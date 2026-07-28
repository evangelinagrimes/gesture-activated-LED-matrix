#include <WiFi.h>
#include <WiFiUdp.h>
#include <esp_wifi.h>
#include <esp_system.h>
#include <Adafruit_NeoPixel.h>

#define LED_PIN 5
#define NUMPIXELS 256
#define HEARTBEAT_INTERVAL 5000  // Print heartbeat every 5000ms (5 seconds)
#define WIFI_RECONNECT_INTERVAL 5000  // How often to retry WiFi.reconnect() while down
#define RECONNECT_PULSE_PERIOD 1500   // ms for one full breathe cycle of the "reconnecting" LED
#define RECONNECT_PULSE_MIN_INTERVAL 30  // ms between pixel updates while pulsing (avoid oversaturating the NeoPixel bus)
#define RECONNECT_PULSE_MIN_LEVEL 20   // raw (pre-gamma) low end of the breathe ramp
#define RECONNECT_PULSE_MAX_LEVEL 220  // raw (pre-gamma) high end -- safe to go this bright since pure magenta (R==B, G==0) never matches a gesture color
// After this many plain WiFi.reconnect() calls fail in a row, escalate to a
// full disconnect + WiFi.begin() cycle -- reconnect() re-associates with the
// same cached state, which can get stuck if that state is what's stale.
#define WIFI_HARD_RESET_AFTER_ATTEMPTS 3

// Set to true for full step-by-step debug output, false for minimal/normal operation
#define DEBUG_VERBOSE false

#if DEBUG_VERBOSE
  #define DEBUG_PRINT(x) Serial.print(x)
  #define DEBUG_PRINTLN(x) Serial.println(x)
#else
  #define DEBUG_PRINT(x)
  #define DEBUG_PRINTLN(x)
#endif

// WiFi credentials
const char* WIFI_SSID = "GL-AR300M-1ab";
const char* WIFI_PASSWORD = "goodlife";

// Must match ESP32_PORT / LOCAL_UDP_PORT in gesture.py.
const unsigned int LISTEN_PORT = 4210;  // gesture commands arrive here
const unsigned int REPLY_PORT = 4211;   // debug/status messages go here

Adafruit_NeoPixel pixels(NUMPIXELS, LED_PIN, NEO_GRB + NEO_KHZ800);
WiFiUDP udp;

char packetBuffer[255];
unsigned long lastHeartbeat = 0;
unsigned long lastGestureReceived = 0;
unsigned long lastReconnectAttempt = 0;
unsigned long lastPulseUpdate = 0;
unsigned long disconnectedSince = 0;
unsigned int reconnectAttempts = 0;
int gestureCount = 0;

// Set from onWiFiEvent() the instant the radio disconnects -- read and
// cleared by maintainWiFi() on its next pass. volatile because it's written
// from the WiFi driver's event context, not the main loop().
volatile bool wifiEventDisconnectPending = false;
volatile uint8_t lastDisconnectReason = 0;

// setup() blocks in connectWiFi() until the first connect succeeds, so this
// starts true; maintainWiFi() flips it as the radio drops/recovers.
bool wifiWasConnected = true;

// Address of whoever last sent us a gesture -- debug/status messages are
// sent back there. Empty until the first gesture datagram arrives.
IPAddress lastSenderIP;
bool haveSender = false;

// Prints locally over USB (always) and, once we know where gesture.py is
// listening, also sends the same message back to it over UDP so it shows
// up in the [ESP32]-prefixed terminal log.
void sendStatus(const String& message) {
  Serial.println(message);
  if (!haveSender) return;
  udp.beginPacket(lastSenderIP, REPLY_PORT);
  udp.print(message);
  udp.print("\n");
  udp.endPacket();
}

void setColor(uint8_t r, uint8_t g, uint8_t b) {
  DEBUG_PRINT("setColor() R=");
  DEBUG_PRINT(r);
  DEBUG_PRINT(" G=");
  DEBUG_PRINT(g);
  DEBUG_PRINT(" B=");
  DEBUG_PRINTLN(b);

  for (int i = 0; i < NUMPIXELS; i++) {
    pixels.setPixelColor(i, pixels.Color(r, g, b));
  }
  pixels.show();
}

void processGesture(String gesture) {
  gesture.trim();  // Remove any whitespace/newline chars

  // Always report this one line, even with debug off, so you have basic visibility
  sendStatus("Gesture: " + gesture);

  if (gesture == "thumbs_up") {
    setColor(0, 255, 0);
  } else if (gesture == "thumbs_down") {
    setColor(255, 0, 0);
  } else if (gesture == "peace") {
    setColor(0, 0, 255);
  } else if (gesture == "open_palm") {
    setColor(255, 255, 255);
  } else if (gesture == "fist") {
    setColor(255, 165, 0);
  } else if (gesture == "none") {
    setColor(0, 0, 0);
  } else {
    sendStatus("WARNING: Unknown gesture: " + gesture);
  }

  gestureCount++;
  lastGestureReceived = millis();
}

void printHeartbeat() {
  String msg = "[HEARTBEAT] Uptime: " + String(millis() / 1000) + "s | Gestures: " + String(gestureCount);

  if (lastGestureReceived == 0) {
    msg += " | none yet";
  } else {
    msg += " | last " + String((millis() - lastGestureReceived) / 1000) + "s ago";
  }

  sendStatus(msg);
}

// Human-readable wifi_err_reason_t decoding (values from ESP-IDF's
// esp_wifi_types.h, stable across core versions). Distinguishes router-side
// kicks (AUTH/ASSOC expiry, explicit AUTH_FAIL) from link-quality/power
// symptoms (BEACON_TIMEOUT, HANDSHAKE_TIMEOUT) so Serial output gives a real
// lead instead of just "disconnected".
String wifiDisconnectReasonToString(uint8_t reason) {
  switch (reason) {
    case 2:   return "AUTH_EXPIRE (router expired our authentication)";
    case 3:   return "AUTH_LEAVE";
    case 4:   return "ASSOC_EXPIRE (router expired our association, e.g. idle timeout)";
    case 5:   return "ASSOC_TOOMANY (router hit its client limit)";
    case 8:   return "ASSOC_LEAVE (we or the router tore down the association)";
    case 15:  return "4WAY_HANDSHAKE_TIMEOUT (weak signal or wrong password)";
    case 200: return "BEACON_TIMEOUT (missed router beacons -- weak signal, interference, or the MCU stalled from a power sag)";
    case 201: return "NO_AP_FOUND (SSID not visible -- router down, out of range, or wrong band)";
    case 202: return "AUTH_FAIL (password rejected)";
    case 203: return "ASSOC_FAIL";
    case 204: return "HANDSHAKE_TIMEOUT";
    case 205: return "CONNECTION_FAIL";
    default:  return "code " + String(reason) + " (see wifi_err_reason_t)";
  }
}

// Printed once at boot. ESP_RST_BROWNOUT is the smoking gun for a power
// sag (the 3.3V rail dipping below the brownout threshold, usually because
// a USB port/cable can't supply the ~500mA the radio draws on TX bursts);
// ESP_RST_POWERON just means a normal cold boot.
void printResetReason() {
  Serial.print("Reset reason: ");
  switch (esp_reset_reason()) {
    case ESP_RST_POWERON:   Serial.println("power-on"); break;
    case ESP_RST_BROWNOUT:  Serial.println("BROWNOUT -- check your 5V supply/cable, this is a power problem, not a WiFi one"); break;
    case ESP_RST_PANIC:     Serial.println("software panic/crash"); break;
    case ESP_RST_INT_WDT:
    case ESP_RST_TASK_WDT:
    case ESP_RST_WDT:       Serial.println("watchdog timeout"); break;
    case ESP_RST_DEEPSLEEP: Serial.println("woke from deep sleep"); break;
    case ESP_RST_SW:        Serial.println("software reset (e.g. esp_restart())"); break;
    default:                Serial.println("code " + String((int)esp_reset_reason())); break;
  }
}

// Registered with WiFi.onEvent() so a disconnect is logged (with its reason
// code) the instant it happens, instead of waiting for the next WiFi.status()
// poll in maintainWiFi() to notice the radio dropped. Keep this handler
// minimal -- it runs in the WiFi driver's event context, not loop() -- and
// let maintainWiFi() do the actual reconnect work.
void onWiFiEvent(WiFiEvent_t event, WiFiEventInfo_t info) {
  if (event == ARDUINO_EVENT_WIFI_STA_DISCONNECTED) {
    lastDisconnectReason = info.wifi_sta_disconnected.reason;
    wifiEventDisconnectPending = true;
  }
}

// Breathes the LED matrix through magenta (red+blue, no green -- a mix no
// real gesture color uses) so "trying to reconnect" reads as alive and
// active rather than a color that could just be a frozen/crashed board.
// Throttled to RECONNECT_PULSE_MIN_INTERVAL so it doesn't hammer the
// NeoPixel bus every loop() iteration.
//
// Gamma-corrected via pixels.gamma8(): human brightness perception is
// roughly logarithmic, so a *linear* PWM ramp confined to the low end of
// the 0-255 range (as this originally was, 10-70) compresses almost
// entirely into "looks dim and basically constant" -- it reads as static
// even though the raw value is genuinely animating. gamma8() expands the
// low end back out so the same swing actually looks like it's pulsing.
void updateReconnectPulse() {
  if (millis() - lastPulseUpdate < RECONNECT_PULSE_MIN_INTERVAL) return;
  lastPulseUpdate = millis();

  unsigned long phase = millis() % RECONNECT_PULSE_PERIOD;
  unsigned long half = RECONNECT_PULSE_PERIOD / 2;
  uint8_t rawLevel = (phase < half)
      ? map(phase, 0, half, RECONNECT_PULSE_MIN_LEVEL, RECONNECT_PULSE_MAX_LEVEL)
      : map(phase - half, 0, half, RECONNECT_PULSE_MAX_LEVEL, RECONNECT_PULSE_MIN_LEVEL);
  uint8_t level = pixels.gamma8(rawLevel);
  setColor(level, 0, level);
}

void connectWiFi() {
  Serial.print("Connecting to WiFi ");
  Serial.print(WIFI_SSID);
  WiFi.mode(WIFI_STA);
  // Modem sleep (WiFi power-save) is the single most common cause of an
  // ESP32 that connects fine and then drops a few seconds later -- it can
  // miss beacons on routers with aggressive/nonstandard DTIM settings.
  // Disable it outright since this board is always mains/USB powered.
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    updateReconnectPulse();  // otherwise the matrix just sits dark while we wait
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Connected. IP address: ");
  Serial.println(WiFi.localIP());
  Serial.print("Set ESP32_HOST in gesture.py to: ");
  Serial.println(WiFi.localIP());
}

// Call every loop() iteration. Detects a dropped WiFi connection, pulses a
// distinct "reconnecting" color that can't be mistaken for any real
// gesture color (none of them mix red+blue with green at zero), and
// retries periodically rather than hammering the radio every iteration.
void maintainWiFi() {
  // Log the reason the instant the event fires, rather than waiting for the
  // WL_CONNECTED check below to notice the radio dropped.
  if (wifiEventDisconnectPending) {
    wifiEventDisconnectPending = false;
    Serial.print("WiFi disconnect reason: ");
    Serial.println(wifiDisconnectReasonToString(lastDisconnectReason));
  }

  if (WiFi.status() == WL_CONNECTED) {
    if (!wifiWasConnected) {
      unsigned long downtimeS = (millis() - disconnectedSince) / 1000;
      Serial.println("WiFi reconnected.");
      Serial.print("IP address: ");
      Serial.println(WiFi.localIP());
      udp.begin(LISTEN_PORT);  // rebind -- the network interface reset under us
      wifiWasConnected = true;
      sendStatus("WiFi reconnected after " + String(reconnectAttempts) +
                 " attempt(s), down for " + String(downtimeS) + "s");
      reconnectAttempts = 0;
    }
    return;
  }

  if (wifiWasConnected) {
    Serial.println("WiFi connection lost. Attempting to reconnect...");
    disconnectedSince = millis();
    reconnectAttempts = 0;
    wifiWasConnected = false;
  }

  updateReconnectPulse();

  if (millis() - lastReconnectAttempt >= WIFI_RECONNECT_INTERVAL) {
    lastReconnectAttempt = millis();
    reconnectAttempts++;
    if (reconnectAttempts % WIFI_HARD_RESET_AFTER_ATTEMPTS == 0) {
      // Plain reconnect() re-associates using the driver's existing state,
      // which can get stuck; periodically force a full teardown/rebuild
      // instead. All non-blocking -- these just kick off the process and
      // the next WL_CONNECTED poll picks up when it finishes.
      Serial.println("Escalating to full WiFi restart (plain reconnect isn't working)...");
      WiFi.disconnect(true);
      WiFi.mode(WIFI_STA);
      WiFi.setSleep(false);
      WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    } else {
      WiFi.reconnect();
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("ESP32 Gesture LED Controller starting...");
  printResetReason();

  pixels.begin();
  setColor(0, 0, 0);  // Start with LEDs off

  WiFi.onEvent(onWiFiEvent);
  connectWiFi();
  udp.begin(LISTEN_PORT);

  Serial.print("Ready. Listening for gesture data on UDP port ");
  Serial.println(LISTEN_PORT);
  lastHeartbeat = millis();
}

void loop() {
  maintainWiFi();

  int packetSize = udp.parsePacket();
  if (packetSize > 0) {
    lastSenderIP = udp.remoteIP();
    haveSender = true;

    int len = udp.read(packetBuffer, sizeof(packetBuffer) - 1);
    packetBuffer[len > 0 ? len : 0] = '\0';
    processGesture(String(packetBuffer));
  }

  if (millis() - lastHeartbeat >= HEARTBEAT_INTERVAL) {
    printHeartbeat();
    lastHeartbeat = millis();
  }
}
