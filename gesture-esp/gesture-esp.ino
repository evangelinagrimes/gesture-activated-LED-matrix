#include <string.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <esp_wifi.h>
#include <esp_system.h>
#include <Adafruit_NeoPixel.h>

#define LED_PIN 5
#define NUMPIXELS 256
#define HEARTBEAT_INTERVAL 5000  // Print heartbeat every 5000ms (5 seconds)
#define WIFI_RECONNECT_INTERVAL 5000  // How often to retry WiFi.reconnect() while down
#define WIFI_CONNECT_TIMEOUT_MS 15000  // setup() gives up waiting and moves on to loop() after this long with no router
#define RECONNECT_PULSE_PERIOD 1500   // ms for one full breathe cycle of the "reconnecting" LED
#define RECONNECT_PULSE_MIN_INTERVAL 30  // ms between pixel updates while pulsing (avoid oversaturating the NeoPixel bus)
#define RECONNECT_PULSE_MIN_LEVEL 20   // raw (pre-gamma) low end of the breathe ramp
#define RECONNECT_PULSE_MAX_LEVEL 220  // raw (pre-gamma) high end -- safe to go this bright since pure magenta (R==B, G==0) never matches a gesture color
// After this many plain WiFi.reconnect() calls fail in a row, escalate to a
// full disconnect + WiFi.begin() cycle -- reconnect() re-associates with the
// same cached state, which can get stuck if that state is what's stale.
#define WIFI_HARD_RESET_AFTER_ATTEMPTS 3
// While a command (serial or UDP, frame or label) has arrived within this
// window, maintainWiFi() leaves the panel alone instead of pulsing over it
// -- a live serial link means the board isn't actually unattended just
// because WiFi is down. See maintainWiFi().
#define COMMAND_IDLE_BEFORE_PULSE_MS 10000

// Caps total current draw so a bright sprite frame can't sag the 5V rail
// and brown the board out -- 256 WS2812 pixels at full white is ~15A,
// well past what most USB/5V supplies can deliver. Applies to every
// pixels.show() call (setColor(), renderFrame(), the reconnect pulse).
#define MAX_BRIGHTNESS 40  // out of 255

// Set to true for full step-by-step debug output, false for minimal/normal operation
#define DEBUG_VERBOSE false

#if DEBUG_VERBOSE
  #define DEBUG_PRINT(x) Serial.print(x)
  #define DEBUG_PRINTLN(x) Serial.println(x)
#else
  #define DEBUG_PRINT(x)
  #define DEBUG_PRINTLN(x)
#endif

// WiFi credentials -- kept out of this tracked file. Copy
// .env/secrets.h.example to .env/secrets.h (gitignored) and fill in your
// real network name/password there; this just defines WIFI_SSID/WIFI_PASSWORD.
// If secrets.h doesn't exist (e.g. a fresh clone), fall back to empty
// credentials rather than failing to compile -- WiFi.begin() will simply
// never connect, and the bounded connect timeout (WIFI_CONNECT_TIMEOUT_MS)
// takes over, same as being on the serial transport with no router at all.
#if __has_include(".env/secrets.h")
  #include ".env/secrets.h"
#else
  #warning "gesture-esp/.env/secrets.h not found -- WiFi will not connect. Copy .env/secrets.h.example to .env/secrets.h (optional, only needed for the UDP transport)."
  #define WIFI_SSID ""
  #define WIFI_PASSWORD ""
#endif

// Must match ESP32_PORT / LOCAL_UDP_PORT in gesture.py.
const unsigned int LISTEN_PORT = 4210;  // gesture commands arrive here
const unsigned int REPLY_PORT = 4211;   // debug/status messages go here

// --- LED matrix geometry -------------------------------------------------
// Adafruit_NeoPixel only knows about a flat pixel count; these constants
// and xyToIndex() are what give it 2-D meaning. If a sprite frame comes out
// mirrored or rotated on the real panel, these are the knobs to change (and
// reflash) -- send the "test" command (a lit top-left corner pixel, then a
// row sweep top-to-bottom) to see which one you need.
#define MATRIX_WIDTH 16
#define MATRIX_HEIGHT 16
#define MATRIX_SERPENTINE true    // rows alternate direction (most common WS2812 panel wiring)
#define MATRIX_FLIP_X false
#define MATRIX_FLIP_Y false
#define MATRIX_ROTATION 0         // 0, 90, 180, or 270 -- applied before flip/serpentine

// 4-byte magic prefix identifying a sprite-frame payload, followed by
// "<label>\n<768 bytes raw RGB, row-major from the top-left>". Must match
// FRAME_MAGIC in sprites.py. Doesn't collide with any gesture label, so a
// payload not starting with this falls through to the plain-label path.
const char FRAME_MAGIC[] = "IMG1";
#define FRAME_MAGIC_LEN 4

Adafruit_NeoPixel pixels(NUMPIXELS, LED_PIN, NEO_GRB + NEO_KHZ800);
WiFiUDP udp;

uint8_t rxBuffer[1024];  // shared scratch buffer for one UDP datagram or serial frame
unsigned long lastHeartbeat = 0;
unsigned long lastGestureReceived = 0;
unsigned long lastReconnectAttempt = 0;
unsigned long lastPulseUpdate = 0;
unsigned long disconnectedSince = 0;
unsigned int reconnectAttempts = 0;
int gestureCount = 0;

// Set the instant *any* command (serial or UDP) is successfully parsed --
// see handleCommand() and COMMAND_IDLE_BEFORE_PULSE_MS above. 0 means
// "never", which intentionally behaves like "idle" so the very first WiFi
// connect attempt in setup() still pulses (nothing could have arrived yet).
unsigned long lastCommandReceived = 0;

// Set from onWiFiEvent() the instant the radio disconnects -- read and
// cleared by maintainWiFi() on its next pass. volatile because it's written
// from the WiFi driver's event context, not the main loop().
volatile bool wifiEventDisconnectPending = false;
volatile uint8_t lastDisconnectReason = 0;

// Whether WiFi was up as of the last maintainWiFi() pass. Starts false:
// connectWiFi() now waits only up to WIFI_CONNECT_TIMEOUT_MS (it used to
// block here forever), so setup() can reach loop() with no router present
// at all -- this must reflect the *real* outcome of that bounded wait, not
// assume success, or maintainWiFi() would broadcast a spurious "WiFi
// connection lost" on every WiFi-less boot. connectWiFi() sets the real
// value once it knows it.
bool wifiWasConnected = false;

// Address of whoever last sent us a gesture over UDP -- debug/status
// messages are sent back there. Empty until the first UDP datagram arrives
// (irrelevant for the serial transport, which is always full-duplex).
IPAddress lastSenderIP;
bool haveSender = false;

// --- Serial framing state (see pollSerial()) ------------------------------
// Wire format for one framed message:
//   0xA5 0x5A <uint16 LE payload_len> <payload bytes> <xor checksum>
// Necessary because a sprite frame's raw RGB bytes can be any value,
// including 0x0A -- a newline-terminated read would corrupt it. Must match
// SerialTransport in transport.py.
#define SERIAL_SYNC1 0xA5
#define SERIAL_SYNC2 0x5A
#define SERIAL_MAX_PAYLOAD 1024        // must match SerialTransport.MAX_PAYLOAD in transport.py
#define SERIAL_MAX_BYTES_PER_LOOP 1024 // ceiling per pollSerial() call so it can't stall maintainWiFi()
#define SERIAL_FRAME_TIMEOUT_MS 250    // resync if a frame stalls mid-parse this long
#define SERIAL_LINE_MAX 128            // bare ASCII fallback line buffer (manual Serial Monitor typing)

enum SerialParseState { SP_SYNC1, SP_SYNC2, SP_LEN_LO, SP_LEN_HI, SP_PAYLOAD, SP_CHECKSUM };
SerialParseState serialState = SP_SYNC1;
uint8_t serialRxBuffer[SERIAL_MAX_PAYLOAD];
uint16_t serialPayloadLen = 0;
uint16_t serialPayloadIdx = 0;
uint8_t serialChecksum = 0;
unsigned long serialFrameStart = 0;

char serialLineBuffer[SERIAL_LINE_MAX];
uint16_t serialLineLen = 0;

// Prints locally over USB (always) and, once we know where gesture.py is
// listening over UDP, also sends the same message back to it there so it
// shows up in the [ESP32]-prefixed terminal log. Over the serial transport
// this Serial.println() alone is the entire return channel -- gesture.py's
// SerialTransport reads it directly, no framing needed on this side.
void sendStatus(const String& message) {
  Serial.println(message);
  if (!haveSender) return;
  udp.beginPacket(lastSenderIP, REPLY_PORT);
  udp.print(message);
  udp.print("\n");
  udp.endPacket();
}

// The subnet broadcast address (e.g. 192.168.8.255), computed from the
// current IP/mask rather than hardcoded so it still works if the router's
// subnet ever changes.
IPAddress broadcastAddress() {
  IPAddress ip = WiFi.localIP();
  IPAddress mask = WiFi.subnetMask();
  IPAddress bcast;
  for (int i = 0; i < 4; i++) {
    bcast[i] = ip[i] | (~mask[i] & 0xFF);
  }
  return bcast;
}

// Like sendStatus(), but also broadcasts on the local subnet. gesture.py's
// UDP listener accepts a datagram from any address, so this reaches it even
// if haveSender is still false (no gesture has ever been sent this boot) --
// used only for the handful of messages that matter for the connection log
// (boot diagnostics, reconnect summaries) even on a session with no gestures.
void broadcastStatus(const String& message) {
  sendStatus(message);
  if (WiFi.status() != WL_CONNECTED) return;  // no IP yet to compute a broadcast address from
  udp.beginPacket(broadcastAddress(), REPLY_PORT);
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

// Maps a logical (x,y) -- x,y in [0, MATRIX_WIDTH)/[0, MATRIX_HEIGHT) --
// to the flat NeoPixel index for however this panel is actually wired.
// Order: rotation, then flips, then serpentine row reversal. This is the
// one place panel wiring is encoded; see the MATRIX_* constants above.
uint16_t xyToIndex(uint8_t x, uint8_t y) {
  uint8_t rx = x, ry = y;

#if MATRIX_ROTATION == 90
  rx = y;
  ry = (MATRIX_WIDTH - 1) - x;
#elif MATRIX_ROTATION == 180
  rx = (MATRIX_WIDTH - 1) - x;
  ry = (MATRIX_HEIGHT - 1) - y;
#elif MATRIX_ROTATION == 270
  rx = (MATRIX_HEIGHT - 1) - y;
  ry = x;
#endif
  // (Assumes a square panel, true for the 16x16 this firmware targets --
  // a non-square rotation would need separate rotated width/height.)

  if (MATRIX_FLIP_X) rx = (MATRIX_WIDTH - 1) - rx;
  if (MATRIX_FLIP_Y) ry = (MATRIX_HEIGHT - 1) - ry;

  if (MATRIX_SERPENTINE && (ry % 2 == 1)) {
    rx = (MATRIX_WIDTH - 1) - rx;
  }

  return (uint16_t)ry * MATRIX_WIDTH + rx;
}

// Renders a 16x16 row-major RGB frame (MATRIX_WIDTH*MATRIX_HEIGHT*3 bytes,
// as produced by sprites.encode_frame() in the Python side) onto the panel.
void renderFrame(const uint8_t* rgb) {
  for (uint8_t y = 0; y < MATRIX_HEIGHT; y++) {
    for (uint8_t x = 0; x < MATRIX_WIDTH; x++) {
      size_t srcIdx = ((size_t)y * MATRIX_WIDTH + x) * 3;
      pixels.setPixelColor(xyToIndex(x, y), pixels.Color(rgb[srcIdx], rgb[srcIdx + 1], rgb[srcIdx + 2]));
    }
  }
  pixels.show();
}

// Blocking manual diagnostic: lights the (0,0) pixel alone, then sweeps
// each row top-to-bottom, so xyToIndex()'s MATRIX_* constants can be
// confirmed (or corrected) by eye against the physical panel. Triggered by
// sending/typing the "test" label like any other gesture -- see
// processGesture(). Intentionally blocks loop() for a few seconds, the same
// way connectWiFi() already does; this is a rarely-used manual command, not
// part of normal operation.
void runOrientationTest() {
  sendStatus("Running orientation test: corner pixel, then row sweep top-to-bottom...");
  pixels.clear();
  pixels.show();
  delay(200);

  pixels.setPixelColor(xyToIndex(0, 0), pixels.Color(255, 255, 255));
  pixels.show();
  delay(800);

  for (uint8_t y = 0; y < MATRIX_HEIGHT; y++) {
    pixels.clear();
    for (uint8_t x = 0; x < MATRIX_WIDTH; x++) {
      pixels.setPixelColor(xyToIndex(x, y), pixels.Color(0, 128, 255));
    }
    pixels.show();
    delay(150);
  }

  pixels.clear();
  pixels.show();
  sendStatus("Orientation test complete.");
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
  } else if (gesture == "test") {
    runOrientationTest();
  } else {
    sendStatus("WARNING: Unknown gesture: " + gesture);
  }

  gestureCount++;
  lastGestureReceived = millis();
}

// Shared sink for both transports: dispatches a complete, already-framed
// message (raw bytes, NOT null-terminated -- a sprite frame is arbitrary
// binary and must never be treated as a C string) to either the sprite
// renderer or the original flat-color gesture path.
void handleCommand(const uint8_t* payload, size_t len) {
  lastCommandReceived = millis();

  if (len >= FRAME_MAGIC_LEN && memcmp(payload, FRAME_MAGIC, FRAME_MAGIC_LEN) == 0) {
    int nl = -1;
    for (size_t i = FRAME_MAGIC_LEN; i < len; i++) {
      if (payload[i] == '\n') { nl = (int)i; break; }
    }
    if (nl < 0) {
      sendStatus("WARNING: malformed IMG1 frame (no label terminator)");
      return;
    }

    String label;
    for (size_t i = FRAME_MAGIC_LEN; i < (size_t)nl; i++) label += (char)payload[i];

    size_t pixelDataLen = len - (nl + 1);
    const size_t expected = (size_t)MATRIX_WIDTH * MATRIX_HEIGHT * 3;
    if (pixelDataLen != expected) {
      sendStatus("WARNING: bad frame for '" + label + "': got " + String((unsigned)pixelDataLen) +
                 " pixel bytes, expected " + String((unsigned)expected));
      return;
    }

    sendStatus("Gesture: " + label);
    renderFrame(payload + nl + 1);
    gestureCount++;
    lastGestureReceived = millis();
    return;
  }

  // Not a sprite frame -- treat the whole payload as a plain ASCII label,
  // exactly like the original wire protocol.
  String label;
  for (size_t i = 0; i < len; i++) label += (char)payload[i];
  processGesture(label);
}

// Accumulates bytes into a line buffer, dispatching to handleCommand() as a
// plain label on '\n'. This is the fallback path when serial input isn't
// our sync-framed format -- e.g. someone typing a label into the Arduino
// Serial Monitor by hand.
void handleSerialLineByte(uint8_t b) {
  if (b == '\n') {
    if (serialLineLen > 0) {
      handleCommand((const uint8_t*)serialLineBuffer, serialLineLen);
      serialLineLen = 0;
    }
    return;
  }
  if (b == '\r') return;  // tolerate CRLF line endings
  if (serialLineLen < SERIAL_LINE_MAX - 1) {
    serialLineBuffer[serialLineLen++] = (char)b;
  }
}

// Byte-at-a-time state machine for the serial transport's sync/length/
// checksum framing (see the SERIAL_* constants above). Falls back to
// treating input as a bare newline-terminated ASCII line whenever the
// first byte of a message isn't the sync word. Drains at most
// SERIAL_MAX_BYTES_PER_LOOP bytes per call so a burst of input can't stall
// maintainWiFi().
void pollSerial() {
  int budget = SERIAL_MAX_BYTES_PER_LOOP;
  while (budget-- > 0 && Serial.available() > 0) {
    uint8_t b = (uint8_t)Serial.read();

    if (serialState != SP_SYNC1 && millis() - serialFrameStart > SERIAL_FRAME_TIMEOUT_MS) {
      // Stalled mid-frame -- resync rather than waiting forever for bytes
      // that may never come (a partial write, or the sender restarted).
      serialState = SP_SYNC1;
    }

    switch (serialState) {
      case SP_SYNC1:
        if (b == SERIAL_SYNC1) {
          serialState = SP_SYNC2;
          serialFrameStart = millis();
        } else {
          handleSerialLineByte(b);
        }
        break;

      case SP_SYNC2:
        if (b == SERIAL_SYNC2) {
          serialState = SP_LEN_LO;
        } else {
          // The SYNC1 byte we consumed was actually ordinary line content
          // (unlikely from a human typing ASCII, but be correct about it).
          handleSerialLineByte(SERIAL_SYNC1);
          handleSerialLineByte(b);
          serialState = SP_SYNC1;
        }
        break;

      case SP_LEN_LO:
        serialPayloadLen = b;
        serialState = SP_LEN_HI;
        break;

      case SP_LEN_HI:
        serialPayloadLen |= ((uint16_t)b << 8);
        if (serialPayloadLen == 0 || serialPayloadLen > SERIAL_MAX_PAYLOAD) {
          sendStatus("WARNING: serial frame length " + String(serialPayloadLen) + " out of range, resyncing");
          serialState = SP_SYNC1;
        } else {
          serialPayloadIdx = 0;
          serialChecksum = 0;
          serialState = SP_PAYLOAD;
        }
        break;

      case SP_PAYLOAD:
        serialRxBuffer[serialPayloadIdx++] = b;
        serialChecksum ^= b;
        if (serialPayloadIdx >= serialPayloadLen) {
          serialState = SP_CHECKSUM;
        }
        break;

      case SP_CHECKSUM:
        if (b == serialChecksum) {
          handleCommand(serialRxBuffer, serialPayloadLen);
        } else {
          sendStatus("WARNING: serial frame checksum mismatch, dropping");
        }
        serialState = SP_SYNC1;
        break;
    }
  }
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
  String prefix = "reason " + String(reason) + " ";
  switch (reason) {
    case 2:   return prefix + "AUTH_EXPIRE (router expired our authentication)";
    case 3:   return prefix + "AUTH_LEAVE";
    case 4:   return prefix + "ASSOC_EXPIRE (router expired our association, e.g. idle timeout)";
    case 5:   return prefix + "ASSOC_TOOMANY (router hit its client limit)";
    case 8:   return prefix + "ASSOC_LEAVE (we or the router tore down the association)";
    case 15:  return prefix + "4WAY_HANDSHAKE_TIMEOUT (weak signal or wrong password)";
    case 200: return prefix + "BEACON_TIMEOUT (missed router beacons -- weak signal, interference, or the MCU stalled from a power sag)";
    case 201: return prefix + "NO_AP_FOUND (SSID not visible -- router down, out of range, or wrong band)";
    case 202: return prefix + "AUTH_FAIL (password rejected)";
    case 203: return prefix + "ASSOC_FAIL";
    case 204: return prefix + "HANDSHAKE_TIMEOUT";
    case 205: return prefix + "CONNECTION_FAIL";
    default:  return prefix + "(see wifi_err_reason_t)";
  }
}

// ESP_RST_BROWNOUT is the smoking gun for a power sag (the 3.3V rail
// dipping below the brownout threshold, usually because a USB port/cable
// can't supply the ~500mA the radio draws on TX bursts); ESP_RST_POWERON
// just means a normal cold boot.
String resetReasonString() {
  switch (esp_reset_reason()) {
    case ESP_RST_POWERON:   return "power-on";
    case ESP_RST_BROWNOUT:  return "BROWNOUT -- check your 5V supply/cable, this is a power problem, not a WiFi one";
    case ESP_RST_PANIC:     return "software panic/crash";
    case ESP_RST_INT_WDT:
    case ESP_RST_TASK_WDT:
    case ESP_RST_WDT:       return "watchdog timeout";
    case ESP_RST_DEEPSLEEP: return "woke from deep sleep";
    case ESP_RST_SW:        return "software reset (e.g. esp_restart())";
    default:                return "code " + String((int)esp_reset_reason());
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
// NeoPixel bus every loop() iteration. Gated by the caller in maintainWiFi()
// so it can't stomp on a sprite while a live serial link is still driving
// the panel -- see COMMAND_IDLE_BEFORE_PULSE_MS.
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

// Waits up to WIFI_CONNECT_TIMEOUT_MS for a first connection, then gives up
// and returns either way -- this board must not depend on WiFi to be
// useful, since the primary transport is USB serial. maintainWiFi() keeps
// retrying in the background for as long as the board runs.
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

  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < WIFI_CONNECT_TIMEOUT_MS) {
    updateReconnectPulse();  // otherwise the matrix just sits dark while we wait
    delay(500);
    Serial.print(".");
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("Connected. IP address: ");
    Serial.println(WiFi.localIP());
    Serial.print("Set ESP32_HOST in gesture.py to: ");
    Serial.println(WiFi.localIP());

    // Broadcast (not just unicast) so gesture.py's connection log picks this
    // up even if it hasn't sent a gesture yet this session -- this is the
    // one place the boot-time reset reason (brownout vs normal) ever leaves
    // the board over UDP. Over serial, the Serial.println() calls above and
    // resetReasonString() in setup() already cover it.
    broadcastStatus("ESP32 booted. WiFi ready, IP=" + WiFi.localIP().toString() +
                     ". Reset reason: " + resetReasonString());
  } else {
    Serial.println("WiFi not connected within timeout -- continuing without it.");
    Serial.println("Serial link still works. maintainWiFi() will keep retrying in the background.");
  }

  // Must reflect the real outcome above, not assume success -- see the
  // wifiWasConnected declaration comment for why.
  wifiWasConnected = (WiFi.status() == WL_CONNECTED);
}

// Call every loop() iteration. Detects a dropped WiFi connection, pulses a
// distinct "reconnecting" color that can't be mistaken for any real
// gesture color (none of them mix red+blue with green at zero) -- unless a
// command has arrived recently over any transport, in which case the panel
// is left showing whatever it's currently showing -- and retries
// periodically rather than hammering the radio every iteration.
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
      // broadcastStatus, not sendStatus: this is the main thing the
      // connection log needs, so it must reach gesture.py even if no
      // gesture has ever been sent (haveSender still false).
      broadcastStatus("WiFi reconnected after " + String(reconnectAttempts) +
                       " attempt(s), down for " + String(downtimeS) + "s. Last disconnect " +
                       wifiDisconnectReasonToString(lastDisconnectReason));
      reconnectAttempts = 0;
    }
    return;
  }

  if (wifiWasConnected) {
    disconnectedSince = millis();
    reconnectAttempts = 0;
    wifiWasConnected = false;
    // broadcastStatus() prints to Serial too, so this alone covers what
    // used to be a separate Serial.println() here. Best-effort over UDP --
    // the link may already be too far gone to actually deliver it, but if
    // the drop is graceful it can still get out.
    broadcastStatus("WiFi connection lost, attempting to reconnect...");
  }

  // Only pulse if nothing has driven the panel recently -- a live serial
  // link (or the ESP32 having never connected to WiFi at all this boot,
  // lastCommandReceived == 0) should keep showing the current sprite rather
  // than being interrupted by the reconnect animation every 5 seconds.
  bool commandRecent = (lastCommandReceived != 0) &&
                        (millis() - lastCommandReceived < COMMAND_IDLE_BEFORE_PULSE_MS);
  if (!commandRecent) {
    updateReconnectPulse();
  }

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
  Serial.print("Reset reason: ");
  Serial.println(resetReasonString());

  pixels.begin();
  pixels.setBrightness(MAX_BRIGHTNESS);
  setColor(0, 0, 0);  // Start with LEDs off

  WiFi.onEvent(onWiFiEvent);
  connectWiFi();       // bounded -- returns even with no router present
  udp.begin(LISTEN_PORT);

  Serial.print("Ready. Listening for gesture data on UDP port ");
  Serial.print(LISTEN_PORT);
  Serial.println(" and on this USB serial connection.");
  lastHeartbeat = millis();
}

void loop() {
  maintainWiFi();
  pollSerial();

  int packetSize = udp.parsePacket();
  if (packetSize > 0) {
    lastSenderIP = udp.remoteIP();
    haveSender = true;

    int len = udp.read(rxBuffer, sizeof(rxBuffer));
    if (len > 0) {
      handleCommand(rxBuffer, (size_t)len);
    }
  }

  if (millis() - lastHeartbeat >= HEARTBEAT_INTERVAL) {
    printHeartbeat();
    lastHeartbeat = millis();
  }
}
