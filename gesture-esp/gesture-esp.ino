#include <WiFi.h>
#include <WiFiUdp.h>
#include <Adafruit_NeoPixel.h>

#define LED_PIN 5
#define NUMPIXELS 256
#define HEARTBEAT_INTERVAL 5000  // Print heartbeat every 5000ms (5 seconds)
#define WIFI_RECONNECT_INTERVAL 5000  // How often to retry WiFi.reconnect() while down

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
const char* WIFI_SSID = "GL-AR300-1ab";
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
int gestureCount = 0;

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

void connectWiFi() {
  Serial.print("Connecting to WiFi ");
  Serial.print(WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Connected. IP address: ");
  Serial.println(WiFi.localIP());
  Serial.print("Set ESP32_HOST in gesture.py to: ");
  Serial.println(WiFi.localIP());
}

// Call every loop() iteration. Detects a dropped WiFi connection, shows a
// distinct "reconnecting" color that can't be mistaken for any real
// gesture color (none of them mix red+blue with green at zero), and
// retries periodically rather than hammering the radio every iteration.
void maintainWiFi() {
  if (WiFi.status() == WL_CONNECTED) {
    if (!wifiWasConnected) {
      Serial.println("WiFi reconnected.");
      Serial.print("IP address: ");
      Serial.println(WiFi.localIP());
      udp.begin(LISTEN_PORT);  // rebind -- the network interface reset under us
      wifiWasConnected = true;
    }
    return;
  }

  if (wifiWasConnected) {
    Serial.println("WiFi connection lost. Attempting to reconnect...");
    setColor(60, 0, 60);  // dim magenta = "reconnecting"
    wifiWasConnected = false;
  }

  if (millis() - lastReconnectAttempt >= WIFI_RECONNECT_INTERVAL) {
    lastReconnectAttempt = millis();
    WiFi.reconnect();
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("ESP32 Gesture LED Controller starting...");

  pixels.begin();
  setColor(0, 0, 0);  // Start with LEDs off

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
