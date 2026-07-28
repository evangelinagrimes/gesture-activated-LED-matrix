#include <WiFi.h>
#include <WiFiUdp.h>
#include <Adafruit_NeoPixel.h>

#define LED_PIN 5
#define NUMPIXELS 256
#define WIFI_RECONNECT_INTERVAL 5000  // how often to retry WiFi.reconnect() while down

// WiFi credentials
const char* WIFI_SSID = "GL-AR300M-1ab";
const char* WIFI_PASSWORD = "goodlife";

// Must match ESP32_PORT / REPLY_PORT in send_image.py.
const unsigned int LISTEN_PORT = 4210;  // image frames arrive here
const unsigned int REPLY_PORT = 4211;   // ack goes back here

Adafruit_NeoPixel pixels(NUMPIXELS, LED_PIN, NEO_GRB + NEO_KHZ800);
WiFiUDP udp;

uint8_t imageBuffer[NUMPIXELS];
unsigned long lastReconnectAttempt = 0;

void setColor(uint8_t r, uint8_t g, uint8_t b) {
  for (int i = 0; i < NUMPIXELS; i++) {
    pixels.setPixelColor(i, pixels.Color(r, g, b));
  }
  pixels.show();
}

// Renders a raw single-byte-per-pixel grayscale frame (NUMPIXELS bytes,
// row-major) pushed from the PC -- see send_image() in send_image.py.
// Each byte becomes an equal R=G=B pixel value.
void displayImage(const uint8_t* gray) {
  for (int i = 0; i < NUMPIXELS; i++) {
    uint8_t v = gray[i];
    pixels.setPixelColor(i, pixels.Color(v, v, v));
  }
  pixels.show();
}

void connectWiFi() {
  Serial.print("Connecting to WiFi ");
  Serial.print(WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);  // avoid missing the occasional incoming datagram to modem-sleep power-save
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Connected. IP address: ");
  Serial.println(WiFi.localIP());
  Serial.print("Set ESP32_HOST in send_image.py to: ");
  Serial.println(WiFi.localIP());
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("ESP32 LED Matrix starting...");

  pixels.begin();
  setColor(0, 0, 0);  // start with LEDs off

  connectWiFi();
  udp.begin(LISTEN_PORT);

  Serial.print("Ready. Listening for image frames on UDP port ");
  Serial.println(LISTEN_PORT);
}

void loop() {
  // This project only needs to be reachable when you want to push an
  // image, not continuously live -- so WiFi handling here is just a
  // simple periodic retry while down, with no escalation, diagnostics, or
  // visual "reconnecting" indicator to maintain.
  if (WiFi.status() != WL_CONNECTED) {
    if (millis() - lastReconnectAttempt >= WIFI_RECONNECT_INTERVAL) {
      lastReconnectAttempt = millis();
      WiFi.reconnect();
    }
    return;
  }

  int packetSize = udp.parsePacket();
  if (packetSize == NUMPIXELS) {
    IPAddress sender = udp.remoteIP();
    udp.read(imageBuffer, NUMPIXELS);
    displayImage(imageBuffer);
    Serial.println("Displayed image frame.");

    udp.beginPacket(sender, REPLY_PORT);
    udp.print("OK: displayed image frame\n");
    udp.endPacket();
  } else if (packetSize > 0) {
    // Not a full image frame -- drain and ignore it.
    uint8_t discard[64];
    while (udp.read(discard, sizeof(discard)) == sizeof(discard)) {}
  }
}
