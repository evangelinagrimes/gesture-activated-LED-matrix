#include <Adafruit_NeoPixel.h>

#define LED_PIN 5
#define NUMPIXELS 256
#define FRAME_MARKER 'I'       // sync byte marking the start of an image frame
#define FRAME_TIMEOUT_MS 2000  // abort a partially-received frame after this long

// Must match FRAME_MARKER / BAUD_RATE in send_image.py. Frame shape on the
// wire: FRAME_MARKER + NUMPIXELS grayscale bytes (row-major) + 1 checksum
// byte (sum of the payload bytes, mod 256).
Adafruit_NeoPixel pixels(NUMPIXELS, LED_PIN, NEO_GRB + NEO_KHZ800);

uint8_t imageBuffer[NUMPIXELS];

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

// Called once FRAME_MARKER has already been consumed from the stream.
// Blocks (with a timeout, so a dropped/short frame can't hang the board
// forever) until NUMPIXELS payload bytes + 1 checksum byte have arrived,
// then verifies the checksum. Serial has no packet boundaries the way UDP
// did, so the checksum is what catches a corrupted or desynced frame
// instead of silently drawing garbage.
bool readFrame(uint8_t* out) {
  size_t received = 0;
  bool haveChecksum = false;
  uint8_t checksum = 0;
  unsigned long start = millis();

  while (received < NUMPIXELS || !haveChecksum) {
    if (millis() - start > FRAME_TIMEOUT_MS) return false;
    if (!Serial.available()) continue;
    uint8_t b = Serial.read();
    if (received < NUMPIXELS) {
      out[received++] = b;
    } else {
      checksum = b;
      haveChecksum = true;
    }
  }

  uint8_t sum = 0;
  for (size_t i = 0; i < NUMPIXELS; i++) sum += out[i];
  return sum == checksum;
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  pixels.begin();
  setColor(0, 0, 0);  // start with LEDs off

  Serial.println("ESP32 LED Matrix starting...");
  Serial.println("Ready. Waiting for image frames over serial.");
}

void loop() {
  if (!Serial.available()) return;

  uint8_t b = Serial.read();
  if (b != FRAME_MARKER) return;  // not a frame start -- ignore and keep scanning

  if (readFrame(imageBuffer)) {
    displayImage(imageBuffer);
    Serial.println("OK: displayed image frame");
  } else {
    Serial.println("ERR: frame timeout or checksum mismatch");
  }
}
