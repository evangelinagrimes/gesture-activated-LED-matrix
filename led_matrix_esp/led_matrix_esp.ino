#include <Adafruit_NeoPixel.h>

#define LED_PIN 5
#define MATRIX_WIDTH 16
#define MATRIX_HEIGHT 16
#define NUMPIXELS (MATRIX_WIDTH * MATRIX_HEIGHT)
#define FRAME_MARKER 'I'       // sync byte marking the start of an image frame
#define FRAME_TIMEOUT_MS 2000  // abort a partially-received frame after this long

// Most pre-made 16x16 WS2812 panels wire alternate rows in reverse
// (serpentine/zigzag) to shorten the trace run between rows, rather than
// every row running left-to-right (raster). If the image comes out
// looking scrambled/shredded despite send_image.py reporting a valid
// checksum, flip this and reflash -- it's the classic tell.
#define MATRIX_SERPENTINE true

// Render style for incoming image frames. Some source images (markers,
// logos, text) are conceptually two-tone; resizing/anti-aliasing leaves
// intermediate gray values that look murky, and true black pixels are
// simply LEDs-off -- physically indistinguishable from "nothing
// displayed." BINARIZE thresholds every pixel to one of two solid,
// independently configurable colors instead of smooth grayscale, so both
// classes of pixel are always clearly visible.
#define BINARIZE true
#define BINARIZE_THRESHOLD 128  // gray value >= this -> COLOR_ON, else COLOR_OFF
#define COLOR_ON_R 255          // default: white, for the brighter/foreground class
#define COLOR_ON_G 255
#define COLOR_ON_B 255
#define COLOR_OFF_R 0           // default: black (LEDs off) -- set to any RGB to make
#define COLOR_OFF_G 0           // the "other" class its own visible color instead
#define COLOR_OFF_B 0

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
// row-major, top-left to bottom-right) pushed from the PC -- see
// send_image() in send_image.py. Remaps row-major source coordinates onto
// the matrix's actual physical wiring order (see MATRIX_SERPENTINE above)
// before writing. Each byte becomes either a solid COLOR_ON/COLOR_OFF
// (if BINARIZE) or an equal R=G=B grayscale value.
void displayImage(const uint8_t* gray) {
  for (int y = 0; y < MATRIX_HEIGHT; y++) {
    for (int x = 0; x < MATRIX_WIDTH; x++) {
      int physX = (MATRIX_SERPENTINE && (y % 2 == 1)) ? (MATRIX_WIDTH - 1 - x) : x;
      uint8_t v = gray[y * MATRIX_WIDTH + x];
      uint32_t color = BINARIZE
          ? (v >= BINARIZE_THRESHOLD
                 ? pixels.Color(COLOR_ON_R, COLOR_ON_G, COLOR_ON_B)
                 : pixels.Color(COLOR_OFF_R, COLOR_OFF_G, COLOR_OFF_B))
          : pixels.Color(v, v, v);
      pixels.setPixelColor(y * MATRIX_WIDTH + physX, color);
    }
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
