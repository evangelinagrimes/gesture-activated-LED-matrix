# LED Matrix

Push a static image to a 256-LED NeoPixel matrix from a PC over WiFi/UDP.
This is a static, on-demand project: there's no live detection loop and
no requirement that the PC and ESP32 stay continuously connected — you
run a script, it sends one image, the matrix updates.

## How it works

```
 hex image file --> send_image.py --> UDP (256-byte frame) --> ESP32 --> NeoPixel matrix
                                    <---------- UDP ack -----
```

**`send_image.py`** (runs on your PC) reads a hex-literal image file,
converts it to a grayscale byte-per-pixel frame sized to the matrix, and
sends it as a single raw UDP datagram to the ESP32. It waits briefly for
an acknowledgment as a courtesy, but not hearing one back isn't treated as
an error -- the two devices don't need to stay connected between pushes.

**`led_matrix_esp/led_matrix_esp.ino`** (runs on the ESP32) listens for
that datagram and renders it directly to the NeoPixel matrix. It
reconnects to WiFi in the background if the link drops, but does nothing
fancier than that -- there's no live session to keep alive between image
pushes.

### Image format

Point `IMAGE_PATH` (top of `send_image.py`), or pass a path as a command
line argument, at a hex-literal image file — either:

- **image2cpp-style, any resolution** — a `..._width`/`..._height`
  declaration plus a packed pixel array at 1, 4, or 8 bits per pixel
  (auto-detected from the byte count). The source image does **not**
  need to already be 16x16: `load_image_bytes()` downsamples
  (area-averaged) or upsamples (nearest-neighbor) it to the matrix's
  resolution automatically.
- **A flat 256-byte dump** — if the file has no width/height metadata,
  it's assumed to already be exactly `NUMPIXELS` (256) grayscale bytes,
  one per LED, row-major.

Either way, the parser just scans the file for `0xNN` tokens, so it
doesn't care whether the surrounding syntax is a `.h` C header or a plain
comma-separated `.txt`. See [`images/README.md`](images/README.md) for
more detail, and drop your own files in `images/`.

## Hardware

- ESP32 DevKit V1 (or similar)
- WS2812/NeoPixel matrix, 256 pixels (16x16), data line on GPIO 5 (`LED_PIN`)
- A stable 5V supply for the ESP32 + LED matrix

## Setup

### ESP32 firmware

1. Open `led_matrix_esp/led_matrix_esp.ino` in the Arduino IDE.
2. Install the `Adafruit NeoPixel` library (Library Manager). `WiFi` and
   `WiFiUdp` ship with the ESP32 Arduino core.
3. Set `WIFI_SSID` / `WIFI_PASSWORD` to your network.
4. Flash it, then open the Serial Monitor at 115200 baud — it prints the
   IP address once connected. Note that IP down.

### Python side

```
pip install -r requirements.txt
```

1. Set `ESP32_HOST` in `send_image.py` to the IP address from the Serial
   Monitor (a static IP or DHCP reservation is strongly recommended — if
   the ESP32's address changes, images silently go nowhere).
2. Run it:

   ```
   python send_image.py                    # sends the image at IMAGE_PATH
   python send_image.py images/pic1.txt    # or send a specific file
   ```

## Tuning knobs

`MATRIX_WIDTH` / `MATRIX_HEIGHT` (top of `send_image.py`) describe the
physical matrix and must match `NUMPIXELS` in `led_matrix_esp.ino`.
`ACK_TIMEOUT_S` controls how long the script waits for the ESP32's
acknowledgment before giving up (non-fatal either way).
