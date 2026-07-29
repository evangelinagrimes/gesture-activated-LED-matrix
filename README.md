# LED Matrix

Push a static image to a 256-LED NeoPixel matrix from a PC over a direct
USB serial connection. This is a static, on-demand project: there's no
live detection loop and no requirement that the PC and ESP32 stay
continuously connected — you run a script, it sends one image, the
matrix updates.

## How it works

```
 hex image file --> send_image.py --> USB serial (framed 256-byte image) --> ESP32 --> NeoPixel matrix
                                    <------------- OK / ERR line -------
```

**`send_image.py`** (runs on your PC) reads a hex-literal image file,
converts it to a grayscale byte-per-pixel frame sized to the matrix, and
writes it to the ESP32's serial port as one framed message (a marker
byte + the 256 pixel bytes + a checksum). It waits briefly for the
ESP32's OK/ERR response as a courtesy, but not hearing one back isn't
treated as an error -- the two devices don't need to stay connected
between pushes.

**`led_matrix_esp/led_matrix_esp.ino`** (runs on the ESP32) watches its
serial input for that marker byte, reads the frame that follows, checks
the checksum, and renders it directly to the NeoPixel matrix if it's
intact. There's no WiFi, no network config, and no live session to keep
alive between image pushes -- just the USB cable.

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

- ESP32 DevKit V1 (or similar), connected to the PC by USB
- WS2812/NeoPixel matrix, 256 pixels (16x16), data line on GPIO 5 (`LED_PIN`)
- A stable 5V supply for the ESP32 + LED matrix

## Setup

### ESP32 firmware

1. Open `led_matrix_esp/led_matrix_esp.ino` in the Arduino IDE.
2. Install the `Adafruit NeoPixel` library (Library Manager).
3. Flash it, then open the Serial Monitor at 115200 baud to confirm it
   prints `Ready. Waiting for image frames over serial.` — then close the
   Serial Monitor again (only one program can hold the port open at a time,
   and `send_image.py` needs it next).

### Python side

```
pip install -r requirements.txt
```

1. Find the ESP32's COM port in Windows' Device Manager, under
   "Ports (COM & LPT)", once it's plugged in (shows as something like
   "Silicon Labs CP210x" or "USB-SERIAL CH340"). Set `SERIAL_PORT` at the
   top of `send_image.py` to it.
2. Run it:

   ```
   python send_image.py                    # sends the image at IMAGE_PATH
   python send_image.py images/pic1.txt    # or send a specific file
   ```

   If the configured port can't be opened, the script lists the serial
   ports it actually found to help you pick the right one.

## Tuning knobs

`MATRIX_WIDTH` / `MATRIX_HEIGHT` (top of `send_image.py`) describe the
physical matrix and must match `NUMPIXELS` in `led_matrix_esp.ino`.
`BAUD_RATE` must match `Serial.begin()` there too. `SERIAL_TIMEOUT_S`
controls how long the script waits for the ESP32's OK/ERR response
before giving up (non-fatal either way); `BOOT_SETTLE_S` is how long it
waits after opening the port before writing, since opening it resets most
ESP32 boards (DTR/RTS toggling the auto-reset circuit) and the board
needs to finish `setup()` first.
