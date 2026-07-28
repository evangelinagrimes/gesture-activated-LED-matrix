# LED Matrix Controller

Drive a 256-LED NeoPixel matrix from a PC over WiFi, two ways: wave at a
webcam and light it up a color per hand gesture, or push a static image
(from a hex byte dump) to display as a full-frame grayscale picture.

## How it works

```
 webcam --> led_matrix_controller.py (MediaPipe + OpenCV) --> UDP --> ESP32 --> NeoPixel matrix
             (or: hex image file  ------------------------->)
                                                  <-- UDP status/heartbeat --
```

**`led_matrix_controller.py`** (runs on your PC) does two things:

- Captures webcam frames with OpenCV and classifies the hand in frame
  using a hybrid approach (see below), sending a short gesture label to
  the ESP32 over UDP.
- On demand (press `i`), reads a hex-literal image file and pushes it as a
  raw 256-byte grayscale frame to the ESP32 over the same UDP link.

### Gesture detection

1. MediaPipe's pretrained `GestureRecognizer` (Tasks API) handles Thumbs Up,
   Thumbs Down, Peace Sign, Open Palm, and Fist, each gated by a confidence
   threshold (`MIN_GESTURE_CONFIDENCE`).
2. Anything the pretrained model doesn't recognize (or isn't confident
   about) falls back to hand-written geometric rules in
   `classify_gesture()`, which also covers a few gestures the pretrained
   model has no category for: Pointing, OK Sign, Rock On.

A gesture has to hold steady for `GESTURE_STABLE_FRAMES` consecutive frames
(`GestureDebouncer`) before it's sent, so a single misclassified frame
doesn't chatter the WiFi link. Confirmed gestures are sent as
newline-terminated UDP datagrams (e.g. `thumbs_up\n`) to the ESP32.

**`gesture-esp/gesture-esp.ino`** (runs on the ESP32) listens for those
datagrams and sets the whole NeoPixel matrix to a solid color per gesture.
It also sends status/heartbeat messages back to the PC over UDP, which
`led_matrix_controller.py` prints with an `[ESP32]` prefix and uses to track whether the
board is still reachable.

### Gesture → color mapping

| Gesture      | LED color        |
|--------------|-------------------|
| `thumbs_up`  | Green             |
| `thumbs_down`| Red               |
| `peace`      | Blue              |
| `open_palm`  | White             |
| `fist`       | Orange            |
| `none`       | Off               |

Gestures the ESP32 doesn't recognize (`Pointing`, `OK Sign`, `Rock On`,
`Unknown`) are sent as `none`.

### Image display

Point `IMAGE_PATH` (top of `led_matrix_controller.py`) at a hex-literal
byte dump — either a C header like image2cpp's
`const uint8_t image[] PROGMEM = {0xFF, 0xBF, ...};` or a plain
comma-separated `.txt` — containing exactly `NUMPIXELS` (256) bytes: one
grayscale brightness value per LED, row-major. `load_image_bytes()` just
scans the file for `0xNN` tokens, so it doesn't care which of the two
formats you use.

Press `i` while the script is running to push it to the matrix as one raw
UDP datagram. The ESP32 (`displayImage()` in `gesture-esp.ino`) tells this
apart from a gesture command purely by packet size — a 256-byte datagram
is always an image frame, never a gesture label.

## Hardware

- ESP32 DevKit V1 (or similar)
- WS2812/NeoPixel matrix, 256 pixels, data line on GPIO 5 (`LED_PIN`)
- A webcam on the PC running `led_matrix_controller.py`
- A stable 5V supply for the ESP32 + LED matrix — a weak USB cable/port is a
  common cause of drops under WiFi TX load; the firmware logs a `BROWNOUT`
  reset reason at boot if this is happening (see below)

## Setup

### ESP32 firmware

1. Open `gesture-esp/gesture-esp.ino` in the Arduino IDE.
2. Install the `Adafruit NeoPixel` library (Library Manager). The `WiFi`,
   `WiFiUdp`, `esp_wifi`, and `esp_system` headers ship with the ESP32
   Arduino core.
3. Set `WIFI_SSID` / `WIFI_PASSWORD` to your network.
4. Flash it, then open the Serial Monitor at 115200 baud — it prints the
   IP address once connected. Note that IP down.

### Python side

```
pip install -r requirements.txt
```

1. Set `ESP32_HOST` in `led_matrix_controller.py` to the IP address from the Serial
   Monitor (a static IP or DHCP reservation is strongly recommended — if
   the ESP32's address changes, gestures silently go nowhere).
2. Run it:

   ```
   python led_matrix_controller.py
   ```

   On first run this downloads MediaPipe's pretrained gesture recognizer
   model (`gesture_recognizer.task`, ~8MB) into the project folder.
3. In the video window: `q` quits, `i` pushes the image at `IMAGE_PATH`
   to the matrix.

## Connection monitoring & diagnostics

WiFi drops are a known rough edge of this project (ESP32 boards can be
sensitive to router power-save settings and marginal power supplies), so
both sides have diagnostics beyond a simple connected/disconnected flag:

- **ESP32**: on a drop, the LED matrix breathes a dim magenta (a color no
  gesture uses) instead of freezing or going dark, so "still trying to
  reconnect" is visible at a glance. It escalates from a plain
  `WiFi.reconnect()` to a full re-associate after a few failed attempts,
  logs the specific WiFi disconnect reason (`wifi_err_reason_t`, e.g.
  `BEACON_TIMEOUT` vs `AUTH_FAIL`) via `WiFi.onEvent()`, and reports the
  reset reason at boot (`BROWNOUT` is the tell for a power supply problem).
- **`led_matrix_controller.py`**: the on-screen overlay shows elapsed downtime and an
  estimated reconnect-attempt count while `UNREACHABLE`, then the ESP32's
  real attempt count once it recovers. Every connection-relevant event
  (drops, recoveries, disconnect reasons) is also appended to
  `esp32_connection.log` next to the script — including from an
  independent background ICMP ping check, so a full network-level drop can
  be told apart from the ESP32 just going quiet at the app layer. This log
  is what to check after an unattended drop.

## Testing

```
python -m unittest test_gestures.py
```

`test_gestures.py` unit-tests the pure classification logic
(`classify_gesture`, `GestureDebouncer`, `resolve_gesture`,
`pick_driving_hand`) against synthetic hand landmarks — no camera or ESP32
required.

## Tuning knobs

The constants at the top of `led_matrix_controller.py` control most of the tunable
behavior: gesture confidence thresholds, hand-detection/tracking
confidence, debounce frame count, and the ESP32 unreachable-timeout. See
the comments next to each for what turning it up or down does.
