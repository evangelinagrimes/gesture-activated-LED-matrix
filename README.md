# Gesture-Activated LED Matrix

Wave at a webcam, light up a 16x16 (256-LED) matrix with a picture of the
gesture you made. A Python script watches your hand through a camera,
classifies the gesture, and sends it to an ESP32 over USB serial (default)
or WiFi, which renders a sprite -- or, for any gesture with no sprite yet,
falls back to a solid color -- on a NeoPixel matrix.

## How it works

```
 webcam --> gesture.py (MediaPipe + OpenCV) --> transport --> ESP32 --> NeoPixel matrix
                                                    <-- status/heartbeat --
```

`transport` is USB serial by default (`TRANSPORT = "serial"` in
`gesture.py`, talking to `SERIAL_PORT`, e.g. `COM3`) so the whole pipeline
works with no router at all. Set `TRANSPORT = "udp"` to use the original
WiFi link instead. Both send the exact same payload; see
[Transports](#transports) below.

**`gesture.py`** (runs on your PC) captures webcam frames with OpenCV and
classifies the hand in frame using a hybrid approach:

1. MediaPipe's pretrained `GestureRecognizer` (Tasks API) handles Thumbs Up,
   Thumbs Down, Peace Sign, Open Palm, and Fist, each gated by a confidence
   threshold (`MIN_GESTURE_CONFIDENCE`).
2. Anything the pretrained model doesn't recognize (or isn't confident
   about) falls back to hand-written geometric rules in
   `classify_gesture()`, which also covers a few gestures the pretrained
   model has no category for: Pointing, OK Sign, Rock On, Middle Finger.

A gesture has to hold steady for `GESTURE_STABLE_FRAMES` consecutive frames
(`GestureDebouncer`) before it's sent, so a single misclassified frame
doesn't chatter the link. The current gesture is also re-sent every
`FRAME_REFRESH_S` seconds (and immediately after a reconnect), so a dropped
UDP datagram, an unplugged cable, or an ESP32 reboot can't leave stale
artwork on the panel.

**`gesture-esp/gesture-esp.ino`** (runs on the ESP32) listens on both the
serial port and UDP, renders whatever it's told to onto the NeoPixel panel,
and reports status/heartbeat messages back, which `gesture.py` prints with
an `[ESP32]` prefix and shows in the debug dashboard.

### Sprites → what actually lights up

Each gesture label looks up a 16x16 image in `sprites/` (`thumbs_up.png`,
`fist.hex`, etc.) and streams it to the ESP32 pixel-for-pixel. A label with
no sprite file falls back to the original flat-color behavior:

| Gesture         | Sprite file (if present)    | Fallback color (if not) |
|-----------------|-------------------------------|--------------------------|
| `thumbs_up`     | `sprites/thumbs_up.*`         | Green                    |
| `thumbs_down`   | `sprites/thumbs_down.*`       | Red                      |
| `peace`         | `sprites/peace.*`             | Blue                     |
| `open_palm`     | `sprites/open_palm.*`         | White                    |
| `fist`          | `sprites/fist.*`              | Orange                   |
| `ok_sign`       | `sprites/ok_sign.*`           | Yellow                   |
| `middle_finger` | `sprites/middle_finger.*`     | Purple                   |
| `none`          | `sprites/none.*`              | Off                      |

Gestures the ESP32 doesn't recognize (`Pointing`, `Rock On`, `Unknown`) are
sent as `none`. See [sprites/README.md](sprites/README.md) for the two
supported file formats (image or hand-authored hex text) and how to check
new artwork with `python -m sprites` before plugging anything in. The repo
ships placeholder `.hex` pixel art for all seven gestures so the whole
pipeline works out of the box -- drop a same-named `.png` in next
to one to replace it, no code change or reflash needed.

## Transports

Set `TRANSPORT` at the top of `gesture.py`:

- **`"serial"` (default)** -- USB cable to `SERIAL_PORT` (`COM3` by
  default) at `SERIAL_BAUD` (115200). Doesn't depend on WiFi or a router at
  all. Frames are wrapped in a small sync/length/checksum frame (see
  `transport.py`'s `SerialTransport`) because raw sprite pixel data can
  contain any byte value, including `\n`, so it can't be newline-delimited
  the way plain gesture labels are.
- **`"udp"`** -- the original WiFi link (`ESP32_HOST`/`ESP32_PORT`), with
  its own ICMP ping monitor and WiFi reconnect diagnostics (see
  `transport.py`'s `UdpTransport`).

Two things worth knowing about the serial transport:

1. **Opening the port resets the board.** Most ESP32 dev boards wire DTR to
   the reset pin, so `gesture.py` starting up reboots the ESP32 -- expect
   the boot banner every time. Harmless; the periodic frame re-send
   restores the panel automatically.
2. **Only one process can own the port at a time.** If the Arduino IDE's
   Serial Monitor has it open, `gesture.py`'s open will fail (the debug
   dashboard shows the OS error). Close the Serial Monitor first.

## Debug dashboard

A second window ("ESP32 Gesture Debug", Tk) opens alongside the OpenCV
video feed, showing: which transport is active and whether it's open;
heartbeat/ping liveness and downtime; the ESP32's own uptime, gesture
count, and boot/reset reason; the raw vs. debounced vs. sent gesture and
debounce progress; per-sprite load status with a reload button; a live
zoomed preview of the exact frame being sent; camera FPS/frame time; and a
scrolling log of everything the ESP32 has reported, with connection-related
lines highlighted. Press `r` in the OpenCV window (or click "Reload
sprites" in the dashboard) to hot-reload `sprites/` without restarting.

If Tk isn't available on your machine, `gesture.py` logs a warning and
keeps running with just the OpenCV window.

## LED matrix geometry

The panel is treated as 16x16, wired serpentine (rows alternate direction)
by default -- see the `MATRIX_*` constants near the top of
`gesture-esp.ino`. If sprite artwork comes out mirrored or rotated on your
physical panel, send/type the `test` gesture label: it lights the top-left
pixel alone, then sweeps rows top-to-bottom, so you can read off which of
`MATRIX_SERPENTINE` / `MATRIX_FLIP_X` / `MATRIX_FLIP_Y` / `MATRIX_ROTATION`
to change (then reflash).

## Hardware

- ESP32 DevKit V1 (or similar)
- WS2812/NeoPixel matrix, 16x16 (256 pixels), data line on GPIO 5 (`LED_PIN`)
- A USB cable (serial transport) and/or WiFi network (UDP transport)
- A webcam on the PC running `gesture.py`
- A stable 5V supply for the ESP32 + LED matrix — a weak USB cable/port is a
  common cause of drops under load; the firmware caps brightness
  (`MAX_BRIGHTNESS`, default 40/255) so a bright sprite frame can't pull
  more current than the supply can deliver, and logs a `BROWNOUT` reset
  reason at boot if a power problem is happening anyway (see below)

## Setup

### ESP32 firmware

1. Open `gesture-esp/gesture-esp.ino` in the Arduino IDE.
2. Install the `Adafruit NeoPixel` library (Library Manager). The `WiFi`,
   `WiFiUdp`, `esp_wifi`, and `esp_system` headers ship with the ESP32
   Arduino core.
3. (Optional, only needed for the UDP transport) copy
   `gesture-esp/.env/secrets.h.example` to `gesture-esp/.env/secrets.h` and
   fill in your real `WIFI_SSID`/`WIFI_PASSWORD`. `secrets.h` is gitignored
   so your password never gets committed -- without it, WiFi connects on a
   bounded timeout (`WIFI_CONNECT_TIMEOUT_MS`) and the board boots and
   serves the serial transport fine with no router at all.
4. Flash it, then open the Serial Monitor at 115200 baud to confirm it
   boots (`Ready. Listening for gesture data on UDP port ... and on this
   USB serial connection.`). If you're using the UDP transport, note the
   IP address it prints. **Close the Serial Monitor afterward** -- see the
   serial gotchas above.

### Python side

```
pip install -r requirements.txt
```

1. Leave `TRANSPORT = "serial"` and set `SERIAL_PORT` to your ESP32's COM
   port (Device Manager on Windows, `ls /dev/tty.*` on macOS, `ls
   /dev/ttyUSB*`/`ttyACM*` on Linux) -- or set `TRANSPORT = "udp"` and
   `ESP32_HOST` to the IP from the Serial Monitor (a static IP or DHCP
   reservation is strongly recommended for that path).
2. Run it:

   ```
   python gesture.py
   ```

   On first run this downloads MediaPipe's pretrained gesture recognizer
   model (`gesture_recognizer.task`, ~8MB) into the project folder.
3. Press `q` in the video window to quit, `r` to hot-reload `sprites/`.

## Connection monitoring & diagnostics

Both transports have diagnostics beyond a simple connected/disconnected
flag, all surfaced in the debug dashboard as well as the console/log file:

- **ESP32**: on a WiFi drop, the LED matrix breathes a dim magenta (a color
  no gesture/sprite uses) instead of freezing or going dark -- unless a
  command has arrived recently over serial, in which case the current
  sprite is left alone instead of being interrupted (`COMMAND_IDLE_BEFORE_PULSE_MS`).
  It escalates from a plain `WiFi.reconnect()` to a full re-associate after
  a few failed attempts, logs the specific WiFi disconnect reason
  (`wifi_err_reason_t`, e.g. `BEACON_TIMEOUT` vs `AUTH_FAIL`) via
  `WiFi.onEvent()`, and reports the reset reason at boot (`BROWNOUT` is the
  tell for a power supply problem).
- **`gesture.py`**: the on-screen overlay and dashboard show elapsed
  downtime (and, on the UDP transport, an estimated reconnect-attempt count
  while `UNREACHABLE`, then the ESP32's real attempt count once it
  recovers). Every connection-relevant event (drops, recoveries, disconnect
  reasons) is also appended to `esp32_connection.log` next to the script --
  including, on the UDP transport, from an independent background ICMP
  ping check, so a full network-level drop can be told apart from the
  ESP32 just going quiet at the app layer. This log is what to check after
  an unattended drop.

## Testing

```
python -m unittest test_gestures.py test_sprites.py test_transport.py
```

- `test_gestures.py` -- the pure classification logic (`classify_gesture`,
  `GestureDebouncer`, `resolve_gesture`, `pick_driving_hand`) against
  synthetic hand landmarks.
- `test_sprites.py` -- sprite loading (image and hex formats), resizing,
  alpha compositing, and `encode_frame()`'s exact byte layout.
- `test_transport.py` -- the serial transport's sync/length/checksum
  framing (including a payload deliberately containing `0x0A` and the sync
  bytes themselves) and a real loopback-UDP round trip.

None of these need a camera or an ESP32. Also useful for checking artwork
specifically:

```
python -m sprites
```

prints a load-status table and a terminal color preview of every sprite in
`sprites/`.

## Tuning knobs

The constants at the top of `gesture.py` control most of the tunable
behavior: transport selection, gesture confidence thresholds,
hand-detection/tracking confidence, debounce frame count, frame refresh
interval, and the ESP32 unreachable-timeout. See the comments next to each
for what turning it up or down does. The matrix geometry and brightness cap
are in the `MATRIX_*`/`MAX_BRIGHTNESS` constants near the top of
`gesture-esp.ino`.
