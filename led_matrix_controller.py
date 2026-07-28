"""
LED Matrix Controller
======================
Drives a NeoPixel LED matrix over WiFi/UDP from a PC in two ways:

1. Real-time hand gesture detection (MediaPipe Tasks GestureRecognizer +
   OpenCV) -- wave at the webcam, the matrix lights up a color per gesture.
2. Pushing a static image, read from a hex-literal byte dump (.h or .txt),
   to be displayed as a full-frame grayscale picture on the matrix.

NOTE: Gesture detection uses mediapipe.tasks (the current,
actively-maintained API), NOT the old mp.solutions.hands API, which is
deprecated and broken on many recent pip installs (see:
github.com/google-ai-edge/mediapipe issues #6200, #6204, #6261).

Install dependencies:
    pip install mediapipe opencv-python

On first run, this script auto-downloads MediaPipe's pretrained gesture
recognizer model file (gesture_recognizer.task, ~8MB) into the same folder
as this script.

Run:
    python3 led_matrix_controller.py

Press 'q' to quit, 'i' to push the image at IMAGE_PATH to the matrix.

Gesture classification is a hybrid: MediaPipe's pretrained recognizer
handles Thumbs Up, Thumbs Down, Peace Sign, Open Palm, and Fist (each with
a confidence score, gated by MIN_GESTURE_CONFIDENCE below). Anything it
doesn't recognize falls back to hand-written geometric rules in
`classify_gesture()`, which also cover a few extra gestures the pretrained
model has no category for: Pointing, OK Sign, Rock On.

You can extend `classify_gesture()` with your own logic/gestures.

--------------------------------------------------------------------------
ESP32 WiFi protocol (UDP)
--------------------------------------------------------------------------
Two datagram shapes go to ESP32_HOST:ESP32_PORT, told apart purely by size
(see loop() in gesture-esp.ino):

- A short newline-terminated UTF-8 label (e.g. b"thumbs_up\n") is a
  gesture command. Keep this label set in sync with whatever the ESP32
  firmware switches on:

    thumbs_up
    thumbs_down
    peace
    open_palm
    fist
    none

  Any gesture classified by `classify_gesture()` that isn't one of the
  above (e.g. "Pointing", "OK Sign", "Rock On", "Unknown") is reported to
  the ESP32 as "none". A gesture must hold steady for
  GESTURE_STABLE_FRAMES consecutive frames before it is sent, to avoid
  chattering the link on single-frame misclassifications.

- A raw datagram of exactly NUMPIXELS bytes is a full-frame image: one
  grayscale brightness byte per LED, row-major. See send_image() and
  load_image_bytes().

This script also binds LOCAL_UDP_PORT to receive debug/status datagrams
back from the ESP32 (printed with an "[ESP32]" prefix). The ESP32 can
reply to whatever address/port a gesture datagram arrived from (the
standard UDP request/reply pattern), or push a datagram to
LOCAL_UDP_PORT unsolicited at any time, e.g. a boot/WiFi-connected
message before this script has sent any gesture yet.

Connection drops/recoveries (from both the ESP32's own UDP status lines
and an independent ICMP ping check) are appended to esp32_connection.log
next to this script, so a drop that happens unattended can still be
diagnosed afterward -- see CONNECTION_LOG_PATH.
--------------------------------------------------------------------------
"""

import os
import math
import re
import time
import atexit
import socket
import logging
import subprocess
import threading
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gesture_recognizer.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)

# WiFi UDP connection to the ESP32. Fill in your ESP32's IP (a static IP or
# DHCP reservation on your router is strongly recommended -- if it moves,
# gestures silently go nowhere).
ESP32_HOST = "192.168.8.182"   # <-- set to your ESP32's IP
ESP32_PORT = 4210              # port the ESP32 listens on for gesture commands / images
LOCAL_UDP_PORT = 4211          # port this script listens on for ESP32 debug messages

# Must match NUMPIXELS in gesture-esp.ino -- also doubles as the exact byte
# count an image datagram must be, since that's how the ESP32 tells an
# image frame apart from a gesture command (see displayImage() there).
NUMPIXELS = 256
# Path to the hex-literal image to push with the 'i' key. Point this at
# your own .h or .txt -- load_image_bytes() just scans for `0xNN` tokens,
# so either a `const uint8_t image[] PROGMEM = {0xFF, ...};` style .h file
# or a plain comma-separated hex dump works.
IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "example.h")

# --- Tuning knobs -------------------------------------------------------
# Minimum confidence from the pretrained recognizer before a gesture counts.
# Lower  -> more responsive, more false positives.
MIN_GESTURE_CONFIDENCE = 0.5
# Lower these if hands aren't picked up in poor light or at a distance.
MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
# Consecutive frames a gesture must hold before it is sent to the ESP32.
GESTURE_STABLE_FRAMES = 4
# OK-sign pinch distance, as a fraction of hand size (scale-invariant).
OK_SIGN_RATIO = 0.35
# Minimum thumb offset from the wrist, as a fraction of hand size, before
# committing to Thumbs Up vs Thumbs Down.
THUMB_DIRECTION_MARGIN = 0.3
# Camera capture size. MediaPipe downsamples internally regardless, so a
# higher capture resolution just costs more per-frame CPU for nothing.
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
# Hands to track per frame. Only one hand ever drives ESP32 output
# (pick_driving_hand), so tracking a second hand costs a full extra
# inference pass for no effect unless you want the on-screen overlay for
# a second hand too.
NUM_HANDS = 1
# Ceiling on ESP32 messages drained per read_esp32_output() call, so a
# chatty or malfunctioning ESP32 can't stall the frame loop indefinitely.
MAX_ESP32_MESSAGES_PER_READ = 20
# How long without hearing anything from the ESP32 (it heartbeats every
# HEARTBEAT_INTERVAL=5s in gesture-esp.ino) before treating it as
# unreachable. A bit over 2x that, to tolerate a couple of dropped
# heartbeats before flagging it.
ESP32_TIMEOUT_S = 12.0
# Must match WIFI_RECONNECT_INTERVAL in gesture-esp.ino. Used only to
# estimate how many reconnect attempts the ESP32 has likely made while
# unreachable -- there's no way to hear the real count until it reconnects,
# since no network path exists while WiFi is down.
ESP32_RECONNECT_INTERVAL_S = 5.0
# How long to keep showing the ESP32's own reconnect report (attempts +
# downtime, sent once right after it reconnects) before reverting to the
# normal gesture status line.
RECONNECT_NOTE_DISPLAY_S = 5.0
# How often the background thread pings ESP32_HOST at the OS/ICMP level.
# This is independent of the UDP heartbeat: if the ESP32 has actually
# dropped off the network (vs. just gone quiet at the app layer), ICMP will
# fail immediately rather than waiting out ESP32_TIMEOUT_S, and it also
# still works if the ESP32's firmware is wedged but the network stack isn't.
PING_INTERVAL_S = 3.0
PING_TIMEOUT_MS = 1000

# Canned categories from MediaPipe's pretrained gesture recognizer. The
# unmapped ones ("None", "Pointing_Up", "ILoveYou") fall through to the
# geometric rules in classify_gesture().
MP_GESTURE_LABELS = {
    "Thumb_Up": "thumbs_up",
    "Thumb_Down": "thumbs_down",
    "Victory": "peace",
    "Open_Palm": "open_palm",
    "Closed_Fist": "fist",
}

# Maps classify_gesture() output to the label strings the ESP32 expects.
# Anything not listed here (e.g. "Pointing", "OK Sign", "Rock On",
# "Unknown") is sent as "none".
GESTURE_LABELS = {
    "Thumbs Up": "thumbs_up",
    "Thumbs Down": "thumbs_down",
    "Peace Sign": "peace",
    "Open Palm": "open_palm",
    "Fist": "fist",
}

sock = None
last_esp32_contact = None  # time.monotonic() of the last datagram received from the ESP32

# Parses the ESP32's "WiFi reconnected after N attempt(s), down for Xs"
# status line (see maintainWiFi() in gesture-esp.ino).
RECONNECT_REPORT_RE = re.compile(r"^WiFi reconnected after (\d+) attempt")
last_reconnect_report = None      # (attempts: int, message: str) from the ESP32's own count
last_reconnect_report_time = None  # time.monotonic() it arrived

# Persistent record of connection failures/recoveries, independent of the
# on-screen overlay, so a drop that happens while no one's watching the
# window can still be diagnosed afterward. Kept separate from the console
# print()s (which include high-frequency stuff like heartbeats/gestures) --
# this file is connection-events only.
CONNECTION_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "esp32_connection.log")
conn_logger = logging.getLogger("esp32_connection")
conn_logger.setLevel(logging.INFO)
conn_logger.propagate = False
_log_handler = logging.FileHandler(CONNECTION_LOG_PATH)
_log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
conn_logger.addHandler(_log_handler)

# Any ESP32 status line containing one of these (case-insensitive) is
# connection-relevant and gets a copy in CONNECTION_LOG_PATH, not just the
# console. Matches the wording used in gesture-esp.ino's broadcastStatus()
# calls (boot/reset-reason, "WiFi connection lost", "WiFi reconnected...").
CONNECTION_LOG_KEYWORDS = ("wifi", "reset reason")


def _is_connection_related(line: str) -> bool:
    lower = line.lower()
    return any(keyword in lower for keyword in CONNECTION_LOG_KEYWORDS)


# --- Independent ICMP ping monitor --------------------------------------
# Runs on its own thread so it never blocks the frame loop. This is
# deliberately separate from the UDP-heartbeat-based esp32_seems_connected()
# check above: that one only tells you the *application* has gone quiet,
# which can't distinguish "ESP32 fell off the network" from "ESP32 is alive
# but its UDP send is failing" -- a raw ICMP ping tells you whether the host
# is reachable on the network at all, independent of anything the ESP32
# firmware is doing.
ping_ok = None  # None = no ping attempted yet
_ping_lock = threading.Lock()
_ping_stop = threading.Event()


def _ping_once(host: str, timeout_ms: int) -> bool:
    if os.name == "nt":
        cmd = ["ping", "-n", "1", "-w", str(timeout_ms), host]
    else:
        cmd = ["ping", "-c", "1", "-W", str(max(1, timeout_ms // 1000)), host]
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=(timeout_ms / 1000) + 2,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def ping_monitor_loop():
    """Background thread body: ping ESP32_HOST periodically, logging transitions."""
    global ping_ok
    while not _ping_stop.is_set():
        ok = _ping_once(ESP32_HOST, PING_TIMEOUT_MS)
        with _ping_lock:
            first_result = ping_ok is None
            changed = not first_result and ok != ping_ok
            ping_ok = ok
        if first_result:
            conn_logger.info(f"[PING] Initial check: {ESP32_HOST} is "
                              f"{'reachable' if ok else 'NOT reachable'}")
        elif changed:
            if ok:
                conn_logger.info(f"[PING] {ESP32_HOST} is reachable again")
            else:
                conn_logger.warning(f"[PING] {ESP32_HOST} stopped responding to ICMP "
                                     f"(host off the network, WiFi dropped, or ICMP blocked)")
        _ping_stop.wait(PING_INTERVAL_S)


def open_esp32_link():
    """Open the UDP socket used to talk to the ESP32, if possible.

    Failing to bind is non-fatal: gesture detection should keep working even
    when the ESP32 isn't reachable. UDP has no handshake, so unlike the old
    serial connection this always "succeeds" unless the local port is
    already in use.
    """
    global sock
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("", LOCAL_UDP_PORT))
        sock.setblocking(False)
        print(f"Listening for ESP32 on UDP port {LOCAL_UDP_PORT}, "
              f"sending gestures to {ESP32_HOST}:{ESP32_PORT}")
    except OSError as e:
        print(f"Warning: could not open UDP socket on port {LOCAL_UDP_PORT}: {e}")
        print("Continuing without ESP32 connection.")
        sock = None


def close_esp32_link():
    """Close the UDP socket, if it's open."""
    if sock is not None:
        sock.close()
        print("ESP32 connection closed.")


def send_gesture(gesture_name: str):
    """Send a gesture label to the ESP32 as a newline-terminated UTF-8 UDP datagram."""
    if sock is None:
        return
    try:
        sock.sendto(f"{gesture_name}\n".encode("utf-8"), (ESP32_HOST, ESP32_PORT))
        print(f"Sent gesture '{gesture_name}' to ESP32 at {ESP32_HOST}:{ESP32_PORT}")
    except OSError as e:
        print(f"Warning: failed to send gesture over UDP: {e}")


# Matches a hex byte literal like 0xFF or 0x0a. Deliberately format-agnostic
# about what's around it, so it works equally on a full C header (variable
# declaration, PROGMEM attribute, braces, comments) or a bare comma/newline
# separated hex dump -- it just pulls out every `0xNN` token in the file.
HEX_TOKEN_RE = re.compile(r"0[xX][0-9A-Fa-f]{2}")


def load_image_bytes(path: str) -> bytes:
    """Extract a flat grayscale byte-per-pixel image from a hex-literal file.

    Expects exactly NUMPIXELS bytes (one brightness value per LED, row-major)
    -- raises ValueError with a count mismatch if the file doesn't match, so
    a wrong/partial file fails loudly instead of drawing a corrupted image.
    """
    with open(path, "r") as f:
        text = f.read()
    values = [int(tok, 16) for tok in HEX_TOKEN_RE.findall(text)]
    if len(values) != NUMPIXELS:
        raise ValueError(
            f"found {len(values)} hex byte(s), expected exactly {NUMPIXELS} "
            f"(one grayscale byte per LED)"
        )
    return bytes(values)


def send_image(path: str = IMAGE_PATH):
    """Push a full-frame grayscale image to the ESP32 as a single raw UDP
    datagram of NUMPIXELS bytes -- no text framing, since the ESP32 tells
    this apart from a gesture command purely by packet size (see
    displayImage() in gesture-esp.ino).
    """
    if sock is None:
        return
    try:
        image_bytes = load_image_bytes(path)
    except (OSError, ValueError) as e:
        print(f"Warning: could not load image from {path}: {e}")
        return
    try:
        sock.sendto(image_bytes, (ESP32_HOST, ESP32_PORT))
        print(f"Sent image frame ({len(image_bytes)} px) from {path} to ESP32 at {ESP32_HOST}:{ESP32_PORT}")
    except OSError as e:
        print(f"Warning: failed to send image over UDP: {e}")


def read_esp32_output():
    """Print any buffered debug/status datagrams the ESP32 has sent back.

    Non-blocking: a socket with setblocking(False) raises BlockingIOError
    immediately when no datagram is waiting, so this never stalls the
    detection loop.
    """
    global last_esp32_contact, last_reconnect_report, last_reconnect_report_time
    if sock is None:
        return
    for _ in range(MAX_ESP32_MESSAGES_PER_READ):
        try:
            data, _addr = sock.recvfrom(1024)
        except BlockingIOError:
            break
        except ConnectionResetError:
            # Windows-only quirk: sending a UDP datagram to a port with no
            # listener can make a *later* recv on this socket raise
            # WinError 10054, even though UDP has no real "connection" to
            # reset. It just means nothing was there to read -- same as
            # BlockingIOError for our purposes.
            break
        except OSError as e:
            print(f"Warning: failed to read from ESP32: {e}")
            break
        last_esp32_contact = time.monotonic()  # any datagram at all counts as "still alive"
        line = data.decode("utf-8", errors="ignore").strip()
        if line:
            print(f"[ESP32] {line}")
            if _is_connection_related(line):
                conn_logger.info(f"[ESP32] {line}")
            match = RECONNECT_REPORT_RE.match(line)
            if match:
                last_reconnect_report = (int(match.group(1)), line)
                last_reconnect_report_time = time.monotonic()


def esp32_seems_connected():
    """Whether the ESP32 has said anything within ESP32_TIMEOUT_S.

    Returns None if we've never heard from it at all, which is distinct
    from having heard from it before and then gone quiet.
    """
    if last_esp32_contact is None:
        return None
    return (time.monotonic() - last_esp32_contact) < ESP32_TIMEOUT_S

# Landmark indices for fingertips and their corresponding lower joints
FINGER_TIPS = [4, 8, 12, 16, 20]        # thumb, index, middle, ring, pinky
FINGER_PIPS = [3, 6, 10, 14, 18]        # joint just below each tip


def ensure_model():
    """Download the gesture recognizer model file if it isn't already present.

    Downloads to a temp file and renames on success, so an interrupted
    download (Ctrl+C, dropped connection) can't leave a truncated .task file
    behind that silently poisons every later run.
    """
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading hand gesture recognizer model (one-time, ~8MB)...")
    partial_path = MODEL_PATH + ".part"
    try:
        urllib.request.urlretrieve(MODEL_URL, partial_path)
        os.replace(partial_path, MODEL_PATH)
        print("Model downloaded to", MODEL_PATH)
    except BaseException:
        if os.path.exists(partial_path):
            os.remove(partial_path)
        raise


def distance(a, b, aspect=1.0):
    """Euclidean distance between two landmarks.

    Normalized landmark x/y are scaled by frame width/height independently,
    so x must be converted into y-units (aspect = width / height) before the
    distances are comparable in any orientation.
    """
    return math.hypot((a.x - b.x) * aspect, a.y - b.y)


def hand_scale(landmarks, aspect=1.0):
    """Reference length (wrist -> middle-finger MCP) for normalizing distances."""
    return max(distance(landmarks[0], landmarks[9], aspect), 1e-6)


def fingers_up(landmarks, aspect=1.0):
    """Return a list of booleans [thumb, index, middle, ring, pinky]
    indicating whether each finger is extended.

    Uses distance-from-palm rather than raw y-coordinates so this holds up
    regardless of how the hand is rotated or tilted toward the camera.
    """
    fingers = []

    # Thumb: a folded thumb curls in toward the pinky side of the palm, so
    # compare distance to the pinky MCP rather than the wrist.
    palm_ref = landmarks[17]
    fingers.append(distance(landmarks[4], palm_ref, aspect) > distance(landmarks[3], palm_ref, aspect))

    # Other four fingers: extended when the tip is farther from the wrist
    # than its pip joint is.
    wrist = landmarks[0]
    for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        fingers.append(distance(landmarks[tip], wrist, aspect) > distance(landmarks[pip], wrist, aspect))

    return fingers


def classify_gesture(landmarks, aspect=1.0):
    f = fingers_up(landmarks, aspect)
    thumb, index, middle, ring, pinky = f
    count_extended = sum(f)
    scale = hand_scale(landmarks, aspect)

    # OK sign: thumb tip and index tip close together, index curled (an
    # extended index is what actually distinguishes an open palm), and the
    # other three fingers extended.
    if (distance(landmarks[4], landmarks[8], aspect) / scale < OK_SIGN_RATIO
            and not index and middle and ring and pinky):
        return "OK Sign"

    if count_extended == 0:
        return "Fist"
    if count_extended == 5:
        return "Open Palm"
    if index and middle and not ring and not pinky and not thumb:
        return "Peace Sign"
    if index and not middle and not ring and not pinky and not thumb:
        return "Pointing"
    if thumb and pinky and not index and not middle and not ring:
        return "Rock On"
    if thumb and not index and not middle and not ring and not pinky:
        # Thumb up vs down based on vertical position of thumb tip vs wrist,
        # with a deadband so a near-horizontal thumb doesn't get guessed at.
        offset = landmarks[0].y - landmarks[4].y  # positive = thumb above wrist
        margin = THUMB_DIRECTION_MARGIN * scale
        if offset > margin:
            return "Thumbs Up"
        elif offset < -margin:
            return "Thumbs Down"
        return "Unknown"

    return "Unknown"


def resolve_gesture(gestures, landmarks, aspect=1.0):
    """Return (esp32_label, display_text) for one hand.

    Prefers the pretrained recognizer when it is confident; otherwise falls
    back to the geometric rules, which cover the gestures it has no category
    for (OK Sign, Rock On, Pointing).
    """
    if gestures:
        top = gestures[0]  # highest-scoring category for this hand
        if top.category_name in MP_GESTURE_LABELS and top.score >= MIN_GESTURE_CONFIDENCE:
            return MP_GESTURE_LABELS[top.category_name], f"{top.category_name} {top.score:.2f}"

    gesture = classify_gesture(landmarks, aspect)
    return GESTURE_LABELS.get(gesture, "none"), gesture


def pick_driving_hand(result):
    """Index of the hand that should drive ESP32 output: the largest
    apparent hand, i.e. the one nearest the camera."""
    if not result.hand_landmarks:
        return None
    return max(range(len(result.hand_landmarks)),
               key=lambda i: hand_scale(result.hand_landmarks[i]))


class GestureDebouncer:
    """Require the same label on N consecutive frames before it counts.

    Filters single-frame misclassifications and brief tracking dropouts so
    they don't chatter the ESP32 WiFi link.
    """

    def __init__(self, stable_frames):
        self.stable_frames = stable_frames
        self._candidate = None
        self._run_length = 0
        self._confirmed = None

    def update(self, label):
        if label == self._candidate:
            self._run_length += 1
        else:
            self._candidate = label
            self._run_length = 1
        if self._run_length >= self.stable_frames:
            self._confirmed = self._candidate
        return self._confirmed


# Hand connections for drawing (pairs of landmark indices), same topology
# used by MediaPipe's legacy drawing utils.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]


def draw_landmarks(frame, landmarks, w, h):
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (255, 255, 255), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)


def main():
    ensure_model()
    open_esp32_link()
    atexit.register(close_esp32_link)

    conn_logger.info(f"Session started. Connection events for {ESP32_HOST} logged to {CONNECTION_LOG_PATH}")
    ping_thread = threading.Thread(target=ping_monitor_loop, daemon=True)
    ping_thread.start()
    atexit.register(_ping_stop.set)

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)

    # CAP_DSHOW avoids a multi-second startup stall the default MSMF backend
    # is prone to on Windows.
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    start_time = time.perf_counter()
    last_timestamp_ms = 0
    last_sent_gesture = None
    esp32_was_connected = False  # tracks the previous frame's state, to log the transition once
    esp32_ever_dropped = False  # guards the "reachable again" log so it doesn't fire on first-ever contact
    debouncer = GestureDebouncer(GESTURE_STABLE_FRAMES)

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            frame = cv2.flip(frame, 1)  # mirror for natural interaction
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            h, w, _ = frame.shape
            aspect = w / h

            # Real elapsed time, forced strictly increasing (the VIDEO-mode
            # tracker requires monotonically increasing timestamps).
            timestamp_ms = max(int((time.perf_counter() - start_time) * 1000), last_timestamp_ms + 1)
            last_timestamp_ms = timestamp_ms
            result = recognizer.recognize_for_video(mp_image, timestamp_ms)

            esp32_gesture = "none"
            driving_hand = pick_driving_hand(result)

            for i, landmarks in enumerate(result.hand_landmarks):
                # The frame is mirrored above before inference, so MediaPipe
                # sees a mirrored world and its handedness call is reversed
                # relative to the physical hand.
                mp_label = result.handedness[i][0].category_name if i < len(result.handedness) else None
                label = {"Left": "Right", "Right": "Left"}.get(mp_label, "?")
                gestures = result.gestures[i] if i < len(result.gestures) else []
                esp32_label, display_text = resolve_gesture(gestures, landmarks, aspect)

                if i == driving_hand:
                    esp32_gesture = esp32_label

                draw_landmarks(frame, landmarks, w, h)

                wrist = landmarks[0]
                x, y = int(wrist.x * w), int(wrist.y * h)
                marker = "-> " if i == driving_hand else ""
                cv2.putText(
                    frame,
                    f"{marker}{label}: {display_text}",
                    (x - 40, y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            confirmed = debouncer.update(esp32_gesture)
            if confirmed is not None and confirmed != last_sent_gesture:
                send_gesture(confirmed)
                last_sent_gesture = confirmed

            read_esp32_output()

            esp32_connected = esp32_seems_connected()
            if esp32_connected is False and esp32_was_connected:
                msg = f"ESP32 has not responded in over {ESP32_TIMEOUT_S:.0f}s -- check its WiFi connection and power."
                print(f"Warning: {msg}")
                conn_logger.warning(f"[UDP] {msg}")
                esp32_ever_dropped = True
            elif esp32_connected and not esp32_was_connected and esp32_ever_dropped:
                # Guarded by esp32_ever_dropped so this doesn't fire on the
                # very first contact of the session (not a "recovery").
                conn_logger.info("[UDP] ESP32 reachable again")
                esp32_ever_dropped = False
            esp32_was_connected = bool(esp32_connected)

            if esp32_connected is None:
                status_text, status_color = "ESP32: waiting for first contact...", (0, 200, 255)
            elif esp32_connected:
                status_text, status_color = f"ESP32: {last_sent_gesture or '...'}", (0, 200, 255)
                # Briefly show the ESP32's own reconnect telemetry (real
                # attempt count + downtime) right after it comes back.
                if (last_reconnect_report_time is not None
                        and time.monotonic() - last_reconnect_report_time < RECONNECT_NOTE_DISPLAY_S):
                    attempts, _msg = last_reconnect_report
                    status_text += f" (reconnected after {attempts} attempt(s))"
            else:
                downtime_s = time.monotonic() - last_esp32_contact
                # No network path exists to ask the ESP32 for its real
                # attempt count while it's down, so estimate from elapsed
                # downtime and its known retry interval.
                est_attempts = int(downtime_s // ESP32_RECONNECT_INTERVAL_S)
                status_text = f"ESP32: UNREACHABLE ({downtime_s:.0f}s, ~{est_attempts} reconnect attempts)"
                status_color = (0, 0, 255)  # BGR red

            cv2.putText(
                frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("LED Matrix Controller (q: quit, i: send image)", frame)
            key = cv2.waitKey(5) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("i"):
                send_image(IMAGE_PATH)
    except KeyboardInterrupt:
        print("Interrupted, shutting down.")
    finally:
        recognizer.close()
        cap.release()
        cv2.destroyAllWindows()
        read_esp32_output()  # flush any remaining buffered ESP32 messages
        close_esp32_link()


if __name__ == "__main__":
    main()
