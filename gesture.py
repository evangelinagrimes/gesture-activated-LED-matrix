"""
Real-Time Hand Gesture Detection
==================================
Uses MediaPipe's new Tasks API (GestureRecognizer) + OpenCV to detect hand
gestures live from your webcam.

NOTE: This uses mediapipe.tasks (the current, actively-maintained API),
NOT the old mp.solutions.hands API, which is deprecated and broken on
many recent pip installs (see: github.com/google-ai-edge/mediapipe issues
#6200, #6204, #6261). This version avoids that problem entirely.

Install dependencies:
    pip install -r requirements.txt

On first run, this script auto-downloads MediaPipe's pretrained gesture
recognizer model file (gesture_recognizer.task, ~8MB) into the same folder
as this script.

Run:
    python3 gesture.py

Press 'q' to quit, 'r' to hot-reload sprites/ without restarting.

Gesture classification is a hybrid: MediaPipe's pretrained recognizer
handles Thumbs Up, Thumbs Down, Peace Sign, Open Palm, and Fist (each with
a confidence score, gated by MIN_GESTURE_CONFIDENCE below). Anything it
doesn't recognize falls back to hand-written geometric rules in
`classify_gesture()`, which also cover a few extra gestures the pretrained
model has no category for: Pointing, OK Sign, Rock On.

You can extend `classify_gesture()` with your own logic/gestures.

--------------------------------------------------------------------------qqqqq
ESP32 transport: USB serial (default) or WiFi UDP
--------------------------------------------------------------------------
Set TRANSPORT below to pick the link. "serial" (default) talks to the
ESP32 over a USB cable (SERIAL_PORT/SERIAL_BAUD) and doesn't depend on WiFi
at all -- see transport.py's SerialTransport. "udp" is the original WiFi
link (ESP32_HOST/ESP32_PORT), with its own ICMP ping monitor and reconnect
diagnostics -- see transport.py's UdpTransport.

Whichever transport is active, gesture.py sends one message per confirmed
gesture change, re-sending the same message every FRAME_REFRESH_S seconds
(and immediately after a reconnect) so a dropped datagram or an ESP32
reboot can't leave stale artwork on the panel indefinitely.

Each message is either:
  - a sprite frame, b"IMG1<label>\\n<768 bytes raw RGB>" (see sprites.py),
    when sprites/<label>.* exists, or
  - a plain label, b"<label>\\n", when it doesn't -- the ESP32 falls back
    to its original flat gesture->color behavior for that label.

Keep the label set in sync with whatever the ESP32 firmware switches on:

    thumbs_up
    thumbs_down
    peace
    open_palm
    fist
    none

Any gesture classified by `classify_gesture()` that isn't one of the
above (e.g. "Pointing", "OK Sign", "Rock On", "Unknown") is reported to
the ESP32 as "none". A gesture must hold steady for GESTURE_STABLE_FRAMES
consecutive frames before it is sent, to avoid chattering the link on
single-frame misclassifications.

Status/debug lines the ESP32 sends back (heartbeats, boot/reset reason,
WiFi reconnect reports) are printed with an "[ESP32]" prefix and surfaced
in the debug dashboard (see dashboard.py) -- a second Tk window opened
alongside the OpenCV video feed.

Connection drops/recoveries are appended to esp32_connection.log next to
this script, so a drop that happens unattended can still be diagnosed
afterward -- see CONNECTION_LOG_PATH.
--------------------------------------------------------------------------
"""

import os
import math
import re
import time
import atexit
import logging
from collections import deque

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import transport
import sprites
import dashboard

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gesture_recognizer.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)

# --- Transport selection -------------------------------------------------
# "serial" (default): USB cable, doesn't depend on WiFi at all.
# "udp": the original WiFi link, with ICMP ping + WiFi reconnect diagnostics.
TRANSPORT = "serial"

# Used when TRANSPORT == "serial".
SERIAL_PORT = "COM3"
SERIAL_BAUD = 115200
# How often to retry opening the port after it's unavailable/dropped (cable
# unplugged, or the ESP32 reset and briefly disappeared from the OS).
SERIAL_REOPEN_INTERVAL_S = 2.0

# Used when TRANSPORT == "udp". A static IP or DHCP reservation on your
# router is strongly recommended -- if it moves, gestures silently go nowhere.
ESP32_HOST = "192.168.8.182"   # <-- set to your ESP32's IP
ESP32_PORT = 4210              # port the ESP32 listens on for gesture commands
LOCAL_UDP_PORT = 4211          # port this script listens on for ESP32 debug messages

# How often to re-send the currently-displayed gesture, regardless of
# whether it changed. UDP can drop a datagram and USB can get unplugged --
# either way, the render is idempotent, so this also self-heals an ESP32
# reboot (including the DTR-triggered reset opening a serial port causes on
# most dev boards) without needing a new gesture to arrive first.
FRAME_REFRESH_S = 2.0

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
# How long without hearing anything from the ESP32 (it heartbeats every
# HEARTBEAT_INTERVAL=5s in gesture-esp.ino, over whichever transport is
# active) before treating it as unreachable. A bit over 2x that, to
# tolerate a couple of dropped heartbeats before flagging it.
ESP32_TIMEOUT_S = 12.0
# Must match WIFI_RECONNECT_INTERVAL in gesture-esp.ino. Used only to
# estimate how many reconnect attempts the ESP32 has likely made while
# unreachable over UDP -- there's no way to hear the real count until it
# reconnects, since no network path exists while WiFi is down.
ESP32_RECONNECT_INTERVAL_S = 5.0
# How long to keep showing the ESP32's own reconnect report (attempts +
# downtime, sent once right after it reconnects) before reverting to the
# normal gesture status line.
RECONNECT_NOTE_DISPLAY_S = 5.0
# How often the background thread pings ESP32_HOST at the OS/ICMP level
# (TRANSPORT == "udp" only -- meaningless over a wired USB link).
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

link = None  # the active Transport instance, set in main()
last_esp32_contact = None  # time.monotonic() of the last line received from the ESP32

# --- ESP32 status-line parsing (transport-agnostic: both SerialTransport
# and UdpTransport hand read_esp32_output() plain decoded lines) ---------
HEARTBEAT_RE = re.compile(r"^\[HEARTBEAT\] Uptime: (\d+)s \| Gestures: (\d+)")
BOOT_RE = re.compile(r"^ESP32 booted\..*Reset reason: (.+)$")
RESET_REASON_RE = re.compile(r"^Reset reason: (.+)$")
LAST_DISCONNECT_RE = re.compile(r"Last disconnect (.+)$")

esp32_uptime_s = None
esp32_gesture_count = None
esp32_boot_reason = None
esp32_last_disconnect_reason = None
reconnects_this_session = 0
messages_received = 0
esp32_log = deque(maxlen=200)  # (text, is_connection_related), oldest first -- feeds the dashboard log

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


def create_transport():
    if TRANSPORT == "serial":
        return transport.SerialTransport(SERIAL_PORT, SERIAL_BAUD, SERIAL_REOPEN_INTERVAL_S)
    elif TRANSPORT == "udp":
        return transport.UdpTransport(ESP32_HOST, ESP32_PORT, LOCAL_UDP_PORT,
                                       PING_INTERVAL_S, int(PING_TIMEOUT_MS), conn_logger)
    else:
        raise ValueError(f"Unknown TRANSPORT {TRANSPORT!r} (expected 'serial' or 'udp')")


def read_esp32_output():
    """Drain and process buffered status/debug lines from the ESP32.

    Transport-agnostic: link.read_lines() already returns decoded,
    newline-split text regardless of whether it came from a UDP datagram or
    the serial port.
    """
    global last_esp32_contact, esp32_uptime_s, esp32_gesture_count
    global esp32_boot_reason, esp32_last_disconnect_reason, reconnects_this_session, messages_received

    for line in link.read_lines():
        last_esp32_contact = time.monotonic()
        messages_received += 1
        print(f"[ESP32] {line}")
        is_conn = _is_connection_related(line)
        esp32_log.append((line, is_conn))
        if is_conn:
            conn_logger.info(f"[ESP32] {line}")

        m = HEARTBEAT_RE.match(line)
        if m:
            esp32_uptime_s = int(m.group(1))
            esp32_gesture_count = int(m.group(2))
            continue

        m = BOOT_RE.match(line) or RESET_REASON_RE.match(line)
        if m:
            esp32_boot_reason = m.group(1)

        if transport.RECONNECT_REPORT_RE.match(line):
            reconnects_this_session += 1
            m2 = LAST_DISCONNECT_RE.search(line)
            if m2:
                esp32_last_disconnect_reason = m2.group(1)


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
        import urllib.request
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
    they don't chatter the ESP32 link.
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

    def progress(self):
        """(candidate, run_length) for the debug dashboard."""
        return self._candidate, self._run_length


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
    global link

    ensure_model()

    link = create_transport()
    link.open()
    atexit.register(link.close)

    sprite_cache, sprite_statuses = sprites.load_all_sprites()
    print(f"Sprites loaded from {sprites.SPRITE_DIR}:")
    for label, status in sprite_statuses.items():
        print(f"  {label:12s} {status}")

    dash = dashboard.Dashboard.create()
    if dash is not None:
        atexit.register(dash.destroy)

    conn_logger.info(f"Session started. Transport={TRANSPORT}. "
                      f"Connection events logged to {CONNECTION_LOG_PATH}")

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
    last_sent_time = None
    last_sent_kind = None
    last_frame_send_time = 0.0
    current_preview_rgb = None
    current_fallback_color = None
    esp32_was_connected = False  # tracks the previous frame's state, to log the transition once
    esp32_ever_dropped = False  # guards the "reachable again" log so it doesn't fire on first-ever contact
    debouncer = GestureDebouncer(GESTURE_STABLE_FRAMES)
    fps = 0.0
    frame_ms = 0.0
    fps_window_start = time.perf_counter()
    fps_frame_count = 0

    def dispatch(label):
        """Send `label` now (a sprite frame if one is loaded, else the
        original plain-label datagram/message) and update all the
        bookkeeping the refresh timer and dashboard need."""
        nonlocal last_sent_gesture, last_sent_time, last_sent_kind
        nonlocal current_preview_rgb, current_fallback_color, last_frame_send_time
        rgb = sprite_cache.get(label)
        if rgb is not None:
            payload = sprites.encode_frame(label, rgb)
            kind = "FRAME"
        else:
            payload = f"{label}\n".encode("utf-8")
            kind = "LABEL (fallback color)"
        link.send(payload)
        print(f"Sent gesture '{label}' via {TRANSPORT} ({kind})")
        last_sent_gesture = label
        last_sent_time = time.monotonic()
        last_sent_kind = kind
        last_frame_send_time = last_sent_time
        current_preview_rgb = rgb
        current_fallback_color = sprites.FALLBACK_COLORS.get(label, (0, 0, 0))

    try:
        while cap.isOpened():
            loop_start = time.perf_counter()
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            frame = cv2.flip(frame, 1)  # mirror for natural interaction
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

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
                dispatch(confirmed)
            elif last_sent_gesture is not None and time.monotonic() - last_frame_send_time >= FRAME_REFRESH_S:
                dispatch(last_sent_gesture)  # idempotent refresh -- see FRAME_REFRESH_S

            read_esp32_output()

            esp32_connected = esp32_seems_connected()
            if esp32_connected is False and esp32_was_connected:
                msg = f"ESP32 has not responded in over {ESP32_TIMEOUT_S:.0f}s -- check its connection and power."
                print(f"Warning: {msg}")
                conn_logger.warning(f"[{TRANSPORT.upper()}] {msg}")
                esp32_ever_dropped = True
            elif esp32_connected and not esp32_was_connected:
                if esp32_ever_dropped:
                    # Guarded by esp32_ever_dropped so this doesn't fire on the
                    # very first contact of the session (not a "recovery").
                    conn_logger.info(f"[{TRANSPORT.upper()}] ESP32 reachable again")
                    esp32_ever_dropped = False
                if last_sent_gesture is not None:
                    # Restore the panel right away instead of waiting up to
                    # FRAME_REFRESH_S after a reconnect.
                    dispatch(last_sent_gesture)
            esp32_was_connected = bool(esp32_connected)

            if esp32_connected is None:
                status_text, status_color = "ESP32: waiting for first contact...", (0, 200, 255)
            elif esp32_connected:
                status_text, status_color = f"ESP32: {last_sent_gesture or '...'}", (0, 200, 255)
                # Briefly show the ESP32's own reconnect telemetry (real
                # attempt count + downtime) right after it comes back --
                # only tracked over the UDP transport.
                report_time = getattr(link, "last_reconnect_report_time", None)
                if report_time is not None and time.monotonic() - report_time < RECONNECT_NOTE_DISPLAY_S:
                    attempts, _msg = link.last_reconnect_report
                    status_text += f" (reconnected after {attempts} attempt(s))"
            else:
                downtime_s = time.monotonic() - last_esp32_contact
                est_attempts = int(downtime_s // ESP32_RECONNECT_INTERVAL_S)
                status_text = f"ESP32: UNREACHABLE ({downtime_s:.0f}s)"
                if TRANSPORT == "udp":
                    # No network path exists to ask the ESP32 for its real
                    # attempt count while WiFi is down, so estimate from
                    # elapsed downtime and its known retry interval.
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

            cv2.imshow("Gesture Detection (press 'q' to quit, 'r' to reload sprites)", frame)

            fps_frame_count += 1
            now = time.perf_counter()
            if now - fps_window_start >= 1.0:
                fps = fps_frame_count / (now - fps_window_start)
                fps_frame_count = 0
                fps_window_start = now
            frame_ms = (time.perf_counter() - loop_start) * 1000

            if dash is not None and not dash.closed:
                link_status = link.status()
                reconnect_report = link_status.get("last_reconnect_report")
                candidate, run_length = debouncer.progress()
                state = dashboard.DashboardState(
                    transport_name=link_status.get("transport", TRANSPORT),
                    transport_address=link_status.get("address", "?"),
                    transport_open=link_status.get("open", False),
                    transport_reopen_attempts=link_status.get("reopen_attempts", 0),
                    transport_last_error=link_status.get("last_error"),
                    frames_sent=link_status.get("frames_sent", 0),
                    bytes_sent=link_status.get("bytes_sent", 0),
                    heartbeat_state=esp32_connected,
                    last_contact_age_s=(time.monotonic() - last_esp32_contact) if last_esp32_contact else None,
                    downtime_s=(time.monotonic() - last_esp32_contact)
                    if (esp32_connected is False and last_esp32_contact) else None,
                    ping_ok=link_status.get("ping_ok"),
                    reconnects_this_session=reconnects_this_session,
                    last_reconnect_report=reconnect_report[1] if reconnect_report else None,
                    last_disconnect_reason=esp32_last_disconnect_reason,
                    esp32_uptime_s=esp32_uptime_s,
                    esp32_gesture_count=esp32_gesture_count,
                    esp32_boot_reason=esp32_boot_reason,
                    raw_label=esp32_gesture,
                    debounce_candidate=candidate,
                    debounce_run_length=run_length,
                    debounce_stable_frames=GESTURE_STABLE_FRAMES,
                    confirmed_label=confirmed,
                    last_sent_label=last_sent_gesture,
                    last_sent_age_s=(time.monotonic() - last_sent_time) if last_sent_time else None,
                    send_kind=last_sent_kind,
                    sprite_statuses=sprite_statuses,
                    preview_rgb=current_preview_rgb,
                    preview_fallback_color=current_fallback_color,
                    fps=fps,
                    frame_ms=frame_ms,
                    messages_sent=link_status.get("frames_sent", 0),
                    messages_received=messages_received,
                    log_lines=list(esp32_log),
                )
                dash.update(state)
                if dash.consume_reload_request():
                    sprite_cache, sprite_statuses = sprites.load_all_sprites()
                    print("Sprites reloaded.")
                    if last_sent_gesture is not None:
                        dispatch(last_sent_gesture)

            key = cv2.waitKey(5) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                sprite_cache, sprite_statuses = sprites.load_all_sprites()
                print("Sprites reloaded.")
                if last_sent_gesture is not None:
                    dispatch(last_sent_gesture)
    except KeyboardInterrupt:
        print("Interrupted, shutting down.")
    finally:
        recognizer.close()
        cap.release()
        cv2.destroyAllWindows()
        read_esp32_output()  # flush any remaining buffered ESP32 messages
        link.close()
        if dash is not None:
            dash.destroy()


if __name__ == "__main__":
    main()
