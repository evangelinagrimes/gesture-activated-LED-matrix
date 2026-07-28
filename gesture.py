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
    pip install mediapipe opencv-python pyserial

On first run, this script auto-downloads MediaPipe's pretrained gesture
recognizer model file (gesture_recognizer.task, ~8MB) into the same folder
as this script.

Run:
    python3 gesture_detection.py

Press 'q' to quit.

Gesture classification is a hybrid: MediaPipe's pretrained recognizer
handles Thumbs Up, Thumbs Down, Peace Sign, Open Palm, and Fist (each with
a confidence score, gated by MIN_GESTURE_CONFIDENCE below). Anything it
doesn't recognize falls back to hand-written geometric rules in
`classify_gesture()`, which also cover a few extra gestures the pretrained
model has no category for: Pointing, OK Sign, Rock On.

You can extend `classify_gesture()` with your own logic/gestures.

--------------------------------------------------------------------------
ESP32 serial protocol
--------------------------------------------------------------------------
Detected gestures are sent to the ESP32 over USB serial as newline-
terminated, UTF-8 strings (e.g. b"thumbs_up\n"). Keep this label set in
sync with whatever the Arduino sketch switches on:

    thumbs_up
    thumbs_down
    peace
    open_palm
    fist
    none

Any gesture classified by `classify_gesture()` that isn't one of the
above (e.g. "Pointing", "OK Sign", "Rock On", "Unknown") is reported to
the ESP32 as "none". A gesture must hold steady for GESTURE_STABLE_FRAMES
consecutive frames before it is sent, to avoid chattering the serial line
on single-frame misclassifications.
--------------------------------------------------------------------------
"""

import os
import math
import time
import atexit
import urllib.request

import cv2
import mediapipe as mp
import serial
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gesture_recognizer.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)

# Serial connection to the ESP32. On Windows this looks like "COM3"; on
# Mac/Linux it looks like "/dev/tty.usbserial-XXXX" or "/dev/ttyUSB0".
SERIAL_PORT = "COM4"
BAUD_RATE = 115200

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

ser = None


def open_serial():
    """Open the serial connection to the ESP32, if possible.

    Failing to connect is non-fatal: gesture detection should keep working
    even when the ESP32 isn't plugged in.
    """
    global ser
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # give the ESP32 time to reset and establish the connection
        print(f"Connected to ESP32 on {SERIAL_PORT}")
    except serial.SerialException as e:
        print(f"Warning: could not open serial port {SERIAL_PORT}: {e}")
        print("Continuing without ESP32 connection.")
        ser = None


def close_serial():
    """Close the serial connection, if it's open."""
    if ser is not None and ser.is_open:
        ser.close()
        print("Serial connection closed.")


def send_gesture(gesture_name: str):
    """Send a gesture label to the ESP32 as a newline-terminated UTF-8 string."""
    if ser is None:
        return
    try:
        ser.write(f"{gesture_name}\n".encode("utf-8"))
        print(f"Sent gesture '{gesture_name}' to ESP32 on {SERIAL_PORT}")
    except (serial.SerialException, OSError) as e:
        print(f"Warning: failed to send gesture over serial: {e}")


def read_esp32_output():
    """Print any buffered debug/status lines the ESP32 has sent back.

    Non-blocking: only reads while ser.in_waiting reports bytes already
    sitting in the buffer, so this never stalls the detection loop.
    """
    if ser is None:
        return
    try:
        while ser.in_waiting > 0:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                print(f"[ESP32] {line}")
    except (serial.SerialException, OSError) as e:
        print(f"Warning: failed to read from serial: {e}")

# Landmark indices for fingertips and their corresponding lower joints
FINGER_TIPS = [4, 8, 12, 16, 20]        # thumb, index, middle, ring, pinky
FINGER_PIPS = [3, 6, 10, 14, 18]        # joint just below each tip


def ensure_model():
    """Download the gesture recognizer model file if it isn't already present."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand gesture recognizer model (one-time, ~8MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded to", MODEL_PATH)


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def hand_scale(landmarks):
    """Reference length (wrist -> middle-finger MCP) for normalizing distances."""
    return max(distance(landmarks[0], landmarks[9]), 1e-6)


def fingers_up(landmarks):
    """Return a list of booleans [thumb, index, middle, ring, pinky]
    indicating whether each finger is extended.

    Uses distance-from-palm rather than raw y-coordinates so this holds up
    regardless of how the hand is rotated or tilted toward the camera.
    """
    fingers = []

    # Thumb: a folded thumb curls in toward the pinky side of the palm, so
    # compare distance to the pinky MCP rather than the wrist.
    palm_ref = landmarks[17]
    fingers.append(distance(landmarks[4], palm_ref) > distance(landmarks[3], palm_ref))

    # Other four fingers: extended when the tip is farther from the wrist
    # than its pip joint is.
    wrist = landmarks[0]
    for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        fingers.append(distance(landmarks[tip], wrist) > distance(landmarks[pip], wrist))

    return fingers


def classify_gesture(landmarks):
    f = fingers_up(landmarks)
    thumb, index, middle, ring, pinky = f
    count_extended = sum(f)
    scale = hand_scale(landmarks)

    # OK sign: thumb tip and index tip close together, other fingers extended
    if distance(landmarks[4], landmarks[8]) / scale < OK_SIGN_RATIO and middle and ring and pinky:
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


def resolve_gesture(gestures, landmarks):
    """Return (esp32_label, display_text) for one hand.

    Prefers the pretrained recognizer when it is confident; otherwise falls
    back to the geometric rules, which cover the gestures it has no category
    for (OK Sign, Rock On, Pointing).
    """
    if gestures:
        top = gestures[0]  # highest-scoring category for this hand
        if top.category_name in MP_GESTURE_LABELS and top.score >= MIN_GESTURE_CONFIDENCE:
            return MP_GESTURE_LABELS[top.category_name], f"{top.category_name} {top.score:.2f}"

    gesture = classify_gesture(landmarks)
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
    they don't chatter the ESP32 serial line.
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


def draw_landmarks(frame, landmarks):
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (255, 255, 255), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)


def main():
    ensure_model()
    open_serial()
    atexit.register(close_serial)

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    start_time = time.perf_counter()
    last_timestamp_ms = 0
    last_sent_gesture = None
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

            # Real elapsed time, forced strictly increasing (the VIDEO-mode
            # tracker requires monotonically increasing timestamps).
            timestamp_ms = max(int((time.perf_counter() - start_time) * 1000), last_timestamp_ms + 1)
            last_timestamp_ms = timestamp_ms
            result = recognizer.recognize_for_video(mp_image, timestamp_ms)

            esp32_gesture = "none"
            driving_hand = pick_driving_hand(result)

            if result.hand_landmarks:
                for i, (landmarks, handedness) in enumerate(zip(result.hand_landmarks, result.handedness)):
                    label = handedness[0].category_name  # "Left" / "Right"
                    gestures = result.gestures[i] if i < len(result.gestures) else []
                    esp32_label, display_text = resolve_gesture(gestures, landmarks)

                    if i == driving_hand:
                        esp32_gesture = esp32_label

                    draw_landmarks(frame, landmarks)

                    h, w, _ = frame.shape
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

            cv2.putText(
                frame,
                f"ESP32: {last_sent_gesture or '...'}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Gesture Detection (press 'q' to quit)", frame)
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("Interrupted, shutting down.")
    finally:
        recognizer.close()
        cap.release()
        cv2.destroyAllWindows()
        read_esp32_output()  # flush any remaining buffered ESP32 messages
        close_serial()


if __name__ == "__main__":
    main()
