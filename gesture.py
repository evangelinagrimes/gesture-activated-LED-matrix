"""
Real-Time Hand Gesture Detection
==================================
Uses MediaPipe's new Tasks API (HandLandmarker) + OpenCV to detect hand
gestures live from your webcam.

NOTE: This uses mediapipe.tasks (the current, actively-maintained API),
NOT the old mp.solutions.hands API, which is deprecated and broken on
many recent pip installs (see: github.com/google-ai-edge/mediapipe issues
#6200, #6204, #6261). This version avoids that problem entirely.

Install dependencies:
    pip install mediapipe opencv-python pyserial

On first run, this script auto-downloads the hand landmark model file
(hand_landmarker.task, ~7MB) into the same folder as this script.

Run:
    python3 gesture_detection.py

Press 'q' to quit.

Detected gestures: Fist, Open Palm, Thumbs Up, Thumbs Down, Peace Sign,
Pointing (index finger), OK Sign, Rock On.

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
the ESP32 as "none".
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

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Serial connection to the ESP32. On Windows this looks like "COM3"; on
# Mac/Linux it looks like "/dev/tty.usbserial-XXXX" or "/dev/ttyUSB0".
SERIAL_PORT = "COM4"
BAUD_RATE = 115200

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
    """Download the hand landmark model file if it isn't already present."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (one-time, ~7MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded to", MODEL_PATH)


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def fingers_up(landmarks, handedness_label):
    """Return a list of booleans [thumb, index, middle, ring, pinky]
    indicating whether each finger is extended."""
    fingers = []

    # Thumb: compare x-position relative to handedness (mirrored image)
    if handedness_label == "Right":
        fingers.append(landmarks[4].x < landmarks[3].x)
    else:
        fingers.append(landmarks[4].x > landmarks[3].x)

    # Other four fingers: tip above (smaller y) than pip joint = extended
    for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        fingers.append(landmarks[tip].y < landmarks[pip].y)

    return fingers


def classify_gesture(landmarks, handedness_label):
    f = fingers_up(landmarks, handedness_label)
    thumb, index, middle, ring, pinky = f
    count_extended = sum(f)

    # OK sign: thumb tip and index tip close together, other fingers extended
    if distance(landmarks[4], landmarks[8]) < 0.05 and middle and ring and pinky:
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
        # Thumb up vs down based on vertical position of thumb tip vs wrist
        if landmarks[4].y < landmarks[0].y:
            return "Thumbs Up"
        else:
            return "Thumbs Down"

    return "Unknown"


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
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    frame_timestamp_ms = 0
    last_sent_gesture = None

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            frame = cv2.flip(frame, 1)  # mirror for natural interaction
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            frame_timestamp_ms += 33  # roughly assumes ~30fps; must be increasing
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            esp32_gesture = "none"

            if result.hand_landmarks:
                for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                    label = handedness[0].category_name  # "Left" / "Right"
                    gesture = classify_gesture(landmarks, label)

                    # Only the first detected hand drives the ESP32 output.
                    if esp32_gesture == "none":
                        esp32_gesture = GESTURE_LABELS.get(gesture, "none")

                    draw_landmarks(frame, landmarks)

                    h, w, _ = frame.shape
                    wrist = landmarks[0]
                    x, y = int(wrist.x * w), int(wrist.y * h)
                    cv2.putText(
                        frame,
                        f"{label}: {gesture}",
                        (x - 40, y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

            if esp32_gesture != last_sent_gesture:
                send_gesture(esp32_gesture)
                last_sent_gesture = esp32_gesture

            read_esp32_output()

            cv2.imshow("Gesture Detection (press 'q' to quit)", frame)
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("Interrupted, shutting down.")
    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()
        read_esp32_output()  # flush any remaining buffered ESP32 messages
        close_serial()


if __name__ == "__main__":
    main()