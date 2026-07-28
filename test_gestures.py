"""
Unit tests for the pure classification functions in led_matrix_controller.py.

No camera or serial hardware required — these build synthetic 21-point
hand landmark sets by hand and feed them straight into classify_gesture()
and friends. Run with:

    python -m unittest test_gestures.py
"""

import math
import unittest

import led_matrix_controller as gesture


class LM:
    """Minimal stand-in for mediapipe's NormalizedLandmark."""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.z = 0.0


def _fill_missing(landmarks, default=(0.5, 0.85)):
    for i in range(21):
        if landmarks[i] is None:
            landmarks[i] = LM(*default)
    return landmarks


def make_hand(thumb=False, index=False, middle=False, ring=False, pinky=False):
    """Build a synthetic upright hand with the given fingers extended.

    Landmark coordinates are "physical" (square-pixel / aspect=1) units:
    correct as-is for aspect=1.0 tests, and convertible to any other aspect
    via to_aspect() below.
    """
    lm = [None] * 21
    lm[0] = LM(0.5, 0.9)    # wrist
    lm[9] = LM(0.5, 0.6)    # middle MCP (defines hand_scale)
    lm[17] = LM(0.6, 0.62)  # pinky MCP (thumb's palm reference)

    if thumb:
        lm[3] = LM(0.35, 0.75)
        lm[4] = LM(0.25, 0.7)    # far from pinky MCP -> extended
    else:
        lm[3] = LM(0.55, 0.63)
        lm[4] = LM(0.58, 0.62)   # close to pinky MCP -> folded

    tips = {"index": (8, 6), "middle": (12, 10), "ring": (16, 14), "pinky": (20, 18)}
    extended_y = {"index": 0.3, "middle": 0.25, "ring": 0.3, "pinky": 0.35}
    states = {"index": index, "middle": middle, "ring": ring, "pinky": pinky}
    for name, (tip, pip) in tips.items():
        lm[pip] = LM(0.5, 0.55)
        lm[tip] = LM(0.5, extended_y[name] if states[name] else 0.58)

    return _fill_missing(lm)


def to_aspect(physical_hand, aspect):
    """Re-express a physical-units hand as normalized landmarks for a frame
    of the given aspect ratio (width / height).

    x_norm = x_physical / aspect, y_norm = y_physical, matching how
    classify_gesture(..., aspect) undoes the distortion: this lets the same
    physical hand shape be classified at any aspect and expect an identical
    result if (and only if) the aspect correction is working.
    """
    return [LM(p.x / aspect, p.y) for p in physical_hand]


class ClassifyGestureTests(unittest.TestCase):
    def test_fist(self):
        hand = make_hand()
        self.assertEqual(gesture.classify_gesture(hand), "Fist")

    def test_open_palm(self):
        hand = make_hand(thumb=True, index=True, middle=True, ring=True, pinky=True)
        self.assertEqual(gesture.classify_gesture(hand), "Open Palm")

    def test_peace_sign(self):
        hand = make_hand(index=True, middle=True)
        self.assertEqual(gesture.classify_gesture(hand), "Peace Sign")

    def test_pointing(self):
        hand = make_hand(index=True)
        self.assertEqual(gesture.classify_gesture(hand), "Pointing")

    def test_rock_on(self):
        hand = make_hand(thumb=True, pinky=True)
        self.assertEqual(gesture.classify_gesture(hand), "Rock On")

    def test_thumbs_up(self):
        hand = [None] * 21
        hand[0] = LM(0.5, 0.9)
        hand[9] = LM(0.5, 0.6)
        hand[17] = LM(0.6, 0.62)
        hand[3] = LM(0.35, 0.75)
        hand[4] = LM(0.25, 0.65)  # above the wrist, far from pinky MCP
        for tip, pip in zip(gesture.FINGER_TIPS[1:], gesture.FINGER_PIPS[1:]):
            hand[pip] = LM(0.5, 0.55)
            hand[tip] = LM(0.5, 0.58)  # curled
        hand = _fill_missing(hand)
        self.assertEqual(gesture.classify_gesture(hand), "Thumbs Up")

    def test_thumbs_down(self):
        hand = [None] * 21
        hand[0] = LM(0.5, 0.9)
        hand[9] = LM(0.5, 0.6)
        hand[17] = LM(0.6, 0.62)
        hand[3] = LM(0.35, 0.95)
        hand[4] = LM(0.25, 1.05)  # below the wrist, far from pinky MCP
        for tip, pip in zip(gesture.FINGER_TIPS[1:], gesture.FINGER_PIPS[1:]):
            hand[pip] = LM(0.5, 0.55)
            hand[tip] = LM(0.5, 0.58)  # curled
        hand = _fill_missing(hand)
        self.assertEqual(gesture.classify_gesture(hand), "Thumbs Down")

    def test_ok_sign(self):
        # Index folded (curled tip/pip from make_hand), middle/ring/pinky
        # extended, thumb tip nudged onto the folded index tip.
        hand = make_hand(thumb=False, index=False, middle=True, ring=True, pinky=True)
        hand[4] = LM(hand[8].x - 0.02, hand[8].y + 0.01)
        self.assertEqual(gesture.classify_gesture(hand), "OK Sign")

    def test_ok_sign_not_confused_with_open_palm(self):
        """C5 regression: an extended index with the thumb resting nearby
        must not be misread as an OK sign — it's an open palm."""
        hand = make_hand(thumb=True, index=True, middle=True, ring=True, pinky=True)
        # Nudge the thumb tip close to the index tip, as an open hand at
        # rest might have it, without curling the index.
        hand[4] = LM(hand[8].x + 0.02, hand[8].y + 0.02)
        result = gesture.classify_gesture(hand)
        self.assertNotEqual(result, "OK Sign")


class AspectInvarianceTests(unittest.TestCase):
    """The whole point of C1: the same physical hand must classify
    identically regardless of the frame's aspect ratio."""

    CASES = [
        ("Fist", dict()),
        ("Open Palm", dict(thumb=True, index=True, middle=True, ring=True, pinky=True)),
        ("Peace Sign", dict(index=True, middle=True)),
        ("Pointing", dict(index=True)),
        ("Rock On", dict(thumb=True, pinky=True)),
    ]

    def test_aspect_invariance_across_common_ratios(self):
        for name, states in self.CASES:
            physical_hand = make_hand(**states)
            baseline = gesture.classify_gesture(physical_hand, aspect=1.0)
            self.assertEqual(baseline, name)  # sanity: baseline matches the label
            for aspect in (4 / 3, 16 / 9, 9 / 16):
                reprojected = to_aspect(physical_hand, aspect)
                result = gesture.classify_gesture(reprojected, aspect=aspect)
                self.assertEqual(
                    result, baseline,
                    f"{name} classified as {result!r} at aspect={aspect:.3f}, "
                    f"expected {baseline!r} (same physical hand)",
                )


class RotationInvarianceTests(unittest.TestCase):
    def test_open_palm_rotated_90_degrees(self):
        hand = make_hand(thumb=True, index=True, middle=True, ring=True, pinky=True)
        origin = (hand[0].x, hand[0].y)
        rotated = _rotate_about(hand, origin, math.pi / 2)
        self.assertEqual(gesture.classify_gesture(hand), "Open Palm")
        self.assertEqual(gesture.classify_gesture(rotated), "Open Palm")


def _rotate_about(landmarks, origin, angle_rad):
    ox, oy = origin
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    out = []
    for p in landmarks:
        dx, dy = p.x - ox, p.y - oy
        out.append(LM(ox + dx * cos_a - dy * sin_a, oy + dx * sin_a + dy * cos_a))
    return out


class GestureDebouncerTests(unittest.TestCase):
    def test_requires_consecutive_frames_before_confirming(self):
        d = gesture.GestureDebouncer(3)
        seq = ["none", "fist", "fist", "fist", "fist", "peace", "fist", "fist", "fist"]
        outs = [d.update(x) for x in seq]
        self.assertEqual(
            outs,
            [None, None, None, "fist", "fist", "fist", "fist", "fist", "fist"],
        )


class ResolveGestureTests(unittest.TestCase):
    class Category:
        def __init__(self, name, score):
            self.category_name = name
            self.score = score

    def test_high_confidence_recognizer_result_wins(self):
        label, text = gesture.resolve_gesture([self.Category("Thumb_Up", 0.9)], None)
        self.assertEqual(label, "thumbs_up")

    def test_low_confidence_falls_back_to_rules(self):
        fist = make_hand()
        label, text = gesture.resolve_gesture([self.Category("Thumb_Up", 0.1)], fist)
        self.assertEqual(label, "fist")
        self.assertEqual(text, "Fist")


class PickDrivingHandTests(unittest.TestCase):
    class FakeResult:
        def __init__(self, hand_landmarks):
            self.hand_landmarks = hand_landmarks

    def test_larger_hand_wins(self):
        small = [LM(0.5, 0.5)] + [LM(0, 0)] * 8 + [LM(0.52, 0.52)] + [LM(0, 0)] * 10
        big = [LM(0.5, 0.5)] + [LM(0, 0)] * 8 + [LM(0.9, 0.9)] + [LM(0, 0)] * 10
        result = self.FakeResult([small, big])
        self.assertEqual(gesture.pick_driving_hand(result), 1)

    def test_no_hands_returns_none(self):
        result = self.FakeResult([])
        self.assertIsNone(gesture.pick_driving_hand(result))


if __name__ == "__main__":
    unittest.main()
