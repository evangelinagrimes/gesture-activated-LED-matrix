"""
Sprite loading for the LED matrix.
=====================================================
Loads a 16x16 RGB image per gesture label from sprites/ next to this script,
so the matrix can show a picture (a thumbs-up shape for "thumbs_up", etc.)
instead of a flat color. This is pure/hardware-free -- no camera, no ESP32 --
so it's covered by test_sprites.py and can be exercised standalone with:

    python -m sprites

which prints a load-status table and an ANSI terminal preview of every
sprite, the fastest way to check new artwork before plugging anything in.

File formats, checked in this order per label (first match wins, so a PNG
overrides a placeholder .hex with no code change):

    <label>.png / .jpg / .jpeg / .bmp / .gif   -- any size; resized to 16x16
    <label>.hex / <label>.txt                  -- 256 whitespace-separated
                                                   hex color tokens, see
                                                   parse_hex_sprite()

A label with no sprite file returns None from load_sprite() -- gesture.py's
caller falls back to sending the plain gesture label instead of a frame, so
the ESP32's original flat-color behavior (see FALLBACK_COLORS below) still
runs for anything without artwork yet.
"""

import os

import cv2
import numpy as np

MATRIX_WIDTH = 16
MATRIX_HEIGHT = 16

SPRITE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sprites")

# The wire label set from gesture.py's GESTURE_LABELS/MP_GESTURE_LABELS,
# plus "none". A missing "none" sprite is fine -- it falls back to the
# existing off/black behavior, which is what a driving hand with no
# recognized gesture already does today.
DEFAULT_LABELS = ("thumbs_up", "thumbs_down", "peace", "open_palm", "fist", "none")

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
HEX_EXTS = (".hex", ".txt")

# 4-byte magic prefix for a sprite-frame datagram/payload. Doesn't collide
# with any existing gesture label, so the firmware can dispatch on it and
# the plain-label wire format stays byte-compatible. Must match FRAME_MAGIC
# in gesture-esp.ino's handleCommand().
FRAME_MAGIC = b"IMG1"

# Mirrors processGesture()'s setColor() calls in gesture-esp.ino -- what the
# panel shows today for a label with no sprite. Used only so the dashboard
# preview can show what's actually on the panel in that case.
FALLBACK_COLORS = {
    "thumbs_up": (0, 255, 0),
    "thumbs_down": (255, 0, 0),
    "peace": (0, 0, 255),
    "open_palm": (255, 255, 255),
    "fist": (255, 165, 0),
    "none": (0, 0, 0),
}


class SpriteLoadError(Exception):
    """A sprite file exists but couldn't be decoded/parsed."""


def _find_sprite_file(label: str):
    for ext in IMAGE_EXTS + HEX_EXTS:
        path = os.path.join(SPRITE_DIR, label + ext)
        if os.path.isfile(path):
            return path
    return None


def _load_image_sprite(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SpriteLoadError(f"could not decode image: {path}")

    if img.ndim == 2:  # grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.shape[2] == 4:
        # Composite onto black using the alpha channel -- a transparent
        # background should read as "off" pixels on the panel, not garbage.
        b, g, r, a = cv2.split(img)
        alpha = a.astype(np.float32) / 255.0
        b = (b.astype(np.float32) * alpha).astype(np.uint8)
        g = (g.astype(np.float32) * alpha).astype(np.uint8)
        r = (r.astype(np.float32) * alpha).astype(np.uint8)
        img = cv2.merge([b, g, r])

    h, w = img.shape[:2]
    if (w, h) != (MATRIX_WIDTH, MATRIX_HEIGHT):
        # Downscaling uses area averaging (fewer artifacts); upscaling uses
        # nearest-neighbor so hand-authored pixel art stays crisp instead of
        # getting blurred.
        interp = cv2.INTER_AREA if (w > MATRIX_WIDTH or h > MATRIX_HEIGHT) else cv2.INTER_NEAREST
        img = cv2.resize(img, (MATRIX_WIDTH, MATRIX_HEIGHT), interpolation=interp)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.uint8)


def _parse_hex_color(token: str):
    """Parse one color token: RRGGBB, RGB (shorthand), or '.'/'-' for off."""
    t = token.strip()
    if t in (".", "-"):
        return (0, 0, 0)
    if len(t) == 3:
        t = "".join(c * 2 for c in t)
    if len(t) != 6:
        raise SpriteLoadError(f"invalid color token {token!r} (expected RRGGBB, RGB, or '.'/'-' for off)")
    try:
        r = int(t[0:2], 16)
        g = int(t[2:4], 16)
        b = int(t[4:6], 16)
    except ValueError:
        raise SpriteLoadError(f"invalid hex color token {token!r}")
    return (r, g, b)


def parse_hex_sprite(text: str) -> np.ndarray:
    """Parse a 16x16 sprite from whitespace-separated hex color tokens.

    Whole-line comments start with '#' (no inline comments -- that would be
    ambiguous with a token like "#fff" and this file format has no leading
    '#' on color tokens). Requires exactly MATRIX_WIDTH*MATRIX_HEIGHT tokens,
    read row-major from the top-left, so a malformed file fails with a clear
    count mismatch instead of silently drawing garbage.
    """
    tokens = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens.extend(stripped.split())

    expected = MATRIX_WIDTH * MATRIX_HEIGHT
    if len(tokens) != expected:
        raise SpriteLoadError(f"expected {expected} color tokens, found {len(tokens)}")

    pixels = [_parse_hex_color(t) for t in tokens]
    return np.array(pixels, dtype=np.uint8).reshape(MATRIX_HEIGHT, MATRIX_WIDTH, 3)


def _load_hex_sprite(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return parse_hex_sprite(text)


def load_sprite(label: str):
    """Return a (16,16,3) uint8 RGB array for `label`, or None if no sprite file exists.

    Raises SpriteLoadError if a file exists but can't be decoded/parsed, so a
    malformed sprite fails loudly rather than silently falling back to the
    flat-color behavior.
    """
    path = _find_sprite_file(label)
    if path is None:
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext in HEX_EXTS:
        return _load_hex_sprite(path)
    return _load_image_sprite(path)


def load_all_sprites(labels=DEFAULT_LABELS):
    """Load every sprite in `labels`.

    Returns (sprites, statuses): sprites maps label -> (16,16,3) uint8 array
    for everything that loaded; statuses maps every label in `labels` to a
    human-readable string ("loaded (thumbs_up.png)" / "missing" / "error: ...")
    for the dashboard's sprite table.
    """
    sprites = {}
    statuses = {}
    for label in labels:
        path = _find_sprite_file(label)
        if path is None:
            statuses[label] = "missing"
            continue
        try:
            arr = load_sprite(label)
            sprites[label] = arr
            statuses[label] = f"loaded ({os.path.basename(path)})"
        except SpriteLoadError as e:
            statuses[label] = f"error: {e}"
        except Exception as e:  # defensive: a bad file must never crash the detector
            statuses[label] = f"error: {e}"
    return sprites, statuses


def encode_frame(label: str, rgb: np.ndarray) -> bytes:
    """Encode a sprite frame: FRAME_MAGIC + label + newline + 768 raw RGB bytes.

    Shared by both transports -- UdpTransport sends this as one datagram
    as-is; SerialTransport wraps it in its own sync/length/checksum framing
    before writing it to the port.
    """
    if rgb.shape != (MATRIX_HEIGHT, MATRIX_WIDTH, 3):
        raise ValueError(f"expected shape ({MATRIX_HEIGHT}, {MATRIX_WIDTH}, 3), got {rgb.shape}")
    return FRAME_MAGIC + label.encode("utf-8") + b"\n" + rgb.astype(np.uint8).tobytes()


def _ansi_preview(rgb: np.ndarray) -> str:
    """Render a (16,16,3) array as a terminal preview: two spaces of truecolor
    background per pixel, one text row per matrix row.

    Deliberately ASCII-only (no Unicode block characters) -- Windows' default
    console codepage (cp1252) can't encode those and would crash print()
    outright, whereas a plain ANSI escape + space renders fine (as a solid
    color swatch) in any terminal that supports 24-bit color, and degrades to
    harmless plain spaces in one that doesn't.
    """
    lines = []
    for y in range(MATRIX_HEIGHT):
        row = []
        for x in range(MATRIX_WIDTH):
            r, g, b = (int(v) for v in rgb[y, x])
            row.append(f"\x1b[48;2;{r};{g};{b}m  ")
        row.append("\x1b[0m")
        lines.append("".join(row))
    return "\n".join(lines)


def main():
    sprites, statuses = load_all_sprites()
    print(f"Sprite directory: {SPRITE_DIR}\n")
    for label in DEFAULT_LABELS:
        print(f"{label:12s} {statuses.get(label, 'missing')}")
        if label in sprites:
            print(_ansi_preview(sprites[label]))
        print()


if __name__ == "__main__":
    main()
