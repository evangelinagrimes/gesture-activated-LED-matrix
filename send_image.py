"""
Send Image to LED Matrix
=========================
Pushes a static image, read from a hex-literal byte dump, to an ESP32-driven
NeoPixel LED matrix over WiFi/UDP. This project doesn't run continuously or
need a persistent connection -- it's just "load an image, send it, done."

Run:
    python send_image.py [path/to/image.h]

If no path is given, IMAGE_PATH below is used.

--------------------------------------------------------------------------
Image format
--------------------------------------------------------------------------
load_image_bytes() accepts either:

- image2cpp-style: `..._width`/`..._height` declarations plus a packed
  pixel array at 1, 4, or 8 bits per pixel (auto-detected from the byte
  count). The source image does not need to already be MATRIX_WIDTH x
  MATRIX_HEIGHT -- it's downsampled (area-averaged) or upsampled
  (nearest-neighbor) automatically. See images/README.md for details.
- A flat NUMPIXELS-byte grayscale dump, if no width/height metadata is
  found in the file.

Either way, this just scans for `0xNN` tokens, so it doesn't care whether
the surrounding syntax is a `.h` C header or a plain comma-separated `.txt`.

--------------------------------------------------------------------------
ESP32 WiFi protocol (UDP)
--------------------------------------------------------------------------
The image is sent to ESP32_HOST:ESP32_PORT as a single raw UDP datagram of
exactly NUMPIXELS bytes -- no text framing needed, since the ESP32
(loop() in led_matrix_esp.ino) tells this apart from anything else purely
by that exact size. The ESP32 sends a short ack back to REPLY_PORT once
it's displayed the frame; this script waits briefly for it purely as a
courtesy, since the two devices don't need to stay continuously connected
-- not hearing one back isn't treated as fatal, just a hint to check the
ESP32 if the matrix didn't update.
--------------------------------------------------------------------------
"""

import os
import re
import sys
import socket

import cv2
import numpy as np

# WiFi UDP connection to the ESP32. Fill in your ESP32's IP (a static IP or
# DHCP reservation on your router is strongly recommended -- if it moves,
# the image silently goes nowhere).
ESP32_HOST = "192.168.8.182"   # <-- set to your ESP32's IP
ESP32_PORT = 4210              # port the ESP32 listens on for image frames
REPLY_PORT = 4211              # port this script listens on for the ESP32's ack
ACK_TIMEOUT_S = 2.0            # how long to wait for the ack before giving up (non-fatal)

# Physical dimensions of the LED matrix. Their product must match NUMPIXELS
# in led_matrix_esp.ino -- that's also the exact byte count the image
# datagram must be, since that's how the ESP32 tells it apart from
# anything else.
MATRIX_WIDTH = 16
MATRIX_HEIGHT = 16
NUMPIXELS = MATRIX_WIDTH * MATRIX_HEIGHT

# Default image to send if no path is given on the command line.
IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "pic1.txt")

# Matches a hex byte literal like 0xFF or 0x0a. Deliberately format-agnostic
# about what's around it, so it works equally on a full C header (variable
# declaration, PROGMEM attribute, braces, comments) or a bare comma/newline
# separated hex dump -- it just pulls out every `0xNN` token in the file.
HEX_TOKEN_RE = re.compile(r"0[xX][0-9A-Fa-f]{2}")

# Matches image2cpp-style `..._width = N;` / `..._height = N;` declarations
# (any prefix, e.g. pic1_width, image_height) so the source resolution can
# be recovered regardless of what the generator named its variables.
WIDTH_RE = re.compile(r"\w*width\w*\s*=\s*(\d+)", re.IGNORECASE)
HEIGHT_RE = re.compile(r"\w*height\w*\s*=\s*(\d+)", re.IGNORECASE)


def _parse_dimensions(text: str):
    """Return (width, height) from image2cpp-style declarations, or None if
    the file has no such metadata (i.e. it's assumed to already be a flat
    NUMPIXELS-byte dump matching the matrix 1:1)."""
    w_match = WIDTH_RE.search(text)
    h_match = HEIGHT_RE.search(text)
    if not w_match or not h_match:
        return None
    return int(w_match.group(1)), int(h_match.group(1))


def _detect_bit_depth(token_count: int, width: int, height: int) -> int:
    """Infer pixels-per-byte packing from the byte count alone.

    image2cpp (and similar tools) pack multiple low-bit-depth pixels per
    byte, row-aligned (each row starts on a byte boundary, so a row whose
    width isn't a multiple of the packing factor gets padded). Rather than
    trust a possibly-wrong size comment in the source (a `/8` size
    declaration is arithmetically incorrect for non-multiple-of-8 widths --
    integer truncation), just check which depth's *actual* row-padded size
    matches the real token count.
    """
    candidates = {
        8: width * height,                # 1 byte/pixel, direct grayscale
        4: ((width + 1) // 2) * height,    # 2 px/byte (4-bit grayscale)
        1: ((width + 7) // 8) * height,    # 8 px/byte (1-bit monochrome)
    }
    for depth, expected in candidates.items():
        if token_count == expected:
            return depth
    raise ValueError(
        f"{token_count} hex byte(s) doesn't match any known packing for a "
        f"{width}x{height} image (tried 8/4/1 bits per pixel: "
        f"{candidates[8]}/{candidates[4]}/{candidates[1]} byte(s) expected)"
    )


def _unpack_bitmap(raw: bytes, width: int, height: int, bit_depth: int) -> np.ndarray:
    """Expand a row-padded, MSB-first packed bitmap into an 8-bit HxW
    grayscale array (0-255)."""
    pixels_per_byte = 8 // bit_depth
    row_bytes = -(-width // pixels_per_byte)  # ceil division
    max_val = (1 << bit_depth) - 1
    scale = 255 // max_val  # e.g. 1-bit -> 255, 4-bit -> 17, 8-bit -> 1

    img = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        row_start = y * row_bytes
        for x in range(width):
            byte_val = raw[row_start + x // pixels_per_byte]
            shift = bit_depth * (pixels_per_byte - 1 - (x % pixels_per_byte))
            img[y, x] = ((byte_val >> shift) & max_val) * scale
    return img


def load_image_bytes(path: str) -> bytes:
    """Load a hex-literal image file and return exactly NUMPIXELS grayscale
    bytes (one per LED, row-major), ready to hand to send_image().

    Supports two shapes of input:

    - image2cpp-style: `..._width`/`..._height` declarations plus a packed
      pixel array at 1, 4, or 8 bits per pixel (auto-detected from the byte
      count -- see _detect_bit_depth()), row-aligned. The source image is
      downsampled (area-averaged) or upsampled (nearest-neighbor) to
      MATRIX_WIDTH x MATRIX_HEIGHT as needed -- it does not need to already
      match the matrix's resolution.
    - No width/height metadata found: assumed to already be a flat
      NUMPIXELS-byte grayscale dump matching the matrix 1:1.

    Either way, this just scans for `0xNN` tokens, so it doesn't care
    whether the surrounding syntax is a C header or a bare hex dump.
    """
    with open(path, "r") as f:
        text = f.read()
    raw = bytes(int(tok, 16) for tok in HEX_TOKEN_RE.findall(text))

    dims = _parse_dimensions(text)
    if dims is None:
        if len(raw) != NUMPIXELS:
            raise ValueError(
                f"found {len(raw)} hex byte(s) and no width/height metadata, "
                f"expected exactly {NUMPIXELS} (one grayscale byte per LED)"
            )
        return raw

    width, height = dims
    bit_depth = _detect_bit_depth(len(raw), width, height)
    img = _unpack_bitmap(raw, width, height, bit_depth)

    interpolation = cv2.INTER_AREA if (width, height) > (MATRIX_WIDTH, MATRIX_HEIGHT) else cv2.INTER_NEAREST
    resized = cv2.resize(img, (MATRIX_WIDTH, MATRIX_HEIGHT), interpolation=interpolation)
    return resized.astype(np.uint8).tobytes()


def send_image(path: str = IMAGE_PATH):
    """Load path and push it to the ESP32 as a single raw UDP datagram of
    NUMPIXELS bytes. Waits briefly for the ESP32's ack as a courtesy --
    not receiving one isn't fatal, since the link doesn't need to stay up.
    """
    image_bytes = load_image_bytes(path)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("", REPLY_PORT))
        sock.settimeout(ACK_TIMEOUT_S)
        sock.sendto(image_bytes, (ESP32_HOST, ESP32_PORT))
        print(f"Sent image frame ({len(image_bytes)} px) from {path} to ESP32 at {ESP32_HOST}:{ESP32_PORT}")
        try:
            data, _addr = sock.recvfrom(256)
            print(f"[ESP32] {data.decode('utf-8', errors='ignore').strip()}")
        except socket.timeout:
            print(f"No response from ESP32 within {ACK_TIMEOUT_S:.0f}s -- "
                  f"check its power/WiFi if the matrix didn't update.")
    except OSError as e:
        print(f"Warning: failed to send image over UDP: {e}")
    finally:
        sock.close()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH
    try:
        send_image(path)
    except (OSError, ValueError) as e:
        print(f"Error: could not load image from {path}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
