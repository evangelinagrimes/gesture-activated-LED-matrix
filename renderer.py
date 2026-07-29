"""
Rendering Pipeline for LED Matrix
==================================
Turns a LoadedImage (see image_loader.py) plus a RenderSettings into the
exact RGB pixel array the ESP32 will display, the 768-byte wire payload for
that array, and a scaled-up preview image for on-screen display.

This is deliberately the single place where "what does the panel show"
gets decided: render_rgb() is the one function that feeds both
to_payload() (what actually gets sent) and to_preview_image() (what the
GUI shows on screen), so the preview cannot drift from the panel -- there
is no second code path that could compute something different.

Rendering modes:

- Two-tone (the default): each pixel is thresholded against
  RenderSettings.threshold and rendered as one of two solid colors
  (color_on/color_off). Good for markers, logos, and text, where
  intermediate gray from resizing/anti-aliasing just looks murky and true
  black is otherwise indistinguishable from the matrix being off.
  With `smooth=True`, pixels are ramped between color_off and color_on
  instead of hard-thresholded -- this degrades to plain grayscale when
  color_off=black and color_on=white, so smooth two-tone is a strict
  superset of the old firmware's non-binarized behavior.
- Full color: uses the source image's actual RGB values directly (only
  available when the source is a raster image -- see LoadedImage.rgb).

Brightness is applied last, as a uniform scale on the final RGB output --
not to the grayscale input -- so it can never interact with the threshold
decision (scaling the input first could push an already-dim pixel below
the threshold and flip it off, which would be a surprising side effect of
"brightness").
--------------------------------------------------------------------------
"""

from dataclasses import dataclass, replace

import numpy as np
from PIL import Image

from image_loader import LoadedImage, MATRIX_WIDTH, MATRIX_HEIGHT

TWO_TONE = "two_tone"
FULL_COLOR = "full_color"


@dataclass(frozen=True)
class RenderSettings:
    mode: str = TWO_TONE
    threshold: int = 128
    color_on: tuple = (255, 255, 255)
    color_off: tuple = (0, 0, 0)
    smooth: bool = False       # ramp between color_off/color_on instead of hard threshold
    brightness: int = 100      # percent, 0-100

    def replace(self, **changes) -> "RenderSettings":
        return replace(self, **changes)


DEFAULTS = RenderSettings()


def render_rgb(img: LoadedImage, settings: RenderSettings) -> np.ndarray:
    """Return the (MATRIX_HEIGHT, MATRIX_WIDTH, 3) uint8 array the panel
    will display for this image under these settings."""
    if settings.mode == FULL_COLOR:
        if img.rgb is None:
            raise ValueError(
                "full_color mode requires a source image with real color data "
                "(a raster image, not a hex dump) -- use two_tone mode instead"
            )
        rgb = img.rgb.astype(np.float32)
    else:
        on = np.array(settings.color_on, dtype=np.float32)
        off = np.array(settings.color_off, dtype=np.float32)
        gray = img.gray.astype(np.float32)
        if settings.smooth:
            t = (gray / 255.0)[..., None]
            rgb = off + (on - off) * t
        else:
            mask = (gray >= settings.threshold)[..., None]
            rgb = np.where(mask, on, off)

    rgb = rgb * (settings.brightness / 100.0)
    return np.clip(rgb + 0.5, 0, 255).astype(np.uint8)


def to_payload(rgb: np.ndarray) -> bytes:
    """Flatten an (H, W, 3) RGB array to the 768-byte row-major R,G,B wire
    payload. Explicit ascontiguousarray so a future transpose/slice upstream
    can't silently reorder the bytes on the wire."""
    return np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()


def to_preview_image(rgb: np.ndarray, scale: int = 24) -> Image.Image:
    """Scale an (H, W, 3) RGB array up for on-screen display. NEAREST is
    mandatory here -- any smoothing would make the preview lie about where
    pixel boundaries actually fall on the physical matrix."""
    img = Image.fromarray(rgb, "RGB")
    return img.resize((MATRIX_WIDTH * scale, MATRIX_HEIGHT * scale), Image.NEAREST)
