# sprites/

One file per gesture label, named `<label>.<ext>`. The labels that matter
are the wire label set from `gesture.py`: `thumbs_up`, `thumbs_down`,
`peace`, `open_palm`, `fist`, `ok_sign`, `middle_finger`, and `none`.

A label with no file here just falls back to the original flat-color
behavior (`processGesture()` in `gesture-esp.ino`), so it's fine to add
these one at a time.

## Formats

Checked in this order per label -- first match wins:

1. **Images** -- `.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`. Any size; resized
   to 16x16 (nearest-neighbor when upscaling, so pixel art stays crisp;
   area-averaging when downscaling a larger source photo/image). A
   transparent background is composited onto black, i.e. transparent reads
   as "off" pixels on the panel.

2. **Hex text** -- `.hex` or `.txt`. Exactly 256 whitespace-separated
   color tokens, read row-major starting top-left. Each token is one of:
   - `RRGGBB` -- a 6-digit hex color, e.g. `FF8800`
   - `RGB` -- 3-digit shorthand, e.g. `F80` == `FF8800`
   - `.` or `-` -- pixel off (black)

   Lines whose first non-whitespace character is `#` are comments and are
   skipped entirely. There's no inline-comment syntax -- a color token never
   starts with `#`, so `#` at the start of a line is unambiguous.

   Example (first two rows of a 16x16 file):
   ```
   # my_gesture.hex
   . . . . . . FF0000 FF0000 . . . . . . . .
   . . . . . FF0000 FF0000 FF0000 FF0000 . . . . . . .
   ...
   ```

Dropping `thumbs_up.png` in next to the existing `thumbs_up.hex` makes the
PNG take effect immediately (no code change, no reflash) -- press `r` in the
OpenCV window to hot-reload without restarting `gesture.py`, or just restart
it.

## Checking your artwork

```
python -m sprites
```

prints a load-status table and an ANSI terminal preview of every sprite --
no ESP32 or webcam required. Re-run it after adding or editing a file.

## What's here now

The seven `.hex` files in this folder are placeholder pixel art (not real
photos) -- a thumb-and-fist shape for thumbs_up/thumbs_down, a V-sign for
peace, a splayed hand for open_palm, a rounded blob for fist, a ring for
ok_sign, a raised finger over a fist for middle_finger -- so the whole
pipeline (load, encode, transport, render) is testable with something
recognizable before real artwork exists. They reuse the same colors as the
original flat-color fallback (green/red/blue/white/orange/yellow/purple).
Replace them freely -- see the formats above.
