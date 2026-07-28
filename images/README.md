# Images

Drop your hex-literal image file(s) here, then either point `IMAGE_PATH`
(top of `send_image.py`) at the one you want to send by default, or pass
its path as a command-line argument: `python send_image.py images/foo.h`.

`load_image_bytes()` accepts either:

- **image2cpp-style, any resolution** — a `..._width`/`..._height`
  declaration plus a packed pixel array at 1, 4, or 8 bits per pixel
  (auto-detected from the byte count), e.g.:

  ```c
  const uint32_t pic1_width = 800;
  const uint32_t pic1_height = 480;
  const uint8_t pic1_data[...] = { 0xFF, 0xFF, /* ... */ };
  ```

  The source image does **not** need to already be 16x16 -- it's
  automatically downsampled (area-averaged) or upsampled
  (nearest-neighbor) to the matrix's resolution (`MATRIX_WIDTH` x
  `MATRIX_HEIGHT`, 16x16 by default). This is the format image2cpp
  (javl.github.io/image2cpp) and similar tools export.

- **A flat 256-byte dump** — if the file has no width/height metadata,
  it's assumed to already be exactly `NUMPIXELS` (256) grayscale bytes,
  one per LED, row-major.

Either way, the parser just scans the file for `0xNN` tokens, so it
doesn't care whether the surrounding syntax is a `.h` C header or a plain
comma-separated `.txt` dump.

A wrong/unrecognized byte count raises an error naming exactly what was
expected, rather than silently drawing a corrupted image.
