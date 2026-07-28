# Images

Drop your hex-literal image file(s) here and point `IMAGE_PATH` (top of
`led_matrix_controller.py`) at the one you want the `i` key to push to the
matrix.

Accepted format: either one works, since `load_image_bytes()` just scans
the file for `0xNN` tokens and ignores everything else around them.

- A C header (image2cpp-style), e.g.:

  ```c
  const uint8_t example[] PROGMEM = {
      0xFF, 0xFF, 0xFF, 0xFF, /* ... */
  };
  ```

- A plain comma-separated `.txt` dump of the same bytes.

The file must contain **exactly 256** hex bytes — one grayscale
brightness value per LED on the 16x16 matrix, row-major (left-to-right,
top-to-bottom). `send_image()` will refuse to send anything that doesn't
match that count, so a partial or wrongly-sized file fails loudly instead
of drawing garbage.
