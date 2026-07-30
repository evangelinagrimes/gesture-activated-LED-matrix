"""
Unit tests for sprites.py.

No camera, ESP32, or real sprites/ folder required -- these write small
PNG/hex files to a scratch directory (monkeypatching sprites.SPRITE_DIR so
the repo's real sprites/ is never touched) and exercise loading, hex
parsing, and frame encoding directly. Run with:

    python -m unittest test_sprites.py
"""

import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np

import sprites


class SpriteDirTestCase(unittest.TestCase):
    """Points sprites.SPRITE_DIR at a scratch directory for the duration of each test."""

    def setUp(self):
        self._orig_dir = sprites.SPRITE_DIR
        self._tmp = tempfile.mkdtemp(prefix="sprites_test_")
        sprites.SPRITE_DIR = self._tmp

    def tearDown(self):
        sprites.SPRITE_DIR = self._orig_dir
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _write(self, name, content):
        path = os.path.join(self._tmp, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    @staticmethod
    def _hex_body(token):
        """256 copies of `token`, formatted as 16 rows of 16 tokens."""
        tokens = [token] * 256
        return "\n".join(" ".join(tokens[r * 16:(r + 1) * 16]) for r in range(16))


class HexParsingTests(SpriteDirTestCase):
    def test_valid_hex_sprite_shape_and_values(self):
        arr = sprites.parse_hex_sprite(self._hex_body("FF0000"))
        self.assertEqual(arr.shape, (16, 16, 3))
        self.assertEqual(arr.dtype, np.uint8)
        self.assertTrue((arr == [255, 0, 0]).all())

    def test_shorthand_and_off_tokens(self):
        row = " ".join(["F00"] * 8 + ["."] * 8)
        text = "\n".join([row] * 16)
        arr = sprites.parse_hex_sprite(text)
        self.assertTrue((arr[0, 0] == [255, 0, 0]).all())
        self.assertTrue((arr[0, 8] == [0, 0, 0]).all())

    def test_comment_lines_skipped(self):
        text = "# a comment\n" + self._hex_body("00FF00") + "\n# trailing comment\n"
        arr = sprites.parse_hex_sprite(text)
        self.assertTrue((arr == [0, 255, 0]).all())

    def test_wrong_token_count_raises(self):
        text = " ".join(["FF0000"] * 255)  # one short
        with self.assertRaises(sprites.SpriteLoadError):
            sprites.parse_hex_sprite(text)

    def test_invalid_token_raises(self):
        text = " ".join(["FF0000"] * 255 + ["ZZZZZZ"])
        with self.assertRaises(sprites.SpriteLoadError):
            sprites.parse_hex_sprite(text)

    def test_load_sprite_from_hex_file(self):
        self._write("thumbs_up.hex", self._hex_body("0000FF"))
        arr = sprites.load_sprite("thumbs_up")
        self.assertIsNotNone(arr)
        self.assertEqual(arr.shape, (16, 16, 3))
        self.assertTrue((arr == [0, 0, 255]).all())


class ImageLoadingTests(SpriteDirTestCase):
    def test_exact_size_image_channel_order_converted_to_rgb(self):
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        img[:, :] = (0, 128, 255)  # BGR, as OpenCV writes it
        cv2.imwrite(os.path.join(self._tmp, "fist.png"), img)
        arr = sprites.load_sprite("fist")
        self.assertEqual(arr.shape, (16, 16, 3))
        self.assertEqual(arr.dtype, np.uint8)
        self.assertTrue((arr[0, 0] == [255, 128, 0]).all())  # BGR->RGB swap

    def test_larger_image_is_resized_down(self):
        img = np.full((64, 64, 3), (0, 255, 0), dtype=np.uint8)  # BGR green
        cv2.imwrite(os.path.join(self._tmp, "peace.png"), img)
        arr = sprites.load_sprite("peace")
        self.assertEqual(arr.shape, (16, 16, 3))
        self.assertTrue((arr == [0, 255, 0]).all())

    def test_transparent_pixels_composite_to_black(self):
        img = np.zeros((16, 16, 4), dtype=np.uint8)
        img[:, :, 0:3] = (0, 0, 255)  # BGR red, but...
        img[:, :, 3] = 0             # ...fully transparent
        cv2.imwrite(os.path.join(self._tmp, "open_palm.png"), img)
        arr = sprites.load_sprite("open_palm")
        self.assertTrue((arr == [0, 0, 0]).all())

    def test_missing_sprite_returns_none(self):
        self.assertIsNone(sprites.load_sprite("thumbs_down"))

    def test_image_takes_precedence_over_hex(self):
        self._write("fist.hex", self._hex_body("FFFFFF"))
        img = np.zeros((16, 16, 3), dtype=np.uint8)  # black
        cv2.imwrite(os.path.join(self._tmp, "fist.png"), img)
        arr = sprites.load_sprite("fist")
        self.assertTrue((arr == [0, 0, 0]).all())  # PNG (black) wins over hex (white)


class LoadAllSpritesTests(SpriteDirTestCase):
    def test_statuses_for_missing_loaded_and_broken(self):
        self._write("fist.hex", self._hex_body("112233"))
        self._write("peace.hex", "only a few tokens")  # malformed

        sprite_map, statuses = sprites.load_all_sprites(("fist", "peace", "thumbs_up"))

        self.assertIn("fist", sprite_map)
        self.assertTrue(statuses["fist"].startswith("loaded"))

        self.assertNotIn("peace", sprite_map)
        self.assertTrue(statuses["peace"].startswith("error"))

        self.assertNotIn("thumbs_up", sprite_map)
        self.assertEqual(statuses["thumbs_up"], "missing")


class EncodeFrameTests(unittest.TestCase):
    def test_byte_layout(self):
        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        rgb[0, 0] = (1, 2, 3)
        rgb[15, 15] = (250, 251, 252)

        payload = sprites.encode_frame("thumbs_up", rgb)

        self.assertTrue(payload.startswith(b"IMG1thumbs_up\n"))
        header_len = len(b"IMG1thumbs_up\n")
        pixel_bytes = payload[header_len:]
        self.assertEqual(len(pixel_bytes), 16 * 16 * 3)
        self.assertEqual(pixel_bytes[0:3], bytes([1, 2, 3]))
        self.assertEqual(pixel_bytes[-3:], bytes([250, 251, 252]))

    def test_wrong_shape_raises(self):
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            sprites.encode_frame("fist", rgb)


if __name__ == "__main__":
    unittest.main()
