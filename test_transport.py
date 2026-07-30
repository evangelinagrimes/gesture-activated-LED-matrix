"""
Unit tests for transport.py.

No ESP32 or real serial port required. SerialTransport is tested against a
fake pyserial-shaped object (write()/in_waiting/read()) swapped in for
_ser, so these never touch a COM port. UdpTransport is tested over a real
loopback UDP socket, which needs no hardware either. Run with:

    python -m unittest test_transport.py
"""

import socket
import time
import unittest
from unittest import mock

import transport


class FakeSerial:
    """Minimal stand-in for pyserial's Serial: enough of write()/in_waiting/
    read() for SerialTransport to drive, plus feed() for tests to inject
    incoming bytes."""

    def __init__(self):
        self.is_open = True
        self.written = bytearray()
        self._rx = bytearray()

    def write(self, data):
        self.written.extend(data)
        return len(data)

    def close(self):
        self.is_open = False

    @property
    def in_waiting(self):
        return len(self._rx)

    def read(self, n):
        chunk = bytes(self._rx[:n])
        del self._rx[:n]
        return chunk

    def feed(self, data: bytes):
        self._rx.extend(data)


def _make_serial_transport():
    """A SerialTransport with a FakeSerial already 'open', so send()/read_lines()
    never attempt a real port open."""
    st = transport.SerialTransport("COM_TEST", 115200, reopen_interval_s=2.0)
    st._ser = FakeSerial()
    return st, st._ser


def _decode_serial_frames(data: bytes):
    """Pure-Python mirror of the ESP32's pollSerial() state machine
    (gesture-esp.ino): sync word -> uint16 LE length -> payload -> XOR
    checksum. Used only to independently verify SerialTransport's framing
    round-trips correctly -- including payloads that contain 0x0A and the
    sync bytes themselves, which is the entire reason this framing exists
    instead of newline-termination.

    Returns a list of (payload: bytes, checksum_ok: bool).
    """
    SYNC1, SYNC2 = 0xA5, 0x5A
    frames = []
    i = 0
    n = len(data)
    while i < n:
        if data[i] != SYNC1:
            i += 1
            continue
        if i + 1 >= n or data[i + 1] != SYNC2:
            i += 1
            continue
        if i + 4 > n:
            break
        length = data[i + 2] | (data[i + 3] << 8)
        start = i + 4
        end = start + length
        if end + 1 > n:
            break
        payload = data[start:end]
        checksum = data[end]
        expected = 0
        for b in payload:
            expected ^= b
        frames.append((bytes(payload), checksum == expected))
        i = end + 1
    return frames


class SerialFramingTests(unittest.TestCase):
    def test_sync_word_and_length_prefix(self):
        st, fake = _make_serial_transport()
        payload = b"thumbs_up\n"

        self.assertTrue(st.send(payload))

        frame = bytes(fake.written)
        self.assertEqual(frame[0:2], bytes([0xA5, 0x5A]))
        length = frame[2] | (frame[3] << 8)
        self.assertEqual(length, len(payload))
        self.assertEqual(frame[4:4 + length], payload)

    def test_checksum_is_xor_of_payload(self):
        st, fake = _make_serial_transport()
        payload = bytes([1, 2, 3, 4, 5])

        st.send(payload)

        frame = bytes(fake.written)
        checksum = frame[-1]
        expected = 0
        for b in payload:
            expected ^= b
        self.assertEqual(checksum, expected)

    def test_round_trip_payload_containing_newline_and_sync_bytes(self):
        st, fake = _make_serial_transport()
        # Deliberately includes 0x0A (would corrupt a newline-terminated
        # protocol) and 0xA5/0x5A (the sync word itself), plus a realistic
        # IMG1 sprite frame payload.
        payload = (bytes([0xA5, 0x5A, 0x0A, 0x00, 0xFF, 0x0A, 0xA5])
                   + b"IMG1none\n" + bytes(range(256)))

        st.send(payload)

        decoded = _decode_serial_frames(bytes(fake.written))
        self.assertEqual(len(decoded), 1)
        decoded_payload, checksum_ok = decoded[0]
        self.assertTrue(checksum_ok)
        self.assertEqual(decoded_payload, payload)

    def test_oversized_payload_rejected(self):
        st, _fake = _make_serial_transport()
        with self.assertRaises(ValueError):
            st.send(bytes(transport.SerialTransport.MAX_PAYLOAD + 1))

    def test_read_lines_splits_on_newline(self):
        st, fake = _make_serial_transport()
        fake.feed(b"Gesture: fist\n[HEARTBEAT] Uptime: 5s | Gestures: 1 | last 2s ago\n")

        lines = st.read_lines()

        self.assertEqual(lines, ["Gesture: fist", "[HEARTBEAT] Uptime: 5s | Gestures: 1 | last 2s ago"])

    def test_read_lines_handles_partial_line_across_calls(self):
        st, fake = _make_serial_transport()
        fake.feed(b"partial ")
        self.assertEqual(st.read_lines(), [])

        fake.feed(b"line\n")
        self.assertEqual(st.read_lines(), ["partial line"])

    def test_status_reports_open_and_counters(self):
        st, _fake = _make_serial_transport()
        st.send(b"fist\n")

        status = st.status()

        self.assertEqual(status["transport"], "serial")
        self.assertTrue(status["open"])
        self.assertEqual(status["frames_sent"], 1)
        self.assertGreater(status["bytes_sent"], 0)


class UdpTransportLoopbackTests(unittest.TestCase):
    """No ESP32 needed -- a plain loopback UDP socket plays its part."""

    def test_send_and_receive_round_trip(self):
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_sock.bind(("127.0.0.1", 0))
        server_sock.settimeout(2.0)
        server_port = server_sock.getsockname()[1]

        # Avoid spawning a real ping subprocess in this test.
        with mock.patch.object(transport, "_ping_once", return_value=True):
            t = transport.UdpTransport("127.0.0.1", server_port, 0,
                                        ping_interval_s=9999, ping_timeout_ms=100)
            self.assertTrue(t.open())
            try:
                t.send(b"thumbs_up\n")
                data, _addr = server_sock.recvfrom(1024)
                self.assertEqual(data, b"thumbs_up\n")

                local_port = t._sock.getsockname()[1]
                server_sock.sendto(b"Gesture: thumbs_up\n", ("127.0.0.1", local_port))

                deadline = time.monotonic() + 2.0
                lines = []
                while time.monotonic() < deadline and not lines:
                    lines = t.read_lines()
                self.assertEqual(lines, ["Gesture: thumbs_up"])
            finally:
                t.close()
                server_sock.close()

    def test_reconnect_report_regex_matches_firmware_message(self):
        # Exact wording from maintainWiFi() in gesture-esp.ino.
        line = "WiFi reconnected after 3 attempt(s), down for 12s. Last disconnect reason 8 ASSOC_LEAVE"
        match = transport.RECONNECT_REPORT_RE.match(line)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "3")


if __name__ == "__main__":
    unittest.main()
