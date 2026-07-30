"""
Transport layer between gesture.py and the ESP32.
=====================================================
Two interchangeable ways to reach the ESP32, behind one small interface so
the frame loop in gesture.py never has to branch on which link is active:

    open() / close() / is_open
    send(payload: bytes) -> bool
    read_lines() -> list[str]      # non-blocking; decoded status/debug lines
    status() -> dict               # link facts for the debug dashboard

SerialTransport (USB, primary) wraps every outgoing payload in a small
sync/length/checksum frame -- see FRAME_SYNC below -- because a raw sprite
payload can contain any byte value, including 0x0A, and serial has no
built-in message boundary the way a UDP datagram does. UdpTransport is the
original WiFi link: datagrams already have boundaries, so no framing is
needed there, and it additionally owns the independent ICMP ping monitor
(ping is meaningless over a wired USB link).

Both transports carry the exact same payload bytes end to end -- either a
plain "<label>\\n" (see sprites.encode_frame's sibling, the fallback path in
gesture.py) or an "IMG1<label>\\n<768 bytes RGB>" sprite frame -- so the
firmware's handleCommand() dispatch logic doesn't need to know or care which
transport a message arrived over.
"""

import os
import re
import socket
import subprocess
import threading
import time

try:
    import serial  # pyserial
except ImportError:  # pragma: no cover - exercised only when pyserial is missing
    serial = None

# Parses the ESP32's "WiFi reconnected after N attempt(s), down for Xs"
# status line (see maintainWiFi() in gesture-esp.ino). WiFi-specific, so it
# lives with UdpTransport, but the ESP32 still prints it over Serial too
# when WiFi happens to be enabled -- SerialTransport just doesn't parse it.
RECONNECT_REPORT_RE = re.compile(r"^WiFi reconnected after (\d+) attempt")


class SerialTransport:
    """USB serial link to the ESP32 (default transport -- doesn't depend on WiFi).

    Framing on the write side:

        0xA5 0x5A <uint16 LE payload_len> <payload bytes> <xor checksum>

    Sync word, explicit length (so the firmware never has to guess where a
    frame ends), XOR checksum over the payload. The firmware falls back to
    treating input as a bare newline-terminated ASCII label whenever the
    first byte isn't the sync word, so typing a label into the Arduino
    Serial Monitor by hand still works.

    The read side needs none of that: the firmware's sendStatus() calls
    Serial.println() first (before it ever touches UDP), so status/debug
    lines arrive as plain newline-terminated UTF-8 with zero firmware
    changes required.
    """

    FRAME_SYNC = bytes([0xA5, 0x5A])
    MAX_PAYLOAD = 1024  # must match SERIAL_MAX_PAYLOAD in gesture-esp.ino

    def __init__(self, port: str, baud: int, reopen_interval_s: float):
        if serial is None:
            raise RuntimeError(
                "pyserial is required for TRANSPORT='serial'. "
                "Install it with: pip install pyserial (see requirements.txt)."
            )
        self.port = port
        self.baud = baud
        self.reopen_interval_s = reopen_interval_s

        self._ser = None
        self._rx_buf = bytearray()
        self._last_open_attempt = 0.0
        self.reopen_attempts = 0
        self.last_error = None
        self.frames_sent = 0
        self.bytes_sent = 0

    @property
    def is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def open(self) -> bool:
        self._last_open_attempt = time.monotonic()
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=0)
            self.last_error = None
            print(f"Opened serial port {self.port} @ {self.baud} baud")
            return True
        except serial.SerialException as e:
            self._ser = None
            self.last_error = str(e)
            self.reopen_attempts += 1
            return False

    def close(self):
        if self._ser is not None:
            try:
                self._ser.close()
            except serial.SerialException:
                pass
            self._ser = None
            print("Serial connection closed.")

    def _ensure_open(self) -> bool:
        """(Re)open the port if it isn't currently open, throttled to reopen_interval_s.

        Necessary because both a cable unplug and an ESP32-side reset (e.g.
        from opening the port, which toggles DTR on most boards) can drop
        the port out from under us mid-session.
        """
        if self.is_open:
            return True
        if time.monotonic() - self._last_open_attempt < self.reopen_interval_s:
            return False
        return self.open()

    def send(self, payload: bytes) -> bool:
        if not self._ensure_open():
            return False
        if len(payload) > self.MAX_PAYLOAD:
            raise ValueError(f"payload too large for serial framing: {len(payload)} > {self.MAX_PAYLOAD}")

        checksum = 0
        for b in payload:
            checksum ^= b
        frame = self.FRAME_SYNC + len(payload).to_bytes(2, "little") + payload + bytes([checksum])

        try:
            self._ser.write(frame)
            self.frames_sent += 1
            self.bytes_sent += len(frame)
            return True
        except serial.SerialException as e:
            self.last_error = str(e)
            print(f"Warning: failed to write to serial port: {e}")
            self.close()
            return False

    def read_lines(self) -> list:
        """Drain whatever the OS buffer has and return complete decoded lines.

        Non-blocking (opened with timeout=0): in_waiting/read never stall
        the frame loop even if the ESP32 has gone quiet.
        """
        if not self._ensure_open():
            return []
        try:
            n = self._ser.in_waiting
            if n:
                self._rx_buf.extend(self._ser.read(n))
        except (serial.SerialException, OSError) as e:
            self.last_error = str(e)
            print(f"Warning: failed to read from serial port: {e}")
            self.close()
            return []

        lines = []
        while True:
            idx = self._rx_buf.find(b"\n")
            if idx == -1:
                break
            raw = bytes(self._rx_buf[:idx])
            del self._rx_buf[:idx + 1]
            line = raw.decode("utf-8", errors="ignore").strip()
            if line:
                lines.append(line)
        return lines

    def status(self) -> dict:
        return {
            "transport": "serial",
            "address": f"{self.port} @ {self.baud} baud",
            "open": self.is_open,
            "reopen_attempts": self.reopen_attempts,
            "last_error": self.last_error,
            "frames_sent": self.frames_sent,
            "bytes_sent": self.bytes_sent,
            "ping_ok": None,  # not applicable over a wired link
        }


def _ping_once(host: str, timeout_ms: int) -> bool:
    if os.name == "nt":
        cmd = ["ping", "-n", "1", "-w", str(timeout_ms), host]
    else:
        cmd = ["ping", "-c", "1", "-W", str(max(1, timeout_ms // 1000)), host]
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=(timeout_ms / 1000) + 2,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


class UdpTransport:
    """WiFi UDP link to the ESP32 (fallback transport -- set TRANSPORT = "udp").

    One socket serves both directions: bound to local_port to receive status
    datagrams, and sends gesture/frame datagrams to host:port. Also runs an
    independent ICMP ping monitor on a background thread -- this is deliberately
    separate from the UDP-heartbeat-based liveness check in gesture.py, which
    only tells you the *application* has gone quiet and can't distinguish "ESP32
    fell off the network" from "ESP32 is alive but its UDP send is failing".
    """

    def __init__(self, host: str, port: int, local_port: int,
                 ping_interval_s: float, ping_timeout_ms: int, conn_logger=None):
        self.host = host
        self.port = port
        self.local_port = local_port
        self.ping_interval_s = ping_interval_s
        self.ping_timeout_ms = ping_timeout_ms
        self._conn_logger = conn_logger

        self._sock = None
        self.last_error = None
        self.frames_sent = 0
        self.bytes_sent = 0

        # None = no ping attempted yet.
        self.ping_ok = None
        self._ping_lock = threading.Lock()
        self._ping_stop = threading.Event()
        self._ping_thread = None

        # (attempts: int, message: str) from the ESP32's own reconnect report,
        # and time.monotonic() it arrived -- for the dashboard/status overlay.
        self.last_reconnect_report = None
        self.last_reconnect_report_time = None

    @property
    def is_open(self) -> bool:
        return self._sock is not None

    def open(self) -> bool:
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.bind(("", self.local_port))
            self._sock.setblocking(False)
            print(f"Listening for ESP32 on UDP port {self.local_port}, "
                  f"sending gestures to {self.host}:{self.port}")
            self._start_ping_thread()
            return True
        except OSError as e:
            print(f"Warning: could not open UDP socket on port {self.local_port}: {e}")
            print("Continuing without ESP32 connection.")
            self._sock = None
            self.last_error = str(e)
            return False

    def close(self):
        self._ping_stop.set()
        if self._sock is not None:
            self._sock.close()
            self._sock = None
            print("ESP32 connection closed.")

    def send(self, payload: bytes) -> bool:
        if self._sock is None:
            return False
        try:
            self._sock.sendto(payload, (self.host, self.port))
            self.frames_sent += 1
            self.bytes_sent += len(payload)
            return True
        except OSError as e:
            self.last_error = str(e)
            print(f"Warning: failed to send over UDP: {e}")
            return False

    def read_lines(self, max_messages: int = 20) -> list:
        """Drain up to max_messages buffered datagrams, so a chatty or
        malfunctioning ESP32 can't stall the frame loop indefinitely."""
        if self._sock is None:
            return []
        lines = []
        for _ in range(max_messages):
            try:
                data, _addr = self._sock.recvfrom(1024)
            except BlockingIOError:
                break
            except ConnectionResetError:
                # Windows-only quirk: sending a UDP datagram to a port with no
                # listener can make a *later* recv on this socket raise
                # WinError 10054, even though UDP has no real "connection" to
                # reset. Same as BlockingIOError for our purposes.
                break
            except OSError as e:
                self.last_error = str(e)
                print(f"Warning: failed to read from ESP32: {e}")
                break
            line = data.decode("utf-8", errors="ignore").strip()
            if line:
                lines.append(line)
                match = RECONNECT_REPORT_RE.match(line)
                if match:
                    self.last_reconnect_report = (int(match.group(1)), line)
                    self.last_reconnect_report_time = time.monotonic()
        return lines

    def status(self) -> dict:
        with self._ping_lock:
            ping_ok = self.ping_ok
        return {
            "transport": "udp",
            "address": f"{self.host}:{self.port}",
            "open": self.is_open,
            "ping_ok": ping_ok,
            "last_error": self.last_error,
            "frames_sent": self.frames_sent,
            "bytes_sent": self.bytes_sent,
            "last_reconnect_report": self.last_reconnect_report,
            "last_reconnect_report_time": self.last_reconnect_report_time,
        }

    # --- Independent ICMP ping monitor ----------------------------------

    def _start_ping_thread(self):
        self._ping_stop.clear()
        self._ping_thread = threading.Thread(target=self._ping_monitor_loop, daemon=True)
        self._ping_thread.start()

    def _ping_monitor_loop(self):
        while not self._ping_stop.is_set():
            ok = _ping_once(self.host, self.ping_timeout_ms)
            with self._ping_lock:
                first_result = self.ping_ok is None
                changed = not first_result and ok != self.ping_ok
                self.ping_ok = ok
            if self._conn_logger is not None:
                if first_result:
                    self._conn_logger.info(f"[PING] Initial check: {self.host} is "
                                            f"{'reachable' if ok else 'NOT reachable'}")
                elif changed:
                    if ok:
                        self._conn_logger.info(f"[PING] {self.host} is reachable again")
                    else:
                        self._conn_logger.warning(f"[PING] {self.host} stopped responding to ICMP "
                                                   f"(host off the network, WiFi dropped, or ICMP blocked)")
            self._ping_stop.wait(self.ping_interval_s)
