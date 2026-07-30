"""
Debug dashboard for the LED matrix gesture detector.
=====================================================
A second, independent Tk window alongside the OpenCV video feed, showing
everything useful for diagnosing the pipeline: which transport is active
and whether it's actually connected, the raw/debounced/sent gesture, what
the ESP32 itself is reporting back, which sprites loaded, and a live zoomed
preview of the exact 16x16 frame currently being sent.

Driven from gesture.py's existing per-frame loop via update(state) then
pump() (-> root.update()) -- no new thread, no event loop of its own.
Dashboard.create() is the entry point: it swallows Tk import/construction
failures so a machine without Tk (e.g. a headless install) just runs
without this window instead of crashing the detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

try:
    import tkinter as tk
except ImportError:  # pragma: no cover - platform without Tk
    tk = None

import numpy as np

MATRIX_WIDTH = 16
MATRIX_HEIGHT = 16
PREVIEW_SCALE = 14  # on-screen pixels per matrix cell

COLOR_OK = "#2ecc71"
COLOR_WARN = "#f1c40f"
COLOR_BAD = "#e74c3c"
COLOR_UNKNOWN = "#95a5a6"
COLOR_TEXT = "#dddddd"


@dataclass
class DashboardState:
    """Everything the dashboard needs for one refresh. gesture.py builds one
    of these per frame and passes it to Dashboard.update() -- this is the
    entire interface between gesture.py and Tk, so gesture.py never touches
    a Tk widget directly."""

    # Transport
    transport_name: str = "?"
    transport_address: str = "?"
    transport_open: bool = False
    transport_reopen_attempts: int = 0
    transport_last_error: Optional[str] = None
    frames_sent: int = 0
    bytes_sent: int = 0

    # Link / liveness
    heartbeat_state: Optional[bool] = None  # None=never heard, True=ok, False=quiet
    last_contact_age_s: Optional[float] = None
    downtime_s: Optional[float] = None
    ping_ok: Optional[bool] = None  # meaningful only when transport_name == "udp"
    reconnects_this_session: int = 0
    last_reconnect_report: Optional[str] = None
    last_disconnect_reason: Optional[str] = None

    # ESP32 telemetry
    esp32_uptime_s: Optional[int] = None
    esp32_gesture_count: Optional[int] = None
    esp32_boot_reason: Optional[str] = None

    # Gesture
    raw_label: str = "none"
    debounce_candidate: Optional[str] = None
    debounce_run_length: int = 0
    debounce_stable_frames: int = 0
    confirmed_label: Optional[str] = None
    last_sent_label: Optional[str] = None
    last_sent_age_s: Optional[float] = None
    send_kind: Optional[str] = None  # "FRAME" or "LABEL (fallback color)"

    # Sprites: label -> status string (see sprites.load_all_sprites)
    sprite_statuses: dict = field(default_factory=dict)

    # Preview: exactly one of these should be set
    preview_rgb: Optional[np.ndarray] = None       # (16,16,3) uint8
    preview_fallback_color: Optional[tuple] = None  # (r,g,b), used when no sprite

    # Perf
    fps: float = 0.0
    frame_ms: float = 0.0
    messages_sent: int = 0
    messages_received: int = 0

    # Log: list of (text, is_connection_related) tuples, oldest first.
    # Dashboard only renders lines beyond what it already has.
    log_lines: list = field(default_factory=list)


def _fmt_age(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    return f"{seconds:.0f}s ago"


def _fmt_secs(seconds) -> str:
    if seconds is None:
        return "-"
    return f"{seconds:.0f}s"


class Dashboard:
    """Owns the Tk window. Use Dashboard.create(), not the constructor."""

    def __init__(self, root):
        self.root = root
        self.closed = False
        self.reload_requested = False
        self._label_widgets = {}
        self._var = {}
        self._log_len = 0
        self._build()

    @classmethod
    def create(cls) -> Optional["Dashboard"]:
        if tk is None:
            print("Warning: tkinter is not available -- running without the debug dashboard.")
            return None
        try:
            root = tk.Tk()
            root.title("ESP32 Gesture Debug")
        except tk.TclError as e:
            print(f"Warning: could not open the debug dashboard window: {e}")
            return None
        dashboard = cls(root)
        root.protocol("WM_DELETE_WINDOW", dashboard._on_close)
        return dashboard

    def _on_close(self):
        self.closed = True

    def consume_reload_request(self) -> bool:
        """Returns True (once) if the 'Reload sprites' button was clicked since the last call."""
        requested = self.reload_requested
        self.reload_requested = False
        return requested

    def pump(self):
        """Process pending Tk events. Call once per frame from the main loop."""
        if self.closed:
            return
        try:
            self.root.update()
        except tk.TclError:
            self.closed = True

    def destroy(self):
        if not self.closed:
            try:
                self.root.destroy()
            except tk.TclError:
                pass
        self.closed = True

    # --- layout ----------------------------------------------------------

    def _build(self):
        self.root.geometry("620x820")
        self.root.configure(bg="#2b2b2b")

        outer = tk.Frame(self.root, bg="#2b2b2b")
        outer.pack(fill="both", expand=True, padx=8, pady=8)

        left = tk.Frame(outer, bg="#2b2b2b")
        left.pack(side="left", fill="both", expand=True)
        right = tk.Frame(outer, bg="#2b2b2b", width=200)
        right.pack(side="right", fill="y", padx=(8, 0))

        self._build_section(left, "Transport", [
            ("transport", "Transport"),
            ("address", "Address"),
            ("status", "Status"),
            ("reopen_attempts", "Reopen attempts"),
            ("last_error", "Last error"),
            ("frames_sent", "Frames sent"),
            ("bytes_sent", "Bytes sent"),
        ])
        self._build_section(left, "Link", [
            ("heartbeat", "Heartbeat"),
            ("downtime", "Downtime"),
            ("ping", "Ping (ICMP)"),
            ("reconnects", "Reconnects (session)"),
            ("last_reconnect", "Last reconnect"),
            ("last_disconnect", "Last disconnect reason"),
        ])
        self._build_section(left, "ESP32 Telemetry", [
            ("uptime", "Uptime"),
            ("gesture_count", "Gesture count"),
            ("boot_reason", "Boot/reset reason"),
        ])
        self._build_section(left, "Gesture", [
            ("raw", "Raw (this frame)"),
            ("debounce", "Debounce"),
            ("confirmed", "Confirmed"),
            ("last_sent", "Last sent"),
            ("send_kind", "Send kind"),
        ])
        self._build_text_section(left, "Sprites", "sprites", height=7)
        self._build_section(left, "Perf", [
            ("fps", "Camera FPS"),
            ("frame_ms", "Frame loop"),
            ("messages_sent", "Messages sent"),
            ("messages_received", "Messages received"),
        ])

        # --- Preview -------------------------------------------------
        preview_box = tk.LabelFrame(right, text="Preview", bg="#2b2b2b", fg=COLOR_TEXT)
        preview_box.pack(fill="x")
        canvas_size = MATRIX_WIDTH * PREVIEW_SCALE
        self._preview_canvas = tk.Canvas(preview_box, width=canvas_size, height=canvas_size,
                                          bg="black", highlightthickness=1, highlightbackground="#444")
        self._preview_canvas.pack(padx=6, pady=6)
        self._preview_cells = [
            [
                self._preview_canvas.create_rectangle(
                    x * PREVIEW_SCALE, y * PREVIEW_SCALE,
                    (x + 1) * PREVIEW_SCALE, (y + 1) * PREVIEW_SCALE,
                    fill="black", outline="#222",
                )
                for x in range(MATRIX_WIDTH)
            ]
            for y in range(MATRIX_HEIGHT)
        ]

        tk.Button(right, text="Reload sprites (r)", command=self._on_reload_click).pack(
            fill="x", pady=(8, 0))

        # --- Log -------------------------------------------------------
        log_box = tk.LabelFrame(self.root, text="ESP32 Log", bg="#2b2b2b", fg=COLOR_TEXT)
        log_box.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        scrollbar = tk.Scrollbar(log_box)
        scrollbar.pack(side="right", fill="y")
        self._log_text = tk.Text(log_box, height=10, wrap="none", yscrollcommand=scrollbar.set,
                                  bg="#1e1e1e", fg=COLOR_TEXT, insertbackground=COLOR_TEXT)
        self._log_text.pack(fill="both", expand=True)
        scrollbar.config(command=self._log_text.yview)
        self._log_text.tag_config("conn", foreground=COLOR_WARN)
        self._log_text.config(state="disabled")

    def _on_reload_click(self):
        self.reload_requested = True

    def _build_section(self, parent, title, rows):
        box = tk.LabelFrame(parent, text=title, bg="#2b2b2b", fg=COLOR_TEXT, padx=6, pady=4)
        box.pack(fill="x", pady=(0, 6))
        for key, label_text in rows:
            row = tk.Frame(box, bg="#2b2b2b")
            row.pack(fill="x")
            tk.Label(row, text=label_text + ":", width=20, anchor="w",
                     bg="#2b2b2b", fg="#999999").pack(side="left")
            var = tk.StringVar(value="-")
            value_label = tk.Label(row, textvariable=var, anchor="w", bg="#2b2b2b", fg=COLOR_TEXT)
            value_label.pack(side="left", fill="x", expand=True)
            self._var[key] = var
            self._label_widgets[key] = value_label

    def _build_text_section(self, parent, title, key, height):
        box = tk.LabelFrame(parent, text=title, bg="#2b2b2b", fg=COLOR_TEXT, padx=6, pady=4)
        box.pack(fill="x", pady=(0, 6))
        text = tk.Text(box, height=height, wrap="none", bg="#1e1e1e", fg=COLOR_TEXT)
        text.pack(fill="x")
        text.config(state="disabled")
        self._var[key] = text

    def _set(self, key, text, color=COLOR_TEXT):
        var = self._var.get(key)
        if var is not None:
            var.set(text)
        widget = self._label_widgets.get(key)
        if widget is not None:
            widget.config(fg=color)

    # --- update ------------------------------------------------------

    def update(self, state: DashboardState):
        if self.closed:
            return

        # Transport
        self._set("transport", state.transport_name)
        self._set("address", state.transport_address)
        if state.transport_open:
            self._set("status", "OPEN", COLOR_OK)
        else:
            self._set("status", "CLOSED (retrying)", COLOR_BAD)
        self._set("reopen_attempts", str(state.transport_reopen_attempts))
        self._set("last_error", state.transport_last_error or "-",
                   COLOR_WARN if state.transport_last_error else COLOR_TEXT)
        self._set("frames_sent", str(state.frames_sent))
        self._set("bytes_sent", str(state.bytes_sent))

        # Link
        if state.heartbeat_state is None:
            self._set("heartbeat", "waiting for first contact...", COLOR_UNKNOWN)
        elif state.heartbeat_state:
            self._set("heartbeat", f"OK ({_fmt_age(state.last_contact_age_s)})", COLOR_OK)
        else:
            self._set("heartbeat", f"QUIET ({_fmt_age(state.last_contact_age_s)})", COLOR_BAD)
        self._set("downtime", _fmt_secs(state.downtime_s),
                   COLOR_BAD if state.downtime_s else COLOR_TEXT)
        if state.transport_name != "udp":
            self._set("ping", "n/a (serial)", COLOR_UNKNOWN)
        elif state.ping_ok is None:
            self._set("ping", "checking...", COLOR_UNKNOWN)
        elif state.ping_ok:
            self._set("ping", "OK", COLOR_OK)
        else:
            self._set("ping", "FAIL", COLOR_BAD)
        self._set("reconnects", str(state.reconnects_this_session))
        self._set("last_reconnect", state.last_reconnect_report or "-")
        self._set("last_disconnect", state.last_disconnect_reason or "-")

        # ESP32 telemetry
        self._set("uptime", _fmt_secs(state.esp32_uptime_s))
        self._set("gesture_count",
                   str(state.esp32_gesture_count) if state.esp32_gesture_count is not None else "-")
        self._set("boot_reason", state.esp32_boot_reason or "-")

        # Gesture
        self._set("raw", state.raw_label)
        if state.debounce_candidate:
            self._set("debounce",
                       f"{state.debounce_candidate} ({state.debounce_run_length}/{state.debounce_stable_frames})")
        else:
            self._set("debounce", "-")
        self._set("confirmed", state.confirmed_label or "-")
        if state.last_sent_label:
            self._set("last_sent", f"{state.last_sent_label} ({_fmt_age(state.last_sent_age_s)})")
        else:
            self._set("last_sent", "-")
        self._set("send_kind", state.send_kind or "-")

        # Sprites table
        sprites_widget = self._var["sprites"]
        sprites_widget.config(state="normal")
        sprites_widget.delete("1.0", "end")
        for label, status in state.sprite_statuses.items():
            sprites_widget.insert("end", f"{label:12s} {status}\n")
        sprites_widget.config(state="disabled")

        # Preview
        self._update_preview(state)

        # Perf
        self._set("fps", f"{state.fps:.1f}")
        self._set("frame_ms", f"{state.frame_ms:.1f} ms")
        self._set("messages_sent", str(state.messages_sent))
        self._set("messages_received", str(state.messages_received))

        # Log (append-only)
        self._update_log(state.log_lines)

        self.pump()

    def _update_preview(self, state: DashboardState):
        if state.preview_rgb is not None:
            rgb = state.preview_rgb
            for y in range(MATRIX_HEIGHT):
                for x in range(MATRIX_WIDTH):
                    r, g, b = (int(v) for v in rgb[y, x])
                    self._preview_canvas.itemconfig(self._preview_cells[y][x],
                                                     fill=f"#{r:02x}{g:02x}{b:02x}")
        else:
            r, g, b = state.preview_fallback_color or (0, 0, 0)
            fill = f"#{r:02x}{g:02x}{b:02x}"
            for y in range(MATRIX_HEIGHT):
                for x in range(MATRIX_WIDTH):
                    self._preview_canvas.itemconfig(self._preview_cells[y][x], fill=fill)

    def _update_log(self, log_lines):
        if len(log_lines) <= self._log_len:
            # Log was cleared/reset (e.g. new session) -- redraw from scratch.
            if len(log_lines) < self._log_len:
                self._log_text.config(state="normal")
                self._log_text.delete("1.0", "end")
                self._log_text.config(state="disabled")
                self._log_len = 0
            else:
                return
        new_lines = log_lines[self._log_len:]
        self._log_len = len(log_lines)
        self._log_text.config(state="normal")
        for text, is_conn in new_lines:
            tag = ("conn",) if is_conn else ()
            self._log_text.insert("end", text + "\n", tag)
        self._log_text.see("end")
        self._log_text.config(state="disabled")
