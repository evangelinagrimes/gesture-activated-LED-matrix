"""
LED Matrix GUI
===============
Interactive control panel for the ESP32 LED matrix: load an image, pick
two-tone or full-color rendering, adjust colors/threshold/brightness with
an instant on-screen preview, and push it to the matrix either live as you
drag a slider or on demand with a Send button.

Run:
    python gui.py

Built on the same image_loader.py / renderer.py / matrix_link.py that
send_image.py (the scriptable CLI) uses -- render_rgb() in renderer.py is
the single function that feeds both the preview shown here and the actual
wire payload, so the preview cannot drift from what the panel shows.

Requires the RGB888 firmware (v2) -- see led_matrix_esp/led_matrix_esp.ino
and matrix_link.py's module docstring for the wire protocol. Connecting to
older grayscale-only firmware will show a warning that the board didn't
answer a ping.

--------------------------------------------------------------------------
Threading
--------------------------------------------------------------------------
One long-lived daemon worker thread owns the MatrixLink exclusively (it is
not thread-safe). The GUI thread never touches serial directly, and the
worker thread never touches Tk widgets directly -- they communicate only
through three queues:

  cmd_q   (GUI -> worker)  "connect"/"disconnect"/"quit" commands
  frame_q (GUI -> worker)  maxsize=1, latest-wins -- see _push_frame()
  evt_q   (worker -> GUI)  status/error/ack events, drained by _drain_events()
                           via root.after(), which is safe to call from the
                           GUI thread's own event loop (unlike calling it
                           from the worker thread, which is not safe)

frame_q is capped at 1 and drained-then-filled by the producer (the GUI
thread) because a ttk.Scale fires on every motion event -- a few seconds
of dragging is 100+ callbacks, and at ~78ms per round trip an unbounded
queue would mean several seconds of visible lag after you release the
mouse, all of it already-stale frames. A 60ms UI-side debounce (just under
that round trip) on top of the drain means at most one frame is ever in
flight and one queued, and the *last* slider value always wins.
--------------------------------------------------------------------------
"""

import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk

from PIL import ImageTk

import image_loader
import matrix_link
import renderer

if sys.platform == "win32":
    # Tk 8.6 isn't DPI-aware on its own, which blurs the preview on a scaled
    # display -- and the preview being crisp is the whole point of this UI.
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

PREVIEW_SCALE = 24        # on-screen pixels per matrix pixel
SEND_DEBOUNCE_MS = 60     # just under the ~78ms serial round trip
EVENT_POLL_MS = 50        # how often the GUI thread drains evt_q
CMD_POLL_TIMEOUT_S = 0.05  # how often the worker thread checks cmd_q
WORKER_JOIN_TIMEOUT_S = 2.5

DEFAULT_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "attempt1.txt")

STATUS_COLORS = {"info": "#1a1a1a", "warning": "#a06000", "error": "#b00020"}


def _to_hex(rgb: tuple) -> str:
    return "#%02x%02x%02x" % rgb


class MatrixApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("LED Matrix Control")
        root.geometry("920x600")
        root.minsize(820, 520)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Image/render state (GUI thread only)
        self.img: "image_loader.LoadedImage | None" = None
        self.rgb = None                # last-rendered (H, W, 3) uint8 array
        self.color_on = renderer.DEFAULTS.color_on
        self.color_off = renderer.DEFAULTS.color_off
        self._photo = None             # must stay referenced or PhotoImage goes blank
        self._canvas_img_id = None
        self._send_job = None          # after() id for the send debounce
        self._ports_cache = []         # [(device, description), ...] from the last refresh

        # Connection state (GUI thread's mirror of the worker's real state --
        # the GUI thread never touches MatrixLink directly)
        self.connected = False

        # Worker thread + queues (see module docstring)
        self.cmd_q = queue.Queue()
        self.frame_q = queue.Queue(maxsize=1)
        self.evt_q = queue.Queue()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)

        # tk variables backing the controls
        self.var_mode = tk.StringVar(value=renderer.TWO_TONE)
        self.var_threshold = tk.IntVar(value=renderer.DEFAULTS.threshold)
        self.var_smooth = tk.BooleanVar(value=renderer.DEFAULTS.smooth)
        self.var_brightness = tk.IntVar(value=renderer.DEFAULTS.brightness)
        self.var_live = tk.BooleanVar(value=True)

        self._build_ui()
        self._on_refresh_ports()
        self._worker_thread.start()
        self.root.after(EVENT_POLL_MS, self._drain_events)

        self._load_image_path(DEFAULT_IMAGE_PATH)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=0)
        outer.rowconfigure(0, weight=1)

        self._build_preview(outer)

        right = ttk.Frame(outer)
        right.grid(row=0, column=1, sticky="n", padx=(12, 0))
        self._build_connection(right)
        self._build_image(right)
        self._build_render(right)
        self._build_send(right)

        self._build_statusbar(outer)

    def _build_preview(self, parent):
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=0, sticky="n")

        size = image_loader.MATRIX_WIDTH * PREVIEW_SCALE
        self.canvas = tk.Canvas(frame, width=size, height=size, bg="black", highlightthickness=1,
                                 highlightbackground="#444")
        self.canvas.pack()

        self.lbl_image_path = ttk.Label(frame, text="(no image loaded)")
        self.lbl_image_path.pack(anchor="w", pady=(8, 0))
        self.lbl_image_info = ttk.Label(frame, text="", foreground="#666")
        self.lbl_image_info.pack(anchor="w")
        ttk.Label(frame, text="Logical view -- serpentine wiring is applied on the ESP32.",
                  foreground="#666", wraplength=size).pack(anchor="w", pady=(4, 0))

    def _build_connection(self, parent):
        box = ttk.LabelFrame(parent, text="Connection", padding=8)
        box.pack(fill="x", pady=(0, 8))
        box.columnconfigure(0, weight=1)

        self.combo_port = ttk.Combobox(box, state="readonly")
        self.combo_port.grid(row=0, column=0, sticky="ew")
        ttk.Button(box, text="Refresh", command=self._on_refresh_ports).grid(row=0, column=1, padx=(6, 0))

        self.btn_connect = ttk.Button(box, text="Connect", command=self._on_toggle_connection)
        self.btn_connect.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        self.lbl_firmware = ttk.Label(box, text="", foreground="#666")
        self.lbl_firmware.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

    def _build_image(self, parent):
        box = ttk.LabelFrame(parent, text="Image", padding=8)
        box.pack(fill="x", pady=(0, 8))
        ttk.Button(box, text="Open Image...", command=self._on_open_image).pack(fill="x")

    def _build_render(self, parent):
        box = ttk.LabelFrame(parent, text="Rendering", padding=8)
        box.pack(fill="x", pady=(0, 8))

        mode_row = ttk.Frame(box)
        mode_row.pack(fill="x")
        ttk.Radiobutton(mode_row, text="Two-tone", variable=self.var_mode,
                         value=renderer.TWO_TONE, command=self._on_mode_changed).pack(side="left")
        self.radio_full_color = ttk.Radiobutton(
            mode_row, text="Full color", variable=self.var_mode,
            value=renderer.FULL_COLOR, command=self._on_mode_changed)
        self.radio_full_color.pack(side="left", padx=(10, 0))

        self.lbl_threshold = ttk.Label(box, text="Threshold")
        self.lbl_threshold.pack(anchor="w", pady=(8, 0))
        self.scale_threshold = ttk.Scale(box, from_=0, to=255, variable=self.var_threshold,
                                          command=self._on_slider_changed)
        self.scale_threshold.pack(fill="x")

        self.chk_smooth = ttk.Checkbutton(box, text="Smooth ramp (instead of hard threshold)",
                                           variable=self.var_smooth, command=self._on_settings_changed)
        self.chk_smooth.pack(anchor="w", pady=(4, 0))

        colors_row = ttk.Frame(box)
        colors_row.pack(fill="x", pady=(8, 0))
        self.btn_color_on = tk.Button(colors_row, text="Color ON", bg=_to_hex(self.color_on),
                                       command=lambda: self._on_pick_color("on"))
        self.btn_color_on.pack(side="left", fill="x", expand=True)
        self.btn_color_off = tk.Button(colors_row, text="Color OFF", bg=_to_hex(self.color_off),
                                        fg="white", command=lambda: self._on_pick_color("off"))
        self.btn_color_off.pack(side="left", fill="x", expand=True, padx=(6, 0))

        ttk.Label(box, text="Brightness").pack(anchor="w", pady=(8, 0))
        ttk.Scale(box, from_=0, to=100, variable=self.var_brightness,
                  command=self._on_slider_changed).pack(fill="x")

        ttk.Button(box, text="Reset to defaults", command=self._on_reset).pack(fill="x", pady=(8, 0))

    def _build_send(self, parent):
        box = ttk.LabelFrame(parent, text="Send", padding=8)
        box.pack(fill="x")
        self.btn_send = ttk.Button(box, text="Send to Matrix", command=self._on_send, state="disabled")
        self.btn_send.pack(fill="x")
        ttk.Checkbutton(box, text="Live update", variable=self.var_live).pack(anchor="w", pady=(6, 0))

    def _build_statusbar(self, parent):
        self.lbl_status = ttk.Label(parent, text="Ready.", relief="sunken", anchor="w", padding=(6, 3))
        self.lbl_status.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))

    # ------------------------------------------------------------------
    # Rendering / preview
    # ------------------------------------------------------------------

    def _current_settings(self) -> renderer.RenderSettings:
        return renderer.RenderSettings(
            mode=self.var_mode.get(),
            threshold=int(round(self.var_threshold.get())),
            color_on=self.color_on,
            color_off=self.color_off,
            smooth=self.var_smooth.get(),
            brightness=int(round(self.var_brightness.get())),
        )

    def _on_slider_changed(self, _value):
        # ttk.Scale's command fires with a float string; round it back to an
        # int so the numeric value (and the label, if one is added later)
        # doesn't read like 127.94827586.
        self.var_threshold.set(int(round(self.var_threshold.get())))
        self.var_brightness.set(int(round(self.var_brightness.get())))
        self._on_settings_changed()

    def _on_mode_changed(self):
        self._update_control_states()
        self._on_settings_changed()

    def _update_control_states(self):
        full_color = self.var_mode.get() == renderer.FULL_COLOR
        state = "disabled" if full_color else "normal"
        self.scale_threshold.configure(state=state)
        self.chk_smooth.configure(state=state)
        self.btn_color_on.configure(state=state)
        self.btn_color_off.configure(state=state)

        has_color = self.img is not None and self.img.rgb is not None
        self.radio_full_color.configure(state=("normal" if has_color else "disabled"))

    def _on_settings_changed(self, *_):
        if self.img is None:
            return
        self._rerender()
        self._update_preview()
        if self.var_live.get() and self.connected:
            self._schedule_send()

    def _rerender(self):
        try:
            self.rgb = renderer.render_rgb(self.img, self._current_settings())
        except ValueError:
            # Shouldn't normally happen -- the full-color radio is disabled
            # without color data -- but fail safe rather than crash.
            self.rgb = None

    def _update_preview(self):
        if self.rgb is None:
            return
        preview = renderer.to_preview_image(self.rgb, scale=PREVIEW_SCALE)
        self._photo = ImageTk.PhotoImage(preview)
        if self._canvas_img_id is None:
            self._canvas_img_id = self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        else:
            self.canvas.itemconfig(self._canvas_img_id, image=self._photo)

    def _on_reset(self):
        self.var_mode.set(renderer.DEFAULTS.mode)
        self.var_threshold.set(renderer.DEFAULTS.threshold)
        self.var_smooth.set(renderer.DEFAULTS.smooth)
        self.var_brightness.set(renderer.DEFAULTS.brightness)
        self.color_on = renderer.DEFAULTS.color_on
        self.color_off = renderer.DEFAULTS.color_off
        self.btn_color_on.configure(bg=_to_hex(self.color_on))
        self.btn_color_off.configure(bg=_to_hex(self.color_off))
        self._update_control_states()
        self._on_settings_changed()

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _on_open_image(self):
        path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[
                ("Hex image", "*.h *.hpp *.c *.inc *.txt"),
                ("Image", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"),
                ("All files", "*.*"),
            ],
            parent=self.root,
        )
        if not path:
            return
        self._load_image_path(path)

    def _load_image_path(self, path: str):
        try:
            img = image_loader.load_image(path)
        except image_loader.ImageLoadError as e:
            # Keep whatever was previously loaded -- a bad file shouldn't
            # blank out a perfectly good preview.
            messagebox.showerror("Could not load image", str(e), parent=self.root)
            self._set_status(f"Could not load {os.path.basename(path)}: {e}", "error")
            return

        self.img = img
        self.lbl_image_path.configure(text=os.path.basename(path))
        self.lbl_image_info.configure(text=f"{img.source_size[0]}x{img.source_size[1]} source")
        if img.rgb is None and self.var_mode.get() == renderer.FULL_COLOR:
            self.var_mode.set(renderer.TWO_TONE)
        self._update_control_states()
        self._on_settings_changed()
        self.btn_send.configure(state=("normal" if self.connected else "disabled"))
        self._set_status(f"Loaded {os.path.basename(path)}.", "info")

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _on_refresh_ports(self):
        self._ports_cache = matrix_link.list_ports()
        if self._ports_cache:
            self.combo_port["values"] = [f"{d} — {desc}" for d, desc in self._ports_cache]
            self.combo_port.current(0)
            self.btn_connect.configure(state="normal")
        else:
            self.combo_port["values"] = ["(no serial ports found)"]
            self.combo_port.current(0)
            self.btn_connect.configure(state="disabled")

    def _selected_port(self):
        idx = self.combo_port.current()
        if not self._ports_cache or idx < 0 or idx >= len(self._ports_cache):
            return None
        return self._ports_cache[idx][0]

    def _on_toggle_connection(self):
        if self.connected:
            self.cmd_q.put(("disconnect", None))
            return
        port = self._selected_port()
        if not port:
            self._set_status("No serial port selected -- plug in the ESP32 and hit Refresh.", "error")
            return
        self.btn_connect.configure(state="disabled")
        self._set_status(f"Connecting to {port}...", "info")
        self.cmd_q.put(("connect", port))

    def _set_connected(self, connected: bool, info: str = ""):
        self.connected = connected
        self.btn_connect.configure(state="normal", text=("Disconnect" if connected else "Connect"))
        self.lbl_firmware.configure(text=info if connected else "")
        self.btn_send.configure(state=("normal" if (connected and self.img is not None) else "disabled"))

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    def _schedule_send(self):
        if self._send_job is not None:
            self.root.after_cancel(self._send_job)
        self._send_job = self.root.after(SEND_DEBOUNCE_MS, self._push_frame)

    def _push_frame(self):
        self._send_job = None
        if self.rgb is None or not self.connected:
            return
        payload = renderer.to_payload(self.rgb)
        # Latest-wins: drop whatever's queued before adding the new one, so
        # the worker never sends a stale frame from mid-drag (see module
        # docstring).
        try:
            self.frame_q.get_nowait()
        except queue.Empty:
            pass
        self.frame_q.put_nowait(payload)

    def _on_send(self):
        if self.rgb is None:
            self._set_status("No image loaded.", "error")
            return
        if not self.connected:
            self._set_status("Not connected.", "error")
            return
        self._push_frame()

    # ------------------------------------------------------------------
    # Worker thread -- the only thing that ever touches MatrixLink
    # ------------------------------------------------------------------

    def _worker(self):
        link = None
        while True:
            try:
                cmd, arg = self.cmd_q.get(timeout=CMD_POLL_TIMEOUT_S)
            except queue.Empty:
                cmd, arg = None, None

            if cmd == "quit":
                if link is not None:
                    link.close()
                return

            elif cmd == "connect":
                if link is not None:
                    link.close()
                    link = None
                try:
                    link = matrix_link.MatrixLink(arg)
                    link.open()
                    fw_id = link.ping()
                    if fw_id is None:
                        time.sleep(matrix_link.BOOT_SETTLE_S)
                        fw_id = link.ping()
                    if fw_id is None:
                        self.evt_q.put(("connected", ""))
                        self.evt_q.put(("warning",
                                         "Port open, but the firmware didn't answer -- "
                                         "is it flashed with the RGB firmware?"))
                    else:
                        self.evt_q.put(("connected", fw_id))
                except matrix_link.MatrixLinkError as e:
                    link = None
                    self.evt_q.put(("error", str(e)))

            elif cmd == "disconnect":
                if link is not None:
                    link.close()
                    link = None
                self.evt_q.put(("disconnected", ""))

            if link is not None and link.is_open:
                try:
                    payload = self.frame_q.get_nowait()
                except queue.Empty:
                    continue
                try:
                    ack = link.send_rgb(payload)
                    self.evt_q.put(("ack", ack or ""))
                except matrix_link.MatrixLinkError as e:
                    link.close()
                    link = None
                    self.evt_q.put(("disconnected", str(e)))

    def _drain_events(self):
        try:
            while True:
                kind, payload = self.evt_q.get_nowait()
                if kind == "connected":
                    self._set_connected(True, payload)
                    self._set_status(f"Connected. {payload}" if payload else "Connected.", "info")
                elif kind == "disconnected":
                    self._set_connected(False)
                    if payload:
                        self._set_status(f"Disconnected: {payload}", "error")
                        self._on_refresh_ports()
                    else:
                        self._set_status("Disconnected.", "info")
                elif kind == "error":
                    self._set_connected(False)
                    self._set_status(f"Connection failed: {payload}", "error")
                    self._on_refresh_ports()
                elif kind == "warning":
                    self._set_status(payload, "warning")
                elif kind == "ack":
                    self._set_status(f"Sent. ESP32: {payload}" if payload else "Sent (no response).", "info")
        except queue.Empty:
            pass
        self.root.after(EVENT_POLL_MS, self._drain_events)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _set_status(self, text: str, kind: str = "info"):
        self.lbl_status.configure(text=text, foreground=STATUS_COLORS.get(kind, "#1a1a1a"))

    def _on_pick_color(self, which: str):
        current = self.color_on if which == "on" else self.color_off
        _, hex_str = colorchooser.askcolor(
            color=_to_hex(current), parent=self.root,
            title=f"Choose {'ON' if which == 'on' else 'OFF'} color")
        if hex_str is None:
            return  # cancelled
        rgb = tuple(int(hex_str[i:i + 2], 16) for i in (1, 3, 5))
        if which == "on":
            self.color_on = rgb
            self.btn_color_on.configure(bg=hex_str)
        else:
            self.color_off = rgb
            self.btn_color_off.configure(bg=hex_str)
        self._on_settings_changed()

    def _on_close(self):
        if self._send_job is not None:
            self.root.after_cancel(self._send_job)
        self.cmd_q.put(("quit", None))
        self._worker_thread.join(timeout=WORKER_JOIN_TIMEOUT_S)
        self.root.destroy()


def main():
    root = tk.Tk()
    MatrixApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
