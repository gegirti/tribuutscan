"""The viewer's right-hand panels: one small class per group of settings or actions.

A Panel has a header that folds its body; `build(body)` makes its widgets once and `refresh()` updates
them (the viewer calls it about 30 times a second). Panels only call the viewer App's methods and read
its state, so adding a panel is one class here plus one entry in viewer.PANELS.
"""
import math
import tkinter as tk

from . import cloud

BOLD = ("TkDefaultFont", 10, "bold")


class Panel(tk.Frame):
    title = ""

    def __init__(self, parent, app, open_=True):
        super().__init__(parent, bd=1, relief="groove")
        self.app, self.open, self.shown = app, open_, {}
        self.header = tk.Button(self, anchor="w", relief="flat", font=BOLD, command=self.toggle)
        self.header.pack(fill="x")
        self.body = tk.Frame(self)
        self.build(self.body)
        self._place()

    def _place(self):
        self.header.configure(text=("▾ " if self.open else "▸ ") + self.title)
        if self.open:
            self.body.pack(fill="x", padx=6, pady=(0, 6))
        else:
            self.body.pack_forget()

    def toggle(self):
        self.open = not self.open
        self._place()

    def set(self, widget, **options):
        """configure() only when something changed (refresh runs every tick)."""
        if self.shown.get(widget) != options:
            self.shown[widget] = options
            widget.configure(**options)

    def build(self, body):
        pass

    def refresh(self):
        pass


class ScannerPanel(Panel):
    """Depth range, IR exposure and gain, colour brightness, Auto IR."""
    title = "Scanner"

    def build(self, body):
        row = tk.Frame(body)
        row.pack(fill="x", pady=(2, 4))
        tk.Label(row, text="Range").pack(side="left")
        tk.OptionMenu(row, self.app.range_var, *self.app.ranges, command=self.app.choose_range).pack(side="left", fill="x")
        for label, node, cid in self.app.sliders:
            self.app.slider(body, label, node, cid)
        buttons = tk.Frame(body)
        buttons.pack(fill="x", pady=(4, 0))
        tk.Button(buttons, text="Auto IR", command=self.app.start_auto).pack(side="left", expand=True, fill="x")
        tk.Button(buttons, text="Reset defaults", command=self.app.reset_controls).pack(side="left", expand=True, fill="x")


class CapturePanel(Panel):
    """Save a frame set, record raw sets (with IMU samples), mesh the latest recording."""
    title = "Capture"

    def build(self, body):
        tk.Button(body, text="Save frame (s)", command=self.app.save).pack(fill="x")
        self.record = tk.Button(body, command=self.app.toggle_record)
        self.record.pack(fill="x")
        self.mesh = tk.Button(body, command=self.app.mesh)
        self.mesh.pack(fill="x")

    def refresh(self):
        self.set(self.record, text="Stop recording (r)" if self.app.rec["recorder"] else "Record (r)")
        busy = self.app.job["process"] is not None
        self.set(self.mesh, text="Meshing..." if busy else "Mesh last recording (m)", state="disabled" if busy else "normal")


class ScanPanel(Panel):
    """Live scan into one merged cloud (worker process), continue, browser 3D view."""
    title = "Scan"
    STATES = {"idle": ("Start scan (c)", "normal"), "scanning": ("Stop scan (c)", "normal"),
              "saving": ("Saving...", "disabled"), "optimising": ("Optimising...", "disabled")}

    def build(self, body):
        row = tk.Frame(body)
        row.pack(fill="x")
        tk.OptionMenu(row, self.app.preset, *cloud.PRESETS_MM).pack(side="left")
        self.scan = tk.Button(row, command=self.app.toggle_scan)
        self.scan.pack(side="left", expand=True, fill="x")
        row = tk.Frame(body)
        row.pack(fill="x")
        tk.Button(row, text="Continue last", command=self.app.continue_scan).pack(side="left", expand=True, fill="x")
        tk.Button(row, text="Open 3D view", command=self.app.open_view).pack(side="left", expand=True, fill="x")
        row = tk.Frame(body)
        row.pack(fill="x")
        tk.Checkbutton(row, text="Remove base", variable=self.app.remove_base).pack(side="left")
        tk.Checkbutton(row, text="keep raw frames", variable=self.app.keep_raw).pack(side="left")

    def refresh(self):
        text, state = self.STATES[self.app.scan_state]
        self.set(self.scan, text=text, state=state)


class ImuPanel(Panel):
    """The IMU live: a horizon from gravity (tilt), the turn rate, still or moving."""
    title = "IMU"
    W, H = 240, 120

    def build(self, body):
        tk.Checkbutton(body, text="IMU on", variable=self.app.imu_wanted, command=self.app.toggle_imu).pack(anchor="w")
        self.canvas = tk.Canvas(body, width=self.W, height=self.H, bg="#9ecbea", highlightthickness=0)
        self.canvas.pack(pady=2)
        self.ground = self.canvas.create_polygon(0, 0, 0, 0, fill="#8d6e4a", outline="")
        self.canvas.create_line(self.W / 2 - 30, self.H / 2, self.W / 2 - 8, self.H / 2, width=3)
        self.canvas.create_line(self.W / 2 + 8, self.H / 2, self.W / 2 + 30, self.H / 2, width=3)
        self.bar = self.canvas.create_rectangle(0, self.H - 8, 0, self.H, fill="#c62828", outline="")
        self.tilt = tk.Label(body, anchor="w", font=("TkFixedFont", 10))
        self.tilt.pack(fill="x")
        self.motion = tk.Label(body, anchor="w", font=("TkFixedFont", 10))
        self.motion.pack(fill="x")
        self.ticks = 0

    def refresh(self):
        self.ticks += 1
        if self.ticks % 2:  # 15 Hz is plenty for a horizon
            return
        st = self.app.imu.state() if self.app.imu is not None else None
        if st is None:
            self.set(self.tilt, text=self.app.imu_note or ("IMU lost: switch it off and on" if self.app.imu else "IMU off"))
            self.set(self.motion, text="", fg="black")
            self.canvas.coords(self.ground, 0, 0, 0, 0)
            self.canvas.coords(self.bar, 0, self.H - 8, 0, self.H)
            return
        w, h = self.W, self.H
        offset = st["pitch"] / 45 * h / 2  # +-45 degrees of pitch fill the view
        a = math.radians(st["roll"])
        dx, dy = math.cos(a) * w * 2, math.sin(a) * w * 2
        cx, cy = w / 2, h / 2 + offset
        nx, ny = -math.sin(a) * h * 4, math.cos(a) * h * 4  # towards the ground side
        self.canvas.coords(self.ground, cx - dx, cy - dy, cx + dx, cy + dy, cx + dx + nx, cy + dy + ny, cx - dx + nx, cy - dy + ny)
        self.canvas.coords(self.bar, 0, h - 8, w * min(st["turn"], 90) / 90, h)
        self.set(self.tilt, text=f"pitch {st['pitch']:+6.1f} deg   roll {st['roll']:+6.1f} deg")
        state = "still" if st["still"] else "moving"
        self.set(self.motion, text=f"turn {st['turn']:5.1f} deg/s {state:6}  {st['rate']:3.0f} Hz  {st['temp']:4.1f} C",
                 fg="#2e7d32" if st["still"] else "#ef6c00")


class ViewPanel(Panel):
    """What the previews show."""
    title = "View"

    def build(self, body):
        tk.Checkbutton(body, text="Find markers (ringed in colour)", variable=self.app.find_markers,
                       command=self.app.toggle_markers).pack(anchor="w")
