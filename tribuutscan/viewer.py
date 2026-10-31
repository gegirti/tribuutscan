"""Live viewer: depth in the middle, colour and IR on the left, setting panels on the right.

Run with `python -m tribuutscan.viewer`. Panels (panels.py, each folds):
- Scanner: Range switches close range (15-33 cm) / long range (~20-72 cm, an estimate until calibrated);
  the streams pause about 10 s while it switches (vendor.py). IR exposure and gain are controls on the
  depth function: they change the IR sensor that depth is computed from (14-bit USB depth, usbdepth.py,
  then owns them). Auto IR steps them until the IR image has little clipping (clipping costs depth).
- Capture: `s` saves the current frame set to captures/<timestamp>/; `r` records raw sets (and IMU
  samples, imu.csv) to captures/<timestamp>-rec/; `m` fuses the latest recording into mesh.ply
  (fuse.py in the Open3D .venv).
- Scan: `c` merges frames live into one compact cloud (cloud.py) in a worker process; Open 3D view
  shows it growing in the browser. Stop saves captures/<timestamp>-scan/.
- IMU: the scanner's accelerometer and gyroscope (imu.py): tilt from gravity, turn rate.
- View: markers (white dot in a black ring) are found in the colour image a few times a second and
  ringed in the colour view; saved frames list them in info.json.
The distance bar reads the centre of the depth image. Events go to the message line, the console
and captures/viewer.log. One viewer at a time (captures/.viewer.lock). When the scanner resets its USB
connection (it does so on its own now and then), the viewer reconnects by itself.
"""
import collections
import fcntl
import multiprocessing
import os
import queue
import subprocess
import threading
import time
import tkinter as tk
import tkinter.messagebox
import webbrowser
from pathlib import Path

import numpy as np

from . import frames as fr
from . import calib, cloud, imu, livescan, markers, panels, pointcloud, usbdepth, v4l2, vendor
from .depth import decode_depth
from .image import depth_to_rgb, ppm, yuyv_pixels

SIDE_W, DEPTH_W = 400, 800  # preview widths: colour and IR on the left, depth in the middle
PREVIEW_FPS = 15  # previews redraw at most this often; every frame is still captured and paired

# (label, node, control id): what the Scanner panel's sliders drive.
SLIDERS = [
    ("IR exposure", "depth", v4l2.CID_EXPOSURE),
    ("IR gain", "depth", v4l2.CID_GAIN),
    ("Colour brightness", "color", v4l2.CID_BRIGHTNESS),
]
PANELS = [panels.ScannerPanel, panels.CapturePanel, panels.ScanPanel, panels.ImuPanel, panels.ViewPanel]

# Auto IR: keep clipped pixels between CLIP_LO and CLIP_HI percent; brighten only if the mean is low.
CLIP_LO, CLIP_HI, MEAN_LO = 0.3, 2.0, 35
GAIN_CAP = 2000  # auto never raises gain past this (gain clipped badly in the 25 cm test)
AUTO_STEP_MS, AUTO_MAX_STEPS = 600, 16

RECORD_FPS = 10  # about 49 MB/s of raw data

GOOD, EDGE, BAD = "#2e7d32", "#ef6c00", "#c62828"

REPO = Path(__file__).resolve().parent.parent
VENV_PYTHON = REPO / ".venv" / "bin" / "python"  # has Open3D; see fuse.py


LOG = fr.CAPTURES / "viewer.log"
LOCK = fr.CAPTURES / ".viewer.lock"
RETRY_S = 10.0  # after a failed reconnect, try again this much later


def latest_recording():
    recordings = sorted(fr.CAPTURES.glob("*-rec"))
    return recordings[-1] if recordings else None


def preview_index(size, out):
    """Source indices that sample `size` pixels down (or up) to `out` (nearest neighbour)."""
    return np.arange(out) * size // out


def fit(img, width):
    """An image resampled to `width` pixels across (aspect kept)."""
    h, w = img.shape[:2]
    return img[np.ix_(preview_index(h, round(h * width / w)), preview_index(w, width))]


class Stream(threading.Thread):
    """Reads one node in the background; keeps the latest raw frame, its preview and stats.

    `stamp` is the camera's capture time (kernel timestamp, time.monotonic() scale); the last few
    frames are kept so `nearest(t)` can pair streams by time instead of taking whatever is latest.
    """

    def __init__(self, cap, render):
        super().__init__(daemon=True)
        self.cap = cap
        self.resumed, self.idle = threading.Event(), threading.Event()  # pause() / resume(cap)
        self.render = render
        self.latest = None
        self.version = 0  # bumped with each new preview (the window redraws only on change)
        self.stats = {}
        self.raw = None
        self.stamp = 0.0
        self.recent = collections.deque(maxlen=12)  # (stamp, raw): ~0.4 s, so pairing works under load
        self.fps = 0.0

    def run(self):
        while True:
            try:
                self._stream()
            except OSError:  # the device went away mid-stream: wait for resume() like after pause()
                pass
            self.idle.set()
            self.resumed.wait()
            self.resumed.clear()
            self.idle.clear()

    def _stream(self):
        n, t0, rendered = 0, time.time(), 0.0
        for buf in self.cap.frames():
            if len(buf) < self.cap.sizeimage:  # truncated USB transfer
                continue
            self.raw, self.stamp = buf, self.cap.last_time
            self.recent.append((self.stamp, buf))
            n += 1  # the fps shown is the camera's rate, not the preview's
            if time.time() - t0 >= 1:
                self.fps, n, t0 = n / (time.time() - t0), 0, time.time()
            if self.cap.ready() or time.time() - rendered < 1 / PREVIEW_FPS:
                continue  # behind, or a preview is fresh enough: every frame still goes to `recent`
            rendered = time.time()
            self.latest, self.stats = self.render(self.cap, buf)
            self.version += 1

    def nearest(self, t):
        """(raw, stamp) of the kept frame captured closest to time t."""
        recent = list(self.recent)
        if not recent:
            return self.raw, self.stamp
        stamp, raw = min(recent, key=lambda sr: abs(sr[0] - t))
        return raw, stamp

    def pause(self, timeout=8.0):
        """Close the capture and wait until this thread has let go of it (a pending read ends within 5 s),
        so a new capture can't be read through the old one's file descriptor."""
        self.cap.close()
        self.idle.wait(timeout)
        self.fps = 0.0

    def resume(self, cap):
        self.cap = cap
        self.recent.clear()
        self.idle.clear()  # running again from now on, before the thread wakes (the reconnect check reads it)
        self.resumed.set()

    def stop(self):
        self.cap.close()


MARKER_HZ = 5  # marker search rate; one search takes ~0.1 s at half resolution
MARKER_RADII = (3, 4, 5, 7)  # dot radius in half-resolution pixels (smaller ones were glints)
found_markers = {"on": True, "list": [], "stamp": 0.0}  # list in full-resolution pixels


def find_markers_loop(color):
    """Background thread: search the latest colour frame for markers at half resolution."""
    while True:
        t = time.time()
        if found_markers["on"] and color.raw and color.stamp != found_markers["stamp"]:
            found_markers["stamp"] = color.stamp
            luma = np.frombuffer(color.raw, np.uint8, color.cap.width * color.cap.height * 2)[0::2]
            found_markers["list"] = markers.detect_fast(luma.reshape(color.cap.height, color.cap.width))
        elif not found_markers["on"]:
            found_markers["list"] = []
        time.sleep(max(0.0, 1 / MARKER_HZ - (time.time() - t)))


def render_color(cap, buf, width=SIDE_W):
    ys = preview_index(cap.height, round(cap.height * width / cap.width))
    v, u = np.meshgrid(ys, preview_index(cap.width, width), indexing="ij")  # convert only the shown pixels
    rgb = yuyv_pixels(buf, cap.width, u.ravel(), v.ravel()).reshape(v.shape + (3,))
    hits, s = found_markers["list"], width / cap.width
    if hits:
        rgb = markers.draw(rgb, [(x * s, y * s, r * s, c) for x, y, r, c in hits])
    return ppm(rgb), {"text": f"markers {len(hits)}" if found_markers["on"] else ""}


def render_ir(cap, buf, width=SIDE_W):
    g = np.frombuffer(buf, np.uint8, cap.width * cap.height).reshape(cap.height, cap.width)
    g = fit(g[:, ::-1], width)  # IR is mirrored relative to colour/depth
    mean, clipped = float(g.mean()), 100 * float((g >= 250).mean())
    stats = {"mean": mean, "clipped": clipped, "text": f"mean {mean:.0f}  clipped {clipped:.1f}%"}
    return ppm(np.repeat(g[..., None], 3, axis=2)), stats


def render_depth(cap, buf, width=DEPTH_W):
    d = decode_depth(buf, cap.width, cap.height, cap.bytesperline)
    h, w = d.shape
    centre = d[h * 3 // 8:h * 5 // 8, w * 3 // 8:w * 5 // 8]
    valid = centre[centre > 0]
    stats = {
        "text": f"valid {100 * (d > 0).mean():.0f}%",
        "centre_valid": valid.size / centre.size,
        "centre_cm": float(pointcloud.depth_to_mm(np.median(valid))) / 10 if valid.size else None,
    }
    return ppm(depth_to_rgb(fit(d, width))), stats  # colour-map only the shown pixels


def distance_hint(stats):
    """(text, colour) for the distance bar, from the depth centre stats (the active depth mode's window)."""
    near, far = pointcloud.window_cm()
    cm = stats.get("centre_cm")
    if cm is None or stats["centre_valid"] < 0.2:
        return f"Centre: no depth. Too close or too far; aim for {near + 3:.0f}-{far - 4:.0f} cm", BAD
    text = f"Centre: ~{cm:.1f} cm" + (" (estimate)" if calib.CAMERA.law_estimate else "")
    if cm < near + 2:
        return text + " - near the close edge, move back", EDGE
    if cm > far - 3:
        return text + " - near the far edge, move closer", EDGE
    return text + " - good", GOOD


IR_EXPOSURE_MODE = 2  # IR node's auto exposure menu: 2 lets the depth function's IR exposure and gain act;
# in 0 (auto, seen after a power-up on 2026-10-31) the sliders and Auto IR changed nothing


def ir_exposure_by_hand(nodes, log):
    try:
        v4l2.set_control(nodes["ir"], v4l2.CID_EXPOSURE_AUTO, IR_EXPOSURE_MODE)
    except OSError as e:
        log(f"IR exposure mode not set ({e.strerror}): the IR sliders may do nothing")


RANGES = {"Close range 15-33 cm": calib.CLOSE, "Long range ~20-72 cm (estimate)": calib.LONG}
SWITCH_SETTLE_S = 6.0  # the scanner may reset its USB connection a few seconds after its streams stop


def depth_title(cap):
    kind = "14-bit USB" if isinstance(cap, usbdepth.DepthCapture) else "10-bit"
    return f"Depth, {kind}, {'long' if cap.depth_mode.startswith('Long') else 'close'} range (near = red)"


def wait_for_scanner(timeout=15.0):
    """v4l2.discover() once the scanner's three nodes are back (after a stop or USB reset)."""
    end = time.monotonic() + timeout
    while True:
        try:
            return v4l2.discover()
        except (RuntimeError, OSError):
            if time.monotonic() > end:
                raise RuntimeError("the scanner did not come back")
            time.sleep(0.5)


def node_controls(target):
    """Controls of a V4L2 node path, or of the USB depth capture (it owns the depth function's controls)."""
    return target.controls() if isinstance(target, usbdepth.DepthCapture) else v4l2.controls(target)


def get_control(target, cid):
    return target.get_control(cid) if isinstance(target, usbdepth.DepthCapture) else v4l2.get_control(target, cid)


def set_control(target, cid, value):
    if isinstance(target, usbdepth.DepthCapture):
        target.set_control(cid, value)
    else:
        v4l2.set_control(target, cid, value)


class App:
    """The viewer's state and actions. The panels (panels.py) call these methods; `tick` runs in the Tk
    thread about 30 times a second. The range switch and the record loop run in their own threads and
    touch no Tk objects (they leave flags and the message for `tick`)."""

    def __init__(self, root):
        self.root, self.message = root, ""
        self.nodes = v4l2.discover()
        ir_exposure_by_hand(self.nodes, self.log)
        opened = []  # closed again if a later one fails, so a retry finds the scanner free
        try:
            for open_one in (lambda: fr.open_depth(self.nodes, self.log),
                             lambda: v4l2.Capture(self.nodes["color"], "YUYV", fr.W, fr.H),
                             lambda: v4l2.Capture(self.nodes["ir"], "GREY", fr.W, fr.H)):
                opened.append(open_one())
        except BaseException:
            for c in opened:
                c.close()
            raise
        cap, color, ir = opened
        self.targets = {**self.nodes, "depth": cap if isinstance(cap, usbdepth.DepthCapture) else self.nodes["depth"]}
        self.streams = {"Colour": Stream(color, render_color), "IR (flipped)": Stream(ir, render_ir),
                        "Depth": Stream(cap, render_depth)}
        self.color, self.ir, self.depth = self.streams.values()
        self.controls = {}  # label -> (node name, control id, info, scale); targets[node] is looked up at use
        self.auto = {"steps": 0, "running": False}
        self.switching = {"busy": False, "done": None}  # a range switch in progress / finished (its thread)
        self.rec = {"recorder": None, "thread": None}
        self.job = {"process": None, "dir": None}
        self.live = {"proc": None, "commands": None, "replies": None, "url": None, "open": False,
                     "inflight": 0, "sent": 0.0}
        self.scan_state = "idle"  # idle | scanning | saving | optimising
        self.imu, self.imu_note, self.imu_want = None, "", usbdepth.available()
        self.retry_at = 0.0  # time.monotonic() before which a lost scanner is not reconnected again
        self.lost = False  # a stream lost its device and the scanner is not back yet
        self.ranges, self.sliders = RANGES, SLIDERS  # for the Scanner panel
        self.range_names = {mode: label for label, mode in RANGES.items()}
        self.range_var = tk.StringVar(value=self.range_name())
        self.preset, self.keep_raw = tk.StringVar(value="small"), tk.BooleanVar(value=True)
        self.imu_wanted, self.find_markers = tk.BooleanVar(value=self.imu_want), tk.BooleanVar(value=True)
        self.panels, self.images, self.shown = [], {}, {}

    # --- window ---------------------------------------------------------------------------------
    def build_window(self):
        root = self.root
        root.title("triBUUTSCAN (Ferret Pro)")
        left, middle, right = tk.Frame(root), tk.Frame(root), tk.Frame(root)
        left.grid(row=0, column=0, sticky="n", padx=(8, 4), pady=8)
        middle.grid(row=0, column=1, sticky="n", padx=4, pady=8)
        right.grid(row=0, column=2, sticky="n", padx=(4, 8), pady=8)
        self.views = {}
        for name, parent, width in (("Colour", left, SIDE_W), ("IR (flipped)", left, SIDE_W), ("Depth", middle, DEPTH_W)):
            title = tk.Label(parent, text=depth_title(self.depth.cap) if name == "Depth" else name, font=panels.BOLD)
            title.pack(anchor="w")
            blank = tk.PhotoImage(width=width, height=width * fr.H // fr.W)
            self.images[name] = blank
            image = tk.Label(parent, image=blank, bg="black", bd=0)
            image.pack()
            if name == "Depth":
                bar = tk.Frame(parent)
                bar.pack(fill="x", pady=(4, 0))
                self.guide = tk.Label(bar, anchor="center", font=("TkDefaultFont", 14, "bold"))
                self.guide.pack(side="left", fill="x", expand=True)
                self.retry = tk.Button(bar, text="Retry now", font=panels.BOLD, command=self.retry_now)  # shown while lost
            stats = tk.Label(parent, anchor="w", font=("TkFixedFont", 9))
            stats.pack(fill="x", pady=(0, 6))
            self.views[name] = (title, image, stats)
        self.note = tk.Label(root, anchor="w", font=("TkDefaultFont", 11))
        self.note.grid(row=1, column=0, columnspan=3, sticky="we", padx=8, pady=(0, 8))
        for cls in PANELS:
            panel = cls(right, self)
            panel.pack(fill="x", pady=(0, 6))
            self.panels.append(panel)
        for key, action in (("s", self.save), ("r", self.toggle_record), ("m", self.mesh), ("c", self.toggle_scan)):
            root.bind(key, lambda _event, action=action: action())
        root.protocol("WM_DELETE_WINDOW", self.close)

    def start(self):
        threading.Thread(target=find_markers_loop, args=(self.color,), daemon=True).start()
        for s in self.streams.values():
            s.start()
        on_usb = isinstance(self.depth.cap, usbdepth.DepthCapture)
        self.log(f"viewer started (pid {os.getpid()}, depth {'14-bit over USB' if on_usb else '10-bit V4L2'}, "
                 f"{calib.CAMERA.mode}{', distances are estimates' if calib.CAMERA.law_estimate else ''})")
        if self.imu_want:
            self.imu_on()
        else:
            self.imu_note = "IMU needs udev/70-tribuutscan.rules (see README)"
        self.tick()

    def log(self, text):
        """Show a message in the window, print it, and append it to captures/viewer.log."""
        self.message = text
        line = f"[{time.strftime('%H:%M:%S')}] {text}"
        print(line, flush=True)
        try:
            LOG.parent.mkdir(exist_ok=True)
            with LOG.open("a") as f:
                f.write(time.strftime("%Y-%m-%d ") + line + "\n")
        except OSError:
            pass

    def tick(self):
        if not self.switching["busy"] and time.monotonic() > self.retry_at \
                and any(s.idle.is_set() for s in self.streams.values()):  # a stream lost its device
            self.switching["busy"], self.lost = True, True
            self.log("lost the scanner (unplugged, or it reset its USB connection): reconnecting ...")
            threading.Thread(target=self.reconnect, daemon=True).start()
        if self.switching["done"]:  # back in the Tk thread after a range switch or a reconnect
            self.switching.update(done=None, busy=False)
            self.views["Depth"][0].configure(text=depth_title(self.depth.cap))
            self.range_var.set(self.range_name())
            self.imu_wanted.set(self.imu is not None or (self.imu_want and not self.imu_note))
            for label, (node, cid, _, scale) in self.controls.items():  # the scanner may have reset them
                try:
                    set_control(self.targets[node], cid, int(scale.get()))
                except (OSError, AttributeError) as e:
                    self.log(f"{label}: {getattr(e, 'strerror', None) or e}")
        for name, s in self.streams.items():
            _, image, stats = self.views[name]
            if s.latest is not None and self.shown.get(name) != s.version:  # redraw only a changed preview
                self.shown[name] = s.version
                self.images[name] = tk.PhotoImage(data=s.latest, format="PPM")
                image.configure(image=self.images[name])
            stats.configure(text=f"{s.fps:4.1f} fps   {s.stats.get('text', '')}")
        if self.lost:
            wait = self.retry_at - time.monotonic()
            self.guide.configure(fg=BAD, text="Reconnecting to the scanner ..." if self.switching["busy"] or wait <= 0 else
                                 f"Scanner disconnected: plug it in or replug it. Next try in {wait:.0f} s")
        elif self.depth.stats:
            text, colour = distance_hint(self.depth.stats)
            self.guide.configure(text=text, fg=colour)
        retry_shown = self.retry.winfo_ismapped()
        if self.lost and not self.switching["busy"] and not retry_shown:
            self.retry.pack(side="right", padx=(6, 0))
        elif retry_shown and (not self.lost or self.switching["busy"]):
            self.retry.pack_forget()
        self.check_mesh()
        self.check_scan()
        recorder = self.rec["recorder"]
        if recorder:
            self.message = (f"REC {recorder.count} sets, {recorder.megabytes:.0f} MB"
                            + (f", {recorder.imu_rows} IMU samples" if recorder.imu_rows else ""))
        self.note.configure(text=self.message)
        for panel in self.panels:
            panel.refresh()
        self.root.after(33, self.tick)

    def close(self):
        self.log("viewer closed")
        if self.rec["recorder"]:
            self.stop_recording()
            self.rec["thread"].join(timeout=2)  # let the last set and the files close cleanly
        if self.live["proc"] is not None:
            if self.scan_state == "scanning":
                self.live["commands"].put(("stop",))  # save what was scanned
                self.log("scan stopped by closing, saving ...")
            self.live["commands"].put(("quit",))
            self.live["proc"].join(timeout=15)
            if self.live["proc"].is_alive():
                self.live["proc"].terminate()
        self.imu_off()  # while the depth capture still carries the vendor commands
        for s in self.streams.values():
            try:
                s.stop()
            except OSError:  # the scanner is already gone
                pass
        self.root.destroy()

    # --- scanner settings ------------------------------------------------------------------------
    def slider(self, parent, label, node, cid):
        """A Scale for one control (see SLIDERS), or None if the node does not have it."""
        info = next((c for c in node_controls(self.targets[node]).values() if c["id"] == cid), None)
        if info is None:
            return None

        def set_value(value):
            try:
                set_control(self.targets[node], cid, int(float(value)))
            except (OSError, AttributeError) as e:
                self.log(f"{label}: {getattr(e, 'strerror', None) or e}")

        scale = tk.Scale(parent, label=label, orient="horizontal", from_=info["min"], to=info["max"],
                         resolution=info["step"] or 1, command=set_value)
        scale.set(info["value"])
        scale.pack(fill="x")
        self.controls[label] = (node, cid, info, scale)
        return scale

    def reset_controls(self):
        for _, _, info, scale in self.controls.values():
            scale.set(info["default"])  # triggers set_value

    def control_values(self):
        return {label: get_control(self.targets[node], cid) for label, (node, cid, _, _) in self.controls.items()}

    def start_auto(self):
        if self.auto["running"]:  # a second click would start a second chain of steps beside the first
            return
        if "IR exposure" in self.controls and "IR gain" in self.controls and not self.switching["busy"]:
            self.auto["steps"], self.auto["running"] = 0, True
            self.auto_step()

    def auto_step(self):
        """One Auto IR step: darken while clipping, brighten while dim, else stop."""
        exposure, gain = self.controls["IR exposure"], self.controls["IR gain"]
        e, g = exposure[3].get(), gain[3].get()
        clipped, mean = self.ir.stats.get("clipped"), self.ir.stats.get("mean")
        self.auto["steps"] += 1
        if clipped is None or self.auto["steps"] > AUTO_MAX_STEPS:
            self.log("Auto IR: gave up")
            self.auto["running"] = False
            return
        if clipped > CLIP_HI and g > gain[2]["min"]:
            gain[3].set(max(gain[2]["min"], g * 0.7))
        elif clipped > CLIP_HI and e > exposure[2]["min"]:
            exposure[3].set(max(exposure[2]["min"], e * 0.7))
        elif clipped < CLIP_LO and mean < MEAN_LO and e < exposure[2]["max"]:
            exposure[3].set(min(exposure[2]["max"], e * 1.4))
        elif clipped < CLIP_LO and mean < MEAN_LO and g < GAIN_CAP:
            gain[3].set(min(GAIN_CAP, g * 1.3))
        else:
            self.log(f"Auto IR: done (exposure {e}, gain {g})")
            self.auto["running"] = False
            return
        self.message = f"Auto IR: step {self.auto['steps']}"
        self.root.after(AUTO_STEP_MS, self.auto_step)

    def range_name(self):
        return self.range_names.get(calib.CAMERA.mode, calib.CAMERA.mode)

    def choose_range(self, label):
        mode = RANGES[label]
        busy = self.rec["recorder"] or self.scan_state != "idle" or self.auto["running"] or self.switching["busy"]
        if mode == calib.CAMERA.mode or busy:
            if busy:
                self.log("stop recording, scanning and Auto IR before switching the range")
            self.range_var.set(self.range_name())
            return
        self.switching["busy"] = True
        self.log(f"switching to {label}: the streams pause for about 10 s")
        threading.Thread(target=self.switch_range, args=(mode,), daemon=True).start()

    def switch_range(self, mode):
        """Background thread: pause the streams, switch the scanner's depth mode, start them again."""
        try:
            self.imu_off()  # its commands go through the depth capture, which is about to close
            for s in self.streams.values():
                s.pause()
            time.sleep(SWITCH_SETTLE_S)
            for attempt in range(2):
                try:
                    found = wait_for_scanner()
                    if vendor.camera_users():
                        raise RuntimeError("another program has the scanner open")
                    ch = vendor.Channel(found["depth"])
                    try:
                        ch.switch_depth_mode(mode)
                    finally:
                        ch.close()
                    break
                except OSError:  # it re-enumerated under us: once more
                    if attempt:
                        raise
                    time.sleep(3)
            self.log(f"depth range switched to {mode}")
        except Exception as e:  # noqa: BLE001 - shown to the user
            self.log(f"range switch failed: {e}")
        finally:
            self.reopen()
            if self.imu_want:
                self.imu_on()
            self.switching["done"] = True

    def reconnect(self):
        """Background thread, after a stream lost its device (the scanner reset itself): stop what
        depended on the old connection, then open everything again; the depth range is read anew
        (a reset puts the scanner back in close range)."""
        try:
            if self.rec["recorder"]:
                self.stop_recording()
                self.log("recording stopped: the scanner reset itself")
            if self.scan_state == "scanning":
                self.toggle_scan()  # stops and saves; frames after a reset may be in another range
                self.log("scan stopped and saved: the scanner reset itself")
            self.imu_off(quiet=True)
            for s in self.streams.values():
                s.pause()
            if self.reopen():
                self.lost = False
                self.log(f"scanner back ({calib.CAMERA.mode})")
                if self.imu_want:
                    self.imu_on()
            else:
                self.retry_at = time.monotonic() + RETRY_S
        finally:
            self.switching["done"] = True

    def retry_now(self):
        """The Retry now button: reconnect at the next tick instead of waiting for the timer."""
        self.retry_at = 0.0
        self.log("retrying now ...")

    def reopen(self):
        """Open all three streams again (node paths can change after the scanner re-enumerates).
        Returns False (and says so) if they could not be opened."""
        try:
            self.nodes.update(wait_for_scanner())
            ir_exposure_by_hand(self.nodes, self.log)  # a USB reset puts it back to auto
            self.color.resume(v4l2.Capture(self.nodes["color"], "YUYV", fr.W, fr.H))
            self.ir.resume(v4l2.Capture(self.nodes["ir"], "GREY", fr.W, fr.H))
            cap = fr.open_depth(self.nodes, self.log)
            self.targets.update(self.nodes, depth=cap if isinstance(cap, usbdepth.DepthCapture) else self.nodes["depth"])
            self.depth.resume(cap)
            return True
        except Exception as e:  # noqa: BLE001 - shown to the user; the viewer stays up
            self.log(f"could not open the scanner ({e}): is it plugged in? trying again in {RETRY_S:.0f} s")
            return False

    # --- IMU -------------------------------------------------------------------------------------
    def vendor_channel(self):
        """The command channel: through the USB depth capture while it owns the depth function."""
        cap = self.depth.cap
        return vendor.Channel(usb=cap) if isinstance(cap, usbdepth.DepthCapture) else vendor.Channel(self.nodes["depth"])

    def imu_on(self):
        if self.imu is not None:
            return
        try:
            ch = self.vendor_channel()
            try:
                self.imu = imu.switch_on(ch)
            finally:
                ch.close()
        except (OSError, RuntimeError, vendor.VendorError) as e:
            self.imu, self.imu_note = None, f"IMU not available: {e}"
            self.log(self.imu_note)
            return
        self.imu_note = ""
        recorder = self.rec["recorder"]
        if recorder:
            self.imu.listener = recorder.write_imu
        s = self.imu.settings
        self.log(f"IMU on: gyro {s['gyro_hz']} Hz +-{s['gyro_dps']} deg/s, accel {s['accel_hz']} Hz +-{s['accel_g']} g")

    def imu_off(self, quiet=False):
        stream, self.imu = self.imu, None
        if stream is None:
            return
        stream.listener = None
        stream.stop()  # reading first: interface 6 goes back to usbhid
        try:
            ch = self.vendor_channel()
            try:
                imu.switch_off(ch)
            finally:
                ch.close()
        except (OSError, RuntimeError, vendor.VendorError) as e:
            if not quiet:  # after a reset the IMU is off anyway
                self.log(f"IMU not switched off ({e}); it stops at the next power-up")

    def toggle_imu(self):
        self.imu_want = self.imu_wanted.get()
        if self.switching["busy"]:
            return  # the switch brings the IMU back as wanted
        if self.imu_want:
            self.imu_on()
            self.imu_wanted.set(self.imu is not None)
        else:
            self.imu_off()
            self.imu_note = ""
            self.log("IMU off")

    def toggle_markers(self):
        found_markers.update(on=self.find_markers.get())

    # --- capture -----------------------------------------------------------------------------------
    def save(self):
        if self.switching["busy"]:
            self.log("switching the range: save when the streams are back")
            return
        color, ir, depth = self.color, self.ir, self.depth
        if not (color.raw and ir.raw and depth.raw):
            self.log("nothing to save yet")
            return
        frames = fr.decode(color.raw, ir.raw, depth.raw, depth.cap.bytesperline)
        hits = markers.detect(markers.luma(frames["color"]), rgb=frames["color"])
        out = fr.save(frames, extra={"controls": self.control_values(),
                                     "markers": [[round(x, 2), round(y, 2), round(r, 2)] for x, y, r, _ in hits]})
        xyz, rgb = pointcloud.to_points(frames["depth"], pointcloud.register_color(frames["color"]))
        pointcloud.write_ply(out / "cloud.ply", xyz, rgb)
        self.log(f"saved {out.name} (with cloud.ply, {len(hits)} markers)")

    def record_loop(self, recorder):
        last = 0.0
        color, ir, depth = self.color, self.ir, self.depth
        while self.rec["recorder"] is recorder:
            t = time.time()
            if depth.stamp != last and color.raw and ir.raw:
                d_raw, last = depth.raw, depth.stamp
                (c_raw, c_t), (i_raw, i_t) = color.nearest(last), ir.nearest(last)
                recorder.write({"color": c_raw, "ir": i_raw, "depth": d_raw}, {"color": c_t, "ir": i_t, "depth": last})
            time.sleep(max(0.0, 1 / RECORD_FPS - (time.time() - t)))
        recorder.close()

    def stop_recording(self):
        recorder, self.rec["recorder"] = self.rec["recorder"], None
        if self.imu is not None:
            self.imu.listener = None
        self.log(f"recorded {recorder.count} sets"
                 + (f" and {recorder.imu_rows} IMU samples" if recorder.imu_rows else "") + f" to {recorder.dir.name}")

    def toggle_record(self):
        if self.rec["recorder"]:
            self.stop_recording()
            return
        if not self.depth.raw or self.switching["busy"]:
            self.log("nothing to record yet")
            return
        extra = {"controls": self.control_values(), "fps": RECORD_FPS}
        if self.imu is not None:
            extra["imu"] = self.imu.settings
        recorder = fr.Recorder(self.depth.cap.bytesperline, extra=extra)
        self.rec["recorder"] = recorder
        if self.imu is not None:
            self.imu.listener = recorder.write_imu
        self.rec["thread"] = threading.Thread(target=self.record_loop, args=(recorder,), daemon=True)
        self.rec["thread"].start()
        self.log(f"recording started: {recorder.dir.name}" + (" (with IMU)" if self.imu is not None else ""))

    def mesh(self):
        """Fuse the latest recording in a separate process so the live view keeps running."""
        job = self.job
        if job["process"]:
            self.log(f"still meshing {job['dir'].name}")
            return
        if self.rec["recorder"]:
            self.log("stop recording first")
            return
        directory = latest_recording()
        if directory is None:
            self.log("no recording yet: press r to record")
            return
        if not VENV_PYTHON.exists():
            self.log("meshing needs Open3D in .venv (see fuse.py)")
            return
        job["process"] = subprocess.Popen(
            [str(VENV_PYTHON), "-m", "tribuutscan.fuse", str(directory)], cwd=REPO,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        job["dir"] = directory
        self.log(f"meshing {directory.name} (takes about a minute) ...")

    def check_mesh(self):
        job = self.job
        process = job["process"]
        if process is None or process.poll() is None:
            return
        lines = [ln for ln in process.stdout.read().splitlines() if ln.startswith(("mesh:", "Traceback"))]
        if process.returncode == 0:
            summary = lines[-1].split(" -> ")[0] if lines else "done"
            self.log(f"meshed {job['dir'].name} ({summary}) -> mesh.ply")
        else:
            self.log(f"meshing {job['dir'].name} failed (exit {process.returncode})")
        job["process"] = None

    # --- live scan ---------------------------------------------------------------------------------
    def start_worker(self):
        live = self.live
        ctx = multiprocessing.get_context("spawn")
        if VENV_PYTHON.exists():
            ctx.set_executable(str(VENV_PYTHON))  # has Open3D: tracking
        live["commands"], live["replies"] = ctx.Queue(), ctx.Queue()
        live["proc"] = ctx.Process(target=livescan.run, args=(live["commands"], live["replies"]), daemon=True)
        live["proc"].start()

    def toggle_scan(self, continue_from=None):
        live = self.live
        if self.scan_state == "scanning":
            self.scan_state = "saving"
            live["commands"].put(("stop",))
            self.log("scan stopped, saving ...")
            return
        if self.scan_state != "idle" or self.switching["busy"]:
            return
        if live["proc"] is None:
            self.start_worker()
        out = fr.CAPTURES / (time.strftime("%Y%m%d-%H%M%S") + "-scan")
        self.scan_state, live["inflight"] = "scanning", 0
        live["commands"].put(("start", self.preset.get(), out, self.keep_raw.get(), continue_from, calib.CAMERA.mode))
        self.log(f"scan started: {out.name} ({self.preset.get()}, point distance {cloud.PRESETS_MM[self.preset.get()]} mm)"
                 + (f", continuing {continue_from.name}" if continue_from else ""))

    def continue_scan(self):
        """Go on with the newest saved scan (saved by this version: it has merged.npz)."""
        if self.scan_state != "idle":
            return
        saved = sorted(d for d in fr.CAPTURES.glob("*-scan") if (d / "merged.npz").exists())
        if not saved:
            self.log("nothing to continue: no saved scan with its frames yet")
            return
        self.toggle_scan(continue_from=saved[-1])

    def open_view(self):
        live = self.live
        if live["url"]:
            webbrowser.open(live["url"])
            return
        if live["proc"] is None:
            self.start_worker()
        live["open"] = True  # opens when the worker reports ready
        self.log("starting the 3D view ...")

    def check_scan(self):
        """Hand the worker a new frame when it is free; handle its replies."""
        live, color, ir, depth = self.live, self.color, self.ir, self.depth
        if self.scan_state == "scanning" and live["inflight"] < 2 and color.raw and ir.raw and depth.raw \
                and depth.stamp != live["sent"]:  # one frame in work, one waiting: the worker never idles
            d_raw, d_t = depth.raw, depth.stamp
            live["sent"] = d_t
            live["inflight"] += 1
            (c_raw, c_t), (i_raw, i_t) = color.nearest(d_t), ir.nearest(d_t)
            live["commands"].put(("frame", c_raw, i_raw, d_raw, depth.cap.bytesperline, (c_t, i_t, d_t)))
        while live["replies"] is not None:
            try:
                kind, *rest = live["replies"].get_nowait()
            except queue.Empty:
                break
            if kind == "ready":
                live["url"] = rest[0]
                if live["open"] and rest[0]:
                    webbrowser.open(rest[0])
            elif kind == "log":
                self.log(rest[0])
            elif kind == "frame_done":
                live["inflight"] = max(0, live["inflight"] - 1)
                if self.scan_state == "scanning":
                    merged, frames, points, tracking, cue = rest
                    held = ", ".join(([f"{cue['markers']} markers"] if cue.get("markers") else [])
                                     + (["features"] if cue.get("features") else [])) or "shape"
                    hint = ("   TRACKING LOST: move back slowly to where it was" if not tracking else
                            "   FLAT AREA: aim at an edge, a corner or markers" if cue.get("flat") else "")
                    self.message = f"SCAN {merged} of {frames} frames merged, {points:,} points | held by {held}{hint}"
            elif kind == "saved":
                path, points = rest
                self.log(f"scan saved: {points:,} points -> {Path(path).parent.name}/cloud.ply; optimising ...")
                self.scan_state = "optimising"
            elif kind == "optimised":
                self.log(rest[0])
                self.scan_state = "idle"


def refuse(text):
    print(text, flush=True)
    root = tk.Tk()
    root.withdraw()
    tkinter.messagebox.showerror("triBUUTSCAN", text)
    root.destroy()


def scanner_problem(error):
    """What to tell the user when the scanner can't be opened at start."""
    if usbdepth.find() is None:
        return "The scanner is not connected.\n\nPlug it in (or replug it), wait a few seconds, then press Retry."
    return (f"The scanner is connected but could not be opened:\n{error}\n\nClose other programs that use it "
            "(CrealityScan, a measure or calibrate run), or replug it, wait a few seconds, then press Retry.")


def open_app(root, ask=tkinter.messagebox.askretrycancel):
    """The App, or None if the user gives up: while the scanner can't be opened, say why and offer Retry."""
    while True:
        try:
            return App(root)
        except (RuntimeError, OSError) as e:
            print(f"scanner not available: {e}", flush=True)
            if not ask("triBUUTSCAN", scanner_problem(e), parent=root):
                return None


def main():
    LOCK.parent.mkdir(exist_ok=True)
    lock = open(LOCK, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # released by the system when this process ends
    except BlockingIOError:
        refuse("The viewer is already running: use that window, or close it first.")
        return
    root = tk.Tk()
    root.withdraw()  # only the Retry question until the scanner opens
    app = open_app(root)
    if app is None:
        root.destroy()
        return
    root.deiconify()
    app.build_window()
    app.start()
    root.mainloop()


if __name__ == "__main__":
    main()
