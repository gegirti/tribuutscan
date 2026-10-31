"""Live viewer: colour, IR and depth side by side. Run with `python -m tribuutscan.viewer`.

Press `s` or click Save to write the current frame set to captures/<timestamp>/.
Press `r` or click Record to start/stop recording raw frame sets to captures/<timestamp>-rec/.
Sliders set IR exposure/gain and colour brightness live. IR exposure and gain are controls on the
depth function: they change the IR sensor that depth is computed from. Depth is 14-bit over USB
(usbdepth.py, which then also owns those two controls) when allowed, else 10-bit V4L2.
Range switches the scanner between close range (15-33 cm) and long range (~20-72 cm, an estimate until
calibrated); the streams pause for about 10 s while it switches (vendor.py). Auto IR steps them until the IR
image has little clipping (clipping costs depth; see CLAUDE.md).
The distance line reads the centre of the depth image; distances and the window are estimates.
Press `m` or click Mesh to fuse the latest recording into mesh.ply (runs fuse.py in the Open3D .venv).
Events are printed to the console and appended to captures/viewer.log.
Press `c` or click Start scan to merge frames live into one compact cloud (cloud.py) in a worker
process; Open 3D view shows it growing in the browser. Stop saves captures/<timestamp>-scan/.
Markers (white dot in a black ring) are found in the colour image a few times a second and ringed
in the colour view; saved frames list them in info.json.
"""
import collections
import multiprocessing
import os
import queue
import subprocess
import threading
import time
import tkinter as tk
import webbrowser
from pathlib import Path

import numpy as np

from . import frames as fr
from . import calib, cloud, livescan, markers, pointcloud, usbdepth, v4l2, vendor
from .depth import decode_depth
from .image import depth_to_rgb, ppm, yuyv_pixels, yuyv_to_rgb

PANE_W = 480
PREVIEW_FPS = 15  # previews redraw at most this often; every frame is still captured and paired

# (label, node, control id, column): what the sliders drive and which image they sit under.
SLIDERS = [
    ("Colour brightness", "color", v4l2.CID_BRIGHTNESS, 0),
    ("IR exposure", "depth", v4l2.CID_EXPOSURE, 1),
    ("IR gain", "depth", v4l2.CID_GAIN, 1),
]

# Auto IR: keep clipped pixels between CLIP_LO and CLIP_HI percent; brighten only if the mean is low.
CLIP_LO, CLIP_HI, MEAN_LO = 0.3, 2.0, 35
GAIN_CAP = 2000  # auto never raises gain past this (gain clipped badly in the 25 cm test)
AUTO_STEP_MS, AUTO_MAX_STEPS = 600, 16

RECORD_FPS = 10  # about 43 MB/s of raw data

GOOD, EDGE, BAD = "#2e7d32", "#ef6c00", "#c62828"

REPO = Path(__file__).resolve().parent.parent
VENV_PYTHON = REPO / ".venv" / "bin" / "python"  # has Open3D; see fuse.py


LOG = fr.CAPTURES / "viewer.log"


def latest_recording():
    recordings = sorted(fr.CAPTURES.glob("*-rec"))
    return recordings[-1] if recordings else None


def fit(rgb):
    step = max(1, rgb.shape[1] // PANE_W)
    return rgb[::step, ::step]


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


def render_color(cap, buf):
    step = max(1, cap.width // PANE_W)  # convert only the preview's pixels (4x less work than all)
    v, u = np.mgrid[0:cap.height:step, 0:cap.width:step]
    rgb = yuyv_pixels(buf, cap.width, u.ravel(), v.ravel()).reshape(v.shape + (3,))
    hits = found_markers["list"]
    if hits:
        rgb = markers.draw(rgb, [(x / step, y / step, r / step, c) for x, y, r, c in hits])
    return ppm(rgb), {"text": f"markers {len(hits)}" if found_markers["on"] else ""}


def render_ir(cap, buf):
    g = np.frombuffer(buf, np.uint8, cap.width * cap.height).reshape(cap.height, cap.width)
    g = fit(g[:, ::-1])  # IR is mirrored relative to colour/depth
    mean, clipped = float(g.mean()), 100 * float((g >= 250).mean())
    stats = {"mean": mean, "clipped": clipped, "text": f"mean {mean:.0f}  clipped {clipped:.1f}%"}
    return ppm(np.repeat(g[..., None], 3, axis=2)), stats


def render_depth(cap, buf):
    d = decode_depth(buf, cap.width, cap.height, cap.bytesperline)
    h, w = d.shape
    centre = d[h * 3 // 8:h * 5 // 8, w * 3 // 8:w * 5 // 8]
    valid = centre[centre > 0]
    stats = {
        "text": f"valid {100 * (d > 0).mean():.0f}%",
        "centre_valid": valid.size / centre.size,
        "centre_cm": float(pointcloud.depth_to_mm(np.median(valid))) / 10 if valid.size else None,
    }
    return ppm(depth_to_rgb(fit(d))), stats  # colour-map only the preview's pixels


def distance_hint(stats):
    """(text, colour) for the guide line, from the depth centre stats (the active depth mode's window)."""
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
    return f"Depth, {kind}, {'long' if cap.depth_mode.startswith('Long') else 'close'} range (raw, near=red)"


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


def main():
    nodes = v4l2.discover()
    ir_exposure_by_hand(nodes, print)
    depth_cap = fr.open_depth(nodes)
    on_usb = isinstance(depth_cap, usbdepth.DepthCapture)
    targets = {**nodes, "depth": depth_cap if on_usb else nodes["depth"]}  # where each slider's control lives
    streams = {
        "Colour": Stream(v4l2.Capture(nodes["color"], "YUYV", fr.W, fr.H), render_color),
        "IR (flipped)": Stream(v4l2.Capture(nodes["ir"], "GREY", fr.W, fr.H), render_ir),
        "Depth": Stream(depth_cap, render_depth),
    }
    color, ir, depth = streams.values()

    root = tk.Tk()
    root.title("triBUUTSCAN (Ferret Pro)")
    labels, titles = {}, {}
    for col, name in enumerate(streams):
        titles[name] = tk.Label(root, text=depth_title(depth_cap) if name == "Depth" else name)
        titles[name].grid(row=0, column=col)
        labels[name] = tk.Label(root)
        labels[name].grid(row=1, column=col)
    guide = tk.Label(root, anchor="center", font=("TkDefaultFont", 14, "bold"))
    guide.grid(row=2, column=0, columnspan=3, sticky="we", padx=6)
    status = tk.Label(root, anchor="w")
    status.grid(row=3, column=0, columnspan=2, sticky="we")
    note = tk.Label(root, anchor="center", font=("TkDefaultFont", 12))  # messages: save, record, mesh
    note.grid(row=4, column=0, columnspan=3, sticky="we", padx=6)
    message = {"text": ""}

    def log(text):
        """Show a message in the window, print it, and append it to captures/viewer.log."""
        message["text"] = text
        line = f"[{time.strftime('%H:%M:%S')}] {text}"
        print(line, flush=True)
        try:
            LOG.parent.mkdir(exist_ok=True)
            with LOG.open("a") as f:
                f.write(time.strftime("%Y-%m-%d ") + line + "\n")
        except OSError:
            pass

    columns = [tk.Frame(root) for _ in streams]
    for col, frame in enumerate(columns):
        frame.grid(row=5, column=col, sticky="n")
    controls = {}  # label -> (node name, control id, info, scale); targets[node] is looked up at use
    for label, node, cid, col in SLIDERS:
        info = next((c for c in node_controls(targets[node]).values() if c["id"] == cid), None)
        if info is None:
            continue

        def set_value(value, node=node, cid=cid, label=label):
            try:
                set_control(targets[node], cid, int(float(value)))
            except (OSError, AttributeError) as e:
                log(f"{label}: {getattr(e, 'strerror', None) or e}")

        scale = tk.Scale(columns[col], label=label, orient="horizontal", length=PANE_W - 40, from_=info["min"],
                         to=info["max"], resolution=info["step"] or 1, command=set_value)
        scale.set(info["value"])
        scale.pack(padx=10)
        controls[label] = (node, cid, info, scale)

    def reset():
        for _, _, info, scale in controls.values():
            scale.set(info["default"])  # triggers set_value

    def control_values():
        return {label: get_control(targets[node], cid) for label, (node, cid, _, _) in controls.items()}

    buttons = tk.Frame(columns[2])
    buttons.pack(pady=10)
    tk.Button(buttons, text="Reset defaults", command=reset).pack(fill="x")

    auto = {"steps": 0, "running": False}
    switching = {"busy": False, "done": None}  # a range switch in progress / finished (set by its thread)

    def auto_step():
        """One Auto IR step: darken while clipping, brighten while dim, else stop."""
        exposure, gain = controls["IR exposure"], controls["IR gain"]
        e, g = exposure[3].get(), gain[3].get()
        clipped, mean = ir.stats.get("clipped"), ir.stats.get("mean")
        auto["steps"] += 1
        if clipped is None or auto["steps"] > AUTO_MAX_STEPS:
            log("Auto IR: gave up")
            auto["running"] = False
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
            log(f"Auto IR: done (exposure {e}, gain {g})")
            auto["running"] = False
            return
        message["text"] = f"Auto IR: step {auto['steps']}"
        root.after(AUTO_STEP_MS, auto_step)

    def start_auto():
        if "IR exposure" in controls and "IR gain" in controls and not switching["busy"]:
            auto["steps"], auto["running"] = 0, True
            auto_step()

    tk.Button(buttons, text="Auto IR", command=start_auto).pack(fill="x")

    def save(_event=None):
        if switching["busy"]:
            log("switching the range: save when the streams are back")
            return
        if not (color.raw and ir.raw and depth.raw):
            log("nothing to save yet")
            return
        frames = fr.decode(color.raw, ir.raw, depth.raw, depth.cap.bytesperline)
        hits = markers.detect(markers.luma(frames["color"]), rgb=frames["color"])
        out = fr.save(frames, extra={"controls": control_values(),
                                     "markers": [[round(x, 2), round(y, 2), round(r, 2)] for x, y, r, _ in hits]})
        xyz, rgb = pointcloud.to_points(frames["depth"], pointcloud.register_color(frames["color"]))
        pointcloud.write_ply(out / "cloud.ply", xyz, rgb)
        log(f"saved {out.name} (with cloud.ply, {len(hits)} markers)")

    rec = {"recorder": None}

    def record_loop(recorder):
        last = 0.0
        while rec["recorder"] is recorder:
            t = time.time()
            if depth.stamp != last and color.raw and ir.raw:
                d_raw, last = depth.raw, depth.stamp
                (c_raw, c_t), (i_raw, i_t) = color.nearest(last), ir.nearest(last)
                recorder.write({"color": c_raw, "ir": i_raw, "depth": d_raw}, {"color": c_t, "ir": i_t, "depth": last})
            time.sleep(max(0.0, 1 / RECORD_FPS - (time.time() - t)))
        recorder.close()

    def toggle_record(_event=None):
        recorder = rec["recorder"]
        if recorder:
            rec["recorder"] = None
            record_button.configure(text="Record (r)")
            log(f"recorded {recorder.count} sets to {recorder.dir.name}")
            return
        if not depth.raw:
            log("nothing to record yet")
            return
        recorder = fr.Recorder(depth.cap.bytesperline, extra={"controls": control_values(), "fps": RECORD_FPS})
        rec["recorder"] = recorder
        rec["thread"] = threading.Thread(target=record_loop, args=(recorder,), daemon=True)
        rec["thread"].start()
        log(f"recording started: {recorder.dir.name}")
        record_button.configure(text="Stop recording (r)")

    job = {"process": None, "dir": None}

    def mesh(_event=None):
        """Fuse the latest recording in a separate process so the live view keeps running."""
        if job["process"]:
            log(f"still meshing {job['dir'].name}")
            return
        if rec["recorder"]:
            log("stop recording first")
            return
        directory = latest_recording()
        if directory is None:
            log("no recording yet: press r to record")
            return
        if not VENV_PYTHON.exists():
            log("meshing needs Open3D in .venv (see fuse.py)")
            return
        job["process"] = subprocess.Popen(
            [str(VENV_PYTHON), "-m", "tribuutscan.fuse", str(directory)], cwd=REPO,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        job["dir"] = directory
        mesh_button.configure(text="Meshing...", state="disabled")
        log(f"meshing {directory.name} (takes about a minute) ...")

    def check_mesh():
        process = job["process"]
        if process is None or process.poll() is None:
            return
        lines = [ln for ln in process.stdout.read().splitlines() if ln.startswith(("mesh:", "Traceback"))]
        if process.returncode == 0:
            summary = lines[-1].split(" -> ")[0] if lines else "done"
            log(f"meshed {job['dir'].name} ({summary}) -> mesh.ply")
        else:
            log(f"meshing {job['dir'].name} failed (exit {process.returncode})")
        job["process"] = None
        mesh_button.configure(text="Mesh last recording (m)", state="normal")

    # Actions: buttons for now, a menu later.
    actions = tk.Frame(root)
    actions.grid(row=3, column=2, sticky="e")
    tk.Button(actions, text="Save frame (s)", command=save).pack(side="left")
    record_button = tk.Button(actions, text="Record (r)", command=toggle_record)
    record_button.pack(side="left")
    mesh_button = tk.Button(actions, text="Mesh last recording (m)", command=mesh)
    mesh_button.pack(side="left")
    root.bind("s", save)
    root.bind("r", toggle_record)
    root.bind("m", mesh)

    live = {"proc": None, "commands": None, "replies": None, "url": None, "open": False,
            "scanning": False, "inflight": 0, "sent": 0.0}

    def start_worker():
        ctx = multiprocessing.get_context("spawn")
        if VENV_PYTHON.exists():
            ctx.set_executable(str(VENV_PYTHON))  # has Open3D: tracking
        live["commands"], live["replies"] = ctx.Queue(), ctx.Queue()
        live["proc"] = ctx.Process(target=livescan.run, args=(live["commands"], live["replies"]), daemon=True)
        live["proc"].start()

    def toggle_scan(_event=None, continue_from=None):
        if live["scanning"]:
            live["scanning"] = False
            live["commands"].put(("stop",))
            scan_button.configure(text="Saving...", state="disabled")
            log("scan stopped, saving ...")
            return
        if live["proc"] is None:
            start_worker()
        out = fr.CAPTURES / (time.strftime("%Y%m%d-%H%M%S") + "-scan")
        live.update(scanning=True, inflight=0)
        live["commands"].put(("start", preset.get(), out, keep_raw.get(), continue_from, calib.CAMERA.mode))
        scan_button.configure(text="Stop scan (c)")
        log(f"scan started: {out.name} ({preset.get()}, point distance {cloud.PRESETS_MM[preset.get()]} mm)"
            + (f", continuing {continue_from.name}" if continue_from else ""))

    def continue_scan():
        """Go on with the newest saved scan (saved by this version: it has merged.npz)."""
        if live["scanning"]:
            return
        saved = sorted(d for d in fr.CAPTURES.glob("*-scan") if (d / "merged.npz").exists())
        if not saved:
            log("nothing to continue: no saved scan with its frames yet")
            return
        toggle_scan(continue_from=saved[-1])

    def open_view():
        if live["url"]:
            webbrowser.open(live["url"])
            return
        if live["proc"] is None:
            start_worker()
        live["open"] = True  # opens when the worker reports ready
        log("starting the 3D view ...")

    def check_scan():
        """Hand the worker a new frame when it is free; handle its replies."""
        if live["scanning"] and live["inflight"] < 2 and color.raw and ir.raw and depth.raw \
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
                log(rest[0])
            elif kind == "frame_done":
                live["inflight"] = max(0, live["inflight"] - 1)
                if live["scanning"]:
                    merged, frames, points, tracking, cue = rest
                    held = ", ".join(([f"{cue['markers']} markers"] if cue.get("markers") else [])
                                     + (["features"] if cue.get("features") else [])) or "shape"
                    hint = ("   TRACKING LOST: move back slowly to where it was" if not tracking else
                            "   FLAT AREA: aim at an edge, a corner or markers" if cue.get("flat") else "")
                    message["text"] = f"SCAN {merged} of {frames} frames merged, {points:,} points | held by {held}{hint}"
            elif kind == "saved":
                path, points = rest
                log(f"scan saved: {points:,} points -> {Path(path).parent.name}/cloud.ply; optimising ...")
                scan_button.configure(text="Optimising...", state="disabled")
            elif kind == "optimised":
                log(rest[0])
                scan_button.configure(text="Start scan (c)", state="normal")

    scan_box = tk.Frame(columns[2])
    scan_box.pack(pady=4)
    preset = tk.StringVar(value="small")
    tk.OptionMenu(scan_box, preset, *cloud.PRESETS_MM).pack(side="left")
    scan_button = tk.Button(scan_box, text="Start scan (c)", command=toggle_scan)
    scan_button.pack(side="left")
    tk.Button(scan_box, text="Continue last", command=continue_scan).pack(side="left")
    tk.Button(scan_box, text="Open 3D view", command=open_view).pack(side="left")
    keep_raw = tk.BooleanVar(value=True)  # debugging: replay a live scan offline with cloud.py
    tk.Checkbutton(scan_box, text="keep raw frames", variable=keep_raw).pack(side="left")
    root.bind("c", toggle_scan)

    range_names = {mode: label for label, mode in RANGES.items()}
    range_var = tk.StringVar(value=range_names.get(calib.CAMERA.mode, calib.CAMERA.mode))

    def reopen():
        """Open all three streams again (node paths can change after the scanner re-enumerates)."""
        try:
            nodes.update(wait_for_scanner())
            ir_exposure_by_hand(nodes, log)  # a USB reset puts it back to auto
            color.resume(v4l2.Capture(nodes["color"], "YUYV", fr.W, fr.H))
            ir.resume(v4l2.Capture(nodes["ir"], "GREY", fr.W, fr.H))
            cap = fr.open_depth(nodes, log)
            targets.update(nodes, depth=cap if isinstance(cap, usbdepth.DepthCapture) else nodes["depth"])
            depth.resume(cap)
        except Exception as e:  # noqa: BLE001 - shown to the user; the viewer stays up
            log(f"could not restart the streams ({e}): close the viewer and start it again")

    def switch_range(mode):
        """Background thread: pause the streams, switch the scanner's depth mode, start them again."""
        try:
            for s in streams.values():
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
            log(f"depth range switched to {mode}")
        except Exception as e:  # noqa: BLE001 - shown to the user
            log(f"range switch failed: {e}")
        finally:
            reopen()
            switching["done"] = True

    def choose_range(label):
        mode = RANGES[label]
        busy = rec["recorder"] or live["scanning"] or auto["running"] or switching["busy"]
        if mode == calib.CAMERA.mode or busy:
            if busy:
                log("stop recording, scanning and Auto IR before switching the range")
            range_var.set(range_names.get(calib.CAMERA.mode, calib.CAMERA.mode))
            return
        switching["busy"] = True
        log(f"switching to {label}: the streams pause for about 10 s")
        threading.Thread(target=switch_range, args=(mode,), daemon=True).start()

    range_box = tk.Frame(columns[2])
    range_box.pack(pady=4)
    tk.Label(range_box, text="Range").pack(side="left")
    tk.OptionMenu(range_box, range_var, *RANGES, command=choose_range).pack(side="left")

    show = tk.BooleanVar(value=True)
    tk.Checkbutton(columns[0], text="Find markers", variable=show,
                   command=lambda: found_markers.update(on=show.get())).pack()
    threading.Thread(target=find_markers_loop, args=(color,), daemon=True).start()

    for s in streams.values():
        s.start()
    images = {}

    shown = {}

    def tick():
        if switching["done"]:  # back in the Tk thread after a range switch
            switching.update(done=None, busy=False)
            titles["Depth"].configure(text=depth_title(depth.cap))
            range_var.set(range_names.get(calib.CAMERA.mode, calib.CAMERA.mode))
            for label, (node, cid, _, scale) in controls.items():  # the scanner may have reset them
                try:
                    set_control(targets[node], cid, int(scale.get()))
                except (OSError, AttributeError) as e:
                    log(f"{label}: {getattr(e, 'strerror', None) or e}")
        for name, s in streams.items():
            if s.latest is not None and shown.get(name) != s.version:  # redraw only a changed preview
                shown[name] = s.version
                images[name] = tk.PhotoImage(data=s.latest, format="PPM")
                labels[name].configure(image=images[name])
        if depth.stats:
            text, colour = distance_hint(depth.stats)
            guide.configure(text=text, fg=colour)
        fps = "  ".join(f"{n.split()[0]}: {s.fps:4.1f} fps {s.stats.get('text', '')}" for n, s in streams.items())
        check_mesh()
        check_scan()
        recorder = rec["recorder"]
        if recorder:
            message["text"] = f"REC {recorder.count} sets, {recorder.megabytes:.0f} MB"
        status.configure(text=fps)
        note.configure(text=message["text"])
        root.after(33, tick)

    def close():
        log("viewer closed")
        if rec["recorder"]:
            rec["recorder"] = None
            rec["thread"].join(timeout=2)  # let the last set and the files close cleanly
        if live["proc"] is not None:
            if live["scanning"]:
                live["commands"].put(("stop",))  # save what was scanned
                log("scan stopped by closing, saving ...")
            live["commands"].put(("quit",))
            live["proc"].join(timeout=15)
            if live["proc"].is_alive():
                live["proc"].terminate()
        for s in streams.values():
            s.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    log(f"viewer started (pid {os.getpid()}, depth {'14-bit over USB' if on_usb else '10-bit V4L2'}, "
        f"{calib.CAMERA.mode}{', distances are estimates' if calib.CAMERA.law_estimate else ''})")
    tick()
    root.mainloop()


if __name__ == "__main__":
    main()
