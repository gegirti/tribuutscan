"""Live scanning worker: merges frames from the viewer into one cloud, in its own process.

The viewer starts it with the .venv Python when there is one (so ICP tracking works) and talks
to it over two queues. The worker also serves the browser 3D view of the growing cloud
(view3d page), so cloud data never passes back through the viewer.

Commands:  ("start", preset, out_dir, keep_raw, continue_from or None, depth mode name, remove_base)
           | ("frame", yuyv, grey, depth_buf, bytesperline, (t_colour, t_ir, t_depth))
           | ("stop",) | ("quit",)   (times: camera capture times, seconds)
Replies:   ("ready", url) | ("log", text) | ("frame_done", merged, frames, points, tracking, cue, view)
           | ("saved", path, points)
           cue: {"markers": used, "features": ORB start, "flat": shape alone slides, "base": base found}
           view: what the frame was, at 1/VIEW_STEP of its size (uint8: 0 other, 1 base, 2 object), or None

With keep_raw, the frames the worker processed are also written to out_dir/<stamp>-rec/ (the
viewer's recording format), so a scan can be replayed offline with cloud.py for debugging.
"""
import io
import multiprocessing
import queue
from pathlib import Path
import site
import sys
import threading
import time

import numpy as np

from . import calib
from . import cloud as cl
from . import frames as fr
from . import pointcloud as pc
from . import view3d
from .depth import decode_depth

PAYLOAD_EVERY_S = 0.5  # rebuild the page's cloud at most this often
LIVE_MIN_SEEN = 1  # the page shows every cube while scanning; the saved cloud uses cloud.MIN_SEEN
VIEW_STEP = cl.LABEL_STEP  # the frame's object mask goes back to the viewer at 1/8 of its size (160 x 100)
view_mask = cl.view_mask  # the same labels the scan keeps for fusing again


class LiveSource:
    """view3d source backed by the running Scanner (or the last one)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.scanner, self.name, self.scan_id = None, "no scan yet", 0
        self.phase, self.out_dir, self.busy = "idle", None, threading.Lock()  # scanning / optimising / done
        self.cached, self.built_key, self.built_at = ("0", view3d.pack_arrays(np.empty((0, 3))), {}), None, 0.0

    def current(self):
        with self.lock:
            s = self.scanner
            if s is None:
                return "empty", self.cached[1], {"name": self.name, "points": 0, "session": {}, "phase": self.phase}
            key = (self.scan_id, s.stats["merged"], s.stats["tracking"], s.cue.get("flat"), self.phase,
                   id(s.cloud), s.cloud.point_mm)
            if key != self.built_key and time.time() - self.built_at >= PAYLOAD_EVERY_S:
                xyz, rgb, seen = s.output_cloud(connect=False).points(LIVE_MIN_SEEN)  # region, base as chosen
                payload = view3d.pack_arrays(xyz @ cl.FLIP.T / 1000, rgb, seen)
                info = {"name": self.name, "points": len(xyz), "phase": self.phase,
                        "mesh": bool(self.out_dir and (self.out_dir / "mesh.ply").exists()), "session": {
                    "preset": s.preset, "point_distance_mm": s.cloud.point_mm,
                    "raw_points": s.cloud.raw_points,
                    **{k: s.stats[k] for k in ("frames", "merged", "lost", "relocked", "tracking")},
                    "cue": dict(s.cue)}}
                self.cached = ("-".join(map(str, key)), payload, info)
                self.built_key, self.built_at = key, time.time()
            return self.cached


    def process(self, action):
        """A Process-menu action from the page, on the last finished scan."""
        from . import optimise
        if self.scanner is None or self.phase in ("scanning", "optimising"):
            return "not now: no finished scan (or it is still being scanned or optimised)"
        with self.busy:
            self.phase = "optimising"
            try:
                return optimise.process(self.scanner, self.out_dir, action, log=lambda m: None)
            finally:
                self.phase = "done"
                self.built_key = None

    def mesh_path(self):
        return self.out_dir / "mesh.ply" if self.out_dir else None

    def ply_bytes(self):
        """The scan's cloud.ply: the saved file once the scan is done, else built from the cloud so far."""
        if self.phase in ("done", "idle") and self.out_dir and (self.out_dir / "cloud.ply").exists():
            return (self.out_dir / "cloud.ply").read_bytes()
        with self.lock:
            s = self.scanner
            if s is None:
                return None
            xyz, rgb, normals, seen, _ = cl.ply_arrays(s.output_cloud(), list(s.poses))
        buf = io.BytesIO()
        pc.write_ply(buf, xyz, rgb, normals=normals, seen=seen)
        return buf.getvalue()

    def download_name(self):
        return self.out_dir.name if self.out_dir else "live-scan"


def run(commands, replies, port=view3d.PORT):
    """Worker process main loop."""
    # A spawned process inherits the parent's sys.path (system Python). Put this interpreter's own
    # site-packages (the .venv) FIRST, or old system packages shadow the venv's and Open3D fails to import.
    own = [path for path in site.getsitepackages() if path.startswith(sys.prefix)]
    for path in own:
        site.addsitedir(path)
    sys.path[:] = [path for path in sys.path if path in own] + [path for path in sys.path if path not in own]
    source = LiveSource()
    try:
        server = view3d.serve(source, port)
    except OSError as e:
        replies.put(("log", f"3D view not available: port {port}: {e.strerror}"))
        server = None
    replies.put(("ready", f"http://localhost:{port}/" if server else None))
    out_dir, keep_raw, recorder = None, False, None
    parent = multiprocessing.parent_process()
    timing = {"wait": 0.0, "raw": 0.0, "decode": 0.0, "feed": 0.0, "frames": 0}  # per scan, logged at Stop
    while True:
        t_wait = time.perf_counter()
        try:
            cmd = commands.get(timeout=1.0)
        except queue.Empty:
            if parent is not None and not parent.is_alive():  # viewer killed: don't linger holding the port
                if source.scanner is not None and out_dir is not None and not (out_dir / "cloud.ply").exists():
                    source.scanner.save(out_dir)  # keep what was scanned
                if recorder is not None:
                    recorder.close()
                cmd = ("quit",)
            else:
                continue
        kind = cmd[0]
        timing["wait"] += time.perf_counter() - t_wait
        if kind == "start":
            timing = {k: 0 for k in timing}
            _, preset, out_dir, keep_raw, continue_from, depth_mode, remove_base = cmd
            calib.use_mode(depth_mode)  # the law of the scanner's mode, before any frame is turned into points
            recorder = None
            log = lambda m: replies.put(("log", m))
            with source.lock:
                if continue_from:  # go on with a saved scan: it re-finds its place, then merges into it
                    source.scanner = cl.Scanner.continue_from(continue_from, log=log)
                    source.scanner.remove_base = remove_base
                    log(f"continuing {Path(continue_from).name}: show the scanner a part it has seen")
                else:
                    source.scanner = cl.Scanner(preset, log=log, remove_base=remove_base)
                source.scan_id += 1
                source.name = f"live: {out_dir.name}"
                source.phase, source.out_dir = "scanning", out_dir
        elif kind == "frame" and source.scanner is not None:
            _, yuyv, grey, depth_buf, bytesperline, (t_c, t_i, t_d) = cmd
            t0 = time.perf_counter()
            if keep_raw:
                if recorder is None:
                    recorder = fr.Recorder(bytesperline, root=out_dir, extra={"note": "frames a live scan processed"})
                    with source.lock:  # its frame 0 is the scanner's next index (continued scans count on)
                        source.scanner.add_source(recorder.dir.resolve(), source.scanner.stats["frames"])
                recorder.write({"color": yuyv, "ir": grey, "depth": depth_buf}, {"color": t_c, "ir": t_i, "depth": t_d})
            t1 = time.perf_counter()
            frame = {"yuyv": yuyv, "depth": decode_depth(depth_buf, fr.W, fr.H, bytesperline),  # colour decoded per point
                     "t": t_d, "skew_ms": (t_c - t_d) * 1000}
            t2 = time.perf_counter()
            with source.lock:
                s = source.scanner
                s.feed(frame)
                with s.cloud_lock:
                    points = int((s.cloud.seen >= cl.MIN_SEEN).sum())
            timing["raw"] += t1 - t0
            timing["decode"] += t2 - t1
            timing["feed"] += time.perf_counter() - t2
            timing["frames"] += 1
            replies.put(("frame_done", s.stats["merged"], s.stats["frames"], points, s.stats["tracking"],
                         dict(s.cue, base=s.scene.plane is not None), view_mask(s.last_view)))
        elif kind == "frame":
            replies.put(("frame_done", 0, 0, 0))
        elif kind == "stop" and source.scanner is not None:
            if recorder is not None:
                recorder.close()
                recorder = None
            with source.lock:
                path, kept = source.scanner.save(out_dir)
                source.name = out_dir.name
                source.phase = "optimising"
            replies.put(("saved", str(path), kept))
            n = max(timing["frames"], 1)
            replies.put(("log", f"worker: {timing['frames']} frames; per frame: tracking {1000 * timing['feed'] / n:.0f} ms, "
                                f"raw copy {1000 * timing['raw'] / n:.0f} ms, decode {1000 * timing['decode'] / n:.0f} ms, "
                                f"waiting for frames {1000 * timing['wait'] / n:.0f} ms"))
            try:  # Optimise right away (user decision): loop closures, pose graph, merge again
                from . import optimise
                with source.busy:
                    message = optimise.process(source.scanner, out_dir, "optimise", log=lambda m: None)
                replies.put(("optimised", message))
            except ImportError:
                replies.put(("optimised", "not optimised: needs Open3D (.venv)"))
            finally:
                source.phase, source.built_key = "done", None
        elif kind == "quit":
            if server:
                server.shutdown()
            return
