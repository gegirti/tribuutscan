"""Live scanning worker: merges frames from the viewer into one cloud, in its own process.

The viewer starts it with the .venv Python when there is one (so ICP tracking works) and talks
to it over two queues. The worker also serves the browser 3D view of the growing cloud
(view3d page), so cloud data never passes back through the viewer.

Commands:  ("start", preset, out_dir) | ("frame", yuyv, y10, bytesperline) | ("stop",) | ("quit",)
Replies:   ("ready", url) | ("log", text) | ("frame_done", merged, frames, points) | ("saved", path, points)
"""
import site
import sys
import threading
import time

import numpy as np

from . import cloud as cl
from . import frames as fr
from . import view3d
from .depth import decode_y10
from .image import yuyv_to_rgb

PAYLOAD_EVERY_S = 0.5  # rebuild the page's cloud at most this often


class LiveSource:
    """view3d source backed by the running Scanner (or the last one)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.scanner, self.name, self.scan_id = None, "no scan yet", 0
        self.cached, self.built_key, self.built_at = ("0", view3d.pack_arrays(np.empty((0, 3))), {}), None, 0.0

    def current(self):
        with self.lock:
            s = self.scanner
            if s is None:
                return "empty", self.cached[1], {"name": self.name, "points": 0, "session": {}}
            key = (self.scan_id, s.stats["merged"])
            if key != self.built_key and time.time() - self.built_at >= PAYLOAD_EVERY_S:
                xyz, rgb, seen = s.cloud.points(cl.MIN_SEEN)
                payload = view3d.pack_arrays(xyz @ cl.FLIP.T / 1000, rgb, seen)
                info = {"name": self.name, "points": len(xyz), "session": {
                    "preset": s.preset, "point_distance_mm": s.cloud.point_mm,
                    "raw_points": s.cloud.raw_points, **{k: s.stats[k] for k in ("frames", "merged", "lost")}}}
                self.cached = (f"{key[0]}-{key[1]}", payload, info)
                self.built_key, self.built_at = key, time.time()
            return self.cached


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
    out_dir = None
    while True:
        cmd = commands.get()
        kind = cmd[0]
        if kind == "start":
            _, preset, out_dir = cmd
            with source.lock:
                source.scanner = cl.Scanner(preset, log=lambda m: replies.put(("log", m)))
                source.scan_id += 1
                source.name = f"live: {out_dir.name}"
        elif kind == "frame" and source.scanner is not None:
            _, yuyv, y10, bytesperline = cmd
            frame = {"color": yuyv_to_rgb(yuyv, fr.W, fr.H), "depth": decode_y10(y10, fr.W, fr.H, bytesperline)}
            with source.lock:
                s = source.scanner
                s.feed(frame)
                points = int((s.cloud.seen >= cl.MIN_SEEN).sum())
            replies.put(("frame_done", s.stats["merged"], s.stats["frames"], points))
        elif kind == "frame":
            replies.put(("frame_done", 0, 0, 0))
        elif kind == "stop" and source.scanner is not None:
            with source.lock:
                path, kept = source.scanner.save(out_dir)
                source.name = out_dir.name
            replies.put(("saved", str(path), kept))
        elif kind == "quit":
            if server:
                server.shutdown()
            return
