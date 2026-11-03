"""Live scanning worker: merges frames from the viewer into one cloud, in its own process.

The viewer starts it with the .venv Python when there is one (so ICP tracking works) and talks
to it over two queues. The worker also serves the browser 3D view of the growing cloud
(view3d page), so cloud data never passes back through the viewer.

Commands:  ("start", preset, out_dir, keep_raw, continue_from or None, depth mode name, remove_base[, tracker])
           (tracker: "gpu", the default, which falls back to the CPU solve without a GPU; or "joint")
           | ("frame", yuyv, grey or None, depth_buf, bytesperline, (t_colour, t_ir, t_depth))
           | ("stop"[, mesh]) | ("quit",)   (times: camera capture times, seconds; mesh: make the mesh after
           Optimise, fusion on the GPU, so Stop ends with a finished model)
Replies:   ("ready", url) | ("log", text) | ("optimised", message, a mesh follows) | ("meshed", message)
           | ("frame_done", merged, frames, points, tracking, cue, view, model, t_depth) | ("saved", path, points)
           cue: {"markers": used, "features": ORB start, "flat": shape alone slides, "base": base found,
                 "turning": the object turns and the scanner stands (Scanner.check_room)}
           view: what the frame was, at 1/VIEW_STEP of its size (uint8: 0 other, 1 base, 2 object), or None
           model: the live model view of that frame (gpu.preview_codes at 1/gputrack.PREVIEW_STEP: its piece
           against the model built so far), or None without the GPU tracker; t_depth: the frame's depth time

With keep_raw, the frames the worker processed are also written to out_dir/<stamp>-rec/ (the
viewer's recording format, without IR), so a scan can be replayed offline with cloud.py for
debugging. The copy stops (the scan goes on) when the disk is nearly full (frames.Recorder). It is
written beside tracking (RawCopy): in the first live GPU scan it took 34 ms of every frame.
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
from .depth import LazyFrame, decode_strided

PAYLOAD_EVERY_S = 0.5  # rebuild the page's cloud at most this often
LIVE_MIN_SEEN = 1  # the page shows every cube while scanning; the saved cloud uses cloud.MIN_SEEN
VIEW_STEP = cl.LABEL_STEP  # the frame's object mask goes back to the viewer at 1/8 of its size (160 x 100)
view_mask = cl.view_mask  # the same labels the scan keeps for fusing again


class RawCopy:
    """frames.Recorder writes on a thread of their own (zlib and file writes release the GIL). put()
    waits while QUEUE sets are pending, so no frame is left out; `error` is the DiskFull that stopped it."""

    QUEUE = 30

    def __init__(self, recorder):
        self.recorder, self.error = recorder, None
        self.queue = queue.Queue(self.QUEUE)
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def _write(self):
        while (item := self.queue.get()) is not None:
            if self.error is None:
                try:
                    self.recorder.write(*item)
                except fr.DiskFull as e:
                    self.error = e

    def put(self, bufs, stamps):
        self.queue.put((bufs, stamps))

    def close(self):
        self.queue.put(None)
        self.thread.join()
        self.recorder.close()


class Reader(threading.Thread):
    """Takes the viewer's commands off its queue beside tracking, in the order they came, frames as
    (*command, every-4th-pixel depth, cleared rows): depth.decode_strided, with the garbled-row check.
    The full depth stays packed (depth.LazyFrame): the GPU unpacks it for tracking, the merge process
    for itself (the full decode here took 14 ms a frame beside tracking, holding the GIL)."""

    def __init__(self, commands):
        super().__init__(daemon=True)
        self.commands, self.out = commands, queue.Queue()
        self.decode_s = 0.0  # time spent decoding (beside tracking)

    def run(self):
        while True:
            cmd = self.commands.get()
            if cmd[0] == "frame":
                t = time.perf_counter()
                _, _, _, depth_buf, bytesperline, _ = cmd
                cmd = (*cmd, *decode_strided(depth_buf, fr.W, fr.H, bytesperline))
                self.decode_s += time.perf_counter() - t
            self.out.put(cmd)
            if cmd[0] == "quit":
                return


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
                xyz, rgb, seen = s.live_points(LIVE_MIN_SEEN)  # region, base as chosen
                payload = view3d.pack_arrays(xyz @ cl.FLIP.T / 1000, rgb, seen)
                info = {"name": self.name, "points": len(xyz), "phase": self.phase, "session": {
                    "preset": s.preset, "point_distance_mm": s.cloud.point_mm,
                    "raw_points": s.cloud.raw_points,
                    **{k: s.stats[k] for k in ("frames", "merged", "lost", "relocked", "tracking")},
                    "cue": dict(s.cue)}}
                self.cached = ("-".join(map(str, key)), payload, info)
                self.built_key, self.built_at = key, time.time()
            return self.cached[0], self.cached[1], dict(self.cached[2], mesh=view3d.mesh_version(self.mesh_path()))


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
    timing = {"wait": 0.0, "raw": 0.0, "feed": 0.0, "frames": 0}  # per scan, logged at Stop
    reader = Reader(commands)
    reader.start()
    decoded_at_start = 0.0
    while True:
        t_wait = time.perf_counter()
        try:
            cmd = reader.out.get(timeout=1.0)
        except queue.Empty:
            if parent is not None and not parent.is_alive():  # viewer killed: don't linger holding the port
                if source.scanner is not None and out_dir is not None and not (out_dir / "cloud.ply").exists():
                    source.scanner.close()
                    source.scanner.save(out_dir)  # keep what was scanned
                if recorder is not None:
                    recorder.close()
                cmd = ("quit",)
            else:
                continue
        kind = cmd[0]
        timing["wait"] += time.perf_counter() - t_wait
        if kind == "start":
            timing, decoded_at_start = {k: 0 for k in timing}, reader.decode_s
            _, preset, out_dir, keep_raw, continue_from, depth_mode, remove_base, *tracker = cmd
            calib.use_mode(depth_mode)  # the law of the scanner's mode, before any frame is turned into points
            recorder = None
            log = lambda m: replies.put(("log", m))
            with source.lock:
                if continue_from:  # go on with a saved scan: it re-finds its place, then merges into it
                    source.scanner = cl.Scanner.continue_from(continue_from, log=log)
                    source.scanner.remove_base = remove_base
                    log(f"continuing {Path(continue_from).name}: show the scanner a part it has seen")
                else:
                    source.scanner = cl.Scanner(preset, log=log, remove_base=remove_base,
                                                track_with=tracker[0] if tracker else "gpu", live_view=True,
                                                merge_process=True)
                source.scan_id += 1
                source.name = f"live: {out_dir.name}"
                source.phase, source.out_dir = "scanning", out_dir
        elif kind == "frame" and source.scanner is not None:
            _, yuyv, grey, depth_buf, bytesperline, (t_c, t_i, t_d), depth4, cleared = cmd
            t0 = time.perf_counter()
            if keep_raw:
                try:
                    if recorder is None:
                        recorder = RawCopy(fr.Recorder(bytesperline, root=out_dir,
                                                       extra={"note": "frames a live scan processed"}))
                        with source.lock:  # its frame 0 is the scanner's next index (continued scans count on)
                            source.scanner.add_source(recorder.recorder.dir.resolve(), source.scanner.stats["frames"])
                    if recorder.error is not None:
                        raise recorder.error
                    recorder.put({"color": yuyv, "ir": grey, "depth": depth_buf}, {"color": t_c, "ir": t_i, "depth": t_d})
                except fr.DiskFull as e:
                    keep_raw = False
                    if recorder is not None:
                        recorder.close()
                        recorder = None
                    replies.put(("log", f"raw copy stopped, the scan goes on: {e}"))
            t2 = time.perf_counter()
            frame = LazyFrame(yuyv=yuyv, depth_raw=depth_buf, bytesperline=bytesperline, depth4=depth4, cleared=cleared,
                              shape=(fr.H, fr.W), t=t_d, skew_ms=(t_c - t_d) * 1000)  # colour per point
            with source.lock:
                s = source.scanner
                s.feed(frame)
                points = s.points_kept  # (the cloud lock is held through each ~90 ms merge: no waiting for it)
            timing["raw"] += t2 - t0
            timing["feed"] += time.perf_counter() - t2
            timing["frames"] += 1
            replies.put(("frame_done", s.stats["merged"], s.stats["frames"], points, s.stats["tracking"],
                         dict(s.cue, base=s.scene.plane is not None, turning=s.object_turns), view_mask(s.last_view),
                         s.view_codes, t_d))
        elif kind == "frame":
            replies.put(("frame_done", 0, 0, 0))
        elif kind == "stop" and source.scanner is not None:
            if recorder is not None:
                recorder.close()
                recorder = None
            with source.lock:
                source.scanner.close()  # the GPU tracker's memory back (Optimise and the mesh come next)
                path, kept = source.scanner.save(out_dir)
                source.name = out_dir.name
                source.phase = "optimising"
            replies.put(("saved", str(path), kept))
            n = max(timing["frames"], 1)
            replies.put(("log", f"worker: {timing['frames']} frames; per frame: tracking {1000 * timing['feed'] / n:.0f} ms, "
                                f"raw copy {1000 * timing['raw'] / n:.0f} ms, waiting for frames "
                                f"{1000 * timing['wait'] / n:.0f} ms, decode beside tracking "
                                f"{1000 * (reader.decode_s - decoded_at_start) / n:.0f} ms"
                                f"; lost {source.scanner.stats['lost']}, found again {source.scanner.stats['relocked']} "
                                f"({source.scanner.stats['shape_refinds']} by shape)"
                                + (f"; GPU {source.scanner.stats['gpu_ms']}" if source.scanner.stats.get("gpu_ms") else "")))
            make_mesh = len(cmd) > 1 and bool(cmd[1])
            try:  # Optimise right away (user decision): loop closures, pose graph, merge again
                from . import optimise
                with source.busy:
                    message = optimise.process(source.scanner, out_dir, "optimise", log=lambda m: None)
                replies.put(("optimised", message, make_mesh))
                if make_mesh:  # and the mesh (the 3D page's Make mesh, ~15-20 s on the GPU): a finished model
                    source.phase = "meshing"
                    try:
                        with source.busy:
                            message = optimise.process(source.scanner, out_dir, "mesh", log=lambda m: None)
                    except Exception as e:  # (the viewer waits for this reply)
                        message = f"mesh not made: {e}"
                    replies.put(("meshed", message))
            except ImportError:
                replies.put(("optimised", "not optimised: needs Open3D (.venv)", False))
            finally:
                source.phase, source.built_key = "done", None
        elif kind == "quit":
            if server:
                server.shutdown()
            return
