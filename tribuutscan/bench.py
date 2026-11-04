"""Time each stage of the scanning pipeline on a recording; compare runs.

    .venv/bin/python -m tribuutscan.bench [recording] [--frames N] [--tracker gpu] [--label x] [--against captures/bench/<run>.json]
    .venv/bin/python -m tribuutscan.bench --live [recording] [--frames N] [--tracker gpu|joint]

The recording (default: the marker recording, evaluate's "markers") is replayed through cloud.Scanner
as cloud.py does. Per stage: calls, ms per call and ms per frame, wall clock. Stages are timed
inclusively (a solve inside a tracking step counts in both), and work in the merge thread is listed
apart, since it runs beside tracking. Warm: the first WARMUP frames are left out (imports, first
allocations; GPU engines compile their kernels there). Disk is left out: reading and decoding a set is
its own row, not part of a frame. Also noted: peak memory, GPU memory and busy time where the driver
reports them (/proc/self/fdinfo drm-total-* / drm-engine-*: i915, xe, amdgpu and others), and the
machine and library versions, so runs from other computers can be compared.
Writes captures/bench/<stamp>-<label>.json.

--live replays the recording through the live scan worker instead (livescan.run in its own process,
as the viewer starts it), handing it each frame set's raw buffers whenever it has fewer than two in
hand, as the viewer does, but none dropped: the worker's own frame rate, as a simulated camera. Its
scan is saved and optimised in a temporary directory, then deleted.
"""
import functools
import json
import os
import platform
import resource
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import calib
from .sysinfo import gpu_usage, parse_fdinfo  # noqa: F401 (parse_fdinfo kept here by name: tests and older notes use bench's)
from . import cloud as cl
from . import frames as fr

BENCH = fr.CAPTURES / "bench"
WARMUP = 10


def stages():
    """(owner, attribute, label) of every timed step, in pipeline order."""
    from . import features, fuse, gputrack, scene, track
    return [
        (cl.Scanner, "feed", "frame (all of feed)"),
        (cl, "frame_points", "depth -> points"),
        (cl, "cleaned_frame", "clean depth (merging)"),
        (cl, "isolate_mask", "object piece"),
        (track, "levels", "frame levels"),
        (cl.Scanner, "frame_markers", "markers"),
        (features, "extract", "ORB extract"),
        (features, "relative_pose", "ORB pose"),
        (cl.Scanner, "model_ready", "model (rebuild, waits for merges)"),
        (fuse, "levels", "model levels"),
        (cl.Scanner, "joint_step", "tracking step"),
        (track, "solve", "solve"),
        (cl.Scanner, "gpu_source", "GPU: frame maps"),
        (cl.Scanner, "gpu_step", "GPU: tracking step"),
        (gputrack.Tracker, "solve", "GPU: solve (ray cast + ICP)"),
        (gputrack.Tracker, "fuse", "GPU: fuse"),
        (cl.Scanner, "relock", "re-find"),
        (scene.Scene, "observe", "base: observe"),
        (scene.Scene, "masks", "base: masks"),
        (cl, "view_mask", "labels"),
        (cl.Scanner, "merge", "merge"),
        (cl.VoxelCloud, "add", "cloud add"),
    ]


class StageTimer:
    """Wraps the stage functions while active (`with`); restores them on exit."""

    def __init__(self, steps):
        self.steps, self.on, self.saved = steps, False, []
        self.seconds, self.calls = defaultdict(float), defaultdict(int)

    def __enter__(self):
        for owner, name, label in self.steps:
            fn = owner.__dict__[name]
            self.saved.append((owner, name, fn))
            setattr(owner, name, self._timed(fn, label))
        return self

    def __exit__(self, *exc):
        for owner, name, fn in reversed(self.saved):
            setattr(owner, name, fn)
        self.saved = []

    def _timed(self, fn, label):
        @functools.wraps(fn)
        def timed(*args, **kwargs):
            if not self.on:
                return fn(*args, **kwargs)
            key = label if threading.current_thread() is threading.main_thread() else label + " [merge thread]"
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self.seconds[key] += time.perf_counter() - t0
                self.calls[key] += 1
        return timed


def machine():
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    for name in ("open3d", "cv2"):
        try:
            versions[name] = __import__(name).__version__
        except ImportError:
            versions[name] = None
    cpu, info = platform.processor(), Path("/proc/cpuinfo")
    if info.exists():
        cpu = next((line.split(":", 1)[1].strip() for line in info.read_text().splitlines()
                    if line.startswith("model name")), cpu)
    return {"system": platform.platform(), "cpu": cpu, "threads": os.cpu_count(), **versions}


def run(directory, frames=None, log=print, tracker="joint"):
    """Replay a recording with the stages timed; returns the report (dict). tracker: "joint" (the CPU
    solve) or "gpu" (gputrack)."""
    scanner = cl.Scanner("small", log=lambda m: None, track_with=tracker)
    reading, measured, n = 0.0, 0, 0
    with StageTimer(stages()) as timer:
        source = fr.read_recording(directory)
        while frames is None or n < frames:
            t0 = time.perf_counter()
            frame = next(source, None)
            if frame is None:
                break
            timer.on = n >= WARMUP
            if timer.on:
                reading += time.perf_counter() - t0
                measured += 1
            scanner.feed(frame, n)
            n += 1
            if n % 100 == 0:
                log(f"frame {n}")
        scanner.flush()
    scanner.close()
    per_frame = max(measured, 1)
    rows = []
    for _, _, label in stages():
        for key in (label, label + " [merge thread]"):
            if timer.calls.get(key):
                rows.append({"stage": key, "calls": timer.calls[key],
                             "ms_per_call": round(1000 * timer.seconds[key] / timer.calls[key], 2),
                             "ms_per_frame": round(1000 * timer.seconds[key] / per_frame, 2)})
    return {
        "recording": str(Path(directory).resolve()), "frames": n, "measured": measured, "warmup": WARMUP,
        "tracker": scanner.track_with, "gpu_ms": scanner.stats.get("gpu_ms"),
        "camera": {"source": calib.CAMERA.source, "fingerprint": calib.CAMERA.fingerprint()},
        "tracked": sum(p is not None for _, _, p in scanner.track_log), "merged": scanner.stats["merged"],
        "read_ms_per_frame": round(1000 * reading / per_frame, 2),
        "stages": rows,
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "gpu": gpu_usage(), "machine": machine(),
    }


def live(directory, frames=None, tracker="gpu", log=print):
    """Frames per second of the live scan worker on a recording (see --live); returns the report."""
    import multiprocessing
    import queue
    import shutil
    import tempfile
    from . import livescan
    ctx = multiprocessing.get_context("spawn")
    commands, replies = ctx.Queue(), ctx.Queue()
    proc = ctx.Process(target=livescan.run, args=(commands, replies, 0), daemon=True)  # any free port
    proc.start()
    messages, done, tracked = [], 0, 0

    def reply():
        nonlocal done, tracked
        while True:
            try:
                kind, *rest = replies.get(timeout=5)
                break
            except queue.Empty:
                if not proc.is_alive():
                    raise RuntimeError("the live worker died (its traceback is above)")
        if kind == "frame_done":
            done += 1
            tracked += bool(rest[3]) if len(rest) > 3 else 0
        elif kind == "log":
            messages.append(rest[0])
        return kind, rest

    while reply()[0] != "ready":
        pass
    BENCH.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="live-", dir=BENCH))
    try:
        source, sent, t0 = fr.read_raw(directory), 0, None
        for bufs in source:
            if t0 is None:  # the recording's depth mode is known once its first set is read
                commands.put(("start", "small", tmp / "scan", False, None, calib.CAMERA.mode, True, tracker))
                t0 = time.perf_counter()
            while sent - done >= 2:
                reply()
            commands.put(("frame", bufs["color"], None, bufs["depth"], bufs["bytesperline"], bufs.get("t", (0.0, 0.0, 0.0))))
            sent += 1
            if frames is not None and sent >= frames:
                break
        while done < sent:
            reply()
        elapsed = time.perf_counter() - t0
        commands.put(("stop",))
        t1 = time.perf_counter()
        kind = None
        while kind != "optimised":
            kind, rest = reply()
            if kind == "saved":
                t_saved = time.perf_counter()
        report = {"recording": str(Path(directory).resolve()), "tracker": tracker, "frames": sent, "tracked": tracked,
                  "fps": round(sent / elapsed, 2), "ms_per_frame": round(1000 * elapsed / max(sent, 1), 1),
                  "save_s": round(t_saved - t1, 1), "optimise_s": round(time.perf_counter() - t_saved, 1),
                  "optimised": rest[0], "worker_log": messages, "machine": machine()}
    finally:
        commands.put(("quit",))
        proc.join(timeout=30)
        shutil.rmtree(tmp, ignore_errors=True)
    return report


def show(report, against=None):
    old = {r["stage"]: r for r in (against or {}).get("stages", [])}
    print(f"{report['measured']} frames timed (of {report['frames']}; {report['tracked']} tracked, "
          f"{report['merged']} merged), camera {report['camera']['source']}")
    head = f"{'stage':40s} {'calls':>6s} {'ms/call':>8s} {'ms/frame':>9s}"
    print(head + (f" {'before':>9s} {'change':>7s}" if against else ""))
    for r in report["stages"]:
        line = f"{r['stage']:40s} {r['calls']:6d} {r['ms_per_call']:8.2f} {r['ms_per_frame']:9.2f}"
        if against:
            b = old.get(r["stage"])
            line += (f" {b['ms_per_frame']:9.2f} {(r['ms_per_frame'] / b['ms_per_frame'] - 1) * 100:+6.0f}%"
                     if b and b["ms_per_frame"] else f" {'-':>9s}")
        print(line)
    print(f"{'read + decode (disk, not in a frame)':40s} {'':6s} {'':8s} {report['read_ms_per_frame']:9.2f}")
    print(f"peak memory {report['peak_rss_mb']} MB; GPU {report['gpu'] or 'not used'}")


def main():
    args = sys.argv[1:]
    opt = lambda name, default=None: args[args.index(name) + 1] if name in args else default
    values = {opt(f) for f in ("--frames", "--label", "--against", "--tracker")}
    paths = [a for a in args if not a.startswith("--") and a not in values]
    if paths:
        directory = Path(paths[0])
    else:
        from .evaluate import dataset_path
        directory = dataset_path("markers")
    frames = int(opt("--frames")) if opt("--frames") else None
    if "--live" in args:
        report = live(directory, frames, opt("--tracker", "gpu"))
        print(json.dumps({k: v for k, v in report.items() if k != "machine"}, indent=1))
        BENCH.mkdir(parents=True, exist_ok=True)
        out = BENCH / f"{time.strftime('%Y%m%d-%H%M%S')}-live-{report['tracker']}.json"
        out.write_text(json.dumps(report, indent=2))
        print(f"-> {out}")
        return
    against = json.loads(Path(opt("--against")).read_text()) if opt("--against") else None
    report = run(directory, frames, tracker=opt("--tracker", "joint"))
    show(report, against)
    BENCH.mkdir(parents=True, exist_ok=True)
    out = BENCH / f"{time.strftime('%Y%m%d-%H%M%S')}-{opt('--label', 'run')}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
