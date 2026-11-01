"""Time each stage of the scanning pipeline on a recording; compare runs.

    .venv/bin/python -m tribuutscan.bench [recording] [--frames N] [--label x] [--against captures/bench/<run>.json]

The recording (default: the marker recording, evaluate's "markers") is replayed through cloud.Scanner
as cloud.py does. Per stage: calls, ms per call and ms per frame, wall clock. Stages are timed
inclusively (a solve inside a tracking step counts in both), and work in the merge thread is listed
apart, since it runs beside tracking. Warm: the first WARMUP frames are left out (imports, first
allocations; GPU engines compile their kernels there). Disk is left out: reading and decoding a set is
its own row, not part of a frame. Also noted: peak memory, GPU memory and busy time where the driver
reports them (/proc/self/fdinfo drm-total-* / drm-engine-*: i915, xe, amdgpu and others), and the
machine and library versions, so runs from other computers can be compared.
Writes captures/bench/<stamp>-<label>.json.
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
from . import cloud as cl
from . import frames as fr

BENCH = fr.CAPTURES / "bench"
WARMUP = 10


def stages():
    """(owner, attribute, label) of every timed step, in pipeline order."""
    from . import features, fuse, scene, track
    return [
        (cl.Scanner, "feed", "frame (all of feed)"),
        (cl, "frame_points", "depth -> points"),
        (cl.Scanner, "cleaned", "clean depth"),
        (cl, "isolate_mask", "object piece"),
        (track, "levels", "frame levels"),
        (cl.Scanner, "frame_markers", "markers"),
        (features, "extract", "ORB extract"),
        (features, "relative_pose", "ORB pose"),
        (cl.Scanner, "model_ready", "model (rebuild, waits for merges)"),
        (fuse, "levels", "model levels"),
        (cl.Scanner, "joint_step", "tracking step"),
        (track, "solve", "solve"),
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


def parse_fdinfo(texts):
    """GPU memory (MiB per region) and busy time (s per engine) from DRM fdinfo texts, one entry per
    client (fds sharing a client are counted once). None when no fd is a DRM client."""
    clients = {}
    for text in texts:
        fields = dict(line.split(":", 1) for line in text.splitlines() if ":" in line)
        fields = {k.strip(): v.strip() for k, v in fields.items()}
        if "drm-client-id" in fields:
            clients[(fields.get("drm-pdev"), fields["drm-client-id"])] = fields
    if not clients:
        return None
    units = {"": 1 / 2 ** 20, "KiB": 1 / 1024, "MiB": 1.0, "GiB": 1024.0}
    memory, busy = defaultdict(float), defaultdict(float)
    for fields in clients.values():
        for key, value in fields.items():
            number, _, unit = value.partition(" ")
            if key.startswith(("drm-total-", "drm-memory-")) and number.isdigit():
                memory[key.split("-", 2)[2]] += int(number) * units.get(unit, 0)
            elif key.startswith("drm-engine-") and not key.startswith("drm-engine-capacity-") and unit == "ns":
                busy[key[len("drm-engine-"):]] += int(number) / 1e9
    return {"driver": next(iter(clients.values())).get("drm-driver"),
            "memory_mib": {k: round(v, 1) for k, v in memory.items()},
            "busy_s": {k: round(v, 3) for k, v in busy.items()}}


def gpu_usage():
    texts = []
    for fd in os.listdir("/proc/self/fdinfo") if os.path.isdir("/proc/self/fdinfo") else []:
        try:
            texts.append(Path(f"/proc/self/fdinfo/{fd}").read_text())
        except OSError:
            pass
    return parse_fdinfo(texts)


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


def run(directory, frames=None, log=print):
    """Replay a recording with the stages timed; returns the report (dict)."""
    scanner = cl.Scanner("small", log=lambda m: None)
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
        "camera": {"source": calib.CAMERA.source, "fingerprint": calib.CAMERA.fingerprint()},
        "tracked": sum(p is not None for _, _, p in scanner.track_log), "merged": scanner.stats["merged"],
        "read_ms_per_frame": round(1000 * reading / per_frame, 2),
        "stages": rows,
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "gpu": gpu_usage(), "machine": machine(),
    }


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
    values = {opt(f) for f in ("--frames", "--label", "--against")}
    paths = [a for a in args if not a.startswith("--") and a not in values]
    if paths:
        directory = Path(paths[0])
    else:
        from .evaluate import dataset_path
        directory = dataset_path("markers")
    frames = int(opt("--frames")) if opt("--frames") else None
    against = json.loads(Path(opt("--against")).read_text()) if opt("--against") else None
    report = run(directory, frames)
    show(report, against)
    BENCH.mkdir(parents=True, exist_ok=True)
    out = BENCH / f"{time.strftime('%Y%m%d-%H%M%S')}-{opt('--label', 'run')}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
