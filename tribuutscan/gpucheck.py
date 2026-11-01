"""Check the GPU engine on this machine, no scanner needed: correctness and speed on a synthetic sphere.

    .venv/bin/python -m tribuutscan.gpucheck [--frames N] [--out report.json]

A 60 mm sphere is seen from N poses around it at 25 cm (1280 x 800 depth, the scanner's lens), each
frame fused, ray cast from its pose with one ICP pass against it, then meshed. On each WebGPU adapter
the machine has (its GPU and the CPU fallback, llvmpipe): the mesh must lie on the sphere (median
distance under 0.05 mm; the CPU fallback fuses 12 frames, it is slow). Prints the times per step and a
report to send back (README: Compatibility).
"""
import json
import platform
import sys
import time

import numpy as np

from . import gpu

RADIUS_MM, CENTRE_MM, ORBIT_MM = 60.0, np.array([0.0, 0.0, 300.0]), 250.0
K = np.array([[960.0, 0, 639.5], [0, 960.0, 399.5], [0, 0, 1]])
SIZE = (1280, 800)


def look_at(eye, target):
    """Camera -> world pose (mm) for a camera at `eye` looking at `target` (y down, z forward)."""
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross([0.0, -1.0, 0.0], z)
    x /= np.linalg.norm(x)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([x, np.cross(z, x), z], 1)
    pose[:3, 3] = eye
    return pose


def frame(pose):
    """Depth (mm) of the sphere seen from `pose`, and a colour pattern on it."""
    d = gpu.rays(K, SIZE) @ pose[:3, :3].T
    o = pose[:3, 3] - CENTRE_MM
    a, b, c = (d * d).sum(-1), 2 * (d @ o), o @ o - RADIUS_MM ** 2
    disc = b * b - 4 * a * c
    with np.errstate(invalid="ignore"):
        s = np.where(disc > 0, (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a), 0)
    p = pose[:3, 3] + d * s[..., None] - CENTRE_MM
    rgb = np.stack([128 + 100 * np.sign(np.sin(p[..., 0] / 8)), 128 + 100 * np.sign(np.sin(p[..., 1] / 8)),
                    np.full(s.shape, 128.0)], -1)
    return s.astype(np.float32), np.where(s[..., None] > 0, rgb, 0).astype(np.uint8)


def run(kind, n_frames, log=print):
    from . import gpu_wgpu
    layout = gpu.Layout(CENTRE_MM - RADIUS_MM - 10, CENTRE_MM + RADIUS_MM + 10)
    e = gpu_wgpu.Engine(layout, K, SIZE, kind, 600.0)
    rows, tick = [], None
    for k in range(n_frames):
        a = 2 * np.pi * k / n_frames
        eye = CENTRE_MM + ORBIT_MM * np.array([np.sin(a), 0.35 * np.sin(3 * a), -np.cos(a)])
        pose = look_at(eye, CENTRE_MM)
        depth, rgb = frame(pose)
        t0 = time.perf_counter()
        e.fuse(depth, rgb, pose)
        t1 = time.perf_counter()
        row = {"fuse_ms": 1000 * (t1 - t0), **{f"gpu_{k2}_ms": v for k2, v in e.gpu_ms.items()}}
        e.raycast(pose)
        row["gpu_raycast_ms"] = e.gpu_ms["raycast"]
        sums = e.icp(pose, pose, CENTRE_MM)
        row["gpu_icp_ms"], row["icp_pairs"] = e.gpu_ms["icp"], int(sums[27])
        rows.append(row)
        if k == 3 and tick is None:
            e.calibrate_ticks()
            tick = e.tick_ns
    t = time.perf_counter()
    v, tri, _ = e.mesh(2)
    mesh_s = time.perf_counter() - t
    bricks = e.counts()[0]
    e.close()
    dist = np.abs(np.linalg.norm(v - CENTRE_MM, axis=1) - RADIUS_MM)
    warm = rows[5:] or rows
    return {"device": e.device_name, "compile_s": round(e.compile_s, 3), "timestamp_ns_per_unit": tick,
            "median_ms": {k: round(float(np.median([r[k] for r in warm])), 3) for k in warm[0] if k != "icp_pairs"},
            "mesh_s": round(mesh_s, 3), "triangles": int(len(tri)), "bricks": bricks,
            "mesh_to_sphere_mm": {"median": round(float(np.median(dist)), 4), "p99": round(float(np.percentile(dist, 99)), 4)},
            "icp_pairs": rows[-1]["icp_pairs"]}, (v, tri)


def main():
    opt = lambda name: sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None
    n = int(opt("--frames") or 36)
    report = {"system": platform.platform(), "python": platform.python_version(), "numpy": np.__version__}
    try:
        import wgpu
        from . import gpu_wgpu
        report["wgpu"] = wgpu.__version__
        found = gpu_wgpu.adapters()
    except Exception as e:  # no wgpu, or no adapter at all
        report["error"] = f"WebGPU not available: {e}"
        print(json.dumps(report, indent=1))
        sys.exit(1)
    report["adapters"] = {k: {"device": a.info["device"], "backend": a.info["backend_type"], "driver": a.info["description"]}
                          for k, a in found.items()}
    ok = "gpu" in found
    for kind in ("gpu", "cpu"):
        if kind not in found:
            continue
        print(f"{kind}: {found[kind].info['device']} ...", flush=True)
        try:
            report[kind] = run(kind, n if kind == "gpu" else min(n, 12))[0]
            ok &= report[kind]["mesh_to_sphere_mm"]["median"] < 0.05
        except Exception as e:
            report[kind] = {"error": repr(e)}
            ok = False
    report["passed"] = bool(ok)
    print(json.dumps(report, indent=1))
    if opt("--out"):
        with open(opt("--out"), "w") as f:
            json.dump(report, f, indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
