"""GPU engine bake-off (Phase 2): the same frames through each engine, timed and checked.

    .venv/bin/python -m tribuutscan.bakeoff dump captures/<stamp>-scan <frames.npz>
    .venv/bin/python -m tribuutscan.bakeoff run <frames.npz> [wgpu-gpu wgpu-cpu] [--frames N] [--out report.json]
    .venv/bin/python -m tribuutscan.bakeoff slice captures/<stamp>-scan [--seconds S] [--capacity BRICKS] [--cpu] [--out report.json]

`slice` runs the winner's whole slice (WebGPU) on a saved scan, as fusion does it: frames read from the
recording, prepared and fused on the GPU, each ray cast from its pose with one ICP pass against it, then
the mesh; again and again for --seconds (stability), with a small --capacity to fill the brick pool.

The OpenCL engine it was measured against (gpu_cl.py) was removed after the decision; it is in commit
401cdf9 (git show 401cdf9:tribuutscan/gpu_cl.py), its numbers in CLAUDE.md.

`dump` prepares a scan's merged frames once, exactly as fusion.fuse does (cleaned, masked, colour per
pixel, lens taken out), with their poses. `run` fuses them in each engine and per frame ray casts the
volume from the frame's pose and takes one ICP-sum pass of the frame against that ray cast. Checked:
the volume voxel by voxel against Open3D's (fusion.Engine, the Phase 1 reference), the frame maps and
ICP sums against the numpy references (gpu.py), the ray-cast depth against Open3D's ray cast. Timed:
each kernel on the device (timestamps / profiling events) and each step's wall clock with its uploads,
launches and readbacks, warm (the first WARM frames left out).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import gpu

WARM = 5
ENGINES = ("wgpu-gpu", "wgpu-cpu")


def dump(scan_dir, out):
    from . import cloud as cl
    from . import fusion
    from .calib import CAMERA
    scanner = cl.Scanner.restore(Path(scan_dir), log=lambda m: None)
    pinhole = fusion.Pinhole()
    arrays, poses, lo, hi = {}, [], [], []
    for k, (frame, pose, labels) in enumerate(fusion.scan_frames(scanner, "merged", lambda: fusion.check_camera(scan_dir))):
        mask = None if labels is None else fusion.label_mask(labels, cl.LABEL_STEP, frame["depth"].shape)
        depth, rgb = fusion.frame_images(frame, mask)
        depth, rgb = pinhole(depth), pinhole(rgb)
        arrays[f"d{k}"], arrays[f"c{k}"] = depth, np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
        poses.append(pose)
        v, u = np.nonzero(depth)
        pts = gpu.rays(pinhole.K, depth.shape[::-1])[v, u] * depth[v, u, None]
        world = pts @ pose[:3, :3].T + pose[:3, 3] * 1000
        lo.append(np.percentile(world, 0.1, 0))
        hi.append(np.percentile(world, 99.9, 0))
    np.savez_compressed(out, **arrays, poses=np.array(poses), K=pinhole.K, lo=np.min(lo, 0) - 8, hi=np.max(hi, 0) + 8,
                        far_mm=CAMERA.window_mm[1] + 20)
    print(f"{len(poses)} frames -> {out}")


class Frames:
    """The dumped frames, read one by one (each npz array once)."""

    def __init__(self, path):
        self.f = np.load(path)
        self.poses, self.K, self.far_mm = self.f["poses"], self.f["K"], float(self.f["far_mm"])
        self.lo, self.hi = self.f["lo"], self.f["hi"]

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, k):
        return self.f[f"d{k}"], self.f[f"c{k}"], gpu.pose_mm(self.poses[k])


def open_engine(name, layout, K, size, far_mm):
    from . import gpu_wgpu
    return gpu_wgpu.Engine(layout, K, size, name.split("-")[1], far_mm, nearest=False)  # Open3D's pixel rule


def open3d_volume(frames, n):
    from . import fusion
    e = fusion.Engine(frames.K, gpu.VOXEL_MM, gpu.TRUNC_VOXELS, frames.far_mm)
    for k in range(n):
        d, c, pose = frames[k]
        p = pose.copy()
        p[:3, 3] /= 1000
        e.integrate(d, c.astype(np.float32), p)
    return e


def open3d_voxels(o):
    import open3d.core as o3c
    hm = o.grid.hashmap()
    bi = hm.active_buf_indices().to(o3c.int64).numpy()
    keys = hm.key_tensor().numpy()[bi]
    ts, wt = o.grid.attribute("tsdf").numpy(), o.grid.attribute("weight").numpy()
    return keys, ts.reshape(len(ts), -1)[bi], wt.reshape(len(wt), -1)[bi]


def compare_volume(engine, layout, keys, ot, ow):
    """Voxel by voxel against Open3D's: bricks, weights, TSDF (normalised) where both have seen it."""
    index, tw, _ = engine.volume()
    tg, wg = gpu.unpack_tsdf(tw)
    bc = (keys - np.round(layout.origin / layout.brick_mm)).astype(np.int64)
    inside = ((bc >= 0) & (bc < layout.dims)).all(1)
    flat = bc[inside, 0] + layout.dims[0] * (bc[inside, 1] + layout.dims[1] * bc[inside, 2])
    slots = index[flat]
    have = slots < layout.capacity
    ot, ow = ot[inside][have].ravel(), ow[inside][have].ravel()
    gt, gw = tg.reshape(-1, 512)[slots[have]].ravel(), wg.reshape(-1, 512)[slots[have]].ravel()
    both = (ow > 0) & (gw > 0)
    diff = np.abs(ot[both] - gt[both])
    return {"open3d_bricks": int(len(keys)), "outside_volume": int((~inside).sum()), "bricks": engine.counts()[0],
            "open3d_bricks_missing": int((~have).sum()), "voxels_seen": int(((ow > 0) | (gw > 0)).sum()),
            "weight_differs": int((ow != gw).sum()), "tsdf_median": float(np.median(diff)),
            "tsdf_p99": float(np.percentile(diff, 99)), "tsdf_over_0.01": float((diff > 0.01).mean())}


def open3d_raycast(o, pose, K, size, far_mm, min_weight=gpu.RAY_MIN_WEIGHT):
    import open3d.core as o3c
    hm = o.grid.hashmap()
    keys = hm.key_tensor()[hm.active_buf_indices().to(o3c.int64)]
    p = pose.copy()
    p[:3, 3] /= 1000
    r = o.grid.ray_cast(keys, o3c.Tensor(K), o3c.Tensor(np.linalg.inv(p)), size[0], size[1], ["depth"], 1000.0, 0.05,
                        far_mm / 1000, min_weight, gpu.TRUNC_VOXELS, 8)
    return r["depth"].numpy()[..., 0]


def run(path, names, n_frames=None, out=None, log=print):
    frames = Frames(path)
    n = min(n_frames or len(frames), len(frames))
    layout = gpu.Layout(frames.lo, frames.hi)
    d0 = frames[0][0]
    size = (d0.shape[1], d0.shape[0])
    log(f"{n} frames {size}, volume {layout.dims} bricks ({layout.bricks}), origin {layout.origin}")
    t = time.perf_counter()
    o = open3d_volume(frames, n)
    log(f"open3d reference: {o.blocks()} bricks, {time.perf_counter() - t:.1f} s")
    keys, ot, ow = open3d_voxels(o)
    check = sorted({0, n // 2, n - 1})
    centre = (frames.lo + frames.hi) / 2  # ICP twists about the volume's centre
    report = {"frames": n, "size": size, "volume_bricks": layout.dims.tolist(), "engines": {}}
    for name in names:
        try:
            e = open_engine(name, layout, frames.K, size, frames.far_mm)
        except Exception as ex:  # an engine this machine cannot run
            log(f"{name}: not available ({ex})")
            report["engines"][name] = {"error": str(ex)}
            continue
        log(f"{name}: {e.device_name}, compile {e.compile_s:.2f} s")
        calibrated = False
        rows, checks = [], {"maps": [], "icp": [], "ray_vs_open3d": []}
        for k in range(n):
            d, c, pose = frames[k]
            if k == WARM and hasattr(e, "calibrate_ticks"):  # a volume to ray cast now exists
                e.calibrate_ticks()
                calibrated = True
            t0 = time.perf_counter()
            e.fuse(d, c, pose)
            t1 = time.perf_counter()
            fuse_gpu = dict(e.gpu_ms)
            e.raycast(pose)
            t2 = time.perf_counter()
            ray_gpu = e.gpu_ms.get("raycast")
            sums = e.icp(pose, pose, centre)
            t3 = time.perf_counter()
            icp_gpu = e.gpu_ms.get("icp")
            rows.append({"fuse_ms": 1000 * (t1 - t0), "raycast_ms": 1000 * (t2 - t1), "icp_ms": 1000 * (t3 - t2),
                         **{f"gpu_{key}_ms": v for key, v in fuse_gpu.items()}, "gpu_raycast_ms": ray_gpu, "gpu_icp_ms": icp_gpu})
            if k in check:
                t4 = time.perf_counter()
                vm, nm = e.frame_maps()
                rv, rn = e.model_maps()
                rows[-1]["readback_maps_ms"] = 1000 * (time.perf_counter() - t4)
                ref_v, ref_n = gpu.maps(d, frames.K)
                checks["maps"].append({"vertex_max_mm": float(np.abs(vm - ref_v).max()), "normal_max": float(np.abs(nm - ref_n).max())})
                ref = gpu.icp_sums(vm.astype(np.float64), nm.astype(np.float64), pose, rv.astype(np.float64),
                                   rn.astype(np.float64), np.linalg.inv(pose), frames.K, centre)
                checks["icp"].append({"pairs": int(sums[27]), "rel_max": float(np.abs(sums - ref).max() / np.abs(ref).max())})
                if k == n - 1:  # Open3D's ray cast needs the full volume: only at the end
                    od = open3d_raycast(o, pose, frames.K, size, frames.far_mm)
                    gd = rv[..., 3]
                    both = (od > 0) & (gd > 0)
                    dd = np.abs(od[both] - gd[both])
                    checks["ray_vs_open3d"].append({"hits": int((gd > 0).sum()), "open3d_hits": int((od > 0).sum()),
                                                    "both": int(both.sum()), "median_mm": float(np.median(dd)),
                                                    "p99_mm": float(np.percentile(dd, 99))})
        e.sync()
        warm = rows[WARM:] if len(rows) > WARM else rows
        keys_all = sorted({key for r in warm for key in r})
        med = {key: float(np.median([r[key] for r in warm if r.get(key) is not None]))
               for key in keys_all if any(r.get(key) is not None for r in warm)}
        result = {"device": e.device_name, "compile_s": round(e.compile_s, 3),
                  "tick_ns": round(getattr(e, "tick_ns", 1.0), 2) if calibrated else None, "median": {k: round(v, 3) for k, v in med.items()},
                  "volume": compare_volume(e, layout, keys, ot, ow), "counts": e.counts(), "checks": checks}
        report["engines"][name] = result
        log(json.dumps({name: result}, indent=1))
        if hasattr(e, "close"):
            e.close()
    if out:
        Path(out).write_text(json.dumps(report, indent=1))
    return report


def slice_run(scan_dir, seconds=0.0, capacity=gpu.CAPACITY, kind="gpu", log=print):
    import resource
    from . import bench, fusion, gpu_wgpu
    from . import cloud as cl
    from .calib import CAMERA
    scanner = cl.Scanner.restore(Path(scan_dir), log=lambda m: None)
    lo, hi = fusion.scan_bounds(scanner)
    passes, start, tick = [], time.time(), None
    while True:
        e, rows, t_read = None, [], time.perf_counter()
        for frame, pose, labels in fusion.scan_frames(scanner, "merged", lambda: fusion.check_camera(scan_dir), yuyv=True):
            read_ms = 1000 * (time.perf_counter() - t_read)
            if e is None:
                pin = fusion.Pinhole()
                layout = gpu.Layout(lo, hi, capacity=capacity)
                e = gpu_wgpu.Engine(layout, pin.K, frame["depth"].shape[::-1], kind, CAMERA.window_mm[1] + 20)
                e.set_camera(CAMERA, pin.index)
                centre = (lo + hi) / 2
                if tick:
                    e.tick_ns = tick
            t0 = time.perf_counter()
            if e.fuse_frame(frame, labels, pose):
                t1 = time.perf_counter()
                row = {"read_ms": read_ms, "fuse_ms": 1000 * (t1 - t0), **{f"gpu_{k}_ms": v for k, v in e.gpu_ms.items()}}
                pm = gpu.pose_mm(pose)
                e.raycast(pm)
                row["gpu_raycast_ms"] = e.gpu_ms["raycast"]
                e.icp(pm, pm, centre)
                row["gpu_icp_ms"] = e.gpu_ms["icp"]
                row["raycast_icp_ms"] = 1000 * (time.perf_counter() - t1)
                rows.append(row)
                if len(rows) == WARM and tick is None:
                    e.calibrate_ticks()
                    tick = e.tick_ns
            t_read = time.perf_counter()
        t = time.perf_counter()
        v, tri, _ = e.mesh(4)
        mesh_s = time.perf_counter() - t
        warm = rows[WARM:]
        med = {k: round(float(np.median([r[k] for r in warm])), 3) for k in warm[0]}
        prep = sum(med.get(f"gpu_{k}_ms", 0) for k in ("prep_clean", "prep_erode", "prep_finish", "prep_remap", "maps"))
        usage = bench.gpu_usage() or {}
        bricks, full, outside = e.counts()
        passes.append({"frames": len(rows), "median": med, "gpu_prep_and_maps_ms": round(prep, 3), "mesh_s": round(mesh_s, 3),
                       "triangles": int(len(tri)), "pool_bricks": bricks, "pool_full_refused": full, "outside": outside,
                       "gpu_memory_mib": usage.get("memory_mib"), "max_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
                       "elapsed_s": round(time.time() - start, 1)})
        log(json.dumps(passes[-1]))
        e.close()
        if time.time() - start >= seconds:
            break
    return {"device_tick_ns": tick, "capacity": capacity, "passes": passes}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = lambda name: sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None
    if args[0] == "dump":
        dump(args[1], args[2])
    elif args[0] == "slice":
        values = {opt("--seconds"), opt("--capacity"), opt("--out")}
        report = slice_run(args[1], float(opt("--seconds") or 0), int(opt("--capacity") or gpu.CAPACITY),
                           "cpu" if "--cpu" in sys.argv else "gpu")
        if opt("--out"):
            Path(opt("--out")).write_text(json.dumps(report, indent=1))
    elif args[0] == "run":
        values = {opt("--frames"), opt("--out")}
        names = [a for a in args[2:] if a not in values] or list(ENGINES)
        run(args[1], names, int(opt("--frames")) if opt("--frames") else None, opt("--out"))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
