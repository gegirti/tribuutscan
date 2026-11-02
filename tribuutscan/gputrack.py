"""Tracking on the GPU: each frame matched against a ray cast of the scan's own volume (KinectFusion's
frame-to-model tracking), with track.solve's contract.

The frame's piece (cloud.Scanner's centre piece, uncleaned: depth edges and steep faces hold flat
objects) becomes vertex and normal maps on the GPU. The volume (gpu_wgpu.Engine) holds the merged frames,
fused as for the mesh (cleaned, object and base labels); it is ray cast from the predicted pose, and each
frame pixel is matched to the model pixel it projects to. The GPU sums the point-to-plane normal
equations (gpu.icp_sums: Tukey, reach, normal gate, the base left out of the model by the kept plane),
the CPU adds the marker terms and the motion prior and takes the step (track.pose_terms,
track.gauss_newton): the same solve, levels and info as track.solve, so the Scanner's gates stay.

Weights: a pixel counts by its footprint, z^2 / (fx fy) mm^2 for every sampled pixel, over the level's
voxel area, so the shape weighs what track.solve's voxel-downsampled points weighed against the
markers and the prior.
"""
import os
import time

import numpy as np

from . import gpu
from . import track

LEVELS = ((0.006, 0.03), (0.003, 0.012), (0.002, 0.006))  # fuse.ICP_LEVELS: (voxel, reach), metres
STRIDES = (4, 2, 1)  # frame pixels taking part per level (every n-th each way)
ITERATIONS = (3, 3, 8)  # ICP passes per level at most (each stops early once converged). track.ITERATIONS
    # (5, 5, 8) took 10 passes a frame; (3, 3, 8) the same ATE (synth 0.35 -> 0.37, synthhard 0.33 -> 0.28 mm),
    # (2, 2, 6) lost synth (2.4 mm)
NORMAL_STEP = 2  # frame normals from neighbours this many pixels away (synthfp: 2, 3 or 4 px the same)
# Off, measured (synthfp: the sensor's fixed per-pixel error, against which tracking drifts ~0.7 mm in its
# first frames and the cloud's p90 to the true surface is 0.70 mm, the CPU's 0.33; without the fixed
# error it is 0.18 mm, better than the CPU): depth blurred (same-surface mean) in the frame and the
# fused frames, 3 px: p90 0.63; model normals from ray-cast hits 8 px apart (MODEL_NORMAL_PX): p90 0.42,
# but the noisy box's error 0.37 -> 1.74 mm (rounded edges); TSDF gradients over 2-3 mm lost tracking.
BLUR_PX = 0
MODEL_NORMAL_PX = 0
HALF_MM = 400.0  # the volume reaches this far from the first frame's piece each way (a dense brick index)
RAY_MARGIN_MM, RAY_MARGIN_PX = 30.0, 16  # the ray cast covers the piece's pixels and this much around them
    # (the coarse level's reach seen at the piece's nearest point): the rest of the view is never matched.
    # Hand scan (long range, the object a sixth of the view): ray cast 11.9 ms over the whole image
TRUNC_VOXELS = 6.0  # 3 mm: the mesh's 1.5 mm (fusion.TRUNC_VOXELS) gave worse tracking on the thin plate (ATE 36 mm; 2 mm: 19, 3 mm: 6.8)
KEYFRAME_MM = 2.0  # keyframe points (Optimise): one per cell of this size, as track.levels' finest level
PREVIEW_STEP = 2  # the live view (preview): every 2nd pixel each way, 640 x 400


def kind():
    """The WebGPU adapter to track on ("gpu", or "cpu" with TRIBUUTSCAN_GPU=cpu), or None (no wgpu, no
    adapter, or TRIBUUTSCAN_GPU=0)."""
    choice = os.environ.get("TRIBUUTSCAN_GPU", "auto").lower()
    if choice in ("0", "no", "off"):
        return None
    try:
        from . import gpu_wgpu
        found = gpu_wgpu.adapters()
    except Exception:
        return None
    want = "cpu" if choice == "cpu" else "gpu"
    return want if want in found else None


def grid_points(xyz, cell_mm=KEYFRAME_MM):
    """One point per occupied cell (the mean of its points): a keyframe's points for Optimise."""
    keys = np.floor(xyz / cell_mm).astype(np.int64)
    _, inv, count = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    out = np.zeros((len(count), 3))
    for k in range(3):
        out[:, k] = np.bincount(inv, xyz[:, k], len(count))
    return out / count[:, None]


class Tracker:
    """The volume and the per-frame solve. `camera` (calib.Camera) in the recording's depth mode, frames
    of `size` (w, h); the volume is centred on `centre_mm` (world)."""

    def __init__(self, camera, size, centre_mm, adapter="gpu", far_mm=None):
        from . import fusion, gpu_wgpu
        self.pinhole = fusion.Pinhole(camera)
        self.K = self.pinhole.K
        centre = np.asarray(centre_mm, float)
        layout = gpu.Layout(centre - HALF_MM, centre + HALF_MM, fusion.VOXEL_MM, TRUNC_VOXELS)
        far = camera.window_mm[1] + 20 if far_mm is None else far_mm
        self.engine = gpu_wgpu.Engine(layout, self.K, size, adapter, far, timestamps=False)  # wall clock below
        self.engine.set_camera(camera, self.pinhole.index)
        self.timing = {"source": 0.0, "raycast": 0.0, "icp": 0.0, "fuse": 0.0, "preview": 0.0}  # wall s, readbacks in
        self.frames, self.passes = 0, 0  # frames solved, ICP passes run
        self.rect = None  # the ray cast's pixels (source)
        self.ref = None  # the pose (mm) the last ray cast was made from
        self.unpacked_frame = None  # the frame whose bytes the engine unpacked last (its depth is on the GPU)

    def source(self, frame, pix, step, near_mm=None):
        """This frame's piece (pixels `pix`, sampled every `step`-th each way) as the maps to track; with
        `near_mm` (the piece's nearest distance) the ray casts cover only the piece's part of the view. A
        frame with the scanner's bytes ("depth_raw": depth.LazyFrame) is unpacked here (Engine.unpack)."""
        h, w = frame["shape"] if "shape" in frame else frame["depth"].shape
        keep = np.zeros(((h + step - 1) // step, (w + step - 1) // step), bool)
        keep[pix[:, 1] // step, pix[:, 0] // step] = True
        self.rect = None
        if near_mm is not None and len(pix):
            m = int(RAY_MARGIN_MM * self.K[0, 0] / max(near_mm, 1.0)) + RAY_MARGIN_PX + step
            lo, hi = pix.min(0) - m, pix.max(0) + m + 1
            self.rect = (max(lo[0], 0), max(lo[1], 0), min(hi[0], w), min(hi[1], h))
        t0 = time.perf_counter()
        if "depth_raw" in frame:
            self.engine.unpack(frame["depth_raw"], frame["bytesperline"], frame.get("cleared"))
            self.engine.track_source(None, keep, step, NORMAL_STEP, BLUR_PX)
            self.unpacked_frame = frame
        else:
            self.engine.track_source(frame["depth"], keep, step, NORMAL_STEP, BLUR_PX)
            self.unpacked_frame = None
        self.timing["source"] += time.perf_counter() - t0

    def depth(self, frame):
        """The full depth of the frame source() unpacked last (read back), or None for any other frame."""
        return self.engine.unpacked() if frame is self.unpacked_frame else None

    def solve(self, init, prior=None, marker_pairs=None, radius_mm=60.0, centre_mm=(0, 0, 0), base=None,
              marker_sigma_mm=track.MARKER_SIGMA_MM):
        """Refine `init` (camera -> world, metres) against the volume, ray cast from `init`; prior, marker
        pairs (camera mm, world mm) and the returned (transform, info) as track.solve. `centre_mm`: the
        object's centre (world), which the GPU's sums turn about; `base`: (n, c) with n . x + c > 0
        above the base (world mm), or None."""
        e = self.engine
        ref = self.ref = gpu.pose_mm(init)
        t0 = time.perf_counter()
        e.raycast(ref, normal_px=MODEL_NORMAL_PX, rect=self.rect)
        e.sync()
        t1 = time.perf_counter()
        self.timing["raycast"] += t1 - t0
        self.frames += 1
        c = np.asarray(centre_mm, float)
        back = np.eye(6)  # twist about the centre -> about the origin (as track.solve): J0 = Jc back
        back[3:, :3] = -track.hat(c)
        plane = (0.0, 0.0, 0.0, 1.0) if base is None else (*base[0], base[1])
        t, w_prior, last = init.copy(), track.prior_weights(prior), None
        fx, fy = self.K[0, 0], self.K[1, 1]
        for level, ((voxel, reach), stride, iters) in enumerate(zip(LEVELS, STRIDES, ITERATIONS)):
            sigmas = track.MARKER_GATE_SIGMAS[0] if level < len(LEVELS) - 1 else track.MARKER_GATE_SIGMAS[1]
            area = stride ** 2 / (fx * fy * (voxel * 1000) ** 2)
            for _ in range(iters):
                s = e.icp(gpu.pose_mm(t), ref, c, area=area, reach_mm=reach * 1000, stride=stride, base=plane)
                self.passes += 1
                if s[31] < 10:
                    self.timing["icp"] += time.perf_counter() - t1
                    return t, summary(last, radius_mm)
                a = np.zeros((6, 6))
                a[np.triu_indices(6)] = s[:21]
                a = a + np.triu(a, 1).T
                h, g = back.T @ a @ back, back.T @ s[21:27]
                used, mres = track.pose_terms(t, h, g, marker_pairs, sigmas * marker_sigma_mm, marker_sigma_mm,
                                              prior, w_prior)
                last = (a, s, used, mres)
                t, done = track.gauss_newton(t, h, g)
                if done:
                    break
        self.timing["icp"] += time.perf_counter() - t1
        return t, summary(last, radius_mm)

    def preview(self, base=None, min_weight=4.0):
        """The live view of the frame just solved (gpu_wgpu.Engine.preview: its piece against the model as
        ray cast for it), uint8 at 1/PREVIEW_STEP; `base` as solve's. None before any solve."""
        if self.ref is None:
            return None
        t0 = time.perf_counter()
        plane = (0.0, 0.0, 0.0, 1.0) if base is None else (*base[0], base[1])
        out = self.engine.preview(self.ref, plane, min_weight, PREVIEW_STEP)
        self.timing["preview"] += time.perf_counter() - t0
        return out

    def fuse(self, frame, labels, pose_m):
        """Fuse a merged frame (its 1/8 labels; pose camera -> world, metres) into the volume."""
        t0 = time.perf_counter()
        fused = self.engine.fuse_frame(frame, labels, pose_m, blur_px=BLUR_PX, depth_ready=frame is self.unpacked_frame)
        self.engine.sync()
        self.timing["fuse"] += time.perf_counter() - t0
        return fused

    def report(self):
        """ms per solved frame for each step (wall clock, readbacks included), ICP passes per frame."""
        n = max(self.frames, 1)
        return {**{k: round(1000 * v / n, 2) for k, v in self.timing.items()}, "passes": round(self.passes / n, 1),
                "device": self.engine.device_name}

    def close(self):
        self.engine.close()


def summary(last, radius_mm):
    """track.solve's info from the last pass's sums."""
    if last is None:
        return track.summary(None, None, 0.0, 0, None, radius_mm)
    a, s, used, mres = last
    rmse = float(np.sqrt(s[29] / s[27])) if s[27] > 0 else None
    return track.summary(a, rmse, float(s[31] / max(s[30], 1)), used, mres, radius_mm)
