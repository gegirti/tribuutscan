"""Rough point-cloud export from a raw depth map plus colour. NOT metrically calibrated.

    python3 -m ferret.pointcloud                    # grab a frame set now
    python3 -m ferret.pointcloud captures/<stamp>   # use a saved set

Assumptions, all unverified:
  * raw depth -> distance uses the hand-measured fit  z_cm = A / (v + B)  (see README);
  * pinhole camera with a guessed horizontal field of view and centred principal point;
  * colour -> depth registration is a rough affine ESTIMATE read off box edges in one frame
    (REG below): depth_x = AX * colour_x + BX, depth_y = colour_y + BY. Not a calibration.
Output units are millimetres, x right, y up, z away from the scanner.
"""
import sys
from pathlib import Path

import numpy as np

from . import frames as fr

A, B = 31634.0, 998.0  # z_cm = A / (v + B), fitted on 20/25/30 cm readings
HFOV_DEG = 60.0  # guess
REG = {"ax": 1.134, "bx": -30.6, "ay": 1.0, "by": 5.0}  # eyeballed from one frame, see docstring


def register_color(color, reg=REG):
    """Resample colour onto the depth pixel grid using the rough affine estimate."""
    h, w = color.shape[:2]
    u = np.arange(w)
    v = np.arange(h)
    src_x = np.clip(np.round((u - reg["bx"]) / reg["ax"]).astype(int), 0, w - 1)
    src_y = np.clip(np.round((v - reg["by"]) / reg["ay"]).astype(int), 0, h - 1)
    return color[src_y][:, src_x]


def depth_to_mm(v):
    return 10.0 * A / (v.astype(np.float64) + B)


def to_points(depth, rgb, hfov_deg=HFOV_DEG):
    h, w = depth.shape
    mask = depth > 0
    f = (w / 2) / np.tan(np.radians(hfov_deg) / 2)
    v, u = np.nonzero(mask)
    z = depth_to_mm(depth[v, u])
    x = (u - w / 2) * z / f
    y = -(v - h / 2) * z / f
    return np.stack([x, y, z], 1).astype(np.float32), rgb[v, u]


def write_ply(path, xyz, rgb):
    rec = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(rec)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(rec.tobytes())


def main():
    if len(sys.argv) > 1:
        directory = Path(sys.argv[1])
        data = np.load(directory / "frames.npz")
        frames = {k: data[k] for k in data.files}
    else:
        frames = fr.grab()
        directory = fr.save(frames)
    xyz, rgb = to_points(frames["depth"], register_color(frames["color"]))
    out = directory / "cloud.ply"
    write_ply(out, xyz, rgb)
    print(f"{len(xyz)} points -> {out}")
    if len(xyz):
        print("z range mm: %.0f .. %.0f" % (xyz[:, 2].min(), xyz[:, 2].max()))


if __name__ == "__main__":
    main()
