"""Point-cloud export from a raw depth map plus colour.

    python3 -m tribuutscan.pointcloud                    # grab a frame set now
    python3 -m tribuutscan.pointcloud captures/<stamp>   # use a saved set

The camera model (depth law, intrinsics, colour registration) comes from calib.CAMERA: the board
calibration (captures/calibration.json), else the scanner's factory calibration (captures/factory.json,
kept whenever the scanner is opened), else ESTIMATES (hand-measured depth fit, guessed field of view,
colour registration eyeballed from one frame): metric only with one of the first two.
Output units are millimetres, x right, y up, z away from the scanner.
"""
import json
import sys
from pathlib import Path

import numpy as np

from . import calib
from . import frames as fr
from .calib import CAMERA


def window_cm():
    """The active depth mode's window (cm): beyond it depth is 0."""
    return tuple(z / 10 for z in CAMERA.window_mm)


def color_source(u, v, w, h, z_mm=None):
    """Colour-image pixel (x, y) (rounded, clipped) for depth pixel (u, v), at distance z_mm if known."""
    x, y = CAMERA.colour_px(u, v, z_mm)
    return np.clip(np.round(x).astype(int), 0, w - 1), np.clip(np.round(y).astype(int), 0, h - 1)


def register_color(color, depth=None):
    """Resample colour onto the depth pixel grid, through each pixel's own distance when the raw
    `depth` is given (the cameras sit apart, so the match moves with distance), else at DEFAULT_Z_MM."""
    vv, uu = np.mgrid[0:CAMERA.height, 0:CAMERA.width]
    z = None if depth is None else np.where(depth > 0, CAMERA.depth_mm(np.maximum(depth, 1)), calib.DEFAULT_Z_MM)
    src_x, src_y = color_source(uu, vv, color.shape[1], color.shape[0], z)
    return color[src_y, src_x]


def depth_to_mm(v):
    return CAMERA.depth_mm(v)


def to_points(depth, rgb):
    mask = depth > 0
    v, u = np.nonzero(mask)
    xyz = CAMERA.backproject(u, v, depth_to_mm(depth[v, u]))
    xyz[:, 1] *= -1  # y up for this export
    return xyz.astype(np.float32), rgb[v, u]


def write_ply(path, xyz, rgb, normals=None, seen=None):
    """Binary PLY with colour; optional per-point normals and `seen` count (frames that saw it)."""
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    if normals is not None:
        fields += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    if seen is not None:
        fields += [("seen", "<u2")]
    rec = np.empty(len(xyz), dtype=fields)
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["red"], rec["green"], rec["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    if normals is not None:
        rec["nx"], rec["ny"], rec["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    if seen is not None:
        rec["seen"] = np.minimum(seen, 65535)
    types = {"<f4": "float", "u1": "uchar", "<u2": "ushort"}
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(rec)}\n"
        + "".join(f"property {types[t]} {name}\n" for name, t in fields)
        + "end_header\n"
    )
    if hasattr(path, "write"):  # a file object, e.g. io.BytesIO for a download
        path.write(header.encode() + rec.tobytes())
        return
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(rec.tobytes())


def main():
    if len(sys.argv) > 1:
        directory = Path(sys.argv[1])
        data = np.load(directory / "frames.npz")
        frames = {k: data[k] for k in data.files}
        info = directory / "info.json"
        calib.use_mode(json.loads(info.read_text()).get("depth_mode", calib.CLOSE) if info.exists() else calib.CLOSE)
    else:
        frames = fr.grab()
        directory = fr.save(frames)
    xyz, rgb = to_points(frames["depth"], register_color(frames["color"], frames["depth"]))
    out = directory / "cloud.ply"
    write_ply(out, xyz, rgb)
    print(f"{len(xyz)} points -> {out}")
    if len(xyz):
        print("z range mm: %.0f .. %.0f" % (xyz[:, 2].min(), xyz[:, 2].max()))


if __name__ == "__main__":
    main()
