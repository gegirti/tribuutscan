"""The camera model in one place: depth intrinsics, depth law, depth window, colour <-> depth pixels.

Loads captures/calibration.json (to be written by calibrate.py) when it exists. Otherwise it uses the
ESTIMATES below and `CAMERA.estimate` is True, so output can say the numbers are estimates.
"""
import json
from pathlib import Path

import numpy as np

CALIBRATION = Path(__file__).resolve().parent.parent / "captures" / "calibration.json"
DEFAULT_Z_MM = 250.0  # distance assumed for colour <-> depth pixels when it is not known
ESTIMATES = {
    "estimate": True,
    "size": [1280, 800],
    "depth": {"hfov_deg": 60.0},  # a guess; principal point assumed at the image centre
    "law": {"a_mm": 316340.0, "b": 998.0},  # z_mm = a / (v + b), fitted on hand-held 20/25/30 cm readings
    "window_mm": [170.0, 320.0],  # hypothesis from hand-held wall readings
    "colour_from_depth": {"ax": 1.134, "bx": -30.6, "ay": 1.0, "by": 5.0},  # eyeballed from one frame
}


def distort(x, y, d):
    """Normalised camera coordinates through the OpenCV lens model (k1, k2, p1, p2, k3)."""
    k1, k2, p1, p2, k3 = (list(d) + [0.0] * 5)[:5]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    return (x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
            y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)


def undistort(xd, yd, d, iterations=6):
    """Inverse of distort() by fixed-point iteration (enough for scanner lenses)."""
    x, y = xd, yd
    for _ in range(iterations):
        dx, dy = distort(x, y, d)
        x, y = x + (xd - dx), y + (yd - dy)
    return x, y


class Camera:
    """Depth camera intrinsics (pixels; x right, y down, z forward), depth law and colour mapping.

    With a calibration that has a "colour" block, colour <-> depth pixels go through 3D (the two
    cameras sit apart, so the offset changes with distance: pass the distance where it is known);
    the estimates use the old distance-independent affine mapping.
    """

    def __init__(self, spec):
        self.spec = spec
        self.estimate = bool(spec.get("estimate", False))
        self.width, self.height = spec["size"]
        d = spec["depth"]
        if "fx" in d:
            self.fx, self.fy, self.cx, self.cy = d["fx"], d["fy"], d["cx"], d["cy"]
        else:
            self.fx = self.fy = (self.width / 2) / np.tan(np.radians(d["hfov_deg"]) / 2)
            self.cx, self.cy = self.width / 2, self.height / 2
        self.dist = np.array(d.get("dist", []), float)
        self.a, self.b = spec["law"]["a_mm"], spec["law"]["b"]
        self.window_mm = tuple(spec["window_mm"])
        self.reg = spec["colour_from_depth"]
        colour = spec.get("colour")
        self.colour = None if colour is None else {
            "K": np.array(colour["K"], float), "dist": np.array(colour["dist"], float),
            "R": np.array(colour["R"], float), "T": np.array(colour["T_mm"], float)}
        self._grid = None  # undistorted normalised coordinates of every depth pixel (when distorted)

    def depth_mm(self, raw):
        """Distance along the camera axis (mm) for raw depth values (nearer = larger)."""
        return self.a / (np.asarray(raw, np.float64) + self.b)

    def raw_from_mm(self, z_mm):
        return self.a / np.asarray(z_mm, np.float64) - self.b

    def normalised(self, u, v):
        """Undistorted normalised coordinates (x/z, y/z) of depth pixels (u, v)."""
        x = (np.asarray(u, np.float64) - self.cx) / self.fx
        y = (np.asarray(v, np.float64) - self.cy) / self.fy
        if not np.any(self.dist):
            return x, y
        u_arr, v_arr = np.asarray(u), np.asarray(v)
        if np.issubdtype(u_arr.dtype, np.integer) and np.issubdtype(v_arr.dtype, np.integer):
            if self._grid is None:  # once: every pixel of the depth image
                vv, uu = np.mgrid[0:self.height, 0:self.width]
                self._grid = undistort((uu - self.cx) / self.fx, (vv - self.cy) / self.fy, self.dist)
            return self._grid[0][v_arr, u_arr], self._grid[1][v_arr, u_arr]
        return undistort(x, y, self.dist)

    def backproject(self, u, v, z_mm):
        """(N, 3) camera points (mm) for pixels (u, v) at distances z_mm."""
        x, y = self.normalised(u, v)
        z = np.asarray(z_mm, np.float64)
        return np.stack([x * z, y * z, z * np.ones_like(x)], -1)

    def colour_px(self, u, v, z_mm=None):
        """Colour-image pixel (float) seen at depth pixel (u, v), at distance z_mm when known."""
        if self.colour is None:
            r = self.reg
            return (np.asarray(u, np.float64) - r["bx"]) / r["ax"], (np.asarray(v, np.float64) - r["by"]) / r["ay"]
        c = self.colour
        p = self.backproject(u, v, DEFAULT_Z_MM if z_mm is None else z_mm) @ c["R"].T + c["T"]
        xd, yd = distort(p[..., 0] / p[..., 2], p[..., 1] / p[..., 2], c["dist"])
        return xd * c["K"][0, 0] + c["K"][0, 2], yd * c["K"][1, 1] + c["K"][1, 2]

    def depth_px(self, x, y, z_mm=None):
        """Depth-image pixel (float) of colour pixel (x, y), at distance z_mm when known."""
        if self.colour is None:
            r = self.reg
            return r["ax"] * np.asarray(x, np.float64) + r["bx"], r["ay"] * np.asarray(y, np.float64) + r["by"]
        c = self.colour
        xn, yn = undistort((np.asarray(x, np.float64) - c["K"][0, 2]) / c["K"][0, 0],
                           (np.asarray(y, np.float64) - c["K"][1, 2]) / c["K"][1, 1], c["dist"])
        z = DEFAULT_Z_MM if z_mm is None else np.asarray(z_mm, np.float64)
        p = (np.stack([xn * z, yn * z, z * np.ones_like(xn)], -1) - c["T"]) @ c["R"]  # colour -> depth camera
        xd, yd = distort(p[..., 0] / p[..., 2], p[..., 1] / p[..., 2], self.dist)
        return xd * self.fx + self.cx, yd * self.fy + self.cy

    def intrinsic_matrix(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])


def use(spec):
    """Switch the shared camera model in place (every module holds this same CAMERA object). Synthetic
    data is made and read with ESTIMATES, a pinhole camera, whatever is calibrated."""
    CAMERA.__init__(spec)


def load(path=CALIBRATION):
    path = Path(path)
    return Camera(json.loads(path.read_text()) if path.exists() else ESTIMATES)


CAMERA = load()
