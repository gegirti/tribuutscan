"""The camera model in one place: depth intrinsics, depth law, depth window, colour <-> depth pixels.

Loads captures/calibration.json (to be written by calibrate.py) when it exists. Otherwise it uses the
ESTIMATES below and `CAMERA.estimate` is True, so output can say the numbers are estimates.
"""
import json
from pathlib import Path

import numpy as np

CALIBRATION = Path(__file__).resolve().parent.parent / "captures" / "calibration.json"
ESTIMATES = {
    "estimate": True,
    "size": [1280, 800],
    "depth": {"hfov_deg": 60.0},  # a guess; principal point assumed at the image centre
    "law": {"a_mm": 316340.0, "b": 998.0},  # z_mm = a / (v + b), fitted on hand-held 20/25/30 cm readings
    "window_mm": [170.0, 320.0],  # hypothesis from hand-held wall readings
    "colour_from_depth": {"ax": 1.134, "bx": -30.6, "ay": 1.0, "by": 5.0},  # eyeballed from one frame
}


class Camera:
    """Depth camera intrinsics (pixels; x right, y down, z forward), depth law and colour mapping."""

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
        self.a, self.b = spec["law"]["a_mm"], spec["law"]["b"]
        self.window_mm = tuple(spec["window_mm"])
        self.reg = spec["colour_from_depth"]

    def depth_mm(self, raw):
        """Distance along the camera axis (mm) for raw depth values (nearer = larger)."""
        return self.a / (np.asarray(raw, np.float64) + self.b)

    def raw_from_mm(self, z_mm):
        return self.a / np.asarray(z_mm, np.float64) - self.b

    def backproject(self, u, v, z_mm):
        """(N, 3) camera points (mm) for pixels (u, v) at distances z_mm."""
        return np.stack([(u - self.cx) * z_mm / self.fx, (v - self.cy) * z_mm / self.fy, z_mm], -1)

    def colour_px(self, u, v):
        """Colour-image pixel (float) seen at depth pixel (u, v)."""
        r = self.reg
        return (np.asarray(u, np.float64) - r["bx"]) / r["ax"], (np.asarray(v, np.float64) - r["by"]) / r["ay"]

    def depth_px(self, x, y):
        """Depth-image pixel (float) of colour pixel (x, y)."""
        r = self.reg
        return r["ax"] * np.asarray(x, np.float64) + r["bx"], r["ay"] * np.asarray(y, np.float64) + r["by"]

    def intrinsic_matrix(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])


def load(path=CALIBRATION):
    path = Path(path)
    return Camera(json.loads(path.read_text()) if path.exists() else ESTIMATES)


CAMERA = load()
