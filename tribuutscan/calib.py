"""The camera model in one place: depth intrinsics, depth law, depth window, colour <-> depth pixels.

Three sources, the first found wins (`CAMERA.source`):
- "board": captures/calibration.json, written by calibrate.py from a printed board
- "factory": captures/factory.json, the scanner's own calibration, kept by frames.open_depth each time it
  opens the scanner (`note_factory`)
- "estimate": the ESTIMATES below; `CAMERA.estimate` is True, so output can say the numbers are estimates
TRIBUUTSCAN_CAMERA=board|factory|estimate forces one (if it exists; else the estimates).
The depth law and window depend on the scanner's depth mode (vendor.py): `use_mode` picks them. The
top-level law is close range; other modes come from a "modes" block (a board calibration gets one from
the disparity offsets in factory.json), else from LONG_B_SHIFT (`CAMERA.law_estimate` is True then).
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

CALIBRATION = Path(__file__).resolve().parent.parent / "captures" / "calibration.json"
FACTORY = CALIBRATION.parent / "factory.json"
DEFAULT_Z_MM = 250.0  # distance assumed for colour <-> depth pixels when it is not known
CLOSE, LONG = "Close_range_mode Default", "Long_range_mode Default"
# A raw depth unit (Y10 scale) is a quarter pixel of disparity, and each depth mode is the same law with
# another disparity offset: b = 4 x offset. Both units read so far (ours and one other, 2026-11-01) have
# 218 in close and 100 in long range; without this scanner's own offsets, long range is the close law with
# b - 472 (it was -471.3 from one flat surface, and a hand-measured 50 cm read 175.7, which -472 puts at
# 501.8 mm). The window is what raw 1023..1 then spans (~20-72 cm).
DEFAULT_OFFSETS = {CLOSE: 218.0, LONG: 100.0}
LONG_B_SHIFT = 4 * (DEFAULT_OFFSETS[LONG] - DEFAULT_OFFSETS[CLOSE])
COLOUR_STRETCH = 800 / 720  # the 1280x800 colour mode is the 1280x720 picture stretched to 800 rows
ESTIMATES = {
    "estimate": True,
    "size": [1280, 800],
    "depth": {"hfov_deg": 60.0},  # a guess; principal point assumed at the image centre
    "law": {"a_mm": 316340.0, "b": 998.0},  # z_mm = a / (v + b), fitted on hand-held 20/25/30 cm readings
    "window_mm": [170.0, 320.0],  # hypothesis from hand-held wall readings
    "colour_from_depth": {"ax": 1.134, "bx": -30.6, "ay": 1.0, "by": 5.0},  # eyeballed from one frame
}


def distort(x, y, d):
    """Normalised camera coordinates through the OpenCV lens model (k1, k2, p1, p2, k3, and the rational
    model's k4, k5, k6 when given: the factory colour model uses them)."""
    k1, k2, p1, p2, k3, k4, k5, k6 = (list(d) + [0.0] * 8)[:8]
    r2 = x * x + y * y
    radial = (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3) / (1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3)
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
        self.source = spec.get("source", "estimate" if self.estimate else "board")
        self.width, self.height = spec["size"]
        d = spec["depth"]
        if "fx" in d:
            self.fx, self.fy, self.cx, self.cy = d["fx"], d["fy"], d["cx"], d["cy"]
        else:
            self.fx = self.fy = (self.width / 2) / np.tan(np.radians(d["hfov_deg"]) / 2)
            self.cx, self.cy = self.width / 2, self.height / 2
        self.dist = np.array(d.get("dist", []), float)
        self.base_law = (spec["law"]["a_mm"], spec["law"]["b"], tuple(spec["window_mm"]))  # close range
        self.set_mode(CLOSE)
        self.reg = spec["colour_from_depth"]
        colour = spec.get("colour")
        self.colour = None if colour is None else {
            "K": np.array(colour["K"], float), "dist": np.array(colour["dist"], float),
            "R": np.array(colour["R"], float), "T": np.array(colour["T_mm"], float)}
        self._grid = None  # undistorted normalised coordinates of every depth pixel (when distorted)

    def set_mode(self, name):
        """Use the depth law and window of the scanner's depth mode `name`."""
        a, b, window = self.base_law
        own = self.spec.get("modes", {}).get(name)
        if own:
            a, b, window = own["law"]["a_mm"], own["law"]["b"], tuple(own["window_mm"])
            estimate = bool(own.get("estimate", False))
        elif name.startswith("Long_range"):
            b += LONG_B_SHIFT
            window = tuple(law_window(a, b))
            estimate = True
        else:
            estimate = self.estimate or name != CLOSE  # Interleave / Narrow_View: same law, unverified
        self.mode, self.a, self.b, self.window_mm, self.law_estimate = name, a, b, window, estimate

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

    def fingerprint(self):
        """Short hash of what turns raw frames into points (size, depth lens, colour model, the active
        mode's law): a scan is fused again only under the model it was tracked with."""
        blob = json.dumps({**{k: self.spec.get(k) for k in ("size", "depth", "colour")}, "law": [self.a, self.b]},
                          sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]


def law_window(a, b):
    """What raw 1023 (near) .. 1 (far) spans under z = a / (v + b), mm."""
    return [a / (1023 + b), a / (1 + b)]


def mode_block(a, b, estimate=False):
    return {"law": {"a_mm": a, "b": b}, "window_mm": law_window(a, b), "estimate": estimate}


def affine_at(k_depth, k_colour, rot, trans, z_mm):
    """The old affine depth->colour pixel mapping, refitted for distance z_mm (for code paths that
    still use it). The real mapping depends on distance (parallax), which Camera handles."""
    vv, uu = np.mgrid[100:700:50, 100:1180:60]
    u, v = uu.ravel().astype(float), vv.ravel().astype(float)
    pts = np.stack([(u - k_depth[0, 2]) / k_depth[0, 0] * z_mm, (v - k_depth[1, 2]) / k_depth[1, 1] * z_mm,
                    np.full(len(u), z_mm)], 1)
    c = pts @ np.asarray(rot).T + np.asarray(trans).ravel()
    x = c[:, 0] / c[:, 2] * k_colour[0, 0] + k_colour[0, 2]
    y = c[:, 1] / c[:, 2] * k_colour[1, 1] + k_colour[1, 2]
    ax, bx = np.polyfit(x, u, 1)  # depth u = ax * colour x + bx (the old REG form)
    ay, by = np.polyfit(y, v, 1)
    return {"ax": float(ax), "bx": float(bx), "ay": float(ay), "by": float(by)}


def from_factory(fac, mode, offsets=None):
    """A spec in calibration.json's format from the scanner's factory calibration (as
    vendor.Channel.factory_calibration returns it, read in depth mode `mode`; `offsets` are the disparity
    offsets of modes read before). The factory frame is our depth AND colour images mirrored left-right
    (the IR stream's orientation), with colour = R depth + T. Checked on our unit (2026-11-01) on the
    board calibration's views: colour lands on depth within 0.88 px median, a 24.00 mm square measures
    24.06 mm, and the close law is within 0.5 mm of the board's."""
    w, h = ESTIMATES["size"]
    dc = fac["depth_calib"]
    offsets = {**(offsets or {}), mode: dc["disp_offset"]}
    a = 4 * dc["unit"] * dc["baseline"] * dc["focal"]
    b = 4 * offsets.get(CLOSE, DEFAULT_OFFSETS[CLOSE])
    modes = {m: mode_block(a, 4 * off) for m, off in offsets.items() if m != CLOSE}
    if CLOSE not in offsets:  # close range not read from this scanner yet
        modes[CLOSE] = mode_block(a, b, estimate=True)
    cam = next((c for c in fac["cameras"] if (c["depth"]["w"], c["depth"]["h"]) == (w, h)), fac["cameras"][0])
    d = cam["depth"]
    m = np.diag([-1.0, 1.0, 1.0])
    k_depth = np.array([[d["fx"], 0, w - 1 - d["cx"]], [0, d["fy"], d["cy"]], [0, 0, 1]])
    spec = {"estimate": False, "source": "factory", "size": [w, h],
            "depth": {"fx": d["fx"], "fy": d["fy"], "cx": w - 1 - d["cx"], "cy": d["cy"], "dist": []},  # rectified
            "law": {"a_mm": a, "b": b}, "window_mm": law_window(a, b), "modes": modes,
            "colour_from_depth": ESTIMATES["colour_from_depth"], "disp_offsets": offsets, "factory": fac}
    # the colour record that serves 1280x720 colour: its crop places that picture in the record's frame
    prof = next((p for p in fac["profiles"] if tuple(p["colour"]) == (1280, 720) and tuple(p["depth"]) == (w, h)
                 and p["record"] < len(fac["cameras"])), None)
    if prof is not None:
        rec = fac["cameras"][prof["record"]]
        c, cd, s = rec["colour"], rec["colour_dist"], prof["scale"]
        cx720, cy720 = c["cx"] * s + prof["crop"][0], c["cy"] * s + prof["crop"][1]
        k_colour = np.array([[c["fx"] * s, 0, w - 1 - cx720],
                             [0, c["fy"] * s * COLOUR_STRETCH, (cy720 + 0.5) * COLOUR_STRETCH - 0.5], [0, 0, 1]])
        rot, trans = m @ np.reshape(rec["R"], (3, 3)) @ m, m @ np.asarray(rec["T_mm"])
        spec["colour"] = {"K": k_colour.tolist(),
                          "dist": [cd["k1"], cd["k2"], cd["p1"], -cd["p2"], cd["k3"], cd["k4"], cd["k5"], cd["k6"]],
                          "R": rot.tolist(), "T_mm": trans.tolist()}  # mirroring flips p2's sign
        spec["colour_from_depth"] = affine_at(k_depth, k_colour, rot, trans, DEFAULT_Z_MM)
    return spec


def note_factory(fac, mode, path=None):
    """Keep the factory calibration read in depth mode `mode` in captures/factory.json, with the disparity
    offset of every mode read so far (forgotten when another scanner turns up), and reload CAMERA from
    the chosen source. The caller then picks the mode (`use_mode`)."""
    path = Path(path or FACTORY)
    old = json.loads(path.read_text()) if path.exists() else {}
    dc, old_dc = fac["depth_calib"], old.get("factory", {}).get("depth_calib", {})
    same = all(old_dc.get(k) == dc[k] for k in ("baseline", "focal", "unit"))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")  # replaced in one step: a scan worker may be loading it
    tmp.write_text(json.dumps(from_factory(fac, mode, old.get("disp_offsets") if same else None), indent=2))
    tmp.replace(path)
    use(chosen_spec(factory_path=path))


def chosen_spec(board_path=None, factory_path=None):
    """The spec of the first source found (board, factory), or ESTIMATES; TRIBUUTSCAN_CAMERA forces one.
    A board calibration takes the disparity offsets read from the scanner for its other modes."""
    paths = {"board": Path(board_path or CALIBRATION), "factory": Path(factory_path or FACTORY)}
    forced = os.environ.get("TRIBUUTSCAN_CAMERA", "").lower()
    if forced and forced not in (*paths, "estimate"):
        raise ValueError(f"TRIBUUTSCAN_CAMERA={forced}: use board, factory or estimate")
    for source in [forced] if forced else ["board", "factory"]:
        if source in paths and paths[source].exists():
            spec = {**json.loads(paths[source].read_text()), "source": source}
            if source == "board" and paths["factory"].exists():
                offsets = json.loads(paths["factory"].read_text()).get("disp_offsets", {})
                if CLOSE in offsets:
                    a, b = spec["law"]["a_mm"], spec["law"]["b"]
                    spec["modes"] = {**{m: mode_block(a, b + 4 * (off - offsets[CLOSE]))
                                        for m, off in offsets.items() if m != CLOSE}, **spec.get("modes", {})}
            return spec
    return ESTIMATES


def use(spec):
    """Switch the shared camera model in place (every module holds this same CAMERA object). Synthetic
    data is made and read with ESTIMATES, a pinhole camera, whatever is calibrated."""
    CAMERA.__init__(spec)


def use_mode(name):
    """Switch the shared camera model to the law and window of depth mode `name`."""
    CAMERA.set_mode(name)


def load(path=None):
    """The camera from `path` (a calibration.json-format file), or from the chosen source."""
    return Camera(json.loads(Path(path).read_text()) if path else chosen_spec())


CAMERA = load()
