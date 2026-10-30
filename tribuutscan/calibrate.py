"""Calibrate the scanner with a printed ChArUco board: the colour and IR cameras, where they sit
relative to each other, and the depth law. Writes captures/calibration.json, which calib.CAMERA then
uses instead of the estimates. OpenCV (.venv).

    .venv/bin/python -m tribuutscan.calibrate board                      # captures/calib/board.svg
    .venv/bin/python -m tribuutscan.calibrate capture [square_mm]        # live: space keeps a view, q ends
    .venv/bin/python -m tribuutscan.calibrate solve captures/calib/<stamp> [square_mm]

Print board.svg at 100% (no "fit to page"), on matte paper, fixed flat to something rigid, and
measure one square with a ruler: pass that as square_mm. Keep about 25 views at 18-31 cm, tilted
different ways, the whole board or most of it in view of both cameras. The IR image shows the
projector's dots; it is median-blurred before detection (if detection still fails, tape over the
projector and light the board with a lamp: the depth stream then stops, but IR keeps the board).
"""
import base64
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import frames as fr

BOARD_SQUARES = (11, 8)
SQUARE_MM, MARKER_MM = 24.0, 18.0  # as drawn; measure the print and pass the real square size
DICTIONARY = cv2.aruco.DICT_5X5_100
CALIB = fr.CAPTURES / "calib"
IR_BLUR = 5  # median blur against the projector's dot pattern


def board(square_mm=SQUARE_MM):
    return cv2.aruco.CharucoBoard(BOARD_SQUARES, square_mm, square_mm * MARKER_MM / SQUARE_MM,
                                  cv2.aruco.getPredefinedDictionary(DICTIONARY))


def write_board(path):
    """An A4-landscape SVG with the board at exact millimetre size (embedded 10 px/mm image)."""
    w_mm, h_mm = BOARD_SQUARES[0] * SQUARE_MM, BOARD_SQUARES[1] * SQUARE_MM
    img = board().generateImage((int(w_mm * 10), int(h_mm * 10)), marginSize=0, borderBits=1)
    png = base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode()
    x, y = (297 - w_mm) / 2, (210 - h_mm) / 2
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="297mm" height="210mm" viewBox="0 0 297 210">\n'
        f'<rect width="297" height="210" fill="white"/>\n'
        f'<image x="{x}" y="{y}" width="{w_mm}" height="{h_mm}" href="data:image/png;base64,{png}"/>\n'
        f'<text x="{x}" y="{y + h_mm + 6}" font-size="4" font-family="sans-serif">triBUUTSCAN calibration board: '
        f'{BOARD_SQUARES[0]}x{BOARD_SQUARES[1]} squares of {SQUARE_MM} mm. Print at 100%, then measure one square.</text>\n'
        f'</svg>\n')
    return path


def ir_for_detection(ir):
    return cv2.medianBlur(ir, IR_BLUR)


def detect(grey, brd):
    """(object points (N, 3) mm, image points (N, 2), corner ids (N,)) or None."""
    corners, ids, _, _ = cv2.aruco.CharucoDetector(brd).detectBoard(grey)
    if ids is None or len(ids) < 8:
        return None
    obj, img = brd.matchImagePoints(corners, ids)
    return obj.reshape(-1, 3).astype(np.float32), img.reshape(-1, 2).astype(np.float32), ids.ravel()


def load_views(directory):
    return [dict(np.load(p)) for p in sorted(Path(directory).glob("view_*.npz"))]


def plane_depth(k, dist, rvec, tvec, us, vs, square_mm=None):
    """Distance along the IR camera axis (mm) of the board plane at pixels (us, vs)."""
    rot = cv2.Rodrigues(rvec)[0]
    n, c = rot[:, 2], rot @ np.zeros(3) + tvec.ravel()  # board normal and a point on it (camera frame)
    rays = cv2.undistortPoints(np.stack([us, vs], 1).astype(np.float64).reshape(-1, 1, 2), k, dist).reshape(-1, 2)
    rays = np.c_[rays, np.ones(len(rays))]
    t = (n @ c) / (rays @ n)
    return rays[:, 2] * t


def fit_law(raw, z):
    """Fit both v = A/z + B and v = A/z^2 + B (raw depth v, distance z mm); return the better as the
    a, b of z = a / (v + b) when the first wins, with both residuals (mm) for the record."""
    out = {}
    for name, feature in (("a/z", 1 / z), ("a/z^2", 1 / z ** 2)):
        m = np.c_[feature, np.ones(len(z))]
        (a_coef, b_coef), *_ = np.linalg.lstsq(m, raw, rcond=None)
        if name == "a/z":
            z_hat = a_coef / (raw - b_coef)  # v = A/z + B  ->  z = A / (v - B)
        else:
            z_hat = np.sqrt(np.maximum(a_coef / np.maximum(raw - b_coef, 1e-9), 0))
        res = np.abs(z_hat - z)
        out[name] = {"A": float(a_coef), "B": float(b_coef), "median_mm": float(np.median(res)),
                     "p90_mm": float(np.percentile(res, 90))}
    best = min(out, key=lambda k: out[k]["median_mm"])
    return best, out


def solve(directory, square_mm=SQUARE_MM, out=None, log=print):
    """Calibrate from captured views; write calibration.json (default captures/calibration.json)."""
    brd = board(square_mm)
    views = load_views(directory)
    size = (fr.W, fr.H)
    col, irs, both = [], [], []
    for v in views:
        c = detect(cv2.cvtColor(v["color"], cv2.COLOR_RGB2GRAY), brd)
        i = detect(ir_for_detection(v["ir"]), brd)
        if c is not None:
            col.append(c)
        if i is not None:
            irs.append((i, v))
        if c is not None and i is not None:
            common = np.intersect1d(c[2], i[2])
            if len(common) >= 8:
                ci, ii = [np.searchsorted(x[2], common) for x in (c, i)]
                both.append((c[0][ci], i[1][ii], c[1][ci]))
    log(f"views {len(views)}: board found in colour {len(col)}, IR {len(irs)}, both {len(both)}")
    if len(col) < 5 or len(irs) < 5 or len(both) < 5:
        raise RuntimeError("too few views with the board found (need 5+ in colour, IR and both)")
    rms_c, k_c, d_c, _, _ = cv2.calibrateCamera([c[0] for c in col], [c[1] for c in col], size, None, None)
    rms_i, k_i, d_i, rv, tv = cv2.calibrateCamera([i[0][0] for i in irs], [i[0][1] for i in irs], size, None, None)
    rms_s, *_, rot, trans, _, _ = cv2.stereoCalibrate([b[0] for b in both], [b[1] for b in both], [b[2] for b in both],
                                                       k_i, d_i, k_c, d_c, size, flags=cv2.CALIB_FIX_INTRINSIC)
    # depth law: true distance of the board plane (from the IR pose) against raw depth inside the board
    raws, zs = [], []
    for (det, v), r, t in zip(irs, rv, tv):
        hull = cv2.convexHull(det[1].astype(np.float32)).astype(np.int32)
        mask = np.zeros(v["depth"].shape, np.uint8)
        cv2.fillConvexPoly(mask, hull, 1)
        mask = cv2.erode(mask, np.ones((15, 15), np.uint8)) > 0
        vv, uu = np.nonzero(mask & (v["depth"] > 0))
        if len(uu) < 100:
            continue
        pick = np.random.default_rng(0).choice(len(uu), min(4000, len(uu)), replace=False)
        raws.append(v["depth"][vv[pick], uu[pick]].astype(np.float64))
        zs.append(plane_depth(k_i, d_i, r, t, uu[pick], vv[pick]))
    raw, z = np.concatenate(raws), np.concatenate(zs)
    law_name, laws = fit_law(raw, z)
    a_law = laws["a/z"]
    report = {"views": len(views), "colour_views": len(col), "ir_views": len(irs), "stereo_views": len(both),
              "reprojection_px": {"colour": float(rms_c), "ir": float(rms_i), "stereo": float(rms_s)},
              "depth_law": {"better_form": law_name, **laws}, "square_mm": square_mm,
              "baseline_mm": float(np.linalg.norm(trans))}
    spec = {
        "estimate": False, "size": list(size),
        "depth": {"fx": float(k_i[0, 0]), "fy": float(k_i[1, 1]), "cx": float(k_i[0, 2]), "cy": float(k_i[1, 2]),
                  "dist": d_i.ravel().tolist()},
        "law": {"a_mm": float(a_law["A"]), "b": float(-a_law["B"])},  # z = A / (v - B)  ->  a / (v + b)
        "window_mm": [float(np.percentile(z, 0.5)), float(np.percentile(z, 99.5))],
        "colour": {"K": k_c.tolist(), "dist": d_c.ravel().tolist(), "R": rot.tolist(), "T_mm": trans.ravel().tolist()},
        "colour_from_depth": affine_at(k_i, k_c, rot, trans, 250.0),
        "report": report,
    }
    out = Path(out or fr.CAPTURES / "calibration.json")
    out.write_text(json.dumps(spec, indent=2))
    log(json.dumps(report, indent=2))
    log(f"-> {out}")
    return spec


def affine_at(k_depth, k_colour, rot, trans, z_mm):
    """The old affine depth->colour pixel mapping, refitted for distance z_mm (for code paths that
    still use it). The real mapping depends on distance (parallax), which calib.Camera handles."""
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


def capture(square_mm=SQUARE_MM):
    """Live window (colour + IR with the detected board); space keeps a view, q finishes and solves."""
    import tkinter as tk
    from . import v4l2
    from .depth import decode_y10
    from .image import ppm, yuyv_to_rgb
    brd = board(square_mm)
    out = CALIB / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    nodes = v4l2.discover()
    caps = {"color": v4l2.Capture(nodes["color"], "YUYV", fr.W, fr.H),
            "ir": v4l2.Capture(nodes["ir"], "GREY", fr.W, fr.H),
            "depth": v4l2.Capture(nodes["depth"], "Y10", fr.W, fr.H)}
    root = tk.Tk()
    root.title("triBUUTSCAN calibration: space keeps a view, q finishes")
    panes = [tk.Label(root), tk.Label(root)]
    for i, p in enumerate(panes):
        p.grid(row=0, column=i)
    status = tk.Label(root, font=("TkDefaultFont", 13))
    status.grid(row=1, column=0, columnspan=2)
    state = {"views": 0, "last": None, "images": []}

    def grab():
        bufs = {}
        for k, cap in caps.items():
            b = cap.read(timeout=1.0)
            if b is not None and len(b) >= cap.sizeimage:
                bufs[k] = b
        if len(bufs) == 3:
            state["last"] = fr.decode(bufs["color"], bufs["ir"], bufs["depth"], caps["depth"].bytesperline)
        return state["last"]

    def tick():
        frame = grab()
        if frame is not None:
            grey = cv2.cvtColor(frame["color"], cv2.COLOR_RGB2GRAY)
            c, i = detect(grey, brd), detect(ir_for_detection(frame["ir"]), brd)
            shows = []
            for img, det in ((frame["color"].copy(), c), (np.repeat(frame["ir"][..., None], 3, 2), i)):
                if det is not None:
                    for x, y in det[1].astype(int):
                        cv2.circle(img, (int(x), int(y)), 6, (255, 0, 255), 2)
                shows.append(tk.PhotoImage(data=ppm(np.ascontiguousarray(img[::2, ::2])), format="PPM"))
            state["images"] = shows
            for p, im in zip(panes, shows):
                p.configure(image=im)
            status.configure(text=f"colour: {0 if c is None else len(c[2])} corners   IR: {0 if i is None else len(i[2])} "
                                  f"corners   views kept: {state['views']} (aim for ~25, 18-31 cm, tilted)")
        root.after(50, tick)

    def keep(_e=None):
        if state["last"] is not None:
            np.savez_compressed(out / f"view_{state['views']:03d}.npz", **state["last"])
            state["views"] += 1

    def finish(_e=None):
        for cap in caps.values():
            cap.close()
        root.destroy()

    root.bind("<space>", keep)
    root.bind("q", finish)
    root.protocol("WM_DELETE_WINDOW", finish)
    tick()
    root.mainloop()
    if state["views"] >= 5:
        solve(out, square_mm)
    return out


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "board":
        CALIB.mkdir(parents=True, exist_ok=True)
        print(write_board(CALIB / "board.svg"))
    elif cmd == "capture":
        capture(float(sys.argv[2]) if len(sys.argv) > 2 else SQUARE_MM)
    elif cmd == "solve":
        solve(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else SQUARE_MM)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
