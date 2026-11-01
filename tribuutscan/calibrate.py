"""Calibrate the scanner with a printed ChArUco board: the colour and IR cameras, where they sit
relative to each other, and the depth law. Writes captures/calibration.json, which calib.CAMERA then
uses instead of the estimates. OpenCV (.venv).

    .venv/bin/python -m tribuutscan.calibrate board                      # captures/calib/board.svg
    .venv/bin/python -m tribuutscan.calibrate capture [square_mm] [captures/calib/<stamp>]
                                                         # live: keeps views by itself, q ends; a folder continues
    .venv/bin/python -m tribuutscan.calibrate solve captures/calib/<stamp> [square_mm]

Print board.svg at 100% (no "fit to page"), on matte paper, and tape it FLAT to something rigid
and flat (a desk): a page on a spiral pad bowed 6 mm, which the depth map showed. Measure one
square with a ruler: pass that as square_mm. Keep about 25 views at 18-31 cm, tilted different
ways, the whole board or most of it in view of both cameras.

The IR image shows the board only through the projector's dots, which break up the ArUco codes
(they cannot be read in IR at 30 cm). The board is therefore found in colour, and IR is aligned to
it: see ir_corners().
"""
import base64
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import calib
from . import frames as fr

BOARD_SQUARES = (11, 8)
SQUARE_MM, MARKER_MM = 24.0, 18.0  # as drawn; measure the print and pass the real square size
DICTIONARY = cv2.aruco.DICT_5X5_100
CALIB = fr.CAPTURES / "calib"
IR_SIGMA = 2.0  # px: blurs the projector's dots into grey levels (white squares bright, black dark)
MARGIN_MM = 8.0  # white paper around the board in the alignment template
MIN_CORRELATION = 0.5  # image alignment accepted above this
# auto-keep: corners move < STILL_PX per update (~4 a second) for STILL_S, and lie NEW_PX from kept views.
# Colour and IR are captured 0.4 ms apart, so a slowly moving hand-held scanner is fine.
STILL_PX, STILL_S, NEW_PX = 3.0, 0.5, 40.0


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


def detect(grey, brd):
    """(object points (N, 3) mm, image points (N, 2), corner ids (N,)) sorted by id, or None."""
    corners, ids, _, _ = cv2.aruco.CharucoDetector(brd).detectBoard(grey)
    if ids is None or len(ids) < 8:
        return None
    # the detector's corners sit about 0.5 px off (on a rendered board and on the scanner's colour
    # image); refining them on the image removes that
    corners = cv2.cornerSubPix(grey, np.float32(corners), (5, 5), (-1, -1),
                               (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))
    obj, img = brd.matchImagePoints(corners, ids)
    order = np.argsort(ids.ravel())
    return obj.reshape(-1, 3).astype(np.float32)[order], img.reshape(-1, 2).astype(np.float32)[order], ids.ravel()[order]


def template(brd, px_per_mm):
    """The board as printed (float image, white paper margin) at px_per_mm, and the 3x3 matrix taking
    board mm to its pixels (square edges fall between pixels)."""
    side = brd.getSquareLength()
    size = tuple(int(round(n * side * px_per_mm)) for n in brd.getChessboardSize())
    m = int(round(MARGIN_MM * px_per_mm))
    img = cv2.copyMakeBorder(brd.generateImage(size, marginSize=0, borderBits=1), m, m, m, m,
                             cv2.BORDER_CONSTANT, value=255)
    return np.float32(img), np.array([[px_per_mm, 0, m - 0.5], [0, px_per_mm, m - 0.5], [0, 0, 1.0]])


def ecc(tpl, image, warp, motion, iterations=100):
    """cv2.findTransformECC: (correlation, warp taking tpl pixels into image), or None if it fails."""
    try:
        return cv2.findTransformECC(tpl, image, np.float32(warp), motion,
                                    (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, iterations, 1e-6), None, 5)
    except cv2.error:
        return None


def ir_board(ir, col, brd, scales=(0.125, 0.25, 0.5)):
    """Whole-board alignment in IR: (homography board mm -> IR px, correlation, blurred IR) or None.

    Starts from the colour detection `col` moved by the estimated colour -> depth mapping (rough is
    enough), then aligns the board's image to the blurred IR image (ECC), coarse to fine.
    """
    h_c, _ = cv2.findHomography(np.float64(col[0][:, :2]), np.float64(col[1]))
    if h_c is None:
        return None
    r = calib.ESTIMATES["colour_from_depth"]
    to_ir = np.array([[r["ax"], 0, r["bx"]], [0, r["ay"], r["by"]], [0, 0, 1.0]]) @ h_c
    grey = cv2.GaussianBlur(np.float32(ir), (0, 0), IR_SIGMA)
    rho = 0.0
    for s in scales:
        tpl, to_px = template(brd, 4 * s)  # about the IR image's own resolution at that size (ECC cost grows with it)
        scale = np.array([[s, 0, s / 2 - 0.5], [0, s, s / 2 - 0.5], [0, 0, 1.0]])  # full-size px -> resized px
        warp = scale @ to_ir @ np.linalg.inv(to_px)
        found = ecc(tpl, cv2.resize(grey, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), warp / warp[2, 2],
                    cv2.MOTION_HOMOGRAPHY)
        if found is None:
            return None
        rho, warp = found
        to_ir = np.linalg.inv(scale) @ np.float64(warp) @ to_px
    return to_ir, rho, grey


def ir_corners(ir, col, brd, px_per_mm=6.0):
    """Board corners in the IR image, like detect(), for the corners found in colour (`col`); or None.

    The projector's dots break up the codes and the corners in IR (corner refinement there wandered
    2-3 px with the dots), so corners come from aligning images instead: the whole board first
    (ir_board), then each corner by a shift-only alignment of the board's image within one square of
    it, on the IR image rectified by the whole-board fit. That per-corner step keeps lens distortion,
    which no single homography holds. On a still board the corners scattered 0.03 px between frames;
    on synthetic dot-lit views they land 0.26 px (median) from the truth, 0.1 px of it a constant offset.
    """
    found = ir_board(ir, col, brd)
    if found is None or found[1] < MIN_CORRELATION:
        return None
    to_ir, _, grey = found
    tpl, to_px = template(brd, px_per_mm)
    warp = to_ir @ np.linalg.inv(to_px)  # template px -> IR px
    size = (tpl.shape[1], tpl.shape[0])
    rect = cv2.warpPerspective(grey, warp, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    inside = cv2.warpPerspective(np.ones_like(grey), warp, size, flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    r = int(round(brd.getSquareLength() * px_per_mm))
    obj = brd.getChessboardCorners()
    ids, pts = [], []
    for k in col[2]:
        cx, cy = (to_px @ (obj[k, 0], obj[k, 1], 1.0))[:2]
        x0, y0 = int(round(cx)) - r, int(round(cy)) - r
        box = np.s_[y0:y0 + 2 * r + 1, x0:x0 + 2 * r + 1]
        if x0 < 0 or y0 < 0 or x0 + 2 * r >= size[0] or y0 + 2 * r >= size[1] or inside[box].min() < 1:
            continue  # partly outside the IR image
        found = ecc(tpl[box], rect[box], np.eye(2, 3), cv2.MOTION_TRANSLATION, 50)
        if found is None or found[0] < MIN_CORRELATION or np.hypot(*found[1][:, 2]) > 3 * px_per_mm:
            continue
        p = warp @ (cx + found[1][0, 2], cy + found[1][1, 2], 1.0)
        ids.append(k)
        pts.append(p[:2] / p[2])
    if len(ids) < 8:
        return None
    ids = np.array(ids)
    return obj[ids].astype(np.float32), np.float32(pts), ids


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


def calibrate_robust(dets, numbers, size, name, log, limit=3.0):
    """calibrateCamera on detections; views with an error over `limit` x the median (and over 0.5 px)
    are dropped once and the rest solved again. -> rms, K, dist, kept view numbers, dropped, rvecs, tvecs."""
    obj, img = [d[0] for d in dets], [d[1] for d in dets]
    rms, k, dist, rv, tv, _, _, errors = cv2.calibrateCameraExtended(obj, img, size, None, None)
    errors = errors.ravel()
    bad = (errors > limit * np.median(errors)) & (errors > 0.5)
    if bad.any():
        log(f"{name}: dropped views {[n for n, b in zip(numbers, bad) if b]} "
            f"(errors {np.round(errors[bad], 2).tolist()} px, median {np.median(errors):.2f} px)")
        rms, k, dist, rv, tv = cv2.calibrateCamera([o for o, b in zip(obj, bad) if not b],
                                                   [i for i, b in zip(img, bad) if not b], size, None, None)
    return (rms, k, dist, [n for n, b in zip(numbers, bad) if not b], [n for n, b in zip(numbers, bad) if b],
            rv, tv)


def solve(directory, square_mm=SQUARE_MM, out=None, log=print):
    """Calibrate from captured views; write calibration.json (default captures/calibration.json)."""
    brd = board(square_mm)
    views = load_views(directory)
    modes = {str(v.get("depth_mode", calib.CLOSE)) for v in views}
    if modes != {calib.CLOSE}:  # the solve writes the close-range law; a long-range one needs its own block
        raise RuntimeError(f"views taken in {sorted(modes)}: only {calib.CLOSE} can be solved for now")
    size = (fr.W, fr.H)
    col, irs, both = [], [], []
    for n, v in enumerate(views):
        c = detect(cv2.cvtColor(v["color"], cv2.COLOR_RGB2GRAY), brd)
        i = ir_corners(v["ir"], c, brd) if c is not None else None
        if c is not None:
            col.append((c, n))
        if i is not None:
            irs.append((i, v, n))
        if c is not None and i is not None:
            common = np.intersect1d(c[2], i[2])
            if len(common) >= 8:
                ci, ii = [np.searchsorted(x[2], common) for x in (c, i)]
                both.append((c[0][ci], i[1][ii], c[1][ci], n))
    log(f"views {len(views)}: board found in colour {len(col)}, IR {len(irs)}, both {len(both)}")
    if len(col) < 5 or len(irs) < 5 or len(both) < 5:
        raise RuntimeError("too few views with the board found (need 5+ in colour, IR and both)")
    rms_c, k_c, d_c, kept_c, dropped_c, _, _ = calibrate_robust([c for c, _ in col], [n for _, n in col], size,
                                                                "colour", log)
    rms_i, k_i, d_i, kept_i, dropped_i, rv, tv = calibrate_robust([i for i, _, _ in irs], [n for _, _, n in irs], size,
                                                                  "IR", log)
    col = [x for x in col if x[1] in kept_c]
    irs = [x for x in irs if x[2] in kept_i]
    both = [b for b in both if b[3] not in dropped_c + dropped_i]
    rms_s, *_, rot, trans, _, _ = cv2.stereoCalibrate([b[0] for b in both], [b[1] for b in both], [b[2] for b in both],
                                                       k_i, d_i, k_c, d_c, size, flags=cv2.CALIB_FIX_INTRINSIC)
    # depth law: true distance of the board plane (from the IR pose) against raw depth inside the board
    raws, zs = [], []
    for (det, v, _), r, t in zip(irs, rv, tv):
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
    seen = [float(np.percentile(z, 0.5)), float(np.percentile(z, 99.5))]
    a_law = laws["a/z"]
    report = {"views": len(views), "colour_views": len(col), "ir_views": len(irs), "stereo_views": len(both),
              "dropped_views": {"colour": dropped_c, "ir": dropped_i},
              "reprojection_px": {"colour": float(rms_c), "ir": float(rms_i), "stereo": float(rms_s)},
              "depth_law": {"better_form": law_name, **laws}, "square_mm": square_mm,
              "baseline_mm": float(np.linalg.norm(trans)), "depth_seen_mm": seen}
    spec = {
        "estimate": False, "size": list(size),
        "depth": {"fx": float(k_i[0, 0]), "fy": float(k_i[1, 1]), "cx": float(k_i[0, 2]), "cy": float(k_i[1, 2]),
                  "dist": d_i.ravel().tolist()},
        "law": {"a_mm": float(a_law["A"]), "b": float(-a_law["B"])},  # z = A / (v - B)  ->  a / (v + b)
        # the depth window: what the views saw, but never narrower than the estimate (the near end
        # observed is only where the board was held, not the scanner's limit)
        "window_mm": [min(calib.ESTIMATES["window_mm"][0], seen[0]), max(calib.ESTIMATES["window_mm"][1], seen[1])],
        "colour": {"K": k_c.tolist(), "dist": d_c.ravel().tolist(), "R": rot.tolist(), "T_mm": trans.ravel().tolist()},
        "colour_from_depth": calib.affine_at(k_i, k_c, rot, trans, 250.0),
        "report": report,
    }
    out = Path(out or fr.CAPTURES / "calibration.json")
    out.write_text(json.dumps(spec, indent=2))
    log(json.dumps(report, indent=2))
    log(f"-> {out}")
    return spec


def board_place(frame, col, to_ir, k_rough):
    """(distance cm from the depth map, tilt degrees) of the board; both rough, for guidance."""
    outline = cv2.perspectiveTransform(np.float64([[0, 0], [264, 0], [264, 192], [0, 192]]).reshape(-1, 1, 2), to_ir)
    mask = np.zeros(frame["depth"].shape, np.uint8)
    cv2.fillConvexPoly(mask, outline.reshape(-1, 2).astype(np.int32), 1)
    raw = frame["depth"][(mask > 0) & (frame["depth"] > 0)]
    cm = float(calib.CAMERA.depth_mm(np.median(raw))) / 10 if raw.size > 100 else None
    ok, rvec, _ = cv2.solvePnP(col[0], col[1], k_rough, None)
    tilt = float(np.degrees(np.arccos(min(1.0, abs(cv2.Rodrigues(rvec)[0][2, 2]))))) if ok else None
    return cm, tilt


def corner_shift(a, b):
    """Median movement (px) of the corners two detections share; inf if they share fewer than 8."""
    common, ia, ib = np.intersect1d(a[2], b[2], return_indices=True)
    return float(np.median(np.linalg.norm(a[1][ia] - b[1][ib], axis=1))) if len(common) >= 8 else np.inf


def capture(square_mm=SQUARE_MM, out=None):
    """Live window (colour + IR with the board found). A view is kept by itself when the board has been
    still for STILL_S at a new place (hold the scanner still); space keeps one at once; q finishes and
    solves. With `out` an earlier session folder, its views count and new ones are added.

    The camera and the board search run in a thread: in the window's own thread they froze it for
    0.25 s per update and over a second per kept view, and a session's window was closed mid-way.
    """
    import threading
    import tkinter as tk
    import traceback
    from . import v4l2
    from .image import ppm
    brd = board(square_mm)
    out = Path(out) if out else CALIB / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    k_rough = np.array([[1000.0, 0, fr.W / 2], [0, 1000.0, fr.H / 2], [0, 0, 1]])  # tilt guidance only
    state = {"kept": [], "places": [], "prev": None, "still": None, "flash": 0.0, "note": "", "logged": 0.0,
             "show": None, "keep_now": False, "stop": False}
    for v in load_views(out):  # an earlier session's views: they count, and new views must differ from them
        col = detect(cv2.cvtColor(v["color"], cv2.COLOR_RGB2GRAY), brd)
        found = ir_board(v["ir"], col, brd) if col is not None else None
        if found is not None:
            state["kept"].append(col)
            state["places"].append(board_place(v, col, found[0], k_rough))
    if state["kept"]:
        print(f"continuing {out.name}: {len(state['kept'])} views", flush=True)
    nodes = v4l2.discover()
    caps = {"color": v4l2.Capture(nodes["color"], "YUYV", fr.W, fr.H),
            "ir": v4l2.Capture(nodes["ir"], "GREY", fr.W, fr.H),
            "depth": fr.open_depth(nodes)}

    def keep(frame, col, place):
        i = ir_corners(frame["ir"], col, brd)
        state["still"] = None  # the next keep needs another still moment
        if i is None or len(i[2]) < 20:
            state["note"] = "IR could not place the board here: come a little closer or tilt less"
            return
        n = len(list(out.glob("view_*.npz")))
        np.savez_compressed(out / f"view_{n:03d}.npz", **frame, depth_mode=calib.CAMERA.mode)
        state["kept"].append(col)
        state["places"].append(place)
        state["flash"], state["note"] = time.monotonic() + 0.6, ""
        cm, tilt = place
        print(f"kept view {n}: {len(col[2])} corners in colour, {len(i[2])} in IR, "
              f"{'?' if cm is None else f'{cm:.1f}'} cm, tilt {'?' if tilt is None else f'{tilt:.0f}'} deg", flush=True)

    def update():
        """One frame set: find the board, maybe keep the view, leave a preview for the window."""
        got = fr.paired(caps, timeout=0.5)
        if got is None:
            return
        frame = fr.decode(got["color"][0], got["ir"][0], got["depth"][0], caps["depth"].bytesperline)
        col = detect(cv2.cvtColor(frame["color"], cv2.COLOR_RGB2GRAY), brd)
        found = ir_board(frame["ir"], col, brd, scales=(0.125, 0.25)) if col is not None else None
        aligned = found is not None and found[1] >= MIN_CORRELATION
        place = board_place(frame, col, found[0], k_rough) if aligned else (None, None)
        now = time.monotonic()
        moved = corner_shift(col, state["prev"]) if col is not None and state["prev"] is not None else np.inf
        state["prev"] = col
        state["still"] = (state["still"] or now) if moved < STILL_PX else None
        new = col is not None and all(corner_shift(col, k) > NEW_PX for k in state["kept"])
        if aligned and (state["keep_now"] or (new and state["still"] and now - state["still"] >= STILL_S)):
            state["keep_now"] = False
            keep(frame, col, place)
        grey = found[2] if found is not None else cv2.GaussianBlur(np.float32(frame["ir"]), (0, 0), IR_SIGMA)
        shows = [frame["color"][::2, ::2].copy(),
                 np.repeat(cv2.normalize(grey[::2, ::2], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)[..., None], 3, 2)]
        if col is not None:
            for x, y in col[1] / 2:
                cv2.circle(shows[0], (int(x), int(y)), 4, (255, 0, 255), 2)
        if aligned:
            p = cv2.perspectiveTransform(np.float64(brd.getChessboardCorners()[:, :2]).reshape(-1, 1, 2), found[0])
            for x, y in p.reshape(-1, 2) / 2:
                cv2.circle(shows[1], (int(x), int(y)), 4, (0, 255, 0), 2)
        cm, tilt = place
        text = (f"colour: {0 if col is None else len(col[2])} corners   IR: {'board found' if aligned else 'no board'}   "
                f"{'' if cm is None else f'{cm:.1f} cm (estimate)   '}{'' if tilt is None else f'tilt {tilt:.0f} deg   '}"
                f"{'still' if state['still'] else 'moving'}{'' if new or col is None else ', seen already'}   {state['note']}")
        places = list(state["places"])
        near = sum(1 for c, _ in places if c is not None and c < 22)
        far = sum(1 for c, _ in places if c is not None and c > 27)
        tilted = sum(1 for _, t in places if t is not None and t > 20)
        tally = (f"kept {len(places)} views (aim for ~25): {near} near (< 22 cm), {len(places) - near - far} middle, "
                 f"{far} far (> 27 cm); {tilted} tilted over 20 deg   |   space keeps now, q finishes")
        state["show"] = ([ppm(np.ascontiguousarray(s)) for s in shows], text, tally, now < state["flash"])
        if now - state["logged"] >= 2:  # a trace of what the window showed, for debugging
            print(f"status: {text.strip()} (moved {moved:.1f} px)", flush=True)
            state["logged"] = now

    def work():
        while not state["stop"]:
            try:
                update()
            except Exception as e:
                if state["stop"]:
                    return  # cameras closed while reading
                if isinstance(e, OSError) and e.errno == 19:
                    print("scanner disconnected", flush=True)
                    state["note"] = "SCANNER DISCONNECTED: close this window, reconnect, start again (views are kept)"
                    return
                traceback.print_exc()
                time.sleep(0.5)

    root = tk.Tk()
    root.title("triBUUTSCAN calibration: hold the scanner still to keep a view, q finishes")
    panes = [tk.Label(root, bd=6), tk.Label(root, bd=6)]
    for i, p in enumerate(panes):
        p.grid(row=0, column=i)
    status = tk.Label(root, font=("TkDefaultFont", 13))
    status.grid(row=1, column=0, columnspan=2)
    tally_label = tk.Label(root, font=("TkDefaultFont", 15, "bold"))
    tally_label.grid(row=2, column=0, columnspan=2)
    images = []

    def refresh():  # the window's thread only shows the newest preview
        show, state["show"] = state["show"], None
        if show is not None:
            data, text, tally, flash = show
            images[:] = [tk.PhotoImage(data=d, format="PPM") for d in data]
            for p, im in zip(panes, images):
                p.configure(image=im, bg="#22cc44" if flash else root.cget("bg"))
            status.configure(text=text)
            tally_label.configure(text=tally)
        elif state["note"].startswith("SCANNER"):
            status.configure(text=state["note"])
        root.after(30, refresh)

    def keep_now(_e=None):
        state["keep_now"] = True

    def finish(how):
        print(f"window closed: {how}", flush=True)
        state["stop"] = True
        worker.join(timeout=5)
        for cap in caps.values():
            cap.close()
        root.destroy()

    root.bind("<space>", keep_now)
    root.bind("q", lambda _e: finish("q pressed"))
    root.protocol("WM_DELETE_WINDOW", lambda: finish("window close button"))
    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    refresh()
    root.mainloop()
    if len(state["kept"]) >= 5:
        solve(out, square_mm, out=out / "calibration.json")  # checked before it replaces captures/calibration.json
    return out


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "board":
        CALIB.mkdir(parents=True, exist_ok=True)
        print(write_board(CALIB / "board.svg"))
    elif cmd == "capture":
        capture(float(sys.argv[2]) if len(sys.argv) > 2 else SQUARE_MM, sys.argv[3] if len(sys.argv) > 3 else None)
    elif cmd == "solve":
        solve(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else SQUARE_MM)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
