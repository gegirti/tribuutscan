"""Find scanning markers: uncoded circular targets (bright dot in a dark ring, e.g. 6 mm / 10 mm).

The standard sticker is a retroreflective white dot inside a black ring. Seen at an angle it is an
ellipse whose centre stays a stable point, which is what tracking uses. Detection here is numpy only:
a bright centre whose surroundings are dark on all four sides, at a few dot sizes, then the centre
is refined to the brightness-weighted centroid of the dot.

    python3 -m tribuutscan.markers captures/<stamp>   # writes markers.png and prints the list
"""
import sys
from pathlib import Path

import numpy as np

RADII = (4, 6, 9, 13)  # dot radius in pixels to look for (at 1280x800, about 15-50 cm away)
MIN_CONTRAST = 60  # dot minus ring brightness, 0-255. Shaded markers (~40) are missed: below
                   # ~60 key labels, logos and highlights look the same, so reliability wins.
MAX_TINT = 60  # dots are near-white: mean (max - min) over RGB; only checked when colour is given
MIN_DOT = 80  # the dot itself must be at least this bright
MAX_ELONGATION = 3.0  # dot long axis / short axis; a tilted marker is an ellipse, slivers are longer
MIN_RING_DARK = 0.9  # share of the band just outside the dot that must be dark


def box_means(img, half):
    """Mean over a (2*half+1)^2 window centred on each pixel (edges padded)."""
    k = 2 * half + 1
    p = np.pad(img, half + 1, mode="edge").astype(np.float64)
    s = p.cumsum(0).cumsum(1)
    h, w = img.shape
    total = s[k:k + h, k:k + w] - s[:h, k:k + w] - s[k:k + h, :w] + s[:h, :w]
    return total / (k * k)


def score(gray, r):
    """Contrast of a bright dot of radius r against the darkest of four ring patches around it."""
    dot = box_means(gray, max(1, int(0.6 * r)))
    patch = box_means(gray, max(1, int(0.35 * r)))
    d = int(round(1.35 * r))  # ring patch centres: inside the black ring, outside the dot
    shifted = [np.roll(patch, shift, axis) for shift, axis in ((d, 0), (-d, 0), (d, 1), (-d, 1))]
    ring = np.max(shifted, axis=0)  # the brightest side must still be dark
    s = dot - ring
    s[dot < MIN_DOT] = 0
    return s


def detect(gray, radii=RADII, rgb=None):
    """Return a list of (x, y, radius, contrast) for markers in an 8-bit greyscale image.

    Pass the colour image as `rgb` to also reject coloured dots (markers are white).
    """
    gray = np.asarray(gray, np.float64)
    h, w = gray.shape
    best = np.zeros_like(gray)
    best_r = np.zeros(gray.shape, np.int16)
    for r in radii:
        s = score(gray, r)
        better = s > best
        best[better], best_r[better] = s[better], r
    margin = 2 * max(radii)
    best[:margin], best[-margin:], best[:, :margin], best[:, -margin:] = 0, 0, 0, 0
    ys, xs = np.nonzero(best > MIN_CONTRAST)
    order = np.argsort(-best[ys, xs])
    found = []
    for y, x in zip(ys[order], xs[order]):
        r = best_r[y, x]
        if any((x - fx) ** 2 + (y - fy) ** 2 < (2 * max(r, fr)) ** 2 for fx, fy, fr, _ in found):
            continue  # non-maximum suppression
        cx, cy, rr, elongation, ring_dark = refine(gray, x, y, r)
        if elongation > MAX_ELONGATION or ring_dark < MIN_RING_DARK or rr < min(radii):  # glints are tiny
            continue
        if rgb is not None:
            k = max(1, int(0.6 * rr))
            dot = rgb[y - k:y + k + 1, x - k:x + k + 1].reshape(-1, 3).astype(np.int16)
            if (dot.max(1) - dot.min(1)).mean() > MAX_TINT:
                continue
        found.append((cx, cy, rr, float(best[y, x])))
    return found


def refine(gray, x, y, r):
    """Centre, radius, elongation and ring darkness of the dot near (x, y), searched at radius r.

    An ellipse is fitted to the bright core (moments). The ellipse must be filled, and the band just
    outside it (the black ring, out to 1.6x the dot size; a 6/10 mm marker's ring ends at 1.67x)
    must be dark all round. A paper corner or edge runs through that band and fails.
    """
    k = int(2.2 * r) + 1
    win = gray[y - k:y + k + 1, x - k:x + k + 1]
    if win.shape != (2 * k + 1, 2 * k + 1):
        return float(x), float(y), float(r), np.inf, 0.0
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    core = np.hypot(xx, yy) <= 1.3 * r
    lo, hi = np.percentile(win[core], 10), win[core].max()
    mask = win > (lo + hi) / 2
    pts = mask & core
    if pts.sum() < 5:
        return float(x), float(y), float(r), np.inf, 0.0
    wts = np.where(pts, win - lo, 0)
    cx, cy = (wts * xx).sum() / wts.sum(), (wts * yy).sum() / wts.sum()
    cov = np.cov(np.stack([xx[pts] - cx, yy[pts] - cy]))
    evals = np.linalg.eigvalsh(cov)
    if evals[0] <= 1e-6:
        return float(x), float(y), float(r), np.inf, 0.0
    d = np.stack([xx - cx, yy - cy], -1)
    m2 = np.einsum("...i,ij,...j->...", d, np.linalg.inv(cov), d) / 4  # 1.0 on the fitted ellipse edge
    inside, band = m2 <= 0.8, (m2 >= 1.25 ** 2) & (m2 <= 1.6 ** 2)
    fill = mask[inside].mean() if inside.any() else 0.0
    ring_dark = float((~mask)[band].mean()) if band.any() else 0.0
    if fill < 0.9:
        ring_dark = 0.0
    return (float(x + cx), float(y + cy), float(2 * np.sqrt(np.sqrt(evals[0] * evals[1]))),
            float(np.sqrt(evals[1] / evals[0])), ring_dark)


def luma(rgb):
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def draw(rgb, found, color=(255, 0, 255)):
    """Return a copy of rgb with a ring drawn round each marker."""
    out = rgb.copy()
    h, w = out.shape[:2]
    t = np.linspace(0, 2 * np.pi, 64)
    for x, y, r, _ in found:
        for rr in (r + 3, r + 4):
            px = np.clip(np.round(x + rr * np.cos(t)).astype(int), 0, w - 1)
            py = np.clip(np.round(y + rr * np.sin(t)).astype(int), 0, h - 1)
            out[py, px] = color
    return out


RING = (1.8, 4.0)  # depth ring around a dot, in dot radii: beyond the 6/10 mm black ring (1.67)
RING_MIN_PX = 30  # valid depth pixels needed in the ring
RING_MAX_RMS_MM = 1.0  # ring must be flat to this (estimate; depth noise ~1 mm)
HALF_RADII = (3, 4, 5, 7)  # dot radii to search at half resolution (smaller ones were glints)


def detect_half(rgb):
    """Markers in full-resolution pixels, searched at half resolution (~4x faster, same hits)."""
    half = rgb[::2, ::2]
    return [(2 * x, 2 * y, 2 * r, c) for x, y, r, c in detect(luma(half), HALF_RADII, rgb=half)]


def locate(depth, found):
    """3D centre (mm, depth camera: x right, y down, z forward) and unit normal of each marker, or None.

    The retroreflective dot itself has no depth, so a plane is fitted to the valid depth in a ring
    around it and the camera ray through the dot's centre is intersected with that plane. Colour
    pixels map to depth pixels through calib.CAMERA (an estimate until calibrated).
    """
    from .calib import CAMERA
    h, w = depth.shape
    out = []
    for x, y, r, _ in found:
        u0, v0 = CAMERA.depth_px(x, y)
        rd = max(r * CAMERA.reg["ax"], 2.0)
        k = int(RING[1] * rd) + 1
        u_lo, u_hi, v_lo, v_hi = int(u0) - k, int(u0) + k + 1, int(v0) - k, int(v0) + k + 1
        if u_lo < 0 or v_lo < 0 or u_hi > w or v_hi > h:
            out.append(None)
            continue
        win = depth[v_lo:v_hi, u_lo:u_hi]
        vv, uu = np.mgrid[v_lo:v_hi, u_lo:u_hi]
        dist = np.hypot(uu - u0, vv - v0)
        ring = (dist >= RING[0] * rd) & (dist <= RING[1] * rd) & (win > 0)
        if ring.sum() < RING_MIN_PX:
            out.append(None)
            continue
        pts = CAMERA.backproject(uu[ring], vv[ring], CAMERA.depth_mm(win[ring]))
        centre = pts.mean(0)
        _, sv, vt = np.linalg.svd(pts - centre, full_matrices=False)
        normal = vt[2]
        if sv[2] / np.sqrt(len(pts)) > RING_MAX_RMS_MM:
            out.append(None)  # not flat: the marker sits on an edge or the depth is noisy
            continue
        ray = CAMERA.backproject(u0, v0, 1.0)
        point = ray * (normal @ centre) / (normal @ ray)
        out.append((point, normal if normal @ point < 0 else -normal))
    return out


def main():
    from .image import write_png
    directory = Path(sys.argv[1])
    data = np.load(directory / "frames.npz")
    rgb = data["color"]
    found = detect(luma(rgb), rgb=rgb)
    write_png(directory / "markers.png", draw(rgb, found))
    print(f"{len(found)} markers -> {directory / 'markers.png'}")
    for x, y, r, c in found:
        print(f"  x {x:7.1f}  y {y:6.1f}  r {r:4.1f}  contrast {c:5.0f}")


if __name__ == "__main__":
    main()
