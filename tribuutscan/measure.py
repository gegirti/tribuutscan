"""Depth statistics for a flat target: raw values, flatness and noise in mm.

    python3 -m tribuutscan.measure 30                        # target is 30 cm from the scanner
    TRIBUUTSCAN_DEPTH=y10 python3 -m tribuutscan.measure 30  # the same with 10-bit V4L2 depth

Appends a row to captures/measure.csv. plane_rms_mm is the spread of the centre region around its
best-fit plane (per frame, median over frames): depth noise plus the rounding to raw steps.
temporal_mm is each pixel's spread over the frames. step_mm is one raw step at the median distance
(14-bit steps are 1/16 of a 10-bit step). Distances use calib.CAMERA (board calibration, else the
scanner's factory calibration, else estimates).
"""
import csv
import sys
from pathlib import Path

import numpy as np

from . import frames as fr
from . import v4l2
from .calib import CAMERA
from .depth import decode_depth

FRAMES = 30
WARMUP = 10  # frames dropped after the start, as frames.grab does
ROI = 100  # half-size of the central window, in pixels


def plane_rms(points):
    """RMS distance (mm) of (N, 3) points from their least-squares plane."""
    centred = points - points.mean(0)
    return float(np.linalg.svd(centred, full_matrices=False)[1][-1] / np.sqrt(len(points)))


def main():
    cm = float(sys.argv[1]) if len(sys.argv) > 1 else None
    stack = []
    with fr.open_depth(v4l2.discover()) as cap:
        fmt = "Y14" if cap.bytesperline >= cap.width * 14 // 8 else "Y10"
        for k, buf in enumerate(cap.frames()):
            if k >= WARMUP and len(buf) >= cap.sizeimage:
                stack.append(decode_depth(buf, cap.width, cap.height, cap.bytesperline))
            if len(stack) == FRAMES:
                break
    d = np.stack(stack).astype(np.float64)
    h, w = d.shape[1:]
    rows, cols = slice(h // 2 - ROI, h // 2 + ROI), slice(w // 2 - ROI, w // 2 + ROI)
    roi = d[:, rows, cols]
    valid = roi[roi > 0]
    vv, uu = np.mgrid[rows, cols]
    rms = [plane_rms(CAMERA.backproject(uu[f > 0], vv[f > 0], CAMERA.depth_mm(f[f > 0]))) for f in roi
           if (f > 0).sum() > 1000]
    always = (roi > 0).all(0)
    z = float(np.median(CAMERA.depth_mm(valid))) if valid.size else None
    row = {
        "cm": cm,
        "format": fmt,
        "frame_valid": round(float((d > 0).mean()), 3),
        "roi_valid": round(float((roi > 0).mean()), 3),
        "roi_median": round(float(np.median(valid)), 3) if valid.size else None,
        "z_mm": round(z, 2) if z else None,
        "plane_rms_mm": round(float(np.median(rms)), 4) if rms else None,
        "temporal_mm": round(float(np.median(CAMERA.depth_mm(roi[:, always]).std(0))), 4) if always.any() else None,
        "step_mm": round(z * z / CAMERA.a / (16 if fmt == "Y14" else 1), 4) if z else None,
    }
    print(row)
    out = Path("captures/measure.csv")
    out.parent.mkdir(exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=row)
        if new:
            wr.writeheader()
        wr.writerow(row)


if __name__ == "__main__":
    main()
