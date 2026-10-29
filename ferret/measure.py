"""Record raw depth statistics for a flat target at a known distance.

    python3 -m ferret.measure 30        # target is 30 cm from the scanner

Appends a row to captures/scale.csv. Use several distances, then fit raw value vs distance.
"""
import csv
import sys
from pathlib import Path

import numpy as np

from . import v4l2
from .depth import decode_y10

FRAMES = 30
ROI = 100  # half-size of the central window, in pixels


def main():
    cm = float(sys.argv[1])
    dev = v4l2.discover()["depth"]
    stack = []
    with v4l2.Capture(dev, "Y10", 1280, 800) as cap:
        for buf in cap.frames():
            if len(buf) >= cap.sizeimage:
                stack.append(decode_y10(buf, cap.width, cap.height, cap.bytesperline))
            if len(stack) == FRAMES:
                break
    d = np.stack(stack)
    h, w = d.shape[1:]
    roi = d[:, h // 2 - ROI : h // 2 + ROI, w // 2 - ROI : w // 2 + ROI]
    valid = roi[roi > 0]
    row = {
        "cm": cm,
        "frame_valid": round(float((d > 0).mean()), 3),
        "roi_valid": round(float((roi > 0).mean()), 3),
        "roi_median": float(np.median(valid)) if valid.size else None,
        "roi_p10": float(np.percentile(valid, 10)) if valid.size else None,
        "roi_p90": float(np.percentile(valid, 90)) if valid.size else None,
        "frame_max": int(d.max()),
    }
    print(row)
    out = Path("captures/scale.csv")
    out.parent.mkdir(exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=row)
        if new:
            wr.writeheader()
        wr.writerow(row)


if __name__ == "__main__":
    main()
