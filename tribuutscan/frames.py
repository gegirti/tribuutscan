"""Grab and save synchronised-ish frame sets (colour, IR, depth), all 1280x800.

Colour is requested at 1280x800 so it shares the depth map's pixel grid. Whether the two are truly
registered (different cameras, different baselines) is not verified; identity is assumed.
"""
import json
import time
from pathlib import Path

import numpy as np

from . import v4l2
from .depth import decode_y10
from .image import depth_to_rgb, write_png, yuyv_to_rgb

W, H = 1280, 800
CAPTURES = Path(__file__).resolve().parent.parent / "captures"


def decode(color_buf, ir_buf, depth_buf, depth_bytesperline):
    return {
        "color": yuyv_to_rgb(color_buf, W, H),
        "ir": np.frombuffer(ir_buf, np.uint8, W * H).reshape(H, W)[:, ::-1].copy(),  # flipped to match colour/depth
        "depth": decode_y10(depth_buf, W, H, depth_bytesperline),
    }


def grab(warmup=10):
    """Open all three nodes, discard warm-up frames, and return one decoded frame set."""
    nodes = v4l2.discover()
    with v4l2.Capture(nodes["color"], "YUYV", W, H) as c, v4l2.Capture(nodes["ir"], "GREY", W, H) as i, \
            v4l2.Capture(nodes["depth"], "Y10", W, H) as d:
        last = {}
        for _ in range(warmup):
            for key, cap in (("color", c), ("ir", i), ("depth", d)):
                buf = cap.read()
                if buf is not None and len(buf) >= cap.sizeimage:
                    last[key] = buf
        if len(last) != 3:
            raise RuntimeError(f"no usable frames from: {set(('color', 'ir', 'depth')) - set(last)}")
        return decode(last["color"], last["ir"], last["depth"], d.bytesperline)


def save(frames, root=CAPTURES):
    """Write a frame set to captures/<timestamp>/ and return the directory."""
    out = Path(root) / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    write_png(out / "color.png", frames["color"])
    write_png(out / "ir.png", frames["ir"])
    write_png(out / "depth_preview.png", depth_to_rgb(frames["depth"]))
    np.savez_compressed(out / "frames.npz", **frames)
    valid = frames["depth"][frames["depth"] > 0]
    (out / "info.json").write_text(json.dumps({
        "size": [W, H],
        "depth_valid_fraction": round(float((frames["depth"] > 0).mean()), 4),
        "depth_median": float(np.median(valid)) if valid.size else None,
        "note": "ir is flipped horizontally to match colour/depth (sensor output is mirrored); depth is raw, units unknown",
    }, indent=2))
    return out
