"""Small image helpers (numpy only): colour conversion, depth colouring, PNG/PPM writing."""
import struct
import zlib

import numpy as np

DEPTH_MAX = 900  # raw depth: ~280 at 25 cm, ~860 at 17 cm (estimate); display scale only


def yuyv_to_rgb(buf, w, h):
    y = np.frombuffer(buf, np.uint8, w * h * 2).reshape(h, w // 2, 4)
    luma = y[..., [0, 2]].reshape(h, w).astype(np.float32)
    u = np.repeat(y[..., 1], 2, axis=1).astype(np.float32) - 128
    v = np.repeat(y[..., 3], 2, axis=1).astype(np.float32) - 128
    rgb = np.stack([luma + 1.402 * v, luma - 0.344 * u - 0.714 * v, luma + 1.772 * u], -1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def yuyv_pixels(buf, w, xs, ys):
    """RGB (uint8) of selected pixels of a YUYV frame, without converting the whole image."""
    y = np.frombuffer(buf, np.uint8)
    pair = (ys * w + (xs & ~1)) * 2  # each 2-pixel group is Y0 U Y1 V
    luma = y[pair + (xs & 1) * 2].astype(np.float32)
    u = y[pair + 1].astype(np.float32) - 128
    v = y[pair + 3].astype(np.float32) - 128
    rgb = np.stack([luma + 1.402 * v, luma - 0.344 * u - 0.714 * v, luma + 1.772 * u], -1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def ramp(x):
    """Cheap blue->red colour ramp for x in [0, 1] -> uint8 RGB."""
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def depth_to_rgb(d, vmax=DEPTH_MAX):
    rgb = ramp(np.clip(d / vmax, 0, 1))
    rgb[d == 0] = 0
    return rgb


def ppm(rgb):
    h, w = rgb.shape[:2]
    return b"P6 %d %d 255\n" % (w, h) + rgb.tobytes()


def write_png(path, img):
    """Write an 8-bit greyscale (H, W) or RGB (H, W, 3) array as PNG."""
    h, w = img.shape[:2]
    color_type = 0 if img.ndim == 2 else 2
    rows = np.ascontiguousarray(img).reshape(h, -1)
    raw = b"".join(b"\x00" + rows[y].tobytes() for y in range(h))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        f.write(chunk(b"IEND", b""))
