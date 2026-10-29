"""Decode the depth node's packed 10-bit stream.

The node reports pixel format 'Y10 ' but with bytesperline = width * 10 / 8: pixels are an
MSB-first 10-bit bitstream, 4 pixels per 5 bytes. This is NOT the MIPI Y10P layout (which keeps
the low bits in the 5th byte). Values are ~1..200, nearer = larger; units are not yet known.
"""
import numpy as np


def decode_y10(buf, width, height, bytesperline):
    """Return a (height, width) uint16 array of raw depth values (0 = invalid)."""
    rows = np.frombuffer(buf, dtype=np.uint8, count=height * bytesperline).reshape(height, bytesperline)
    g = rows[:, : width // 4 * 5].reshape(height, width // 4, 5).astype(np.uint16)
    out = np.empty((height, width // 4, 4), dtype=np.uint16)
    out[..., 0] = (g[..., 0] << 2) | (g[..., 1] >> 6)
    out[..., 1] = ((g[..., 1] & 0x3F) << 4) | (g[..., 2] >> 4)
    out[..., 2] = ((g[..., 2] & 0x0F) << 6) | (g[..., 3] >> 2)
    out[..., 3] = ((g[..., 3] & 0x03) << 8) | g[..., 4]
    return out.reshape(height, width)


def encode_y10(values):
    """Inverse of decode_y10 (for tests)."""
    h, w = values.shape
    v = values.astype(np.uint16).reshape(h, w // 4, 4)
    out = np.empty((h, w // 4, 5), dtype=np.uint8)
    out[..., 0] = v[..., 0] >> 2
    out[..., 1] = ((v[..., 0] & 0x3) << 6) | (v[..., 1] >> 4)
    out[..., 2] = ((v[..., 1] & 0xF) << 4) | (v[..., 2] >> 6)
    out[..., 3] = ((v[..., 2] & 0x3F) << 2) | (v[..., 3] >> 8)
    out[..., 4] = v[..., 3] & 0xFF
    return out.reshape(h, -1).tobytes()
