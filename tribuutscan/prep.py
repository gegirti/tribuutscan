"""Clean a raw depth frame before tracking: drop the pixels that are most likely wrong. numpy only.

In order: outside the depth window; flying pixels where depth jumps between neighbours; surfaces
seen nearly edge-on (their depth is poor); a 1-pixel erosion of every edge; isolated speckle.
Thresholds are estimates for this scanner (depth noise ~1 mm at 25 cm).
"""
import numpy as np

from .calib import CAMERA

JUMP_MM, JUMP_SHARE = 3.0, 0.015  # a neighbour farther than max(3 mm, 1.5% of z) is a depth jump
MAX_VIEW_DEG = 80.0  # surfaces seen at more than this from head-on are dropped
SPECKLE_WIN, SPECKLE_MIN = 7, 0.5  # a pixel needs half of its 7x7 window valid


def box_sum(a, k):
    """Sum over a k x k window around each pixel (zero padded)."""
    r = k // 2
    p = np.pad(a, r + 1).cumsum(0).cumsum(1)
    h, w = a.shape
    return p[k:k + h, k:k + w] - p[:h, k:k + w] - p[k:k + h, :w] + p[:h, :w]


def clean(raw, stride=1):
    """Return a copy of a raw depth frame with doubtful pixels set to 0 (invalid).

    With `stride`, `raw` is the every-`stride`-th-pixel grid of a frame (neighbours are `stride`
    pixels apart); cleaning only what tracking uses is ~stride^2 times cheaper.
    """
    z = np.where(raw > 0, CAMERA.depth_mm(raw), np.nan).astype(np.float32)
    near, far = CAMERA.window_mm
    with np.errstate(invalid="ignore"):
        ok = (z >= near) & (z <= far)
    z = np.where(ok, z, np.nan)
    # neighbour jumps (4-neighbourhood); a missing neighbour is not a jump
    limit = np.maximum(JUMP_MM, JUMP_SHARE * np.nan_to_num(z))
    jump = np.zeros_like(ok)
    for dz in (np.abs(z[:, 1:] - z[:, :-1]), np.abs(z[1:] - z[:-1])):
        big = np.nan_to_num(dz, nan=0.0)
        if dz.shape[1] < z.shape[1]:
            jump[:, 1:] |= big > limit[:, 1:]
            jump[:, :-1] |= big > limit[:, :-1]
        else:
            jump[1:] |= big > limit[1:]
            jump[:-1] |= big > limit[:-1]
    ok &= ~jump
    # grazing surfaces: normal from neighbours vs the viewing ray
    h, w = raw.shape
    vv, uu = np.mgrid[0:h, 0:w].astype(np.float32) * stride
    grid = CAMERA.backproject(uu, vv, z).astype(np.float32)
    dx, dy = np.full_like(grid, np.nan), np.full_like(grid, np.nan)
    dx[:, 1:-1] = grid[:, 2:] - grid[:, :-2]
    dy[1:-1] = grid[2:] - grid[:-2]
    n = np.cross(dx, dy)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.abs((n * grid).sum(-1)) / (np.linalg.norm(n, axis=-1) * np.linalg.norm(grid, axis=-1))
        ok &= ~(cos < np.cos(np.radians(MAX_VIEW_DEG)))
    # erode edges by one pixel, then drop speckle
    valid = ok.astype(np.int32)
    inner = np.zeros_like(ok)
    inner[1:-1, 1:-1] = (valid[1:-1, 1:-1] & valid[:-2, 1:-1] & valid[2:, 1:-1]
                         & valid[1:-1, :-2] & valid[1:-1, 2:]).astype(bool)
    ok = inner
    ok &= box_sum(ok.astype(np.int32), SPECKLE_WIN) >= SPECKLE_MIN * SPECKLE_WIN ** 2
    return np.where(ok, raw, 0).astype(raw.dtype)
