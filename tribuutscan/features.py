"""Colour features (ORB) as a starting pose for tracking: they handle movements too large for ICP to
find on its own. OpenCV (.venv).

Keypoints are found in the colour image registered to the depth grid (pointcloud.register_color),
contrast-equalised (CLAHE), and lifted to 3D with the depth at the keypoint; two frames are matched
by descriptor (ratio test) and the 3D pairs by RANSAC + Kabsch. In "still" mode the whole image is
used (the scene does not move); when the object moves, only the object.

Measured on saved scans (frames 5 apart, 3D-verified matches): black part 36 on the object but 97
over the whole image; kettle 382 on the object; synthetic 180-480. Dark or plain scenes give few:
the result is then None and tracking falls back to the motion prediction.
"""
import cv2
import numpy as np

from . import pointcloud as pc
from .calib import CAMERA
from .markermap import kabsch

MIN_INLIERS = 25
INLIER_MM = 3.0
RATIO = 0.8
RANSAC_ROUNDS = 100
SCALE = 2  # ORB runs on a half-resolution image: about 4x faster, enough keypoints

_orb = cv2.ORB_create(1500, fastThreshold=7)
_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)


def luma_of(frame):
    """Brightness (H, W) uint8 of a frame set: straight from the YUYV bytes when it has them."""
    depth = frame["depth"]
    h, w = depth.shape
    if "yuyv" in frame:
        return np.frombuffer(frame["yuyv"], np.uint8, w * h * 2)[0::2].reshape(h, w)
    rgb = frame["color"]
    return (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114).astype(np.uint8)


def extract(luma, depth, mask_pixels=None):
    """(points mm (N, 3) in the camera, descriptors (N, 32) uint8) of keypoints that have depth.

    luma: colour-camera brightness (H, W) uint8; mask_pixels: optional (N, 2) depth-grid pixels to
    search around. Works at 1/SCALE resolution on the depth grid.
    """
    h, w = depth.shape
    v, u = np.mgrid[0:h:SCALE, 0:w:SCALE]
    z = None
    if CAMERA.colour is not None:  # calibrated: each grid pixel's own distance (median where none)
        raw = depth[::SCALE, ::SCALE]
        z = np.where(raw > 0, CAMERA.depth_mm(np.maximum(raw, 1)),
                     np.median(CAMERA.depth_mm(raw[raw > 0])) if (raw > 0).any() else 250.0)
    src_x, src_y = pc.color_source(u, v, w, h, z)  # depth-grid pixel -> colour pixel (registration)
    grey = _clahe.apply(np.ascontiguousarray(luma[src_y, src_x]))
    mask = None
    if mask_pixels is not None:
        mask = np.zeros(grey.shape, np.uint8)
        mask[mask_pixels[:, 1] // SCALE, mask_pixels[:, 0] // SCALE] = 255
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    keypoints, desc = _orb.detectAndCompute(grey, mask)
    if desc is None:
        return np.empty((0, 3)), None
    uv = np.round(np.array([k.pt for k in keypoints]) * SCALE).astype(int)
    h, w = depth.shape
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    raw = np.zeros(len(uv), depth.dtype)
    raw[inside] = depth[uv[inside, 1], uv[inside, 0]]
    ok = raw > 0
    pts = CAMERA.backproject(uv[ok, 0], uv[ok, 1], CAMERA.depth_mm(raw[ok]))
    return pts, desc[ok]


def relative_pose(current, reference, rng):
    """Transform (4x4, metres) taking `current` camera coordinates to `reference`'s, or None.

    current, reference: (points mm, descriptors) from extract().
    """
    (pa, da), (pb, db) = current, reference
    if da is None or db is None or len(da) < 3 or len(db) < 3:
        return None, 0
    good = [m for m, n in (p for p in _matcher.knnMatch(da, db, k=2) if len(p) == 2) if m.distance < RATIO * n.distance]
    if len(good) < MIN_INLIERS:
        return None, len(good)
    a, b = pa[[m.queryIdx for m in good]], pb[[m.trainIdx for m in good]]
    best, best_mask = None, None
    for _ in range(RANSAC_ROUNDS):
        s = rng.choice(len(good), 3, replace=False)
        t = kabsch(a[s], b[s])
        inl = np.linalg.norm(a @ t[:3, :3].T + t[:3, 3] - b, axis=1) < INLIER_MM
        if best_mask is None or inl.sum() > best_mask.sum():
            best, best_mask = t, inl
    if best_mask.sum() < MIN_INLIERS:
        return None, int(best_mask.sum())
    t = kabsch(a[best_mask], b[best_mask])  # refined on all inliers
    t[:3, 3] /= 1000
    return t, int(best_mask.sum())
