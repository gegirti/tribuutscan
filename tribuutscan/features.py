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
RANSAC_ROUNDS = 200

_orb = cv2.ORB_create(1500, fastThreshold=7)
_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)


def extract(rgb, depth, mask_pixels=None):
    """(points mm (N, 3) in the camera, descriptors (N, 32) uint8) of keypoints that have depth.

    rgb: colour image (H, W, 3); mask_pixels: optional (N, 2) depth-grid pixels to search around.
    """
    grey = _clahe.apply(cv2.cvtColor(np.ascontiguousarray(pc.register_color(rgb)), cv2.COLOR_RGB2GRAY))
    mask = None
    if mask_pixels is not None:
        mask = np.zeros(grey.shape, np.uint8)
        mask[mask_pixels[:, 1], mask_pixels[:, 0]] = 255
        mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))
    keypoints, desc = _orb.detectAndCompute(grey, mask)
    if desc is None:
        return np.empty((0, 3)), None
    uv = np.round([k.pt for k in keypoints]).astype(int)
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
