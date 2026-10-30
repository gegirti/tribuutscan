"""What is object and what is table, decided once and kept: scan modes and a stable object region.

Modes:
  still      the object stands still on a table and the scanner moves (default, like Creality's
             handheld mode). The table plane is found once (it must agree in 3 of the first 5
             frames) and kept in WORLD coordinates; each frame re-fits it near its kept place.
  turntable  the scanner stays still and the object turns (turntable or by hand on the table).
             The plane is found once and kept in CAMERA coordinates; only the object is tracked.
  free       no support plane assumed (an object held in the hand): each frame keeps the piece at
             the image centre, as the first version did.

The object is everything more than OBJECT_ABOVE_MM above the plane that lies inside a cylinder
around the object (its starting footprint + REGION_MARGIN_MM) or is connected to what does. The
cylinder grows as new parts of the object come into view, but only by REGION_GROW_MM per frame, so
a neighbouring object touching it is not swallowed at once.
"""
import numpy as np

from . import cloud as cl

MODES = ("still", "turntable", "free")
VOTE_FRAMES, VOTES_NEEDED = 5, 3  # the plane must be found in 3 of the first 5 frames
SAME_PLANE_DEG, SAME_PLANE_MM = 5.0, 5.0
OBJECT_ABOVE_MM = 3.0
REGION_MARGIN_MM, REGION_GROW_MM = 20.0, 15.0
REFIT_BAND_MM = 10.0  # the kept plane is re-fitted to the frame's points within this band


def plane_to(plane, transform_mm):
    """Plane (n, c) with n.p + c = 0 moved by a rigid transform (4x4, translation in mm)."""
    n, c = plane
    r, t = transform_mm[:3, :3], transform_mm[:3, 3]
    n2 = r @ n
    return n2, c - n2 @ t


def mm_transform(pose):
    """A pose (translation in metres) with its translation in mm."""
    out = pose.copy()
    out[:3, 3] = pose[:3, 3] * 1000
    return out


class Scene:
    """Keeps the support plane and the object's region for one scan."""

    def __init__(self, mode="still", rng=None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode, self.rng = mode, rng or np.random.default_rng(0)
        self.plane = None  # (n, c): n points up out of the table, in the region frame
        self.votes, self.seen_frames = [], 0
        self.axis, self.radius, self.top = None, None, None  # object cylinder, region frame (mm)
        self.late_plane = False  # the plane was settled after objects had been merged

    def observe(self, xyz_cam, pix, shape, cam_to_region):
        """Vote for a support plane during the first frames. cam_to_region: 4x4, mm."""
        if self.mode == "free" or self.plane is not None or self.seen_frames >= VOTE_FRAMES + 20:
            return
        self.seen_frames += 1
        found = cl.support_plane(xyz_cam, self.rng, pix, shape)
        if found is None:
            return
        self.votes.append(plane_to(found, cam_to_region))
        for n, c in self.votes:
            agree = [(n2, c2) for n2, c2 in self.votes
                     if np.degrees(np.arccos(np.clip(n @ n2, -1, 1))) < SAME_PLANE_DEG and abs(c - c2) < SAME_PLANE_MM]
            if len(agree) >= VOTES_NEEDED:
                n_avg = np.mean([a[0] for a in agree], axis=0)
                n_avg /= np.linalg.norm(n_avg)
                self.plane = (n_avg, float(np.mean([a[1] for a in agree])))
                self.late_plane = self.seen_frames > VOTE_FRAMES
                return

    def plane_in_camera(self, xyz_cam, cam_to_region):
        """The kept plane in this frame's camera coordinates, re-fitted to the frame's own points.

        The kept plane, moved with a (possibly predicted) pose, is only a prior: points within
        REFIT_BAND_MM of it are fitted again (least squares, then once more without outliers), so a
        few mm of pose error cannot let the table leak into the object or cut the object's base.
        """
        prior = plane_to(self.plane, np.linalg.inv(cam_to_region))
        n, c = prior
        for band in (REFIT_BAND_MM, cl.PLANE_TOL_MM):
            near = np.abs(xyz_cam @ n + c) < band
            if near.sum() < 50:
                return prior if band == REFIT_BAND_MM else (n, c)
            pts = xyz_cam[near]
            centre = pts.mean(0)
            n_fit = np.linalg.svd(pts - centre, full_matrices=False)[2][2]
            n_fit = n_fit if n_fit @ prior[0] > 0 else -n_fit
            n, c = n_fit, -n_fit @ centre
        return n, c

    def object_mask(self, xyz_cam, pix, shape, cam_to_region):
        """Which points (camera coordinates) are the object. Without a plane: the piece at the centre."""
        if self.plane is None:
            return cl.isolate_mask(xyz_cam, pix, shape, self.rng)[0]
        n, c = self.plane_in_camera(xyz_cam, cam_to_region)
        h = xyz_cam @ n + c
        above = h > OBJECT_ABOVE_MM
        mask = np.zeros(len(xyz_cam), bool)
        if not above.any():
            return mask
        idx = np.nonzero(above)[0]
        pts = xyz_cam[idx]
        to_cam = np.linalg.inv(cam_to_region)
        if self.axis is None:  # first view: the object is the piece above the table at the image centre
            first = cl.centre_piece(pts, pix[idx], shape)
            self._fit_region(pts[first] @ cam_to_region[:3, :3].T + cam_to_region[:3, 3],
                             self.plane[0])
        axis = to_cam[:3, :3] @ self.axis + to_cam[:3, 3]
        flat = pts - np.outer((pts - axis) @ n, n)  # projected onto the table
        r = np.linalg.norm(flat - axis, axis=1)
        inside = (r <= self.radius) & (h[idx] <= self.top)
        piece = cl.connected_piece(pts, inside) if inside.any() else inside
        if piece.any():  # grow the region toward what is connected, a little per frame
            reach = r[piece].max() + REGION_MARGIN_MM
            self.radius = min(max(self.radius, reach), self.radius + REGION_GROW_MM)
            self.top = min(max(self.top, h[idx][piece].max() + REGION_MARGIN_MM), self.top + REGION_GROW_MM)
        mask[idx[piece]] = True
        return mask

    def height(self, xyz_region):
        n, c = self.plane
        return xyz_region @ n + c

    def _fit_region(self, pts, n):
        if not len(pts):
            pts = np.zeros((1, 3))
        centre = pts.mean(0)
        self.axis = centre - (centre @ n + self.plane[1]) * n  # the centroid dropped onto the table
        flat = pts - np.outer((pts - self.axis) @ n, n)
        self.radius = float(np.linalg.norm(flat - self.axis, axis=1).max()) + REGION_MARGIN_MM
        self.top = float((pts @ n + self.plane[1]).max()) + REGION_MARGIN_MM
