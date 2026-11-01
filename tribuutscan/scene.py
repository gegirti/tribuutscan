"""What is object and what is table, decided once and kept: scan modes and a stable object region.

Modes:
  still      the object stands still on a table and the scanner moves (default, like Creality's
             handheld mode). The base (table) plane is found once and kept in WORLD coordinates (3
             frames must agree, looked for during the first VOTE_LIMIT frames, and the object at the
             image centre must stand on it); each frame re-fits it near its kept place.
  turntable  the scanner stays still and the object turns (turntable or by hand on the table).
             The plane is found once and kept in CAMERA coordinates; only the object is tracked.
  free       no support plane assumed (an object held in the hand): each frame keeps the piece at
             the image centre, as the first version did.

The object is kept in a cylinder standing on the plane (its starting footprint + REGION_MARGIN_MM).
In each frame it is the piece more than OBJECT_ABOVE_MM above the plane that the middle of the view
hits (the piece tracking uses), if that piece reaches the cylinder's core; otherwise whatever is
connected to the core. The cylinder grows toward what is seen in the middle of the view, by at most
REGION_GROW_MM per frame, so a neighbour at the edge of the view does not pull it out (on the marker
scan it grew to 317 mm while it followed everything connected). The base is the plane's points inside
the cylinder: merged too, so "remove base" can be decided when the cloud is saved (cloud.Scanner).

Until a base is found (or never, e.g. an object held in the hand) the object is each frame's centre
piece inside a ball around the first one, which grows slowly and at most to BALL_MAX times its start.
"""
import numpy as np

from . import cloud as cl

MODES = ("still", "turntable", "free")
VOTE_FRAMES, VOTES_NEEDED = 5, 3  # 3 votes must agree; found after VOTE_FRAMES frames counts as late
VOTE_LIMIT = 300  # frames to keep looking for the base
# The base may be a thin rim around an object that fills the view: on the marker part the table was
# 7-9% of the points and 55-151 cm2, under the 10% / 80 cm2 that tracking's per-frame plane test uses.
# Votes must agree in world coordinates, which a slice through a curved surface does not do.
BASE_MIN_SHARE, BASE_MIN_CM2, BASE_TRIES = 0.03, 30.0, 5
CONTACT_MM = 10.0  # the object at the image centre must come this close to a plane for it to be its base
FOOTPRINT_CELL_MM, MAX_UNDER = 5.0, 0.5  # ...and at most half the plane may lie inside the object's outline
BALL_MAX = 2.0
SAME_PLANE_DEG, SAME_PLANE_MM = 5.0, 5.0
OBJECT_ABOVE_MM = 3.0
REGION_MARGIN_MM, REGION_GROW_MM = 20.0, 15.0
GROW_WINDOW = 0.5  # the region grows only toward what is seen in the middle of the view (this share of
                   # its width and height): what the scanner is aimed at, not a neighbour at the edge
REFIT_BAND_MM = 10.0  # the kept plane is re-fitted to the frame's points within this band...
REFIT_MAX_DEG, REFIT_MAX_MM = 3.0, 5.0  # ...and a re-fit that turns or moves it more is not taken


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


def stands_on(xyz, pix, shape, n, c):
    """Does the object at the image centre stand on plane (n, c)? Its centre piece above the plane must
    come within CONTACT_MM of it (a table under the object does, a wall behind it does not), and the
    plane's points must lie AROUND the object's outline seen along the normal, not inside it: under the
    object a table is hidden, while one of the object's own flat sides lies within its outline."""
    h = xyz @ n + c
    idx = np.nonzero(h > OBJECT_ABOVE_MM)[0]
    if len(idx) < 50:
        return False
    piece = cl.centre_piece(xyz[idx], pix[idx], shape)
    if not piece.any() or h[idx][piece].min() >= CONTACT_MM:
        return False
    a = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    a /= np.linalg.norm(a)
    cells = lambda p: np.unique(np.floor(np.c_[p @ a, p @ np.cross(n, a)] / FOOTPRINT_CELL_MM).astype(np.int64)
                                @ np.array([1, 1_000_003], np.int64))
    outline = cells(xyz[idx][piece])
    on = cells(xyz[np.abs(h) < cl.PLANE_TOL_MM])
    return bool(len(on) and np.isin(on, outline).mean() <= MAX_UNDER)


class Scene:
    """Keeps the support plane and the object's region for one scan."""

    def __init__(self, mode="still", rng=None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode, self.rng = mode, rng or np.random.default_rng(0)
        self.plane = None  # (n, c): n points up out of the table, in the region frame
        self.votes, self.seen_frames = [], 0
        self.axis, self.radius, self.top = None, None, None  # object cylinder, region frame (mm)
        self.ball = None  # (centre, radius, start radius) while no base is known, region frame (mm)
        self.late_plane = False  # the plane was settled after objects had been merged
        self.off_centre = 0  # frames whose view centre missed the object (masks fell back to the core)

    def observe(self, xyz_cam, pix, shape, cam_to_region):
        """Vote for a support plane during the first frames. cam_to_region: 4x4, mm."""
        if self.mode == "free" or self.plane is not None or self.seen_frames >= VOTE_LIMIT:
            return
        self.seen_frames += 1
        found = cl.support_plane(xyz_cam, self.rng, pix, shape, tries=BASE_TRIES, min_share=BASE_MIN_SHARE,
                                 min_cm2=BASE_MIN_CM2, accept=lambda n, c: stands_on(xyz_cam, pix, shape, n, c))
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
        Once the object's region is known, only points outside its footprint take part: the lower
        edge of an object standing up, or a face of it lying low, is near the table too, and on the
        marker scan such re-fits tipped the plane 20-64 degrees in 276 of 581 frames. A re-fit that
        still turns the plane more than REFIT_MAX_DEG or moves it more than REFIT_MAX_MM (at the
        object, or at the view's centre before the region is known) is not taken.
        """
        to_cam = np.linalg.inv(cam_to_region)
        prior = plane_to(self.plane, to_cam)
        pts_all, where = xyz_cam, None
        if self.axis is not None:
            where = to_cam[:3, :3] @ self.axis + to_cam[:3, 3]
            n0 = prior[0]
            flat = xyz_cam - np.outer((xyz_cam - where) @ n0, n0)
            pts_all = xyz_cam[np.linalg.norm(flat - where, axis=1) > self.radius - REGION_MARGIN_MM]
        n, c = prior
        for band in (REFIT_BAND_MM, cl.PLANE_TOL_MM):
            near = np.abs(pts_all @ n + c) < band
            if near.sum() < 50:
                return prior if band == REFIT_BAND_MM else (n, c)
            pts = pts_all[near]
            centre = pts.mean(0)
            n_fit = np.linalg.svd(pts - centre, full_matrices=False)[2][2]
            n_fit = n_fit if n_fit @ prior[0] > 0 else -n_fit
            at = where if where is not None else centre
            if (np.degrees(np.arccos(np.clip(n_fit @ prior[0], -1, 1))) > REFIT_MAX_DEG
                    or abs(n_fit @ (at - centre) - (prior[0] @ at + prior[1])) > REFIT_MAX_MM):
                return prior
            n, c = n_fit, -n_fit @ centre
        return n, c

    def object_mask(self, xyz_cam, pix, shape, cam_to_region):
        """Which points (camera coordinates) are the object (see `masks`)."""
        return self.masks(xyz_cam, pix, shape, cam_to_region)[0]

    def masks(self, xyz_cam, pix, shape, cam_to_region):
        """(object, base) masks of the points (camera coordinates). The base is the plane's points
        inside the object's cylinder; empty while no plane is known (then the object is the centre
        piece inside the ball)."""
        mask = np.zeros(len(xyz_cam), bool)
        if self.plane is None:
            return self._ball_mask(xyz_cam, pix, shape, cam_to_region), mask.copy()
        n, c = self.plane_in_camera(xyz_cam, cam_to_region)
        h = xyz_cam @ n + c
        above = h > OBJECT_ABOVE_MM
        base = np.zeros(len(xyz_cam), bool)
        if not above.any():
            return mask, base
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
        core = inside & (r <= self.radius - REGION_MARGIN_MM) & (h[idx] <= self.top - REGION_MARGIN_MM)
        piece = cl.centre_piece(pts, pix[idx], shape)  # what the middle of the view hits...
        if not (piece & core).any():  # ...unless that is not the object: then whatever is in its core
            self.off_centre += 1
            piece = cl.connected_piece(pts, core) if core.any() else core
        hh, ww = shape
        u, v = pix[idx, 0], pix[idx, 1]
        aimed = piece & (np.abs(u - ww / 2) <= GROW_WINDOW * ww / 2) & (np.abs(v - hh / 2) <= GROW_WINDOW * hh / 2)
        if aimed.any():  # grow the region toward what the middle of the view sees, a little per frame
            reach = r[aimed].max() + REGION_MARGIN_MM
            self.radius = min(max(self.radius, reach), self.radius + REGION_GROW_MM)
            self.top = min(max(self.top, h[idx][aimed].max() + REGION_MARGIN_MM), self.top + REGION_GROW_MM)
        piece &= (r <= self.radius) & (h[idx] <= self.top)
        mask[idx[piece]] = True
        near = np.nonzero((h > -cl.PLANE_TOL_MM) & ~above)[0]  # the base under and around the object
        flat_near = xyz_cam[near] - np.outer((xyz_cam[near] - axis) @ n, n)
        base[near[np.linalg.norm(flat_near - axis, axis=1) <= self.radius]] = True
        return mask, base

    def _ball_mask(self, xyz_cam, pix, shape, cam_to_region):
        """No base known: the frame's centre piece (cl.isolate_mask), kept inside a ball around the
        first one; the ball grows by REGION_GROW_MM per frame toward what is kept, up to BALL_MAX."""
        keep = cl.isolate_mask(xyz_cam, pix, shape, self.rng)[0]
        if self.mode != "still" or not keep.any():
            return keep
        idx = np.nonzero(keep)[0]
        world = xyz_cam[idx] @ cam_to_region[:3, :3].T + cam_to_region[:3, 3]
        if self.ball is None:
            centre = world.mean(0)
            r0 = float(np.linalg.norm(world - centre, axis=1).max()) + REGION_MARGIN_MM
            self.ball = (centre, r0, r0)
        centre, radius, r0 = self.ball
        d = np.linalg.norm(world - centre, axis=1)
        inside = d <= radius
        if inside.any():
            grown = min(max(radius, d[inside].max() + REGION_MARGIN_MM), radius + REGION_GROW_MM, BALL_MAX * r0)
            self.ball = (centre, grown, r0)
        keep[idx[~inside]] = False
        return keep

    def state(self):
        """The kept plane and region, as plain JSON values."""
        listed = lambda v: None if v is None else np.asarray(v).tolist()
        return {"mode": self.mode, "plane": None if self.plane is None else [listed(self.plane[0]), self.plane[1]],
                "axis": listed(self.axis), "radius": self.radius, "top": self.top,
                "ball": None if self.ball is None else [listed(self.ball[0]), self.ball[1], self.ball[2]]}

    def load(self, state):
        if state.get("plane"):
            self.plane = (np.array(state["plane"][0]), float(state["plane"][1]))
        if state.get("axis") is not None:
            self.axis, self.radius, self.top = np.array(state["axis"]), state["radius"], state["top"]
        if state.get("ball") is not None:
            self.ball = (np.array(state["ball"][0]), state["ball"][1], state["ball"][2])

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
