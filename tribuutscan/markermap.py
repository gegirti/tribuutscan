"""A map of the scanning markers in world coordinates, and matching a frame's markers to it. numpy only.

Markers are uncoded (all look alike), so they are matched by geometry: with a predicted pose, by
nearest neighbour and then by checking that the distances between markers agree (they do not change
as the scanner moves); without one (re-finding, continuing a scan), by comparing the side lengths of
marker triangles. Symmetric or collinear layouts are refused: place markers irregularly.
Positions are mm; poses are camera -> world 4x4 in metres (like cloud.Scanner).
"""
import itertools

import numpy as np

TRUSTED = 2  # sightings before a marker is used to re-find without a prediction
MATCH_MM = 10.0  # predicted position to map marker: nearest neighbour within this
PAIR_MM = 2.0  # distances between matched markers must agree to this
SIDE_MM, SIDE_SHARE = 2.0, 0.01  # triangle sides agree to 2 mm + 1%
INLIER_MM = 3.0  # a marker fits a pose if it lands this close to its map marker
MIN_TRIANGLE_HEIGHT_MM = 15.0  # a 3-marker match needs a triangle at least this "fat"


def kabsch(src, dst):
    """Rigid transform (4x4, same units as the points) that best maps src (N, 3) onto dst (N, 3)."""
    cs, cd = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - cs).T @ (dst - cd))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = r, cd - r @ cs
    return t


def to_pose(t_mm):
    """A 4x4 with mm translation as a pose with metres."""
    out = t_mm.copy()
    out[:3, 3] = t_mm[:3, 3] / 1000
    return out


def apply(pose, pts_mm):
    return pts_mm @ pose[:3, :3].T + pose[:3, 3] * 1000


def triangle_height(a, b, c):
    """Smallest height of the triangle abc (0 when collinear)."""
    area2 = np.linalg.norm(np.cross(b - a, c - a))
    return area2 / max(np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(a - c), 1e-9)


def consistent(frame_pts, map_pts, tol=PAIR_MM):
    """Largest subset of matched pairs whose pairwise distances agree (greedy: drop the worst)."""
    keep = list(range(len(frame_pts)))
    while len(keep) >= 2:
        f, m = frame_pts[keep], map_pts[keep]
        err = np.abs(np.linalg.norm(f[:, None] - f[None], axis=2) - np.linalg.norm(m[:, None] - m[None], axis=2))
        worst = err.max(1)
        if worst.max() <= tol:
            break
        keep.pop(int(np.argmax(worst)))
    return keep


def search(frame_pts, map_pts):
    """Best transform (4x4, mm) placing frame_pts on map_pts by triangle matching, or None if there
    are too few inliers or an equally good, different answer (a symmetric layout)."""
    if len(frame_pts) < 3 or len(map_pts) < 3:
        return None
    sides = lambda p: np.sort([np.linalg.norm(p[0] - p[1]), np.linalg.norm(p[1] - p[2]), np.linalg.norm(p[2] - p[0])])
    map_tris = [(idx, sides(map_pts[list(idx)])) for idx in itertools.combinations(range(len(map_pts)), 3)
                if triangle_height(*map_pts[list(idx)]) > 5.0]
    best = []
    for fidx in itertools.combinations(range(len(frame_pts)), 3):
        f3 = frame_pts[list(fidx)]
        if triangle_height(*f3) <= 5.0:
            continue
        fs = sides(f3)
        for midx, ms in map_tris:
            if np.all(np.abs(fs - ms) <= SIDE_MM + SIDE_SHARE * ms):
                for perm in itertools.permutations(midx):  # which map marker is which frame marker
                    t = kabsch(f3, map_pts[list(perm)])
                    moved = f3 @ t[:3, :3].T + t[:3, 3]
                    if np.linalg.norm(moved - map_pts[list(perm)], axis=1).max() > INLIER_MM:
                        continue
                    all_moved = frame_pts @ t[:3, :3].T + t[:3, 3]
                    d = np.linalg.norm(all_moved[:, None] - map_pts[None], axis=2).min(1)
                    best.append(((d < INLIER_MM).sum(), triangle_height(*f3), t))
    if not best:
        return None
    best.sort(key=lambda b: (-b[0], -b[1]))
    inliers, height, t = best[0]
    moved = frame_pts @ t[:3, :3].T + t[:3, 3]
    differs = lambda t2: np.linalg.norm(frame_pts @ t2[:3, :3].T + t2[:3, 3] - moved, axis=1).max() > INLIER_MM
    rivals = [b for b in best[1:] if b[0] == inliers and differs(b[2])]  # an equally good, different answer
    if rivals or inliers < 3 or (inliers == 3 and height < MIN_TRIANGLE_HEIGHT_MM):
        return None  # too few, or ambiguous (a symmetric layout)
    return t


class MarkerMap:
    def __init__(self, add_mm=INLIER_MM):
        self.add_mm = add_mm  # a sighting this close to a map marker is that marker (else a new one)
        self.pos = np.empty((0, 3))  # world mm, running means
        self.count = np.empty(0, int)  # sightings; a false detection rarely gets a second one

    def __len__(self):
        return len(self.pos)

    def match_predicted(self, frame_pts, pose):
        """[(frame index, map index)] for markers near where `pose` puts them, distance-checked."""
        if not len(self) or not len(frame_pts):
            return []
        world = apply(pose, frame_pts)
        d = np.linalg.norm(world[:, None] - self.pos[None], axis=2)
        pairs = [(i, int(np.argmin(d[i]))) for i in range(len(world))
                 if d[i].min() < MATCH_MM and int(np.argmin(d[:, np.argmin(d[i])])) == i]  # mutual nearest
        if len(pairs) < 2:
            return pairs
        f, m = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
        return [pairs[k] for k in consistent(frame_pts[f], self.pos[m])]

    def match_global(self, frame_pts):
        """Pose (4x4, metres) placing the frame's markers on the map without a prediction, or None.

        Triangles are matched against markers seen at least TRUSTED times; the pose is then refined
        on every map marker that fits (the camera position is sensitive to marker noise: a few mm
        of lever arm when the markers are nearly coplanar and 25 cm away).
        """
        t = search(frame_pts, self.pos[self.count >= TRUSTED])
        if t is None:
            return None
        moved = frame_pts @ t[:3, :3].T + t[:3, 3]
        d = np.linalg.norm(moved[:, None] - self.pos[None], axis=2)
        pairs = [(i, int(np.argmin(d[i]))) for i in range(len(frame_pts)) if d[i].min() < INLIER_MM]
        return self.pose_from(frame_pts, pairs) if len(pairs) >= 3 else None

    def pose_from(self, frame_pts, pairs):
        """Pose (metres) from matched pairs (at least 3), by Kabsch."""
        f = frame_pts[[p[0] for p in pairs]]
        m = self.pos[[p[1] for p in pairs]]
        return to_pose(kabsch(f, m))

    def add(self, frame_pts, pose):
        """Fold this frame's markers (camera mm) into the map at a known pose."""
        if not len(frame_pts):
            return
        for w in apply(pose, frame_pts):
            if len(self):
                d = np.linalg.norm(self.pos - w, axis=1)
                k = int(np.argmin(d))
                if d[k] < self.add_mm:  # the pose is final here: a marker lands within a few mm
                    self.count[k] += 1
                    self.pos[k] += (w - self.pos[k]) / self.count[k]
                    continue
            self.pos = np.vstack([self.pos, w])
            self.count = np.append(self.count, 1)
