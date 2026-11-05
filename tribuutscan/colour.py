"""The object's colours, learned while it is tracked: which chroma its pixels show, against what surrounds it.

Depth alone picks the object as the piece connected to where it should be (cloud.isolate_mask). That goes wrong
where depth splits the object or the guess is off: on 2026-11-02's platform scan, a fresh start on the hand seen
edge-on took one finger (216 cells, against 1590 for the whole hand), and every frame after fitted it at 2-20%.
In colour the green hand on the white platform is plainly one object. So while frames track, the object's
pixels (label 2) and the rest of the view (label 0) are counted into two chroma histograms (YUV's U and V, 32 x
32 bins: brightness and shading count little); a point's log ratio of the two says how much more it looks like
the object than its surroundings. Near-black and blown-out pixels say nothing (their chroma is noise).

When object and surroundings share their colours (a grey part on a grey table) the histograms overlap and the
model stays silent (`informative`): depth alone decides, as before. Nothing here assumes a colour.
numpy only; the scan's random stream is not touched (subsampling is by stride).
"""
import numpy as np

from . import pointcloud as pc
from .image import yuyv_pixels

BINS = 32  # per chroma axis (8 levels of U or V per bin)
MIN_LUMA, MAX_LUMA = 25, 245  # outside this, a pixel's chroma is noise: it scores 0
MIN_SAMPLES = 400  # each histogram needs this many before the model says anything
MAX_OVERLAP = 0.6  # Bhattacharyya coefficient of the two histograms above which the model stays silent
SEED_LOG_RATIO = 1.0  # points at least e (2.7) times likelier object than surroundings seed the object's piece
DECAY = 0.97  # each learn() keeps this much of what was counted before (light and view change during a scan)
SAMPLES = 3000  # at most this many points of each kind per learn()
EPS = 1e-4


def chroma(rgb):
    """(luma, u, v) of uint8 RGB rows (BT.601, the camera's YUYV)."""
    rgb = np.asarray(rgb, np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.169 * r - 0.331 * g + 0.5 * b + 128
    v = 0.5 * r - 0.419 * g - 0.081 * b + 128
    return luma, u, v


def point_rgb(frame, xyz, pix):
    """Colours of a frame's depth points (camera mm, depth pixels), from its "color" image or raw "yuyv"."""
    w = frame["shape"][1] if "shape" in frame else frame["depth"].shape[1]
    h = frame["shape"][0] if "shape" in frame else frame["depth"].shape[0]
    x, y = pc.color_source(pix[:, 0], pix[:, 1], w, h, xyz[:, 2])
    return frame["color"][y, x] if "color" in frame else yuyv_pixels(frame["yuyv"], w, x, y)


def has_colour(frame):
    return "color" in frame or "yuyv" in frame


def bins(rgb):
    """(flat bin index, usable) per colour: usable where the luma is neither near black nor blown out."""
    luma, u, v = chroma(rgb)
    iu = np.clip((u * BINS / 256).astype(int), 0, BINS - 1)
    iv = np.clip((v * BINS / 256).astype(int), 0, BINS - 1)
    return iu * BINS + iv, (luma >= MIN_LUMA) & (luma <= MAX_LUMA)


def smooth(hist):
    """A histogram blurred by one bin each way (sparse counts generalise to neighbouring shades)."""
    p = np.pad(hist, 1, mode="edge")
    return (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] + 4 * p[1:-1, 1:-1]) / 8


class ObjectColours:
    """Two chroma histograms, the object's and its surroundings', and the log ratio between them."""

    def __init__(self):
        self.obj = np.zeros((BINS, BINS))
        self.other = np.zeros((BINS, BINS))
        self.ratio = None  # log ratio per bin, once informative

    def learn(self, frame, xyz, pix, obj, other):
        """Count a tracked frame's object points (bool mask `obj`) and the rest of its view (`other`)."""
        if not has_colour(frame):
            return
        added = []
        for mask in (obj, other):
            idx = np.nonzero(mask)[0]
            idx = idx[::max(1, len(idx) // SAMPLES)]
            b, usable = bins(point_rgb(frame, xyz[idx], pix[idx])) if len(idx) else (np.zeros(0, int), np.zeros(0, bool))
            added.append(np.bincount(b[usable], minlength=BINS * BINS).reshape(BINS, BINS))
        self.obj = self.obj * DECAY + added[0]
        self.other = self.other * DECAY + added[1]
        self.ratio = None
        if self.informative():
            p, q = smooth(self.obj), smooth(self.other)
            self.ratio = np.log((p / p.sum() + EPS) / (q / q.sum() + EPS))

    def overlap(self):
        """Bhattacharyya coefficient of the two histograms (1: the same colours, 0: none shared)."""
        p, q = self.obj / max(self.obj.sum(), EPS), self.other / max(self.other.sum(), EPS)
        return float(np.sqrt(p * q).sum())

    def informative(self):
        return self.obj.sum() >= MIN_SAMPLES and self.other.sum() >= MIN_SAMPLES and self.overlap() <= MAX_OVERLAP

    def score(self, frame, xyz, pix):
        """Log ratio per point (0 for unusable pixels), or None while the model is silent."""
        if self.ratio is None or not has_colour(frame):
            return None
        b, usable = bins(point_rgb(frame, xyz, pix))
        return np.where(usable, self.ratio.ravel()[b], 0.0)

    def seeds(self, frame, xyz, pix, radius_mm):
        """Where the object is in a frame by colour: a bool mask of the points that look like it, in the
        biggest group of them and within `radius_mm` of that group's middle (the rest of the object's colour,
        e.g. split fingers, joins it; the same colour elsewhere in the room does not). None while silent."""
        s = self.score(frame, xyz, pix)
        if s is None:
            return None
        like = s >= SEED_LOG_RATIO
        if like.sum() < MIN_SEEDS_HERE:
            return np.zeros(len(xyz), bool)
        cells = np.floor(xyz[like] / GROUP_CELL_MM).astype(np.int64)
        keys, inv, counts = np.unique(cells[:, 0] * 1_000_003 ** 2 + cells[:, 1] * 1_000_003 + cells[:, 2],
                                      return_inverse=True, return_counts=True)
        centre = xyz[like][inv == np.argmax(counts)].mean(0)  # the densest cell of object colour...
        near = np.linalg.norm(xyz[like] - centre, axis=1) <= radius_mm
        centre = np.median(xyz[like][near], axis=0)  # ...and the middle of what lies around it
        out = np.zeros(len(xyz), bool)
        out[np.nonzero(like)[0][np.linalg.norm(xyz[like] - centre, axis=1) <= radius_mm]] = True
        return out

    def state(self):
        return {"obj": self.obj.tolist(), "other": self.other.tolist()}


MIN_SEEDS_HERE = 20  # (cloud.MIN_SEEDS: isolate_mask takes seeds only from this many)
GROUP_CELL_MM = 30.0  # cells for finding the densest group of object-coloured points
