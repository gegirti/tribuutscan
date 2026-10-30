import numpy as np

from tribuutscan import prep
from tribuutscan.calib import CAMERA


def raw_of(z_mm):
    return np.clip(np.round(CAMERA.raw_from_mm(z_mm)), 1, 1023).astype(np.uint16)


def test_step_edge_flying_pixels_removed_slope_kept():
    h, w = 60, 80
    z = np.full((h, w), 250.0)
    z[:, 40:] = 290.0  # a 40 mm step: pixels next to it are doubtful
    z[:, 39] = 270.0  # a flying pixel between the two surfaces
    out = prep.clean(raw_of(z))
    assert (out[10:50, 39] == 0).all() and (out[10:50, 20] > 0).all() and (out[10:50, 60] > 0).all()
    # a 60-degree slope (steep but not grazing) survives
    vv, uu = np.mgrid[0:h, 0:w]
    slope = 240.0 + (uu - 40) * 0.6 * np.tan(np.radians(60)) * 250 / CAMERA.fx
    kept = prep.clean(raw_of(slope))
    assert (kept[10:50, 10:70] > 0).mean() > 0.9


def test_speckle_removed():
    raw = np.zeros((40, 40), np.uint16)
    raw[20, 20] = raw_of(np.array(250.0))  # a lone valid pixel
    assert prep.clean(raw).sum() == 0
