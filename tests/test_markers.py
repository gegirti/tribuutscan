import numpy as np

from tribuutscan import markers as mk


def sticker(img, x, y, dot, ring, sx=1.0):
    """Paint a white dot in a black ring (ellipse when sx != 1) on a grey image."""
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    d = np.hypot((xx - x) / sx, yy - y)
    img[d <= ring] = 20
    img[d <= dot] = 230


def test_finds_round_and_tilted_markers():
    img = np.full((200, 300), 120.0)
    sticker(img, 80.3, 90.6, 8, 13)
    sticker(img, 200, 100, 7, 12, sx=1.8)  # seen at an angle
    found = sorted(mk.detect(img))
    assert len(found) == 2
    assert abs(found[0][0] - 80.3) < 0.5 and abs(found[0][1] - 90.6) < 0.5


def test_ignores_edges_and_glints():
    img = np.full((200, 300), 20.0)
    img[:, 150:] = 230  # a bright half: an edge, not a marker
    img[100, 60] = 255  # a one-pixel glint
    assert mk.detect(img) == []
