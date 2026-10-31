import numpy as np

from tribuutscan import livescan


def test_view_mask_marks_object_and_base_at_one_eighth():
    pix = np.array([[0, 0], [100, 50], [1279, 799], [640, 400]])
    obj = np.array([False, True, False, True])
    base = np.array([True, False, True, False])
    m = livescan.view_mask((pix, obj, base, (800, 1280)))
    assert m.shape == (100, 160) and m.dtype == np.uint8
    assert (m[0, 0], m[6, 12], m[99, 159], m[50, 80]) == (1, 2, 1, 2) and int((m > 0).sum()) == 4
    assert livescan.view_mask(None) is None
