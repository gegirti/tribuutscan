import struct
import tempfile
from pathlib import Path

import numpy as np

from tribuutscan import pointcloud as pc
from tribuutscan import view3d


def fusion_or_none():
    try:
        from tribuutscan import fusion
    except ImportError:  # needs Open3D (.venv)
        return None
    return fusion


def test_a_wall_fuses_where_it_is():
    """A flat wall 250 mm away, seen from three nearby poses, comes out flat at 250 mm."""
    fusion = fusion_or_none()
    if fusion is None:
        return
    w, h = 160, 100
    K = np.array([[120.0, 0, 80], [0, 120, 50], [0, 0, 1]])
    engine = fusion.Engine(K, voxel_mm=0.5)
    for dx in (-0.004, 0.0, 0.004):  # camera moves sideways (metres); the wall stays at z = 250 mm
        pose = np.eye(4)
        pose[0, 3] = dx
        engine.integrate(np.full((h, w), 250.0, np.float32), np.full((h, w, 3), 128, np.uint8), pose)
    engine.integrate(np.zeros((h, w), np.float32), np.zeros((h, w, 3), np.uint8), np.eye(4))  # no depth: skipped
    vertices, triangles, colours = engine.mesh(min_weight=3)
    assert engine.frames == 3 and engine.empty == 1
    assert len(triangles) > 1000 and np.abs(vertices[:, 2] - 250).max() < 0.05
    assert (colours == 128).all()


def test_pinhole_lookup_undoes_the_lens():
    fusion = fusion_or_none()
    if fusion is None:
        return
    from tribuutscan import calib
    cam = calib.Camera({**calib.ESTIMATES, "size": [64, 40],
                        "depth": {"fx": 50.0, "fy": 50.0, "cx": 31.5, "cy": 19.5, "dist": [0.1, -0.05, 0, 0, 0]}})
    look = fusion.Pinhole(cam)
    image = np.arange(64 * 40, dtype=np.float32).reshape(40, 64)  # each pixel holds its own index
    out = look(image)
    v, u = 30, 55  # a pinhole pixel away from the centre: it must hold the sensor pixel its ray hits
    xd, yd = calib.distort((u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, cam.dist)
    assert out[v, u] == round(yd * cam.fy + cam.cy) * 64 + round(xd * cam.fx + cam.cx)
    flat = calib.Camera({**calib.ESTIMATES, "size": [64, 40], "depth": {"fx": 50.0, "fy": 50.0, "cx": 31.5, "cy": 19.5}})
    assert fusion.Pinhole(flat)(image) is image  # no lens model: nothing to do


def test_label_mask_grows_one_cell_and_scales_up():
    fusion = fusion_or_none()
    if fusion is None:
        return
    labels = np.zeros((5, 6), np.uint8)
    labels[2, 3] = 2
    mask = fusion.label_mask(labels, 8, (40, 48))
    assert mask.shape == (40, 48)
    assert mask[8:32, 16:40].all() and mask.sum() == 9 * 64  # the cell and its 8 neighbours, 8 x 8 px each


def test_crumbs_are_dropped_and_vertices_renumbered():
    fusion = fusion_or_none()
    if fusion is None:
        return
    big = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], float)
    small = big * 0.05 + [50, 0, 0]  # 0.25% of the big one's area
    vertices = np.concatenate([big, small])
    triangles = np.array([[0, 1, 2], [3, 4, 5]])
    keep = fusion.without_crumbs(vertices, triangles)
    assert keep.tolist() == [True, False]
    used, tri = fusion.cut(vertices, triangles, np.array([True, True, True, True, False, True]))
    assert used.tolist() == [0, 1, 2] and tri.tolist() == [[0, 1, 2]]


def test_mesh_ply_round_trip_and_page_payload():
    xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], np.float32)
    rgb = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [9, 9, 9]], np.uint8)
    triangles = np.array([[0, 1, 2], [1, 3, 2]])
    normals = pc.vertex_normals(xyz.astype(float), triangles)
    assert np.allclose(normals, [0, 0, 1])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mesh.ply"
        pc.write_mesh_ply(path, xyz, rgb, triangles, normals)
        rec, faces = view3d.read_ply(path, faces=True)
        payload = view3d.pack_mesh(path)
        assert view3d.read_ply(path)["red"].tolist() == [255, 0, 0, 9]  # vertices alone, as for clouds
    assert faces.tolist() == triangles.tolist()
    assert np.allclose(np.stack([rec["x"], rec["y"], rec["z"]], 1), xyz) and rec["blue"].tolist() == [0, 0, 255, 9]
    n, m = struct.unpack("<II", payload[:8])
    assert (n, m) == (4, 2) and len(payload) == 8 + 4 * 12 + 12 + 2 * 12
    assert np.frombuffer(payload[-24:], "<u4").reshape(2, 3).tolist() == triangles.tolist()
