import numpy as np


def test_compare_takes_meshes_and_a_given_placement():
    try:
        import open3d as o3d
        from tribuutscan import compare
    except ImportError:  # needs Open3D (.venv)
        return
    import tempfile
    from pathlib import Path
    part = o3d.geometry.TriangleMesh.create_box(100, 60, 30)  # mm; a post at one corner breaks the symmetry
    post = o3d.geometry.TriangleMesh.create_cylinder(10, 40).translate((15, 15, 50))
    part += post
    turn = np.eye(4)
    turn[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((0.3, -0.2, 0.5))
    turn[:3, 3] = (40, -25, 10)
    ours = o3d.geometry.TriangleMesh(part).transform(turn)
    with tempfile.TemporaryDirectory() as tmp:
        ref_path, ours_path = Path(tmp) / "ref.ply", Path(tmp) / "mesh.ply"
        o3d.io.write_triangle_mesh(str(ref_path), part)
        o3d.io.write_triangle_mesh(str(ours_path), ours)
        report, placed, d = compare.compare(ours_path, ref_path, seeds=3)
        assert report["our_points"] > 50_000  # sampled over the surface, not the 8 + 40 vertices
        assert report["accuracy_mm"]["median"] < 0.05 and report["completeness"]["within_1mm"] > 0.95
        assert np.allclose(np.asarray(report["transform_mm"]) @ turn, np.eye(4), atol=0.02)
        again, _, _ = compare.compare(ours_path, ref_path, transform=report["transform_mm"])
        assert again["placement"] == "given" and again["accuracy_mm"] == report["accuracy_mm"]
