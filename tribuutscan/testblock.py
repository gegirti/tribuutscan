"""A test block to print and scan: known shape, so a scan's error can be measured (accuracy, small
features, scale). numpy only.

    .venv/bin/python -m tribuutscan.testblock [out_dir]      # captures/testblock/: testblock.stl, .ply

80 x 55 x 25 mm, printable without supports (base down), asymmetric so a scan lines up with it one
way only. Features (mm; x along 80, y along 55, z up from the bed):
- base 80 x 55 x 10; step B (front left) to 18, step A (back left) to 25: step heights 8, 7, 15
- ramp at 45 degrees, x 32-44 (front middle), rising from the base to 22
- dome: a sphere of radius 12 whose top stands 6 above the base (front right)
- bosses: diameter 16, 12 high (back middle); diameter 6, 8 high (back right)
- blind holes from the top: diameter 10, 8 deep and diameter 4, 5 deep in step A; diameter 6, 6 deep
  in the base
- ribs on step B: 1, 2 and 3 mm thick, 5 high, 18 long, 4 mm apart

The shape is a signed distance function (sdf), meshed by marching cubes (gpu.triangle_table) on a
0.25 mm grid that no face of the block lies on (grid points exactly on a face made vertices coincide
and left the mesh open): closed and manifold, every face within MAX_FACE_ERROR_MM of the shape
(0.10 mm measured; checked on the sdf at random points), ~540k triangles (27 MB of STL).
Thinning it (Open3D's quadric decimation, to 150k) left it open and cut corners by up to 1.9 mm, so
it is not thinned. testblock.ply (mm) is compare.py's reference; a scan's error is best measured on
the sdf itself. The print differs from the shape by the printer's tolerance, so measure it with
calipers too (MEASURE lists what).
"""
import struct
import sys
from pathlib import Path

import numpy as np

from . import gpu

GRID_MM, GRID_OFFSET = 0.25, 0.37  # grid step, and its start off the block's round coordinates (in steps)
MAX_FACE_ERROR_MM = 0.15
OUT = Path(__file__).resolve().parent.parent / "captures" / "testblock"
MEASURE = [  # what to measure on the print (calipers), and its size as designed (mm)
    ("length (x)", 80.0), ("width (y)", 55.0), ("height at step A", 25.0), ("height at step B", 18.0),
    ("base height (right side)", 10.0), ("large boss diameter", 16.0), ("large boss height above the base", 12.0),
    ("large hole diameter", 10.0), ("large hole depth", 8.0), ("thickest rib", 3.0),
]


def box(p, lo, hi):
    lo, hi = np.asarray(lo, np.float32), np.asarray(hi, np.float32)
    c, h = (lo + hi) / 2, (hi - lo) / 2
    q = np.abs(p - c) - h
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0)


def cylinder(p, centre_xy, r, z0, z1):
    """Upright cylinder: axis at centre_xy, radius r, from z0 to z1."""
    d = np.stack([np.linalg.norm(p[..., :2] - np.asarray(centre_xy, np.float32), axis=-1) - r,
                  np.maximum(z0 - p[..., 2], p[..., 2] - z1)], -1)
    return np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(d.max(-1), 0)


def sdf(p):
    """Signed distance (mm, inside negative) of the block at points p (..., 3)."""
    solid = np.minimum.reduce([
        box(p, (0, 0, 0), (80, 55, 10)),  # base
        box(p, (0, 25, 10), (30, 55, 25)),  # step A
        box(p, (0, 0, 10), (30, 25, 18)),  # step B
        np.maximum(box(p, (32, 2, 10), (44, 22, 22)),  # ramp: 45 degrees, rising toward +x
                   ((p[..., 2] - 10) - (p[..., 0] - 32)) / np.sqrt(2)),
        np.maximum(np.linalg.norm(p - np.array([64, 14, 4], np.float32), axis=-1) - 12, -p[..., 2]),  # dome
        cylinder(p, (52, 42), 8, 10, 22),  # large boss
        cylinder(p, (72, 47), 3, 10, 18),  # small boss
        box(p, (5, 3, 18), (6, 21, 23)),  # ribs, 1, 2, 3 mm
        box(p, (10, 3, 18), (12, 21, 23)),
        box(p, (16, 3, 18), (19, 21, 23)),
    ])
    holes = np.minimum.reduce([
        cylinder(p, (15, 40), 5, 17, 26),  # 10 mm, 8 deep in step A
        cylinder(p, (25, 49), 2, 20, 26),  # 4 mm, 5 deep in step A
        cylinder(p, (38, 44), 3, 4, 11),  # 6 mm, 6 deep in the base
    ])
    return np.maximum(solid, -holes)


def marching_cubes(values, origin, step):
    """(vertices, triangles) of the zero level of a grid of signed values (inside negative), facing out:
    gpu.triangle_table's cases; the cubes along an edge share its vertex."""
    table, (nx, ny, nz) = gpu.triangle_table(), values.shape
    corner = [np.array((c & 1, c >> 1 & 1, c >> 2 & 1)) for c in range(8)]
    pattern = np.zeros((nx - 1, ny - 1, nz - 1), np.uint8)
    for c, (dx, dy, dz) in enumerate(corner):
        pattern |= (values[dx:nx - 1 + dx, dy:ny - 1 + dy, dz:nz - 1 + dz] < 0).astype(np.uint8) << c
    cube = np.array(np.nonzero((pattern != 0) & (pattern != 255))).T
    kinds = pattern[tuple(cube.T)]
    keys = []  # per triangle corner, kind by kind and triangle by triangle: the edge's key in every such cube
    for kind in np.unique(kinds):
        at = cube[kinds == kind]
        for tri in table[kind]:
            for e in tri:
                a, b = (corner[c] for c in gpu.EDGES[e])
                lo, axis = at + np.minimum(a, b), int(np.argmax(np.abs(b - a)))
                keys.append(((lo[:, 0] * ny + lo[:, 1]) * nz + lo[:, 2]) * 3 + axis)
    uniq, inverse = np.unique(np.concatenate(keys), return_inverse=True)
    axis, cell = uniq % 3, uniq // 3
    p0 = np.stack([cell // (ny * nz), cell // nz % ny, cell % nz], 1)
    p1 = p0.copy()
    p1[np.arange(len(p1)), axis] += 1
    v0, v1 = values[tuple(p0.T)].astype(np.float64), values[tuple(p1.T)].astype(np.float64)
    vertices = origin + (p0 + (p1 - p0) * (v0 / (v0 - v1))[:, None]) * step
    corners = np.split(inverse, np.cumsum([len(k) for k in keys])[:-1])
    return vertices, np.concatenate([np.stack(corners[i:i + 3], 1) for i in range(0, len(corners), 3)])


def build(grid_mm=GRID_MM):
    """The block's mesh (vertices mm, triangles) on a grid of grid_mm."""
    lo = np.array([-1.0, -1.0, -1.0]) + GRID_OFFSET * grid_mm  # no grid point on a face (see the docstring)
    n = np.ceil((np.array([81.0, 56.0, 26.0]) - lo) / grid_mm).astype(int) + 1
    values = np.empty(n, np.float32)
    ys, zs = lo[1] + np.arange(n[1]) * grid_mm, lo[2] + np.arange(n[2]) * grid_mm
    yy, zz = np.meshgrid(ys, zs, indexing="ij")
    for i in range(n[0]):  # one slice of the grid at a time
        p = np.stack([np.full_like(yy, lo[0] + i * grid_mm), yy, zz], -1).astype(np.float32)
        values[i] = sdf(p)
    return marching_cubes(values, lo, grid_mm)


def face_error(vertices, triangles, rng=None):
    """Largest distance (mm) of the shape from random points on the mesh's faces, one per face."""
    rng = rng or np.random.default_rng(0)
    w = rng.dirichlet((1, 1, 1), len(triangles))
    return float(np.abs(sdf((vertices[triangles] * w[:, :, None]).sum(1).astype(np.float32))).max())


def write_stl(path, vertices, triangles):
    """Binary STL (mm)."""
    v = vertices[triangles].astype(np.float32)
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(v), [("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, v
    with open(path, "wb") as f:
        f.write(b"triBUUTSCAN test block, mm".ljust(80, b" ") + struct.pack("<I", len(v)) + rec.tobytes())


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    out.mkdir(parents=True, exist_ok=True)
    vertices, triangles = build()
    error = face_error(vertices, triangles)
    if error > MAX_FACE_ERROR_MM:
        raise RuntimeError(f"mesh faces {error:.3f} mm off the shape (limit {MAX_FACE_ERROR_MM})")
    from . import pointcloud as pc
    write_stl(out / "testblock.stl", vertices, triangles)
    pc.write_mesh_ply(out / "testblock.ply", vertices.astype(np.float32), np.full((len(vertices), 3), 200, np.uint8),
                      triangles, pc.vertex_normals(vertices, triangles))
    print(f"{len(triangles):,} triangles, faces within {error:.3f} mm of the shape -> {out / 'testblock.stl'} "
          "(print) and testblock.ply (compare.py's reference)")
    print("measure the print:\n" + "\n".join(f"  {name:36s} {mm:6.1f} mm as designed" for name, mm in MEASURE))


if __name__ == "__main__":
    main()
