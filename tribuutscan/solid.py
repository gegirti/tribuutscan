"""A scan's fused surface closed into a printable solid: watertight, every edge shared by two triangles.

The fused mesh (fusion.py) is open wherever the scanner never looked: fingertips seen only from the side,
the bottom of a stand, the gap between a wrist and its stand's rim, and many pinholes. A slicer wants a
closed solid, so `close` makes one:

1. screened Poisson (Open3D) on the fused mesh's own vertices and normals: a closed surface that keeps to
   the scanned one (on the fourth platform scan the scan lay a median 0.007 mm from it, p95 0.04 mm) and
   rounds off what was not seen;
2. its largest piece, without degenerate, duplicated or non-manifold faces (Open3D's clean-ups);
3. at the scan's base plane (and the crop box's faces): everything beyond it cut away (triangles crossing
   it split) and each cut loop capped flat, so the object stands as it stood (Poisson bulged below the
   stand into a toothed fringe, where most of its defects were); one plane after the other;
4. the boundary loops left fan-filled, up to SMALL_LOOP edges each.

Open3D's own fill_holes (VTK) was tried first: 110 s on that mesh, and still open and self-intersecting.
"""
import time

import numpy as np

POISSON_DEPTH = 10  # Poisson's octree depth: ~0.25 mm cells over a 250 mm object
SMALL_LOOP = 200  # leftover boundary loops of up to this many edges are fan-filled
ADDED_MM = 1.0  # closed surface further than this from the scanned one counts as added
ON_PLANE_MM = 1e-3  # vertices this close to the cut plane count as below it: no cut point lands within float32's
    # reach of a vertex (~3e-5 mm at 400 mm; at 1e-6 one did, and the STL had two triangles collapsed)
REPAIR_ROUNDS = 6  # repair: rounds of opening a pinched spot a little wider and filling it again


def close(vertices, triangles, colours=None, plane=None, depth=POISSON_DEPTH):
    """(vertices mm, triangles, colours uint8, report) of a closed solid around a fused surface (world mm,
    triangles facing out); `plane` (n, c): the base, n . x + c > 0 above it (scene.Scene.plane), a list of
    such planes (fusion.printing_planes), or None. The report says whether the result is closed
    (`edges_closed`) and how much surface was added."""
    import open3d as o3d
    from .pointcloud import vertex_normals
    t0 = time.time()
    vertices = np.asarray(vertices, np.float64)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices))
    pcd.normals = o3d.utility.Vector3dVector(vertex_normals(vertices, triangles))
    if colours is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colours, np.float64) / 255)
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    labels, counts, _ = mesh.cluster_connected_triangles()
    mesh.remove_triangles_by_mask(np.asarray(labels) != int(np.argmax(counts)))
    mesh.remove_unreferenced_vertices()
    v, t = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    rgb = np.asarray(mesh.vertex_colors) * 255 if colours is not None and mesh.has_vertex_colors() else None
    planes = [] if plane is None else plane if isinstance(plane, list) else [plane]
    report = {"poisson_depth": depth, "cut": len(planes)}
    for plane in planes:
        v, t, rgb = clip(v, t, rgb, plane)
        t = np.concatenate([t, cap(v, t, plane)])
    v, t, rgb, filled = fill_loops(v, t, rgb)
    v, t, rgb = weld(v, t, rgb)
    v, t, rgb, rounds = repair(v, t, rgb)
    report.update(loops_filled=filled, repair_rounds=rounds, edges_closed=edges_closed(t), triangles=int(len(t)))
    report.update(added(vertices, triangles, v, t), seconds=round(time.time() - t0, 1))
    return v, t, None if rgb is None else np.clip(rgb, 0, 255).astype(np.uint8), report


def clip(v, t, rgb, plane):
    """The part of a mesh above `plane` (n . x + c > 0): triangles across it split at the plane, the new
    vertices shared by the two triangles of each cut edge (so the cut loops come out connected)."""
    n, c = np.asarray(plane[0], float), float(plane[1])
    h = v @ n + c
    h[np.abs(h) <= ON_PLANE_MM] = -ON_PLANE_MM
    above = h > 0
    up = above[t].sum(1)
    keep = [t[up == 3]]
    mixed = t[(up == 1) | (up == 2)]
    a = above[mixed]
    odd = np.where(up[(up == 1) | (up == 2)] == 1, np.argmax(a, 1), np.argmin(a, 1))  # the one on its own side
    o, p, q = (mixed[np.arange(len(mixed)), (odd + k) % 3] for k in range(3))  # (cyclic order kept)
    cut_v, cut_rgb, index = [v], [rgb] if rgb is not None else None, {}
    count = [len(v)]

    def cut(i, j):  # the vertex where edge i-j crosses the plane, one per edge
        key = (min(i, j), max(i, j))
        if key not in index:
            s = h[i] / (h[i] - h[j])
            cut_v.append((v[i] + s * (v[j] - v[i]))[None])
            if cut_rgb is not None:
                cut_rgb.append((rgb[i] + s * (rgb[j] - rgb[i]))[None])
            index[key] = count[0]
            count[0] += 1
        return index[key]

    lone_above = above[o]
    new = []
    for oi, pi, qi, top in zip(o.tolist(), p.tolist(), q.tolist(), lone_above.tolist()):
        op, oq = cut(oi, pi), cut(oi, qi)
        if top:  # o alone above: the corner at o
            new.append((oi, op, oq))
        else:  # p and q above: the quad op, p, q, oq
            new += [(op, pi, qi), (op, qi, oq)]
    t = np.concatenate(keep + [np.array(new, np.int64).reshape(-1, 3)])
    v = np.concatenate(cut_v)
    rgb = None if cut_rgb is None else np.concatenate(cut_rgb)
    used = np.zeros(len(v), bool)
    used[t] = True
    remap = np.cumsum(used) - 1
    return v[used], remap[t], None if rgb is None else rgb[used]


def boundary_loops(t):
    """The mesh's boundary loops, each a list of vertex indices in the direction its triangles run along
    them (a closed manifold mesh has none)."""
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    key = lambda a, b: a.astype(np.int64) * (int(t.max()) + 1) + b
    twin = np.isin(key(e[:, 1], e[:, 0]), key(e[:, 0], e[:, 1]))
    nxt = {int(a): int(b) for a, b in e[~twin]}
    loops, seen = [], set()
    for start in nxt:
        if start in seen:
            continue
        loop, x = [], start
        while x not in seen and x in nxt:
            seen.add(x)
            loop.append(x)
            x = nxt[x]
        if loop and x == start:
            loops.append(loop)
    return loops


def cap(v, t, plane):
    """Flat caps over the boundary loops lying on `plane`, facing down out of the solid (ear clipping)."""
    n, c = np.asarray(plane[0], float), float(plane[1])
    a = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    out = []
    for loop in boundary_loops(t):
        pts = v[loop]
        if np.abs(pts @ n + c).max() > 1e-3:
            continue  # not a cut loop
        out.append(ear_clip(np.c_[pts @ a, pts @ b], np.array(loop))[:, ::-1])  # against the loop: facing down
    return np.concatenate(out) if out else np.empty((0, 3), np.int64)


def ear_clip(xy, ids):
    """Triangles (as `ids`) covering a simple polygon xy (N, 2), in the polygon's winding."""
    xy, ids = np.asarray(xy, float), list(ids)
    order = list(range(len(ids)))
    area = 0.5 * np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1])
    sign = 1.0 if area >= 0 else -1.0
    tris, guard = [], 0
    while len(order) > 3 and guard < 4 * len(ids) * len(ids):
        guard += 1
        m = len(order)
        found = False
        for k in range(m):
            i, j, l = order[k - 1], order[k], order[(k + 1) % m]
            pa, pb, pc = xy[i], xy[j], xy[l]
            cross = (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])
            if cross * sign <= 1e-12:
                continue  # reflex or flat corner
            rest = np.array([x for x in order if x not in (i, j, l)])
            if len(rest):
                p = xy[rest]
                d1 = (pb[0] - pa[0]) * (p[:, 1] - pa[1]) - (pb[1] - pa[1]) * (p[:, 0] - pa[0])
                d2 = (pc[0] - pb[0]) * (p[:, 1] - pb[1]) - (pc[1] - pb[1]) * (p[:, 0] - pb[0])
                d3 = (pa[0] - pc[0]) * (p[:, 1] - pc[1]) - (pa[1] - pc[1]) * (p[:, 0] - pc[0])
                if np.any((d1 * sign >= 0) & (d2 * sign >= 0) & (d3 * sign >= 0)):
                    continue  # another corner inside: not an ear
            tris.append((ids[i], ids[j], ids[l]))
            order.pop(k)
            found = True
            break
        if not found:  # (a degenerate polygon: close it as a fan)
            tris += [(ids[order[0]], ids[order[s]], ids[order[s + 1]]) for s in range(1, len(order) - 1)]
            order = order[:3] if len(order) < 3 else []
            break
    if len(order) == 3:
        tris.append(tuple(ids[x] for x in order))
    return np.array(tris, np.int64).reshape(-1, 3)


def fill_loops(v, t, rgb, small=SMALL_LOOP):
    """The boundary loops of up to `small` edges closed by a fan around their centre; (v, t, rgb, count)."""
    new_v, new_rgb, new_t, filled = [v], [rgb] if rgb is not None else None, [t], 0
    nv = len(v)
    for loop in boundary_loops(t):
        if len(loop) > small:
            continue
        centre = nv
        new_v.append(v[loop].mean(0)[None])
        if new_rgb is not None:
            new_rgb.append(rgb[loop].mean(0)[None])
        nv += 1
        new_t.append(np.array([(loop[(k + 1) % len(loop)], loop[k], centre) for k in range(len(loop))], np.int64))
        filled += 1
    return (np.concatenate(new_v), np.concatenate(new_t), None if new_rgb is None else np.concatenate(new_rgb),
            filled)


def repair(v, t, rgb, rounds=REPAIR_ROUNDS):
    """Close what fill_loops could not: around a vertex with two open edges leaving it (a pinched boundary,
    no simple loop to follow) or an edge used twice in one direction (overlapping fills), the triangles are
    taken out and the simple hole left is filled again. (On the fourth platform scan: one spot of Poisson's,
    136 mm up the hand, 6 open edges.) Returns (v, t, rgb, rounds used)."""
    for k in range(rounds):
        if edges_closed(t):
            return v, t, rgb, k
        e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]).astype(np.int64)
        m = int(t.max()) + 1
        fwd, back = e[:, 0] * m + e[:, 1], e[:, 1] * m + e[:, 0]
        uf, cf = np.unique(fwd, return_counts=True)
        doubled = np.isin(fwd, uf[cf > 1])
        open_ = ~np.isin(back, fwd)
        starts, counts = np.unique(e[open_, 0], return_counts=True)
        bad = np.unique(np.concatenate([e[doubled].ravel(), starts[counts > 1]]))
        if not len(bad):  # simple loops too long for fill_loops: fill them all
            v, t, rgb, _ = fill_loops(v, t, rgb, small=len(t))
            continue
        t = t[~np.isin(t, bad).any(1)]
        v, t, rgb, _ = fill_loops(v, t, rgb)
    return v, t, rgb, rounds


def weld(v, t, rgb):
    """Vertices merged where float32 (as an STL stores them, and a slicer joins them by position) cannot
    tell them apart, and the triangles that collapses dropped; repair closes what that opens."""
    u, first, inv = np.unique(v.astype(np.float32), axis=0, return_index=True, return_inverse=True)
    if len(u) == len(v):
        return v, t, rgb
    t = inv.reshape(-1)[t]
    t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])]
    return v[first], t, None if rgb is None else rgb[first]


def edges_closed(t):
    """Is every edge used by exactly two triangles, once in each direction (a closed, oriented surface)?"""
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]).astype(np.int64)
    m = int(t.max()) + 1
    fwd = e[:, 0] * m + e[:, 1]
    back = e[:, 1] * m + e[:, 0]
    return bool(len(np.unique(fwd)) == len(fwd) and np.isin(back, fwd).all())


def added(vertices, triangles, v, t):
    """How much of the closed surface lies further than ADDED_MM from the scanned one: its area (mm2) and
    the largest distance (mm)."""
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(vertices, np.float32)), o3d.core.Tensor(np.asarray(triangles, np.uint32)))
    centres = v[t].mean(1).astype(np.float32)
    d = scene.compute_distance(o3d.core.Tensor(centres)).numpy()
    area = 0.5 * np.linalg.norm(np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]), axis=1)
    far = d > ADDED_MM
    return {"added_mm2": round(float(area[far].sum())), "added_share": round(float(area[far].sum() / area.sum()), 4),
            "added_max_mm": round(float(d.max()), 1) if len(d) else 0.0}
