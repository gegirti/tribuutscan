"""Joint pose refinement: point-to-plane ICP, marker terms and a motion prior in one solve. numpy +
Open3D (.venv) for downsampling, normals and nearest-neighbour correspondences.

Shape alone cannot hold a flat object: sliding along a flat face costs nothing in point-to-plane
ICP. Markers (where seen) and a weak prior toward the predicted motion fill exactly the directions
the shape leaves free, while the thousands of shape terms dominate everywhere else. The solve also
reports how well the shape pins each direction (`degeneracy`: smallest / largest eigenvalue of the
6x6 shape information, rotation scaled by the object's radius); a small value means the shape alone
cannot be trusted, and the scanner can say "flat area: aim at an edge or a marker".

Poses here: 4x4 transforms with translation in metres (like fuse.icp); residuals in mm.
"""
import numpy as np
import open3d as o3d

from . import fuse

TUKEY_MM = 3.0  # point-to-plane residuals beyond this get no weight
POINT_SIGMA_MM = 1.0  # depth noise (estimate)
MARKER_SIGMA_MM = 1.0
MARKER_GATE_SIGMAS = (10.0, 3.0)  # a marker beyond this many sigmas is ignored: loose on the coarse
                                  # levels (the start may be off), 3 sigma on the fine level
PRIOR_MM, PRIOR_DEG = 20.0, 5.0  # how far the solve may wander from the predicted motion, softly
ITERATIONS = (8, 8, 10)  # per ICP level, coarse to fine
DEGENERATE = 0.02  # smallest / largest scaled shape eigenvalue below this: the shape slides


OUTLIER_NEIGHBOURS, OUTLIER_RADIUS_M = 8, 0.004  # a point needs 8 others within 4 mm (surfaces have ~200)


def levels(points_m):
    """Per-level clouds (like fuse.levels) after dropping stray points: scattered clutter and flying
    pixels outnumber the surface once it is downsampled, and biased the coarse levels by 10+ mm."""
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points_m, np.float64)))
    cloud, _ = cloud.remove_radius_outlier(nb_points=OUTLIER_NEIGHBOURS, radius=OUTLIER_RADIUS_M)
    return fuse.levels(cloud)


def hat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def exp_se3(xi_rad_mm):
    """4x4 (translation in metres) for a twist (rotation vector rad, translation mm)."""
    w, v = xi_rad_mm[:3], xi_rad_mm[3:] / 1000
    a = np.linalg.norm(w)
    r = np.eye(3)
    if a > 1e-12:
        k = hat(w / a)
        r = np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = r, v
    return t


def log_rot(r):
    a = np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))
    if a < 1e-9:
        return np.zeros(3)
    return a / (2 * np.sin(a)) * np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])


def solve(source, target, init, prior=None, marker_pairs=None, radius_mm=60.0, marker_sigma_mm=MARKER_SIGMA_MM):
    """Refine `init` (source camera -> target frame, metres) jointly.

    source, target: per-level Open3D clouds (fuse.levels), target with normals.
    prior: the predicted transform (soft pull, PRIOR_MM / PRIOR_DEG) or None.
    marker_pairs: (src_mm (N, 3) in source camera, dst_mm (N, 3) in the target frame) or None.
    Returns (transform, info) with info: fitness, rmse_mm, degeneracy, markers_used, marker_rmse_mm.
    """
    t = init.copy()
    w_prior = None
    if prior is not None:
        w_prior = np.r_[np.full(3, 1 / np.radians(PRIOR_DEG) ** 2), np.full(3, 1 / PRIOR_MM ** 2)]
    last = None  # (h_shape, residuals, n correspondences, n source points, markers used, marker residuals)
    for level, ((voxel, max_dist), src, tgt, iters) in enumerate(zip(fuse.ICP_LEVELS, source, target, ITERATIONS)):
        gate_mm = (MARKER_GATE_SIGMAS[0] if level < len(ITERATIONS) - 1 else MARKER_GATE_SIGMAS[1]) * marker_sigma_mm
        p = np.asarray(src.points)
        q, nq = np.asarray(tgt.points), np.asarray(tgt.normals)
        for _ in range(iters):
            corr = np.asarray(o3d.pipelines.registration.evaluate_registration(
                src, tgt, max_dist, t).correspondence_set)
            if len(corr) < 10:
                return t, summarise(last, radius_mm)
            x = (p[corr[:, 0]] @ t[:3, :3].T + t[:3, 3]) * 1000  # mm, in the target frame
            qq, n = q[corr[:, 1]] * 1000, nq[corr[:, 1]]
            r = np.einsum("ij,ij->i", n, x - qq)
            dist = np.linalg.norm(x - qq, axis=1) / (max_dist * 1000)  # 0..1 of the match distance
            w = (np.where(np.abs(r) < TUKEY_MM, (1 - (r / TUKEY_MM) ** 2) ** 2, 0.0)
                 * (1 - dist ** 2) ** 2) / POINT_SIGMA_MM ** 2  # far matches (beyond a face's edge) count less
            j = np.c_[np.cross(x, n), n]  # d r / d(rotation rad, translation mm), left perturbation
            h_shape = (j * w[:, None]).T @ j
            h, g = h_shape.copy(), (j * (w * r)[:, None]).sum(0)
            used, mres = 0, None
            if marker_pairs is not None and len(marker_pairs[0]):
                m = marker_pairs[0] @ t[:3, :3].T + t[:3, 3] * 1000
                rm = m - marker_pairs[1]
                ok = np.linalg.norm(rm, axis=1) < gate_mm
                for xm, res in zip(m[ok], rm[ok]):
                    jm = np.c_[-hat(xm), np.eye(3)]
                    h += jm.T @ jm / marker_sigma_mm ** 2
                    g += jm.T @ res / marker_sigma_mm ** 2
                used, mres = int(ok.sum()), rm[ok]
            if w_prior is not None:
                d = np.linalg.inv(prior) @ t
                e = np.r_[log_rot(d[:3, :3]), d[:3, 3] * 1000]
                h += np.diag(w_prior)
                g += w_prior * e
            centre = x.mean(0)  # shape information about the object's centre, for the degeneracy measure
            jc = np.c_[np.cross(x - centre, n), n]
            last = ((jc * w[:, None]).T @ jc, r, len(corr), len(p), used, mres)
            h += np.eye(6) * 1e-9 * max(np.trace(h), 1e-9)  # Levenberg-Marquardt damping: stays defined on a flat face
            xi = -np.linalg.solve(h, g)
            t = exp_se3(xi) @ t
            if np.linalg.norm(xi[:3]) < 1e-5 and np.linalg.norm(xi[3:]) < 1e-3:
                break
    return t, summarise(last, radius_mm)


def summarise(last, radius_mm):
    """Fitness, RMSE, degeneracy and marker use from the final iteration."""
    if last is None:
        return {"fitness": 0.0, "rmse_mm": None, "degeneracy": None, "markers_used": 0, "marker_rmse_mm": None}
    h_shape, r, n_corr, n_src, used, mres = last
    scale = np.diag([1 / radius_mm] * 3 + [1.0] * 3)  # rotation as displacement at the object's edge
    ev = np.linalg.eigvalsh(scale @ h_shape @ scale)
    inl = np.abs(r) < TUKEY_MM
    return {"fitness": n_corr / max(n_src, 1),
            "rmse_mm": float(np.sqrt(np.mean(r[inl] ** 2))) if inl.any() else None,
            "degeneracy": float(max(ev[0], 0) / ev[-1]) if ev[-1] > 0 else 0.0,
            "markers_used": used,
            "marker_rmse_mm": float(np.sqrt(np.mean(np.sum(mres ** 2, 1)))) if used else None}
