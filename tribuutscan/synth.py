"""Synthetic scans with known motion: a flat-faced box on a table, with markers, along a tilting path.

    .venv/bin/python -m tribuutscan.synth [out_dir] [frames] [every] [hard]

Rendered with Open3D's RaycastingScene into the viewer's raw recording format (YUYV colour, packed
depth through the estimated depth law, blank IR) plus gt_poses.txt (camera -> first camera, metres,
x right / y down / z forward, as cloud.Scanner keeps poses). Depth gets noise, dropouts, flying
pixels at edges and the estimated depth window, so tracking sees roughly what the scanner gives.
`every` keeps every n-th frame (3 gives the ~7 fps of a slow live worker). `hard` adds what broke
the real tilt scan: a sudden 6 degree jerk, and a stretch where the scanner drifts beyond the depth
window (tracking is lost) and comes back (it must be re-found in the right place).
"""
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.core as o3c

from . import frames as fr
from . import pointcloud as pc
from .calib import CAMERA
from .depth import encode_y10

W, H = fr.W, fr.H
BOX = np.array([0.12, 0.08, 0.06])  # flat-faced object, metres, standing on the table at z = 0
NOISE_MM = 0.5  # depth noise (estimate of the scanner's)
TOP_MARKERS = [(-0.042, 0.021), (0.031, -0.026), (0.047, 0.029), (-0.013, -0.031), (0.004, 0.012)]
SIDE_MARKERS = [(0.02, 0.035), (-0.03, 0.018)]  # on the +x face: (y, z)
TABLE_MARKERS = [(-0.11, 0.07), (0.1, -0.08), (0.13, 0.05), (-0.09, -0.09)]
ALBEDO = {"table": 185, "box": 45, "ring": 15, "dot": 245}


def disc(radius, height, centre, normal):
    """A thin cylinder lying on a face: centred at `centre`, axis along `normal`."""
    m = o3d.geometry.TriangleMesh.create_cylinder(radius, height, resolution=32)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, normal)
    if np.linalg.norm(axis) > 1e-9:
        angle = np.arccos(np.clip(z @ normal, -1, 1))
        m.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(axis / np.linalg.norm(axis) * angle), center=(0, 0, 0))
    m.translate(centre)
    return m


def build_scene():
    """RaycastingScene plus a map from geometry id to material name, and marker centres (world, metres)."""
    scene, material, marker_xyz = o3d.t.geometry.RaycastingScene(), {}, []

    def add(mesh, name):
        material[scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))] = name

    table = o3d.geometry.TriangleMesh.create_box(0.6, 0.6, 0.01)
    table.translate((-0.3, -0.3, -0.01))
    add(table, "table")
    box = o3d.geometry.TriangleMesh.create_box(*BOX)
    box.translate((-BOX[0] / 2, -BOX[1] / 2, 0))
    add(box, "box")
    places = [((x, y, BOX[2]), (0, 0, 1)) for x, y in TOP_MARKERS]
    places += [((BOX[0] / 2, y, z), (1, 0, 0)) for y, z in SIDE_MARKERS]
    places += [((x, y, 0.0), (0, 0, 1)) for x, y in TABLE_MARKERS]
    for centre, normal in places:
        c, n = np.array(centre), np.array(normal, float)
        add(disc(0.005, 0.0002, c + n * 0.0001, n), "ring")
        add(disc(0.003, 0.0002, c + n * 0.0003, n), "dot")
        marker_xyz.append(c)
    return scene, material, np.array(marker_xyz)


def look_at(eye, target):
    """Camera -> world 4x4 for a camera at `eye` looking at `target` (x right, y down, z forward)."""
    f = target - eye
    f /= np.linalg.norm(f)
    x = np.cross(f, (0, 0, 1.0))
    x /= np.linalg.norm(x)
    m = np.eye(4)
    m[:3, :3] = np.stack([x, np.cross(f, x), f], 1)
    m[:3, 3] = eye
    return m


def path(n, hard=False):
    """A hand-held tilt around the box: azimuth -35..40 deg, elevation 60..35 deg, ~25 cm away, shaky."""
    t = np.linspace(0, 1, n)
    az, el = np.radians(-35 + 75 * t), np.radians(60 - 25 * t)
    radius = 0.25 + 0.01 * np.sin(2 * np.pi * t)
    if hard:
        az = az + np.radians(6) * (t > 0.3)  # a jerk between two frames
        radius = radius + 0.09 * np.exp(-(((t - 0.6) / 0.05) ** 2))  # out beyond the window and back
    shake = 0.0015 * np.stack([np.sin(23 * t), np.sin(31 * t + 1), np.sin(19 * t + 2)], 1)
    target = np.array([0, 0, BOX[2] / 2]) + 0.002 * np.stack([np.sin(13 * t), np.cos(17 * t), 0 * t], 1)
    eye = target + radius[:, None] * np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], 1)
    return [look_at(e + s, g) for e, s, g in zip(eye, shake, target)]


def texture(material_name, hit):
    """Albedo per hit point: table has a blotchy print (features), the box a faint grain."""
    base = ALBEDO[material_name]
    if material_name == "table":  # 4 mm cells of random grey: something for feature trackers
        cell = np.floor(hit[:, :2] / 0.004).astype(np.int64)
        noise = ((cell[:, 0] * 73856093) ^ (cell[:, 1] * 19349663)) % 61 - 30
        return base + noise
    if material_name == "box":
        cell = np.floor(hit / 0.001).astype(np.int64)
        return base + ((cell[:, 0] * 73856093 ^ cell[:, 1] * 19349663 ^ cell[:, 2] * 83492791) % 13 - 6)
    return np.full(len(hit), base, float)


def render(scene, material, cam_to_world, rng):
    """(raw depth (H, W) uint16, colour RGB (H, W, 3) uint8 in colour-camera pixels) for one view."""
    world_to_cam = np.linalg.inv(cam_to_world)
    K = o3c.Tensor(CAMERA.intrinsic_matrix(), o3c.float64)
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(K, o3c.Tensor(world_to_cam), W, H)
    ans = scene.cast_rays(rays)
    t = ans["t_hit"].numpy()
    ids = ans["geometry_ids"].numpy().astype(np.int64)
    r = rays.numpy()
    valid = np.isfinite(t)
    hit = r[..., :3] + r[..., 3:] * np.where(valid, t, 0)[..., None]
    z = (hit @ world_to_cam[:3, :3].T + world_to_cam[:3, 3])[..., 2] * 1000  # mm along the camera axis
    shade = np.abs(np.einsum("hwc,hwc->hw", ans["primitive_normals"].numpy(), r[..., 3:] /
                             np.linalg.norm(r[..., 3:], axis=-1, keepdims=True)))
    grey = np.full((H, W), 120.0)
    for gid, name in material.items():
        sel = valid & (ids == gid)
        grey[sel] = texture(name, hit[sel]) * (0.35 + 0.65 * shade[sel])
    # depth: noise, the window, random dropouts, and flying pixels / gaps where depth jumps
    zn = z + rng.normal(0, NOISE_MM, z.shape)
    near, far = pc.WINDOW_CM[0] * 10, pc.WINDOW_CM[1] * 10
    ok = valid & (zn >= near) & (zn <= far) & (rng.random(z.shape) > 0.01)
    jump = np.zeros_like(ok)
    zz = np.where(valid, z, 1e4)
    jump[:, 1:-1] |= np.abs(zz[:, 2:] - zz[:, :-2]) > 8
    jump[1:-1] |= np.abs(zz[2:] - zz[:-2]) > 8
    ok &= ~(jump & (rng.random(z.shape) < 0.6))
    raw = np.zeros((H, W), np.uint16)
    v = np.round(CAMERA.raw_from_mm(zn[ok]))
    raw[ok] = np.clip(v, 1, 1023).astype(np.uint16)
    # colour camera image: sample the depth-grid render through the registration estimate
    ys, xs = np.mgrid[0:H, 0:W]
    u, vv = CAMERA.depth_px(xs, ys)
    u = np.clip(np.round(u).astype(int), 0, W - 1)
    vv = np.clip(np.round(vv).astype(int), 0, H - 1)
    colour = np.clip(grey[vv, u] + rng.normal(0, 2, (H, W)), 0, 255)
    return raw, np.repeat(colour[..., None], 3, axis=2).astype(np.uint8)


def rgb_to_yuyv(rgb):
    """Inverse of image.yuyv_to_rgb (BT.601, full range); chroma averaged over each pixel pair."""
    rgb = rgb.astype(np.float32)
    y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    u = (rgb[..., 2] - y) / 1.772 + 128
    v = (rgb[..., 0] - y) / 1.402 + 128
    out = np.empty((rgb.shape[0], rgb.shape[1] // 2, 4), np.uint8)
    out[..., 0] = np.clip(y[:, 0::2], 0, 255)
    out[..., 1] = np.clip((u[:, 0::2] + u[:, 1::2]) / 2, 0, 255)
    out[..., 2] = np.clip(y[:, 1::2], 0, 255)
    out[..., 3] = np.clip((v[:, 0::2] + v[:, 1::2]) / 2, 0, 255)
    return out.tobytes()


def generate(out_dir, n_frames=160, every=1, seed=0, hard=False):
    """Write a synthetic recording to out_dir (recording format + gt_poses.txt + markers_gt.json)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    scene, material, marker_xyz = build_scene()
    cams = path(n_frames, hard)[::every]
    first_inv = np.linalg.inv(cams[0])
    bpl = W * 10 // 8
    with open(out_dir / "color.raw", "wb") as fc, open(out_dir / "ir.raw", "wb") as fi, \
            open(out_dir / "depth.raw", "wb") as fd, open(out_dir / "stamps.csv", "w") as fs, \
            open(out_dir / "gt_poses.txt", "w") as fg:
        fs.write("index,color_t,ir_t,depth_t\n")
        for i, cam in enumerate(cams):
            raw, colour = render(scene, material, cam, rng)
            fc.write(rgb_to_yuyv(colour))
            fi.write(bytes(fr.FRAME_BYTES["ir"]))
            fd.write(encode_y10(raw))
            t = i * every / 20.0  # 20 fps camera
            fs.write(f"{i},{t:.4f},{t:.4f},{t:.4f}\n")
            fg.write(f"{i} " + " ".join(f"{v:.9f}" for v in (first_inv @ cam).ravel()) + "\n")
    (out_dir / "info.json").write_text(json.dumps({
        "size": [W, H], "depth_bytesperline": bpl, "frame_bytes": fr.FRAME_BYTES,
        "note": "synthetic: flat-faced box on a table with markers, tilting path; ir is blank",
        "frames": len(cams), "every": every, "seed": seed, "hard": hard}, indent=2))
    world_to_first = first_inv
    (out_dir / "markers_gt.json").write_text(json.dumps(
        [list((world_to_first[:3, :3] @ m + world_to_first[:3, 3]) * 1000) for m in marker_xyz]))
    return out_dir


def read_truth(directory):
    """Ground-truth poses of a synthetic recording: dict index -> 4x4 (metres)."""
    t = np.loadtxt(Path(directory) / "gt_poses.txt", ndmin=2)
    return {int(row[0]): row[1:].reshape(4, 4) for row in t}


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else fr.CAPTURES / "synth" / "box-tilt-rec"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 160
    every = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    hard = len(sys.argv) > 4 and sys.argv[4] == "hard"
    print(f"{generate(out, n, every, hard=hard)}: {len(range(0, n, every))} frames")


if __name__ == "__main__":
    main()
