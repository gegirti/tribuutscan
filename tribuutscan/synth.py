"""Synthetic scans with known motion: a flat-faced box on a table, with markers, along a tilting path.

    .venv/bin/python -m tribuutscan.synth [out_dir] [frames] [every] [hard] [--plate] [--pattern MM] [--y14]

Rendered with Open3D's RaycastingScene into the viewer's raw recording format (YUYV colour, packed
depth through the estimated depth law, blank IR), always with the estimated camera (a pinhole; it
is the synthetic world's camera, whatever is calibrated: evaluate.py reads these with it too) plus gt_poses.txt (camera -> first camera, metres,
x right / y down / z forward, as cloud.Scanner keeps poses). Depth gets noise, dropouts, flying
pixels at edges and the estimated depth window, so tracking sees roughly what the scanner gives.
`every` keeps every n-th frame (3 gives the ~7 fps of a slow live worker). `hard` adds what broke
the real tilt scan: a sudden 6 degree jerk, and a stretch where the scanner drifts beyond the depth
window (tracking is lost) and comes back (it must be re-found in the right place).

Closer to the real sensor (options; the defaults render the first datasets byte for byte):
- `pattern_mm`: a fixed per-pixel depth error, the same in every frame. The scanner's flat-target RMS
  is 0.14 / 0.23 / 0.36 mm at 20 / 25 / 30 cm and nearly fixed over time (CLAUDE.md), i.e. a constant
  error in disparity: given here as the RMS at 20 cm, growing with z^2. Fusion cannot average it away
  the way it averages frame noise. Its pattern (two blurred noise fields, 4 and 24 px) is an estimate.
- `noise_mm`: frame-to-frame noise (the scanner's 14-bit depth: 0.015-0.07 mm).
- `bits=14`: 14-bit depth (usbdepth.py), on the same law.
- scene "plate": a 1.8 mm plate standing on the table, markers on both faces, seen from both sides
  (a path half way round it): fusion must keep both faces apart.
Every set also gets true_surface.ply: the object (without the table) in the first camera's frame,
in cloud.ply's convention (metres, y up, z toward the scanner), so outputs can be measured in place.
"""
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.core as o3c

from . import calib
from . import frames as fr
from .calib import CAMERA
from .depth import encode_y10, encode_y14

W, H = fr.W, fr.H
BOX = np.array([0.12, 0.08, 0.06])  # flat-faced object, metres, standing on the table at z = 0
NOISE_MM = 0.5  # depth noise (estimate of the scanner's)
TOP_MARKERS = [(-0.042, 0.021), (0.031, -0.026), (0.047, 0.029), (-0.013, -0.031), (0.004, 0.012)]
SIDE_MARKERS = [(0.02, 0.035), (-0.03, 0.018)]  # on the +x face: (y, z)
TABLE_MARKERS = [(-0.11, 0.07), (0.1, -0.08), (0.13, 0.05), (-0.09, -0.09)]
ALBEDO = {"table": 185, "box": 45, "ring": 15, "dot": 245}
PLATE = np.array([0.0018, 0.10, 0.07])  # thin plate standing on the table: thickness (x), width (y), height (z)
PLATE_MARKERS = [(-0.03, 0.045), (0.025, 0.02), (0.035, 0.055)]  # (y, z) on each face, mirrored on the back


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


def object_parts(kind="box"):
    """[(mesh, material)] of the object and the markers on it (world, metres), and the marker places."""
    if kind == "plate":
        body = o3d.geometry.TriangleMesh.create_box(*PLATE)
        body.translate((-PLATE[0] / 2, -PLATE[1] / 2, 0))
        places = [((PLATE[0] / 2, y, z), (1, 0, 0)) for y, z in PLATE_MARKERS]
        places += [((-PLATE[0] / 2, -y, z), (-1, 0, 0)) for y, z in PLATE_MARKERS]
    else:
        body = o3d.geometry.TriangleMesh.create_box(*BOX)
        body.translate((-BOX[0] / 2, -BOX[1] / 2, 0))
        places = [((x, y, BOX[2]), (0, 0, 1)) for x, y in TOP_MARKERS]
        places += [((BOX[0] / 2, y, z), (1, 0, 0)) for y, z in SIDE_MARKERS]
    return [(body, "box")] + marker_meshes(places), places


def marker_meshes(places):
    parts = []
    for centre, normal in places:
        c, n = np.array(centre), np.array(normal, float)
        parts += [(disc(0.005, 0.0002, c + n * 0.0001, n), "ring"), (disc(0.003, 0.0002, c + n * 0.0003, n), "dot")]
    return parts


def build_scene(kind="box"):
    """RaycastingScene plus a map from geometry id to material name, and marker centres (world, metres)."""
    scene, material = o3d.t.geometry.RaycastingScene(), {}
    table = o3d.geometry.TriangleMesh.create_box(0.6, 0.6, 0.01)
    table.translate((-0.3, -0.3, -0.01))
    parts, places = object_parts(kind)
    table_places = [((x, y, 0.0), (0, 0, 1)) for x, y in TABLE_MARKERS]
    for mesh, name in [(table, "table"), parts[0]] + parts[1:] + marker_meshes(table_places):
        material[scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))] = name
    return scene, material, np.array([c for c, _ in places + table_places], float)


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


def path_around(n):
    """Half way round a standing plate: azimuth -40..220 deg (both faces, edge-on at 90), elevation 30 deg."""
    t = np.linspace(0, 1, n)
    az, el = np.radians(-40 + 260 * t), np.radians(30 + 5 * np.sin(2 * np.pi * t))
    radius = 0.24 + 0.01 * np.sin(3 * np.pi * t)
    shake = 0.0015 * np.stack([np.sin(23 * t), np.sin(31 * t + 1), np.sin(19 * t + 2)], 1)
    target = np.array([0, 0, PLATE[2] / 2])
    eye = target + radius[:, None] * np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], 1)
    return [look_at(e + s, target) for e, s in zip(eye, shake)]


def fixed_pattern(rms_mm_at_200, seed):
    """Per-pixel error in raw (disparity) units, the same in every frame; its depth RMS is
    rms_mm_at_200 at 20 cm and grows with z^2 (dz = z^2 / a per raw unit)."""
    rng = np.random.default_rng(seed + 7919)  # apart from the frame noise: those draws stay as they were
    fy, fx = np.fft.fftfreq(H)[:, None], np.fft.fftfreq(W)[None, :]
    field = np.zeros((H, W))
    for sigma_px in (4.0, 24.0):
        blur = np.exp(-2 * (np.pi * sigma_px) ** 2 * (fx ** 2 + fy ** 2))
        f = np.fft.ifft2(np.fft.fft2(rng.normal(size=(H, W))) * blur).real
        field += f / f.std()
    return field / field.std() * rms_mm_at_200 * CAMERA.a / 200.0 ** 2


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


def render(scene, material, cam_to_world, rng, noise_mm=NOISE_MM, pattern=None, bits=10):
    """(raw depth (H, W) uint16, colour RGB (H, W, 3) uint8 in colour-camera pixels) for one view.
    pattern: fixed raw offsets (fixed_pattern) or None; bits 14: raw is Y14 ((Y10 x 16) + 8)."""
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
    zn = z + rng.normal(0, noise_mm, z.shape)
    near, far = CAMERA.window_mm
    ok = valid & (zn >= near) & (zn <= far) & (rng.random(z.shape) > 0.01)
    jump = np.zeros_like(ok)
    zz = np.where(valid, z, 1e4)
    jump[:, 1:-1] |= np.abs(zz[:, 2:] - zz[:, :-2]) > 8
    jump[1:-1] |= np.abs(zz[2:] - zz[:-2]) > 8
    ok &= ~(jump & (rng.random(z.shape) < 0.6))
    raw = np.zeros((H, W), np.uint16)
    v = CAMERA.raw_from_mm(zn[ok]) + (0 if pattern is None else pattern[ok])
    if bits == 14:
        raw[ok] = np.clip(np.round(v * 16 + 8), 24, 16376).astype(np.uint16)
    else:
        raw[ok] = np.clip(np.round(v), 1, 1023).astype(np.uint16)
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


def generate(out_dir, n_frames=160, every=1, seed=0, hard=False, kind="box", pattern_mm=0.0, noise_mm=NOISE_MM,
             bits=10):
    """Write a synthetic recording to out_dir (recording format + gt_poses.txt + markers_gt.json +
    true_surface.ply). The new options (module docstring) default to the first datasets' rendering."""
    calibrated = CAMERA.spec
    calib.use(calib.ESTIMATES)
    try:
        return _generate(Path(out_dir), n_frames, every, seed, hard, kind, pattern_mm, noise_mm, bits)
    finally:
        calib.use(calibrated)


def first_camera(kind="box", hard=False):
    """Camera -> world of a set's first frame (the start of its path, whatever the frame count)."""
    return (path_around(2) if kind == "plate" else path(2, hard))[0]


def _generate(out_dir, n_frames, every, seed, hard, kind, pattern_mm, noise_mm, bits):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    scene, material, marker_xyz = build_scene(kind)
    cams = (path_around(n_frames) if kind == "plate" else path(n_frames, hard))[::every]
    first_inv = np.linalg.inv(cams[0])
    pattern = fixed_pattern(pattern_mm, seed) if pattern_mm else None
    bpl = W * bits // 8
    blank_ir = (kind, pattern_mm, bits) == ("box", 0.0, 10)  # the first datasets kept a blank IR file
    with open(out_dir / "color.raw", "wb") as fc, open(out_dir / "depth.raw", "wb") as fd, \
            open(out_dir / "stamps.csv", "w") as fs, open(out_dir / "gt_poses.txt", "w") as fg:
        fi = open(out_dir / "ir.raw", "wb") if blank_ir else None
        fs.write("index,color_t,ir_t,depth_t\n")
        for i, cam in enumerate(cams):
            raw, colour = render(scene, material, cam, rng, noise_mm, pattern, bits)
            fc.write(rgb_to_yuyv(colour))
            if fi:
                fi.write(bytes(fr.FRAME_BYTES["ir"]))
            fd.write(encode_y14(raw) if bits == 14 else encode_y10(raw))
            t = i * every / 20.0  # 20 fps camera
            fs.write(f"{i},{t:.4f},{t:.4f},{t:.4f}\n")
            fg.write(f"{i} " + " ".join(f"{v:.9f}" for v in (first_inv @ cam).ravel()) + "\n")
        if fi:
            fi.close()
    info = {"size": [W, H], "depth_bytesperline": bpl, "frame_bytes": {**fr.FRAME_BYTES, "depth": bpl * H},
            "note": "synthetic: flat-faced box on a table with markers, tilting path; ir is blank",
            "frames": len(cams), "every": every, "seed": seed, "hard": hard}
    if not blank_ir:
        info.update(streams=["color", "depth"], kind=kind, pattern_mm_at_200=pattern_mm, noise_mm=noise_mm,
                    note=f"synthetic: {'thin plate seen from both sides' if kind == 'plate' else 'box, tilting path'}, "
                         f"{bits}-bit depth, fixed pattern {pattern_mm} mm RMS at 20 cm, frame noise {noise_mm} mm")
    (out_dir / "info.json").write_text(json.dumps(info, indent=2))
    world_to_first = first_inv
    (out_dir / "markers_gt.json").write_text(json.dumps(
        [list((world_to_first[:3, :3] @ m + world_to_first[:3, 3]) * 1000) for m in marker_xyz]))
    write_true_surface(out_dir, kind, cams[0])
    return out_dir


def write_true_surface(out_dir, kind, first_cam):
    """true_surface.ply: the object and its markers (no table) in the first camera's frame, as cloud.ply
    (metres, y up, z toward the scanner)."""
    to_first = np.diag([1.0, -1.0, -1.0, 1.0]) @ np.linalg.inv(first_cam)  # cloud.FLIP after world -> first camera
    surface = o3d.geometry.TriangleMesh()
    for mesh, _ in object_parts(kind)[0]:
        surface += mesh
    surface.transform(to_first)
    o3d.io.write_triangle_mesh(str(Path(out_dir) / "true_surface.ply"), surface)


def truth_distance(directory, xyz_mm):
    """Distances (mm) from points in a synthetic scan's world (the first camera's frame, mm, as the
    Scanner keeps them) to the set's true surface; writes true_surface.ply first if an older set lacks it."""
    directory = Path(directory)
    surface_file = directory / "true_surface.ply"
    if not surface_file.exists():
        info = json.loads((directory / "info.json").read_text())
        write_true_surface(directory, info.get("kind", "box"), first_camera(info.get("kind", "box"), info.get("hard", False)))
    mesh = o3d.io.read_triangle_mesh(str(surface_file))
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) @ np.diag([1000.0, -1000.0, -1000.0]))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene.compute_distance(o3d.core.Tensor(np.asarray(xyz_mm, np.float32))).numpy()


def read_truth(directory):
    """Ground-truth poses of a synthetic recording: dict index -> 4x4 (metres)."""
    t = np.loadtxt(Path(directory) / "gt_poses.txt", ndmin=2)
    return {int(row[0]): row[1:].reshape(4, 4) for row in t}


def main():
    args, options = [], {}
    rest = iter(sys.argv[1:])
    for a in rest:
        if a == "--pattern":
            options["pattern_mm"] = float(next(rest))
        elif a == "--plate":
            options["kind"] = "plate"
        elif a == "--y14":
            options["bits"] = 14
        else:
            args.append(a)
    out = Path(args[0]) if args else fr.CAPTURES / "synth" / "box-tilt-rec"
    n = int(args[1]) if len(args) > 1 else 160
    every = int(args[2]) if len(args) > 2 else 1
    hard = len(args) > 3 and args[3] == "hard"
    print(f"{generate(out, n, every, hard=hard, **options)}: {len(range(0, n, every))} frames")


if __name__ == "__main__":
    main()


def _to_world(directory):
    """Scan world (the first camera's frame, metres) -> synthetic world (table at z = 0), 4x4."""
    info = json.loads((Path(directory) / "info.json").read_text())
    return first_camera(info.get("kind", "box"), info.get("hard", False))


def height_above_table(directory, xyz_mm):
    """Height (mm) above the synthetic table of points in a scan's world (mm)."""
    m = _to_world(directory)
    return (np.asarray(xyz_mm) / 1000 @ m[:3, :3].T + m[:3, 3])[:, 2] * 1000


def plate_faces(directory, spacing_mm=0.5):
    """Points (scan world, mm) every spacing_mm on the plate's two large faces, front (+x) then back."""
    ys = np.arange(-PLATE[1] / 2, PLATE[1] / 2, spacing_mm / 1000)
    zs = np.arange(0.002, PLATE[2], spacing_mm / 1000)  # from 2 mm up: the foot meets the table
    y, z = [a.ravel() for a in np.meshgrid(ys, zs)]
    to_scan = np.linalg.inv(_to_world(directory))
    faces = []
    for x in (PLATE[0] / 2, -PLATE[0] / 2):
        p = np.stack([np.full_like(y, x), y, z], 1)
        faces.append((p @ to_scan[:3, :3].T + to_scan[:3, 3]) * 1000)
    return faces
