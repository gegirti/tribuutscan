"""The README's GIFs from a saved live scan (kept raw frames): the scan as the viewer showed it, and its mesh turning.

    .venv/bin/python -m tribuutscan.showcase captures/<stamp>-scan [out_dir]   # -> scan.gif, model.gif

scan.gif: the scan's raw copy replayed through the live path (the GPU tracker with its live model view, frames
built as the worker builds them) and drawn with the viewer's own previews: colour on the left, depth with the
model painted on the right; every SCAN_EVERY-th frame. model.gif: mesh.ply turning about the base's up axis
(session.json "crop_axes"), rendered offscreen by Open3D (Filament). Needs .venv (Open3D, wgpu) and ffmpeg.
Check every frame before publishing: the colour view shows the room.
"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

SCAN_EVERY = 9  # every 9th frame of the scan: a ~9 fps live scan plays about 10x faster at GIF_FPS
GIF_FPS = 10
SCAN_WIDTH, SCAN_COLOURS = 720, 128  # (on the 11-05 platform scan: 15 s, 6.5 MB)
TURN_FRAMES, TURN_FPS, TURN_SIZE, TURN_WIDTH = 60, 12, 440, 400
BACKGROUND = [0.082, 0.090, 0.102, 1.0]  # the 3D page's --bg, #15171a


def scan_frames(rec, out, every=SCAN_EVERY):
    """PPMs of the scan as the viewer showed it (c*/d*: colour, depth with the live model); returns the count."""
    from . import cloud as cl
    from . import depth as dp
    from . import frames as fr
    from . import viewer
    s = cl.Scanner("small", log=lambda m: None, track_with="gpu", live_view=True, aim=cl.recording_aim(rec))
    viewer.found_markers["on"] = False

    class Cap:
        width, height, bytesperline = fr.W, fr.H, None

    n = 0
    try:
        for i, b in enumerate(fr.read_raw(rec)):
            Cap.bytesperline = b["bytesperline"]
            depth4, cleared = dp.decode_strided(b["depth"], fr.W, fr.H, b["bytesperline"])
            t_c, _, t_d = b["t"]
            frame = dp.LazyFrame(yuyv=b["color"], depth_raw=b["depth"], bytesperline=b["bytesperline"],
                                 depth4=depth4, cleared=cleared, shape=(fr.H, fr.W), t=t_d, skew_ms=(t_c - t_d) * 1000)
            s.feed(frame, i)
            if i % every:
                continue
            view = cl.view_mask(s.last_view) if s.last_view is not None else None
            viewer.scan_view.update(mask=view, t=time.monotonic(), keep_base=False, lost=not s.stats["tracking"],
                                    model=None if s.view_codes is None else (s.view_codes, b["depth"]))
            (out / f"d{n:04d}.ppm").write_bytes(viewer.render_depth(Cap, b["depth"])[0])
            (out / f"c{n:04d}.ppm").write_bytes(viewer.render_color(Cap, b["color"], width=viewer.DEPTH_W)[0])
            n += 1
    finally:
        s.close()
    return n


def turntable_frames(scan, out, n=TURN_FRAMES, size=TURN_SIZE):
    """PNGs of the scan's mesh.ply turning about the base's up axis, framed on the object."""
    import open3d as o3d
    from open3d.visualization import rendering
    mesh = o3d.io.read_triangle_mesh(str(scan / "mesh.ply"))
    axes = np.array(json.loads((scan / "session.json").read_text())["crop_axes"])  # rows: x, y (up), z
    v = np.asarray(mesh.vertices) @ axes.T
    v -= (v.min(0) + v.max(0)) / 2
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.compute_vertex_normals()
    r = rendering.OffscreenRenderer(size, size)
    r.scene.set_background(BACKGROUND)
    mat = rendering.MaterialRecord()
    mat.shader, mat.base_roughness = "defaultLit", 0.7
    r.scene.add_geometry("mesh", mesh, mat)
    light = np.array([-0.4, -0.8, -0.5])
    r.scene.scene.set_sun_light(light, [1.0, 1.0, 1.0], 75000)
    r.scene.scene.enable_sun_light(True)
    r.scene.set_lighting(rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, light)
    ext = np.ptp(v, 0)
    dist, look = 1.2 * ext.max(), np.array([0.0, 0.12 * ext[1], 0.0])
    for k in range(n):
        a = 2 * np.pi * k / n
        eye = np.array([dist * np.sin(a), dist * 0.55, dist * np.cos(a)]) + look
        r.setup_camera(35.0, look.tolist(), eye.tolist(), [0, 1, 0])
        o3d.io.write_image(str(out / f"t{k:03d}.png"), r.render_to_image())


def gif(inputs, filters, path):
    """ffmpeg: frames (their rate set on the inputs) -> a GIF with its own palette."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex",
                    f"{filters},split[a][b];[a]palettegen=max_colors={SCAN_COLOURS}:stats_mode=diff[p];"
                    "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle", str(path)],
                   check=True)


def main():
    scan = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else scan
    rec = next(scan.glob("*-rec"), None)
    if rec is None:
        sys.exit(f"{scan} has no raw frames (scan with 'keep raw frames' on)")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        n = scan_frames(rec, tmp)
        gif(["-framerate", str(GIF_FPS), "-i", str(tmp / "c%04d.ppm"), "-framerate", str(GIF_FPS), "-i", str(tmp / "d%04d.ppm")],
            f"[0][1]hstack,scale={SCAN_WIDTH}:-1:flags=lanczos", out / "scan.gif")
        turntable_frames(scan, tmp)
        gif(["-framerate", str(TURN_FPS), "-i", str(tmp / "t%03d.png")], f"scale={TURN_WIDTH}:-1:flags=lanczos",
            out / "model.gif")
    print(f"{n} scan frames -> {out / 'scan.gif'}, {TURN_FRAMES} turns -> {out / 'model.gif'}")


if __name__ == "__main__":
    main()
