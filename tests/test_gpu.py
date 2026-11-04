from collections import Counter

import numpy as np

from tribuutscan import gpu


def wgpu_engine_or_none(kind="gpu"):
    """The WebGPU engine module when wgpu and an adapter of `kind` are there (else these tests pass by)."""
    try:
        from tribuutscan import gpu_wgpu
    except ImportError:  # needs wgpu (.venv)
        return None
    try:
        return gpu_wgpu if kind in gpu_wgpu.adapters() else None
    except Exception:
        return None


def kinds():
    """The engines to check here: the GPU, and the CPU fallback (llvmpipe), where present."""
    return [k for k in ("gpu", "cpu") if wgpu_engine_or_none(k) is not None]


def marching_cubes(f):
    """numpy marching cubes with gpu.triangle_table over a grid f (inside < 0): (vertices, triangles)."""
    table, keys, pos, tris = gpu.triangle_table(), {}, [], []
    for i, j, k in np.ndindex(*(np.array(f.shape) - 1)):
        pattern = sum(1 << c for c in range(8) if f[i + (c & 1), j + (c >> 1 & 1), k + (c >> 2 & 1)] < 0)
        tri = []
        for e in (e for t in table[pattern] for e in t):
            a, axis = gpu.EDGES[e][0], gpu.EDGE_AXIS[e]
            ca = (i + (a & 1), j + (a >> 1 & 1), k + (a >> 2 & 1))
            if (ca, axis) not in keys:
                cb = list(ca)
                cb[axis] += 1
                p = np.array(ca, float)
                p[axis] += f[ca] / (f[ca] - f[tuple(cb)])
                keys[ca, axis] = len(pos)
                pos.append(p)
            tri.append(keys[ca, axis])
        tris += [tri[n:n + 3] for n in range(0, len(tri), 3)]
    return np.array(pos), np.array(tris)


def closed_and_oriented(triangles):
    directed = Counter((a, b) for t in triangles for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])))
    return all(c == 1 and directed.get((b, a)) == 1 for (a, b), c in directed.items())


def test_force_cpu_keeps_tracking_and_fusion_off_the_gpu():
    """--force-cpu (viewer, cloud.py, fusion.py) sets TRIBUUTSCAN_GPU=0 for this process and the ones it
    starts: no GPU tracker (gputrack.kind), and fusion takes Open3D's engine (fusion.open_engine)."""
    import os
    before = os.environ.get("TRIBUUTSCAN_GPU")
    try:
        os.environ.pop("TRIBUUTSCAN_GPU", None)
        assert gpu.choice() == "auto" and not gpu.force_cpu(["viewer"]) and "TRIBUUTSCAN_GPU" not in os.environ
        assert gpu.force_cpu(["viewer", "--force-cpu"]) and gpu.choice() == "0"
        try:
            from tribuutscan import fusion, gputrack
        except ImportError:  # needs Open3D (.venv)
            return
        assert gputrack.kind() is None
        K = np.array([[120.0, 0, 80], [0, 120, 50], [0, 0, 1]])
        engine = fusion.open_engine(K, (160, 100), 0.5, 3.0, 600.0, bounds=(np.zeros(3), np.full(3, 50.0)))
        assert isinstance(engine, fusion.Engine)
    finally:
        os.environ.pop("TRIBUUTSCAN_GPU", None)
        if before is not None:
            os.environ["TRIBUUTSCAN_GPU"] = before


def test_triangle_table_gives_closed_surfaces_facing_out():
    g = np.mgrid[0:16, 0:16, 0:16].astype(float)
    v, t = marching_cubes(np.sqrt(((g - 7.6) ** 2).sum(0)) - 5.3)  # a sphere
    assert closed_and_oriented(t)
    n = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
    assert ((v[t].mean(1) - 7.6) * n).sum(1).min() > 0  # outward
    assert np.abs(np.linalg.norm(v - 7.6, axis=1) - 5.3).max() < 0.05  # linear interpolation on a curve
    rng = np.random.default_rng(1)
    f = np.pad(rng.normal(size=(9, 9, 9)), 1, constant_values=1.0)  # every ambiguous case, closed off by the border
    assert closed_and_oriented(marching_cubes(f)[1])


def test_reference_maps_and_icp_sums_find_a_known_shift():
    """The numpy references (what each kernel must return): a curved surface seen twice, the second
    view placed 0.6 mm off; one Gauss-Newton step from the ICP sums takes most of it back."""
    w, h = 120, 90
    K = np.array([[100.0, 0, 60], [0, 100, 45], [0, 0, 1]])
    ray = gpu.rays(K, (w, h))
    def bowl(pose):  # a bowl z = 200 + (x^2 + y^2) / 150 in world, seen from `pose`
        d = ray @ pose[:3, :3].T
        o = pose[:3, 3]
        a, b = (d[..., :2] ** 2).sum(-1) / 150, 2 * (o[:2] * d[..., :2]).sum(-1) / 150 - d[..., 2]
        c = (o[:2] ** 2).sum() / 150 + 200 - o[2]
        with np.errstate(invalid="ignore"):  # rays that miss the bowl: no depth
            return np.nan_to_num(2 * c / (-b + np.sqrt(b * b - 4 * a * c))).astype(np.float32)  # near root, stable at a = 0
    pose = np.eye(4)
    vref, nref = gpu.maps(bowl(pose), K)
    assert np.allclose(nref[45, 60, :3], [0, 0, -1], atol=1e-3)  # bottom of the bowl, facing the camera
    vw = vref.copy()  # the reference camera is the world here
    off = np.eye(4)
    off[:3, 3] = [0.4, -0.3, 0.3]
    sums = gpu.icp_sums(vref, nref, off, vw, nref, np.eye(4), K, (0, 0, 210))
    assert sums[27] > 5000
    step = gpu.solve_icp(sums)
    assert np.linalg.norm(step[3:] + off[:3, 3]) < 0.25 * np.linalg.norm(off[:3, 3])


def test_engine_matches_open3d_and_the_references():
    """Three views of a tilted wall: the WebGPU volume holds Open3D's voxels (fusion.Engine, the CPU
    reference), the frame maps and ICP sums equal the numpy references, the ray cast finds the wall,
    the mesh is Open3D's. On the GPU and on the CPU fallback alike."""
    try:
        import open3d as o3d
        import open3d.core as o3c
        from tribuutscan import fusion
    except ImportError:
        return
    w, h = 320, 200
    K = np.array([[300.0, 0, 160], [0, 300, 100], [0, 0, 1]])
    ray = gpu.rays(K, (w, h))
    def wall(pose):  # z = 250 + 0.2 x (world)
        d, t = ray @ pose[:3, :3].T, pose[:3, 3]
        n = np.array([-0.2, 0, 1.0])
        return ((250 - n @ t) / (d @ n)).astype(np.float32)
    poses = []
    for dx in (-3.0, 0.0, 3.0):
        p = np.eye(4)
        p[0, 3] = dx
        poses.append(p)
    rgb = np.full((h, w, 3), 90, np.uint8)
    ref = fusion.Engine(K, gpu.VOXEL_MM, gpu.TRUNC_VOXELS, 1000)
    for p in poses:
        pm = p.copy()
        pm[:3, 3] /= 1000
        ref.integrate(wall(p), rgb.astype(np.float32), pm)
    hm = ref.grid.hashmap()
    bi = hm.active_buf_indices().to(o3c.int64).numpy()
    keys = hm.key_tensor().numpy()[bi]
    ot = ref.grid.attribute("tsdf").numpy().reshape(-1, 512)[bi]
    ow = ref.grid.attribute("weight").numpy().reshape(-1, 512)[bi]
    layout = gpu.Layout(keys.min(0) * 4.0, (keys.max(0) + 1) * 4.0)
    ov, otri, _ = ref.mesh(2)
    for kind in kinds():
        e = wgpu_engine_or_none(kind).Engine(layout, K, (w, h), kind, 1000, nearest=False)  # Open3D's pixel rule
        for p in poses:
            e.fuse(wall(p), rgb, p)
        assert e.counts() == (len(keys), 0, 0)
        index, tw, cw = e.volume()
        tg, wg = gpu.unpack_tsdf(tw)
        bc = (keys - layout.origin / layout.brick_mm).astype(int)
        slots = index[bc[:, 0] + layout.dims[0] * (bc[:, 1] + layout.dims[1] * bc[:, 2])]
        gt, gwt = tg.reshape(-1, 512)[slots], wg.reshape(-1, 512)[slots]
        both = (ow > 0) & (gwt > 0)
        assert (ow != gwt).mean() < 0.01 and np.median(np.abs(ot[both] - gt[both])) < 1e-4
        assert (gpu.unpack_colour(cw.reshape(-1, 512)[slots][gwt > 0]) == 90).all()
        vm, nm = e.frame_maps()
        rv, rn = gpu.maps(wall(poses[-1]), K)
        assert np.abs(vm - rv).max() < 1e-3 and np.abs(nm - rn).max() < 1e-4
        e.raycast(poses[1])
        mv, mn = e.model_maps()
        hit = mv[..., 3] > 0
        err = np.abs(mv[..., 3][hit] - wall(poses[1])[hit])
        # half a pixel off along the slope: the volume takes the depth of the pixel a voxel falls in
        # (rounded down, as Open3D does), 0.5 px x 0.83 mm x 0.2 = 0.083 mm on this wall
        # (at the rim of what was seen a sample averages only the seen voxels: a few hits up to ~0.3 mm)
        assert hit.mean() > 0.9 and abs(np.median(err) - 0.083) < 0.01 and np.percentile(err, 99.9) < 0.2
        assert err.max() < 0.5
        guess = poses[-1].copy()
        guess[0, 3] += 0.5
        sums = e.icp(guess, poses[1], (0, 0, 250))
        want = gpu.icp_sums(rv, rn, guess, mv.astype(float), mn.astype(float), np.linalg.inv(poses[1]), K, (0, 0, 250))
        # float32 on the device: a pixel or two may fall the other side of a rounding edge
        assert abs(sums[27] - want[27]) <= 2 and np.abs(sums - want).max() < 1e-3 * np.abs(want).max()
        v, t, _ = e.mesh(2)
        assert abs(len(t) - len(otri)) <= 0.01 * len(otri)
        near = wgpu_engine_or_none(kind).Engine(layout, K, (w, h), kind, 1000)  # the default: the nearest pixel
        for p in poses:
            near.fuse(wall(p), rgb, p)
        near.raycast(poses[1])
        nv, near_n = near.model_maps()
        nhit = nv[..., 3] > 0
        assert np.median(np.abs(nv[..., 3][nhit] - wall(poses[1])[nhit])) < 0.02  # no half-pixel offset
        index, tw, _ = near.volume()  # the live view: model seen often enough / too few times / the frame's own
        weight = gpu.voxel_weight(layout, index, gpu.unpack_tsdf(tw)[1], nv[..., :3])
        base = (0.0, 0.0, 1.0, -252.0)  # the wall's nearer part counts as the base
        fv = near.frame_maps()[0]
        for step in (1, 2):
            got = near.preview(poses[1], base, len(poses), step)
            want = gpu.preview_codes(nv, near_n, fv, poses[1], base, weight, len(poses), step)
            assert got.shape == want.shape == (h // step, w // step)
            differ = got.astype(int) - want
            # float32 on the device: a shade may round one step the other way
            assert (np.abs(differ) <= 1).all() and (differ != 0).mean() < 0.01
        for lo, hi in ((1, 1), (2, 127), (128, 255)):
            assert ((got >= lo) & (got <= hi)).any(), (lo, hi)  # every kind of pixel occurs
        near.raycast(poses[1], rect=(13, 7, 41, 30))  # only a window of the view, the rest cleared
        wv, wn = near.model_maps()
        assert (wv[7:30, 13:41] == nv[7:30, 13:41]).all() and (wn[7:30, 13:41] == near_n[7:30, 13:41]).all()
        outside = np.ones(wv.shape[:2], bool)
        outside[7:30, 13:41] = False
        assert not wv[outside].any() and not wn[outside].any() and nhit[outside].any()
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3c.Tensor(ov.astype(np.float32)), o3c.Tensor(otri.astype(np.uint32)))
        assert np.median(scene.compute_distance(o3c.Tensor(v.astype(np.float32))).numpy()) < 1e-3  # Open3D's mesh


def test_gpu_frame_preparation_is_fusions():
    """A raw frame (depth on the Y10 scale, YUYV colour, 1/8 labels) prepared on the GPU equals
    fusion.frame_images + Pinhole: the same pixels kept, the same depth, colour through the colour camera."""
    try:
        from tribuutscan import calib, fusion
    except ImportError:
        return
    if not kinds():
        return
    w, h = 160, 96
    spec = {"size": [w, h], "depth": {"fx": 150.0, "fy": 150.0, "cx": 79.5, "cy": 47.5, "dist": [0.05, -0.02, 0, 0, 0]},
            "law": {"a_mm": 289570.0, "b": 873.4}, "window_mm": [150.0, 330.0],
            "colour_from_depth": {"ax": 1.0, "bx": 0.0, "ay": 1.0, "by": 0.0},
            "colour": {"K": [[140.0, 0, 82.0], [0, 141.0, 46.0], [0, 0, 1]], "dist": [0.02, -0.01, 0.001, 0.0005, 0, 0, 0, 0],
                       "R": np.eye(3).tolist(), "T_mm": [19.6, 0.0, -0.8]}}
    saved = calib.CAMERA.spec, calib.CAMERA.mode
    try:
        calib.use(spec)
        rng = np.random.default_rng(3)
        v, u = np.mgrid[0:h, 0:w]
        raw = (60 + 0.8 * u + 0.3 * v).astype(np.float32)  # a slope, inside the window
        raw[30:60, 70:110] += 80.0  # a box nearer: depth jumps at its rim
        raw[rng.random((h, w)) < 0.02] = 0  # dropouts
        raw[5:8, 5:8] = 250.0  # a speck
        yuyv = rng.integers(0, 256, 2 * w * h, dtype=np.uint8).tobytes()
        labels = np.zeros((h // 8, w // 8), np.uint8)
        labels[2:10, 4:17] = 2
        labels[10, 4:17] = 1
        frame = {"depth": raw, "yuyv": yuyv}
        pin = fusion.Pinhole()
        mask = fusion.label_mask(labels, 8, raw.shape)
        from tribuutscan.image import yuyv_to_rgb
        d_cpu, c_cpu = fusion.frame_images({"depth": raw, "color": yuyv_to_rgb(yuyv, w, h)}, mask)
        d_cpu, c_cpu = pin(d_cpu), pin(c_cpu)
        layout = gpu.Layout([-150, -100, 100], [150, 100, 400])
        for kind in kinds():
            e = wgpu_engine_or_none(kind).Engine(layout, pin.K, (w, h), kind, 350)
            e.set_camera(calib.CAMERA, pin.index)
            assert e.fuse_frame(frame, labels, np.eye(4))
            d_gpu, c_gpu = e.prepared()
            assert ((d_cpu > 0) == (d_gpu > 0)).all() and (d_cpu > 0).sum() > 3000
            ok = d_cpu > 0
            assert np.abs(d_cpu[ok] - d_gpu[ok]).max() < 1e-3
            assert (c_cpu[ok] != c_gpu[ok]).any(1).mean() < 0.01
            # the same frame from the scanner's bytes (unpack: the GPU decodes them), Y14 and Y10, with a
            # garbled run: the depth is depth.decode_depth's, and fusing it reads that depth
            from tribuutscan import depth as dp
            bad = np.round(raw).astype(np.uint16)
            bad[40:46] = (64 * (np.arange(6)[:, None] + 3) + np.arange(w) % 64).astype(np.uint16)
            for packed, bpl in ((dp.encode_y14(bad * 16 + 8), w * 14 // 8), (dp.encode_y10(bad), w * 10 // 8)):
                want = dp.decode_depth(packed, w, h, bpl)
                _, cleared = dp.decode_strided(packed, w, h, bpl)
                assert cleared is not None and not want[cleared].any()
                e.unpack(packed, bpl, cleared)
                assert np.array_equal(e.unpacked(), want.astype(np.float32))
                assert e.fuse_frame({"depth": want, "yuyv": yuyv}, labels, np.eye(4), depth_ready=True)
                d_ready, _ = e.prepared()
                e.fuse_frame({"depth": want, "yuyv": yuyv}, labels, np.eye(4))
                assert np.array_equal(d_ready, e.prepared()[0])
    finally:
        calib.use(saved[0])
        calib.use_mode(saved[1])


def blur(depth, r, tol):
    """numpy reference of the tracking source's blur: each depth the mean of its neighbours within r
    pixels that lie within tol mm of it."""
    h, w = depth.shape
    out = np.zeros_like(depth)
    pad = np.pad(depth, r)
    total, count = np.zeros_like(depth), np.zeros_like(depth)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            zz = pad[r + dy:r + dy + h, r + dx:r + dx + w]
            same = (zz > 0) & (np.abs(zz - depth) < tol)
            total += np.where(same, zz, 0)
            count += same
    ok = depth > 0
    out[ok] = total[ok] / count[ok]
    return out


def test_the_tracking_source_and_icp_options_are_the_references():
    """What tracking runs on the GPU equals numpy: the frame's kept cells (uncleaned, raw -> mm, onto the
    pinhole grid), maps with wider-set normals, and the ICP sums with stride, footprint weights, reach and
    the base left out of the model."""
    try:
        from tribuutscan import calib, fusion
    except ImportError:
        return
    if not kinds():
        return
    w, h = 160, 96
    spec = {"size": [w, h], "depth": {"fx": 150.0, "fy": 150.0, "cx": 79.5, "cy": 47.5, "dist": [0.05, -0.02, 0, 0, 0]},
            "law": {"a_mm": 289570.0, "b": 873.4}, "window_mm": [150.0, 330.0],
            "colour_from_depth": {"ax": 1.0, "bx": 0.0, "ay": 1.0, "by": 0.0}}
    saved = calib.CAMERA.spec, calib.CAMERA.mode
    try:
        calib.use(spec)
        cam = calib.CAMERA
        v, u = np.mgrid[0:h, 0:w]
        z = 240 + 0.002 * ((u - 80.0) ** 2 + (v - 48.0) ** 2) + 0.1 * u  # a tilted bowl
        z[60:, :] = 300.0  # a floor-like step behind it
        raw = cam.raw_from_mm(z).astype(np.float32)
        raw[::17, ::13] = 0  # dropouts
        keep = np.zeros((h // 2, w // 2), bool)
        keep[5:40, 10:70] = True
        pin = fusion.Pinhole()
        kept = np.repeat(np.repeat(keep, 2, 0), 2, 1)
        depth = pin(np.where(kept & (raw > 0), cam.depth_mm(np.maximum(raw, 1)), 0.0))
        vref, nref = gpu.maps(depth, pin.K, step=2)
        layout = gpu.Layout([-150, -100, 100], [150, 100, 400])
        pose = np.eye(4)
        pose[:3, 3] = [0.6, -0.4, 0.5]
        base = (0.0, -0.2, -0.98, 0.98 * 290.0)  # n . q + c > 0: the part nearer than ~290 mm
        options = dict(area=4 / (150.0 * 150.0 * 4.0), reach_mm=6.0, stride=2, base=base)
        for kind in kinds():
            e = wgpu_engine_or_none(kind).Engine(layout, pin.K, (w, h), kind, 350)
            e.set_camera(cam, pin.index)  # no colour model: grey
            e.fuse(pin(np.where(raw > 0, cam.depth_mm(np.maximum(raw, 1)), 0.0)), np.zeros((h, w, 3)), np.eye(4))
            e.raycast(np.eye(4))
            mv, mn = e.model_maps()
            e.track_source(raw, keep, 2, normal_step=2)
            vm, nm = e.frame_maps()
            assert (vm[..., 3] > 0).sum() > 1500 and np.abs(vm - vref).max() < 1e-3 and np.abs(nm - nref).max() < 1e-4
            sums = e.icp(pose, np.eye(4), (0, 0, 250), **options)
            want = gpu.icp_sums(vref, nref, pose, mv.astype(float), mn.astype(float), np.eye(4), pin.K, (0, 0, 250),
                                **options)
            assert want[27] > 300 and want[31] < want[30]  # pairs, some beyond the base or out of reach
            assert abs(sums[27] - want[27]) <= 2 and abs(sums[31] - want[31]) <= 2 and sums[30] == want[30]
            assert np.abs(sums - want).max() < 1e-3 * np.abs(want[:27]).max()
            e.track_source(raw, keep, 2, normal_step=2, blur_px=1, blur_mm=1.0)  # the same-surface mean
            blurred = e.frame_maps()[0][..., 3]
            assert np.abs(e.frame_maps()[0][..., 2] - blur(depth, 1, 1.0)).max() < 1e-3 and (blurred > 0).sum() > 1500
            e.raycast(np.eye(4), normal_px=3)  # model normals from the hits 3 px apart, as the frame's maps
            mv, mn = e.model_maps()
            ok = mn[..., 3] > 0
            want_n = gpu.maps(mv[..., 3], pin.K, step=3)[1]  # hits on the rays of the reference camera
            assert ok.sum() > 50 and np.abs(mn[ok] - want_n[ok]).max() < 1e-3  # (needs all four neighbours hit)
            e.close()
    finally:
        calib.use(saved[0])
        calib.use_mode(saved[1])


def test_a_full_brick_pool_is_counted_not_fatal():
    if not kinds():
        return
    gpu_wgpu = wgpu_engine_or_none(kinds()[0])
    w, h = 160, 100
    K = np.array([[150.0, 0, 80], [0, 150, 50], [0, 0, 1]])
    e = gpu_wgpu.Engine(gpu.Layout([-150, -100, 200], [150, 100, 300], capacity=50), K, (w, h), kinds()[0], 1000)
    e.fuse(np.full((h, w), 250.0, np.float32), np.zeros((h, w, 3), np.uint8), np.eye(4))
    bricks, refused, _ = e.counts()
    assert bricks == 50 and refused > 0
    v, t, _ = e.mesh(1)  # what fitted is still there
    assert len(t) > 0 and np.abs(v[:, 2] - 250).max() < 0.05


def test_close_gives_the_gpu_memory_back():
    """wgpu-py frees a device only through Python's cycle collector; close() must not wait for it."""
    if "gpu" not in kinds():
        return
    from tribuutscan import bench
    if not (bench.gpu_usage() or {}).get("memory_mib"):
        return  # this driver does not report memory per process (Linux DRM fdinfo)
    gpu_wgpu = wgpu_engine_or_none("gpu")
    K = np.array([[300.0, 0, 160], [0, 300, 100], [0, 0, 1]])
    before = sum(bench.gpu_usage()["memory_mib"].values())
    e = gpu_wgpu.Engine(gpu.Layout([-100, -100, 150], [100, 100, 350]), K, (320, 200), "gpu", 1000)
    e.fuse(np.full((200, 320), 250.0, np.float32), np.zeros((200, 320, 3), np.uint8), np.eye(4))
    e.mesh(1)
    assert sum(bench.gpu_usage()["memory_mib"].values()) > before + 50
    e.close()
    assert sum(bench.gpu_usage()["memory_mib"].values()) < before + 10
