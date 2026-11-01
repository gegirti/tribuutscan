"""The GPU volume engine on OpenCL (pyopencl: the GPU vendor's OpenCL driver; PoCL on the CPU).

The same layout, kernels and results as gpu_wgpu.py (gpu.py says what each returns), in OpenCL C
1.2. One difference: OpenCL 1.2 has no indirect dispatch, so the number of bricks a frame touched is
read back before the integration is launched.
"""
import time

import numpy as np
import pyopencl as cl

from . import gpu

SOURCE = r"""
typedef struct {
  float4 pose[4]; float4 inv_pose[4]; float4 ref_inv[4];  // columns
  float4 cam; float4 vol; float4 lim; float4 icp; float4 ray;
  uint4 dims; uint4 img;
} Params;
#define EMPTY 0xffffffffu
#define PENDING 0xfffffffeu
#define FULL 0xfffffffdu
#define NV 30
float3 ray_of(__constant Params *P, float u, float v) {
  return (float3)((u - P->cam.z) / P->cam.x, (v - P->cam.w) / P->cam.y, 1.0f);
}
float3 xform(__constant float4 *m, float3 p) { return m[0].xyz * p.x + m[1].xyz * p.y + m[2].xyz * p.z + m[3].xyz; }
float3 rotate(__constant float4 *m, float3 p) { return m[0].xyz * p.x + m[1].xyz * p.y + m[2].xyz * p.z; }
int brick_index(__constant Params *P, int3 bc) {
  int3 d = (int3)((int)P->dims.x, (int)P->dims.y, (int)P->dims.z);
  if (bc.x < 0 || bc.y < 0 || bc.z < 0 || bc.x >= d.x || bc.y >= d.y || bc.z >= d.z) return -1;
  return bc.x + d.x * (bc.y + d.y * bc.z);
}
float4 vertex(__constant Params *P, __global const float *depth, uint x, uint y) {
  float z = depth[y * P->img.x + x];
  if (z <= 0.0f) return (float4)(0.0f);
  return (float4)(ray_of(P, (float)x, (float)y) * z, 1.0f);
}

__kernel void maps(__constant Params *P, __global const float *depth, __global float4 *vmap, __global float4 *nmap) {
  uint x = get_global_id(0), y = get_global_id(1), w = P->img.x, h = P->img.y;
  if (x >= w || y >= h) return;
  uint i = y * w + x;
  float4 p = vertex(P, depth, x, y);
  vmap[i] = p;
  float4 n = (float4)(0.0f);
  if (p.w > 0.0f && x > 0 && y > 0 && x + 1 < w && y + 1 < h) {
    float4 r = vertex(P, depth, x + 1, y), l = vertex(P, depth, x - 1, y);
    float4 dn = vertex(P, depth, x, y + 1), up = vertex(P, depth, x, y - 1);
    if (r.w * l.w * dn.w * up.w > 0.0f) {
      float3 m = cross(r.xyz - l.xyz, dn.xyz - up.xyz);
      float len = length(m);
      if (len > 0.0f) {
        m = m / len;
        if (dot(m, p.xyz) > 0.0f) m = -m;
        n = (float4)(m, 1.0f);
      }
    }
  }
  nmap[i] = n;
}

void touch(__constant Params *P, __global volatile uint *index, __global volatile uint *mark, __global uint *visible,
           __global volatile uint *counters, __global volatile uint *args, uint b) {
  if (index[b] == EMPTY && atomic_cmpxchg(&index[b], EMPTY, PENDING) == EMPTY) {
    uint slot = atomic_inc(&counters[0]);
    if (slot < P->dims.w) atomic_xchg(&index[b], slot);
    else { atomic_xchg(&index[b], FULL); atomic_inc(&counters[1]); }
  }
  if (atomic_xchg(&mark[b], P->img.z) != P->img.z) {
    uint k = atomic_inc(&args[0]);
    if (k < P->img.w) visible[k] = b;
  }
}

__kernel void allocate(__constant Params *P, __global const float *depth, __global volatile uint *index,
                       __global volatile uint *mark, __global uint *visible, __global volatile uint *counters,
                       __global volatile uint *args) {
  uint x = get_global_id(0) * 4, y = get_global_id(1) * 4;  // every 4th pixel each way (Open3D's down_factor)
  if (x >= P->img.x || y >= P->img.y) return;
  float z = depth[y * P->img.x + x];
  if (z <= 0.0f || z > P->lim.z) return;
  float3 rc = ray_of(P, (float)x, (float)y);
  float z0 = fmax(z - P->lim.x, 0.0f), z1 = fmin(z + P->lim.x, P->lim.z);
  float bmm = P->vol.w * 8.0f;
  for (int k = 0; k < 4; k++) {  // the bricks at four depths along the ray within the truncation
    float3 pw = xform(P->pose, rc * (z0 + (z1 - z0) * (float)k / 3.0f));
    float3 f = floor((pw - P->vol.xyz) / bmm);
    int b = brick_index(P, (int3)((int)f.x, (int)f.y, (int)f.z));
    if (b < 0) atomic_inc(&counters[2]); else touch(P, index, mark, visible, counters, args, (uint)b);
  }
}

uint pack_rgb(float3 c) {
  uint3 q = convert_uint3(clamp(rint(c), 0.0f, 255.0f));
  return q.x | (q.y << 8) | (q.z << 16);
}
float3 rgb_of(uint c) { return (float3)((float)(c & 255u), (float)((c >> 8) & 255u), (float)((c >> 16) & 255u)); }

__kernel void integrate(__constant Params *P, __global const float *depth, __global const uint *colour,
                        __global const uint *index, __global const uint *visible, __global uint *tsdfw, __global uint *colvox) {
  uint b = visible[get_group_id(0)];
  uint slot = index[b];
  if (slot >= P->dims.w) return;
  uint t = get_local_id(0);
  uint3 bc = (uint3)(b % P->dims.x, (b / P->dims.x) % P->dims.y, b / (P->dims.x * P->dims.y));
  uint lx = t % 8, ly = t / 8, w = P->img.x, h = P->img.y;
  float trunc = P->lim.x;
  for (uint lz = 0; lz < 8; lz++) {
    float3 pw = P->vol.xyz + convert_float3(bc * 8 + (uint3)(lx, ly, lz)) * P->vol.w;
    float3 pc = xform(P->inv_pose, pw);
    if (pc.z <= 0.0f) continue;
    float uf = pc.x / pc.z * P->cam.x + P->cam.z, vf = pc.y / pc.z * P->cam.y + P->cam.w;
    if (uf < 0.0f || vf < 0.0f || uf > (float)(w - 1) || vf > (float)(h - 1)) continue;
    uint pix = (uint)vf * w + (uint)uf;
    float d = depth[pix];
    if (d <= 0.0f || d > P->lim.z) continue;
    float sdf = d - pc.z;
    if (sdf < -trunc) continue;
    float s = fmin(sdf, trunc) / trunc;
    uint vi = slot * 512 + lx + 8 * ly + 64 * lz;
    uint old = tsdfw[vi];
    float wold = (float)(old >> 16);
    float told = (float)(((int)(old << 16)) >> 16) / 32767.0f;
    float tnew = (told * wold + s) / (wold + 1.0f);
    int q = (int)rint(clamp(tnew, -1.0f, 1.0f) * 32767.0f);
    tsdfw[vi] = (((uint)q) & 0xffffu) | (min((old >> 16) + 1u, 65535u) << 16);
    colvox[vi] = pack_rgb((rgb_of(colvox[vi]) * wold + rgb_of(colour[pix])) / (wold + 1.0f));
  }
}

float2 at(__global const uint *tsdfw, uint slot, int3 l) {
  uint packed = tsdfw[slot * 512 + l.x + 8 * l.y + 64 * l.z];
  return (float2)((float)(((int)(packed << 16)) >> 16) / 32767.0f, (float)(packed >> 16));
}
float2 voxel(__constant Params *P, __global const uint *index, __global const uint *tsdfw, int3 vi) {
  int3 bc = convert_int3(floor(convert_float3(vi) / 8.0f));
  int b = brick_index(P, bc);
  if (b < 0) return (float2)(0.0f);
  uint slot = index[b];
  if (slot >= P->dims.w) return (float2)(0.0f);
  return at(tsdfw, slot, vi - bc * 8);
}
float2 sample(__constant Params *P, __global const uint *index, __global const uint *tsdfw, float3 p) {
  float3 g = (p - P->vol.xyz) / P->vol.w;
  float3 f = floor(g), a = g - f;
  int3 i0 = convert_int3(f);
  int3 bc = convert_int3(floor(f / 8.0f));
  int3 l = i0 - bc * 8;
  float2 c000, c100, c010, c110, c001, c101, c011, c111;
  if (l.x < 7 && l.y < 7 && l.z < 7) {  // all eight in one brick: one index lookup
    int b = brick_index(P, bc);
    if (b < 0) return (float2)(0.0f);
    uint slot = index[b];
    if (slot >= P->dims.w) return (float2)(0.0f);
    c000 = at(tsdfw, slot, l + (int3)(0, 0, 0));
    c100 = at(tsdfw, slot, l + (int3)(1, 0, 0));
    c010 = at(tsdfw, slot, l + (int3)(0, 1, 0));
    c110 = at(tsdfw, slot, l + (int3)(1, 1, 0));
    c001 = at(tsdfw, slot, l + (int3)(0, 0, 1));
    c101 = at(tsdfw, slot, l + (int3)(1, 0, 1));
    c011 = at(tsdfw, slot, l + (int3)(0, 1, 1));
    c111 = at(tsdfw, slot, l + (int3)(1, 1, 1));
  } else {
    c000 = voxel(P, index, tsdfw, i0 + (int3)(0, 0, 0));
    c100 = voxel(P, index, tsdfw, i0 + (int3)(1, 0, 0));
    c010 = voxel(P, index, tsdfw, i0 + (int3)(0, 1, 0));
    c110 = voxel(P, index, tsdfw, i0 + (int3)(1, 1, 0));
    c001 = voxel(P, index, tsdfw, i0 + (int3)(0, 0, 1));
    c101 = voxel(P, index, tsdfw, i0 + (int3)(1, 0, 1));
    c011 = voxel(P, index, tsdfw, i0 + (int3)(0, 1, 1));
    c111 = voxel(P, index, tsdfw, i0 + (int3)(1, 1, 1));
  }
  if (min(min(min(c000.y, c100.y), min(c010.y, c110.y)), min(min(c001.y, c101.y), min(c011.y, c111.y))) < P->ray.x) return (float2)(0.0f);
  float x00 = mix(c000.x, c100.x, a.x), x10 = mix(c010.x, c110.x, a.x);
  float x01 = mix(c001.x, c101.x, a.x), x11 = mix(c011.x, c111.x, a.x);
  return (float2)(mix(mix(x00, x10, a.y), mix(x01, x11, a.y), a.z), 1.0f);
}
__kernel void raycast(__constant Params *P, __global const uint *index, __global const uint *tsdfw,
                      __global float4 *vout, __global float4 *nout) {
  uint x = get_global_id(0), y = get_global_id(1), w = P->img.x;
  if (x >= w || y >= P->img.y) return;
  uint i = y * w + x;
  vout[i] = (float4)(0.0f);
  nout[i] = (float4)(0.0f);
  float3 dc = normalize(ray_of(P, (float)x, (float)y));
  float3 o = P->pose[3].xyz;
  float3 d = rotate(P->pose, dc);
  int3 tiny = isless(fabs(d), (float3)(1e-9f));
  float3 inv = select(1.0f / d, (float3)(1e30f), tiny);
  float bmm = P->vol.w * 8.0f;
  float3 lo = P->vol.xyz, hi = lo + convert_float3(P->dims.xyz) * bmm;
  float3 ta = select((lo - o) * inv, (float3)(-1e30f), tiny), tb = select((hi - o) * inv, (float3)(1e30f), tiny);
  float tmin = fmax(fmax(fmin(ta.x, tb.x), fmin(ta.y, tb.y)), fmin(ta.z, tb.z));
  float tmax = fmin(fmin(fmax(ta.x, tb.x), fmax(ta.y, tb.y)), fmax(ta.z, tb.z));
  float t = fmax(tmin, P->lim.y / dc.z), tend = fmin(tmax, P->lim.z / dc.z);
  float prev = 2.0f, prev_t = t, hit = -1.0f;
  for (uint s = 0; s < (uint)P->ray.z; s++) {
    if (t >= tend) break;
    float3 p = o + d * t;
    float3 fb = floor((p - lo) / bmm);
    int3 bc = convert_int3(fb);
    int b = brick_index(P, bc);
    int empty = b < 0;
    if (!empty) empty = index[b] >= P->dims.w;
    if (empty) {  // jump to where the ray leaves this brick
      float3 side = lo + (fb + select((float3)(0.0f), (float3)(1.0f), isgreater(d, (float3)(0.0f)))) * bmm;
      float3 te = select((side - o) * inv, (float3)(1e30f), tiny);
      t = fmax(fmin(fmin(te.x, te.y), te.z), t) + 0.01f;
      prev = 2.0f;
      continue;
    }
    float2 sm = sample(P, index, tsdfw, p);
    if (sm.y == 0.0f) { prev = 2.0f; t += P->vol.w; continue; }
    if (prev <= 1.0f && prev > 0.0f && sm.x <= 0.0f) { hit = prev_t + (t - prev_t) * prev / (prev - sm.x); break; }
    if (prev < 0.0f && sm.x > 0.0f) break;
    prev = sm.x;
    prev_t = t;
    t += fmax(sm.x * P->lim.x * P->ray.y, P->vol.w * 0.5f);
  }
  if (hit < 0.0f) return;
  float3 p = o + d * hit;
  vout[i] = (float4)(p, hit * dc.z);
  float e = P->vol.w;
  float2 x1 = sample(P, index, tsdfw, p + (float3)(e, 0, 0)), x0 = sample(P, index, tsdfw, p - (float3)(e, 0, 0));
  float2 y1 = sample(P, index, tsdfw, p + (float3)(0, e, 0)), y0 = sample(P, index, tsdfw, p - (float3)(0, e, 0));
  float2 z1 = sample(P, index, tsdfw, p + (float3)(0, 0, e)), z0 = sample(P, index, tsdfw, p - (float3)(0, 0, e));
  if (x1.y * x0.y * y1.y * y0.y * z1.y * z0.y > 0.0f) {
    float3 grad = (float3)(x1.x - x0.x, y1.x - y0.x, z1.x - z0.x);
    if (length(grad) > 0.0f) nout[i] = (float4)(normalize(grad), 1.0f);
  }
}

__kernel __attribute__((reqd_work_group_size(128, 1, 1)))
void icp(__constant Params *P, __global const float4 *vcur, __global const float4 *ncur, __global const float4 *vref,
         __global const float4 *nref, __global float *partials) {
  __local float sh[128 * NV];
  float acc[NV];
  for (int k = 0; k < NV; k++) acc[k] = 0.0f;
  uint t = get_local_id(0), w = P->img.x, h = P->img.y, n = w * h;
  float tk = P->lim.w;
  for (uint i = get_group_id(0) * 128 + t; i < n; i += get_num_groups(0) * 128) {
    float4 pc = vcur[i];
    if (pc.w == 0.0f) continue;
    float4 nc = ncur[i];
    if (nc.w == 0.0f) continue;
    float3 p = xform(P->pose, pc.xyz);
    float3 qc = xform(P->ref_inv, p);
    if (qc.z <= 0.0f) continue;
    float u = floor(qc.x / qc.z * P->cam.x + P->cam.z + 0.5f), v = floor(qc.y / qc.z * P->cam.y + P->cam.w + 0.5f);
    if (u < 0.0f || v < 0.0f || u > (float)(w - 1) || v > (float)(h - 1)) continue;
    uint j = (uint)v * w + (uint)u;
    float4 q = vref[j], nq = nref[j];
    if (q.w <= 0.0f || nq.w == 0.0f) continue;
    if (dot(rotate(P->pose, nc.xyz), nq.xyz) < P->icp.w) continue;
    float r = dot(nq.xyz, p - q.xyz);
    if (fabs(r) >= tk) continue;
    float e = 1.0f - (r / tk) * (r / tk), wt = e * e;
    float3 a = cross(p - P->icp.xyz, nq.xyz);
    float J[6] = {a.x, a.y, a.z, nq.x, nq.y, nq.z};
    int k = 0;
    for (int r0 = 0; r0 < 6; r0++) for (int c0 = r0; c0 < 6; c0++) acc[k++] += wt * J[r0] * J[c0];
    for (int r0 = 0; r0 < 6; r0++) acc[21 + r0] += wt * J[r0] * r;
    acc[27] += 1.0f;
    acc[28] += wt;
    acc[29] += wt * r * r;
  }
  for (int k = 0; k < NV; k++) sh[t * NV + k] = acc[k];
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint s = 64; s > 0; s >>= 1) {
    if (t < s) for (int k = 0; k < NV; k++) sh[t * NV + k] += sh[(t + s) * NV + k];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (t == 0) for (int k = 0; k < NV; k++) partials[get_group_id(0) * NV + k] = sh[k];
}
"""


def devices():
    """{"gpu": device, "cpu": device} for what this machine's OpenCL drivers offer."""
    out = {}
    for platform in cl.get_platforms():
        for d in platform.get_devices():
            kind = "gpu" if d.type & cl.device_type.GPU else "cpu" if d.type & cl.device_type.CPU else None
            if kind and kind not in out:
                out[kind] = d
    return out


class Engine:
    """TSDF volume (gpu.Layout) on one OpenCL device; depth frames of `size` (w, h) on the pinhole K."""

    def __init__(self, layout, K, size, kind="gpu", far_mm=1000.0):
        dev = devices()[kind]
        self.name, self.device_name = f"opencl-{kind}", dev.name.strip()
        self.ctx = cl.Context([dev])
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        self.layout, self.K, self.size, self.far_mm = layout, np.asarray(K, float), size, far_mm
        w, h = size
        mf = cl.mem_flags
        buf = lambda n: cl.Buffer(self.ctx, mf.READ_WRITE, int(n))
        self.uniform = {k: buf(304) for k in ("frame", "ray", "icp")}
        self.depth, self.colour = buf(4 * w * h), buf(4 * w * h)
        self.vmap, self.nmap, self.rvmap, self.rnmap = (buf(16 * w * h) for _ in range(4))
        self.index, self.mark = buf(4 * layout.bricks), buf(4 * layout.bricks)
        self.visible, self.counters, self.args = buf(4 * gpu.MAX_VISIBLE), buf(16), buf(12)
        self.tsdfw, self.colvox = buf(4 * 512 * layout.capacity), buf(4 * 512 * layout.capacity)
        self.partials = buf(4 * gpu.ICP_GROUPS * gpu.ICP_VALUES)
        q = self.queue
        cl.enqueue_fill_buffer(q, self.index, np.uint32(gpu.EMPTY), 0, 4 * layout.bricks)
        cl.enqueue_fill_buffer(q, self.mark, np.uint32(gpu.EMPTY), 0, 4 * layout.bricks)
        for b in (self.tsdfw, self.colvox, self.counters):
            cl.enqueue_fill_buffer(q, b, np.uint32(0), 0, b.size)
        t = time.perf_counter()
        self.program = cl.Program(self.ctx, SOURCE).build()
        self.kernels = {k: getattr(self.program, k) for k in ("maps", "allocate", "integrate", "raycast", "icp")}
        self.compile_s = time.perf_counter() - t
        self.frames = self.empty = 0
        self.gpu_ms = {}
        self.timestamps = True

    def _params(self, which, **kw):
        cl.enqueue_copy(self.queue, self.uniform[which], np.frombuffer(
            gpu.params(self.layout, self.K, self.size, far_mm=self.far_mm, **kw), np.uint8))

    def _launch(self, kernel, global_size, local_size, *args):
        evt = self.kernels[kernel](self.queue, global_size, local_size, *args)
        self._events.append((kernel, evt))
        return evt

    def _image(self, stride=1):
        w, h = ((n + stride - 1) // stride for n in self.size)
        return ((w + 15) // 16 * 16, (h + 15) // 16 * 16), (16, 16)

    def _finish(self):
        self.queue.finish()
        for kernel, evt in self._events:
            self.gpu_ms[kernel] = (evt.profile.end - evt.profile.start) / 1e6
        self._events = []

    def upload(self, depth_mm, rgb):
        cl.enqueue_copy(self.queue, self.depth, np.ascontiguousarray(depth_mm, np.float32))
        cl.enqueue_copy(self.queue, self.colour, np.ascontiguousarray(gpu.pack_colour(rgb)))

    def integrate(self, depth_mm, rgb, pose_m):
        """Fuse one frame (fusion.Engine's interface: pose camera -> world, translation in metres)."""
        if np.count_nonzero(depth_mm) < 100:
            self.empty += 1
            return
        self.fuse(depth_mm, rgb, gpu.pose_mm(pose_m))

    def fuse(self, depth_mm, rgb, pose):
        """Upload a frame, make its maps, allocate the bricks it touches and integrate them (pose in mm)."""
        self._events = []
        self.upload(depth_mm, rgb)
        self._params("frame", pose=pose, frame=self.frames)
        cl.enqueue_copy(self.queue, self.args, np.array([0, 1, 1], np.uint32))
        u = self.uniform["frame"]
        self._launch("maps", *self._image(), u, self.depth, self.vmap, self.nmap)
        self._launch("allocate", *self._image(gpu.ALLOC_STRIDE), u, self.depth, self.index, self.mark, self.visible,
                     self.counters, self.args)
        n = np.empty(3, np.uint32)
        cl.enqueue_copy(self.queue, n, self.args)  # blocking: the number of bricks to integrate
        groups = min(int(n[0]), gpu.MAX_VISIBLE)
        if groups:
            self._launch("integrate", (groups * 64,), (64,), u, self.depth, self.colour, self.index, self.visible,
                         self.tsdfw, self.colvox)
        self._finish()
        self.frames += 1

    def raycast(self, pose, min_weight=gpu.RAY_MIN_WEIGHT):
        self._events = []
        self._params("ray", pose=pose, min_weight=min_weight)
        self._launch("raycast", *self._image(), self.uniform["ray"], self.index, self.tsdfw, self.rvmap, self.rnmap)
        self._finish()

    def icp(self, pose, ref_pose, centre):
        self._events = []
        self._params("icp", pose=pose, ref_inv=np.linalg.inv(ref_pose), centre=centre)
        self._launch("icp", (gpu.ICP_GROUPS * 128,), (128,), self.uniform["icp"], self.vmap, self.nmap, self.rvmap,
                     self.rnmap, self.partials)
        out = np.empty(gpu.ICP_GROUPS * gpu.ICP_VALUES, np.float32)
        cl.enqueue_copy(self.queue, out, self.partials)
        self._finish()
        return gpu.reduce_icp(out)

    def _read(self, b, dtype, count, shape=None):
        out = np.empty(count, dtype)
        cl.enqueue_copy(self.queue, out, b)
        return out if shape is None else out.reshape(shape)

    def frame_maps(self):
        w, h = self.size
        return self._read(self.vmap, np.float32, 4 * w * h, (h, w, 4)), self._read(self.nmap, np.float32, 4 * w * h, (h, w, 4))

    def model_maps(self):
        w, h = self.size
        return self._read(self.rvmap, np.float32, 4 * w * h, (h, w, 4)), self._read(self.rnmap, np.float32, 4 * w * h, (h, w, 4))

    def counts(self):
        c = self._read(self.counters, np.uint32, 4)
        return int(min(c[0], self.layout.capacity)), int(c[1]), int(c[2])

    def volume(self):
        n = self.counts()[0]
        index = self._read(self.index, np.uint32, self.layout.bricks)
        tw, cv = np.empty(512 * n, np.uint32), np.empty(512 * n, np.uint32)
        if n:
            cl.enqueue_copy(self.queue, tw, self.tsdfw)
            cl.enqueue_copy(self.queue, cv, self.colvox)
        return index, tw, cv

    def sync(self):
        self.queue.finish()
