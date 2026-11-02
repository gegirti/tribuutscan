"""The GPU volume engine on WebGPU (wgpu-py: Vulkan, Metal or DirectX 12 underneath; llvmpipe on the CPU).

Layout, inputs and what each kernel returns: gpu.py. Kernels are WGSL, one shader module each (a
module's bindings are exactly what its kernel uses, so layout="auto" fits). Every kernel reads the
parameter block (gpu.params) at binding 0; a few uniform buffers hold the blocks of the frame being
fused, the ray cast and the ICP, so they can sit in one submission.
"""
import gc
import struct
import time

import numpy as np
import wgpu

from . import gpu

PRELUDE = """
struct Params {
  pose: mat4x4<f32>, inv_pose: mat4x4<f32>, ref_inv: mat4x4<f32>,
  cam: vec4<f32>,   // fx, fy, cx, cy
  vol: vec4<f32>,   // origin xyz (mm), voxel mm
  lim: vec4<f32>,   // truncation mm, near, far, ICP Tukey mm
  icp: vec4<f32>,   // ICP centre xyz, normal gate (cos)
  ray: vec4<f32>,   // ray cast: min weight, step factor, max steps
  dims: vec4<u32>,  // bricks x, y, z, pool capacity
  img: vec4<u32>,   // width, height, frame id, max visible bricks
  trk: vec4<f32>,   // ICP: pixel footprint scale (0: none), reach mm, maps: normal step px; ICP: pixel stride
  base: vec4<f32>,  // ICP: model points with dot(base.xyz, q) + base.w <= 0 are the base, left out
  rect: vec4<u32>,  // ray cast: the pixels x0, y0 (in) to x1, y1 (out) it covers; x1 = 0: the whole image
}
@group(0) @binding(0) var<uniform> P: Params;
const EMPTY: u32 = 0xffffffffu;
const PENDING: u32 = 0xfffffffeu;
const FULL: u32 = 0xfffffffdu;
fn ray_of(u: f32, v: f32) -> vec3<f32> { return vec3<f32>((u - P.cam.z) / P.cam.x, (v - P.cam.w) / P.cam.y, 1.0); }
fn xform(m: mat4x4<f32>, p: vec3<f32>) -> vec3<f32> { return (m * vec4<f32>(p, 1.0)).xyz; }
fn brick_index(bc: vec3<i32>) -> i32 {
  let d = vec3<i32>(P.dims.xyz);
  if (any(bc < vec3<i32>(0)) || any(bc >= d)) { return -1; }
  return bc.x + d.x * (bc.y + d.y * bc.z);
}
"""

MAPS = """
@group(0) @binding(1) var<storage, read> depth: array<f32>;
@group(0) @binding(2) var<storage, read_write> vmap: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> nmap: array<vec4<f32>>;
fn vertex(x: u32, y: u32) -> vec4<f32> {
  let z = depth[y * P.img.x + x];
  if (z <= 0.0) { return vec4<f32>(0.0); }
  return vec4<f32>(ray_of(f32(x), f32(y)) * z, 1.0);
}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let w = P.img.x;
  let h = P.img.y;
  if (g.x >= w || g.y >= h) { return; }
  let i = g.y * w + g.x;
  let p = vertex(g.x, g.y);
  vmap[i] = p;
  var n = vec4<f32>(0.0);
  let k = max(u32(P.trk.z), 1u);  // neighbours this many pixels away
  if (p.w > 0.0 && g.x >= k && g.y >= k && g.x + k < w && g.y + k < h) {
    let r = vertex(g.x + k, g.y);
    let l = vertex(g.x - k, g.y);
    let dn = vertex(g.x, g.y + k);
    let up = vertex(g.x, g.y - k);
    if (r.w * l.w * dn.w * up.w > 0.0) {
      var m = cross(r.xyz - l.xyz, dn.xyz - up.xyz);
      let len = length(m);
      if (len > 0.0) {
        m = m / len;
        if (dot(m, p.xyz) > 0.0) { m = -m; }
        n = vec4<f32>(m, 1.0);
      }
    }
  }
  nmap[i] = n;
}
"""

ALLOCATE = """
@group(0) @binding(1) var<storage, read> depth: array<f32>;
@group(0) @binding(2) var<storage, read_write> index: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> mark: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> visible: array<u32>;
@group(0) @binding(5) var<storage, read_write> counters: array<atomic<u32>, 4>;  // allocated, pool full, outside
@group(0) @binding(6) var<storage, read_write> args: array<atomic<u32>, 3>;      // visible bricks, 1, 1
@group(0) @binding(7) var<storage, read_write> bricks: array<u32>;  // the brick in each pool slot
fn touch(b: u32) {
  if (atomicLoad(&index[b]) == EMPTY) {
    loop {
      let r = atomicCompareExchangeWeak(&index[b], EMPTY, PENDING);
      if (r.exchanged) {
        let slot = atomicAdd(&counters[0], 1u);
        if (slot < P.dims.w) { atomicStore(&index[b], slot); bricks[slot] = b; }
        else { atomicStore(&index[b], FULL); atomicAdd(&counters[1], 1u); }
        break;
      }
      if (r.old_value != EMPTY) { break; }
    }
  }
  if (atomicExchange(&mark[b], P.img.z) != P.img.z) {
    let k = atomicAdd(&args[0], 1u);
    if (k < P.img.w) { visible[k] = b; }
  }
}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gs: vec3<u32>) {
  let g = gs.xy * 4u;  // every 4th pixel each way (Open3D's down_factor)
  if (g.x >= P.img.x || g.y >= P.img.y) { return; }
  let z = depth[g.y * P.img.x + g.x];
  if (z <= 0.0 || z > P.lim.z) { return; }
  // the bricks at four depths along the pixel's ray, from z - truncation to z + truncation (Open3D's rule)
  let rc = ray_of(f32(g.x), f32(g.y));
  let z0 = max(z - P.lim.x, 0.0);
  let z1 = min(z + P.lim.x, P.lim.z);
  let bmm = P.vol.w * 8.0;
  for (var k = 0; k < 4; k++) {
    let pw = xform(P.pose, rc * (z0 + (z1 - z0) * f32(k) / 3.0));
    let b = brick_index(vec3<i32>(floor((pw - P.vol.xyz) / bmm)));
    if (b < 0) { atomicAdd(&counters[2], 1u); } else { touch(u32(b)); }
  }
}
"""

CLAMP = """
@group(0) @binding(1) var<storage, read_write> args: array<u32, 3>;
@compute @workgroup_size(1)
fn main() { args[0] = min(args[0], P.img.w); }
"""

INTEGRATE = """
@group(0) @binding(1) var<storage, read> depth: array<f32>;
@group(0) @binding(2) var<storage, read> colour: array<u32>;
@group(0) @binding(3) var<storage, read> index: array<u32>;
@group(0) @binding(4) var<storage, read> visible: array<u32>;
@group(0) @binding(5) var<storage, read_write> tsdfw: array<u32>;
@group(0) @binding(6) var<storage, read_write> colvox: array<u32>;
fn rgb(c: u32) -> vec3<f32> { return vec3<f32>(f32(c & 255u), f32((c >> 8u) & 255u), f32((c >> 16u) & 255u)); }
fn pack_rgb(c: vec3<f32>) -> u32 {
  let q = vec3<u32>(clamp(round(c), vec3<f32>(0.0), vec3<f32>(255.0)));
  return q.x | (q.y << 8u) | (q.z << 16u);
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let b = visible[wg.x];
  let slot = index[b];
  if (slot >= P.dims.w) { return; }
  let bc = vec3<u32>(b % P.dims.x, (b / P.dims.x) % P.dims.y, b / (P.dims.x * P.dims.y));
  let lx = t % 8u;
  let ly = t / 8u;
  let w = P.img.x;
  let h = P.img.y;
  let trunc = P.lim.x;
  for (var lz = 0u; lz < 8u; lz++) {
    let pw = P.vol.xyz + vec3<f32>(bc * 8u + vec3<u32>(lx, ly, lz)) * P.vol.w;
    let pc = xform(P.inv_pose, pw);
    if (pc.z <= 0.0) { continue; }
    let uf = pc.x / pc.z * P.cam.x + P.cam.z;
    let vf = pc.y / pc.z * P.cam.y + P.cam.w;
    if (uf < LO || vf < LO || uf > f32(w) - HI || vf > f32(h) - HI) { continue; }
    let pix = u32(vf + ROUND) * w + u32(uf + ROUND);
    let d = depth[pix];
    if (d <= 0.0 || d > P.lim.z) { continue; }
    let sdf = d - pc.z;
    if (sdf < -trunc) { continue; }
    let s = min(sdf, trunc) / trunc;
    let vi = slot * 512u + lx + 8u * ly + 64u * lz;
    let old = tsdfw[vi];
    let wold = f32(old >> 16u);
    let told = f32(bitcast<i32>(old << 16u) >> 16u) / 32767.0;
    let tnew = (told * wold + s) / (wold + 1.0);
    let q = i32(round(clamp(tnew, -1.0, 1.0) * 32767.0));
    tsdfw[vi] = (bitcast<u32>(q) & 0xffffu) | (min((old >> 16u) + 1u, 65535u) << 16u);
    colvox[vi] = pack_rgb((rgb(colvox[vi]) * wold + rgb(colour[pix])) / (wold + 1.0));
  }
}
"""

RAYCAST = """
@group(0) @binding(1) var<storage, read> index: array<u32>;
@group(0) @binding(2) var<storage, read> tsdfw: array<u32>;
@group(0) @binding(3) var<storage, read_write> vout: array<vec4<f32>>;  // world xyz, camera depth (0: no hit)
@group(0) @binding(4) var<storage, read_write> nout: array<vec4<f32>>;  // world normal, 1 if known
fn at(slot: u32, l: vec3<i32>) -> vec2<f32> {
  let packed = tsdfw[slot * 512u + u32(l.x) + 8u * u32(l.y) + 64u * u32(l.z)];
  return vec2<f32>(f32(bitcast<i32>(packed << 16u) >> 16u) / 32767.0, f32(packed >> 16u));
}
fn voxel(vi: vec3<i32>) -> vec2<f32> {
  let bc = vec3<i32>(floor(vec3<f32>(vi) / 8.0));
  let b = brick_index(bc);
  if (b < 0) { return vec2<f32>(0.0); }
  let slot = index[u32(b)];
  if (slot >= P.dims.w) { return vec2<f32>(0.0); }
  return at(slot, vi - bc * 8);
}
fn sample(p: vec3<f32>) -> vec2<f32> {  // trilinear TSDF, and 1 if it is known (below)
  let g = (p - P.vol.xyz) / P.vol.w;
  let f = floor(g);
  let a = g - f;
  let i0 = vec3<i32>(f);
  let bc = vec3<i32>(floor(f / 8.0));
  let l = i0 - bc * 8;
  var c000: vec2<f32>;
  var c100: vec2<f32>;
  var c010: vec2<f32>;
  var c110: vec2<f32>;
  var c001: vec2<f32>;
  var c101: vec2<f32>;
  var c011: vec2<f32>;
  var c111: vec2<f32>;
  if (all(l < vec3<i32>(7))) {  // all eight in one brick: one index lookup
    let b = brick_index(bc);
    if (b < 0) { return vec2<f32>(0.0); }
    let slot = index[u32(b)];
    if (slot >= P.dims.w) { return vec2<f32>(0.0); }
    c000 = at(slot, l + vec3<i32>(0, 0, 0));
    c100 = at(slot, l + vec3<i32>(1, 0, 0));
    c010 = at(slot, l + vec3<i32>(0, 1, 0));
    c110 = at(slot, l + vec3<i32>(1, 1, 0));
    c001 = at(slot, l + vec3<i32>(0, 0, 1));
    c101 = at(slot, l + vec3<i32>(1, 0, 1));
    c011 = at(slot, l + vec3<i32>(0, 1, 1));
    c111 = at(slot, l + vec3<i32>(1, 1, 1));
  } else {
    c000 = voxel(i0 + vec3<i32>(0, 0, 0));
    c100 = voxel(i0 + vec3<i32>(1, 0, 0));
    c010 = voxel(i0 + vec3<i32>(0, 1, 0));
    c110 = voxel(i0 + vec3<i32>(1, 1, 0));
    c001 = voxel(i0 + vec3<i32>(0, 0, 1));
    c101 = voxel(i0 + vec3<i32>(1, 0, 1));
    c011 = voxel(i0 + vec3<i32>(0, 1, 1));
    c111 = voxel(i0 + vec3<i32>(1, 1, 1));
  }
  // trilinear over the voxels seen enough; none where they hold less than half the weight (a single
  // frame's band has one-voxel holes where cleaning dropped a pixel)
  let b = vec3<f32>(1.0) - a;
  let w000 = b.x * b.y * b.z * f32(c000.y >= P.ray.x);
  let w100 = a.x * b.y * b.z * f32(c100.y >= P.ray.x);
  let w010 = b.x * a.y * b.z * f32(c010.y >= P.ray.x);
  let w110 = a.x * a.y * b.z * f32(c110.y >= P.ray.x);
  let w001 = b.x * b.y * a.z * f32(c001.y >= P.ray.x);
  let w101 = a.x * b.y * a.z * f32(c101.y >= P.ray.x);
  let w011 = b.x * a.y * a.z * f32(c011.y >= P.ray.x);
  let w111 = a.x * a.y * a.z * f32(c111.y >= P.ray.x);
  let total = w000 + w100 + w010 + w110 + w001 + w101 + w011 + w111;
  if (total < 0.5) { return vec2<f32>(0.0); }
  return vec2<f32>((w000 * c000.x + w100 * c100.x + w010 * c010.x + w110 * c110.x + w001 * c001.x + w101 * c101.x
                    + w011 * c011.x + w111 * c111.x) / total, 1.0);
}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = P.img.x;
  let g = gid.xy + P.rect.xy;
  let whole = P.rect.z == 0u;
  if (g.x >= select(P.rect.z, w, whole) || g.y >= select(P.rect.w, P.img.y, whole)) { return; }
  let i = g.y * w + g.x;
  vout[i] = vec4<f32>(0.0);
  nout[i] = vec4<f32>(0.0);
  let dc = normalize(ray_of(f32(g.x), f32(g.y)));
  let o = P.pose[3].xyz;
  let d = (P.pose * vec4<f32>(dc, 0.0)).xyz;
  let tiny = abs(d) < vec3<f32>(1e-9);
  let inv = select(1.0 / d, vec3<f32>(1e30), tiny);
  let bmm = P.vol.w * 8.0;
  let lo = P.vol.xyz;
  let hi = lo + vec3<f32>(P.dims.xyz) * bmm;
  let ta = select((lo - o) * inv, vec3<f32>(-1e30), tiny);
  let tb = select((hi - o) * inv, vec3<f32>(1e30), tiny);
  let tmin = max(max(min(ta.x, tb.x), min(ta.y, tb.y)), min(ta.z, tb.z));
  let tmax = min(min(max(ta.x, tb.x), max(ta.y, tb.y)), max(ta.z, tb.z));
  var t = max(tmin, P.lim.y / dc.z);
  let tend = min(tmax, P.lim.z / dc.z);
  var prev = 2.0;
  var prev_t = t;
  var hit = -1.0;
  for (var s = 0u; s < u32(P.ray.z); s++) {
    if (t >= tend) { break; }
    let p = o + d * t;
    let bc = vec3<i32>(floor((p - lo) / bmm));
    let b = brick_index(bc);
    var empty = b < 0;
    if (!empty) { empty = index[u32(b)] >= P.dims.w; }
    if (empty) {  // jump to where the ray leaves this brick
      let side = lo + (vec3<f32>(bc) + select(vec3<f32>(0.0), vec3<f32>(1.0), d > vec3<f32>(0.0))) * bmm;
      let te = select((side - o) * inv, vec3<f32>(1e30), tiny);
      t = max(min(min(te.x, te.y), te.z), t) + 0.01;
      prev = 2.0;
      continue;
    }
    let sm = sample(p);
    if (sm.y == 0.0) { prev = 2.0; t += P.vol.w; continue; }
    if (prev <= 1.0 && prev > 0.0 && sm.x <= 0.0) { hit = prev_t + (t - prev_t) * prev / (prev - sm.x); break; }
    if (prev < 0.0 && sm.x > 0.0) { break; }  // came out of a surface from behind
    prev = sm.x;
    prev_t = t;
    t += max(sm.x * P.lim.x * P.ray.y, P.vol.w * 0.5);
  }
  if (hit < 0.0) { return; }
  let p = o + d * hit;
  vout[i] = vec4<f32>(p, hit * dc.z);
  let e = P.vol.w;
  let x1 = sample(p + vec3<f32>(e, 0.0, 0.0)); let x0 = sample(p - vec3<f32>(e, 0.0, 0.0));
  let y1 = sample(p + vec3<f32>(0.0, e, 0.0)); let y0 = sample(p - vec3<f32>(0.0, e, 0.0));
  let z1 = sample(p + vec3<f32>(0.0, 0.0, e)); let z0 = sample(p - vec3<f32>(0.0, 0.0, e));
  if (x1.y * x0.y * y1.y * y0.y * z1.y * z0.y > 0.0) {
    let grad = vec3<f32>(x1.x - x0.x, y1.x - y0.x, z1.x - z0.x);
    if (length(grad) > 0.0) { nout[i] = vec4<f32>(normalize(grad), 1.0); }
  }
}
"""

PREVIEW = """
@group(0) @binding(1) var<storage, read> index: array<u32>;
@group(0) @binding(2) var<storage, read> tsdfw: array<u32>;
@group(0) @binding(3) var<storage, read> rvmap: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> rnmap: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> vmap: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> codes: array<u32>;
fn voxel_weight(vi: vec3<i32>) -> u32 {  // seen-count of a voxel (0 where no brick is allocated)
  let bc = vec3<i32>(floor(vec3<f32>(vi) / 8.0));
  let b = brick_index(bc);
  if (b < 0) { return 0u; }
  let slot = index[u32(b)];
  if (slot >= P.dims.w) { return 0u; }
  let l = vi - bc * 8;
  return tsdfw[slot * 512u + u32(l.x) + 8u * u32(l.y) + 64u * u32(l.z)] >> 16u;
}
fn weight_at(q: vec3<f32>) -> f32 {  // the most of the eight voxels around q (gpu.voxel_weight)
  let c = vec3<i32>(floor((q - P.vol.xyz) / P.vol.w));
  var most = 0u;
  for (var k = 0; k < 8; k++) {
    most = max(most, voxel_weight(c + vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1)));
  }
  return f32(most);
}
fn code(x0: u32, y0: u32, s: u32) -> u32 {  // gpu.preview_codes: the block's first model hit, else its piece
  var piece = false;
  for (var dy = 0u; dy < s; dy++) {
    for (var dx = 0u; dx < s; dx++) {
      let i = (y0 + dy) * P.img.x + x0 + dx;
      let m = rvmap[i];
      if (m.w > 0.0 && dot(P.base.xyz, m.xyz) + P.base.w > 0.0) {
        let n = rnmap[i];
        var facing = 0.6;
        if (n.w > 0.0) { facing = abs(dot(n.xyz, normalize(P.pose[3].xyz - m.xyz))); }
        if (weight_at(m.xyz) >= P.ray.x) { return 128u + u32(facing * 127.0); }
        return 2u + u32(facing * 125.0);
      }
      piece = piece || vmap[i].w > 0.0;
    }
  }
  return select(0u, 1u, piece);
}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {  // four codes (bytes) per word
  let s = max(u32(P.trk.w), 1u);
  let ow = P.img.x / s;
  let pw = (ow + 3u) / 4u;
  if (g.x >= pw || g.y >= P.img.y / s) { return; }
  var word = 0u;
  for (var k = 0u; k < 4u; k++) {
    let ox = g.x * 4u + k;
    if (ox < ow) { word = word | (code(ox * s, g.y * s, s) << (8u * k)); }
  }
  codes[g.y * pw + g.x] = word;
}
"""

ICP = """
@group(0) @binding(1) var<storage, read> vcur: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> ncur: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> vref: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> nref: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> partials: array<f32>;
const NV: u32 = 32u;
var<workgroup> sh: array<f32, 4096>;  // 128 threads x 32 values: 16 kB, WebGPU's guaranteed limit
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32,
        @builtin(num_workgroups) nw: vec3<u32>) {
  var a0 = 0.0;
  var a1 = 0.0;
  var a2 = 0.0;
  var a3 = 0.0;
  var a4 = 0.0;
  var a5 = 0.0;
  var a6 = 0.0;
  var a7 = 0.0;
  var a8 = 0.0;
  var a9 = 0.0;
  var a10 = 0.0;
  var a11 = 0.0;
  var a12 = 0.0;
  var a13 = 0.0;
  var a14 = 0.0;
  var a15 = 0.0;
  var a16 = 0.0;
  var a17 = 0.0;
  var a18 = 0.0;
  var a19 = 0.0;
  var a20 = 0.0;
  var a21 = 0.0;
  var a22 = 0.0;
  var a23 = 0.0;
  var a24 = 0.0;
  var a25 = 0.0;
  var a26 = 0.0;
  var a27 = 0.0;
  var a28 = 0.0;
  var a29 = 0.0;
  var a30 = 0.0;
  var a31 = 0.0;
  let w = P.img.x;
  let h = P.img.y;
  let s = max(u32(P.trk.w), 1u);  // every s-th pixel each way
  let ws = (w + s - 1u) / s;
  let n = ws * ((h + s - 1u) / s);
  let tk = P.lim.w;
  let reach2 = P.trk.y * P.trk.y;
  for (var k = wg.x * 128u + t; k < n; k += nw.x * 128u) {
    let i = (k / ws) * s * w + (k % ws) * s;
    let pc = vcur[i];
    if (pc.w == 0.0) { continue; }
    let nc = ncur[i];
    if (nc.w == 0.0) { continue; }
    a30 += 1.0;
    let p = xform(P.pose, pc.xyz);
    let qc = xform(P.ref_inv, p);
    if (qc.z <= 0.0) { continue; }
    let u = floor(qc.x / qc.z * P.cam.x + P.cam.z + 0.5);
    let v = floor(qc.y / qc.z * P.cam.y + P.cam.w + 0.5);
    if (u < 0.0 || v < 0.0 || u > f32(w - 1u) || v > f32(h - 1u)) { continue; }
    let j = u32(v) * w + u32(u);
    let q = vref[j];
    let nq = nref[j];
    if (q.w <= 0.0 || nq.w == 0.0) { continue; }
    if (dot(P.base.xyz, q.xyz) + P.base.w <= 0.0) { continue; }
    let dv = p - q.xyz;
    let reach = dot(dv, dv) / reach2;
    if (reach >= 1.0) { continue; }
    a31 += 1.0;
    let ncw = (P.pose * vec4<f32>(nc.xyz, 0.0)).xyz;
    if (dot(ncw, nq.xyz) < P.icp.w) { continue; }
    let r = dot(nq.xyz, dv);
    if (abs(r) >= tk) { continue; }
    let e = 1.0 - (r / tk) * (r / tk);
    let f = 1.0 - reach;
    var wt = e * e * f * f;
    if (P.trk.x > 0.0) { wt *= pc.z * pc.z * P.trk.x; }
    let a = cross(p - P.icp.xyz, nq.xyz);
    let j0 = a.x; let j1 = a.y; let j2 = a.z; let j3 = nq.x; let j4 = nq.y; let j5 = nq.z;
    a0 += wt * j0 * j0;
    a1 += wt * j0 * j1;
    a2 += wt * j0 * j2;
    a3 += wt * j0 * j3;
    a4 += wt * j0 * j4;
    a5 += wt * j0 * j5;
    a6 += wt * j1 * j1;
    a7 += wt * j1 * j2;
    a8 += wt * j1 * j3;
    a9 += wt * j1 * j4;
    a10 += wt * j1 * j5;
    a11 += wt * j2 * j2;
    a12 += wt * j2 * j3;
    a13 += wt * j2 * j4;
    a14 += wt * j2 * j5;
    a15 += wt * j3 * j3;
    a16 += wt * j3 * j4;
    a17 += wt * j3 * j5;
    a18 += wt * j4 * j4;
    a19 += wt * j4 * j5;
    a20 += wt * j5 * j5;
    a21 += wt * j0 * r;
    a22 += wt * j1 * r;
    a23 += wt * j2 * r;
    a24 += wt * j3 * r;
    a25 += wt * j4 * r;
    a26 += wt * j5 * r;
    a27 += 1.0;
    a28 += wt;
    a29 += r * r;
  }
  sh[t * NV + 0u] = a0;
  sh[t * NV + 1u] = a1;
  sh[t * NV + 2u] = a2;
  sh[t * NV + 3u] = a3;
  sh[t * NV + 4u] = a4;
  sh[t * NV + 5u] = a5;
  sh[t * NV + 6u] = a6;
  sh[t * NV + 7u] = a7;
  sh[t * NV + 8u] = a8;
  sh[t * NV + 9u] = a9;
  sh[t * NV + 10u] = a10;
  sh[t * NV + 11u] = a11;
  sh[t * NV + 12u] = a12;
  sh[t * NV + 13u] = a13;
  sh[t * NV + 14u] = a14;
  sh[t * NV + 15u] = a15;
  sh[t * NV + 16u] = a16;
  sh[t * NV + 17u] = a17;
  sh[t * NV + 18u] = a18;
  sh[t * NV + 19u] = a19;
  sh[t * NV + 20u] = a20;
  sh[t * NV + 21u] = a21;
  sh[t * NV + 22u] = a22;
  sh[t * NV + 23u] = a23;
  sh[t * NV + 24u] = a24;
  sh[t * NV + 25u] = a25;
  sh[t * NV + 26u] = a26;
  sh[t * NV + 27u] = a27;
  sh[t * NV + 28u] = a28;
  sh[t * NV + 29u] = a29;
  sh[t * NV + 30u] = a30;
  sh[t * NV + 31u] = a31;
  workgroupBarrier();
  for (var s2 = 64u; s2 > 0u; s2 = s2 >> 1u) {
    if (t < s2) { for (var k = 0u; k < NV; k++) { sh[t * NV + k] += sh[(t + s2) * NV + k]; } }
    workgroupBarrier();
  }
  if (t == 0u) { for (var k = 0u; k < NV; k++) { partials[wg.x * NV + k] = sh[k]; } }
}
"""

MODEL_NORMALS = """
@group(0) @binding(1) var<storage, read> vin: array<vec4<f32>>;        // the ray cast: world xyz, depth
@group(0) @binding(2) var<storage, read_write> nout: array<vec4<f32>>;
fn at(x: u32, y: u32) -> vec4<f32> { return vin[y * P.img.x + x]; }
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {  // normals from the hits trk.z pixels away (as maps)
  let w = P.img.x;
  let h = P.img.y;
  if (g.x >= w || g.y >= h) { return; }
  let i = g.y * w + g.x;
  nout[i] = vec4<f32>(0.0);
  let p = vin[i];
  let k = max(u32(P.trk.z), 1u);
  if (p.w <= 0.0 || g.x < k || g.y < k || g.x + k >= w || g.y + k >= h) { return; }
  let r = at(g.x + k, g.y);
  let l = at(g.x - k, g.y);
  let dn = at(g.x, g.y + k);
  let up = at(g.x, g.y - k);
  if (r.w * l.w * dn.w * up.w <= 0.0) { return; }
  var m = cross(r.xyz - l.xyz, dn.xyz - up.xyz);
  let len = length(m);
  if (len <= 0.0) { return; }
  m = m / len;
  if (dot(m, p.xyz - P.pose[3].xyz) > 0.0) { m = -m; }  // facing the camera
  nout[i] = vec4<f32>(m, 1.0);
}
"""

MESH = """
@group(0) @binding(1) var<storage, read> index: array<u32>;
@group(0) @binding(2) var<storage, read> tsdfw: array<u32>;
@group(0) @binding(3) var<storage, read> colvox: array<u32>;
@group(0) @binding(4) var<storage, read> bricks: array<u32>;
@group(0) @binding(5) var<storage, read> table: array<u32>;   // gpu.packed_triangle_table
@group(0) @binding(6) var<storage, read_write> out: array<u32>;  // 3 x 5 u32 per triangle (gpu.assemble_mesh)
@group(0) @binding(7) var<storage, read_write> count: array<atomic<u32>, 2>;  // triangles, cubes left out (full)
var<private> EA: array<u32, 12> = array<u32, 12>(0u, 2u, 4u, 6u, 0u, 1u, 4u, 5u, 0u, 1u, 2u, 3u);
var<private> EB: array<u32, 12> = array<u32, 12>(1u, 3u, 5u, 7u, 2u, 3u, 6u, 7u, 4u, 5u, 6u, 7u);
var<private> EX: array<u32, 12> = array<u32, 12>(0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u, 2u, 2u, 2u, 2u);
struct Corner { f: f32, w: f32, c: u32 }
fn corner(vi: vec3<i32>) -> Corner {
  let bc = vec3<i32>(floor(vec3<f32>(vi) / 8.0));
  let b = brick_index(bc);
  if (b < 0) { return Corner(0.0, 0.0, 0u); }
  let slot = index[u32(b)];
  if (slot >= P.dims.w) { return Corner(0.0, 0.0, 0u); }
  let l = vec3<u32>(vi - bc * 8);
  let k = slot * 512u + l.x + 8u * l.y + 64u * l.z;
  let packed = tsdfw[k];
  return Corner(f32(bitcast<i32>(packed << 16u) >> 16u) / 32767.0, f32(packed >> 16u), colvox[k]);
}
fn rgb(c: u32) -> vec3<f32> { return vec3<f32>(f32(c & 255u), f32((c >> 8u) & 255u), f32((c >> 16u) & 255u)); }
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let s = wg.y * 256u + wg.x;
  if (s >= P.img.z) { return; }
  let b = bricks[s];
  let bc = vec3<i32>(vec3<u32>(b % P.dims.x, (b / P.dims.x) % P.dims.y, b / (P.dims.x * P.dims.y)));
  for (var lz = 0; lz < 8; lz++) {
    let v0 = bc * 8 + vec3<i32>(i32(t % 8u), i32(t / 8u), lz);
    var f: array<f32, 8>;
    var col: array<u32, 8>;
    var pattern = 0u;
    var seen = true;
    for (var c = 0u; c < 8u; c++) {
      let k = corner(v0 + vec3<i32>(i32(c & 1u), i32((c >> 1u) & 1u), i32((c >> 2u) & 1u)));
      if (k.w < P.ray.x) { seen = false; break; }
      f[c] = k.f;
      col[c] = k.c;
      if (k.f < 0.0) { pattern |= 1u << c; }
    }
    if (!seen || pattern == 0u || pattern == 255u) { continue; }
    let first = table[pattern];
    let ntri = (table[pattern + 1u] - first) / 3u;
    if (ntri == 0u) { continue; }
    let base = atomicAdd(&count[0], ntri);
    if (base + ntri > P.img.w) { atomicAdd(&count[1], 1u); continue; }
    for (var q = 0u; q < 3u * ntri; q++) {
      let e = table[first + q];
      let a = EA[e];
      let fa = f[a];
      let tt = fa / (fa - f[EB[e]]);
      let c = mix(rgb(col[a]), rgb(col[EB[e]]), tt);
      let ci = vec3<u32>(clamp(round(c), vec3<f32>(0.0), vec3<f32>(255.0)));
      let va = v0 + vec3<i32>(i32(a & 1u), i32((a >> 1u) & 1u), i32((a >> 2u) & 1u));
      let o = (base * 3u + q) * 5u;
      out[o] = u32(va.x);
      out[o + 1u] = u32(va.y);
      out[o + 2u] = u32(va.z) * 4u + EX[e];
      out[o + 3u] = bitcast<u32>(tt);
      out[o + 4u] = ci.x | (ci.y << 8u) | (ci.z << 16u);
    }
  }
}
"""

MESH_BYTES = 128 * 2 ** 20  # marching cubes output: 60 bytes a triangle, 2.2 million triangles
def integrate_source(nearest):
    """INTEGRATE with the pixel a voxel takes its depth from: the nearest one, or (nearest=False)
    Open3D's, the one it falls in rounded down. Rounding down is half a pixel off along every slope:
    on a synthetic sphere the mesh lay 0.057 mm (median) from the true surface, 0.018 with the nearest."""
    consts = ("const LO = -0.5;\nconst HI = 0.5;\nconst ROUND = 0.5;\n" if nearest
              else "const LO = 0.0;\nconst HI = 1.0;\nconst ROUND = 0.0;\n")
    return consts + INTEGRATE


KERNELS = {"preview": PREVIEW, "model_normals": MODEL_NORMALS, "maps": MAPS, "allocate": ALLOCATE, "clamp": CLAMP, "integrate": INTEGRATE, "raycast": RAYCAST, "icp": ICP,
           "mesh": MESH}
# --- frame preparation (S1-S3): fusion.frame_images + Pinhole on the GPU ----------------------------------
PREP_PRELUDE = """
struct Prep {
  law: vec4<f32>,    // depth law a, b (z = a / (raw + b)), window near, far (mm)
  rules: vec4<f32>,  // jump mm, jump share of z, cos of the steepest view kept, 7x7 speckle count needed
  rot0: vec4<f32>, rot1: vec4<f32>, rot2: vec4<f32>,  // depth -> colour camera, rows of R
  tr: vec4<f32>,     // ... and T (mm)
  ck: vec4<f32>,     // colour fx, fy, cx, cy
  d0: vec4<f32>,     // colour lens k1, k2, p1, p2
  d1: vec4<f32>,     // k3, k4, k5, k6
  box: vec4<u32>,    // the cut-out cleaned: first column, first row, end column, end row
  size: vec4<u32>,   // depth width, height, colour width, height (0: no colour)
  cells: vec4<u32>,  // label step, label columns, label rows, labels used (1) or not (0)
  blur: vec4<f32>,   // track_source: radius px (0: none), neighbours within this many mm
}
@group(0) @binding(0) var<uniform> Q: Prep;
fn in_box(x: i32, y: i32) -> bool { return x >= i32(Q.box.x) && y >= i32(Q.box.y) && x < i32(Q.box.z) && y < i32(Q.box.w); }
"""

PREP_CLEAN = """
@group(0) @binding(1) var<storage, read> raw: array<f32>;
@group(0) @binding(2) var<storage, read> rays: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> mask1: array<u32>;
fn zat(x: i32, y: i32) -> f32 {  // mm inside the depth window, -1 where none (or outside the cut-out)
  if (!in_box(x, y)) { return -1.0; }
  let r = raw[u32(y) * Q.size.x + u32(x)];
  if (r <= 0.0) { return -1.0; }
  let z = Q.law.x / (r + Q.law.y);
  if (z < Q.law.z || z > Q.law.w) { return -1.0; }
  return z;
}
fn point(x: i32, y: i32, z: f32) -> vec3<f32> { let r = rays[u32(y) * Q.size.x + u32(x)]; return vec3<f32>(r * z, z); }
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= Q.size.x || g.y >= Q.size.y) { return; }
  let x = i32(g.x);
  let y = i32(g.y);
  let i = g.y * Q.size.x + g.x;
  mask1[i] = 0u;
  let z = zat(x, y);
  if (z < 0.0) { return; }
  let limit = max(Q.rules.x, Q.rules.y * z);  // prep.clean: a depth jump to a neighbour
  let zr = zat(x + 1, y);
  let zl = zat(x - 1, y);
  let zd = zat(x, y + 1);
  let zu = zat(x, y - 1);
  if ((zr >= 0.0 && abs(z - zr) > limit) || (zl >= 0.0 && abs(z - zl) > limit) ||
      (zd >= 0.0 && abs(z - zd) > limit) || (zu >= 0.0 && abs(z - zu) > limit)) { return; }
  if (zr >= 0.0 && zl >= 0.0 && zd >= 0.0 && zu >= 0.0) {  // a surface seen nearly edge-on
    let n = cross(point(x + 1, y, zr) - point(x - 1, y, zl), point(x, y + 1, zd) - point(x, y - 1, zu));
    let p = point(x, y, z);
    let len = length(n) * length(p);
    if (len > 0.0 && abs(dot(n, p)) / len < Q.rules.z) { return; }
  }
  mask1[i] = 1u;
}
"""

PREP_ERODE = """
@group(0) @binding(1) var<storage, read> mask1: array<u32>;
@group(0) @binding(2) var<storage, read_write> mask2: array<u32>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= Q.size.x || g.y >= Q.size.y) { return; }
  let i = g.y * Q.size.x + g.x;
  mask2[i] = 0u;
  if (g.x <= Q.box.x || g.y <= Q.box.y || g.x + 1u >= Q.box.z || g.y + 1u >= Q.box.w) { return; }  // the cut-out's rim
  let w = Q.size.x;
  mask2[i] = mask1[i] & mask1[i - 1u] & mask1[i + 1u] & mask1[i - w] & mask1[i + w];
}
"""

PREP_FINISH = """
@group(0) @binding(1) var<storage, read> raw: array<f32>;
@group(0) @binding(2) var<storage, read> rays: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> mask2: array<u32>;
@group(0) @binding(4) var<storage, read> labels: array<u32>;
@group(0) @binding(5) var<storage, read> yuyv: array<u32>;
@group(0) @binding(6) var<storage, read_write> zs: array<f32>;
@group(0) @binding(7) var<storage, read_write> cs: array<u32>;
fn labelled(x: i32, y: i32) -> bool {  // fusion.label_mask: the frame's 1/8 labels, grown by one cell
  if (Q.cells.w == 0u) { return true; }
  let cx = x / i32(Q.cells.x);
  let cy = y / i32(Q.cells.x);
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let ax = cx + dx;
      let ay = cy + dy;
      if (ax >= 0 && ay >= 0 && ax < i32(Q.cells.y) && ay < i32(Q.cells.z) && labels[u32(ay) * Q.cells.y + u32(ax)] > 0u) { return true; }
    }
  }
  return false;
}
fn colour_at(p: vec3<f32>) -> u32 {  // calib.Camera.colour_px, rounded and clipped; YUYV -> RGB as image.yuyv_pixels
  let q = vec3<f32>(dot(Q.rot0.xyz, p), dot(Q.rot1.xyz, p), dot(Q.rot2.xyz, p)) + Q.tr.xyz;
  let xn = q.x / q.z;
  let yn = q.y / q.z;
  let r2 = xn * xn + yn * yn;
  let radial = (1.0 + Q.d0.x * r2 + Q.d0.y * r2 * r2 + Q.d1.x * r2 * r2 * r2) / (1.0 + Q.d1.y * r2 + Q.d1.z * r2 * r2 + Q.d1.w * r2 * r2 * r2);
  let xd = xn * radial + 2.0 * Q.d0.z * xn * yn + Q.d0.w * (r2 + 2.0 * xn * xn);
  let yd = yn * radial + Q.d0.z * (r2 + 2.0 * yn * yn) + 2.0 * Q.d0.w * xn * yn;
  let cx = u32(clamp(round(xd * Q.ck.x + Q.ck.z), 0.0, f32(Q.size.z - 1u)));
  let cy = u32(clamp(round(yd * Q.ck.y + Q.ck.w), 0.0, f32(Q.size.w - 1u)));
  let word = yuyv[(cy * Q.size.z + cx) / 2u];
  let luma = f32(select(word & 255u, (word >> 16u) & 255u, (cx & 1u) == 1u));
  let u = f32((word >> 8u) & 255u) - 128.0;
  let v = f32(word >> 24u) - 128.0;
  let c = vec3<u32>(clamp(vec3<f32>(luma + 1.402 * v, luma - 0.344 * u - 0.714 * v, luma + 1.772 * u), vec3<f32>(0.0), vec3<f32>(255.0)));
  return c.x | (c.y << 8u) | (c.z << 16u);
}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= Q.size.x || g.y >= Q.size.y) { return; }
  let x = i32(g.x);
  let y = i32(g.y);
  let i = g.y * Q.size.x + g.x;
  zs[i] = 0.0;
  cs[i] = 0u;
  if (mask2[i] == 0u || !labelled(x, y)) { return; }
  var n = 0u;  // speckle: half of the 7x7 window must be kept
  for (var dy = -3; dy <= 3; dy++) {
    for (var dx = -3; dx <= 3; dx++) {
      let ax = x + dx;
      let ay = y + dy;
      if (in_box(ax, ay)) { n += mask2[u32(ay) * Q.size.x + u32(ax)]; }
    }
  }
  if (f32(n) < Q.rules.w) { return; }
  let z = Q.law.x / (raw[i] + Q.law.y);
  zs[i] = z;
  let r = rays[i];
  cs[i] = select(0x808080u, colour_at(vec3<f32>(r * z, z)), Q.size.z > 0u);  // no colour: grey
}
"""

PREP_REMAP = """
@group(0) @binding(1) var<storage, read> lookup: array<i32>;  // fusion.Pinhole: the sensor pixel of each pinhole pixel
@group(0) @binding(2) var<storage, read> zs: array<f32>;
@group(0) @binding(3) var<storage, read> cs: array<u32>;
@group(0) @binding(4) var<storage, read_write> depth: array<f32>;
@group(0) @binding(5) var<storage, read_write> colour: array<u32>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= Q.size.x || g.y >= Q.size.y) { return; }
  let i = g.y * Q.size.x + g.x;
  let j = lookup[i];
  if (j < 0) { depth[i] = 0.0; colour[i] = 0u; return; }
  depth[i] = zs[u32(j)];
  colour[i] = cs[u32(j)];
}
"""

PREP_SOURCE = """
@group(0) @binding(1) var<storage, read> raw: array<f32>;
@group(0) @binding(2) var<storage, read> lookup: array<i32>;
@group(0) @binding(3) var<storage, read> keep: array<u32>;  // one per cell of cells.x^2 sensor pixels
@group(0) @binding(4) var<storage, read_write> depth: array<f32>;  // (zs: blurred into the frame's depth next)
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {  // what tracking matches: the kept cells, uncleaned
  if (g.x >= Q.size.x || g.y >= Q.size.y) { return; }
  let i = g.y * Q.size.x + g.x;
  depth[i] = 0.0;
  let j = lookup[i];
  if (j < 0) { return; }
  let x = u32(j) % Q.size.x;
  let y = u32(j) / Q.size.x;
  if (keep[(y / Q.cells.x) * Q.cells.y + x / Q.cells.x] == 0u) { return; }
  let r = raw[u32(j)];
  if (r > 0.0) { depth[i] = Q.law.x / (r + Q.law.y); }
}
"""

PREP_BLUR = """
@group(0) @binding(1) var<storage, read> zs: array<f32>;
@group(0) @binding(2) var<storage, read_write> depth: array<f32>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {  // mean of the neighbours on the same surface
  if (g.x >= Q.size.x || g.y >= Q.size.y) { return; }
  let i = g.y * Q.size.x + g.x;
  let z = zs[i];
  depth[i] = z;
  let r = i32(Q.blur.x);
  if (z <= 0.0 || r == 0) { return; }
  var sum = 0.0;
  var n = 0.0;
  for (var dy = -r; dy <= r; dy++) {
    for (var dx = -r; dx <= r; dx++) {
      let x = i32(g.x) + dx;
      let y = i32(g.y) + dy;
      if (x < 0 || y < 0 || x >= i32(Q.size.x) || y >= i32(Q.size.y)) { continue; }
      let zz = zs[u32(y) * Q.size.x + u32(x)];
      if (zz > 0.0 && abs(zz - z) < Q.blur.y) { sum += zz; n += 1.0; }
    }
  }
  depth[i] = sum / n;
}
"""

PREP = {"prep_blur": PREP_BLUR, "prep_clean": PREP_CLEAN, "prep_erode": PREP_ERODE, "prep_finish": PREP_FINISH, "prep_remap": PREP_REMAP,
        "prep_source": PREP_SOURCE}

S = wgpu.BufferUsage


def adapters():
    """{"gpu": adapter, "cpu": adapter} for what this machine has (Vulkan, Metal or DX12 first)."""
    out = {}
    for a in wgpu.gpu.enumerate_adapters_sync():
        info = a.info
        if info["backend_type"] == "OpenGL":
            continue
        kind = "cpu" if info["adapter_type"] == "CPU" else "gpu"
        if kind not in out or (kind == "gpu" and info["adapter_type"] == "DiscreteGPU"):
            out[kind] = a
    return out


class Engine:
    """TSDF volume (gpu.Layout) on one WebGPU adapter; depth frames of `size` (w, h) on the pinhole K."""

    def __init__(self, layout, K, size, kind="gpu", far_mm=1000.0, nearest=True, timestamps=True):
        """timestamps: time each pass on the GPU (gpu_ms) where the adapter can; each submission then
        reads the stamps back, ~0.2 ms (tracking runs ~10 ICP passes a frame without them)."""
        adapter = adapters()[kind]
        self.name = f"wgpu-{kind}"
        self.device_name = adapter.info["device"]
        feats = ["timestamp-query"] if timestamps and "timestamp-query" in adapter.features else []
        self.device = adapter.request_device_sync(required_features=feats)
        self.timestamps = bool(feats)
        self.layout, self.K, self.size, self.far_mm = layout, np.asarray(K, float), size, far_mm
        w, h = size
        dev = self.device
        buf = lambda n, usage=S.STORAGE | S.COPY_DST | S.COPY_SRC: dev.create_buffer(size=int(n), usage=usage)
        self.uniform = {k: buf(gpu.PARAMS_BYTES, S.UNIFORM | S.COPY_DST) for k in ("frame", "ray", "icp", "view")}
        self.depth, self.colour = buf(4 * w * h), buf(4 * w * h)
        self.vmap, self.nmap, self.rvmap, self.rnmap = (buf(16 * w * h) for _ in range(4))
        self.index, self.mark = buf(4 * layout.bricks), buf(4 * layout.bricks)
        self.visible = buf(4 * gpu.MAX_VISIBLE)
        self.counters = buf(16)
        self.args = buf(12, S.STORAGE | S.INDIRECT | S.COPY_DST | S.COPY_SRC)
        self.tsdfw, self.colvox = buf(4 * 512 * layout.capacity), buf(4 * 512 * layout.capacity)
        self.partials = buf(4 * gpu.ICP_GROUPS * gpu.ICP_VALUES)
        self.bricks = buf(4 * layout.capacity)
        self.table = buf(4 * len(gpu.packed_triangle_table()))
        self.out = buf(MESH_BYTES)
        self.tri_count = buf(8)
        self.codes = buf(4 * ((w + 3) // 4) * h)  # preview(): a byte per pixel, four to a word
        q = dev.queue
        q.write_buffer(self.table, 0, gpu.packed_triangle_table())
        q.write_buffer(self.index, 0, np.full(layout.bricks, gpu.EMPTY, np.uint32))
        q.write_buffer(self.mark, 0, np.full(layout.bricks, gpu.EMPTY, np.uint32))
        t = time.perf_counter()
        sources = {**KERNELS, "integrate": integrate_source(nearest)}
        self.nearest = nearest
        self.pipes = {k: dev.create_compute_pipeline(layout="auto", compute={
            "module": dev.create_shader_module(code=PRELUDE + src), "entry_point": "main"}) for k, src in sources.items()}
        self.pipes.update({k: dev.create_compute_pipeline(layout="auto", compute={
            "module": dev.create_shader_module(code=PREP_PRELUDE + src), "entry_point": "main"}) for k, src in PREP.items()})
        self.compile_s = time.perf_counter() - t
        self.prep = None  # set_camera()
        u = self.uniform
        self.groups = {
            "maps": self._group("maps", u["frame"], self.depth, self.vmap, self.nmap),
            "allocate": self._group("allocate", u["frame"], self.depth, self.index, self.mark, self.visible, self.counters,
                                    self.args, self.bricks),
            "clamp": self._group("clamp", u["frame"], self.args),
            "integrate": self._group("integrate", u["frame"], self.depth, self.colour, self.index, self.visible, self.tsdfw,
                                     self.colvox),
            "raycast": self._group("raycast", u["ray"], self.index, self.tsdfw, self.rvmap, self.rnmap),
            "model_normals": self._group("model_normals", u["ray"], self.rvmap, self.rnmap),
            "icp": self._group("icp", u["icp"], self.vmap, self.nmap, self.rvmap, self.rnmap, self.partials),
            "preview": self._group("preview", u["view"], self.index, self.tsdfw, self.rvmap, self.rnmap, self.vmap,
                                   self.codes),
            "mesh": self._group("mesh", u["ray"], self.index, self.tsdfw, self.colvox, self.bricks, self.table, self.out,
                                self.tri_count),
        }
        if self.timestamps:
            self.queries = dev.create_query_set(type="timestamp", count=16)
            self.stamps = buf(8 * 16, S.QUERY_RESOLVE | S.COPY_SRC)
        self.frames = self.empty = 0
        self.gpu_ms = {}
        self.tick_ns = 1.0  # WebGPU timestamps should be ns; wgpu-native hands Vulkan's raw ticks (calibrate_ticks)

    def set_camera(self, camera, lookup=None):
        """Prepare frames on the GPU from here on (fuse_frame, track_source), for `camera` (calib.Camera, in
        its current depth mode) and the pinhole lookup (fusion.Pinhole.index, None when the lens has no
        distortion). Without a colour model (the estimated camera) the volume's colour is grey."""
        from . import prep as cpu_prep
        w, h = self.size
        dev, q = self.device, self.device.queue
        buf = lambda n: dev.create_buffer(size=int(n), usage=S.STORAGE | S.COPY_DST | S.COPY_SRC)
        vv, uu = np.mgrid[0:h, 0:w]
        x, y = camera.normalised(uu.ravel(), vv.ravel())
        b = {"raw": buf(4 * w * h), "rays": buf(8 * w * h), "mask1": buf(4 * w * h), "mask2": buf(4 * w * h),
             "labels": buf(4 * (w // 8 + 2) * (h // 8 + 2)), "yuyv": buf(2 * w * h), "zs": buf(4 * w * h), "zt": buf(4 * w * h), "cs": buf(4 * w * h),
             "lookup": buf(4 * w * h), "cells": buf(4 * w * h),
             "uniform": dev.create_buffer(size=208, usage=S.UNIFORM | S.COPY_DST)}
        q.write_buffer(b["rays"], 0, np.stack([x, y], 1).astype(np.float32))
        q.write_buffer(b["lookup"], 0, (np.arange(w * h) if lookup is None else lookup).astype(np.int32))
        c = camera.colour or {"dist": [], "R": np.eye(3), "T": np.zeros(3), "K": camera.intrinsic_matrix()}
        dist = (list(np.ravel(c["dist"])) + [0.0] * 8)[:8]
        R, T, Kc = np.asarray(c["R"], float), np.ravel(c["T"]), np.asarray(c["K"], float)
        near, far = camera.window_mm
        head = struct.pack("<4f", camera.a, camera.b, near, far)
        head += struct.pack("<4f", cpu_prep.JUMP_MM, cpu_prep.JUMP_SHARE, np.cos(np.radians(cpu_prep.MAX_VIEW_DEG)),
                            cpu_prep.SPECKLE_MIN * cpu_prep.SPECKLE_WIN ** 2)
        head += b"".join(struct.pack("<4f", *R[k], 0) for k in range(3)) + struct.pack("<4f", *T, 0)
        head += struct.pack("<4f", Kc[0, 0], Kc[1, 1], Kc[0, 2], Kc[1, 2]) + struct.pack("<8f", *dist)
        # colour frames are the depth size; (0, 0): no colour
        self.prep = {"buffers": b, "head": head, "colour_size": (w, h) if camera.colour is not None else (0, 0)}
        g = lambda kernel, *names: self._group(kernel, b["uniform"], *(b[n] if isinstance(n, str) else n for n in names))
        self.pipes.update({"prep_remap_blur": self.pipes["prep_remap"], "prep_blur_fused": self.pipes["prep_blur"]})
        self.groups.update({
            "prep_clean": g("prep_clean", "raw", "rays", "mask1"),
            "prep_erode": g("prep_erode", "mask1", "mask2"),
            "prep_finish": g("prep_finish", "raw", "rays", "mask2", "labels", "yuyv", "zs", "cs"),
            "prep_remap": g("prep_remap", "lookup", "zs", "cs", self.depth, self.colour),
            "prep_remap_blur": g("prep_remap", "lookup", "zs", "cs", "zt", self.colour),  # then blurred into depth
            "prep_blur_fused": g("prep_blur", "zt", self.depth),
            "prep_source": g("prep_source", "raw", "lookup", "cells", "zs"),
            "prep_blur": g("prep_blur", "zs", self.depth)})

    def fuse_frame(self, frame, labels, pose_m, step=8, blur_px=0, blur_mm=1.0):
        """Prepare one recorded frame set on the GPU (as fusion.frame_images + Pinhole: cleaned, only on
        its labelled pixels, colour per pixel through the colour camera) and fuse it. `frame` holds
        "depth" (raw, Y10 scale) and "yuyv" (without it, or without a colour model: grey); `labels` its
        1/step labels or None; pose camera -> world in metres; blur_px, blur_mm: as track_source (after
        the cleaning; for a tracking volume, not a mesh). Returns False (and counts it empty) when nothing
        of it is left to fuse."""
        w, h = self.size
        p, q = self.prep, self.device.queue
        if labels is not None:  # the cut-out fusion.frame_images cleans: the grown label cells + 8 px
            on = labels > 0
            rows, cols = np.nonzero(on.any(1))[0], np.nonzero(on.any(0))[0]
            if not len(rows):
                self.empty += 1
                return False
            nr, nc = labels.shape
            first_r, last_r = max(rows[0] - 1, 0) * step, min(min(rows[-1] + 1, nr - 1) * step + step - 1, h - 1)
            first_c, last_c = max(cols[0] - 1, 0) * step, min(min(cols[-1] + 1, nc - 1) * step + step - 1, w - 1)
            r0, r1, c0, c1 = max(first_r - 8, 0), min(last_r + 9, h), max(first_c - 8, 0), min(last_c + 9, w)
            lab = np.zeros((labels.shape[0], labels.shape[1]), np.uint32)
            lab[:] = labels
            q.write_buffer(p["buffers"]["labels"], 0, lab)
            cells = (step, labels.shape[1], labels.shape[0], 1)
        else:
            r0, r1, c0, c1, cells = 0, h, 0, w, (step, 0, 0, 0)
        cw, ch = p["colour_size"] if "yuyv" in frame else (0, 0)
        q.write_buffer(p["buffers"]["uniform"], 0, p["head"] + struct.pack("<4I", c0, r0, c1, r1)
                       + struct.pack("<4I", w, h, cw, ch) + struct.pack("<4I", *cells)
                       + struct.pack("<4f", blur_px, blur_mm, 0, 0))
        q.write_buffer(p["buffers"]["raw"], 0, np.ascontiguousarray(frame["depth"], np.float32))
        if cw:
            q.write_buffer(p["buffers"]["yuyv"], 0, np.frombuffer(frame["yuyv"], np.uint8, 2 * cw * ch))
        self._params("frame", pose=gpu.pose_mm(pose_m), frame=self.frames)
        q.write_buffer(self.args, 0, np.array([0, 1, 1], np.uint32))
        img = self._image_groups()
        remap = [("prep_remap_blur", img), ("prep_blur_fused", img)] if blur_px else [("prep_remap", img)]
        self._run([("prep_clean", img), ("prep_erode", img), ("prep_finish", img), *remap, ("maps", img),
                   ("allocate", self._image_groups(gpu.ALLOC_STRIDE)), ("clamp", (1,)), ("integrate", "indirect")])
        self.frames += 1
        return True

    def track_source(self, raw, keep, step, normal_step=1, blur_px=0, blur_mm=1.0):
        """Make the maps tracking matches (icp) from a frame's raw depth (Y10 scale, sensor grid): only
        where `keep` ((H / step, W / step) bool, one per cell of step x step sensor pixels: the frame's
        piece) is set, uncleaned, on the pinhole grid; each depth the mean of its neighbours within
        `blur_px` that lie within `blur_mm` of it; normals from neighbours `normal_step` pixels away."""
        w, h = self.size
        p, q = self.prep, self.device.queue
        cells = np.zeros(((h + step - 1) // step, (w + step - 1) // step), np.uint32)
        cells[:keep.shape[0], :keep.shape[1]] = keep
        q.write_buffer(p["buffers"]["uniform"], 0, p["head"] + struct.pack("<4I", 0, 0, w, h)
                       + struct.pack("<4I", w, h, 0, 0) + struct.pack("<4I", step, cells.shape[1], cells.shape[0], 1)
                       + struct.pack("<4f", blur_px, blur_mm, 0, 0))
        q.write_buffer(p["buffers"]["raw"], 0, np.ascontiguousarray(raw, np.float32))
        q.write_buffer(p["buffers"]["cells"], 0, cells)
        self._params("frame", normal_step=normal_step)
        img = self._image_groups()
        self._run([("prep_source", img), ("prep_blur", img), ("maps", img)])

    def prepared(self):
        """The last frame as fused: (depth mm, colour RGB) on the pinhole grid (for checks)."""
        w, h = self.size
        d = np.frombuffer(self.device.queue.read_buffer(self.depth), np.float32).reshape(h, w)
        c = gpu.unpack_colour(np.frombuffer(self.device.queue.read_buffer(self.colour), np.uint32).reshape(h, w))
        return d, c

    def calibrate_ticks(self, kernel="raycast", repeats=200):
        """ns per timestamp unit, from many dispatches of `kernel` against the wall clock (for timing
        only; the result is a few percent high: it includes the submission)."""
        if not self.timestamps:
            return
        self.sync()
        t = time.perf_counter()
        self._run([(kernel, self._image_groups())] * 2, repeats=repeats)
        wall = time.perf_counter() - t
        self.tick_ns = wall * 1e9 / self._span

    def _group(self, kernel, *buffers):
        return self.device.create_bind_group(layout=self.pipes[kernel].get_bind_group_layout(0), entries=[
            {"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}} for i, b in enumerate(buffers)])

    def _params(self, which, **kw):
        self.device.queue.write_buffer(self.uniform[which], 0, gpu.params(self.layout, self.K, self.size, far_mm=self.far_mm, **kw))

    def _run(self, passes, repeats=1):
        """Encode passes [(kernel, workgroups (x, y) or "indirect")] into one submission (`repeats` times
        over); GPU ms per pass (from timestamps, when the adapter has them) land in self.gpu_ms."""
        enc = self.device.create_command_encoder()
        for r in range(repeats):
            for k, (kernel, groups) in enumerate(passes):
                ts = None
                if self.timestamps and r == repeats - 1:
                    ts = {"query_set": self.queries, "beginning_of_pass_write_index": 2 * k, "end_of_pass_write_index": 2 * k + 1}
                elif self.timestamps and r == 0 and k == 0:
                    ts = {"query_set": self.queries, "beginning_of_pass_write_index": 15}
                cp = enc.begin_compute_pass(timestamp_writes=ts)
                cp.set_pipeline(self.pipes[kernel])
                cp.set_bind_group(0, self.groups[kernel])
                if groups == "indirect":
                    cp.dispatch_workgroups_indirect(self.args, 0)
                else:
                    cp.dispatch_workgroups(*groups)
                cp.end()
        if self.timestamps:
            enc.resolve_query_set(self.queries, 0, 16, self.stamps, 0)
        self.device.queue.submit([enc.finish()])
        if self.timestamps:
            ticks = np.frombuffer(self.device.queue.read_buffer(self.stamps), np.uint64).astype(np.float64)
            self._span = ticks[2 * len(passes) - 1] - (ticks[15] if repeats > 1 else ticks[0])
            for k, (kernel, _) in enumerate(passes):
                self.gpu_ms[kernel] = (ticks[2 * k + 1] - ticks[2 * k]) * self.tick_ns / 1e6

    def _image_groups(self, stride=1):
        w, h = ((n + stride - 1) // stride for n in self.size)
        return (w + 15) // 16, (h + 15) // 16

    def upload(self, depth_mm, rgb):
        q = self.device.queue
        q.write_buffer(self.depth, 0, np.ascontiguousarray(depth_mm, np.float32))
        q.write_buffer(self.colour, 0, gpu.pack_colour(rgb))

    def integrate(self, depth_mm, rgb, pose_m):
        """Fuse one frame (fusion.Engine's interface: pose camera -> world, translation in metres)."""
        if np.count_nonzero(depth_mm) < 100:
            self.empty += 1
            return
        self.fuse(depth_mm, rgb, gpu.pose_mm(pose_m))

    def fuse(self, depth_mm, rgb, pose):
        """Upload a frame, make its maps, allocate the bricks it touches and integrate them (pose in mm)."""
        self.upload(depth_mm, rgb)
        self._params("frame", pose=pose, frame=self.frames)
        self.device.queue.write_buffer(self.args, 0, np.array([0, 1, 1], np.uint32))
        self._run([("maps", self._image_groups()), ("allocate", self._image_groups(gpu.ALLOC_STRIDE)), ("clamp", (1,)),
                   ("integrate", "indirect")])
        self.frames += 1

    def raycast(self, pose, min_weight=gpu.RAY_MIN_WEIGHT, normal_px=0, rect=None):
        """Ray cast the volume from `pose` (camera -> world, mm) into the model maps (kept on the GPU);
        normals from the TSDF's gradient, or with `normal_px` from the hits that many pixels away (as
        the frame's maps). `rect` (x0, y0, x1, y1): only those pixels, the rest of the maps cleared."""
        img = self._image_groups()
        if rect is None:
            self._params("ray", pose=pose, min_weight=min_weight, normal_step=normal_px)
            self._run([("raycast", img)] + ([("model_normals", img)] if normal_px else []))
            return
        x0, y0, x1, y1 = (int(v) for v in rect)
        x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
        self._params("ray", pose=pose, min_weight=min_weight, normal_step=normal_px, rect=(x0, y0, x1, y1))
        enc = self.device.create_command_encoder()
        enc.clear_buffer(self.rvmap)
        enc.clear_buffer(self.rnmap)
        self.device.queue.submit([enc.finish()])
        self._run([("raycast", ((x1 - x0 + 15) // 16, (y1 - y0 + 15) // 16))]
                  + ([("model_normals", img)] if normal_px else []))

    def icp(self, pose, ref_pose, centre, **options):
        """ICP sums (gpu.icp_sums) of the last fused/uploaded frame's maps at `pose` against the last ray
        cast, which was made from `ref_pose` (both camera -> world, mm). `options`: gpu.icp_sums's tukey,
        gate, area, reach_mm, stride, base."""
        self._params("icp", pose=pose, ref_inv=np.linalg.inv(ref_pose), centre=centre, **options)
        self._run([("icp", (gpu.ICP_GROUPS,))])
        return gpu.reduce_icp(np.frombuffer(self.device.queue.read_buffer(self.partials), np.float32))

    def preview(self, pose, base=(0.0, 0.0, 0.0, 1.0), min_weight=4.0, step=2):
        """The live view (gpu.preview_codes) from the last ray cast, made from `pose` (camera -> world, mm),
        and the last frame's maps: uint8 (h // step, w // step). `base`: (n, c), the surface above it."""
        self._params("view", pose=pose, min_weight=min_weight, stride=step, base=tuple(float(v) for v in base))
        w, h = self.size
        ow, oh = w // step, h // step
        pw = (ow + 3) // 4
        self._run([("preview", ((pw + 15) // 16, (oh + 15) // 16))])
        words = np.frombuffer(self.device.queue.read_buffer(self.codes, 0, 4 * pw * oh), np.uint32)
        return words.view(np.uint8).reshape(oh, 4 * pw)[:, :ow].copy()

    def mesh(self, min_weight=4):
        """(vertices world mm, triangles, colours) of the surface where all eight voxels of a cube were
        seen in at least `min_weight` frames (fusion.Engine.mesh's rule), by marching cubes."""
        n = self.counts()[0]
        if not n:
            return np.empty((0, 3)), np.empty((0, 3), np.int64), np.empty((0, 3), np.uint8)
        max_tris = MESH_BYTES // 60
        self.device.queue.write_buffer(self.uniform["ray"], 0, gpu.params(  # img.z: bricks, img.w: triangle room
            self.layout, self.K, self.size, far_mm=self.far_mm, min_weight=float(min_weight), frame=n, limit=max_tris))
        self.device.queue.write_buffer(self.tri_count, 0, np.zeros(2, np.uint32))
        self._run([("mesh", (256, (n + 255) // 256))])
        tris, full = np.frombuffer(self.device.queue.read_buffer(self.tri_count), np.uint32)
        if full:
            raise RuntimeError(f"mesh of more than {max_tris} triangles: {full} cubes left out")
        raw = np.frombuffer(self.device.queue.read_buffer(self.out, 0, int(tris) * 60), np.uint32) if tris else np.empty(0, np.uint32)
        return gpu.assemble_mesh(raw, self.layout)

    def blocks(self):
        return self.counts()[0]

    def report(self):
        """What fusion's mesh.json notes about the volume."""
        bricks, full, outside = self.counts()
        return {"device": self.device_name, "volume_bricks": self.layout.dims.tolist(), "pool_bricks": bricks,
                "pool_full_refused": full, "touches_outside_volume": outside}

    def frame_maps(self):
        w, h = self.size
        rb = lambda b: np.frombuffer(self.device.queue.read_buffer(b), np.float32).reshape(h, w, 4)
        return rb(self.vmap), rb(self.nmap)

    def model_maps(self):
        w, h = self.size
        rb = lambda b: np.frombuffer(self.device.queue.read_buffer(b), np.float32).reshape(h, w, 4)
        return rb(self.rvmap), rb(self.rnmap)

    def counts(self):
        """(bricks allocated, bricks refused: pool full, brick touches outside the volume)."""
        c = np.frombuffer(self.device.queue.read_buffer(self.counters), np.uint32)
        return int(min(c[0], self.layout.capacity)), int(c[1]), int(c[2])

    def volume(self):
        """(index, tsdf/weight voxels, colour voxels) of the allocated bricks, as numpy arrays."""
        n = self.counts()[0]
        rb = lambda b, size=None: np.frombuffer(self.device.queue.read_buffer(b, 0, size), np.uint32)
        return rb(self.index), rb(self.tsdfw, 4 * 512 * n), rb(self.colvox, 4 * 512 * n)

    def sync(self):
        self.device.queue.read_buffer(self.counters)

    def close(self):
        """Give the GPU memory back now (200-700 MB). wgpu-py's device and queue refer to each other, so a
        device is only freed by Python's cycle collector: waiting for it let a loop of replays hold 3.9
        GB. Destroying the buffers alone frees nothing (measured); dropping the references and
        collecting does."""
        if self.device is None:
            return
        self.device.destroy()
        for name, value in list(vars(self).items()):
            if isinstance(value, (wgpu.GPUObjectBase, dict)) or name in ("prep", "queries", "stamps"):
                setattr(self, name, None)
        self.device = None
        gc.collect()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
