// Murakumo Gallery — Unified Material Showcase Shader
// 16マテリアルを1つのシェーダーで描画する。
// kind (0-15) で切り替え。

struct Camera {
    view_proj: mat4x4<f32>,
    eye_pos: vec3<f32>,
    time: f32,
}

@group(0) @binding(0)
var<uniform> camera: Camera;

// ════════════════════════════════════════════════════
//  Mesh Vertex + Instance
// ════════════════════════════════════════════════════

struct MeshVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

struct InstanceInput {
    @location(3) model_0: vec4<f32>,
    @location(4) model_1: vec4<f32>,
    @location(5) model_2: vec4<f32>,
    @location(6) model_3: vec4<f32>,
    @location(7) kind: f32,
    @location(8) _pad: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) kind: f32,
    @location(3) world_normal: vec3<f32>,
}

@vertex
fn vs_main(vert: MeshVertexInput, inst: InstanceInput) -> VertexOutput {
    let model = mat4x4<f32>(inst.model_0, inst.model_1, inst.model_2, inst.model_3);
    let world = model * vec4<f32>(vert.position, 1.0);
    let world_normal = normalize((model * vec4<f32>(vert.normal, 0.0)).xyz);

    var out: VertexOutput;
    out.clip_position = camera.view_proj * world;
    out.uv = vert.uv;
    out.world_pos = world.xyz;
    out.kind = inst.kind;
    out.world_normal = world_normal;
    return out;
}

// ════════════════════════════════════════════════════
//  Common helpers
// ════════════════════════════════════════════════════

const PI: f32 = 3.14159265;
const TAU: f32 = 6.28318530;

// --- Simplex 2D Noise ---
fn mod289_2(x: vec2<f32>) -> vec2<f32> { return x - floor(x / 289.0) * 289.0; }
fn mod289_3(x: vec3<f32>) -> vec3<f32> { return x - floor(x / 289.0) * 289.0; }
fn permute3(x: vec3<f32>) -> vec3<f32> { return mod289_3((x * 34.0 + 1.0) * x); }

fn simplex2d(v: vec2<f32>) -> f32 {
    let C = vec4<f32>(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
    var i = floor(v + dot(v, C.yy));
    let x0 = v - i + dot(i, C.xx);
    var i1: vec2<f32>;
    if x0.x > x0.y { i1 = vec2<f32>(1.0, 0.0); } else { i1 = vec2<f32>(0.0, 1.0); }
    let x1 = x0 - i1 + C.xx;
    let x2 = x0 + C.zz;
    i = mod289_2(i);
    let p = permute3(permute3(i.y + vec3<f32>(0.0, i1.y, 1.0)) + i.x + vec3<f32>(0.0, i1.x, 1.0));
    var m = max(0.5 - vec3<f32>(dot(x0, x0), dot(x1, x1), dot(x2, x2)), vec3<f32>(0.0));
    m = m * m;
    m = m * m;
    let x_arr = vec3<f32>(2.0 * fract(p * C.www) - 1.0);
    let h_arr = abs(x_arr) - 0.5;
    let ox = floor(x_arr + 0.5);
    let a0 = x_arr - ox;
    m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h_arr * h_arr);
    let g0 = a0.x * x0.x + h_arr.x * x0.y;
    let g1 = a0.y * x1.x + h_arr.y * x1.y;
    let g2 = a0.z * x2.x + h_arr.z * x2.y;
    return 130.0 * dot(m, vec3<f32>(g0, g1, g2));
}

fn simplex_fbm(p: vec2<f32>, octaves: i32) -> f32 {
    var val = 0.0; var amp = 0.5; var freq = 1.0; var pos = p;
    for (var i = 0; i < octaves; i++) {
        val += amp * simplex2d(pos * freq);
        freq *= 2.0; amp *= 0.5;
    }
    return val;
}

// --- 3D Simplex noise (for volumetric ray marching) ---
fn mod289_4(x: vec4<f32>) -> vec4<f32> { return x - floor(x / 289.0) * 289.0; }
fn permute4(x: vec4<f32>) -> vec4<f32> { return mod289_4((x * 34.0 + 1.0) * x); }
fn taylorInvSqrt4(r: vec4<f32>) -> vec4<f32> { return 1.79284291400159 - 0.85373472095314 * r; }

fn simplex3d(v: vec3<f32>) -> f32 {
    let C = vec2<f32>(1.0 / 6.0, 1.0 / 3.0);
    let D = vec4<f32>(0.0, 0.5, 1.0, 2.0);
    var i = floor(v + dot(v, C.yyy));
    let x0 = v - i + dot(i, C.xxx);
    let g = step(x0.yzx, x0.xyz);
    let l = 1.0 - g;
    let i1 = min(g, l.zxy);
    let i2 = max(g, l.zxy);
    let x1 = x0 - i1 + C.xxx;
    let x2 = x0 - i2 + C.yyy;
    let x3 = x0 - D.yyy;
    i = mod289_3(i);
    let p = permute4(permute4(permute4(
        i.z + vec4<f32>(0.0, i1.z, i2.z, 1.0))
      + i.y + vec4<f32>(0.0, i1.y, i2.y, 1.0))
      + i.x + vec4<f32>(0.0, i1.x, i2.x, 1.0));
    let n_ = 0.142857142857;
    let ns = n_ * D.wyz - D.xzx;
    let j = p - 49.0 * floor(p * ns.z * ns.z);
    let x_ = floor(j * ns.z);
    let y_ = floor(j - 7.0 * x_);
    let xx = x_ * ns.x + ns.yyyy;
    let yy = y_ * ns.x + ns.yyyy;
    let h = 1.0 - abs(xx) - abs(yy);
    let b0 = vec4<f32>(xx.xy, yy.xy);
    let b1 = vec4<f32>(xx.zw, yy.zw);
    let s0 = floor(b0) * 2.0 + 1.0;
    let s1 = floor(b1) * 2.0 + 1.0;
    let sh = -step(h, vec4<f32>(0.0));
    let a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    let a1 = b1.xzyw + s1.xzyw * sh.zzww;
    var p0 = vec3<f32>(a0.xy, h.x);
    var p1 = vec3<f32>(a0.zw, h.y);
    var p2 = vec3<f32>(a1.xy, h.z);
    var p3 = vec3<f32>(a1.zw, h.w);
    let norm = taylorInvSqrt4(vec4<f32>(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
    var m = max(0.6 - vec4<f32>(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), vec4<f32>(0.0));
    m = m * m;
    return 42.0 * dot(m * m, vec4<f32>(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}

fn simplex3d_fbm(p: vec3<f32>, octaves: i32) -> f32 {
    var val = 0.0; var amp = 0.5; var freq = 1.0; var pos = p;
    for (var i = 0; i < octaves; i++) {
        val += amp * simplex3d(pos * freq);
        freq *= 2.0; amp *= 0.5;
    }
    return val;
}

// --- Environment reflection ---
fn env_reflect(dir: vec3<f32>, t: f32) -> vec3<f32> {
    let up = dir.y * 0.5 + 0.5;
    let sky = mix(vec3<f32>(0.02, 0.03, 0.08), vec3<f32>(0.08, 0.12, 0.25), up);
    let star_uv = vec2<f32>(atan2(dir.x, dir.z) * 2.0, dir.y * 4.0);
    let star = step(0.98, simplex2d(star_uv * 20.0)) * 0.3;
    let ground = smoothstep(-0.1, -0.3, dir.y) * vec3<f32>(0.05, 0.04, 0.06);
    return sky + vec3<f32>(star) + ground;
}

// --- Legacy hash helpers (still used for cell IDs, glitch, etc.) ---
fn hash11(p: f32) -> f32 {
    var pp = fract(p * 0.1031);
    pp = pp * (pp + 33.33);
    pp = pp * (pp + pp);
    return fract(pp);
}

fn hash21(p: vec2<f32>) -> f32 {
    let pp = fract(p * vec2<f32>(0.1031, 0.1030));
    let pp2 = pp + dot(pp, pp.yx + 33.33);
    return fract((pp2.x + pp2.y) * pp2.x);
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

fn vnoise(p: vec2<f32>) -> f32 {
    return simplex2d(p) * 0.5 + 0.5;
}

fn fbm4(p: vec2<f32>) -> f32 {
    return simplex_fbm(p, 4) * 0.5 + 0.5;
}

fn fbm5(p: vec2<f32>) -> f32 {
    return simplex_fbm(p, 5) * 0.5 + 0.5;
}

fn gradient_noise(p: vec2<f32>) -> f32 {
    return simplex2d(p);
}

fn fbm_grad4(p: vec2<f32>) -> f32 {
    return simplex_fbm(p, 4);
}

// Sphere helper: returns (sphere_normal, n_dot_v, radius_squared)
// Returns discard=true if outside sphere
struct SphereInfo {
    normal: vec3<f32>,
    n_dot_v: f32,
    r: f32,
    valid: bool,
}

fn sphere_from_uv(uv: vec2<f32>, eye_pos: vec3<f32>, world_pos: vec3<f32>) -> SphereInfo {
    var info: SphereInfo;
    let uv_cent = uv - 0.5;
    let r2 = dot(uv_cent, uv_cent) * 4.0;
    if r2 > 1.0 {
        info.valid = false;
        return info;
    }
    info.valid = true;
    info.r = sqrt(r2);
    let nz = sqrt(1.0 - r2);
    info.normal = normalize(vec3<f32>(uv_cent.x * 2.0, uv_cent.y * 2.0, nz));
    let view_dir = normalize(eye_pos - world_pos);
    info.n_dot_v = max(dot(info.normal, view_dir), 0.0);
    return info;
}

// ════════════════════════════════════════════════════
//  0: Bubble — 薄膜干渉シャボン玉
// ════════════════════════════════════════════════════

fn mat_bubble(uv: vec2<f32>, wp: vec3<f32>, ep: vec3<f32>, t: f32) -> vec4<f32> {
    let si = sphere_from_uv(uv, ep, wp);
    if !si.valid { return vec4<f32>(0.0); }

    let uv_cent = uv - 0.5;
    let p = uv_cent * 2.0;
    let view_dir = normalize(ep - wp);

    let thick = 350.0 + p.y * 100.0
        + sin(p.x * 3.0 + p.y * 2.0 + t * 0.4) * 80.0
        + sin(p.y * 7.0 + t * 0.7) * 40.0
        + sin(p.x * 5.0 - p.y * 4.0 + t * 0.2) * 30.0;
    let pd = 2.0 * 1.33 * thick;
    let irid = vec3<f32>(
        0.5 + 0.5 * cos(TAU * pd / 700.0),
        0.5 + 0.5 * cos(TAU * pd / 530.0),
        0.5 + 0.5 * cos(TAU * pd / 430.0),
    );

    let thick_b = thick + 150.0 + sin(p.x * 4.0 - p.y * 3.0 - t * 0.3) * 50.0;
    let pd_b = 2.0 * 1.33 * thick_b;
    let irid_b = vec3<f32>(
        0.5 + 0.5 * cos(TAU * pd_b / 700.0),
        0.5 + 0.5 * cos(TAU * pd_b / 530.0),
        0.5 + 0.5 * cos(TAU * pd_b / 430.0),
    );

    let fresnel = pow(1.0 - si.n_dot_v, 3.0);
    let range_f = max(max(irid.x, irid.y), irid.z) - min(min(irid.x, irid.y), irid.z);
    let mask_f = smoothstep(0.4, 0.95, range_f) * (0.005 + fresnel * 0.025);
    let range_b = max(max(irid_b.x, irid_b.y), irid_b.z) - min(min(irid_b.x, irid_b.y), irid_b.z);
    let mask_b = smoothstep(0.4, 0.95, range_b) * (0.003 + fresnel * 0.012);

    let light = normalize(vec3<f32>(0.3, 1.0, 0.6));
    let h = normalize(light + view_dir);
    let spec = pow(max(dot(si.normal, h), 0.0), 256.0) * 0.7;
    let light2 = normalize(vec3<f32>(-0.6, 0.5, 0.3));
    let h2 = normalize(light2 + view_dir);
    let spec2 = pow(max(dot(si.normal, h2), 0.0), 180.0) * 0.3;

    let sheen = fresnel * vec3<f32>(0.2, 0.25, 0.35) * 0.08;
    let bands = irid * mask_f + irid_b * mask_b;
    let band_lum = max(bands.x, max(bands.y, bands.z));
    let shell = vec3<f32>(0.15, 0.18, 0.25) * 0.08;

    // Environment reflection blended with interference
    let refl_dir = reflect(-view_dir, si.normal);
    let env_col = env_reflect(refl_dir, t);
    let env_contrib = env_col * fresnel * 0.4;

    let emit = shell + sheen + bands + vec3<f32>(spec + spec2) + env_contrib;
    let a = 0.08 + band_lum + spec + spec2 + fresnel * 0.15;
    return vec4<f32>(emit, a);
}

// ════════════════════════════════════════════════════
//  1: Glass — ガラスマテリアル
// ════════════════════════════════════════════════════

fn mat_glass(uv: vec2<f32>, wp: vec3<f32>, ep: vec3<f32>, t: f32) -> vec4<f32> {
    let si = sphere_from_uv(uv, ep, wp);
    if !si.valid { return vec4<f32>(0.0); }

    let view_dir = normalize(ep - wp);
    let fresnel = pow(1.0 - si.n_dot_v, 4.0) * 0.9 + 0.1;

    // Distorted refraction pattern
    let uv_cent = uv - 0.5;
    let refract_offset = si.normal.xy * 0.15;
    let refracted_uv = uv + refract_offset;
    let pattern = sin(refracted_uv.x * 20.0 + t * 0.5) * sin(refracted_uv.y * 15.0 + t * 0.3) * 0.5 + 0.5;

    // Caustic-like highlights
    let caustic = pow(pattern, 3.0) * 0.4;

    let light = normalize(vec3<f32>(0.3, 0.8, 0.5));
    let h = normalize(light + view_dir);
    let spec = pow(max(dot(si.normal, h), 0.0), 512.0) * 1.0;
    let spec2 = pow(max(dot(si.normal, h), 0.0), 128.0) * 0.3;

    let base_color = vec3<f32>(0.15, 0.2, 0.3);

    // Environment reflection for glass
    let refl_dir = reflect(-view_dir, si.normal);
    let env_col = env_reflect(refl_dir, t);

    var color = mix(base_color * (0.3 + caustic), env_col, fresnel);
    color += vec3<f32>(spec + spec2);

    // Rim light
    let rim = pow(1.0 - si.n_dot_v, 6.0);
    color += vec3<f32>(0.4, 0.5, 0.7) * rim * 0.3;

    // Moving specular streak
    let streak_pos = fract(t * 0.08) * 3.0 - 1.0;
    let streak = exp(-pow(uv.x * 0.5 + uv.y * 0.5 - streak_pos, 2.0) * 400.0) * 0.3;
    color += vec3<f32>(streak);

    let alpha = 0.15 + fresnel * 0.4 + spec * 0.5;
    return vec4<f32>(color, clamp(alpha, 0.0, 1.0));
}

// ════════════════════════════════════════════════════
//  2: Portal — ポータルエフェクト
// ════════════════════════════════════════════════════

fn mat_portal(uv: vec2<f32>, wp: vec3<f32>, ep: vec3<f32>, t: f32) -> vec4<f32> {
    let uv_cent = uv - 0.5;
    let dist = length(uv_cent) * 2.0;
    if dist > 1.0 { return vec4<f32>(0.0); }

    let angle = atan2(uv_cent.y, uv_cent.x);

    // Spinning vortex rings
    let ring1 = sin(dist * 15.0 - t * 3.0 + angle * 2.0) * 0.5 + 0.5;
    let ring2 = sin(dist * 10.0 - t * 2.5 - angle * 3.0) * 0.5 + 0.5;
    let ring3 = sin(dist * 20.0 - t * 4.0 + angle * 1.0) * 0.5 + 0.5;

    let combined = ring1 * 0.4 + ring2 * 0.35 + ring3 * 0.25;

    // Edge glow
    let edge = smoothstep(1.0, 0.6, dist);
    let rim = smoothstep(0.5, 1.0, dist) * smoothstep(1.0, 0.85, dist) * 2.0;

    // Core brightness
    let core = exp(-dist * 3.0);

    // Color: purple/cyan vortex
    let col1 = vec3<f32>(0.4, 0.1, 0.8);
    let col2 = vec3<f32>(0.1, 0.6, 0.9);
    let col3 = vec3<f32>(0.9, 0.3, 0.6);

    var color = mix(col1, col2, ring1) * combined;
    color += col3 * rim * 0.5;
    color += vec3<f32>(0.8, 0.9, 1.0) * core * 0.6;

    // Noise distortion
    let noise_val = vnoise(uv * 5.0 + vec2<f32>(t * 0.5, t * 0.3));
    color += col2 * noise_val * 0.2 * edge;

    let alpha = edge * (combined * 0.6 + rim * 0.3 + core * 0.4);
    return vec4<f32>(color, clamp(alpha, 0.0, 1.0));
}

// ════════════════════════════════════════════════════
//  3: Grid — グリッドマテリアル
// ════════════════════════════════════════════════════

fn mat_grid(uv: vec2<f32>, wp: vec3<f32>, ep: vec3<f32>, t: f32) -> vec4<f32> {
    let p = uv * 10.0;
    let grid_x = abs(fract(p.x) - 0.5);
    let grid_y = abs(fract(p.y) - 0.5);

    let line_w = 0.02;
    let line_x = smoothstep(line_w, 0.0, grid_x);
    let line_y = smoothstep(line_w, 0.0, grid_y);
    let grid_val = max(line_x, line_y);

    // Subgrid
    let sp = uv * 40.0;
    let sg_x = smoothstep(0.01, 0.0, abs(fract(sp.x) - 0.5));
    let sg_y = smoothstep(0.01, 0.0, abs(fract(sp.y) - 0.5));
    let subgrid = max(sg_x, sg_y) * 0.15;

    // Pulse wave
    let pulse_dist = length(uv - 0.5);
    let pulse = sin(pulse_dist * 20.0 - t * 3.0) * 0.5 + 0.5;
    let pulse_ring = smoothstep(0.05, 0.0, abs(fract(pulse_dist * 3.0 - t * 0.5) - 0.5)) * 0.4;

    // Intersection dots
    let dot_x = smoothstep(0.08, 0.0, grid_x);
    let dot_y = smoothstep(0.08, 0.0, grid_y);
    let dots = dot_x * dot_y * 0.8;

    let base_col = vec3<f32>(0.05, 0.3, 0.5);
    let bright_col = vec3<f32>(0.2, 0.7, 1.0);

    var color = base_col * grid_val + bright_col * (dots + pulse_ring);
    color += vec3<f32>(0.02, 0.08, 0.12) * subgrid;
    color *= (0.8 + pulse * 0.2);

    // Edge fade
    let edge = smoothstep(0.0, 0.05, uv.x) * smoothstep(1.0, 0.95, uv.x)
             * smoothstep(0.0, 0.05, uv.y) * smoothstep(1.0, 0.95, uv.y);

    let lum = max(color.x, max(color.y, color.z));
    let alpha = clamp(lum * 2.0 + 0.15, 0.0, 1.0) * edge;
    return vec4<f32>(color, alpha);
}

// ════════════════════════════════════════════════════
//  4: Water — 水面
// ════════════════════════════════════════════════════

fn wave_height(p: vec2<f32>, t: f32) -> f32 {
    var height = 0.0;
    var amp = 0.5;
    var freq = 3.0;
    var pos = p;
    for (var i = 0; i < 4; i++) {
        let w1 = sin(pos.x * freq + t * 1.3) * amp;
        let w2 = sin(pos.y * freq * 0.8 + t * 0.9 + 2.1) * amp * 0.7;
        let w3 = sin((pos.x + pos.y) * freq * 0.6 + t * 1.1 + 4.3) * amp * 0.5;
        height += w1 + w2 + w3;
        amp *= 0.5;
        freq *= 2.0;
        let c = cos(0.5);
        let s = sin(0.5);
        pos = vec2<f32>(pos.x * c - pos.y * s, pos.x * s + pos.y * c);
    }
    return height;
}

fn water_caustics(p: vec2<f32>, t: f32) -> f32 {
    let a1 = 0.4; let c1 = cos(a1); let s1 = sin(a1);
    let p1 = vec2<f32>(p.x * c1 - p.y * s1, p.x * s1 + p.y * c1) * 6.0;
    let g1 = sin(p1.x + t * 1.2) * sin(p1.y + t * 0.8);
    let a2 = 1.2; let c2 = cos(a2); let s2 = sin(a2);
    let p2 = vec2<f32>(p.x * c2 - p.y * s2, p.x * s2 + p.y * c2) * 8.0;
    let g2 = sin(p2.x - t * 0.9) * sin(p2.y + t * 1.1);
    let a3 = 2.4; let c3 = cos(a3); let s3 = sin(a3);
    let p3 = vec2<f32>(p.x * c3 - p.y * s3, p.x * s3 + p.y * c3) * 5.0;
    let g3 = sin(p3.x + t * 0.7) * sin(p3.y - t * 1.3);
    let combined = (g1 + g2 + g3) / 3.0;
    return pow(max(combined, 0.0), 2.0);
}

fn mat_water(uv: vec2<f32>, wp: vec3<f32>, ep: vec3<f32>, t: f32) -> vec4<f32> {
    let uv_cent = uv - 0.5;
    let dist = length(uv_cent) * 2.0;
    // Circular mask
    let edge = 1.0 - smoothstep(0.9, 1.0, dist);
    if edge <= 0.0 { return vec4<f32>(0.0); }

    let tt = t * 0.6;

    // Wave normal via central difference
    let eps = 0.008;
    let h_c = wave_height(uv * 2.0, tt);
    let h_r = wave_height((uv + vec2<f32>(eps, 0.0)) * 2.0, tt);
    let h_u = wave_height((uv + vec2<f32>(0.0, eps)) * 2.0, tt);
    let normal = normalize(vec3<f32>((h_c - h_r) / eps * 0.3, 1.0, (h_c - h_u) / eps * 0.3));

    // Fresnel
    let view_dir = normalize(vec3<f32>(0.0, 1.0, 0.3));
    let n_dot_v = max(dot(normal, view_dir), 0.0);
    let fresnel = pow(1.0 - n_dot_v, 5.0) * 0.7 + 0.1;

    // Deep ocean color
    let deep = vec3<f32>(0.01, 0.04, 0.15);
    let mid = vec3<f32>(0.02, 0.12, 0.35);
    let shallow = vec3<f32>(0.05, 0.25, 0.5);
    let depth_t = 0.5 + h_c * 0.3;
    let base = mix(deep, mix(mid, shallow, depth_t), depth_t);

    // Strong caustics
    let c_uv = uv * 2.0 + normal.xz * 0.1;
    let caustic = water_caustics(c_uv, tt) * 1.2;
    let caustic_col = vec3<f32>(0.15, 0.5, 0.7) * caustic;

    // Specular — sun reflection on water
    let light = normalize(vec3<f32>(0.2, 0.9, 0.3));
    let hv = normalize(light + view_dir);
    let spec = pow(max(dot(normal, hv), 0.0), 128.0) * 0.8;
    // Secondary scattered highlights
    let spec2 = pow(max(dot(normal, hv), 0.0), 32.0) * 0.15;

    // Sky reflection
    let sky = vec3<f32>(0.2, 0.35, 0.6);
    var color = mix(base, sky, fresnel * 0.5) + caustic_col + vec3<f32>(spec + spec2);

    // Foam on wave crests
    let foam = smoothstep(0.25, 0.45, h_c) * 0.2;
    color = mix(color, vec3<f32>(0.8, 0.9, 0.95), foam);

    return vec4<f32>(color, edge * 0.95);
}

// ════════════════════════════════════════════════════
//  5: Fire — 炎 (volumetric ray-march)
// ════════════════════════════════════════════════════

fn fire_density(p: vec3<f32>, t: f32) -> f32 {
    // Domain warp for organic turbulence
    let warp = vec3<f32>(
        simplex3d(p * 2.0 + vec3<f32>(t * 0.3, 0.0, 5.2)),
        simplex3d(p * 2.0 + vec3<f32>(0.0, t * 0.2, 9.1)),
        simplex3d(p * 2.0 + vec3<f32>(3.7, t * 0.4, 0.0))
    ) * 0.3;
    let warped = p + warp;

    // Scroll upward — fire rises
    let scroll = vec3<f32>(warped.x, warped.y - t * 2.5, warped.z);

    // Multi-scale noise
    let n1 = simplex3d_fbm(scroll * 2.5, 5) * 0.6;
    let n2 = simplex3d(scroll * 5.0 + vec3<f32>(5.2, 1.3, 3.7)) * 0.25;
    let n3 = simplex3d(scroll * 10.0 + vec3<f32>(1.7, 9.2, 2.1)) * 0.1;
    let n4 = simplex3d(scroll * 20.0) * 0.05; // high-freq detail
    let noise = n1 + n2 + n3 + n4;

    // 3D tapered shape — narrow at top, wide at base
    let radial = length(vec2<f32>(p.x, p.z));
    let width = max(0.0, 0.5 - p.y * 0.4); // narrows with height
    let radial_falloff = smoothstep(width, width * 0.3, radial);
    let height_falloff = smoothstep(1.2, 0.0, p.y) * smoothstep(-0.1, 0.1, p.y);

    let d = (noise * 0.5 + 0.5) * radial_falloff * height_falloff;
    return max(d - 0.1, 0.0) * 3.0;
}

fn fire_color(d: f32, p: vec3<f32>, t: f32) -> vec3<f32> {
    // 5-stop gradient: black → dark red → orange → yellow → white
    let col_black = vec3<f32>(0.1, 0.0, 0.0);
    let col_red = vec3<f32>(0.8, 0.1, 0.0);
    let col_orange = vec3<f32>(1.0, 0.4, 0.0);
    let col_yellow = vec3<f32>(1.0, 0.85, 0.2);
    let col_white = vec3<f32>(1.0, 0.98, 0.9);

    var c: vec3<f32>;
    if d > 0.8 {
        c = mix(col_yellow, col_white, (d - 0.8) / 0.2);
    } else if d > 0.5 {
        c = mix(col_orange, col_yellow, (d - 0.5) / 0.3);
    } else if d > 0.2 {
        c = mix(col_red, col_orange, (d - 0.2) / 0.3);
    } else {
        c = mix(col_black, col_red, d / 0.2);
    }

    // Height-based tint: more blue at very top (hot gas)
    let blue_tint = smoothstep(0.8, 1.2, p.y) * 0.2;
    c = mix(c, vec3<f32>(0.3, 0.4, 0.8), blue_tint);

    return c;
}

fn mat_fire(uv: vec2<f32>, wp: vec3<f32>, ep: vec3<f32>, t: f32) -> vec4<f32> {
    let tt = t * 1.2;

    // Ray from camera through the volume
    let view_dir = normalize(wp - ep);
    let local_x = (uv.x - 0.5) * 2.0;
    let local_y = 1.0 - uv.y;
    let ray_origin = vec3<f32>(local_x, local_y, -0.6);
    let ray_dir = normalize(vec3<f32>(view_dir.x * 0.4, view_dir.y * 0.15, 1.0));

    var accum_color = vec3<f32>(0.0);
    var accum_alpha = 0.0;
    let steps = 32;
    let step_size = 1.2 / f32(steps);

    for (var i = 0; i < steps; i++) {
        let pos = ray_origin + ray_dir * (f32(i) * step_size);
        let d = fire_density(pos, tt);
        if d > 0.001 {
            let fc = fire_color(d, pos, tt);

            // HDR emission — bright core for bloom
            let emission = fc * d * step_size * 15.0;
            accum_color += emission * (1.0 - accum_alpha);
            accum_alpha += d * step_size * 5.0 * (1.0 - accum_alpha);
        }
        if accum_alpha >= 0.97 { break; }
    }

    // Ember particles — small bright dots
    let ember_uv = uv * 8.0;
    let ember_cell = floor(ember_uv);
    let ember_hash = fract(sin(dot(ember_cell, vec2<f32>(127.1, 311.7))) * 43758.5453);
    let ember_y = fract(ember_hash * 3.0 - tt * (0.5 + ember_hash));
    let ember_x = fract(ember_hash * 7.0) + sin(tt * 2.0 + ember_hash * 10.0) * 0.1;
    let ember_dist = length(fract(ember_uv) - vec2<f32>(ember_x, ember_y));
    let ember = smoothstep(0.05, 0.0, ember_dist) * step(0.7, ember_hash) * 0.5;
    accum_color += vec3<f32>(1.0, 0.6, 0.1) * ember;
    accum_alpha = max(accum_alpha, ember * 0.5);

    // Multi-frequency flicker
    let flicker = 0.85 + 0.15 * sin(tt * 15.0) * sin(tt * 9.3 + 1.7) * sin(tt * 23.0 + 3.1);
    accum_color *= flicker;

    let alpha = clamp(accum_alpha, 0.0, 1.0);
    if alpha < 0.01 { return vec4<f32>(0.0); }
    return vec4<f32>(accum_color, alpha);
}

// ════════════════════════════════════════════════════
//  6: Smoke — 煙 (volumetric ray-march with Beer-Lambert)
// ════════════════════════════════════════════════════

fn smoke_density(p: vec3<f32>, t: f32) -> f32 {
    // Domain warping for organic swirls
    let warp = vec3<f32>(
        simplex3d(p * 1.5 + vec3<f32>(t * 0.1, 0.0, 3.7)),
        simplex3d(p * 1.5 + vec3<f32>(0.0, t * 0.08, 7.3)),
        simplex3d(p * 1.5 + vec3<f32>(5.2, t * 0.12, 0.0))
    );
    let warped = p + warp * 0.4;
    // Scroll upward
    let scroll = vec3<f32>(warped.x, warped.y - t * 0.6, warped.z);
    let n = simplex3d_fbm(scroll * 2.5, 4) * 0.5 + 0.5;
    // Radial falloff from center column
    let radial = length(vec2<f32>(p.x, p.z));
    let shape = smoothstep(0.8, 0.2, radial) * smoothstep(0.0, 0.15, p.y) * smoothstep(1.0, 0.7, p.y);
    return max(n * shape - 0.1, 0.0) * 1.5;
}

fn mat_smoke(uv: vec2<f32>, wp: vec3<f32>, ep: vec3<f32>, t: f32) -> vec4<f32> {
    let tt = t * 0.5;
    let center = uv - 0.5;
    let dist = length(center);
    let edge = 1.0 - smoothstep(0.4, 0.5, dist);
    if edge <= 0.0 { return vec4<f32>(0.0); }

    // Map UV to 3D volume
    let ray_origin = vec3<f32>((uv.x - 0.5) * 2.0, 1.0 - uv.y, -0.5);
    let ray_dir = vec3<f32>(0.0, 0.0, 1.0);
    let step_size = 1.0 / 14.0;

    var transmittance = 1.0;
    var accum_color = vec3<f32>(0.0);
    let absorption = 3.5;

    // Light direction for simple scattering
    let light_dir = normalize(vec3<f32>(0.3, 1.0, 0.5));
    let base_col = vec3<f32>(0.28, 0.30, 0.38);
    let scatter_col = vec3<f32>(0.35, 0.38, 0.45);

    for (var i = 0; i < 14; i++) {
        let pos = ray_origin + ray_dir * (f32(i) * step_size);
        let d = smoke_density(pos, tt);
        if d > 0.001 {
            // Beer-Lambert absorption
            let extinct = d * absorption * step_size;
            let tr_step = exp(-extinct);

            // Simple forward scattering approximation
            let light_sample = smoke_density(pos + light_dir * 0.15, tt);
            let light_atten = exp(-light_sample * 1.5);
            let lit_color = mix(base_col, scatter_col, light_atten * 0.6);

            // Accumulate with absorption
            accum_color += lit_color * d * step_size * transmittance * 2.0;
            transmittance *= tr_step;
        }
        if transmittance < 0.05 { break; }
    }

    let alpha = (1.0 - transmittance) * edge * 0.7;
    if alpha < 0.01 { return vec4<f32>(0.0); }
    return vec4<f32>(accum_color, clamp(alpha, 0.0, 1.0));
}

// ════════════════════════════════════════════════════
//  7: Aurora — オーロラ
// ════════════════════════════════════════════════════

fn aurora_n1d(p: f32) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(hash11(i), hash11(i + 1.0), u);
}

fn aurora_band(x: f32, t: f32, freq: f32, phase: f32, speed: f32, amp: f32) -> f32 {
    var wave = sin(x * freq + t * speed + phase) * amp;
    wave += sin(x * freq * 1.7 + t * speed * 0.6 + phase * 2.3) * amp * 0.5;
    wave += sin(x * freq * 0.4 + t * speed * 1.3 + phase * 0.7) * amp * 0.3;
    wave += aurora_n1d(x * 2.0 + t * speed * 0.3) * amp * 0.4;
    return wave;
}

fn mat_aurora(uv: vec2<f32>, wp: vec3<f32>, ep: vec3<f32>, t: f32) -> vec4<f32> {
    let tt = t * 0.5;
    let x = uv.x;
    let y = uv.y;

    let vert_mask = smoothstep(0.0, 0.3, y) * smoothstep(1.0, 0.6, y);
    if vert_mask <= 0.01 { return vec4<f32>(0.0); }

    var intensity = 0.0;
    var cmix = 0.0;

    let b1c = 0.5 + aurora_band(x, tt, 3.0, 0.0, 1.0, 0.15);
    let b1d = abs(y - b1c);
    let b1w = 0.12 + aurora_n1d(x * 3.0 + tt * 0.4) * 0.06;
    let b1 = smoothstep(b1w, 0.0, b1d);
    intensity += b1 * 0.6; cmix += b1 * 0.3;

    let b2c = 0.45 + aurora_band(x, tt, 4.5, 2.1, 0.7, 0.12);
    let b2d = abs(y - b2c);
    let b2w = 0.08 + aurora_n1d(x * 4.0 + tt * 0.6 + 3.0) * 0.05;
    let b2 = smoothstep(b2w, 0.0, b2d);
    intensity += b2 * 0.4; cmix += b2 * 0.7;

    let b3c = 0.55 + aurora_band(x, tt, 6.0, 4.5, 1.3, 0.09);
    let b3d = abs(y - b3c);
    let b3w = 0.03 + aurora_n1d(x * 5.0 + tt * 0.8 + 7.0) * 0.02;
    let b3 = smoothstep(b3w, 0.0, b3d);
    intensity += b3 * 0.8; cmix += b3 * 0.5;

    let b4c = 0.48 + aurora_band(x, tt, 2.0, 1.3, 0.5, 0.18);
    let b4d = abs(y - b4c);
    let b4w = 0.2 + aurora_n1d(x * 2.0 + tt * 0.2 + 5.0) * 0.1;
    let b4 = smoothstep(b4w, 0.0, b4d);
    intensity += b4 * 0.2; cmix += b4 * 0.9;

    if intensity <= 0.01 { return vec4<f32>(0.0); }

    cmix = clamp(cmix / max(intensity, 0.001), 0.0, 1.0);
    let shimmer = vnoise(vec2<f32>(x * 20.0 + tt * 2.0, y * 30.0 + tt * 0.5));
    intensity *= 0.85 + shimmer * 0.15;
    let ray = 0.7 + 0.3 * sin(x * 40.0 + tt * 0.3) * sin(x * 17.0 - tt * 0.7);
    intensity *= ray;

    let col1 = vec3<f32>(0.1, 0.8, 0.4);
    let col2 = vec3<f32>(0.3, 0.2, 0.9);
    let pos_shift = sin(x * 5.0 + tt * 0.3) * 0.3 + 0.5;
    let final_t = mix(cmix, pos_shift, 0.4);
    var aurora_col = mix(col1, col2, final_t);
    aurora_col += vec3<f32>(pow(intensity, 0.5) * 0.3);
    aurora_col *= 1.5;

    let glow = pow(intensity, 0.7);
    let h_fade = smoothstep(0.0, 0.1, x) * smoothstep(1.0, 0.9, x);
    let alpha = clamp(glow * vert_mask * h_fade, 0.0, 1.0);
    return vec4<f32>(aurora_col * glow, alpha);
}

// ════════════════════════════════════════════════════
//  8: Hologram — ホログラム
// ════════════════════════════════════════════════════

fn mat_hologram(uv: vec2<f32>, wp: vec3<f32>, ep: vec3<f32>, t: f32) -> vec4<f32> {
    var uuv = uv;
    let glitch_block = floor(uuv.y * 20.0 + t * 3.0);
    let glitch_rand = hash11(glitch_block + floor(t * 7.0));
    let glitch_active = step(0.85, glitch_rand);
    let glitch_offset = (hash11(glitch_block * 3.7 + t) - 0.5) * 0.15 * 0.5 * glitch_active;
    uuv.x = fract(uuv.x + glitch_offset);

    let scan_y = uuv.y * 40.0 + t * 3.0;
    let scan_line = sin(scan_y * PI) * 0.5 + 0.5;
    let scan_mask = smoothstep(0.3, 0.7, scan_line);

    let bar_pos = fract(t * 0.3);
    let bar_dist = abs(uuv.y - bar_pos);
    let bar_wrap = min(bar_dist, 1.0 - bar_dist);
    let scan_bar = smoothstep(0.05, 0.0, bar_wrap) * 0.5;

    let base = vec3<f32>(0.1, 0.6, 1.0);
    let aberr = 0.004;
    let col_r = base.r * (1.0 + (aberr) * 10.0);
    let col_g = base.g;
    let col_b = base.b * (1.0 + (-aberr) * 10.0);
    var color = vec3<f32>(col_r, col_g, col_b);

    color *= (0.6 + 0.4 * scan_mask);
    color += vec3<f32>(scan_bar) * base;

    let uv_cent = uuv - 0.5;
    let edge = pow(length(uv_cent) * 2.0, 2.0) * 0.5;
    color += base * edge;

    let flicker = 1.0 - 0.15 * (sin(t * 5.0) * sin(t * 13.5) + 1.0) * 0.5;
    let flash = 1.0 + step(0.97, hash11(floor(t * 4.0))) * 2.0;
    color *= flicker * flash;

    color += vec3<f32>(sin(uuv.y * 400.0 + t * 10.0) * 0.03) * base;

    if glitch_active > 0.5 {
        let corrupt = hash21(vec2<f32>(uuv.x * 10.0, glitch_block));
        color = mix(color, vec3<f32>(corrupt, 1.0 - corrupt, corrupt * 0.5), 0.3);
    }

    color = clamp(color, vec3<f32>(0.0), vec3<f32>(5.0));
    let alpha = 0.7 * (0.5 + 0.5 * scan_mask) * flicker;
    return vec4<f32>(color, alpha);
}

// ════════════════════════════════════════════════════
//  9: Crystal — クリスタル
// ════════════════════════════════════════════════════

fn voronoi_crystal(uv: vec2<f32>, scale: f32, t: f32) -> vec3<f32> {
    let p = uv * scale;
    let ip = floor(p);
    let fp = fract(p);
    var min_d = 1.0;
    var sec_d = 1.0;
    var cell_id = vec2<f32>(0.0);
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let nb = vec2<f32>(f32(x), f32(y));
            let pt = hash22(ip + nb);
            let anim = nb + 0.5 + 0.3 * sin(t * 0.5 + TAU * pt);
            let diff = anim - fp;
            let d = length(diff);
            if d < min_d {
                sec_d = min_d;
                min_d = d;
                cell_id = ip + nb;
            } else if d < sec_d {
                sec_d = d;
            }
        }
    }
    return vec3<f32>(min_d, sec_d - min_d, hash21(cell_id));
}

fn mat_crystal(uv: vec2<f32>, wp: vec3<f32>, ep: vec3<f32>, t: f32) -> vec4<f32> {
    let si = sphere_from_uv(uv, ep, wp);
    if !si.valid { return vec4<f32>(0.0); }

    let view_dir = normalize(ep - wp);
    let vor = voronoi_crystal(uv, 6.0, t);
    let edge_dist = vor.y;
    let cell_rand = vor.z;

    let facet_angle = cell_rand * TAU;
    let facet_n = normalize(si.normal + vec3<f32>(cos(facet_angle) * 0.15, sin(facet_angle) * 0.15, 0.0));

    let base = vec3<f32>(0.4, 0.6, 0.9);
    var crystal_col = base * (0.85 + 0.3 * cell_rand);
    crystal_col += vec3<f32>(cell_rand * 0.15, -cell_rand * 0.075, cell_rand * 0.045);

    let l1 = normalize(vec3<f32>(0.5, 0.8, 0.6));
    let h1 = normalize(l1 + view_dir);
    let spec1 = pow(max(dot(facet_n, h1), 0.0), 512.0);
    let l2 = normalize(vec3<f32>(-0.4, 0.3, 0.8));
    let h2 = normalize(l2 + view_dir);
    let spec2 = pow(max(dot(facet_n, h2), 0.0), 256.0);

    let edge_sparkle = pow(smoothstep(0.05, 0.0, edge_dist), 2.0) * 1.0;
    let sparkle = spec1 * 1.5 + spec2 * 0.8 + edge_sparkle;

    let fresnel = pow(1.0 - si.n_dot_v, 3.0);
    let glow_f = 0.5 * (0.5 + 0.5 * sin(t * 1.5 + cell_rand * TAU));
    let internal = base * glow_f * (1.0 - fresnel);

    let rainbow = vec3<f32>(
        0.5 + 0.5 * sin(cell_rand * 20.0),
        0.5 + 0.5 * sin(cell_rand * 20.0 + 2.094),
        0.5 + 0.5 * sin(cell_rand * 20.0 + 4.189)
    );
    let dispersion = rainbow * fresnel * smoothstep(0.15, 0.0, edge_dist) * 0.6;

    // Environment reflection per facet
    let crystal_refl_dir = reflect(-view_dir, facet_n);
    let crystal_env = env_reflect(crystal_refl_dir, t);

    var color = crystal_col * (0.4 + 0.6 * si.n_dot_v) + internal + dispersion + vec3<f32>(sparkle);
    color += crystal_env * fresnel * 0.5;
    color *= (0.7 + 0.3 * smoothstep(0.02, 0.04, edge_dist));

    let alpha = 0.9 * (0.8 + 0.2 * si.n_dot_v);
    return vec4<f32>(color, alpha);
}

// ════════════════════════════════════════════════════
//  10: Metal — 異方性メタル
// ════════════════════════════════════════════════════

fn fbm_metal(p: vec2<f32>) -> f32 {
    var val = 0.0;
    var amp = 0.5;
    var pos = p;
    for (var i = 0; i < 4; i++) {
        val += amp * vnoise(pos);
        pos *= 2.1;
        amp *= 0.5;
    }
    return val;
}

fn mat_metal(uv: vec2<f32>, wp: vec3<f32>, ep: vec3<f32>, t: f32) -> vec4<f32> {
    let si = sphere_from_uv(uv, ep, wp);
    if !si.valid { return vec4<f32>(0.0); }

    let view_dir = normalize(ep - wp);
    let n_dot_v = max(si.n_dot_v, 0.001);
    let uv_cent = uv - 0.5;

    let angle = 0.8;
    let tangent = normalize(vec3<f32>(cos(angle), sin(angle), 0.0));
    let bitangent = normalize(cross(si.normal, tangent));

    let scratch_uv = vec2<f32>(
        dot(uv_cent * 2.0, vec2<f32>(cos(angle), sin(angle))),
        dot(uv_cent * 2.0, vec2<f32>(-sin(angle), cos(angle)))
    );
    let scratch = fbm_metal(vec2<f32>(scratch_uv.x * 2.0, scratch_uv.y * 40.0)) * 0.1;

    let base_col = vec3<f32>(0.8, 0.75, 0.65);
    let roughness = 0.3;
    let anisotropy = 0.7;

    let lights = array<vec3<f32>, 3>(
        normalize(vec3<f32>(0.5, 0.8, 0.6)),
        normalize(vec3<f32>(-0.7, 0.3, 0.5)),
        normalize(vec3<f32>(0.0, -0.5, 0.8))
    );
    let lcols = array<vec3<f32>, 3>(
        vec3<f32>(1.0, 0.95, 0.9),
        vec3<f32>(0.7, 0.8, 1.0),
        vec3<f32>(0.9, 0.85, 0.8)
    );

    var total_spec = vec3<f32>(0.0);
    var total_diff = vec3<f32>(0.0);

    for (var i = 0; i < 3; i++) {
        let L = lights[i];
        let H = normalize(L + view_dir);
        let ndl = max(dot(si.normal, L), 0.0);
        let ndh = max(dot(si.normal, H), 0.0);

        let alpha_t = roughness * (1.0 + anisotropy);
        let alpha_b = roughness * (1.0 - anisotropy * 0.9);
        let D = pow(ndh, 64.0) * 2.0;

        let f0 = 0.8;
        let fres = f0 + (1.0 - f0) * pow(1.0 - max(dot(H, view_dir), 0.0), 5.0);

        total_spec += lcols[i] * D * fres * ndl;
        total_diff += lcols[i] * ndl * 0.1;
    }

    let refl_dir = reflect(-view_dir, si.normal);
    let metal_env = env_reflect(refl_dir, t) * 1.5;
    let env_r = metal_env * (1.0 - roughness * 0.7);

    // Ambient + diffuse + specular + environment
    let ambient = base_col * 0.15;
    var color = ambient + base_col * total_diff + total_spec * base_col + env_r * base_col;
    color += vec3<f32>(scratch) * base_col;
    // Darken edges slightly for shape, but keep opaque
    color *= (0.7 + 0.3 * smoothstep(0.0, 0.3, n_dot_v));

    return vec4<f32>(color, 1.0);
}

// ════════════════════════════════════════════════════
//  11: Neon — ネオングロー
// ════════════════════════════════════════════════════

fn mat_neon(uv: vec2<f32>, wp: vec3<f32>, ep: vec3<f32>, t: f32) -> vec4<f32> {
    let uv_cent = uv - 0.5;
    let dist_c = length(uv_cent) * 2.0;

    let pulse_base = sin(t * 3.0) * 0.5 + 0.5;
    let pulse = 0.7 + 0.3 * pow(pulse_base, 2.0);
    let pulse2 = 1.0 + 0.1 * sin(t * 8.1 + 1.3);
    let total_pulse = pulse * pulse2;

    let ring_r = 0.6;
    let ring_d = abs(dist_c - ring_r);

    let core = smoothstep(0.02, 0.0, ring_d) * 4.0 * total_pulse;

    let glow1 = exp(-ring_d * ring_d / 0.01) * total_pulse;
    let glow2 = exp(-ring_d * ring_d / 0.04) * 0.4 * total_pulse;
    let glow3 = exp(-ring_d * ring_d / 0.16) * 0.15 * total_pulse;
    let total_glow = glow1 + glow2 + glow3;

    let line_d = abs(uv_cent.x);
    let line_core = smoothstep(0.01, 0.0, line_d) * smoothstep(0.4, 0.2, abs(uv_cent.y));
    let line_glow = exp(-line_d * line_d / 0.006) * smoothstep(0.5, 0.1, abs(uv_cent.y)) * 0.3;
    let line_total = (line_core * 1.5 + line_glow) * total_pulse;

    let glow_col = vec3<f32>(1.0, 0.2, 0.5);
    let core_color = mix(glow_col, vec3<f32>(1.0), min(core / 3.0, 1.0));

    var color = core_color * core + glow_col * total_glow + glow_col * line_total;
    color += vec3<f32>(glow_col.r * 1.1, glow_col.g * 0.9, glow_col.b * 1.2) * glow3 * 0.3 * total_pulse;

    let flicker = hash11(floor(t * 30.0)) * 0.05 + 0.975;
    color *= flicker;

    let lum = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    if lum < 0.001 { return vec4<f32>(0.0); }

    let alpha = clamp(lum, 0.0, 1.0);
    return vec4<f32>(color, alpha);
}

// ════════════════════════════════════════════════════
//  12: Shield — エネルギーシールド
// ════════════════════════════════════════════════════

fn hex_distance(p: vec2<f32>, scale: f32) -> vec2<f32> {
    let s = vec2<f32>(1.0, 1.7320508);
    let ps = p * scale;
    let half_s = s * 0.5;
    let a = ps - s * floor(ps / s + 0.5);
    let b = ps - half_s - s * floor((ps - half_s) / s + 0.5);
    let da = abs(a);
    let db = abs(b);
    let d_a = max(da.x * 1.5 + da.y * s.y, da.y * s.y * 2.0);
    let d_b = max(db.x * 1.5 + db.y * s.y, db.y * s.y * 2.0);
    return vec2<f32>(min(d_a, d_b), min(length(a), length(b)));
}

fn mat_shield(uv: vec2<f32>, wp: vec3<f32>, ep: vec3<f32>, t: f32) -> vec4<f32> {
    let si = sphere_from_uv(uv, ep, wp);
    if !si.valid { return vec4<f32>(0.0); }

    let view_dir = normalize(ep - wp);
    let sphere_uv = vec2<f32>(
        atan2(si.normal.x, si.normal.z) / TAU + 0.5,
        si.normal.y * 0.5 + 0.5
    );

    let hex = hex_distance(sphere_uv, 8.0);
    let edge_glow = 1.0 - smoothstep(0.42, 0.5, hex.x);
    let pulse = sin(t * 2.0 + sphere_uv.y * 12.0) * 0.5 + 0.5;
    let pulse_glow = edge_glow * (0.3 + 0.7 * pulse);

    let fresnel = pow(1.0 - si.n_dot_v, 3.0);

    let cell_id = floor(sphere_uv * 8.0);
    let ch = hash21(cell_id);
    let cell_flicker = smoothstep(0.7, 1.0, sin(t * 3.0 + ch * TAU) * 0.5 + 0.5) * 0.3;

    // Auto hit effect
    let hit_phase = fract(t * 0.3);
    let hit_uv = vec2<f32>(sin(t * 0.7) * 0.2, cos(t * 0.5) * 0.2);
    let hit_dist = length((uv - 0.5) - hit_uv);
    let ripple_r = hit_phase * 0.6;
    let ripple = (1.0 - smoothstep(0.0, 0.08, abs(hit_dist - ripple_r))) * (1.0 - hit_phase) * 1.5;

    let base_col = vec3<f32>(0.2, 0.5, 1.0);
    let energy = pulse_glow + fresnel * 1.5 + ripple + cell_flicker;

    // Subtle environment reflection on hex cells
    let shield_refl_dir = reflect(-view_dir, si.normal);
    let shield_env = env_reflect(shield_refl_dir, t);

    let color = base_col * energy + shield_env * edge_glow * 0.2;
    let alpha = (edge_glow * 0.4 + fresnel * 0.5 + ripple * 0.6 + cell_flicker) * 0.8;

    return vec4<f32>(color, clamp(alpha, 0.0, 1.0));
}

// ════════════════════════════════════════════════════
//  13: Warp — 空間歪み
// ════════════════════════════════════════════════════

fn mat_warp(uv: vec2<f32>, wp: vec3<f32>, ep: vec3<f32>, t: f32) -> vec4<f32> {
    let center = vec2<f32>(0.5, 0.5);
    let to_center = uv - center;
    let dist = length(to_center);
    let angle = atan2(to_center.y, to_center.x);

    let radius = 0.5;
    let radius_mask = 1.0 - smoothstep(radius * 0.5, radius, dist);
    if radius_mask < 0.001 { return vec4<f32>(0.0); }

    let strength = 1.5;
    let twist = strength * (1.0 - dist / radius) * TAU;
    let rotation = t * 0.8;
    let spiral_a = angle + twist + rotation;

    let warped_d = dist + sin(spiral_a * 3.0 + t) * 0.02 * strength;
    let warped_uv = center + vec2<f32>(cos(spiral_a), sin(spiral_a)) * warped_d;

    let swirl = fbm_grad4(warped_uv * 4.0 + vec2<f32>(t * 0.3, t * 0.2)) * strength;

    let depth_rings = sin(dist * 30.0 - t * 4.0) * 0.5 + 0.5;
    let tunnel = depth_rings * (1.0 - dist / radius);

    let aberr = strength * 0.03 * (1.0 - dist / radius);
    let dir = normalize(to_center);
    let spiral_r = sin((length(uv + dir * aberr - center) * 20.0 - t * 3.0) + spiral_a * 2.0) * 0.5 + 0.5;
    let spiral_g = sin((dist * 20.0 - t * 3.0) + spiral_a * 2.0 + 2.094) * 0.5 + 0.5;
    let spiral_b = sin((length(uv - dir * aberr - center) * 20.0 - t * 3.0) + spiral_a * 2.0 + 4.189) * 0.5 + 0.5;

    let chromatic = vec3<f32>(spiral_r, spiral_g, spiral_b);

    let edge_d = abs(dist - radius * 0.7);
    let edge_glow = exp(-edge_d * 15.0) * 0.5;

    let core_dark = smoothstep(0.05, 0.15, dist);
    let core_bright = exp(-dist * 20.0) * 2.0;

    let dist_col = vec3<f32>(0.5, 0.2, 0.9);
    let base = chromatic * dist_col * (tunnel * 0.5 + 0.5);
    let noise_tint = vec3<f32>(swirl * 0.3 + 0.7) * dist_col;
    let final_c = (base * 0.6 + noise_tint * 0.4) * core_dark + vec3<f32>(core_bright) * dist_col;
    let with_edge = final_c + dist_col * edge_glow;

    let alpha = radius_mask * (0.4 + tunnel * 0.3 + edge_glow + core_bright * 0.5);
    return vec4<f32>(with_edge, clamp(alpha, 0.0, 1.0));
}

// ════════════════════════════════════════════════════
//  14: Dissolve — ディゾルブ
// ════════════════════════════════════════════════════

fn warped_fbm(p: vec2<f32>, t: f32) -> f32 {
    let q = vec2<f32>(fbm5(p), fbm5(p + vec2<f32>(5.2, 1.3)));
    let r = vec2<f32>(
        fbm5(p + 4.0 * q + vec2<f32>(1.7 + t * 0.1, 9.2)),
        fbm5(p + 4.0 * q + vec2<f32>(8.3, 2.8 + t * 0.12))
    );
    return fbm5(p + 4.0 * r);
}

fn mat_dissolve(uv: vec2<f32>, wp: vec3<f32>, ep: vec3<f32>, t: f32) -> vec4<f32> {
    // Animate progress cyclically
    let progress = (sin(t * 0.5) * 0.5 + 0.5) * 0.85 + 0.05;

    let noise_uv = uv * 4.0;
    let noise_val = warped_fbm(noise_uv, t);

    let dir = normalize(vec2<f32>(1.0, 0.5));
    let dir_bias = dot(uv - 0.5, dir) * 0.5 + 0.5;
    let biased = noise_val * 0.6 + dir_bias * 0.4;

    let threshold = progress;
    let dist_to_edge = biased - threshold;

    if dist_to_edge < 0.0 { return vec4<f32>(0.0); }

    let edge_width = 0.08;
    let edge_zone = smoothstep(0.0, edge_width, dist_to_edge);
    let edge_intensity = 1.0 - edge_zone;

    let edge_col = vec3<f32>(1.0, 0.4, 0.1);
    let hot_core = vec3<f32>(1.0, 1.0, 0.8);
    let glow_color = mix(hot_core, edge_col, smoothstep(0.0, 0.6, edge_intensity * edge_width * 20.0));
    let glow_brightness = edge_intensity * (2.0 + sin(t * 10.0 + noise_val * 20.0) * 0.3);

    let base_color = vec3<f32>(0.8, 0.85, 0.9);
    let surface_detail = vnoise(uv * 20.0 + t * 0.5) * 0.1;
    let surface = base_color + surface_detail;

    let ember_noise = vnoise(uv * 40.0 + vec2<f32>(t * 2.0, t * 1.5));
    let ember = smoothstep(0.85, 0.95, ember_noise) * edge_intensity * 1.5;

    let final_c = mix(surface, glow_color * glow_brightness, edge_intensity) + edge_col * ember;
    return vec4<f32>(final_c, 1.0);
}

// ════════════════════════════════════════════════════
//  15: Lightning — 電撃
// ════════════════════════════════════════════════════

fn lightning_offset(y: f32, seed: f32, ts: f32) -> f32 {
    var offset = 0.0;
    var amp = 0.2;
    var freq = 2.0;
    for (var i = 0; i < 6; i++) {
        let phase = hash11(seed * 7.13 + f32(i) * 3.7 + ts) * TAU;
        offset += sin(y * freq * PI + phase) * amp;
        amp *= 0.6;
        freq *= 1.9;
    }
    let jitter_seed = floor(y * 16.0);
    let jitter = (hash11(jitter_seed * 13.7 + seed + ts) - 0.5) * 0.06;
    return offset + jitter;
}

fn arc_brightness(p: vec2<f32>, s_y: f32, e_y: f32, seed: f32, ts: f32, thick: f32, xc: f32) -> f32 {
    let y_norm = (p.y - s_y) / (e_y - s_y);
    if y_norm < 0.0 || y_norm > 1.0 { return 0.0; }

    let arc_x = xc + lightning_offset(y_norm, seed, ts);
    let d = abs(p.x - arc_x);

    let core = exp(-d * d / (thick * thick * 0.3)) * 1.5;
    let glow = exp(-d * d / (thick * thick * 4.0)) * 0.6;
    let outer = exp(-d * d / (thick * thick * 16.0)) * 0.2;
    let end_fade = smoothstep(0.0, 0.05, y_norm) * smoothstep(1.0, 0.95, y_norm);

    return (core + glow + outer) * end_fade;
}

fn mat_lightning(uv: vec2<f32>, wp: vec3<f32>, ep: vec3<f32>, t: f32) -> vec4<f32> {
    let tt = t * 2.0;
    let ts = floor(tt * 4.0) * 0.1;
    let thick = 0.04;

    var total = 0.0;

    for (var i = 0; i < 4; i++) {
        let seed = f32(i) * 17.31 + 5.7;
        let xs = 0.3 + hash11(seed + 0.1) * 0.4;
        total += arc_brightness(uv, 0.05, 0.95, seed, ts, thick, xs);

        for (var b = 0; b < 3; b++) {
            let bs = seed + f32(b) * 23.17;
            let bc = hash11(bs + ts * 3.0);
            if bc < 0.4 {
                let by = 0.2 + hash11(bs + 1.0 + ts) * 0.6;
                let bx = xs + lightning_offset(by, seed, ts);
                let bd = (hash11(bs + 2.0 + ts) - 0.5) * 0.3;
                let bl = 0.1 + hash11(bs + 3.0) * 0.2;
                let bp = vec2<f32>(uv.x - bd * (uv.y - by) / bl, uv.y);
                total += arc_brightness(bp, by, min(by + bl, 0.95), bs * 7.3, ts, thick * 0.6, bx) * 0.5;
            }
        }
    }

    if total < 0.01 { return vec4<f32>(0.0); }

    let base_col = vec3<f32>(0.3, 0.5, 1.0);
    let white_mix = smoothstep(0.5, 2.0, total);
    let core_col = mix(base_col, vec3<f32>(1.0), white_mix);
    let final_c = core_col * total * 3.0;

    let flicker = 0.8 + 0.2 * sin(tt * 30.0) * sin(tt * 17.0 + 1.3);
    let alpha = clamp(total * 2.0 * flicker, 0.0, 1.0);
    return vec4<f32>(final_c * flicker, alpha);
}

// ════════════════════════════════════════════════════
//  Fragment shader — dispatch by kind
// ════════════════════════════════════════════════════

// ════════════════════════════════════════════════════
//  Mesh-normal versions — use real geometry normals for lighting
// ════════════════════════════════════════════════════

fn mat_bubble_mesh(uv: vec2<f32>, wp: vec3<f32>, ep: vec3<f32>, t: f32, n: vec3<f32>) -> vec4<f32> {
    let view_dir = normalize(ep - wp);
    let n_dot_v = max(dot(n, view_dir), 0.0);
    let fresnel = pow(1.0 - n_dot_v, 3.0);

    // Front surface interference — position-based thickness
    let p = uv * 2.0 - 1.0;
    let thick_f = 400.0
        + p.y * 300.0
        + sin(p.x * 3.0 + p.y * 2.0 + t * 0.4) * 80.0
        + sin(p.y * 7.0 + t * 0.7) * 40.0
        + sin(p.x * 5.0 - p.y * 4.0 + t * 0.2) * 30.0;
    let pd_f = 2.0 * 1.33 * thick_f;
    let irid_f = vec3<f32>(
        0.5 + 0.5 * cos(6.28318 * pd_f / 700.0),
        0.5 + 0.5 * cos(6.28318 * pd_f / 530.0),
        0.5 + 0.5 * cos(6.28318 * pd_f / 430.0),
    );

    // Back surface
    let thick_b = thick_f + 150.0 + sin(p.x * 4.0 - p.y * 3.0 - t * 0.3) * 50.0;
    let pd_b = 2.0 * 1.33 * thick_b;
    let irid_b = vec3<f32>(
        0.5 + 0.5 * cos(6.28318 * pd_b / 700.0),
        0.5 + 0.5 * cos(6.28318 * pd_b / 530.0),
        0.5 + 0.5 * cos(6.28318 * pd_b / 430.0),
    );

    let range_f = max(max(irid_f.x, irid_f.y), irid_f.z) - min(min(irid_f.x, irid_f.y), irid_f.z);
    let mask_f = smoothstep(0.7, 0.95, range_f) * (0.005 + fresnel * 0.025);
    let range_b = max(max(irid_b.x, irid_b.y), irid_b.z) - min(min(irid_b.x, irid_b.y), irid_b.z);
    let mask_b = smoothstep(0.7, 0.95, range_b) * (0.003 + fresnel * 0.012);

    // Specular
    let light = normalize(vec3<f32>(0.3, 1.0, 0.6));
    let h = normalize(light + view_dir);
    let spec_f = pow(max(dot(n, h), 0.0), 256.0) * 0.7;
    let spec_b = pow(max(dot(-n, h), 0.0), 128.0) * 0.1;

    let sheen = fresnel * vec3<f32>(0.2, 0.25, 0.35) * 0.08;
    let light2 = normalize(vec3<f32>(-0.6, 0.5, 0.3));
    let h2 = normalize(light2 + view_dir);
    let spec2 = pow(max(dot(n, h2), 0.0), 180.0) * 0.3;

    let bands = irid_f * mask_f + irid_b * mask_b;
    let band_lum = max(bands.x, max(bands.y, bands.z));
    let shell_alpha = 0.025;
    let shell_color = vec3<f32>(0.15, 0.18, 0.25) * shell_alpha;

    let emit = shell_color + sheen + bands + vec3<f32>(spec_f + spec_b + spec2);
    let a = shell_alpha + band_lum + spec_f + spec_b + spec2;
    return vec4<f32>(emit, a);
}

fn mat_glass_mesh(uv: vec2<f32>, wp: vec3<f32>, ep: vec3<f32>, t: f32, n: vec3<f32>) -> vec4<f32> {
    let view_dir = normalize(ep - wp);
    let n_dot_v = max(dot(n, view_dir), 0.0);
    let fresnel = pow(1.0 - n_dot_v, 4.0) * 0.8 + 0.1;

    let refract_offset = n.xy * 0.1;
    let refracted_uv = uv + refract_offset;
    let pattern = sin(refracted_uv.x * 20.0 + t * 0.5) * sin(refracted_uv.y * 15.0 + t * 0.3) * 0.5 + 0.5;
    let caustic = pow(max(pattern, 0.0), 3.0) * 0.3;

    let light = normalize(vec3<f32>(0.3, 0.8, 0.5));
    let h = normalize(light + view_dir);
    let spec = pow(max(dot(n, h), 0.0), 512.0) * 1.2;
    let spec2 = pow(max(dot(n, h), 0.0), 64.0) * 0.2;

    let base_color = vec3<f32>(0.12, 0.16, 0.25);
    let reflect_color = vec3<f32>(0.3, 0.4, 0.6);

    let refl_dir = reflect(-view_dir, n);
    let env = env_reflect(refl_dir, t);

    var color = mix(base_color * (0.3 + caustic), env * 0.6, fresnel);
    color += vec3<f32>(spec + spec2);

    // Rim light
    let rim = pow(1.0 - n_dot_v, 6.0);
    color += vec3<f32>(0.4, 0.5, 0.7) * rim * 0.2;

    // Streak
    let streak_pos = fract(t * 0.08) * 3.0 - 1.0;
    let streak = exp(-pow(uv.x * 0.5 + uv.y * 0.5 - streak_pos, 2.0) * 400.0) * 0.3;
    color += vec3<f32>(streak);

    let alpha = 0.2 + fresnel * 0.5 + spec * 0.3;
    return vec4<f32>(color, clamp(alpha, 0.0, 1.0));
}

fn mat_metal_mesh(uv: vec2<f32>, wp: vec3<f32>, ep: vec3<f32>, t: f32, n: vec3<f32>) -> vec4<f32> {
    let view_dir = normalize(ep - wp);
    let n_dot_v = max(dot(n, view_dir), 0.001);

    let base_col = vec3<f32>(0.8, 0.75, 0.65);

    // Anisotropic scratch direction
    let angle = 0.8;
    let scratch_uv = vec2<f32>(
        dot(uv * 2.0 - 1.0, vec2<f32>(cos(angle), sin(angle))),
        dot(uv * 2.0 - 1.0, vec2<f32>(-sin(angle), cos(angle)))
    );
    let scratch = vnoise(vec2<f32>(scratch_uv.x * 2.0, scratch_uv.y * 40.0)) * 0.08;

    // 3-light setup using mesh normal
    let lights = array<vec3<f32>, 3>(
        normalize(vec3<f32>(0.5, 0.8, 0.6)),
        normalize(vec3<f32>(-0.7, 0.3, 0.5)),
        normalize(vec3<f32>(0.0, -0.5, 0.8))
    );
    let lcols = array<vec3<f32>, 3>(
        vec3<f32>(1.0, 0.95, 0.9),
        vec3<f32>(0.7, 0.8, 1.0),
        vec3<f32>(0.9, 0.85, 0.8)
    );

    var total_spec = vec3<f32>(0.0);
    var total_diff = vec3<f32>(0.0);

    for (var i = 0; i < 3; i++) {
        let L = lights[i];
        let H = normalize(L + view_dir);
        let ndl = max(dot(n, L), 0.0);
        let ndh = max(dot(n, H), 0.0);
        let D = pow(ndh, 64.0) * 2.0;
        let f0 = 0.8;
        let fres = f0 + (1.0 - f0) * pow(1.0 - max(dot(H, view_dir), 0.0), 5.0);
        total_spec += lcols[i] * D * fres * ndl;
        total_diff += lcols[i] * ndl * 0.15;
    }

    let refl_dir = reflect(-view_dir, n);
    let env = env_reflect(refl_dir, t) * 1.5;

    let ambient = base_col * 0.12;
    var color = ambient + base_col * total_diff + total_spec * base_col + env * base_col * 0.5;
    color += vec3<f32>(scratch) * base_col;
    color *= (0.7 + 0.3 * smoothstep(0.0, 0.3, n_dot_v));

    return vec4<f32>(color, 1.0);
}

fn mat_crystal_mesh(uv: vec2<f32>, wp: vec3<f32>, ep: vec3<f32>, t: f32, n: vec3<f32>) -> vec4<f32> {
    let view_dir = normalize(ep - wp);
    let n_dot_v = max(dot(n, view_dir), 0.0);
    let fresnel = pow(1.0 - n_dot_v, 3.0) * 0.6 + 0.2;

    // Faceted refraction — perturb normal with noise
    let facet_noise = vnoise(uv * 15.0) * 0.3;
    let perturbed_n = normalize(n + vec3<f32>(facet_noise, facet_noise * 0.7, -facet_noise * 0.5));

    // Internal glow
    let internal = vec3<f32>(0.6, 0.4, 0.8) * (0.2 + 0.1 * sin(t * 1.5 + uv.x * 5.0));

    // Specular sparkles
    let light = normalize(vec3<f32>(0.5, 0.8, 0.3));
    let h = normalize(light + view_dir);
    let spec1 = pow(max(dot(perturbed_n, h), 0.0), 256.0) * 1.5;
    let light2 = normalize(vec3<f32>(-0.3, 0.5, -0.7));
    let h2 = normalize(light2 + view_dir);
    let spec2 = pow(max(dot(perturbed_n, h2), 0.0), 128.0) * 0.6;

    // Rainbow dispersion at edges
    let dispersion = fresnel * vec3<f32>(
        0.3 + 0.3 * sin(n_dot_v * 10.0 + t),
        0.3 + 0.3 * sin(n_dot_v * 10.0 + t + 2.1),
        0.3 + 0.3 * sin(n_dot_v * 10.0 + t + 4.2),
    );

    // Environment reflection
    let refl_dir = reflect(-view_dir, perturbed_n);
    let env = env_reflect(refl_dir, t);

    var color = internal * (1.0 - fresnel) + env * fresnel + dispersion;
    color += vec3<f32>(spec1 + spec2);

    return vec4<f32>(color, 0.8 + fresnel * 0.2);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let wp = in.world_pos;
    let ep = camera.eye_pos;
    let t = camera.time;
    let k = u32(in.kind + 0.5);

    let wn = normalize(in.world_normal);

    var col: vec4<f32>;
    switch k {
        case 0u:  { col = mat_bubble_mesh(uv, wp, ep, t, wn); }
        case 1u:  { col = mat_glass_mesh(uv, wp, ep, t, wn); }
        case 2u:  { col = mat_portal(uv, wp, ep, t); }
        case 3u:  { col = mat_grid(uv, wp, ep, t); }
        case 4u:  { col = mat_water(uv, wp, ep, t); }
        case 5u:  { col = mat_fire(uv, wp, ep, t); }
        case 6u:  { col = mat_smoke(uv, wp, ep, t); }
        case 7u:  { col = mat_aurora(uv, wp, ep, t); }
        case 8u:  { col = mat_hologram(uv, wp, ep, t); }
        case 9u:  { col = mat_crystal_mesh(uv, wp, ep, t, wn); }
        case 10u: { col = mat_metal_mesh(uv, wp, ep, t, wn); }
        case 11u: { col = mat_neon(uv, wp, ep, t); }
        case 12u: { col = mat_shield(uv, wp, ep, t); }
        case 13u: { col = mat_warp(uv, wp, ep, t); }
        case 14u: { col = mat_dissolve(uv, wp, ep, t); }
        case 15u: { col = mat_lightning(uv, wp, ep, t); }
        default:  { col = vec4<f32>(1.0, 0.0, 1.0, 1.0); }
    }

    if col.a < 0.001 {
        discard;
    }
    return col;
}

// ════════════════════════════════════════════════════
//  Background — fullscreen gradient
// ════════════════════════════════════════════════════

struct BgOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

var<private> FULLSCREEN: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
);

@vertex
fn vs_bg(@builtin(vertex_index) vi: u32) -> BgOutput {
    var out: BgOutput;
    let p = FULLSCREEN[vi];
    out.clip_position = vec4<f32>(p, 0.999, 1.0);
    out.uv = p * 0.5 + 0.5;
    return out;
}

@fragment
fn fs_bg(in: BgOutput) -> @location(0) vec4<f32> {
    let t = camera.time;
    let y = in.uv.y;

    // Dark gradient from bottom to top
    let col_bottom = vec3<f32>(0.01, 0.01, 0.03);
    let col_top = vec3<f32>(0.03, 0.04, 0.08);
    var color = mix(col_bottom, col_top, y);

    // Subtle animated noise
    let n = vnoise(in.uv * 50.0 + vec2<f32>(t * 0.05, t * 0.03));
    color += vec3<f32>(n * 0.008);

    // Subtle vignette
    let vig = 1.0 - length(in.uv - 0.5) * 0.8;
    color *= vig;

    return vec4<f32>(color, 1.0);
}
