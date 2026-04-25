// Murakumo Gallery — Unified PBR + Procedural Material Shader
// Evaluates procedural effects INLINE in the fragment shader using
// world-space coordinates and mesh normals. NO UV mapping for procedural effects.
// NO prepass textures. Eliminates UV seam artifacts entirely.

// ════════════════════════════════════════════════════
//  Group 0: Camera
// ════════════════════════════════════════════════════

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    position: vec4<f32>,   // xyz = eye position, w = time
    clip_min: vec4<f32>,
    clip_max: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// ════════════════════════════════════════════════════
//  Group 1: Lights
// ════════════════════════════════════════════════════

struct GpuLight {
    direction_or_position_and_type: vec4<f32>,
    color_and_intensity: vec4<f32>,
    extra: vec4<f32>,
};

struct LightUniform {
    ambient_and_count: vec4<f32>,
    lights: array<GpuLight, 8>,
};

@group(1) @binding(0)
var<uniform> light_data: LightUniform;

// Per-material tunable params (shares group 1 with lights to fit
// inside the default `max_bind_groups: 4` limit).
struct MatParamsUniform {
    values: array<vec4<f32>, 46>,
};

@group(1) @binding(1)
var<uniform> mat_params: MatParamsUniform;

/// Read the i-th parameter (0..8) for material kind k.
fn mp(k: u32, i: u32) -> f32 {
    let v = mat_params.values[k * 2u + (i / 4u)];
    let lane = i % 4u;
    if (lane == 0u) { return v.x; }
    if (lane == 1u) { return v.y; }
    if (lane == 2u) { return v.z; }
    return v.w;
}

// ════════════════════════════════════════════════════
//  Group 2: Texture (dummy white for Seimei compat)
// ════════════════════════════════════════════════════

@group(2) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(2) @binding(1)
var s_diffuse: sampler;

// ════════════════════════════════════════════════════
//  Group 3: Shadow Map
// ════════════════════════════════════════════════════

@group(3) @binding(0)
var shadow_map: texture_depth_2d;
@group(3) @binding(1)
var shadow_sampler: sampler_comparison;
@group(3) @binding(2)
var<uniform> light_view_proj: mat4x4<f32>;

// ════════════════════════════════════════════════════
//  Constants
// ════════════════════════════════════════════════════

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;
const SHADOW_MAP_SIZE: f32 = 2048.0;
const SHADOW_BIAS: f32 = 0.05;

// ════════════════════════════════════════════════════
//  PBR Functions (from Seimei's pbr_shadow.wgsl)
// ════════════════════════════════════════════════════

fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3(0.0), vec3(1.0));
}

fn calculate_shadow(world_pos: vec3<f32>) -> f32 {
    let light_pos = light_view_proj * vec4<f32>(world_pos, 1.0);
    let proj_coords = light_pos.xyz / light_pos.w;
    let uv = proj_coords.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || proj_coords.z > 1.0) {
        return 1.0;
    }
    let depth = proj_coords.z - SHADOW_BIAS;
    var shadow: f32 = 0.0;
    let texel_size = 1.0 / SHADOW_MAP_SIZE;
    for (var x: i32 = -1; x <= 1; x++) {
        for (var y: i32 = -1; y <= 1; y++) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
            shadow += textureSampleCompare(shadow_map, shadow_sampler, uv + offset, depth);
        }
    }
    return shadow / 9.0;
}

fn ior_to_f0(ior: f32) -> f32 {
    let r = (ior - 1.0) / (ior + 1.0);
    return r * r;
}

// ════════════════════════════════════════════════════
//  Vertex / Instance Input (matches Seimei's layout)
// ════════════════════════════════════════════════════

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(9) tangent: vec4<f32>,
    @location(10) vertex_color: vec4<f32>,
};

struct InstanceInput {
    @location(3) model_matrix_0: vec4<f32>,
    @location(4) model_matrix_1: vec4<f32>,
    @location(5) model_matrix_2: vec4<f32>,
    @location(6) model_matrix_3: vec4<f32>,
    @location(7) color: vec4<f32>,
    @location(8) material: vec4<f32>,  // [metallic, roughness, kind, emissive]
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) material: vec4<f32>,
    @location(5) object_center: vec3<f32>,
};

// ════════════════════════════════════════════════════
//  Vertex Shader
// ════════════════════════════════════════════════════

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );

    var out: VertexOutput;
    let world_pos = model_matrix * vec4<f32>(vertex.position, 1.0);
    out.clip_position = camera.view_proj * world_pos;
    out.world_position = world_pos.xyz;

    let normal_matrix = mat3x3<f32>(
        model_matrix[0].xyz,
        model_matrix[1].xyz,
        model_matrix[2].xyz,
    );
    out.world_normal = normalize(normal_matrix * vertex.normal);

    out.uv = vertex.uv;
    out.color = instance.color * vertex.vertex_color;
    out.material = instance.material;
    // Extract object center from model matrix (translation column)
    out.object_center = vec3<f32>(
        model_matrix[3].x,
        model_matrix[3].y,
        model_matrix[3].z,
    );
    return out;
}

// ════════════════════════════════════════════════════
//  Noise Functions
// ════════════════════════════════════════════════════

fn mod289_2(x: vec2<f32>) -> vec2<f32> { return x - floor(x / 289.0) * 289.0; }
fn mod289_3(x: vec3<f32>) -> vec3<f32> { return x - floor(x / 289.0) * 289.0; }
fn permute3(x: vec3<f32>) -> vec3<f32> { return mod289_3((x * 34.0 + 1.0) * x); }
fn mod289_4(x: vec4<f32>) -> vec4<f32> { return x - floor(x / 289.0) * 289.0; }
fn permute4(x: vec4<f32>) -> vec4<f32> { return mod289_4((x * 34.0 + 1.0) * x); }
fn taylorInvSqrt4(r: vec4<f32>) -> vec4<f32> { return 1.79284291400159 - 0.85373472095314 * r; }

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
    var mm = max(0.6 - vec4<f32>(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), vec4<f32>(0.0));
    mm = mm * mm;
    return 42.0 * dot(mm * mm, vec4<f32>(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}

fn simplex3d_fbm(p: vec3<f32>, octaves: i32) -> f32 {
    var val = 0.0; var amp = 0.5; var freq = 1.0; var pos = p;
    for (var i = 0; i < octaves; i++) {
        val += amp * simplex3d(pos * freq);
        freq *= 2.0; amp *= 0.5;
    }
    return val;
}

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

fn vnoise(p: vec2<f32>) -> f32 { return simplex2d(p) * 0.5 + 0.5; }
fn fbm4(p: vec2<f32>) -> f32 { return simplex_fbm(p, 4) * 0.5 + 0.5; }
fn fbm5(p: vec2<f32>) -> f32 { return simplex_fbm(p, 5) * 0.5 + 0.5; }
fn fbm_grad4(p: vec2<f32>) -> f32 { return simplex_fbm(p, 4); }

// ════════════════════════════════════════════════════
//  Environment Reflection
// ════════════════════════════════════════════════════

fn env_reflect(dir: vec3<f32>, t: f32) -> vec3<f32> {
    let up = dir.y * 0.5 + 0.5;
    let sky = mix(vec3<f32>(0.02, 0.03, 0.08), vec3<f32>(0.08, 0.12, 0.25), up);
    let star_uv = vec2<f32>(atan2(dir.x, dir.z) * 2.0, dir.y * 4.0);
    let star = step(0.98, simplex2d(star_uv * 20.0)) * 0.3;
    let ground = smoothstep(-0.1, -0.3, dir.y) * vec3<f32>(0.05, 0.04, 0.06);
    return sky + vec3<f32>(star) + ground;
}

// ════════════════════════════════════════════════════
//  PBR Lighting (applied to material output)
// ════════════════════════════════════════════════════

struct MaterialResult {
    albedo: vec3<f32>,
    emission: vec3<f32>,
    metallic: f32,
    roughness: f32,
    alpha: f32,
    normal: vec3<f32>,     // possibly perturbed normal
    is_emissive_only: bool, // if true, skip PBR lighting (fire, neon, etc.)
};

fn apply_pbr_lighting(
    mat: MaterialResult,
    world_pos: vec3<f32>,
    eye_pos: vec3<f32>,
) -> vec4<f32> {
    if mat.is_emissive_only {
        // Emissive-only materials: just tonemap and return
        let mapped = aces_tonemap(mat.emission);
        let gamma = pow(mapped, vec3(1.0 / 2.2));
        return vec4<f32>(gamma, mat.alpha);
    }

    let n = mat.normal;
    let v = normalize(eye_pos - world_pos);
    let n_dot_v = max(dot(n, v), 0.001);

    let f0_dielectric = vec3(0.04);
    let f0 = mix(f0_dielectric, mat.albedo, mat.metallic);

    let light_count = i32(light_data.ambient_and_count.a);

    // Ambient
    let f_ambient = fresnel_schlick(n_dot_v, f0);
    let kd_ambient = (1.0 - f_ambient) * (1.0 - mat.metallic);
    let ambient_base = light_data.ambient_and_count.rgb;
    let ambient = ambient_base * (kd_ambient * mat.albedo + f_ambient * 0.1);

    let shadow = calculate_shadow(world_pos);

    var lo = vec3(0.0);

    for (var i = 0; i < light_count; i = i + 1) {
        let light = light_data.lights[i];
        let light_type = light.direction_or_position_and_type.w;
        let light_color = light.color_and_intensity.rgb;
        let intensity = light.color_and_intensity.a;

        var l: vec3<f32>;
        var attenuation: f32 = 1.0;

        if (light_type < 0.5) {
            l = normalize(light.direction_or_position_and_type.xyz);
        } else {
            let light_vec = light.direction_or_position_and_type.xyz - world_pos;
            let dist = length(light_vec);
            l = light_vec / max(dist, 0.001);
            attenuation = 1.0 / (1.0 + 0.0001 * dist * dist);
        }

        let h = normalize(v + l);
        let n_dot_l = max(dot(n, l), 0.0);
        let n_dot_h = max(dot(n, h), 0.0);
        let h_dot_v = max(dot(h, v), 0.0);

        let ndf = distribution_ggx(n_dot_h, mat.roughness);
        let g = geometry_smith(n_dot_v, n_dot_l, mat.roughness);
        let f = fresnel_schlick(h_dot_v, f0);

        let numerator = ndf * g * f;
        let denominator = 4.0 * n_dot_v * n_dot_l + 0.0001;
        let specular = numerator / denominator;

        let kd = (1.0 - f) * (1.0 - mat.metallic);
        let diffuse = kd * mat.albedo / PI;

        let radiance = light_color * intensity * attenuation;

        var shadow_factor: f32 = 1.0;
        if (i == 0 && light_type < 0.5) {
            shadow_factor = shadow;
        }
        lo = lo + (diffuse + specular) * radiance * n_dot_l * shadow_factor;
    }

    let color = ambient + lo + mat.emission;
    let mapped = aces_tonemap(color);
    let gamma = pow(mapped, vec3(1.0 / 2.2));
    return vec4<f32>(gamma, mat.alpha);
}

// ════════════════════════════════════════════════════
//  Atmospheric fog
// ════════════════════════════════════════════════════

fn apply_fog(color: vec3<f32>, world_pos: vec3<f32>, eye_pos: vec3<f32>) -> vec3<f32> {
    let dist = length(world_pos - eye_pos);
    let fog_factor = exp(-dist * 0.08);
    let fog_color = vec3<f32>(0.01, 0.012, 0.03);
    return mix(fog_color, color, fog_factor);
}

// ════════════════════════════════════════════════════════════════════════
//  MATERIAL FUNCTIONS
//  All procedural effects use world_pos and world_normal (NOT UV!)
//  For sphere materials: world_normal IS the sphere direction — continuous
//  For flat/cube materials: use world_pos for procedural coordinates
// ════════════════════════════════════════════════════════════════════════

/// Smooth hue-cycle color (h ∈ 0..1). Pastel: cosine-based, S≈0.5.
fn hue_color(h: f32) -> vec3<f32> {
    let phase = h * TAU;
    return 0.5 + 0.5 * vec3<f32>(cos(phase), cos(phase + 2.094), cos(phase + 4.189));
}

/// Fully-saturated hue (HSV with S=V=1). h=0 red, 0.083 orange, 0.16 yellow,
/// 0.33 green, 0.5 cyan, 0.66 blue, 0.83 magenta.
fn hue_sat(h: f32) -> vec3<f32> {
    let p = fract(h) * 6.0;
    let f = fract(p);
    let i = i32(p);
    if i == 0 { return vec3<f32>(1.0, f, 0.0); }
    if i == 1 { return vec3<f32>(1.0 - f, 1.0, 0.0); }
    if i == 2 { return vec3<f32>(0.0, 1.0, f); }
    if i == 3 { return vec3<f32>(0.0, 1.0 - f, 1.0); }
    if i == 4 { return vec3<f32>(f, 0.0, 1.0); }
    return vec3<f32>(1.0, 0.0, 1.0 - f);
}

// ── 0: Bubble — Thin-film iridescence ──

fn mat_bubble(wp: vec3<f32>, n: vec3<f32>, ep: vec3<f32>, t: f32) -> MaterialResult {
    var r: MaterialResult;
    // Tunable params
    let p_irid       = max(mp(0u, 0u), 0.0);
    let p_thick_s    = max(mp(0u, 1u), 0.05);
    let p_edge_glow  = max(mp(0u, 2u), 0.0);
    let p_speed      = max(mp(0u, 3u), 0.0);
    let p_alpha      = clamp(mp(0u, 4u), 0.0, 1.0);
    let ts = t * p_speed;

    let view_dir = normalize(ep - wp);
    let n_dot_v = max(dot(n, view_dir), 0.0);
    let fresnel = pow(1.0 - n_dot_v, 3.0);

    // ── Flowing thickness field ──
    let p = n;
    let gravity = (1.0 - p.y) * 0.5 + 0.5;
    let theta = atan2(p.z, p.x) + ts * 0.05;
    let phi = acos(clamp(p.y, -1.0, 1.0));

    let thick = (200.0 + gravity * 200.0
        + sin(theta * 2.0 + phi * 3.0 + ts * 0.12) * 60.0
        + sin(theta * 5.0 - phi * 2.0 + ts * 0.08) * 30.0
        + sin(phi * 7.0 + ts * 0.15) * 20.0
        + sin(theta * 3.0 + phi * 5.0 - ts * 0.1) * 15.0) * p_thick_s;

    // ── Thin-film interference (view-dependent) ──
    let cos_refract = sqrt(1.0 - (1.0 - n_dot_v * n_dot_v) / (1.33 * 1.33));
    let opd = 2.0 * 1.33 * thick * cos_refract;

    // Vivid rainbow: wider wavelength spread for saturated colors
    let irid = vec3<f32>(
        0.5 + 0.5 * cos(TAU * opd / 700.0),
        0.5 + 0.5 * cos(TAU * opd / 530.0),
        0.5 + 0.5 * cos(TAU * opd / 400.0),
    );

    // Back-surface interference
    let opd2 = 2.0 * 1.33 * (thick * 1.15 + 80.0) * cos_refract;
    let irid2 = vec3<f32>(
        0.5 + 0.5 * cos(TAU * opd2 / 700.0),
        0.5 + 0.5 * cos(TAU * opd2 / 530.0),
        0.5 + 0.5 * cos(TAU * opd2 / 400.0),
    );

    let blend = mix(irid, irid2, 0.3);

    // ── Saturate the rainbow — push colors away from gray ──
    let lum = dot(blend, vec3<f32>(0.333));
    let saturated = mix(vec3(lum), blend, 1.8); // boost saturation
    let vivid = max(saturated, vec3(0.0));

    // Color visible across surface, stronger at edges
    let colored = vivid * (0.15 + fresnel * 0.7) * p_irid;

    // ── Sharp white rim glint ──
    let rim = pow(1.0 - n_dot_v, 8.0);
    let glint = vec3<f32>(1.0) * rim * 0.6 * p_edge_glow;

    r.albedo = vec3<f32>(0.0);
    r.emission = colored + glint;
    r.metallic = 0.0;
    r.roughness = 0.02;
    // Very transparent — thin film, not solid glass
    r.alpha = (0.06 + fresnel * 0.35) * (p_alpha * 2.0);
    r.normal = n;
    r.is_emissive_only = true;
    return r;
}

// ── 1: Glass ──

fn mat_glass(wp: vec3<f32>, n: vec3<f32>, ep: vec3<f32>, t: f32) -> MaterialResult {
    var r: MaterialResult;
    // Tunable params
    let p_ior       = clamp(mp(1u, 0u), 1.0, 2.4);
    let p_hue       = clamp(mp(1u, 1u), 0.0, 1.0);
    let p_tint      = clamp(mp(1u, 2u), 0.0, 1.0);
    let p_caustic   = max(mp(1u, 3u), 0.0);
    let p_alpha     = clamp(mp(1u, 4u), 0.0, 1.0);

    let view_dir = normalize(ep - wp);
    let n_dot_v = max(dot(n, view_dir), 0.0);
    // Fresnel scales with IOR — higher IOR → stronger reflection
    let f0 = pow((p_ior - 1.0) / (p_ior + 1.0), 2.0);
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - n_dot_v, 4.0);

    let refract_offset = n.xy * 0.15;
    let pattern_coord = n.xy * 5.0 + refract_offset;
    let pattern = sin(pattern_coord.x * 20.0 + t * 0.5) * sin(pattern_coord.y * 15.0 + t * 0.3) * 0.5 + 0.5;
    let caustic = pow(pattern, 3.0) * 0.4 * p_caustic;

    let refl_dir = reflect(-view_dir, n);
    let env_col = env_reflect(refl_dir, t);
    let tint_col = hue_color(p_hue);
    let base_color = mix(vec3<f32>(0.15, 0.2, 0.3), tint_col * 0.5, p_tint);

    let streak_coord = n.x * 0.5 + n.y * 0.5;
    let streak_pos = fract(t * 0.08) * 3.0 - 1.0;
    let streak = exp(-pow(streak_coord - streak_pos, 2.0) * 400.0) * 0.3;

    let rim = pow(1.0 - n_dot_v, 6.0);

    r.albedo = base_color;
    r.emission = mix(base_color * (0.3 + caustic), env_col, fresnel)
               + vec3<f32>(streak)
               + tint_col * rim * 0.3;
    r.metallic = 0.1;
    r.roughness = 0.05;
    r.alpha = (0.15 + fresnel * 0.4) * (p_alpha * 1.6);
    r.normal = n;
    r.is_emissive_only = false;
    return r;
}

// ── 2: Portal ──

fn mat_portal(wp: vec3<f32>, n: vec3<f32>, ep: vec3<f32>, t: f32, uv: vec2<f32>) -> MaterialResult {
    var r: MaterialResult;
    // Tunable params
    let p_swirl  = max(mp(2u, 0u), 0.1);
    let p_hue    = clamp(mp(2u, 1u), 0.0, 1.0);
    let p_bright = max(mp(2u, 2u), 0.0);
    let p_speed  = max(mp(2u, 3u), 0.0);
    let p_core   = max(mp(2u, 4u), 0.0);
    let ts = t * p_speed;

    let uv_cent = uv - 0.5;
    let dist = length(uv_cent) * 2.0;

    let angle = atan2(uv_cent.y, uv_cent.x);
    let ring1 = sin(dist * 15.0 * p_swirl - ts * 3.0 + angle * 2.0) * 0.5 + 0.5;
    let ring2 = sin(dist * 10.0 * p_swirl - ts * 2.5 - angle * 3.0) * 0.5 + 0.5;
    let ring3 = sin(dist * 20.0 * p_swirl - ts * 4.0 + angle * 1.0) * 0.5 + 0.5;
    let combined = ring1 * 0.4 + ring2 * 0.35 + ring3 * 0.25;

    let edge = smoothstep(1.0, 0.6, dist);
    let rim = smoothstep(0.5, 1.0, dist) * smoothstep(1.0, 0.85, dist) * 2.0;
    let core = exp(-dist * 3.0) * p_core;

    let col1 = hue_color(p_hue);
    let col2 = hue_color(fract(p_hue + 0.18));
    let col3 = hue_color(fract(p_hue + 0.42));

    var color = mix(col1, col2, ring1) * combined;
    color += col3 * rim * 0.5;
    color += vec3<f32>(0.8, 0.9, 1.0) * core * 0.6;

    let noise_val = vnoise(uv * 5.0 + vec2<f32>(ts * 0.5, ts * 0.3));
    color += col2 * noise_val * 0.2 * edge;
    color *= p_bright;

    let alpha = edge * (combined * 0.6 + rim * 0.3 + core * 0.4);

    r.albedo = vec3(0.0);
    r.emission = color;
    r.metallic = 0.0;
    r.roughness = 0.3;
    r.alpha = clamp(alpha, 0.0, 1.0);
    r.normal = n;
    r.is_emissive_only = true;
    return r;
}

// ── 3: Grid ──

fn mat_grid(wp: vec3<f32>, n: vec3<f32>, ep: vec3<f32>, t: f32, uv: vec2<f32>) -> MaterialResult {
    var r: MaterialResult;
    // Tunable params
    let p_density = max(mp(3u, 0u), 1.0);
    let p_lw      = max(mp(3u, 1u), 0.001);
    let p_glow    = max(mp(3u, 2u), 0.0);
    let p_hue     = clamp(mp(3u, 3u), 0.0, 1.0);
    let p_pulse_s = max(mp(3u, 4u), 0.0);

    let p = uv * p_density;
    let grid_x = abs(fract(p.x) - 0.5);
    let grid_y = abs(fract(p.y) - 0.5);
    let line_x = smoothstep(p_lw, 0.0, grid_x);
    let line_y = smoothstep(p_lw, 0.0, grid_y);
    let grid_val = max(line_x, line_y);

    let sp = uv * p_density * 4.0;
    let sg_x = smoothstep(0.01, 0.0, abs(fract(sp.x) - 0.5));
    let sg_y = smoothstep(0.01, 0.0, abs(fract(sp.y) - 0.5));
    let subgrid = max(sg_x, sg_y) * 0.15;

    let pulse_dist = length(uv - 0.5);
    let pulse = sin(pulse_dist * 20.0 - t * 3.0 * p_pulse_s) * 0.5 + 0.5;
    let pulse_ring = smoothstep(0.05, 0.0, abs(fract(pulse_dist * 3.0 - t * 0.5 * p_pulse_s) - 0.5)) * 0.4;

    let dot_x = smoothstep(p_lw * 4.0, 0.0, grid_x);
    let dot_y = smoothstep(p_lw * 4.0, 0.0, grid_y);
    let dots = dot_x * dot_y * 0.8;

    let bright_col = hue_color(p_hue);
    let base_col = bright_col * 0.25;
    var color = base_col * grid_val + bright_col * (dots + pulse_ring) * (1.0 + p_glow);
    color += base_col * 0.4 * subgrid;
    color *= (0.8 + pulse * 0.2);

    let edge = smoothstep(0.0, 0.05, uv.x) * smoothstep(1.0, 0.95, uv.x)
             * smoothstep(0.0, 0.05, uv.y) * smoothstep(1.0, 0.95, uv.y);
    let lum = max(color.x, max(color.y, color.z));

    r.albedo = vec3(0.0);
    r.emission = color;
    r.metallic = 0.3;
    r.roughness = 0.5;
    r.alpha = clamp(lum * 2.0 + 0.15, 0.0, 1.0) * edge;
    r.normal = n;
    r.is_emissive_only = true;
    return r;
}

// ── 4: Water ──

fn wave_height(p: vec2<f32>, t: f32) -> f32 {
    var height = 0.0; var amp = 0.5; var freq = 3.0; var pos = p;
    for (var i = 0; i < 4; i++) {
        let w1 = sin(pos.x * freq + t * 1.3) * amp;
        let w2 = sin(pos.y * freq * 0.8 + t * 0.9 + 2.1) * amp * 0.7;
        let w3 = sin((pos.x + pos.y) * freq * 0.6 + t * 1.1 + 4.3) * amp * 0.5;
        height += w1 + w2 + w3;
        amp *= 0.5; freq *= 2.0;
        let c = cos(0.5); let s = sin(0.5);
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
    return pow(max((g1 + g2 + g3) / 3.0, 0.0), 2.0);
}

fn mat_water(wp: vec3<f32>, n: vec3<f32>, ep: vec3<f32>, t: f32) -> MaterialResult {
    var r: MaterialResult;
    let view_dir = normalize(ep - wp);
    let tt = t * 0.5;
    let is_flat = abs(n.y) > 0.9;

    // ── Wave coordinates ──
    var wave_p: vec2<f32>;
    if is_flat { wave_p = wp.xz * 0.6; } else { wave_p = n.xz * 2.0; }

    // ── Subtle wave normals ──
    let eps = 0.01;
    let h_c = wave_height(wave_p, tt) * 0.25;
    let h_r = wave_height(wave_p + vec2(eps, 0.0), tt) * 0.25;
    let h_u = wave_height(wave_p + vec2(0.0, eps), tt) * 0.25;
    let wave_n = normalize(vec3<f32>((h_c - h_r) / eps * 0.12, 1.0, (h_c - h_u) / eps * 0.12));

    var perturbed_n: vec3<f32>;
    if is_flat { perturbed_n = wave_n; }
    else {
        let tng = normalize(cross(vec3(0.0, 1.0, 0.0), n));
        perturbed_n = normalize(tng * wave_n.x + n * wave_n.y + cross(n, tng) * wave_n.z);
    }

    let n_dot_v = max(dot(perturbed_n, view_dir), 0.0);
    let fresnel = pow(1.0 - n_dot_v, 5.0) * 0.65 + 0.04;

    // ── Depth ──
    var depth: f32;
    if is_flat {
        let pc = vec2<f32>(0.4, 0.2);
        let et = clamp(length(wp.xz - pc) / 1.4, 0.0, 1.0);
        depth = (1.0 - et) * (1.0 - et);
    } else { depth = 0.5; }

    // ── Bottom visible through water (eval rock beneath) ──
    let refr_off = perturbed_n.xz * 0.04 * depth;
    var bottom_coord: vec3<f32>;
    if is_flat {
        bottom_coord = normalize(vec3((wp.x + refr_off.x) * 0.3, (wp.z + refr_off.y) * 0.3, 0.5));
    } else { bottom_coord = n; }
    let bottom = eval_rock(bottom_coord, 3.0);

    // Beer's law absorption: red absorbed first, blue least
    let absorption = exp(-depth * 3.0 * vec3<f32>(2.5, 0.8, 0.4));
    let bottom_seen = bottom.albedo * absorption;

    // Subtle caustics on the bottom
    let c_uv = wave_p * 2.0 + perturbed_n.xz * 0.06;
    let caustic = water_caustics(c_uv, tt) * 0.3 * (1.0 - depth);
    let bottom_lit = bottom_seen + vec3(caustic * 0.08, caustic * 0.15, caustic * 0.2);

    // ── Sky reflection ──
    let refl_dir = reflect(-view_dir, perturbed_n);
    let sky_up = smoothstep(-0.1, 0.4, refl_dir.y);
    let sky = mix(vec3<f32>(0.06, 0.1, 0.18), vec3<f32>(0.18, 0.3, 0.5), sky_up);

    // ── Sun glint (sharp) ──
    let light_dir = normalize(vec3<f32>(0.5, 0.8, 0.6));
    let half_v = normalize(light_dir + view_dir);
    let spec = pow(max(dot(perturbed_n, half_v), 0.0), 512.0);
    let glint = vec3<f32>(1.0, 0.95, 0.88) * spec * 1.2;

    // ── Compose: fresnel blends bottom vs sky ──
    var color = mix(bottom_lit, sky, fresnel) + glint;

    // Deep center gets a blue-green tint (volume of water)
    let water_vol_tint = vec3<f32>(0.02, 0.06, 0.1) * depth;
    color += water_vol_tint;

    // ── Edge alpha ──
    var alpha = 0.92;
    if is_flat {
        let pc = vec2<f32>(0.4, 0.2);
        let ed = length(wp.xz - pc);
        alpha = smoothstep(1.4, 1.2, ed); // fade at pond edge
        alpha *= smoothstep(0.0, 0.1, depth); // fade at very shallow shore
    }

    r.albedo = color;
    r.emission = glint;
    r.metallic = 0.0;
    r.roughness = 0.02;
    r.alpha = alpha;
    r.normal = perturbed_n;
    r.is_emissive_only = false;
    return r;
}

// ── 5: Fire ──

// Fire density field (3D volume)
fn fire_density(p: vec3<f32>, t: f32) -> f32 {
    // Fire rises: scroll Y upward
    let scroll = vec3<f32>(p.x, p.y - t * 1.8, p.z);

    // Strong turbulence warp — break up the shape
    let warp = vec3<f32>(
        simplex3d(scroll * 2.5 + vec3<f32>(t * 0.4, 0.0, 3.7)),
        simplex3d(scroll * 2.0 + vec3<f32>(0.0, t * 0.25, 7.1)),
        simplex3d(scroll * 2.5 + vec3<f32>(5.2, t * 0.35, 0.0))
    );
    let warped = scroll + warp * 0.5;

    // FBM noise (4 octaves for detail)
    let n1 = simplex3d(warped * 2.0) * 0.45;
    let n2 = simplex3d(warped * 4.5 + vec3<f32>(5.2, 1.3, 2.8)) * 0.25;
    let n3 = simplex3d(warped * 9.0 + vec3<f32>(1.7, 9.2, 4.1)) * 0.15;
    let n4 = simplex3d(warped * 18.0 + vec3<f32>(3.1, 6.4, 8.7)) * 0.08;
    let noise = n1 + n2 + n3 + n4;

    // Shape: organic flame — noise distorts the boundary
    let radial = length(vec2<f32>(p.x, p.z));
    let height = (p.y + 1.0) * 0.5;

    // Boundary noise: distort the cone edge so it's ragged
    let edge_noise = simplex3d(vec3<f32>(p.x * 4.0, p.y * 2.0 - t * 3.0, p.z * 4.0)) * 0.2
                   + simplex3d(vec3<f32>(p.x * 8.0, p.y * 4.0 - t * 5.0, p.z * 8.0)) * 0.1;
    let cone_radius = mix(0.55, 0.05, height * height) + edge_noise;

    let shape = smoothstep(cone_radius, cone_radius * 0.15, radial)
              * smoothstep(-0.05, 0.15, height)
              * smoothstep(1.05, 0.5, height);

    return max((noise * 0.5 + 0.5) * shape - 0.08, 0.0) * 3.0;
}

fn mat_fire(wp: vec3<f32>, n: vec3<f32>, ep: vec3<f32>, t: f32, obj_center: vec3<f32>) -> MaterialResult {
    var r: MaterialResult;
    // Tunable params
    let p_height_s   = max(mp(5u, 0u), 0.1);
    let p_density_s  = max(mp(5u, 1u), 0.0);
    let p_speed      = max(mp(5u, 2u), 0.0);
    let p_heat       = max(mp(5u, 3u), 0.0);
    let p_blue_core  = clamp(mp(5u, 4u), 0.0, 1.0);
    let p_brightness = max(mp(5u, 5u), 0.0);
    let p_hue        = mp(5u, 6u);
    let p_white_hot  = clamp(mp(5u, 7u), 0.0, 1.0);
    let ts = t * p_speed;

    // Hue-driven fire palette: dark glow → main hue → near-white at peak
    let main_col = hue_sat(p_hue);
    let bright_col = mix(main_col, vec3<f32>(1.0), p_white_hot);
    let warm_col   = main_col * 0.7;
    let glow_col   = main_col * 0.25;

    let view_dir = normalize(wp - ep);
    let sphere_radius = 1.0;
    let oc = ep - obj_center;
    let b = dot(oc, view_dir);
    let c_val = dot(oc, oc) - sphere_radius * sphere_radius;
    let disc = b * b - c_val;

    if disc < 0.0 {
        r.albedo = vec3(0.0); r.emission = vec3(0.0); r.alpha = 0.0;
        r.metallic = 0.0; r.roughness = 0.8; r.normal = n; r.is_emissive_only = true;
        return r;
    }

    let sqrt_disc = sqrt(disc);
    let t_near = max(-b - sqrt_disc, 0.0);
    let t_far = -b + sqrt_disc;
    let march_dist = t_far - t_near;

    let steps = 40;
    let step_size = march_dist / f32(steps);
    var transmittance = 1.0;
    var accum_color = vec3<f32>(0.0);

    for (var i = 0; i < 40; i++) {
        let ray_t = t_near + (f32(i) + 0.5) * step_size;
        let sample_world = ep + view_dir * ray_t;
        // Compress/extend the volume vertically based on height param
        var local_pos = sample_world - obj_center;
        local_pos = vec3<f32>(local_pos.x, local_pos.y / max(p_height_s, 0.1), local_pos.z);

        let d = fire_density(local_pos, ts) * p_density_s;
        if d > 0.001 {
            let height = (local_pos.y + 1.0) * 0.5;

            let temp = d * (1.0 - height * 0.4) * p_heat;
            var fire_col: vec3<f32>;
            if temp > 0.8 {
                fire_col = mix(main_col, bright_col, clamp((temp - 0.8) / 0.4, 0.0, 1.0));
            } else if temp > 0.5 {
                fire_col = mix(warm_col, main_col, (temp - 0.5) / 0.3);
            } else if temp > 0.2 {
                fire_col = mix(glow_col, warm_col, (temp - 0.2) / 0.3);
            } else {
                fire_col = mix(vec3<f32>(0.0), glow_col, temp / 0.2);
            }

            // Blue core: contrasting hot center near base (kept as a separate look,
            // independent of hue — set Blue Core = 0 for a pure single-color flame).
            let blue_zone = smoothstep(0.3, 0.0, height) * smoothstep(0.3, 0.9, d);
            fire_col = mix(fire_col, vec3<f32>(0.3, 0.5, 1.5), blue_zone * p_blue_core);

            let extinct = d * 8.0 * step_size;
            accum_color += fire_col * d * step_size * transmittance * 5.0 * p_brightness;
            transmittance *= exp(-extinct);
        }
        if transmittance < 0.02 { break; }
    }

    let flicker = 0.88 + 0.12 * sin(ts * 15.0) * sin(ts * 9.3 + 1.7);
    accum_color *= flicker;

    r.albedo = vec3(0.0);
    r.emission = accum_color;
    r.metallic = 0.0;
    r.roughness = 0.8;
    r.alpha = clamp((1.0 - transmittance) * 1.2, 0.0, 1.0);
    r.normal = n;
    r.is_emissive_only = true;
    return r;
}

// ── 6: Smoke ──

fn smoke_density(p: vec3<f32>, t: f32) -> f32 {
    let warp = vec3<f32>(
        simplex3d(p * 1.5 + vec3<f32>(t * 0.1, 0.0, 3.7)),
        simplex3d(p * 1.5 + vec3<f32>(0.0, t * 0.08, 7.3)),
        simplex3d(p * 1.5 + vec3<f32>(5.2, t * 0.12, 0.0))
    );
    let warped = p + warp * 0.4;
    let scroll = vec3<f32>(warped.x, warped.y - t * 0.6, warped.z);
    let nn = simplex3d_fbm(scroll * 2.5, 4) * 0.5 + 0.5;
    let radial = length(vec2<f32>(p.x, p.z));
    let shape = smoothstep(0.8, 0.2, radial) * smoothstep(0.0, 0.15, p.y) * smoothstep(1.0, 0.7, p.y);
    return max(nn * shape - 0.1, 0.0) * 1.5;
}

fn mat_smoke(wp: vec3<f32>, n: vec3<f32>, ep: vec3<f32>, t: f32, obj_center: vec3<f32>) -> MaterialResult {
    var r: MaterialResult;
    // Tunable params
    let p_density_s  = max(mp(6u, 0u), 0.0);
    let p_speed      = max(mp(6u, 1u), 0.0);
    let p_brightness = max(mp(6u, 2u), 0.0);
    let p_detail     = max(mp(6u, 3u), 0.1);
    let tt = t * 0.5 * p_speed;

    // Ray-sphere intersection in world space for volumetric march
    let view_dir = normalize(wp - ep);
    let sphere_radius = 1.0;

    // Ray from camera through this fragment, intersect bounding sphere
    let oc = ep - obj_center;
    let b = dot(oc, view_dir);
    let c_val = dot(oc, oc) - sphere_radius * sphere_radius;
    let disc = b * b - c_val;

    if disc < 0.0 {
        r.albedo = vec3(0.0); r.emission = vec3(0.0); r.alpha = 0.0;
        r.metallic = 0.0; r.roughness = 0.9; r.normal = n; r.is_emissive_only = true;
        return r;
    }

    let sqrt_disc = sqrt(disc);
    let t_near = max(-b - sqrt_disc, 0.0);
    let t_far = -b + sqrt_disc;
    let march_dist = t_far - t_near;

    let steps = 32;
    let step_size = march_dist / f32(steps);
    var transmittance = 1.0;
    var accum_color = vec3<f32>(0.0);
    let absorption = 5.0;
    let light_dir = normalize(vec3<f32>(0.3, 1.0, 0.5));
    let base_col = vec3<f32>(0.12, 0.13, 0.18);
    let scatter_col = vec3<f32>(0.22, 0.25, 0.35);
    let highlight_col = vec3<f32>(0.45, 0.48, 0.6);

    for (var i = 0; i < 32; i++) {
        let ray_t = t_near + (f32(i) + 0.5) * step_size;
        let sample_world = ep + view_dir * ray_t;
        // Convert to local space (centered, unit sphere)
        let local_pos = sample_world - obj_center;

        let d = smoke_density(local_pos * p_detail, tt) * p_density_s;
        if d > 0.001 {
            let extinct = d * absorption * step_size;
            let tr_step = exp(-extinct);

            let ls1 = smoke_density((local_pos + light_dir * 0.12) * p_detail, tt) * p_density_s;
            let ls2 = smoke_density((local_pos + light_dir * 0.25) * p_detail, tt) * p_density_s;
            let light_atten = exp(-(ls1 + ls2 * 0.5) * 2.5);

            let lit_color = mix(base_col, mix(scatter_col, highlight_col, light_atten), light_atten * 0.8);
            accum_color += lit_color * d * step_size * transmittance * 4.0 * p_brightness;
            transmittance *= tr_step;
        }
        if transmittance < 0.02 { break; }
    }

    let alpha = (1.0 - transmittance) * 0.95;

    r.albedo = vec3(0.0);
    r.emission = accum_color;
    r.metallic = 0.0;
    r.roughness = 0.9;
    r.alpha = clamp(alpha, 0.0, 1.0);
    r.normal = n;
    r.is_emissive_only = true;
    return r;
}

// ── 7: Aurora ──

fn aurora_n1d(p: f32) -> f32 {
    let i = floor(p); let f = fract(p); let u = f * f * (3.0 - 2.0 * f);
    return mix(hash11(i), hash11(i + 1.0), u);
}

fn aurora_band(x: f32, t: f32, freq: f32, phase: f32, speed: f32, amp: f32) -> f32 {
    var wave = sin(x * freq + t * speed + phase) * amp;
    wave += sin(x * freq * 1.7 + t * speed * 0.6 + phase * 2.3) * amp * 0.5;
    wave += sin(x * freq * 0.4 + t * speed * 1.3 + phase * 0.7) * amp * 0.3;
    wave += aurora_n1d(x * 2.0 + t * speed * 0.3) * amp * 0.4;
    return wave;
}

fn mat_aurora(wp: vec3<f32>, n: vec3<f32>, ep: vec3<f32>, t: f32, uv: vec2<f32>) -> MaterialResult {
    var r: MaterialResult;
    // Tunable params
    let p_speed     = max(mp(7u, 0u), 0.0);
    let p_intensity = max(mp(7u, 1u), 0.0);
    let p_hue       = clamp(mp(7u, 2u), 0.0, 1.0);
    let p_bands     = max(mp(7u, 3u), 0.2);
    let tt = t * 0.5 * p_speed;
    let x = uv.x; let y = uv.y;
    let vert_mask = smoothstep(0.0, 0.15, y) * smoothstep(1.0, 0.7, y);

    var intensity = 0.0; var cmix = 0.0;
    let b1c = 0.5 + aurora_band(x, tt, 3.0, 0.0, 1.0, 0.18);
    let b1d = abs(y - b1c); let b1w = 0.18 + aurora_n1d(x * 3.0 + tt * 0.4) * 0.10;
    let b1 = smoothstep(b1w, 0.0, b1d); intensity += b1 * 0.8; cmix += b1 * 0.3;
    let b2c = 0.45 + aurora_band(x, tt, 4.5, 2.1, 0.7, 0.15);
    let b2d = abs(y - b2c); let b2w = 0.14 + aurora_n1d(x * 4.0 + tt * 0.6 + 3.0) * 0.08;
    let b2 = smoothstep(b2w, 0.0, b2d); intensity += b2 * 0.6; cmix += b2 * 0.7;
    let b3c = 0.55 + aurora_band(x, tt, 6.0, 4.5, 1.3, 0.12);
    let b3d = abs(y - b3c); let b3w = 0.08 + aurora_n1d(x * 5.0 + tt * 0.8 + 7.0) * 0.05;
    let b3 = smoothstep(b3w, 0.0, b3d); intensity += b3 * 0.9; cmix += b3 * 0.5;
    let b4c = 0.48 + aurora_band(x, tt, 2.0, 1.3, 0.5, 0.20);
    let b4d = abs(y - b4c); let b4w = 0.25 + aurora_n1d(x * 2.0 + tt * 0.2 + 5.0) * 0.12;
    let b4 = smoothstep(b4w, 0.0, b4d); intensity += b4 * 0.4; cmix += b4 * 0.9;
    let b5c = 0.42 + aurora_band(x, tt, 2.5, 3.7, 0.8, 0.16);
    let b5d = abs(y - b5c); let b5w = 0.20 + aurora_n1d(x * 2.5 + tt * 0.3 + 9.0) * 0.08;
    let b5 = smoothstep(b5w, 0.0, b5d); intensity += b5 * 0.3; cmix += b5 * 0.6;

    cmix = clamp(cmix / max(intensity, 0.001), 0.0, 1.0);
    let shimmer = vnoise(vec2<f32>(x * 20.0 + tt * 2.0, y * 30.0 + tt * 0.5));
    intensity *= 0.85 + shimmer * 0.15;
    let ray = 0.7 + 0.3 * sin(x * 40.0 + tt * 0.3) * sin(x * 17.0 - tt * 0.7);
    intensity *= ray;

    let col1 = hue_color(p_hue);
    let col2 = hue_color(fract(p_hue + 0.33));
    let col3 = hue_color(fract(p_hue + 0.66));
    let pos_shift = sin(x * 5.0 * p_bands + tt * 0.3) * 0.3 + 0.5;
    let final_t = mix(cmix, pos_shift, 0.4);
    var aurora_col = mix(col1, col2, final_t);
    aurora_col = mix(aurora_col, col3, smoothstep(0.6, 1.0, intensity) * 0.25);
    aurora_col += vec3<f32>(pow(max(intensity, 0.0), 0.5) * 0.35);
    aurora_col *= 2.0 * p_intensity;

    let glow = pow(max(intensity, 0.0), 0.6);
    let h_fade = smoothstep(0.0, 0.05, x) * smoothstep(1.0, 0.95, x);
    let alpha = clamp(glow * vert_mask * h_fade, 0.0, 1.0);

    r.albedo = vec3(0.0);
    r.emission = aurora_col * glow;
    r.metallic = 0.0;
    r.roughness = 0.4;
    r.alpha = alpha;
    r.normal = n;
    r.is_emissive_only = true;
    return r;
}

// ── 8: Hologram ──

fn mat_hologram(wp: vec3<f32>, n: vec3<f32>, ep: vec3<f32>, t: f32, uv: vec2<f32>) -> MaterialResult {
    var r: MaterialResult;
    // Tunable params
    let p_lines      = max(mp(8u, 0u), 1.0);
    let p_speed      = max(mp(8u, 1u), 0.0);
    let p_hue        = clamp(mp(8u, 2u), 0.0, 1.0);
    let p_glitch_th  = clamp(mp(8u, 3u), 0.0, 1.0);
    let p_brightness = max(mp(8u, 4u), 0.0);
    let ts = t * p_speed;

    var uuv = uv;
    let glitch_block = floor(uuv.y * 20.0 + ts * 3.0);
    let glitch_rand = hash11(glitch_block + floor(ts * 7.0));
    let glitch_active = step(p_glitch_th, glitch_rand);
    let glitch_offset = (hash11(glitch_block * 3.7 + ts) - 0.5) * 0.15 * 0.5 * glitch_active;
    uuv.x = fract(uuv.x + glitch_offset);

    let scan_y = uuv.y * (p_lines * 0.5) + ts * 3.0;
    let scan_line = sin(scan_y * PI) * 0.5 + 0.5;
    let scan_mask = smoothstep(0.3, 0.7, scan_line);

    let bar_pos = fract(ts * 0.3);
    let bar_dist = abs(uuv.y - bar_pos);
    let bar_wrap = min(bar_dist, 1.0 - bar_dist);
    let scan_bar = smoothstep(0.05, 0.0, bar_wrap) * 0.5;

    let base = hue_color(p_hue);
    var color = base;
    color *= (0.6 + 0.4 * scan_mask);
    color += vec3<f32>(scan_bar) * base;

    let uv_cent = uuv - 0.5;
    let edge_v = pow(length(uv_cent) * 2.0, 2.0) * 0.5;
    color += base * edge_v;

    let flicker = 1.0 - 0.15 * (sin(ts * 5.0) * sin(ts * 13.5) + 1.0) * 0.5;
    let flash = 1.0 + step(0.97, hash11(floor(ts * 4.0))) * 2.0;
    color *= flicker * flash * p_brightness;
    color += vec3<f32>(sin(uuv.y * (p_lines * 5.0) + ts * 10.0) * 0.03) * base;

    if glitch_active > 0.5 {
        let corrupt = hash21(vec2<f32>(uuv.x * 10.0, glitch_block));
        color = mix(color, vec3<f32>(corrupt, 1.0 - corrupt, corrupt * 0.5), 0.3);
    }
    color = clamp(color, vec3<f32>(0.0), vec3<f32>(5.0));
    let alpha = 0.7 * (0.5 + 0.5 * scan_mask) * flicker;

    r.albedo = vec3(0.0);
    r.emission = color;
    r.metallic = 0.5;
    r.roughness = 0.2;
    r.alpha = alpha;
    r.normal = n;
    r.is_emissive_only = true;
    return r;
}

// ── 9: Crystal ──

fn voronoi_crystal(p_coord: vec2<f32>, scale: f32, t: f32) -> vec3<f32> {
    let p = p_coord * scale; let ip = floor(p); let fp = fract(p);
    var min_d = 1.0; var sec_d = 1.0; var cell_id = vec2<f32>(0.0);
    for (var y = -1; y <= 1; y++) { for (var x = -1; x <= 1; x++) {
        let nb = vec2<f32>(f32(x), f32(y));
        let pt = hash22(ip + nb);
        let anim = nb + 0.5 + 0.3 * sin(t * 0.5 + TAU * pt);
        let diff = anim - fp; let d = length(diff);
        if d < min_d { sec_d = min_d; min_d = d; cell_id = ip + nb; } else if d < sec_d { sec_d = d; }
    } }
    return vec3<f32>(min_d, sec_d - min_d, hash21(cell_id));
}

fn mat_crystal(wp: vec3<f32>, n: vec3<f32>, ep: vec3<f32>, t: f32) -> MaterialResult {
    var r: MaterialResult;
    // Tunable params
    let p_facet  = max(mp(9u, 0u), 0.1);
    let p_sharp  = clamp(mp(9u, 1u), 0.0, 1.0);
    let p_hue    = clamp(mp(9u, 2u), 0.0, 1.0);
    let p_bright = max(mp(9u, 3u), 0.0);
    let p_disp   = max(mp(9u, 4u), 0.0);

    let view_dir = normalize(ep - wp);
    let n_dot_v = max(dot(n, view_dir), 0.0);
    let fresnel = pow(1.0 - n_dot_v, 3.0) * 0.6 + 0.2;

    let crystal_coords = n.xy * p_facet;
    let vor = voronoi_crystal(crystal_coords, 2.0, t);
    let edge_dist = vor.y;
    let cell_rand = vor.z;

    let facet_angle = cell_rand * TAU;
    let perturbed_n = normalize(n + vec3<f32>(cos(facet_angle) * 0.12 * p_sharp, sin(facet_angle) * 0.12 * p_sharp, 0.0));

    let base = hue_color(p_hue);
    var crystal_col = base * (0.85 + 0.3 * cell_rand);
    crystal_col += vec3<f32>(cell_rand * 0.15, -cell_rand * 0.075, cell_rand * 0.045);

    let internal = base * (0.2 + 0.1 * sin(t * 1.5 + cell_rand * TAU)) * (1.0 - fresnel);
    let dispersion = fresnel * vec3<f32>(
        0.3 + 0.3 * sin(n_dot_v * 10.0 + t),
        0.3 + 0.3 * sin(n_dot_v * 10.0 + t + 2.1),
        0.3 + 0.3 * sin(n_dot_v * 10.0 + t + 4.2),
    ) * smoothstep(0.15, 0.0, edge_dist) * 0.6 * p_disp;

    let refl_dir = reflect(-view_dir, perturbed_n);
    let env = env_reflect(refl_dir, t);

    r.albedo = crystal_col * 0.5;
    r.emission = (internal + dispersion + env * fresnel * 0.5) * p_bright;
    r.metallic = 0.2;
    r.roughness = 0.1;
    r.alpha = 0.9 * (0.8 + 0.2 * n_dot_v);
    r.normal = perturbed_n;
    r.is_emissive_only = false;
    return r;
}

// ── 10: Metal ──

fn mat_metal(wp: vec3<f32>, n: vec3<f32>, ep: vec3<f32>, t: f32) -> MaterialResult {
    var r: MaterialResult;
    // Tunable params
    let p_smooth  = clamp(mp(10u, 0u), 0.0, 1.0);
    let p_hue     = clamp(mp(10u, 1u), 0.0, 1.0);
    let p_sat     = max(mp(10u, 2u), 0.0);
    let p_scratch = max(mp(10u, 3u), 0.0);

    let view_dir = normalize(ep - wp);
    let n_dot_v = max(dot(n, view_dir), 0.001);

    let tinted = hue_color(p_hue);
    let neutral = vec3<f32>(0.8, 0.75, 0.65);
    let base_col = mix(neutral, tinted, p_sat);

    let angle_val = 0.8;
    let scratch_coord = vec2<f32>(
        dot(n.xy * 2.0, vec2<f32>(cos(angle_val), sin(angle_val))),
        dot(n.xy * 2.0, vec2<f32>(-sin(angle_val), cos(angle_val)))
    );
    let scratch = vnoise(vec2<f32>(scratch_coord.x * 2.0, scratch_coord.y * 40.0)) * 0.08 * p_scratch;

    let refl_dir = reflect(-view_dir, n);
    let env = env_reflect(refl_dir, t) * 1.5;

    r.albedo = base_col + vec3<f32>(scratch);
    r.emission = env * base_col * 0.5;
    r.metallic = 1.0;
    // Smoothness 1.0 → roughness 0.05; smoothness 0 → roughness 0.6
    r.roughness = mix(0.6, 0.05, p_smooth);
    r.alpha = 1.0;
    r.normal = n;
    r.is_emissive_only = false;
    return r;
}

// ── 11: Neon ──

fn mat_neon(wp: vec3<f32>, n: vec3<f32>, ep: vec3<f32>, t: f32, uv: vec2<f32>) -> MaterialResult {
    var r: MaterialResult;
    // Tunable params
    let p_glow   = max(mp(11u, 0u), 0.0);
    let p_hue    = clamp(mp(11u, 1u), 0.0, 1.0);
    let p_pulse  = max(mp(11u, 2u), 0.0);
    let p_bright = max(mp(11u, 3u), 0.0);

    let uv_cent = uv - 0.5;
    let dist_c = length(uv_cent) * 2.0;

    let pulse_base = sin(t * 3.0 * p_pulse) * 0.5 + 0.5;
    let pulse = 0.7 + 0.3 * pow(pulse_base, 2.0);
    let pulse2 = 1.0 + 0.1 * sin(t * 8.1 * p_pulse + 1.3);
    let total_pulse = pulse * pulse2;

    let ring_r = 0.6;
    let ring_d = abs(dist_c - ring_r);
    let core = smoothstep(0.02, 0.0, ring_d) * 4.0 * total_pulse;
    let glow1 = exp(-ring_d * ring_d / 0.01) * total_pulse * p_glow;
    let glow2 = exp(-ring_d * ring_d / 0.04) * 0.4 * total_pulse * p_glow;
    let glow3 = exp(-ring_d * ring_d / 0.16) * 0.15 * total_pulse * p_glow;
    let total_glow = glow1 + glow2 + glow3;

    let line_d = abs(uv_cent.x);
    let line_core = smoothstep(0.01, 0.0, line_d) * smoothstep(0.4, 0.2, abs(uv_cent.y));
    let line_glow = exp(-line_d * line_d / 0.006) * smoothstep(0.5, 0.1, abs(uv_cent.y)) * 0.3;
    let line_total = (line_core * 1.5 + line_glow) * total_pulse;

    let glow_col = hue_color(p_hue);
    let core_color = mix(glow_col, vec3<f32>(1.0), min(core / 3.0, 1.0));
    var color = core_color * core + glow_col * total_glow + glow_col * line_total;
    color += vec3<f32>(glow_col.r * 1.1, glow_col.g * 0.9, glow_col.b * 1.2) * glow3 * 0.3 * total_pulse;

    let flicker = hash11(floor(t * 30.0)) * 0.05 + 0.975;
    color *= flicker * p_bright;

    let lum = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));

    r.albedo = vec3(0.0);
    r.emission = color;
    r.metallic = 0.0;
    r.roughness = 0.1;
    r.alpha = clamp(lum, 0.0, 1.0);
    r.normal = n;
    r.is_emissive_only = true;
    return r;
}

// ── 12: Shield ──

fn hex_distance(p: vec2<f32>, scale: f32) -> vec2<f32> {
    let s = vec2<f32>(1.0, 1.7320508); let ps = p * scale; let half_s = s * 0.5;
    let a = ps - s * floor(ps / s + 0.5); let b = ps - half_s - s * floor((ps - half_s) / s + 0.5);
    let da = abs(a); let db = abs(b);
    let d_a = max(da.x * 1.5 + da.y * s.y, da.y * s.y * 2.0);
    let d_b = max(db.x * 1.5 + db.y * s.y, db.y * s.y * 2.0);
    return vec2<f32>(min(d_a, d_b), min(length(a), length(b)));
}

fn mat_shield(wp: vec3<f32>, n: vec3<f32>, ep: vec3<f32>, t: f32) -> MaterialResult {
    var r: MaterialResult;
    // Tunable params
    let p_density = max(mp(12u, 0u), 1.0);
    let p_hue     = clamp(mp(12u, 1u), 0.0, 1.0);
    let p_glow    = max(mp(12u, 2u), 0.0);
    let p_speed   = max(mp(12u, 3u), 0.0);
    let p_ripple  = max(mp(12u, 4u), 0.0);
    let ts = t * p_speed;

    let view_dir = normalize(ep - wp);
    let n_dot_v = max(dot(n, view_dir), 0.0);

    let sphere_uv = vec2<f32>(atan2(n.x, n.z) / TAU + 0.5, n.y * 0.5 + 0.5);
    let hex = hex_distance(sphere_uv, p_density);
    let edge_glow = 1.0 - smoothstep(0.42, 0.5, hex.x);
    let pulse = sin(ts * 2.0 + sphere_uv.y * 12.0) * 0.5 + 0.5;
    let pulse_glow = edge_glow * (0.3 + 0.7 * pulse) * p_glow;

    let fresnel = pow(1.0 - n_dot_v, 3.0);
    let cell_id = floor(sphere_uv * p_density);
    let ch = hash21(cell_id);
    let cell_flicker = smoothstep(0.7, 1.0, sin(ts * 3.0 + ch * TAU) * 0.5 + 0.5) * 0.3;

    let hit_phase = fract(ts * 0.3);
    let hit_n = vec2<f32>(sin(ts * 0.7) * 0.3, cos(ts * 0.5) * 0.3);
    let hit_dist = length(n.xy - hit_n);
    let ripple_r = hit_phase * 0.8;
    let ripple = (1.0 - smoothstep(0.0, 0.1, abs(hit_dist - ripple_r))) * (1.0 - hit_phase) * 1.5 * p_ripple;

    let base_col = hue_color(p_hue);
    let energy = pulse_glow + fresnel * 1.5 + ripple + cell_flicker;

    let shield_refl_dir = reflect(-view_dir, n);
    let shield_env = env_reflect(shield_refl_dir, t);

    r.albedo = vec3(0.0);
    r.emission = base_col * energy + shield_env * edge_glow * 0.2;
    r.metallic = 0.3;
    r.roughness = 0.2;
    r.alpha = clamp((edge_glow * 0.4 + fresnel * 0.5 + ripple * 0.6 + cell_flicker) * 0.8, 0.0, 1.0);
    r.normal = n;
    r.is_emissive_only = true;
    return r;
}

// ── 13: Dissolve ──

fn warped_fbm(p: vec2<f32>, t: f32) -> f32 {
    let q = vec2<f32>(fbm5(p), fbm5(p + vec2<f32>(5.2, 1.3)));
    let rr = vec2<f32>(fbm5(p + 4.0 * q + vec2<f32>(1.7 + t * 0.1, 9.2)), fbm5(p + 4.0 * q + vec2<f32>(8.3, 2.8 + t * 0.12)));
    return fbm5(p + 4.0 * rr);
}

fn mat_dissolve(wp: vec3<f32>, n: vec3<f32>, ep: vec3<f32>, t: f32, uv: vec2<f32>) -> MaterialResult {
    var r: MaterialResult;
    // Tunable params
    let p_thresh   = clamp(mp(13u, 0u), 0.0, 1.0);
    let p_edge_w   = max(mp(13u, 1u), 0.005);
    let p_hue      = clamp(mp(13u, 2u), 0.0, 1.0);
    let p_edge_glow= max(mp(13u, 3u), 0.0);
    let p_auto     = clamp(mp(13u, 4u), 0.0, 1.0);

    // Auto-animated threshold (legacy behavior) blended with manual threshold
    let auto_progress = (sin(t * 0.5) * 0.5 + 0.5) * 0.85 + 0.05;
    let progress = mix(p_thresh, auto_progress, p_auto);
    let noise_uv = uv * 4.0;
    let noise_val = warped_fbm(noise_uv, t);
    let dir = normalize(vec2<f32>(1.0, 0.5));
    let dir_bias = dot(uv - 0.5, dir) * 0.5 + 0.5;
    let biased = noise_val * 0.6 + dir_bias * 0.4;
    let threshold = progress;
    let dist_to_edge = biased - threshold;

    if dist_to_edge < 0.0 {
        r.albedo = vec3(0.0);
        r.emission = vec3(0.0);
        r.metallic = 0.0;
        r.roughness = 0.5;
        r.alpha = 0.0;
        r.normal = n;
        r.is_emissive_only = true;
        return r;
    }

    let edge_width = p_edge_w;
    let edge_zone = smoothstep(0.0, edge_width, dist_to_edge);
    let edge_intensity = 1.0 - edge_zone;

    let edge_col = hue_color(p_hue);
    let hot_core = vec3<f32>(1.0, 1.0, 0.8);
    let glow_color = mix(hot_core, edge_col, smoothstep(0.0, 0.6, edge_intensity * edge_width * 20.0));
    let glow_brightness = edge_intensity * (p_edge_glow + sin(t * 10.0 + noise_val * 20.0) * 0.3);

    let base_color = vec3<f32>(0.8, 0.85, 0.9);
    let surface_detail = vnoise(uv * 20.0 + t * 0.5) * 0.1;
    let surface = base_color + surface_detail;

    let ember_noise = vnoise(uv * 40.0 + vec2<f32>(t * 2.0, t * 1.5));
    let ember = smoothstep(0.85, 0.95, ember_noise) * edge_intensity * 1.5;

    r.albedo = surface * edge_zone;
    r.emission = glow_color * glow_brightness * edge_intensity + edge_col * ember;
    r.metallic = 0.0;
    r.roughness = 0.5;
    r.alpha = 1.0;
    r.normal = n;
    r.is_emissive_only = false;
    return r;
}

// ── 15: Lightning ──

fn lightning_offset(y: f32, seed: f32, ts: f32) -> f32 {
    var offset = 0.0; var amp = 0.2; var freq = 2.0;
    for (var i = 0; i < 6; i++) {
        let phase = hash11(seed * 7.13 + f32(i) * 3.7 + ts) * TAU;
        offset += sin(y * freq * PI + phase) * amp; amp *= 0.6; freq *= 1.9;
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

// 3D arc sample: distance from sample point p (object-local) to a vertical
// arc whose axis is at (xc, zc), with x/z perturbation along the y axis.
// `y_extent` is half the active vertical range (e.g. 0.85 = arcs in y ∈ [-0.85, 0.85]).
fn arc_brightness_3d(p: vec3<f32>, seed: f32, ts: f32, thick: f32, xc: f32, zc: f32, y_extent: f32, glow: f32) -> f32 {
    let y_norm = (p.y + y_extent) / (2.0 * y_extent);
    if y_norm < 0.0 || y_norm > 1.0 { return 0.0; }
    let ox = lightning_offset(y_norm, seed, ts);
    let oz = lightning_offset(y_norm, seed * 1.71 + 3.3, ts);
    let dx = p.x - xc - ox;
    let dz = p.z - zc - oz;
    let d2 = dx * dx + dz * dz;
    let core = exp(-d2 / (thick * thick * 0.25)) * 1.4;
    let g    = exp(-d2 / (thick * thick * 3.0))  * glow;
    let end_fade = smoothstep(0.0, 0.18, y_norm) * smoothstep(1.0, 0.82, y_norm);
    return (core + g) * end_fade;
}

fn mat_lightning(wp: vec3<f32>, n: vec3<f32>, ep: vec3<f32>, t: f32, obj_center: vec3<f32>) -> MaterialResult {
    var r: MaterialResult;

    // ── Tunable params (kind = 14) ──
    let p_thick      = mp(14u, 0u);
    let p_glow       = mp(14u, 1u);
    let p_arc_count  = clamp(mp(14u, 2u), 1.0, 6.0);
    let p_branch     = mp(14u, 3u);
    let p_y_range    = clamp(mp(14u, 4u), 0.3, 1.0);
    let p_speed      = max(mp(14u, 5u), 0.05);
    let p_hue        = clamp(mp(14u, 6u), 0.0, 1.0);
    let p_brightness = max(mp(14u, 7u), 0.0);

    let arc_n = i32(p_arc_count + 0.5);

    // Ray-sphere intersection (radius = 1 in object space)
    let view_dir = normalize(wp - ep);
    let sphere_radius = 1.0;
    let oc = ep - obj_center;
    let b = dot(oc, view_dir);
    let c_val = dot(oc, oc) - sphere_radius * sphere_radius;
    let disc = b * b - c_val;
    if disc < 0.0 {
        r.albedo = vec3(0.0); r.emission = vec3(0.0); r.alpha = 0.0;
        r.metallic = 0.0; r.roughness = 0.3; r.normal = n; r.is_emissive_only = true;
        return r;
    }
    let sqrt_disc = sqrt(disc);
    let t_near = max(-b - sqrt_disc, 0.0);
    let t_far  = -b + sqrt_disc;
    let march_dist = t_far - t_near;

    let tt = t * p_speed;
    let ts = floor(tt * 4.0) * 0.1;

    // 軸位置をシード生成 (最大 6 本だが arc_n まで使用)
    var arc_xc: array<f32, 6>;
    var arc_zc: array<f32, 6>;
    var arc_seed: array<f32, 6>;
    for (var i = 0; i < 6; i++) {
        let s = f32(i) * 17.31 + 5.7;
        arc_seed[i] = s;
        arc_xc[i] = (hash11(s + 0.1) - 0.5) * 0.55;
        arc_zc[i] = (hash11(s + 0.27) - 0.5) * 0.55;
    }

    let steps = 40;
    let step_size = march_dist / f32(steps);
    var transmittance = 1.0;
    var accum = vec3<f32>(0.0);

    for (var i = 0; i < 40; i++) {
        let ray_t = t_near + (f32(i) + 0.5) * step_size;
        let p = ep + view_dir * ray_t - obj_center;

        var total = 0.0;
        for (var a = 0; a < 6; a++) {
            if a >= arc_n { break; }
            total += arc_brightness_3d(p, arc_seed[a], ts, p_thick, arc_xc[a], arc_zc[a], p_y_range, p_glow);
            // 分岐 (枝)
            for (var bi = 0; bi < 2; bi++) {
                let bs = arc_seed[a] + f32(bi) * 23.17;
                let bc = hash11(bs + ts * 3.0);
                if bc < p_branch {
                    let by = (hash11(bs + 1.0 + ts) - 0.5) * (p_y_range * 1.4);
                    let by_norm = (by + p_y_range) / (2.0 * p_y_range);
                    if by_norm > 0.1 && by_norm < 0.9 {
                        let bx = arc_xc[a] + lightning_offset(by_norm, arc_seed[a], ts);
                        let bz = arc_zc[a] + lightning_offset(by_norm, arc_seed[a] * 1.71 + 3.3, ts);
                        let dx = p.x - bx;
                        let dy = p.y - by;
                        let dz = p.z - bz;
                        let bd2 = dx * dx + dy * dy * 0.6 + dz * dz;
                        total += exp(-bd2 / (p_thick * p_thick * 1.2)) * 0.5;
                    }
                }
            }
        }

        if total > 0.001 {
            // Hue: 0 = warm violet, 0.5 = blue, 1 = teal cyan
            let col_a = vec3<f32>(0.7, 0.3, 1.0);   // hue=0
            let col_b = vec3<f32>(0.35, 0.55, 1.0); // hue=0.5
            let col_c = vec3<f32>(0.4, 1.0, 0.95);  // hue=1
            var base_col: vec3<f32>;
            if p_hue < 0.5 {
                base_col = mix(col_a, col_b, p_hue * 2.0);
            } else {
                base_col = mix(col_b, col_c, (p_hue - 0.5) * 2.0);
            }
            let white_mix = smoothstep(0.6, 2.5, total);
            let core_col = mix(base_col, vec3<f32>(1.0), white_mix);

            let extinct = total * 8.0 * step_size;
            accum += core_col * total * step_size * transmittance * (5.0 * p_brightness);
            transmittance *= exp(-extinct);
        }
        if transmittance < 0.02 { break; }
    }

    let flicker = 0.8 + 0.2 * sin(tt * 30.0) * sin(tt * 17.0 + 1.3);
    accum *= flicker;

    r.albedo = vec3(0.0);
    r.emission = accum;
    r.metallic = 0.0;
    r.roughness = 0.3;
    r.alpha = clamp((1.0 - transmittance) * 1.5, 0.0, 1.0);
    r.normal = n;
    r.is_emissive_only = true;
    return r;
}

// ── Shared: 3D Voronoi ──

fn voronoi3d(p: vec3<f32>) -> vec2<f32> {
    let cell = floor(p);
    let frac = fract(p);
    var min_d = 1.0;
    var second_d = 1.0;
    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let neighbor = vec3<f32>(f32(dx), f32(dy), f32(dz));
                let nc = cell + neighbor;
                let point = neighbor + vec3<f32>(
                    fract(sin(dot(nc, vec3(127.1, 311.7, 74.7))) * 43758.5453),
                    fract(sin(dot(nc, vec3(269.5, 183.3, 246.1))) * 43758.5453),
                    fract(sin(dot(nc, vec3(113.5, 271.9, 124.6))) * 43758.5453)
                );
                let dd = length(frac - point);
                if dd < min_d { second_d = min_d; min_d = dd; }
                else if dd < second_d { second_d = dd; }
            }
        }
    }
    return vec2(min_d, second_d);
}

// ── 21: Rock ──
// Shared rock surface function — reused by Lava
struct RockSurface {
    albedo: vec3<f32>,
    normal: vec3<f32>,
    roughness: f32,
};

fn rock_height_sample(n: vec3<f32>, scale: f32) -> f32 {
    let p = n * scale;
    let warp = vec3<f32>(
        simplex3d(p * 0.7 + vec3(0.0, 3.2, 7.1)),
        simplex3d(p * 0.7 + vec3(5.4, 0.0, 2.3)),
        simplex3d(p * 0.7 + vec3(1.8, 6.7, 0.0)),
    );
    let w = p + warp * 0.6;
    let h1 = simplex3d(w * 0.8) * 0.5 + 0.5;
    let h2 = simplex3d(w * 2.0 + vec3(3.1, 7.4, 1.8)) * 0.5 + 0.5;
    let h3 = simplex3d(w * 4.5 + vec3(8.2, 1.6, 5.3)) * 0.5 + 0.5;
    return h1 * 0.35 + h2 * 0.25 + h3 * 0.2;
}

fn rock_height_gradient(n: vec3<f32>, scale: f32, eps: f32) -> vec3<f32> {
    let hpx = rock_height_sample(n + vec3(eps, 0.0, 0.0), scale);
    let hnx = rock_height_sample(n - vec3(eps, 0.0, 0.0), scale);
    let hpy = rock_height_sample(n + vec3(0.0, eps, 0.0), scale);
    let hny = rock_height_sample(n - vec3(0.0, eps, 0.0), scale);
    let hpz = rock_height_sample(n + vec3(0.0, 0.0, eps), scale);
    let hnz = rock_height_sample(n - vec3(0.0, 0.0, eps), scale);
    return vec3<f32>(hpx - hnx, hpy - hny, hpz - hnz);
}

fn eval_rock(n: vec3<f32>, scale: f32) -> RockSurface {
    var rs: RockSurface;
    let p = n * scale;

    // ── Domain warping for organic, non-repeating look ──
    let warp1 = vec3<f32>(
        simplex3d(p * 0.7 + vec3(0.0, 3.2, 7.1)),
        simplex3d(p * 0.7 + vec3(5.4, 0.0, 2.3)),
        simplex3d(p * 0.7 + vec3(1.8, 6.7, 0.0)),
    );
    let warped = p + warp1 * 0.6;

    // ── Height field: FBM for terrain-like bumps ──
    let h1 = simplex3d(warped * 0.8) * 0.5 + 0.5;                     // large mounds
    let h2 = simplex3d(warped * 2.0 + vec3(3.1, 7.4, 1.8)) * 0.5 + 0.5; // medium ridges
    let h3 = simplex3d(warped * 4.5 + vec3(8.2, 1.6, 5.3)) * 0.5 + 0.5; // small bumps
    let h4 = simplex3d(warped * 10.0 + vec3(2.7, 5.9, 3.4)) * 0.5 + 0.5; // fine grain
    let h5 = simplex3d(warped * 20.0 + vec3(6.1, 3.3, 9.7)) * 0.5 + 0.5; // micro detail
    let height = h1 * 0.35 + h2 * 0.25 + h3 * 0.2 + h4 * 0.12 + h5 * 0.08;

    // ── Crevices: sharp dips where height is low ──
    let crevice = smoothstep(0.35, 0.25, height);

    // ── Color: mineral layers, not uniform gray ──
    let dark_rock = vec3<f32>(0.04, 0.035, 0.03);
    let mid_rock = vec3<f32>(0.12, 0.10, 0.08);
    let light_rock = vec3<f32>(0.20, 0.17, 0.14);
    let warm_tint = vec3<f32>(0.14, 0.09, 0.05);  // iron oxide

    // Layer blending based on warped noise
    let layer = simplex3d(warped * 1.2 + vec3(4.4, 2.2, 8.8)) * 0.5 + 0.5;
    var color = mix(mid_rock, light_rock, height);
    color = mix(color, warm_tint, layer * 0.3);          // warm patches
    color = mix(color, dark_rock, crevice * 0.9);         // dark crevices
    color *= 0.6 + height * 0.5;                          // crude AO: low areas darken

    // Exposed faces are lighter
    let exposure = simplex3d(warped * 3.0 + vec3(1.1, 5.5, 2.2)) * 0.5 + 0.5;
    color = mix(color, light_rock * 1.1, exposure * 0.2 * height);

    rs.albedo = color;

    // ── Roughness: mostly rough, polished only where worn ──
    rs.roughness = 0.8 + h4 * 0.15 - height * 0.1;

    // ── Normal: gradient of the height field ──
    let eps = 0.01;
    // Compute gradient by sampling height at 6 offset positions
    let grad = rock_height_gradient(n, scale, eps);

    // Fine detail bump (separate, higher frequency)
    let fg = vec3<f32>(
        simplex3d((n + vec3(eps, 0.0, 0.0)) * scale * 12.0) - simplex3d((n - vec3(eps, 0.0, 0.0)) * scale * 12.0),
        simplex3d((n + vec3(0.0, eps, 0.0)) * scale * 12.0) - simplex3d((n - vec3(0.0, eps, 0.0)) * scale * 12.0),
        simplex3d((n + vec3(0.0, 0.0, eps)) * scale * 12.0) - simplex3d((n - vec3(0.0, 0.0, eps)) * scale * 12.0),
    );

    rs.normal = normalize(n + grad * 1.8 + fg * 0.25);

    return rs;
}

fn mat_rock(wp: vec3<f32>, n: vec3<f32>, ep: vec3<f32>, t: f32) -> MaterialResult {
    var r: MaterialResult;
    // Tunable params
    let p_bumps  = max(mp(21u, 0u), 0.0);
    let p_hue    = clamp(mp(21u, 1u), 0.0, 1.0);
    let p_sat    = max(mp(21u, 2u), 0.0);
    let p_detail = max(mp(21u, 3u), 0.1);

    let rock = eval_rock(n, 3.0 * p_detail);
    let tint = hue_color(p_hue);

    r.albedo = mix(rock.albedo, rock.albedo * tint * 2.0, p_sat * 0.5);
    r.emission = vec3(0.0);
    r.metallic = 0.0;
    r.roughness = rock.roughness;
    r.alpha = 1.0;
    // Blend bump strength: lerp between flat normal and rock normal
    r.normal = normalize(mix(n, rock.normal, p_bumps));
    r.is_emissive_only = false;
    return r;
}

// ── 15: Lava (Rock + molten glow) ──

fn mat_lava(wp: vec3<f32>, n: vec3<f32>, ep: vec3<f32>, t: f32) -> MaterialResult {
    var r: MaterialResult;
    // Tunable params
    let p_flow  = max(mp(15u, 0u), 0.0);
    let p_heat  = max(mp(15u, 1u), 0.0);
    let p_crack = max(mp(15u, 2u), 0.1);
    let p_glow  = max(mp(15u, 3u), 0.0);

    let rock = eval_rock(n, 3.0);

    let tt = t * 0.12 * p_flow;
    let flow_warp = vec3<f32>(
        simplex3d(n * 1.5 + vec3(tt, 0.0, 3.7)) * 0.25,
        simplex3d(n * 1.5 + vec3(0.0, tt * 0.8, 7.1)) * 0.25,
        simplex3d(n * 1.5 + vec3(5.2, tt * 0.6, 0.0)) * 0.25
    );
    let p = n * 2.5 * p_crack + flow_warp;

    let v1 = voronoi3d(p);
    let v2 = voronoi3d(p * 2.0 + vec3(5.0));

    let edge1 = v1.y - v1.x;
    let edge2 = v2.y - v2.x;
    let crack = smoothstep(0.14, 0.02, edge1) * 0.7 + smoothstep(0.12, 0.02, edge2) * 0.3;

    let pulse = sin(n.x * 3.0 + n.y * 5.0 + t * 0.4) * 0.25 + 0.75;
    let heat = crack * pulse * p_heat;

    // Temperature color: deep crack = white hot → orange → dark red
    var lava_col: vec3<f32>;
    if heat > 0.7 { lava_col = mix(vec3<f32>(1.0, 0.7, 0.1), vec3<f32>(1.0, 0.95, 0.8), (heat - 0.7) / 0.3); }
    else if heat > 0.35 { lava_col = mix(vec3<f32>(0.9, 0.15, 0.0), vec3<f32>(1.0, 0.7, 0.1), (heat - 0.35) / 0.35); }
    else { lava_col = mix(vec3<f32>(0.2, 0.01, 0.0), vec3<f32>(0.9, 0.15, 0.0), heat / 0.35); }

    // Blend rock surface with lava glow
    // Non-crack areas: pure rock. Crack areas: darker (glow comes from emission)
    let rock_darkened = rock.albedo * (1.0 - crack * 0.9);
    // Edge of crack: rock gets slightly red-hot tint
    let edge_heat = smoothstep(0.0, 0.2, crack) * (1.0 - smoothstep(0.3, 0.8, crack));
    let heated_rock = mix(rock_darkened, vec3<f32>(0.15, 0.03, 0.01), edge_heat);

    r.albedo = heated_rock;
    r.emission = lava_col * heat * 3.5 * p_glow;
    r.metallic = 0.0;
    // Rock is rough, cracks are smooth (molten)
    r.roughness = mix(rock.roughness, 0.1, crack);
    r.alpha = 1.0;
    // Rock bumps fade near cracks (molten areas are smooth)
    r.normal = mix(rock.normal, n, crack * 0.7);
    r.is_emissive_only = false;
    return r;
}

// ── 16: Ice ──

fn mat_ice(wp: vec3<f32>, n: vec3<f32>, ep: vec3<f32>, t: f32) -> MaterialResult {
    var r: MaterialResult;
    // Tunable params
    let p_frost = max(mp(16u, 0u), 0.1);
    let p_sss   = max(mp(16u, 1u), 0.0);
    let p_hue   = clamp(mp(16u, 2u), 0.0, 1.0);
    let p_rough = clamp(mp(16u, 3u), 0.0, 1.0);

    let view_dir = normalize(ep - wp);
    let n_dot_v = max(dot(n, view_dir), 0.0);
    let fresnel = pow(1.0 - n_dot_v, 4.0) * 0.6 + 0.1;

    let p = n * 5.0 * p_frost;
    let frost = simplex3d(p + vec3<f32>(t * 0.02)) * 0.5 + 0.5;
    let frost2 = simplex3d(p * 3.0 + vec3<f32>(3.7, 1.2, 8.4)) * 0.5 + 0.5;
    let frost_pattern = frost * 0.6 + frost2 * 0.4;

    let sss = pow(1.0 - n_dot_v, 2.0) * 0.4 * p_sss;
    let tint = hue_color(p_hue);
    let sss_color = tint * sss;

    let cool = mix(vec3<f32>(0.7, 0.85, 0.95), vec3<f32>(0.9, 0.95, 1.0), frost_pattern);
    let base = mix(cool, mix(cool, tint, 0.5), 0.4);

    r.albedo = base;
    r.emission = sss_color;
    r.metallic = 0.1;
    r.roughness = mix(p_rough * 0.4, p_rough, frost_pattern);
    r.alpha = 1.0;
    r.normal = n;
    r.is_emissive_only = false;
    return r;
}

// ── 17: Cloud (volumetric) ──

fn cloud_density(p: vec3<f32>, t: f32) -> f32 {
    let wind = vec3<f32>(t * 0.1, 0.0, t * 0.05);
    let pp = p + wind;
    let n1 = simplex3d(pp * 1.5) * 0.5;
    let n2 = simplex3d(pp * 3.0 + vec3(5.2, 1.3, 2.8)) * 0.25;
    let n3 = simplex3d(pp * 6.0 + vec3(1.7, 9.2, 4.1)) * 0.125;
    let noise = n1 + n2 + n3;

    let radial = length(p);
    let shape = smoothstep(1.0, 0.3, radial);
    return max((noise * 0.5 + 0.5) * shape - 0.2, 0.0) * 2.0;
}

fn mat_cloud(wp: vec3<f32>, n: vec3<f32>, ep: vec3<f32>, t: f32, obj_center: vec3<f32>) -> MaterialResult {
    var r: MaterialResult;
    // Tunable params
    let p_density_s  = max(mp(17u, 0u), 0.0);
    let p_speed      = max(mp(17u, 1u), 0.0);
    let p_brightness = max(mp(17u, 2u), 0.0);
    let p_detail     = max(mp(17u, 3u), 0.1);
    let ts = t * p_speed;

    let view_dir = normalize(wp - ep);
    let oc = ep - obj_center;
    let b = dot(oc, view_dir);
    let c_val = dot(oc, oc) - 1.0;
    let disc = b * b - c_val;

    if disc < 0.0 {
        r.albedo = vec3(0.0); r.emission = vec3(0.0); r.alpha = 0.0;
        r.metallic = 0.0; r.roughness = 0.9; r.normal = n; r.is_emissive_only = true;
        return r;
    }

    let sqrt_disc = sqrt(disc);
    let t_near = max(-b - sqrt_disc, 0.0);
    let t_far = -b + sqrt_disc;

    let steps = 32;
    let step_size = (t_far - t_near) / f32(steps);
    var transmittance = 1.0;
    var accum = vec3<f32>(0.0);
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let cloud_col = vec3<f32>(0.95, 0.95, 0.97);
    let shadow_col = vec3<f32>(0.4, 0.45, 0.55);

    for (var i = 0; i < 32; i++) {
        let ray_t = t_near + (f32(i) + 0.5) * step_size;
        let local = ep + view_dir * ray_t - obj_center;
        let d = cloud_density(local * p_detail, ts) * p_density_s;
        if d > 0.001 {
            let ls = cloud_density((local + light_dir * 0.15) * p_detail, ts) * p_density_s;
            let light_atten = exp(-ls * 3.0);
            let lit = mix(shadow_col, cloud_col, light_atten);
            accum += lit * d * step_size * transmittance * 4.0 * p_brightness;
            transmittance *= exp(-d * 5.0 * step_size);
        }
        if transmittance < 0.02 { break; }
    }

    r.albedo = vec3(0.0);
    r.emission = accum;
    r.metallic = 0.0;
    r.roughness = 0.9;
    r.alpha = clamp((1.0 - transmittance) * 1.1, 0.0, 1.0);
    r.normal = n;
    r.is_emissive_only = true;
    return r;
}

// ── 18: Explosion (volumetric) ──

fn explosion_density(p: vec3<f32>, t: f32) -> vec2<f32> {
    // Looping explosion: expand then fade
    let phase = fract(t * 0.25);
    let expand = 0.15 + phase * 0.8;
    let fade = smoothstep(0.9, 1.0, phase);

    // Scale position by expansion
    let pp = p / max(expand, 0.01);

    // Heavy turbulence — explosions are chaotic
    let warp = vec3<f32>(
        simplex3d(pp * 1.8 + vec3(t * 0.6, 0.0, 3.7)),
        simplex3d(pp * 1.8 + vec3(0.0, t * 0.5, 7.1)),
        simplex3d(pp * 1.8 + vec3(5.2, t * 0.4, 0.0))
    );
    let warped = pp + warp * 0.6;

    // FBM noise
    let n1 = simplex3d(warped * 1.5) * 0.5;
    let n2 = simplex3d(warped * 3.0 + vec3(5.2, 1.3, 2.8)) * 0.25;
    let n3 = simplex3d(warped * 6.0 + vec3(1.7, 9.2, 4.1)) * 0.12;
    let n4 = simplex3d(warped * 12.0) * 0.06;
    let noise = n1 + n2 + n3 + n4 + 0.5;

    let radial = length(p);
    // Mushroom shape: rises upward
    let rise = p.y * 0.3; // upper part is denser
    let shell = smoothstep(expand + 0.15, expand * 0.3, radial)
              * (1.0 - fade);
    let density = max(noise * shell - 0.08, 0.0) * 3.0;

    // Temperature: hot core, cool edges, cools over time
    let core_heat = exp(-radial * radial / (expand * expand) * 2.0);
    let temp = core_heat * (1.0 - phase * 0.7) * noise;

    return vec2(density, temp);
}

fn mat_explosion(wp: vec3<f32>, n: vec3<f32>, ep: vec3<f32>, t: f32, obj_center: vec3<f32>) -> MaterialResult {
    var r: MaterialResult;
    // Tunable params
    let p_force = max(mp(18u, 0u), 0.1);
    let p_heat  = max(mp(18u, 1u), 0.0);
    let p_speed = max(mp(18u, 2u), 0.0);
    let p_smoke = clamp(mp(18u, 3u), 0.0, 2.0);
    let ts = t * p_speed;

    let view_dir = normalize(wp - ep);
    let oc = ep - obj_center;
    let b = dot(oc, view_dir);
    let c_val = dot(oc, oc) - 1.0;
    let disc = b * b - c_val;

    if disc < 0.0 {
        r.albedo = vec3(0.0); r.emission = vec3(0.0); r.alpha = 0.0;
        r.metallic = 0.0; r.roughness = 0.8; r.normal = n; r.is_emissive_only = true;
        return r;
    }

    let sqrt_disc = sqrt(disc);
    let t_near = max(-b - sqrt_disc, 0.0);
    let t_far = -b + sqrt_disc;
    let steps = 40;
    let step_size = (t_far - t_near) / f32(steps);
    var transmittance = 1.0;
    var accum = vec3<f32>(0.0);

    for (var i = 0; i < 40; i++) {
        let ray_t = t_near + (f32(i) + 0.5) * step_size;
        let local = (ep + view_dir * ray_t - obj_center) / p_force;
        let dt = explosion_density(local, ts);
        let d = dt.x;
        let temp = dt.y * p_heat;
        if d > 0.001 {
            var col: vec3<f32>;
            if temp > 0.8 { col = mix(vec3(1.0, 0.85, 0.3), vec3(1.0, 0.98, 0.9), clamp((temp - 0.8) / 0.3, 0.0, 1.0)); }
            else if temp > 0.5 { col = mix(vec3(1.0, 0.4, 0.0), vec3(1.0, 0.85, 0.3), (temp - 0.5) / 0.3); }
            else if temp > 0.2 { col = mix(vec3(0.6, 0.08, 0.0), vec3(1.0, 0.4, 0.0), (temp - 0.2) / 0.3); }
            else {
                // Cool = dark smoke
                col = mix(vec3(0.03, 0.02, 0.02) * p_smoke, vec3(0.6, 0.08, 0.0), temp / 0.2);
            }

            accum += col * d * step_size * transmittance * 5.0;
            transmittance *= exp(-d * 7.0 * step_size);
        }
        if transmittance < 0.02 { break; }
    }

    let flicker = 0.9 + 0.1 * sin(ts * 20.0) * sin(ts * 13.0 + 2.1);
    accum *= flicker;

    r.albedo = vec3(0.0);
    r.emission = accum;
    r.metallic = 0.0;
    r.roughness = 0.8;
    r.alpha = clamp((1.0 - transmittance) * 1.1, 0.0, 1.0);
    r.normal = n;
    r.is_emissive_only = true;
    return r;
}

// ── 19: Tornado (volumetric) ──

fn tornado_density(p: vec3<f32>, t: f32) -> f32 {
    let height = (p.y + 1.0) * 0.5; // 0=bottom, 1=top
    let radial = length(vec2(p.x, p.z));
    let angle_val = atan2(p.z, p.x);

    // Funnel: wide at bottom (ground), narrow at top (cloud)
    let funnel_r = mix(0.6, 0.06, pow(height, 1.5));

    // Spiraling twist — increases with height
    let twist_speed = t * 4.0;
    let twist = angle_val + twist_speed + height * 12.0;

    // Multiple spiral arms with noise
    let arm1 = sin(twist * 2.0) * 0.5 + 0.5;
    let arm2 = sin(twist * 3.0 + 2.0) * 0.5 + 0.5;
    let spiral = max(arm1, arm2);

    // Distance from funnel wall
    let wall_dist = abs(radial - funnel_r);
    let wall_thick = 0.08 + spiral * 0.12;
    let wall = exp(-wall_dist * wall_dist / (wall_thick * wall_thick));

    // Inner debris/dust
    let inner = smoothstep(funnel_r, funnel_r * 0.2, radial) * 0.3;

    // Rising debris noise
    let debris = simplex3d(vec3(p.x * 5.0, p.y * 3.0 - t * 3.0, p.z * 5.0)) * 0.5 + 0.5;
    let debris2 = simplex3d(vec3(p.x * 10.0, p.y * 6.0 - t * 5.0, p.z * 10.0)) * 0.3;

    // Edge noise to break up the shape
    let edge_noise = simplex3d(vec3(p.x * 3.0, p.y * 2.0 - t * 2.0, p.z * 3.0)) * 0.15;

    let shape = smoothstep(-0.05, 0.08, height) * smoothstep(1.05, 0.85, height);
    let density = (wall + inner) * (debris + 0.4 + debris2) * shape;

    return max(density + edge_noise - 0.1, 0.0) * 2.0;
}

fn mat_tornado(wp: vec3<f32>, n: vec3<f32>, ep: vec3<f32>, t: f32, obj_center: vec3<f32>) -> MaterialResult {
    var r: MaterialResult;
    // Tunable params
    let p_speed      = max(mp(19u, 0u), 0.0);
    let p_density_s  = max(mp(19u, 1u), 0.0);
    let p_twist      = max(mp(19u, 2u), 0.1);
    let p_brightness = max(mp(19u, 3u), 0.0);
    let ts = t * p_speed;

    let view_dir = normalize(wp - ep);
    let oc = ep - obj_center;
    let b = dot(oc, view_dir);
    let c_val = dot(oc, oc) - 1.0;
    let disc = b * b - c_val;

    if disc < 0.0 {
        r.albedo = vec3(0.0); r.emission = vec3(0.0); r.alpha = 0.0;
        r.metallic = 0.0; r.roughness = 0.7; r.normal = n; r.is_emissive_only = true;
        return r;
    }

    let sqrt_disc = sqrt(disc);
    let t_near = max(-b - sqrt_disc, 0.0);
    let t_far = -b + sqrt_disc;
    let steps = 32;
    let step_size = (t_far - t_near) / f32(steps);
    var transmittance = 1.0;
    var accum = vec3<f32>(0.0);
    let light_dir = normalize(vec3<f32>(0.3, 1.0, 0.5));
    let dust_col = vec3<f32>(0.35, 0.3, 0.25);
    let dark_col = vec3<f32>(0.15, 0.12, 0.1);

    for (var i = 0; i < 32; i++) {
        let ray_t = t_near + (f32(i) + 0.5) * step_size;
        var local = ep + view_dir * ray_t - obj_center;
        // Twist horizontally to vary spiral tightness
        let twist_a = local.y * p_twist * 1.5;
        let cs = cos(twist_a); let sn = sin(twist_a);
        local = vec3<f32>(local.x * cs - local.z * sn, local.y, local.x * sn + local.z * cs);

        let d = tornado_density(local, ts) * p_density_s;
        if d > 0.001 {
            let ls = tornado_density(local + light_dir * 0.12, ts) * p_density_s;
            let light_atten = exp(-ls * 2.0);
            let lit = mix(dark_col, dust_col, light_atten);
            accum += lit * d * step_size * transmittance * 4.0 * p_brightness;
            transmittance *= exp(-d * 5.0 * step_size);
        }
        if transmittance < 0.02 { break; }
    }

    r.albedo = vec3(0.0);
    r.emission = accum;
    r.metallic = 0.0;
    r.roughness = 0.7;
    r.alpha = clamp((1.0 - transmittance) * 1.0, 0.0, 1.0);
    r.normal = n;
    r.is_emissive_only = true;
    return r;
}

// ── 20: Skin (SSS) ──

fn mat_skin(wp: vec3<f32>, n: vec3<f32>, ep: vec3<f32>, t: f32) -> MaterialResult {
    var r: MaterialResult;
    // Tunable params
    let p_sss      = max(mp(20u, 0u), 0.0);
    let p_smooth   = clamp(mp(20u, 1u), 0.0, 1.0);
    let p_hue      = clamp(mp(20u, 2u), 0.0, 1.0);
    let p_sat      = max(mp(20u, 3u), 0.0);

    let view_dir = normalize(ep - wp);
    let n_dot_v = max(dot(n, view_dir), 0.0);

    let p = n * 15.0;
    let pore = simplex3d(p) * 0.3 + simplex3d(p * 3.0) * 0.15;
    // smoothness 1 → roughness 0.18; smoothness 0 → roughness 0.55
    let base_rough = mix(0.55, 0.18, p_smooth);
    let pore_rough = base_rough + pore * 0.15;

    let sss_wrap = pow(1.0 - n_dot_v, 2.5);
    let sss_tint = hue_color(p_hue);
    let sss_color = sss_tint * sss_wrap * 0.35 * p_sss;

    let variation = simplex3d(n * 4.0 + vec3(2.1, 5.3, 7.7)) * 0.05;
    let neutral = vec3<f32>(0.82 + variation, 0.62 + variation * 0.5, 0.48 + variation * 0.3);
    let base = mix(neutral, neutral * sss_tint * 1.5, p_sat * 0.5);

    r.albedo = base;
    r.emission = sss_color;
    r.metallic = 0.0;
    r.roughness = pore_rough;
    r.alpha = 1.0;
    r.normal = n;
    r.is_emissive_only = false;
    return r;
}

// ════════════════════════════════════════════════════
//  Fragment Shader — dispatch by kind, apply PBR
// ════════════════════════════════════════════════════

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let wp = in.world_position;
    let ep = camera.position.xyz;
    let t = camera.position.w;  // time encoded in position.w
    let raw_normal = normalize(in.world_normal);
    let uv = in.uv;

    // Landscape mode: material.w > 0.5 means huge sphere (planet surface)
    // Normal is valid but we need to use world position for texture tiling
    let is_landscape = in.material.w > 0.5;
    var wn: vec3<f32>;
    if is_landscape {
        // Use world position for pattern generation (tiles across surface)
        wn = normalize(vec3<f32>(wp.x * 0.3, wp.z * 0.3, wp.y * 0.3 + 0.5));
    } else {
        wn = raw_normal;
    }

    // Material kind from instance data
    let k = u32(in.material.z + 0.5);

    var mat: MaterialResult;
    switch k {
        case 0u:  { mat = mat_bubble(wp, wn, ep, t); }
        case 1u:  { mat = mat_glass(wp, wn, ep, t); }
        case 2u:  { mat = mat_portal(wp, wn, ep, t, uv); }
        case 3u:  { mat = mat_grid(wp, wn, ep, t, uv); }
        case 4u:  { mat = mat_water(wp, wn, ep, t); }
        case 5u:  { mat = mat_fire(wp, wn, ep, t, in.object_center); }
        case 6u:  { mat = mat_smoke(wp, wn, ep, t, in.object_center); }
        case 7u:  { mat = mat_aurora(wp, wn, ep, t, uv); }
        case 8u:  { mat = mat_hologram(wp, wn, ep, t, uv); }
        case 9u:  { mat = mat_crystal(wp, wn, ep, t); }
        case 10u: { mat = mat_metal(wp, wn, ep, t); }
        case 11u: { mat = mat_neon(wp, wn, ep, t, uv); }
        case 12u: { mat = mat_shield(wp, wn, ep, t); }
        case 13u: { mat = mat_dissolve(wp, wn, ep, t, uv); }
        case 14u: { mat = mat_lightning(wp, wn, ep, t, in.object_center); }
        case 15u: { mat = mat_lava(wp, wn, ep, t); }
        case 16u: { mat = mat_ice(wp, wn, ep, t); }
        case 17u: { mat = mat_cloud(wp, wn, ep, t, in.object_center); }
        case 18u: { mat = mat_explosion(wp, wn, ep, t, in.object_center); }
        case 19u: { mat = mat_tornado(wp, wn, ep, t, in.object_center); }
        case 20u: { mat = mat_skin(wp, wn, ep, t); }
        case 21u: { mat = mat_rock(wp, wn, ep, t); }
        default:  { mat = mat_metal(wp, wn, ep, t); }
    }

    if mat.alpha < 0.001 {
        discard;
    }

    // Override metallic/roughness from material.xy if non-emissive
    if !mat.is_emissive_only {
        // Use instance-provided PBR params, blended with procedural
        mat.metallic = max(mat.metallic, in.material.x);
        mat.roughness = clamp(mix(mat.roughness, in.material.y, 0.5), 0.04, 1.0);
    }

    // Landscape mode: restore real surface normal for lighting
    if is_landscape {
        mat.normal = raw_normal;
    }

    // Apply PBR lighting
    var result = apply_pbr_lighting(mat, wp, ep);

    // Atmospheric fog
    let fogged = apply_fog(result.rgb, wp, ep);
    return vec4<f32>(fogged, result.a);
}
