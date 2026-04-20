// Murakumo — Smoke Material
// 煙/霧シェーダー
//
// Billboard quad上にレイヤードFBMノイズベースの
// ボリューメトリックな煙エフェクトを描画する。

struct SmokeParams {
    density: f32,
    scroll_speed: f32,
    detail_scale: f32,
    opacity: f32,
    color: vec4<f32>,
}

struct SmokeInput {
    uv: vec2<f32>,
    world_pos: vec3<f32>,
    eye_pos: vec3<f32>,
    time: f32,
}

struct SmokeOutput {
    color: vec4<f32>,
    discard_pixel: bool,
}

// Hash for noise generation
fn smoke_hash(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453);
}

fn smoke_hash2(p: vec2<f32>) -> vec2<f32> {
    let px = vec2<f32>(dot(p, vec2<f32>(127.1, 311.7)), dot(p, vec2<f32>(269.5, 183.3)));
    return fract(sin(px) * 43758.5453123);
}

// Smooth 2D noise
fn smoke_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let a = smoke_hash(i);
    let b = smoke_hash(i + vec2<f32>(1.0, 0.0));
    let c = smoke_hash(i + vec2<f32>(0.0, 1.0));
    let d = smoke_hash(i + vec2<f32>(1.0, 1.0));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// FBM with 4 octaves
fn smoke_fbm4(p: vec2<f32>) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var pos = p;

    // Octave 1
    value += amplitude * smoke_noise(pos);
    pos = pos * 2.02 + vec2<f32>(1.3, -0.7);
    amplitude *= 0.5;

    // Octave 2
    value += amplitude * smoke_noise(pos);
    pos = pos * 2.03 + vec2<f32>(-0.4, 2.1);
    amplitude *= 0.5;

    // Octave 3
    value += amplitude * smoke_noise(pos);
    pos = pos * 2.01 + vec2<f32>(1.7, 0.9);
    amplitude *= 0.5;

    // Octave 4
    value += amplitude * smoke_noise(pos);

    return value;
}

// FBM with domain warping for volumetric feel
fn smoke_fbm_warped(p: vec2<f32>, t: f32) -> f32 {
    // First pass: warp coordinates
    let q = vec2<f32>(
        smoke_fbm4(p + vec2<f32>(0.0, 0.0)),
        smoke_fbm4(p + vec2<f32>(5.2, 1.3))
    );

    // Second pass: warp again for more complexity
    let r = vec2<f32>(
        smoke_fbm4(p + 4.0 * q + vec2<f32>(1.7 + t * 0.15, 9.2 - t * 0.1)),
        smoke_fbm4(p + 4.0 * q + vec2<f32>(8.3 - t * 0.12, 2.8 + t * 0.08))
    );

    return smoke_fbm4(p + 4.0 * r);
}

fn smoke_material(input: SmokeInput, params: SmokeParams) -> SmokeOutput {
    var out: SmokeOutput;
    out.discard_pixel = false;

    let uv = input.uv;
    let t = input.time * params.scroll_speed;

    // Soft circular mask
    let center = uv - 0.5;
    let dist = length(center);
    let edge_mask = 1.0 - smoothstep(0.3, 0.5, dist);
    if edge_mask <= 0.0 {
        out.discard_pixel = true;
        return out;
    }

    // Layer 1: Large slow-moving shapes
    let uv1 = vec2<f32>(
        uv.x * params.detail_scale * 0.5 + t * 0.3,
        uv.y * params.detail_scale * 0.5 - t * 0.7
    );
    let layer1 = smoke_fbm_warped(uv1, t);

    // Layer 2: Medium detail, different direction
    let uv2 = vec2<f32>(
        uv.x * params.detail_scale * 1.0 - t * 0.5,
        uv.y * params.detail_scale * 1.0 - t * 1.0
    );
    let layer2 = smoke_fbm4(uv2);

    // Layer 3: Fine detail, swirling
    let swirl_angle = t * 0.2;
    let cos_a = cos(swirl_angle);
    let sin_a = sin(swirl_angle);
    let swirl_uv = vec2<f32>(
        center.x * cos_a - center.y * sin_a,
        center.x * sin_a + center.y * cos_a
    );
    let uv3 = (swirl_uv + 0.5) * params.detail_scale * 2.0 + vec2<f32>(t * 0.2, -t * 1.3);
    let layer3 = smoke_fbm4(uv3);

    // Layer 4: Very fine wispy detail
    let uv4 = vec2<f32>(
        uv.x * params.detail_scale * 3.0 + sin(t * 0.4) * 0.5,
        uv.y * params.detail_scale * 3.0 - t * 1.8
    );
    let layer4 = smoke_noise(uv4);

    // Composite layers with different weights
    let composite = layer1 * 0.4 + layer2 * 0.3 + layer3 * 0.2 + layer4 * 0.1;

    // Apply density
    let smoke_density = pow(composite, 2.0 - params.density) * params.density;

    // Vertical fade: smoke dissipates upward
    let vert_fade = 1.0 - pow(uv.y, 2.5);
    // Bottom fade: subtle
    let bottom_fade = smoothstep(0.0, 0.15, uv.y);

    let final_density = smoke_density * edge_mask * vert_fade * bottom_fade;

    if final_density <= 0.01 {
        out.discard_pixel = true;
        return out;
    }

    // Color: slightly lighter at edges (subsurface scattering approximation)
    let base_color = vec3<f32>(params.color.x, params.color.y, params.color.z);
    let edge_light = smoothstep(0.2, 0.45, dist) * 0.2;
    let top_light = uv.y * 0.1;  // Slightly brighter toward top (backlit)

    var smoke_color = base_color + vec3<f32>(edge_light + top_light);

    // Subtle internal variation
    let color_var = (layer2 - 0.5) * 0.1;
    smoke_color += vec3<f32>(color_var * 0.5, color_var * 0.3, color_var);

    let alpha = clamp(final_density, 0.0, 1.0) * params.opacity;
    out.color = vec4<f32>(smoke_color, alpha);
    return out;
}
