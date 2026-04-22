// Bloom post-processing shader for Murakumo Gallery
// Passes: brightness extract, gaussian blur (H/V), composite with tone mapping

// ── Fullscreen triangle vertex shader (shared) ──

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi & 2u) * 2 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

// ── Bindings ──

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;

// ── Brightness extract ──

const BLOOM_THRESHOLD: f32 = 0.7;
const BLOOM_KNEE: f32 = 0.3;

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@fragment
fn fs_extract(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(src_texture, src_sampler, in.uv).rgb;
    let lum = luminance(color);
    // Soft knee threshold
    let knee_start = BLOOM_THRESHOLD - BLOOM_KNEE;
    let knee_end = BLOOM_THRESHOLD + BLOOM_KNEE;
    let contribution = smoothstep(knee_start, knee_end, lum);
    return vec4<f32>(color * contribution, 1.0);
}

// ── Gaussian blur (9-tap) ──
// Direction is encoded in texel_size: (1/w, 0) for horizontal, (0, 1/h) for vertical.
// We pass direction via the sampler texel offset trick:
// Actually, we use two separate entry points.

// Blur weights for 9-tap kernel (sigma ~2.5)
const OFFSETS: array<f32, 4> = array<f32, 4>(1.0, 2.3529, 4.3529, 6.3529);
const WEIGHTS: array<f32, 4> = array<f32, 4>(0.2270, 0.3162, 0.0702, 0.0030);

fn blur_9tap(uv: vec2<f32>, direction: vec2<f32>) -> vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(src_texture));
    let texel = direction / tex_size;

    var color = textureSample(src_texture, src_sampler, uv) * WEIGHTS[0];
    for (var i = 1u; i < 4u; i = i + 1u) {
        let offset = texel * OFFSETS[i];
        color += textureSample(src_texture, src_sampler, uv + offset) * WEIGHTS[i];
        color += textureSample(src_texture, src_sampler, uv - offset) * WEIGHTS[i];
    }
    return color;
}

@fragment
fn fs_blur_h(in: VertexOutput) -> @location(0) vec4<f32> {
    return blur_9tap(in.uv, vec2<f32>(1.0, 0.0));
}

@fragment
fn fs_blur_v(in: VertexOutput) -> @location(0) vec4<f32> {
    return blur_9tap(in.uv, vec2<f32>(0.0, 1.0));
}

// ── Composite: combine scene + bloom, apply ACES tone mapping ──

@group(0) @binding(2) var bloom_texture: texture_2d<f32>;

fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

const BLOOM_INTENSITY: f32 = 0.35;

@fragment
fn fs_composite(in: VertexOutput) -> @location(0) vec4<f32> {
    let scene = textureSample(src_texture, src_sampler, in.uv).rgb;
    let bloom = textureSample(bloom_texture, src_sampler, in.uv).rgb;

    // Additive bloom with controlled intensity
    let combined = scene + bloom * BLOOM_INTENSITY;

    // ACES tone mapping
    let mapped = aces_tonemap(combined);

    // Gamma correction (linear -> sRGB)
    let gamma = pow(mapped, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(gamma, 1.0);
}
