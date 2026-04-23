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

const BLOOM_THRESHOLD: f32 = 0.85;
const BLOOM_KNEE: f32 = 0.15;

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

const BLOOM_INTENSITY: f32 = 0.25;

// Color grading: lift shadows toward blue, push highlights warm
fn color_grade(color: vec3<f32>) -> vec3<f32> {
    let lum = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));

    // Shadow lift — add subtle blue to dark areas
    let shadow_tint = vec3<f32>(0.03, 0.04, 0.08);
    let shadow_mask = 1.0 - smoothstep(0.0, 0.3, lum);

    // Highlight push — add subtle warmth to bright areas
    let highlight_tint = vec3<f32>(0.05, 0.03, 0.0);
    let highlight_mask = smoothstep(0.5, 0.9, lum);

    var graded = color;
    graded += shadow_tint * shadow_mask;
    graded += highlight_tint * highlight_mask;

    // Slight contrast increase (S-curve)
    graded = clamp(graded, vec3<f32>(0.0), vec3<f32>(1.0));
    graded = graded * graded * (3.0 - 2.0 * graded); // smoothstep-like S-curve
    // Blend with original to keep it subtle
    graded = mix(color, graded, 0.35);

    return graded;
}

@fragment
fn fs_composite(in: VertexOutput) -> @location(0) vec4<f32> {
    let scene = textureSample(src_texture, src_sampler, in.uv).rgb;
    let bloom = textureSample(bloom_texture, src_sampler, in.uv).rgb;

    // Additive bloom with controlled intensity
    let combined = scene + bloom * BLOOM_INTENSITY;

    // ACES tone mapping (output is linear, sRGB surface handles gamma)
    var mapped = aces_tonemap(combined);

    // Color grading — cinematic look
    mapped = color_grade(mapped);

    // Vignette — darken screen corners
    let vignette_uv = (in.uv - 0.5) * 2.0;
    let vignette = 1.0 - smoothstep(0.5, 1.5, length(vignette_uv));
    mapped *= mix(0.7, 1.0, vignette); // subtle darkening at edges

    return vec4<f32>(mapped, 1.0);
}
