// Murakumo — Aurora Material
// オーロラシェーダー
//
// Billboard quad上にカーテン状の縦バンドを描画する。
// 複数周波数のsin波で揺らぎ、色間を補間、ソフトグロー+垂直フェード。

struct AuroraParams {
    curtain_speed: f32,
    wave_amplitude: f32,
    brightness: f32,
    _pad: f32,
    color_primary: vec4<f32>,
    color_secondary: vec4<f32>,
}

struct AuroraInput {
    uv: vec2<f32>,
    world_pos: vec3<f32>,
    eye_pos: vec3<f32>,
    time: f32,
}

struct AuroraOutput {
    color: vec4<f32>,
    discard_pixel: bool,
}

// Smooth noise for aurora distortion
fn aurora_hash(p: f32) -> f32 {
    return fract(sin(p * 127.1) * 43758.5453);
}

fn aurora_noise1d(p: f32) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(aurora_hash(i), aurora_hash(i + 1.0), u);
}

fn aurora_noise2d(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let h = dot(i, vec2<f32>(127.1, 311.7));
    let a = fract(sin(h) * 43758.5453);
    let b = fract(sin(h + 127.1) * 43758.5453);
    let c = fract(sin(h + 311.7) * 43758.5453);
    let d = fract(sin(h + 438.8) * 43758.5453);

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Creates a single aurora curtain band
fn aurora_band(x: f32, t: f32, freq: f32, phase: f32, speed: f32, amp: f32) -> f32 {
    // Multiple sine waves create curtain folds
    var wave = sin(x * freq + t * speed + phase) * amp;
    wave += sin(x * freq * 1.7 + t * speed * 0.6 + phase * 2.3) * amp * 0.5;
    wave += sin(x * freq * 0.4 + t * speed * 1.3 + phase * 0.7) * amp * 0.3;

    // Add noise for natural variation
    wave += aurora_noise1d(x * 2.0 + t * speed * 0.3) * amp * 0.4;

    return wave;
}

fn aurora_material(input: AuroraInput, params: AuroraParams) -> AuroraOutput {
    var out: AuroraOutput;
    out.discard_pixel = false;

    let uv = input.uv;
    let t = input.time * params.curtain_speed;

    // Vertical position (aurora appears in upper portion)
    let y = uv.y;
    let x = uv.x;

    // Vertical fade: aurora strongest in middle-upper region, fades at edges
    let vert_mask = smoothstep(0.0, 0.3, y) * smoothstep(1.0, 0.6, y);
    if vert_mask <= 0.01 {
        out.discard_pixel = true;
        return out;
    }

    // Multiple aurora curtain layers
    var aurora_intensity = 0.0;
    var color_mix = 0.0;

    // Curtain 1: Primary band
    let band1_center = 0.5 + aurora_band(x, t, 3.0, 0.0, 1.0, params.wave_amplitude);
    let band1_dist = abs(y - band1_center);
    let band1_width = 0.12 + aurora_noise1d(x * 3.0 + t * 0.4) * 0.06;
    let band1 = smoothstep(band1_width, 0.0, band1_dist);
    aurora_intensity += band1 * 0.6;
    color_mix += band1 * 0.3;

    // Curtain 2: Secondary band (offset)
    let band2_center = 0.45 + aurora_band(x, t, 4.5, 2.1, 0.7, params.wave_amplitude * 0.8);
    let band2_dist = abs(y - band2_center);
    let band2_width = 0.08 + aurora_noise1d(x * 4.0 + t * 0.6 + 3.0) * 0.05;
    let band2 = smoothstep(band2_width, 0.0, band2_dist);
    aurora_intensity += band2 * 0.4;
    color_mix += band2 * 0.7;

    // Curtain 3: Thin bright accent
    let band3_center = 0.55 + aurora_band(x, t, 6.0, 4.5, 1.3, params.wave_amplitude * 0.6);
    let band3_dist = abs(y - band3_center);
    let band3_width = 0.03 + aurora_noise1d(x * 5.0 + t * 0.8 + 7.0) * 0.02;
    let band3 = smoothstep(band3_width, 0.0, band3_dist);
    aurora_intensity += band3 * 0.8;
    color_mix += band3 * 0.5;

    // Curtain 4: Diffuse glow
    let band4_center = 0.48 + aurora_band(x, t, 2.0, 1.3, 0.5, params.wave_amplitude * 1.2);
    let band4_dist = abs(y - band4_center);
    let band4_width = 0.2 + aurora_noise1d(x * 2.0 + t * 0.2 + 5.0) * 0.1;
    let band4 = smoothstep(band4_width, 0.0, band4_dist);
    aurora_intensity += band4 * 0.2;
    color_mix += band4 * 0.9;

    if aurora_intensity <= 0.01 {
        out.discard_pixel = true;
        return out;
    }

    // Normalize color mix
    color_mix = clamp(color_mix / max(aurora_intensity, 0.001), 0.0, 1.0);

    // Add shimmer/scintillation
    let shimmer = aurora_noise2d(vec2<f32>(x * 20.0 + t * 2.0, y * 30.0 + t * 0.5));
    aurora_intensity *= 0.85 + shimmer * 0.15;

    // Vertical ray structure (vertical streaks)
    let ray = 0.7 + 0.3 * sin(x * 40.0 + t * 0.3) * sin(x * 17.0 - t * 0.7);
    aurora_intensity *= ray;

    // Color interpolation between primary and secondary
    let col_primary = vec3<f32>(params.color_primary.x, params.color_primary.y, params.color_primary.z);
    let col_secondary = vec3<f32>(params.color_secondary.x, params.color_secondary.y, params.color_secondary.z);

    // Add position-based color variation
    let pos_color_shift = sin(x * 5.0 + t * 0.3) * 0.3 + 0.5;
    let final_color_t = mix(color_mix, pos_color_shift, 0.4);
    var aurora_color = mix(col_primary, col_secondary, final_color_t);

    // Brightness boost at peaks
    let peak_boost = pow(aurora_intensity, 0.5) * 0.3;
    aurora_color += vec3<f32>(peak_boost);

    // Apply overall brightness
    aurora_color *= params.brightness;

    // Soft glow: exponential falloff makes it feel luminous
    let glow_intensity = pow(aurora_intensity, 0.7);

    // Edge fade (horizontal)
    let horiz_fade = smoothstep(0.0, 0.1, x) * smoothstep(1.0, 0.9, x);

    let alpha = clamp(glow_intensity * vert_mask * horiz_fade, 0.0, 1.0);

    out.color = vec4<f32>(aurora_color * glow_intensity, alpha);
    return out;
}
