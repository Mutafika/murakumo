// Murakumo — Neon Glow Material
// ネオングローシェーダー
//
// 明るいコアからのフォールオフグロー、
// パルスアニメーション、加算発光を表現する。

struct NeonParams {
    glow_color: vec4<f32>,
    core_brightness: f32,
    glow_radius: f32,
    pulse_speed: f32,
    pulse_amount: f32,
}

struct NeonInput {
    uv: vec2<f32>,
    world_pos: vec3<f32>,
    eye_pos: vec3<f32>,
    time: f32,
}

struct NeonOutput {
    color: vec4<f32>,
    discard_pixel: bool,
}

fn hash_neon(p: f32) -> f32 {
    return fract(sin(p * 127.1) * 43758.5453);
}

fn neon_material(input: NeonInput, params: NeonParams) -> NeonOutput {
    var out: NeonOutput;
    out.discard_pixel = false;

    let t = input.time;
    let uv = input.uv;
    let uv_cent = uv - 0.5;

    // ── Pulse modulation ──
    let pulse_base = sin(t * params.pulse_speed) * 0.5 + 0.5;
    let pulse_sharp = pow(pulse_base, 2.0);
    let pulse = 1.0 - params.pulse_amount + params.pulse_amount * pulse_sharp;

    // Secondary pulse (irregular feel)
    let pulse2 = 1.0 + 0.1 * sin(t * params.pulse_speed * 2.7 + 1.3);

    let total_pulse = pulse * pulse2;

    // ── Core shape: circular ring ──
    let dist_center = length(uv_cent) * 2.0;

    // Neon tube: ring shape
    let ring_radius = 0.6;
    let ring_dist = abs(dist_center - ring_radius);

    // Core line (very bright, thin)
    let core_width = 0.02;
    let core = smoothstep(core_width, 0.0, ring_dist);
    let core_bright = core * params.core_brightness * total_pulse;

    // ── Glow falloff (multiple layers for realistic bloom) ──
    let glow1_radius = params.glow_radius;
    let glow1 = exp(-ring_dist * ring_dist / (glow1_radius * glow1_radius * 0.5 + 0.001));

    let glow2_radius = params.glow_radius * 2.0;
    let glow2 = exp(-ring_dist * ring_dist / (glow2_radius * glow2_radius * 0.5 + 0.001)) * 0.4;

    let glow3_radius = params.glow_radius * 4.0;
    let glow3 = exp(-ring_dist * ring_dist / (glow3_radius * glow3_radius * 0.5 + 0.001)) * 0.15;

    let total_glow = (glow1 + glow2 + glow3) * total_pulse;

    // ── Also add a vertical line accent ──
    let line_dist = abs(uv_cent.x);
    let line_core = smoothstep(0.01, 0.0, line_dist) * smoothstep(0.4, 0.2, abs(uv_cent.y));
    let line_glow = exp(-line_dist * line_dist / (params.glow_radius * params.glow_radius * 0.3 + 0.001))
                  * smoothstep(0.5, 0.1, abs(uv_cent.y)) * 0.3;
    let line_total = (line_core * params.core_brightness * 0.5 + line_glow) * total_pulse;

    // ── Color: core is white-hot, glow is colored ──
    let white_core = vec3<f32>(1.0, 1.0, 1.0);
    let glow_col = params.glow_color.rgb;

    // Core desaturates toward white at high brightness
    let core_color = mix(glow_col, white_core, min(core_bright / params.core_brightness, 1.0));

    var color = core_color * core_bright;
    color = color + glow_col * total_glow;
    color = color + glow_col * line_total;

    // ── Color bleeding (slight hue shift in outer glow) ──
    let bleed_hue = vec3<f32>(
        glow_col.r * 1.1,
        glow_col.g * 0.9,
        glow_col.b * 1.2
    );
    color = color + bleed_hue * glow3 * 0.3 * total_pulse;

    // ── Flickering noise (subtle) ──
    let flicker_noise = hash_neon(floor(t * 30.0)) * 0.05 + 0.975;
    color = color * flicker_noise;

    // ── Additive alpha: proportional to brightness ──
    let luminance = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    let alpha = clamp(luminance * params.glow_color.a, 0.0, 1.0);

    // Discard fully dark pixels
    if luminance < 0.001 {
        out.discard_pixel = true;
        return out;
    }

    out.color = vec4<f32>(color, alpha);
    return out;
}
