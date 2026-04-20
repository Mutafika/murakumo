// Murakumo — Hologram Material
// ホログラムシェーダー
//
// スキャンライン、グリッチ、色収差、フリッカーで
// サイバーパンク風ホログラムを描画する。

struct HologramParams {
    base_color: vec4<f32>,
    scan_line_speed: f32,
    scan_line_density: f32,
    glitch_intensity: f32,
    flicker_speed: f32,
}

struct HologramInput {
    uv: vec2<f32>,
    world_pos: vec3<f32>,
    eye_pos: vec3<f32>,
    time: f32,
}

struct HologramOutput {
    color: vec4<f32>,
    discard_pixel: bool,
}

// Pseudo-random hash
fn hash_holo(p: f32) -> f32 {
    let x = fract(sin(p * 127.1) * 43758.5453);
    return x;
}

fn hash2_holo(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

fn hologram_material(input: HologramInput, params: HologramParams) -> HologramOutput {
    var out: HologramOutput;
    out.discard_pixel = false;

    let t = input.time;
    var uv = input.uv;

    // ── Glitch: horizontal UV offset ──
    let glitch_block = floor(uv.y * 20.0 + t * 3.0);
    let glitch_rand = hash_holo(glitch_block + floor(t * 7.0));
    let glitch_active = step(1.0 - params.glitch_intensity * 0.3, glitch_rand);
    let glitch_offset = (hash_holo(glitch_block * 3.7 + t) - 0.5) * 0.15 * params.glitch_intensity * glitch_active;
    uv.x = uv.x + glitch_offset;

    // Wrap UV
    uv.x = fract(uv.x);

    // ── Scan lines (horizontal, scrolling vertically) ──
    let scan_y = uv.y * params.scan_line_density + t * params.scan_line_speed;
    let scan_line = sin(scan_y * 3.14159) * 0.5 + 0.5;
    let scan_mask = smoothstep(0.3, 0.7, scan_line);

    // Thicker scan bar that scrolls
    let bar_pos = fract(t * params.scan_line_speed * 0.1);
    let bar_dist = abs(uv.y - bar_pos);
    let bar_wrap = min(bar_dist, 1.0 - bar_dist);
    let scan_bar = smoothstep(0.05, 0.0, bar_wrap) * 0.5;

    // ── Color aberration (RGB split) ──
    let aberration = 0.008 * params.glitch_intensity;
    let uv_r = vec2<f32>(uv.x + aberration, uv.y);
    let uv_b = vec2<f32>(uv.x - aberration, uv.y);

    // Base color with aberration
    let col_r = params.base_color.r * (1.0 + (uv_r.x - uv.x) * 10.0);
    let col_g = params.base_color.g;
    let col_b = params.base_color.b * (1.0 + (uv_b.x - uv.x) * 10.0);

    var color = vec3<f32>(col_r, col_g, col_b);

    // Apply scan line modulation
    color = color * (0.6 + 0.4 * scan_mask);
    color = color + vec3<f32>(scan_bar) * params.base_color.rgb;

    // ── Edge glow (fresnel-like using UV distance from center) ──
    let uv_cent = uv - 0.5;
    let edge_dist = length(uv_cent) * 2.0;
    let edge_glow = pow(edge_dist, 2.0) * 0.5;
    color = color + params.base_color.rgb * edge_glow;

    // ── Flicker pulses ──
    let flicker_phase = t * params.flicker_speed;
    let flicker = 1.0 - 0.15 * (sin(flicker_phase) * sin(flicker_phase * 2.7) + 1.0) * 0.5;
    // Occasional bright flash
    let flash_trigger = step(0.97, hash_holo(floor(t * 4.0)));
    let flash = 1.0 + flash_trigger * 2.0;
    color = color * flicker * flash;

    // ── Horizontal interference lines (fine detail) ──
    let fine_lines = sin(uv.y * 400.0 + t * 10.0) * 0.03;
    color = color + vec3<f32>(fine_lines) * params.base_color.rgb;

    // ── Alpha: modulated by scan lines and overall transparency ──
    let alpha = params.base_color.a * (0.5 + 0.5 * scan_mask) * flicker;

    // ── Glitch color corruption ──
    if glitch_active > 0.5 {
        let corrupt = hash2_holo(vec2<f32>(uv.x * 10.0, glitch_block));
        color = mix(color, vec3<f32>(corrupt, 1.0 - corrupt, corrupt * 0.5), 0.3 * params.glitch_intensity);
    }

    // Clamp
    color = clamp(color, vec3<f32>(0.0), vec3<f32>(5.0));

    out.color = vec4<f32>(color, alpha);
    return out;
}
