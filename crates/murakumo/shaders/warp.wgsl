// Murakumo — Warp Material
// 空間歪みシェーダー
//
// Billboard quad上でスパイラルツイストと色収差を伴う
// 空間歪みエフェクトを描画する。

struct WarpParams {
    center: vec2<f32>,
    strength: f32,
    radius: f32,
    rotation_speed: f32,
    _pad0: vec3<f32>,
    distortion_color: vec4<f32>,
}

struct WarpInput {
    uv: vec2<f32>,
    world_pos: vec3<f32>,
    eye_pos: vec3<f32>,
    time: f32,
}

struct WarpOutput {
    color: vec4<f32>,
    discard_pixel: bool,
}

// 2Dノイズ用ハッシュ
fn hash2(p: vec2<f32>) -> vec2<f32> {
    let k = vec2<f32>(0.3183099, 0.3678794);
    let pp = p * k + vec2<f32>(k.y, k.x);
    return -1.0 + 2.0 * fract(16.0 * k * fract(pp.x * pp.y * (pp.x + pp.y)));
}

// グラディエントノイズ
fn gradient_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let n00 = dot(hash2(i + vec2<f32>(0.0, 0.0)), f - vec2<f32>(0.0, 0.0));
    let n10 = dot(hash2(i + vec2<f32>(1.0, 0.0)), f - vec2<f32>(1.0, 0.0));
    let n01 = dot(hash2(i + vec2<f32>(0.0, 1.0)), f - vec2<f32>(0.0, 1.0));
    let n11 = dot(hash2(i + vec2<f32>(1.0, 1.0)), f - vec2<f32>(1.0, 1.0));

    return mix(mix(n00, n10, u.x), mix(n01, n11, u.x), u.y);
}

// FBMノイズ
fn fbm_noise(p: vec2<f32>) -> f32 {
    var val = 0.0;
    var amp = 0.5;
    var pp = p;
    for (var i = 0; i < 4; i = i + 1) {
        val = val + amp * gradient_noise(pp);
        pp = pp * 2.0 + vec2<f32>(1.7, 9.2);
        amp = amp * 0.5;
    }
    return val;
}

fn warp_material(input: WarpInput, params: WarpParams) -> WarpOutput {
    var out: WarpOutput;
    out.discard_pixel = false;

    let t = input.time;
    let uv = input.uv;
    let center = params.center;
    let to_center = uv - center;
    let dist = length(to_center);
    let angle = atan2(to_center.y, to_center.x);

    // 半径外はフェードアウト
    let radius_mask = 1.0 - smoothstep(params.radius * 0.5, params.radius, dist);
    if radius_mask < 0.001 {
        out.discard_pixel = true;
        return out;
    }

    // ── スパイラルツイスト ──
    let twist_angle = params.strength * (1.0 - dist / params.radius) * 6.28318;
    let rotation = t * params.rotation_speed;
    let spiral_angle = angle + twist_angle + rotation;

    // 歪み座標
    let warped_dist = dist + sin(spiral_angle * 3.0 + t) * 0.02 * params.strength;
    let warped_uv = center + vec2<f32>(cos(spiral_angle), sin(spiral_angle)) * warped_dist;

    // ── スワーリングノイズ ──
    let noise_uv = warped_uv * 4.0 + vec2<f32>(t * 0.3, t * 0.2);
    let swirl = fbm_noise(noise_uv) * params.strength;

    // ── トンネル深度イリュージョン ──
    let depth_rings = sin(dist * 30.0 - t * 4.0) * 0.5 + 0.5;
    let tunnel = depth_rings * (1.0 - dist / params.radius);

    // ── 色収差 ──
    let aberration_strength = params.strength * 0.03 * (1.0 - dist / params.radius);
    let dir_to_center = normalize(to_center);

    let uv_r = uv + dir_to_center * aberration_strength;
    let uv_g = uv;
    let uv_b = uv - dir_to_center * aberration_strength;

    // 各チャンネルのスパイラルパターン
    let spiral_r = sin((length(uv_r - center) * 20.0 - t * 3.0) + spiral_angle * 2.0) * 0.5 + 0.5;
    let spiral_g = sin((length(uv_g - center) * 20.0 - t * 3.0) + spiral_angle * 2.0 + 2.094) * 0.5 + 0.5;
    let spiral_b = sin((length(uv_b - center) * 20.0 - t * 3.0) + spiral_angle * 2.0 + 4.189) * 0.5 + 0.5;

    let chromatic = vec3<f32>(spiral_r, spiral_g, spiral_b);

    // ── エッジグロー ──
    let edge_dist = abs(dist - params.radius * 0.7);
    let edge_glow = exp(-edge_dist * 15.0) * 0.5;

    // ── 中心のブラックホール ──
    let core_dark = smoothstep(0.05, 0.15, dist);
    let core_bright = exp(-dist * 20.0) * 2.0;

    // ── 合成 ──
    let dist_color = vec3<f32>(params.distortion_color[0], params.distortion_color[1], params.distortion_color[2]);
    let base = chromatic * dist_color * (tunnel * 0.5 + 0.5);
    let noise_tint = vec3<f32>(swirl * 0.3 + 0.7) * dist_color;
    let final_color = (base * 0.6 + noise_tint * 0.4) * core_dark + vec3<f32>(core_bright) * dist_color;
    let final_with_edge = final_color + dist_color * edge_glow;

    let alpha = radius_mask * (0.4 + tunnel * 0.3 + edge_glow + core_bright * 0.5);

    out.color = vec4<f32>(final_with_edge, clamp(alpha, 0.0, 1.0));
    return out;
}
