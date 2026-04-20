// Murakumo — Dissolve Material
// ノイズディゾルブシェーダー
//
// FBMノイズを使った溶解エフェクト。
// 溶解境界に明るいグロー、方向バイアス対応。

struct DissolveParams {
    progress: f32,
    _pad0: vec3<f32>,
    edge_color: vec4<f32>,
    edge_width: f32,
    noise_scale: f32,
    direction: vec2<f32>,
}

struct DissolveInput {
    uv: vec2<f32>,
    world_pos: vec3<f32>,
    eye_pos: vec3<f32>,
    time: f32,
}

struct DissolveOutput {
    color: vec4<f32>,
    discard_pixel: bool,
}

// 値ノイズ用ハッシュ
fn hash_f(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.13);
    p3 = p3 + dot(p3, vec3<f32>(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33));
    return fract((p3.x + p3.y) * p3.z);
}

// バリューノイズ
fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let a = hash_f(i + vec2<f32>(0.0, 0.0));
    let b = hash_f(i + vec2<f32>(1.0, 0.0));
    let c = hash_f(i + vec2<f32>(0.0, 1.0));
    let d = hash_f(i + vec2<f32>(1.0, 1.0));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// FBMノイズ (5オクターブ)
fn fbm(p: vec2<f32>) -> f32 {
    var val = 0.0;
    var amp = 0.5;
    var freq = 1.0;
    var pp = p;
    for (var i = 0; i < 5; i = i + 1) {
        val = val + amp * value_noise(pp * freq);
        freq = freq * 2.0;
        amp = amp * 0.5;
        pp = pp + vec2<f32>(1.3, 1.7);
    }
    return val;
}

// ドメインワープFBM (より有機的なパターン)
fn warped_fbm(p: vec2<f32>, t: f32) -> f32 {
    let q = vec2<f32>(
        fbm(p + vec2<f32>(0.0, 0.0)),
        fbm(p + vec2<f32>(5.2, 1.3))
    );
    let r = vec2<f32>(
        fbm(p + 4.0 * q + vec2<f32>(1.7, 9.2) + t * 0.1),
        fbm(p + 4.0 * q + vec2<f32>(8.3, 2.8) + t * 0.12)
    );
    return fbm(p + 4.0 * r);
}

fn dissolve_material(input: DissolveInput, params: DissolveParams) -> DissolveOutput {
    var out: DissolveOutput;
    out.discard_pixel = false;

    let uv = input.uv;
    let t = input.time;
    let progress = params.progress;

    // 進行度0は完全表示
    if progress < 0.001 {
        let base_color = vec3<f32>(0.8, 0.85, 0.9);
        out.color = vec4<f32>(base_color, 1.0);
        return out;
    }

    // 進行度1は完全消失
    if progress > 0.999 {
        out.discard_pixel = true;
        return out;
    }

    // ── ノイズ生成 ──
    let noise_uv = uv * params.noise_scale;
    let noise_val = warped_fbm(noise_uv, t);

    // ── 方向バイアス ──
    let dir = normalize(params.direction);
    let dir_bias = dot(uv - 0.5, dir) * 0.5 + 0.5; // 0-1
    let biased_noise = noise_val * 0.6 + dir_bias * 0.4;

    // ── 溶解閾値 ──
    let threshold = progress;
    let dist_to_edge = biased_noise - threshold;

    // 溶解済み領域
    if dist_to_edge < 0.0 {
        out.discard_pixel = true;
        return out;
    }

    // ── エッジグロー ──
    let edge_zone = smoothstep(0.0, params.edge_width, dist_to_edge);
    let edge_intensity = 1.0 - edge_zone;

    // エッジのカラー (中心ほど明るい)
    let edge_col = vec3<f32>(params.edge_color[0], params.edge_color[1], params.edge_color[2]);
    let hot_core = vec3<f32>(1.0, 1.0, 0.8); // 白熱色
    let glow_color = mix(hot_core, edge_col, smoothstep(0.0, 0.6, edge_intensity * params.edge_width * 20.0));
    let glow_brightness = edge_intensity * (2.0 + sin(t * 10.0 + noise_val * 20.0) * 0.3);

    // ── ベースカラー (残存サーフェス) ──
    let base_color = vec3<f32>(0.8, 0.85, 0.9);

    // パターンによる表面ディテール
    let surface_detail = value_noise(uv * 20.0 + t * 0.5) * 0.1;
    let surface = base_color + surface_detail;

    // ── エンバー粒子 (エッジ付近に小さな火花) ──
    let ember_noise = value_noise(uv * 40.0 + vec2<f32>(t * 2.0, t * 1.5));
    let ember = smoothstep(0.85, 0.95, ember_noise) * edge_intensity * 1.5;

    // ── 合成 ──
    let final_color = mix(surface, glow_color * glow_brightness, edge_intensity) + edge_col * ember;
    let alpha = 1.0; // 残存部分は不透明

    out.color = vec4<f32>(final_color, alpha);
    return out;
}
