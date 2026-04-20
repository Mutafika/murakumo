// Murakumo — Lightning Material
// 電撃アークシェーダー
//
// ミッドポイントディスプレイスメントの近似で
// 分岐する電撃を手続き的に描画する。

struct LightningParams {
    color: vec4<f32>,
    arc_count: f32,
    thickness: f32,
    branch_probability: f32,
    intensity: f32,
    speed: f32,
}

struct LightningInput {
    uv: vec2<f32>,
    world_pos: vec3<f32>,
    eye_pos: vec3<f32>,
    time: f32,
}

struct LightningOutput {
    color: vec4<f32>,
    discard_pixel: bool,
}

// ハッシュ関数群
fn hash11(p: f32) -> f32 {
    var pp = fract(p * 0.1031);
    pp = pp * (pp + 33.33);
    pp = pp * (pp + pp);
    return fract(pp);
}

fn hash12(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 = p3 + dot(p3, vec3<f32>(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33));
    return fract((p3.x + p3.y) * p3.z);
}

fn hash13(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// ミッドポイントディスプレイスメントで電撃パスを近似
// yに沿ったxオフセットを返す (0-1区間を細分化)
fn lightning_offset(y: f32, seed: f32, time_seed: f32) -> f32 {
    // 複数周波数のノイズを重ねてジグザグを作る
    var offset = 0.0;
    var amp = 0.2;
    var freq = 2.0;

    for (var i = 0; i < 6; i = i + 1) {
        let phase = hash11(seed * 7.13 + f32(i) * 3.7 + time_seed) * 6.28318;
        offset = offset + sin(y * freq * 3.14159 + phase) * amp;
        amp = amp * 0.6;
        freq = freq * 1.9;
    }

    // ジッター (高周波のランダムオフセット)
    let jitter_seed = floor(y * 16.0);
    let jitter = (hash11(jitter_seed * 13.7 + seed + time_seed) - 0.5) * 0.06;

    return offset + jitter;
}

// アーク1本の輝度を計算 (点pにおける)
fn arc_brightness(
    p: vec2<f32>,
    start_y: f32,
    end_y: f32,
    seed: f32,
    time_seed: f32,
    thickness: f32,
    x_center: f32
) -> f32 {
    let y_norm = (p.y - start_y) / (end_y - start_y);
    if y_norm < 0.0 || y_norm > 1.0 {
        return 0.0;
    }

    let arc_x = x_center + lightning_offset(y_norm, seed, time_seed);
    let dist = abs(p.x - arc_x);

    // コア (非常に明るい細い線)
    let core = exp(-dist * dist / (thickness * thickness * 0.3)) * 1.5;
    // グロー (柔らかく広がる光)
    let glow = exp(-dist * dist / (thickness * thickness * 4.0)) * 0.6;
    // 外側のかすかなグロー
    let outer = exp(-dist * dist / (thickness * thickness * 16.0)) * 0.2;

    // 端でフェードアウト
    let end_fade = smoothstep(0.0, 0.05, y_norm) * smoothstep(1.0, 0.95, y_norm);

    return (core + glow + outer) * end_fade;
}

fn lightning_material(input: LightningInput, params: LightningParams) -> LightningOutput {
    var out: LightningOutput;
    out.discard_pixel = false;

    let uv = input.uv;
    let t = input.time * params.speed;

    // フレームごとに異なる乱数シード (電撃を毎フレーム変える)
    let time_seed = floor(t * 4.0) * 0.1;

    let base_color = vec3<f32>(params.color[0], params.color[1], params.color[2]);
    var total_brightness = 0.0;

    let arc_n = i32(params.arc_count);

    // ── メインアーク描画 ──
    for (var i = 0; i < 8; i = i + 1) {
        if i >= arc_n {
            break;
        }

        let arc_seed = f32(i) * 17.31 + 5.7;
        let x_start = 0.3 + hash11(arc_seed + 0.1) * 0.4;

        // メインアーク
        let main_bright = arc_brightness(
            uv,
            0.05,
            0.95,
            arc_seed,
            time_seed,
            params.thickness,
            x_start
        );
        total_brightness = total_brightness + main_bright;

        // ── 分岐 ──
        for (var b = 0; b < 4; b = b + 1) {
            let branch_seed = arc_seed + f32(b) * 23.17;
            let branch_check = hash11(branch_seed + time_seed * 3.0);

            if branch_check < params.branch_probability {
                // 分岐の開始位置 (メインアーク上のy位置)
                let branch_y = 0.2 + hash11(branch_seed + 1.0 + time_seed) * 0.6;
                let branch_x = x_start + lightning_offset(branch_y, arc_seed, time_seed);

                // 分岐の方向と長さ
                let branch_dir = (hash11(branch_seed + 2.0 + time_seed) - 0.5) * 0.3;
                let branch_len = 0.1 + hash11(branch_seed + 3.0) * 0.2;

                // 分岐アーク (簡易: 方向をずらしたサブアーク)
                let branch_end_y = branch_y + branch_len;
                let branch_x_offset = branch_x + branch_dir;

                let bp = vec2<f32>(uv.x - branch_dir * (uv.y - branch_y) / branch_len, uv.y);
                let branch_bright = arc_brightness(
                    bp,
                    branch_y,
                    min(branch_end_y, 0.95),
                    branch_seed * 7.3,
                    time_seed,
                    params.thickness * 0.6,
                    branch_x
                ) * 0.5;
                total_brightness = total_brightness + branch_bright;
            }
        }
    }

    // ── 閾値判定 ──
    if total_brightness < 0.01 {
        out.discard_pixel = true;
        return out;
    }

    // ── 色の合成 ──
    // コアは白っぽく、外側ほど指定色
    let white_mix = smoothstep(0.5, 2.0, total_brightness);
    let core_color = mix(base_color, vec3<f32>(1.0, 1.0, 1.0), white_mix);
    let final_color = core_color * total_brightness * params.intensity;

    // フリッカー (明滅)
    let flicker = 0.8 + 0.2 * sin(t * 30.0) * sin(t * 17.0 + 1.3);

    let alpha = clamp(total_brightness * params.intensity * flicker, 0.0, 1.0);
    out.color = vec4<f32>(final_color * flicker, alpha);
    return out;
}
