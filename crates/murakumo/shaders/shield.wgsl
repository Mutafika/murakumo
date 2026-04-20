// Murakumo — Shield Material
// エネルギーシールドシェーダー
//
// Billboard quad上でレイマーチした球体に
// 六角グリッドのエネルギーシールドを描画する。

struct ShieldParams {
    color: vec4<f32>,
    hex_scale: f32,
    hit_point: vec2<f32>,
    hit_time: f32,
    opacity: f32,
    pulse_speed: f32,
}

struct ShieldInput {
    uv: vec2<f32>,
    world_pos: vec3<f32>,
    eye_pos: vec3<f32>,
    time: f32,
}

struct ShieldOutput {
    color: vec4<f32>,
    discard_pixel: bool,
}

// 六角グリッド距離関数
// pを六角座標に変換し、最寄りのエッジまでの距離を返す
fn hex_distance(p: vec2<f32>, scale: f32) -> vec2<f32> {
    let s = vec2<f32>(1.0, 1.7320508); // 1, sqrt(3)
    let ps = p * scale;

    let half_s = s * 0.5;
    let a = ps - s * floor(ps / s + 0.5);
    let b = ps - half_s - s * floor((ps - half_s) / s + 0.5);

    let da = abs(a);
    let db = abs(b);
    let d_a = max(da.x * 1.5 + da.y * s.y, da.y * s.y * 2.0);
    let d_b = max(db.x * 1.5 + db.y * s.y, db.y * s.y * 2.0);

    // エッジまでの距離と、セル中心からの距離
    let edge_dist = min(d_a, d_b);
    return vec2<f32>(edge_dist, min(length(a), length(b)));
}

// ハッシュ関数
fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 = p3 + dot(p3, vec3<f32>(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33));
    return fract((p3.x + p3.y) * p3.z);
}

fn shield_material(input: ShieldInput, params: ShieldParams) -> ShieldOutput {
    var out: ShieldOutput;
    out.discard_pixel = false;

    let uv_cent = input.uv - 0.5;
    let r2 = dot(uv_cent, uv_cent) * 4.0;
    if r2 > 1.0 {
        out.discard_pixel = true;
        return out;
    }

    let t = input.time;
    let nz = sqrt(1.0 - r2);
    let sphere_n = normalize(vec3<f32>(uv_cent.x * 2.0, uv_cent.y * 2.0, nz));
    let view_dir = normalize(input.eye_pos - input.world_pos);
    let n_dot_v = max(dot(sphere_n, view_dir), 0.0);
    let r = sqrt(r2);

    // ── 球面UV (equirectangular) ──
    let sphere_uv = vec2<f32>(
        atan2(sphere_n.x, sphere_n.z) / 6.28318 + 0.5,
        sphere_n.y * 0.5 + 0.5
    );

    // ── 六角グリッド ──
    let hex = hex_distance(sphere_uv, params.hex_scale);
    let hex_edge = hex.x;
    let hex_cell = hex.y;

    // エッジのシャープネス
    let edge_glow = 1.0 - smoothstep(0.42, 0.5, hex_edge);

    // ── パルスアニメーション ──
    let pulse = sin(t * params.pulse_speed + sphere_uv.y * 12.0) * 0.5 + 0.5;
    let pulse_glow = edge_glow * (0.3 + 0.7 * pulse);

    // ── フレネルグロー ──
    let fresnel = pow(1.0 - n_dot_v, 3.0);
    let fresnel_glow = fresnel * 1.5;

    // ── ヒットエフェクト ──
    var hit_effect = 0.0;
    let time_since_hit = t - params.hit_time;
    if time_since_hit > 0.0 && time_since_hit < 2.0 {
        let hit_uv = params.hit_point - 0.5;
        let hit_dist = length(uv_cent - hit_uv);
        let ripple_radius = time_since_hit * 0.6;
        let ripple_width = 0.08;
        let ripple = 1.0 - smoothstep(0.0, ripple_width, abs(hit_dist - ripple_radius));
        let fade = 1.0 - time_since_hit / 2.0;
        hit_effect = ripple * fade * 3.0;

        // 第2波
        let ripple_radius2 = time_since_hit * 0.4 - 0.1;
        if ripple_radius2 > 0.0 {
            let ripple2 = 1.0 - smoothstep(0.0, ripple_width * 0.7, abs(hit_dist - ripple_radius2));
            hit_effect = hit_effect + ripple2 * fade * 1.5;
        }
    }

    // ── セル内ランダム明滅 ──
    let cell_id = floor(sphere_uv * params.hex_scale);
    let cell_hash = hash21(cell_id);
    let cell_flicker = smoothstep(0.7, 1.0, sin(t * 3.0 + cell_hash * 6.28318) * 0.5 + 0.5) * 0.3;

    // ── 合成 ──
    let base_color = vec3<f32>(params.color[0], params.color[1], params.color[2]);
    let energy = pulse_glow + fresnel_glow + hit_effect + cell_flicker;
    let final_color = base_color * energy;
    let alpha = (edge_glow * 0.4 + fresnel * 0.5 + hit_effect * 0.6 + cell_flicker) * params.opacity;

    out.color = vec4<f32>(final_color, clamp(alpha, 0.0, 1.0));
    return out;
}
