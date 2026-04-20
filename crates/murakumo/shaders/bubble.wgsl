// Murakumo — Bubble Material
// 薄膜干渉シャボン玉シェーダー
//
// Billboard quad上でレイマーチした球体に
// 前面+背面の2層干渉パターンを描画する。

struct BubbleParams {
    thickness_base: f32,
    gravity_strength: f32,
    band_threshold: f32,
    shell_alpha: f32,
    pop_progress: f32,
}

struct BubbleInput {
    uv: vec2<f32>,           // tex_uv (0-1)
    world_pos: vec3<f32>,
    eye_pos: vec3<f32>,
    time: f32,
}

struct BubbleOutput {
    color: vec4<f32>,
    discard_pixel: bool,
}

fn bubble_material(input: BubbleInput, params: BubbleParams) -> BubbleOutput {
    var out: BubbleOutput;
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

    // ── Front surface ──
    let p = uv_cent * 2.0;
    let thick_f = params.thickness_base
        + p.y * params.gravity_strength
        + sin(p.x * 3.0 + p.y * 2.0 + t * 0.4) * 80.0
        + sin(p.y * 7.0 + t * 0.7) * 40.0
        + sin(p.x * 5.0 - p.y * 4.0 + t * 0.2) * 30.0;
    let pd_f = 2.0 * 1.33 * thick_f;
    let irid_f = vec3<f32>(
        0.5 + 0.5 * cos(6.28318 * pd_f / 700.0),
        0.5 + 0.5 * cos(6.28318 * pd_f / 530.0),
        0.5 + 0.5 * cos(6.28318 * pd_f / 430.0),
    );

    // ── Back surface ──
    let thick_b = thick_f + 150.0
        + sin(p.x * 4.0 - p.y * 3.0 - t * 0.3) * 50.0;
    let pd_b = 2.0 * 1.33 * thick_b;
    let irid_b = vec3<f32>(
        0.5 + 0.5 * cos(6.28318 * pd_b / 700.0),
        0.5 + 0.5 * cos(6.28318 * pd_b / 530.0),
        0.5 + 0.5 * cos(6.28318 * pd_b / 430.0),
    );

    let fresnel = pow(1.0 - n_dot_v, 3.0);

    let range_f = max(max(irid_f.x, irid_f.y), irid_f.z) - min(min(irid_f.x, irid_f.y), irid_f.z);
    let mask_f = smoothstep(params.band_threshold, 0.95, range_f) * (0.005 + fresnel * 0.025);

    let range_b = max(max(irid_b.x, irid_b.y), irid_b.z) - min(min(irid_b.x, irid_b.y), irid_b.z);
    let mask_b = smoothstep(params.band_threshold, 0.95, range_b) * (0.003 + fresnel * 0.012);

    // Specular
    let light = normalize(vec3<f32>(0.3, 1.0, 0.6));
    let h = normalize(light + view_dir);
    let spec_f = pow(max(dot(sphere_n, h), 0.0), 256.0) * 0.7;
    let spec_b = pow(max(dot(-sphere_n, h), 0.0), 128.0) * 0.1;

    // Common
    let sheen = fresnel * vec3<f32>(0.2, 0.25, 0.35) * 0.08;
    let light2 = normalize(vec3<f32>(-0.6, 0.5, 0.3));
    let h2 = normalize(light2 + view_dir);
    let spec2 = pow(max(dot(sphere_n, h2), 0.0), 180.0) * 0.3;
    let bands = irid_f * mask_f + irid_b * mask_b;
    let band_lum = max(bands.x, max(bands.y, bands.z));
    let shell_color = vec3<f32>(0.15, 0.18, 0.25) * params.shell_alpha;

    let pop = params.pop_progress;

    // ── Pop ──
    if pop > 0.01 {
        if pop > 0.35 {
            // Expanding ripple ring
            let ring_t = (pop - 0.35) / 0.65;
            let ring_r = ring_t * 1.5;
            let ring_width = 0.04 * (1.0 - ring_t);
            let ring_dist = abs(r - ring_r);
            if ring_dist > ring_width || ring_t > 0.8 {
                out.discard_pixel = true;
                return out;
            }
            let ring_bright = (1.0 - ring_t) * smoothstep(ring_width, 0.0, ring_dist);
            out.color = vec4<f32>(irid_f * ring_bright * 0.4, ring_bright * 0.3);
            return out;
        }

        // Wobble
        let w = pop / 0.35;
        let wobble_boost = sin(uv_cent.x * 30.0 + t * 20.0) * cos(uv_cent.y * 25.0 + t * 15.0) * w * w * 150.0;
        let thick_fw = thick_f + wobble_boost;
        let pd_fw = 2.0 * 1.33 * thick_fw;
        let irid_fw = vec3<f32>(
            0.5 + 0.5 * cos(6.28318 * pd_fw / 700.0),
            0.5 + 0.5 * cos(6.28318 * pd_fw / 530.0),
            0.5 + 0.5 * cos(6.28318 * pd_fw / 430.0),
        );
        let range_fw = max(max(irid_fw.x, irid_fw.y), irid_fw.z) - min(min(irid_fw.x, irid_fw.y), irid_fw.z);
        let mask_fw = smoothstep(params.band_threshold, 0.95, range_fw) * (0.005 + fresnel * 0.025);
        let bands_w = irid_fw * mask_fw + irid_b * mask_b;
        let band_lum_w = max(bands_w.x, max(bands_w.y, bands_w.z));

        let emit = shell_color + sheen + bands_w + vec3<f32>(spec_f + spec_b + spec2);
        let a = params.shell_alpha + band_lum_w + spec_f + spec_b + spec2;
        out.color = vec4<f32>(emit, a);
        return out;
    }

    // ── Normal bubble ──
    let emit = shell_color + sheen + bands + vec3<f32>(spec_f + spec_b + spec2);
    let a = params.shell_alpha + band_lum + spec_f + spec_b + spec2;
    out.color = vec4<f32>(emit, a);
    return out;
}
