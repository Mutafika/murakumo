// Murakumo — Crystal Material
// クリスタルシェーダー
//
// ボロノイセルによるファセット面、屈折、スパークル、
// 内部発光、虹色分散で宝石的表現を行う。

struct CrystalParams {
    color: vec4<f32>,
    refraction_strength: f32,
    facet_count: f32,
    sparkle_intensity: f32,
    internal_glow: f32,
}

struct CrystalInput {
    uv: vec2<f32>,
    world_pos: vec3<f32>,
    eye_pos: vec3<f32>,
    time: f32,
}

struct CrystalOutput {
    color: vec4<f32>,
    discard_pixel: bool,
}

// 2D random for voronoi
fn hash2_crystal(p: vec2<f32>) -> vec2<f32> {
    let q = vec2<f32>(
        dot(p, vec2<f32>(127.1, 311.7)),
        dot(p, vec2<f32>(269.5, 183.3))
    );
    return fract(sin(q) * 43758.5453);
}

fn hash1_crystal(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

// Voronoi with cell ID and edge distance
fn voronoi_crystal(uv: vec2<f32>, scale: f32, time: f32) -> vec3<f32> {
    let p = uv * scale;
    let i_p = floor(p);
    let f_p = fract(p);

    var min_dist = 1.0;
    var second_dist = 1.0;
    var cell_id = vec2<f32>(0.0);

    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let neighbor = vec2<f32>(f32(x), f32(y));
            let point = hash2_crystal(i_p + neighbor);
            // Slight animation
            let animated = neighbor + 0.5 + 0.3 * sin(time * 0.5 + 6.28318 * point);
            let diff = animated - f_p;
            let d = length(diff);
            if d < min_dist {
                second_dist = min_dist;
                min_dist = d;
                cell_id = i_p + neighbor;
            } else if d < second_dist {
                second_dist = d;
            }
        }
    }

    let edge = second_dist - min_dist;
    return vec3<f32>(min_dist, edge, hash1_crystal(cell_id));
}

fn crystal_material(input: CrystalInput, params: CrystalParams) -> CrystalOutput {
    var out: CrystalOutput;
    out.discard_pixel = false;

    let t = input.time;
    let uv = input.uv;
    let uv_cent = uv - 0.5;

    // Billboard sphere shape
    let r2 = dot(uv_cent, uv_cent) * 4.0;
    if r2 > 1.0 {
        out.discard_pixel = true;
        return out;
    }

    let nz = sqrt(1.0 - r2);
    let sphere_n = normalize(vec3<f32>(uv_cent.x * 2.0, uv_cent.y * 2.0, nz));
    let view_dir = normalize(input.eye_pos - input.world_pos);
    let n_dot_v = max(dot(sphere_n, view_dir), 0.0);

    // ── Voronoi facets ──
    let vor = voronoi_crystal(uv, params.facet_count, t);
    let cell_dist = vor.x;
    let edge_dist = vor.y;
    let cell_rand = vor.z;

    // Facet normal perturbation (each facet has a slightly different orientation)
    let facet_angle = cell_rand * 6.28318;
    let facet_tilt = 0.2 * params.refraction_strength;
    let facet_n = normalize(sphere_n + vec3<f32>(
        cos(facet_angle) * facet_tilt,
        sin(facet_angle) * facet_tilt,
        0.0
    ));

    // ── Refraction offset (fake) ──
    let refract_offset = (facet_n.xy - sphere_n.xy) * params.refraction_strength;
    let refracted_uv = uv + refract_offset;

    // Second voronoi layer for internal depth
    let vor2 = voronoi_crystal(refracted_uv + vec2<f32>(0.3, 0.7), params.facet_count * 0.7, t * 0.8);

    // ── Base crystal color per facet ──
    let facet_hue_shift = cell_rand * 0.15;
    var crystal_col = params.color.rgb * (0.85 + 0.3 * cell_rand);
    crystal_col = crystal_col + vec3<f32>(facet_hue_shift, -facet_hue_shift * 0.5, facet_hue_shift * 0.3);

    // ── Specular highlights (sparkles at edges) ──
    let light1 = normalize(vec3<f32>(0.5, 0.8, 0.6));
    let light2 = normalize(vec3<f32>(-0.4, 0.3, 0.8));
    let h1 = normalize(light1 + view_dir);
    let h2 = normalize(light2 + view_dir);
    let spec1 = pow(max(dot(facet_n, h1), 0.0), 512.0);
    let spec2 = pow(max(dot(facet_n, h2), 0.0), 256.0);

    // Edge sparkle (bright at facet boundaries)
    let edge_sparkle = pow(smoothstep(0.05, 0.0, edge_dist), 2.0) * params.sparkle_intensity;
    let sparkle = (spec1 * 1.5 + spec2 * 0.8 + edge_sparkle) * params.sparkle_intensity;

    // ── Internal glow (varies with viewing angle) ──
    let fresnel = pow(1.0 - n_dot_v, 3.0);
    let glow_factor = params.internal_glow * (0.5 + 0.5 * sin(t * 1.5 + cell_rand * 6.28));
    let internal = params.color.rgb * glow_factor * (1.0 - fresnel) * (0.7 + 0.3 * vor2.z);

    // ── Rainbow dispersion at edges ──
    let dispersion_amount = fresnel * smoothstep(0.15, 0.0, edge_dist);
    let rainbow = vec3<f32>(
        0.5 + 0.5 * sin(cell_rand * 20.0 + 0.0),
        0.5 + 0.5 * sin(cell_rand * 20.0 + 2.094),
        0.5 + 0.5 * sin(cell_rand * 20.0 + 4.189)
    );
    let dispersion = rainbow * dispersion_amount * 0.6;

    // ── Combine ──
    var color = crystal_col * (0.4 + 0.6 * n_dot_v);
    color = color + internal;
    color = color + dispersion;
    color = color + vec3<f32>(sparkle);

    // Fresnel rim
    color = color + params.color.rgb * fresnel * 0.3;

    // Edge darkening (facet boundaries)
    let edge_line = smoothstep(0.02, 0.04, edge_dist);
    color = color * (0.7 + 0.3 * edge_line);

    let alpha = params.color.a * (0.8 + 0.2 * n_dot_v);

    out.color = vec4<f32>(color, alpha);
    return out;
}
