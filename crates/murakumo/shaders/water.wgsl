// Murakumo — Water Material
// 水面シェーダー
//
// Billboard quad上にアニメーションする波、コースティクス、
// フレネル反射、深度吸収を描画する。

struct WaterParams {
    wave_speed: f32,
    wave_scale: f32,
    caustic_intensity: f32,
    depth_fade: f32,
    water_color: vec4<f32>,
}

struct WaterInput {
    uv: vec2<f32>,
    world_pos: vec3<f32>,
    eye_pos: vec3<f32>,
    time: f32,
}

struct WaterOutput {
    color: vec4<f32>,
    discard_pixel: bool,
}

// Simple hash for pseudo-random
fn water_hash(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

// Value noise
fn water_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let a = water_hash(i);
    let b = water_hash(i + vec2<f32>(1.0, 0.0));
    let c = water_hash(i + vec2<f32>(0.0, 1.0));
    let d = water_hash(i + vec2<f32>(1.0, 1.0));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Multi-octave wave height
fn wave_height(p: vec2<f32>, t: f32, scale: f32, speed: f32) -> f32 {
    var height = 0.0;
    var amp = 0.5;
    var freq = scale;
    var pos = p;

    for (var i = 0; i < 4; i++) {
        let wave1 = sin(pos.x * freq + t * speed * 1.3) * amp;
        let wave2 = sin(pos.y * freq * 0.8 + t * speed * 0.9 + 2.1) * amp * 0.7;
        let wave3 = sin((pos.x + pos.y) * freq * 0.6 + t * speed * 1.1 + 4.3) * amp * 0.5;
        height += wave1 + wave2 + wave3;
        amp *= 0.5;
        freq *= 2.0;
        // Rotate for variety
        let c = cos(0.5);
        let s = sin(0.5);
        pos = vec2<f32>(pos.x * c - pos.y * s, pos.x * s + pos.y * c);
    }
    return height;
}

// Caustic pattern using overlapping rotated sine grids
fn caustics(p: vec2<f32>, t: f32) -> f32 {
    let tau = 6.28318;

    // First grid
    let angle1 = 0.4;
    let c1 = cos(angle1);
    let s1 = sin(angle1);
    let p1 = vec2<f32>(p.x * c1 - p.y * s1, p.x * s1 + p.y * c1) * 6.0;
    let grid1 = sin(p1.x + t * 1.2) * sin(p1.y + t * 0.8);

    // Second grid
    let angle2 = 1.2;
    let c2 = cos(angle2);
    let s2 = sin(angle2);
    let p2 = vec2<f32>(p.x * c2 - p.y * s2, p.x * s2 + p.y * c2) * 8.0;
    let grid2 = sin(p2.x - t * 0.9) * sin(p2.y + t * 1.1);

    // Third grid
    let angle3 = 2.4;
    let c3 = cos(angle3);
    let s3 = sin(angle3);
    let p3 = vec2<f32>(p.x * c3 - p.y * s3, p.x * s3 + p.y * c3) * 5.0;
    let grid3 = sin(p3.x + t * 0.7) * sin(p3.y - t * 1.3);

    // Combine: bright caustic lines appear where grids overlap
    let combined = (grid1 + grid2 + grid3) / 3.0;
    return pow(max(combined, 0.0), 2.0);
}

fn water_material(input: WaterInput, params: WaterParams) -> WaterOutput {
    var out: WaterOutput;
    out.discard_pixel = false;

    let uv = input.uv;
    let t = input.time * params.wave_speed;
    let p = (uv - 0.5) * 2.0;

    // Circular mask with soft edges (elliptical for water surface)
    let dist = length(p);
    let edge_mask = 1.0 - smoothstep(0.85, 1.0, dist);
    if edge_mask <= 0.0 {
        out.discard_pixel = true;
        return out;
    }

    // Wave normal calculation via central differences
    let eps = 0.01;
    let h_c = wave_height(uv, t, params.wave_scale, 1.0);
    let h_r = wave_height(uv + vec2<f32>(eps, 0.0), t, params.wave_scale, 1.0);
    let h_u = wave_height(uv + vec2<f32>(0.0, eps), t, params.wave_scale, 1.0);
    let normal = normalize(vec3<f32>(
        (h_c - h_r) / eps * 0.15,
        (h_c - h_u) / eps * 0.15,
        1.0
    ));

    // View direction and fresnel
    let view_dir = normalize(input.eye_pos - input.world_pos);
    let n_dot_v = max(dot(vec3<f32>(normal.x, normal.z, normal.y), view_dir), 0.0);
    let fresnel = pow(1.0 - n_dot_v, 4.0) * 0.8 + 0.2;

    // Base water color with depth absorption
    let depth_factor = exp(-dist * params.depth_fade * 0.5);
    let deep_color = vec3<f32>(params.water_color.x * 0.3, params.water_color.y * 0.4, params.water_color.z * 0.6);
    let shallow_color = vec3<f32>(params.water_color.x, params.water_color.y, params.water_color.z);
    let base_color = mix(deep_color, shallow_color, depth_factor);

    // Caustic pattern
    let caustic_uv = uv + normal.xy * 0.05;
    let caustic_val = caustics(caustic_uv, t * 0.8) * params.caustic_intensity;
    let caustic_color = vec3<f32>(0.3, 0.7, 0.9) * caustic_val * depth_factor;

    // Specular highlights
    let light_dir = normalize(vec3<f32>(0.3, 0.8, 0.5));
    let half_vec = normalize(light_dir + view_dir);
    let spec_n = vec3<f32>(normal.x, normal.z, normal.y);
    let spec = pow(max(dot(spec_n, half_vec), 0.0), 64.0) * 0.6;
    let spec2 = pow(max(dot(spec_n, half_vec), 0.0), 256.0) * 0.3;

    // Reflection approximation (sky color mix)
    let sky_color = vec3<f32>(0.4, 0.6, 0.9);
    let reflect_color = mix(base_color, sky_color, fresnel * 0.6);

    // Combine
    var final_color = reflect_color + caustic_color + vec3<f32>(spec + spec2);

    // Subtle wave foam at peaks
    let foam = smoothstep(0.3, 0.5, h_c) * 0.15;
    final_color = mix(final_color, vec3<f32>(0.9, 0.95, 1.0), foam);

    let alpha = params.water_color.w * edge_mask;
    out.color = vec4<f32>(final_color, alpha);
    return out;
}
