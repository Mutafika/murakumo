// Murakumo — Metal Material (Anisotropic)
// 異方性メタルシェーダー
//
// ブラシ仕上げ金属のストレッチスペキュラー、
// GGX近似、環境反射、方向性スクラッチを表現する。

struct MetalParams {
    base_color: vec4<f32>,
    roughness: f32,
    anisotropy: f32,
    aniso_direction: f32,
    reflectance: f32,
}

struct MetalInput {
    uv: vec2<f32>,
    world_pos: vec3<f32>,
    eye_pos: vec3<f32>,
    time: f32,
}

struct MetalOutput {
    color: vec4<f32>,
    discard_pixel: bool,
}

fn hash_metal(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

// Noise for scratches
fn noise_metal(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash_metal(i);
    let b = hash_metal(i + vec2<f32>(1.0, 0.0));
    let c = hash_metal(i + vec2<f32>(0.0, 1.0));
    let d = hash_metal(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm_metal(p: vec2<f32>) -> f32 {
    var val = 0.0;
    var amp = 0.5;
    var pos = p;
    for (var i = 0; i < 4; i = i + 1) {
        val = val + amp * noise_metal(pos);
        pos = pos * 2.1;
        amp = amp * 0.5;
    }
    return val;
}

// GGX distribution approximation
fn ggx_distribution(n_dot_h: f32, roughness_sq: f32) -> f32 {
    let a2 = roughness_sq * roughness_sq;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (3.14159 * denom * denom + 0.0001);
}

fn metal_material(input: MetalInput, params: MetalParams) -> MetalOutput {
    var out: MetalOutput;
    out.discard_pixel = false;

    let t = input.time;
    let uv = input.uv;
    let uv_cent = uv - 0.5;

    // Sphere shape on billboard
    let r2 = dot(uv_cent, uv_cent) * 4.0;
    if r2 > 1.0 {
        out.discard_pixel = true;
        return out;
    }

    let nz = sqrt(1.0 - r2);
    let sphere_n = normalize(vec3<f32>(uv_cent.x * 2.0, uv_cent.y * 2.0, nz));
    let view_dir = normalize(input.eye_pos - input.world_pos);
    let n_dot_v = max(dot(sphere_n, view_dir), 0.001);

    // ── Tangent / Bitangent for anisotropy ──
    let angle = params.aniso_direction;
    let tangent = normalize(vec3<f32>(cos(angle), sin(angle), 0.0));
    let bitangent = normalize(cross(sphere_n, tangent));
    let actual_tangent = normalize(cross(bitangent, sphere_n));

    // ── Directional scratches via noise ──
    let scratch_uv = vec2<f32>(
        dot(uv_cent * 2.0, vec2<f32>(cos(angle), sin(angle))),
        dot(uv_cent * 2.0, vec2<f32>(-sin(angle), cos(angle)))
    );
    // Stretch in aniso direction for directional scratches
    let scratch_coord = vec2<f32>(scratch_uv.x * 2.0, scratch_uv.y * 40.0);
    let scratch = fbm_metal(scratch_coord) * 0.15 * params.anisotropy;

    // ── Lighting (multiple lights for environment feel) ──
    let light_positions = array<vec3<f32>, 3>(
        normalize(vec3<f32>(0.5, 0.8, 0.6)),
        normalize(vec3<f32>(-0.7, 0.3, 0.5)),
        normalize(vec3<f32>(0.0, -0.5, 0.8))
    );
    let light_colors = array<vec3<f32>, 3>(
        vec3<f32>(1.0, 0.95, 0.9),
        vec3<f32>(0.7, 0.8, 1.0),
        vec3<f32>(0.9, 0.85, 0.8)
    );

    var total_spec = vec3<f32>(0.0);
    var total_diffuse = vec3<f32>(0.0);

    for (var i = 0; i < 3; i = i + 1) {
        let L = light_positions[i];
        let H = normalize(L + view_dir);

        let n_dot_l = max(dot(sphere_n, L), 0.0);
        let n_dot_h = max(dot(sphere_n, H), 0.0);

        // Anisotropic half-vector decomposition
        let h_dot_t = dot(H, actual_tangent);
        let h_dot_b = dot(H, bitangent);

        // Stretched roughness
        let alpha_t = params.roughness * (1.0 + params.anisotropy);
        let alpha_b = params.roughness * (1.0 - params.anisotropy * 0.9);

        // Anisotropic GGX
        let aniso_term = (h_dot_t * h_dot_t) / (alpha_t * alpha_t + 0.0001)
                       + (h_dot_b * h_dot_b) / (alpha_b * alpha_b + 0.0001);
        let D = 1.0 / (3.14159 * alpha_t * alpha_b * pow(aniso_term + n_dot_h * n_dot_h, 2.0) + 0.0001);

        // Fresnel (Schlick)
        let f0 = params.reflectance;
        let fresnel = f0 + (1.0 - f0) * pow(1.0 - max(dot(H, view_dir), 0.0), 5.0);

        // Geometry term (simplified)
        let k = params.roughness * 0.5;
        let g_v = n_dot_v / (n_dot_v * (1.0 - k) + k);
        let g_l = n_dot_l / (n_dot_l * (1.0 - k) + k);
        let G = g_v * g_l;

        let spec = D * fresnel * G / (4.0 * n_dot_v * n_dot_l + 0.001);
        total_spec = total_spec + light_colors[i] * spec * n_dot_l;
        total_diffuse = total_diffuse + light_colors[i] * n_dot_l * 0.1;
    }

    // ── Environment reflection (fake) ──
    let reflect_dir = reflect(-view_dir, sphere_n);
    let env_uv = reflect_dir.xy * 0.5 + 0.5;
    // Gradient sky reflection
    let env_color = mix(
        vec3<f32>(0.2, 0.25, 0.35),
        vec3<f32>(0.8, 0.85, 0.95),
        reflect_dir.y * 0.5 + 0.5
    );
    let env_reflect = env_color * params.reflectance * (1.0 - params.roughness);

    // ── Combine ──
    var color = params.base_color.rgb * total_diffuse;
    color = color + total_spec * params.base_color.rgb;
    color = color + env_reflect * params.base_color.rgb;

    // Add scratches
    color = color + vec3<f32>(scratch) * params.base_color.rgb;

    // Slight darkening at grazing angles (more realistic)
    let rim_darken = smoothstep(0.0, 0.3, n_dot_v);
    color = color * (0.5 + 0.5 * rim_darken);

    out.color = vec4<f32>(color, params.base_color.a);
    return out;
}
