// Murakumo — Fire Material
// 炎シェーダー
//
// Billboard quad上にFBMノイズベースの炎エフェクトを描画する。
// 下から上へスクロールするノイズにカラーグラデーションを適用。

struct FireParams {
    intensity: f32,
    speed: f32,
    turbulence: f32,
    _pad: f32,
    color_bottom: vec4<f32>,
    color_top: vec4<f32>,
}

struct FireInput {
    uv: vec2<f32>,
    world_pos: vec3<f32>,
    eye_pos: vec3<f32>,
    time: f32,
}

struct FireOutput {
    color: vec4<f32>,
    discard_pixel: bool,
}

// Hash function for noise
fn fire_hash(p: vec2<f32>) -> f32 {
    let p2 = vec2<f32>(dot(p, vec2<f32>(127.1, 311.7)), dot(p, vec2<f32>(269.5, 183.3)));
    let s = sin(p2) * 43758.5453123;
    return fract(s.x + s.y);
}

// 2D value noise with smooth interpolation
fn fire_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    // Quintic interpolation for smoother results
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    let a = fire_hash(i);
    let b = fire_hash(i + vec2<f32>(1.0, 0.0));
    let c = fire_hash(i + vec2<f32>(0.0, 1.0));
    let d = fire_hash(i + vec2<f32>(1.0, 1.0));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Fractal Brownian Motion with domain warping
fn fire_fbm(p: vec2<f32>, octaves: i32, turbulence: f32) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;
    var pos = p;

    for (var i = 0; i < octaves; i++) {
        value += amplitude * fire_noise(pos * frequency);
        // Domain warping for more organic look
        let warp = vec2<f32>(
            fire_noise(pos * frequency + vec2<f32>(5.2, 1.3)),
            fire_noise(pos * frequency + vec2<f32>(1.7, 9.2))
        );
        pos += warp * turbulence * 0.3 / frequency;
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    return value;
}

fn fire_material(input: FireInput, params: FireParams) -> FireOutput {
    var out: FireOutput;
    out.discard_pixel = false;

    let uv = input.uv;
    let t = input.time * params.speed;

    // Fire shape: narrower at top, wider at bottom
    let center_x = abs(uv.x - 0.5) * 2.0;
    let taper = mix(0.9, 0.2, uv.y);  // Width shrinks upward
    let shape_mask = 1.0 - smoothstep(taper * 0.6, taper, center_x);

    // Vertical gradient (fire fades at top)
    let vert_fade = 1.0 - pow(uv.y, 1.5);

    // Combined base mask
    let base_mask = shape_mask * vert_fade;
    if base_mask <= 0.01 {
        out.discard_pixel = true;
        return out;
    }

    // Scrolling noise coordinates (moving upward)
    let scroll_uv = vec2<f32>(
        uv.x * 3.0,
        uv.y * 4.0 - t * 1.5
    );

    // Primary fire noise (large shapes)
    let noise1 = fire_fbm(scroll_uv, 5, params.turbulence);

    // Secondary turbulence (faster, smaller)
    let turb_uv = vec2<f32>(
        uv.x * 5.0 + sin(t * 0.7) * 0.3,
        uv.y * 6.0 - t * 2.5
    );
    let noise2 = fire_fbm(turb_uv, 3, params.turbulence * 1.5);

    // Tertiary detail (flickering)
    let flicker_uv = vec2<f32>(
        uv.x * 8.0 + cos(t * 1.3) * 0.2,
        uv.y * 10.0 - t * 3.5
    );
    let noise3 = fire_noise(flicker_uv);

    // Composite noise
    let fire_n = noise1 * 0.5 + noise2 * 0.35 + noise3 * 0.15;

    // Shape the fire: more noise influence at top (makes jagged flames)
    let shaped = fire_n * base_mask;
    let fire_intensity = pow(shaped, 1.2 - uv.y * 0.5) * params.intensity;

    if fire_intensity <= 0.01 {
        out.discard_pixel = true;
        return out;
    }

    // Color gradient: bottom (hot) to top (cooler)
    let color_t = pow(uv.y, 0.8);
    let col_bottom = vec3<f32>(params.color_bottom.x, params.color_bottom.y, params.color_bottom.z);
    let col_top = vec3<f32>(params.color_top.x, params.color_top.y, params.color_top.z);

    // Add a hot white core
    let core_mask = (1.0 - center_x) * (1.0 - uv.y) * fire_intensity;
    let core_color = vec3<f32>(1.0, 0.95, 0.8);

    var fire_color = mix(col_bottom, col_top, color_t);
    fire_color = mix(fire_color, core_color, smoothstep(0.5, 1.2, core_mask));

    // Emissive boost
    fire_color *= fire_intensity;

    // Flickering via time
    let flicker = 0.9 + 0.1 * sin(t * 12.0 + uv.x * 5.0) * sin(t * 7.0 + uv.y * 3.0);
    fire_color *= flicker;

    // Alpha: based on fire intensity, edge fade
    let alpha = clamp(fire_intensity * 1.5, 0.0, 1.0) * base_mask;

    out.color = vec4<f32>(fire_color, alpha);
    return out;
}
