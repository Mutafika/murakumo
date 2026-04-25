//! Per-material tunable parameter definitions.
//!
//! Each material gets up to [`PARAMS_PER_MATERIAL`] floats packed into the
//! `MatParamsUniform` (vec4×2 per material) and accessible from WGSL via
//! the `mp(k, i)` helper. The Rust side owns names + ranges + defaults so
//! the UI can render labeled sliders.

/// Maximum tunable parameters per material. Must match WGSL layout.
pub const PARAMS_PER_MATERIAL: usize = 8;

#[derive(Clone, Copy, Debug)]
pub struct ParamSpec {
    pub name: &'static str,
    pub min: f32,
    pub max: f32,
    pub default: f32,
}

const fn ps(name: &'static str, min: f32, max: f32, default: f32) -> ParamSpec {
    ParamSpec { name, min, max, default }
}

const NONE: &[ParamSpec] = &[];

// ── 0: Bubble ──
const BUBBLE: &[ParamSpec] = &[
    ps("Iridescence", 0.0, 2.5, 1.0),
    ps("Thickness",   0.3, 2.5, 1.0),
    ps("Edge Glow",   0.0, 2.0, 1.0),
    ps("Speed",       0.0, 3.0, 1.0),
    ps("Alpha",       0.0, 1.0, 0.4),
];

// ── 1: Glass ──
const GLASS: &[ParamSpec] = &[
    ps("IOR",       1.0, 2.4, 1.45),
    ps("Hue",       0.0, 1.0, 0.55),
    ps("Tint",      0.0, 1.0, 0.4),
    ps("Caustic",   0.0, 2.0, 1.0),
    ps("Alpha",     0.0, 1.0, 0.5),
];

// ── 2: Portal ──
const PORTAL: &[ParamSpec] = &[
    ps("Swirl",      0.5, 4.0, 1.5),
    ps("Hue",        0.0, 1.0, 0.5),
    ps("Brightness", 0.0, 3.0, 1.0),
    ps("Speed",      0.1, 3.0, 1.0),
    ps("Core",       0.0, 2.0, 1.0),
];

// ── 3: Grid ──
const GRID: &[ParamSpec] = &[
    ps("Density",    4.0, 24.0, 10.0),
    ps("Line Width", 0.005, 0.08, 0.02),
    ps("Glow",       0.0, 3.0, 1.0),
    ps("Hue",        0.0, 1.0, 0.55),
    ps("Pulse",      0.0, 3.0, 1.0),
];

// ── 4: Water ──
const WATER: &[ParamSpec] = &[
    ps("Wave Amp",   0.0, 2.0, 1.0),
    ps("Wave Freq",  0.5, 4.0, 1.0),
    ps("Speed",      0.0, 3.0, 1.0),
    ps("Depth",      0.0, 2.0, 1.0),
    ps("Caustic",    0.0, 3.0, 1.0),
];

// ── 5: Fire ──
const FIRE: &[ParamSpec] = &[
    ps("Height",     0.4, 2.0, 1.0),
    ps("Density",    0.3, 3.0, 1.0),
    ps("Speed",      0.1, 3.0, 1.0),
    ps("Heat",       0.5, 2.5, 1.0),
    ps("Blue Core",  0.0, 1.0, 0.4),
    ps("Brightness", 0.0, 3.0, 1.0),
    ps("Hue",        0.0, 1.0, 0.083),  // 0.083 ≈ orange. 0=red, 0.16=yellow, 0.33=green, 0.5=cyan, 0.66=blue, 0.83=magenta
    ps("White Hot",  0.0, 1.0, 0.6),    // how white the hottest tip becomes
];

// ── 6: Smoke ──
const SMOKE: &[ParamSpec] = &[
    ps("Density",    0.0, 3.0, 1.0),
    ps("Speed",      0.0, 3.0, 1.0),
    ps("Brightness", 0.0, 2.5, 1.0),
    ps("Detail",     0.5, 2.5, 1.0),
];

// ── 7: Aurora ──
const AURORA: &[ParamSpec] = &[
    ps("Speed",      0.1, 3.0, 1.0),
    ps("Intensity",  0.0, 3.0, 1.0),
    ps("Hue",        0.0, 1.0, 0.4),
    ps("Bands",      0.5, 2.5, 1.0),
];

// ── 8: Hologram ──
const HOLOGRAM: &[ParamSpec] = &[
    ps("Lines",      20.0, 200.0, 80.0),
    ps("Speed",      0.0, 5.0, 1.0),
    ps("Hue",        0.0, 1.0, 0.5),
    ps("Glitch",     0.0, 1.0, 0.85),
    ps("Brightness", 0.0, 3.0, 1.0),
];

// ── 9: Crystal ──
const CRYSTAL: &[ParamSpec] = &[
    ps("Facet Scale", 0.5, 8.0, 3.0),
    ps("Sharpness",   0.0, 1.0, 0.6),
    ps("Hue",         0.0, 1.0, 0.6),
    ps("Brightness",  0.0, 3.0, 1.0),
    ps("Dispersion",  0.0, 2.0, 1.0),
];

// ── 10: Metal ──
const METAL: &[ParamSpec] = &[
    ps("Smoothness", 0.0, 1.0, 0.7),
    ps("Hue",        0.0, 1.0, 0.55),
    ps("Saturation", 0.0, 1.5, 0.5),
    ps("Scratch",    0.0, 2.0, 1.0),
];

// ── 11: Neon ──
const NEON: &[ParamSpec] = &[
    ps("Glow",       0.5, 4.0, 1.0),
    ps("Hue",        0.0, 1.0, 0.92),
    ps("Pulse",      0.0, 3.0, 1.0),
    ps("Brightness", 0.0, 3.0, 1.0),
];

// ── 12: Shield ──
const SHIELD: &[ParamSpec] = &[
    ps("Density",    4.0, 16.0, 8.0),
    ps("Hue",        0.0, 1.0, 0.55),
    ps("Glow",       0.0, 3.0, 1.0),
    ps("Speed",      0.0, 3.0, 1.0),
    ps("Ripple",     0.0, 2.0, 1.0),
];

// ── 13: Dissolve ──
const DISSOLVE: &[ParamSpec] = &[
    ps("Threshold",  0.0, 1.0, 0.5),
    ps("Edge Width", 0.01, 0.3, 0.08),
    ps("Hue",        0.0, 1.0, 0.08),
    ps("Edge Glow",  0.0, 4.0, 2.0),
    ps("Auto",       0.0, 1.0, 1.0),
];

// ── 14: Lightning ──
const LIGHTNING: &[ParamSpec] = &[
    ps("Thickness",  0.005, 0.08, 0.025),
    ps("Glow",       0.0, 1.0, 0.25),
    ps("Arc Count",  1.0, 6.0, 3.0),
    ps("Branch %",   0.0, 1.0, 0.35),
    ps("Y Range",    0.5, 1.0, 0.85),
    ps("Speed",      0.5, 4.0, 2.0),
    ps("Hue",        0.0, 1.0, 0.6),
    ps("Brightness", 0.5, 3.0, 1.5),
];

// ── 15: Lava ──
const LAVA: &[ParamSpec] = &[
    ps("Flow",       0.0, 3.0, 1.0),
    ps("Heat",       0.5, 3.0, 1.0),
    ps("Crack",      0.5, 3.0, 1.0),
    ps("Glow",       0.0, 3.0, 1.0),
];

// ── 16: Ice ──
const ICE: &[ParamSpec] = &[
    ps("Frost",      0.5, 3.0, 1.0),
    ps("SSS",        0.0, 1.5, 1.0),
    ps("Hue",        0.0, 1.0, 0.55),
    ps("Roughness",  0.0, 0.6, 0.15),
];

// ── 17: Cloud ──
const CLOUD: &[ParamSpec] = &[
    ps("Density",    0.0, 3.0, 1.0),
    ps("Speed",      0.0, 3.0, 1.0),
    ps("Brightness", 0.0, 2.5, 1.0),
    ps("Detail",     0.5, 3.0, 1.0),
];

// ── 18: Explosion ──
const EXPLOSION: &[ParamSpec] = &[
    ps("Force",      0.3, 2.0, 1.0),
    ps("Heat",       0.5, 3.0, 1.0),
    ps("Speed",      0.1, 3.0, 1.0),
    ps("Smoke",      0.0, 2.0, 1.0),
];

// ── 19: Tornado ──
const TORNADO: &[ParamSpec] = &[
    ps("Speed",      0.5, 5.0, 1.0),
    ps("Density",    0.5, 3.0, 1.0),
    ps("Twist",      0.5, 4.0, 1.0),
    ps("Brightness", 0.0, 2.5, 1.0),
];

// ── 20: Skin ──
const SKIN: &[ParamSpec] = &[
    ps("Subsurface", 0.0, 2.0, 1.0),
    ps("Smoothness", 0.0, 1.0, 0.65),
    ps("Hue",        0.0, 1.0, 0.06),
    ps("Saturation", 0.0, 1.5, 1.0),
];

// ── 21: Rock ──
const ROCK: &[ParamSpec] = &[
    ps("Bumpiness",  0.0, 2.0, 1.0),
    ps("Hue",        0.0, 1.0, 0.08),
    ps("Saturation", 0.0, 1.5, 1.0),
    ps("Detail",     0.5, 2.5, 1.0),
];

// ── 22: Field (terrain reuses Rock material) ──
const FIELD: &[ParamSpec] = &[
    ps("Bumpiness",  0.0, 2.0, 1.0),
    ps("Hue",        0.0, 1.0, 0.08),
    ps("Saturation", 0.0, 1.5, 1.0),
    ps("Detail",     0.5, 2.5, 1.0),
];

pub fn material_params(material_index: usize) -> &'static [ParamSpec] {
    match material_index {
        0  => BUBBLE,
        1  => GLASS,
        2  => PORTAL,
        3  => GRID,
        4  => WATER,
        5  => FIRE,
        6  => SMOKE,
        7  => AURORA,
        8  => HOLOGRAM,
        9  => CRYSTAL,
        10 => METAL,
        11 => NEON,
        12 => SHIELD,
        13 => DISSOLVE,
        14 => LIGHTNING,
        15 => LAVA,
        16 => ICE,
        17 => CLOUD,
        18 => EXPLOSION,
        19 => TORNADO,
        20 => SKIN,
        21 => ROCK,
        22 => FIELD,
        _ => NONE,
    }
}

pub fn default_values(material_index: usize) -> [f32; PARAMS_PER_MATERIAL] {
    let mut out = [0.0f32; PARAMS_PER_MATERIAL];
    for (i, spec) in material_params(material_index).iter().enumerate() {
        if i < PARAMS_PER_MATERIAL {
            out[i] = spec.default;
        }
    }
    out
}
