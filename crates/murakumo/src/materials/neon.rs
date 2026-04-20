//! ネオングローマテリアル
//!
//! 明るいコアラインからのフォールオフグローと
//! パルスアニメーションでネオン管を表現する。

use super::Material;

/// ネオングローのパラメータ
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NeonParams {
    /// グロー色 RGBA (default: [1.0, 0.1, 0.5, 1.0])
    pub glow_color: [f32; 4],
    /// コアの明るさ (default: 3.0)
    pub core_brightness: f32,
    /// グローの半径 (default: 0.3)
    pub glow_radius: f32,
    /// パルス速度 (default: 2.0)
    pub pulse_speed: f32,
    /// パルス量 (default: 0.3)
    pub pulse_amount: f32,
}

impl Default for NeonParams {
    fn default() -> Self {
        Self {
            glow_color: [1.0, 0.1, 0.5, 1.0],
            core_brightness: 3.0,
            glow_radius: 0.3,
            pulse_speed: 2.0,
            pulse_amount: 0.3,
        }
    }
}

pub struct Neon {
    pub params: NeonParams,
}

impl Default for Neon {
    fn default() -> Self {
        Self { params: NeonParams::default() }
    }
}

impl Material for Neon {
    fn shader_source(&self) -> &str {
        include_str!("../../shaders/neon.wgsl")
    }

    fn uniform_bytes(&self) -> Vec<u8> {
        bytemuck::bytes_of(&self.params).to_vec()
    }
}
