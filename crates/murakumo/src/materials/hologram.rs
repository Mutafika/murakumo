//! ホログラムマテリアル
//!
//! スキャンラインとグリッチエフェクトで
//! サイバーパンク風ホログラム表現を行う。

use super::Material;

/// ホログラムのパラメータ
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HologramParams {
    /// 基本色 RGBA (default: [0.0, 0.8, 1.0, 0.7])
    pub base_color: [f32; 4],
    /// スキャンラインのスクロール速度 (default: 2.0)
    pub scan_line_speed: f32,
    /// スキャンラインの密度 (default: 80.0)
    pub scan_line_density: f32,
    /// グリッチの強度 (default: 0.3)
    pub glitch_intensity: f32,
    /// フリッカーの速度 (default: 8.0)
    pub flicker_speed: f32,
}

impl Default for HologramParams {
    fn default() -> Self {
        Self {
            base_color: [0.0, 0.8, 1.0, 0.7],
            scan_line_speed: 2.0,
            scan_line_density: 80.0,
            glitch_intensity: 0.3,
            flicker_speed: 8.0,
        }
    }
}

pub struct Hologram {
    pub params: HologramParams,
}

impl Default for Hologram {
    fn default() -> Self {
        Self { params: HologramParams::default() }
    }
}

impl Material for Hologram {
    fn shader_source(&self) -> &str {
        include_str!("../../shaders/hologram.wgsl")
    }

    fn uniform_bytes(&self) -> Vec<u8> {
        bytemuck::bytes_of(&self.params).to_vec()
    }
}
