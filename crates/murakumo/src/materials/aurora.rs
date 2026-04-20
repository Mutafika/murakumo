//! オーロラマテリアル
//!
//! Billboard quad上にカーテン状の縦バンドを描画する。
//! 複数の周波数のsin波で揺らぎ、プライマリ/セカンダリ色間を補間し、
//! ソフトグローと垂直方向のフェードを適用する。

use super::Material;

/// オーロラのパラメータ
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AuroraParams {
    /// カーテンの流れ速度 (default: 0.8)
    pub curtain_speed: f32,
    /// 波の振幅 (default: 0.3)
    pub wave_amplitude: f32,
    /// 明るさ (default: 1.5)
    pub brightness: f32,
    pub _pad: f32,
    /// プライマリ色 RGBA (default: [0.1, 0.9, 0.4, 1.0] — 緑)
    pub color_primary: [f32; 4],
    /// セカンダリ色 RGBA (default: [0.3, 0.2, 0.9, 1.0] — 紫)
    pub color_secondary: [f32; 4],
}

impl Default for AuroraParams {
    fn default() -> Self {
        Self {
            curtain_speed: 0.8,
            wave_amplitude: 0.3,
            brightness: 1.5,
            _pad: 0.0,
            color_primary: [0.1, 0.9, 0.4, 1.0],
            color_secondary: [0.3, 0.2, 0.9, 1.0],
        }
    }
}

pub struct Aurora {
    pub params: AuroraParams,
}

impl Default for Aurora {
    fn default() -> Self {
        Self { params: AuroraParams::default() }
    }
}

impl Material for Aurora {
    fn shader_source(&self) -> &str {
        include_str!("../../shaders/aurora.wgsl")
    }

    fn uniform_bytes(&self) -> Vec<u8> {
        bytemuck::bytes_of(&self.params).to_vec()
    }
}
