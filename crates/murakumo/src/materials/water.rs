//! 水面マテリアル
//!
//! Billboard quad上にアニメーションする波の法線マッピング、
//! コースティクスパターン、フレネル反射、深度による色吸収を描画する。

use super::Material;

/// 水面のパラメータ
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WaterParams {
    /// 波のアニメーション速度 (default: 1.0)
    pub wave_speed: f32,
    /// 波のスケール (default: 8.0)
    pub wave_scale: f32,
    /// コースティクスの強度 (default: 0.6)
    pub caustic_intensity: f32,
    /// 深度フェード (default: 2.0)
    pub depth_fade: f32,
    /// 水の色 RGBA (default: [0.05, 0.2, 0.4, 0.85])
    pub water_color: [f32; 4],
}

impl Default for WaterParams {
    fn default() -> Self {
        Self {
            wave_speed: 1.0,
            wave_scale: 8.0,
            caustic_intensity: 0.6,
            depth_fade: 2.0,
            water_color: [0.05, 0.2, 0.4, 0.85],
        }
    }
}

pub struct Water {
    pub params: WaterParams,
}

impl Default for Water {
    fn default() -> Self {
        Self { params: WaterParams::default() }
    }
}

impl Material for Water {
    fn shader_source(&self) -> &str {
        include_str!("../../shaders/water.wgsl")
    }

    fn uniform_bytes(&self) -> Vec<u8> {
        bytemuck::bytes_of(&self.params).to_vec()
    }
}
