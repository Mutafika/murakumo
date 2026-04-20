//! クリスタルマテリアル
//!
//! ボロノイセルによるファセット表面と屈折、
//! スパークル、内部発光で宝石的表現を行う。

use super::Material;

/// クリスタルのパラメータ
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CrystalParams {
    /// 基本色 RGBA (default: [0.4, 0.6, 1.0, 0.9])
    pub color: [f32; 4],
    /// 屈折の強さ (default: 0.15)
    pub refraction_strength: f32,
    /// ファセット数 (default: 12.0)
    pub facet_count: f32,
    /// スパークルの強度 (default: 1.5)
    pub sparkle_intensity: f32,
    /// 内部発光の強さ (default: 0.4)
    pub internal_glow: f32,
}

impl Default for CrystalParams {
    fn default() -> Self {
        Self {
            color: [0.4, 0.6, 1.0, 0.9],
            refraction_strength: 0.15,
            facet_count: 12.0,
            sparkle_intensity: 1.5,
            internal_glow: 0.4,
        }
    }
}

pub struct Crystal {
    pub params: CrystalParams,
}

impl Default for Crystal {
    fn default() -> Self {
        Self { params: CrystalParams::default() }
    }
}

impl Material for Crystal {
    fn shader_source(&self) -> &str {
        include_str!("../../shaders/crystal.wgsl")
    }

    fn uniform_bytes(&self) -> Vec<u8> {
        bytemuck::bytes_of(&self.params).to_vec()
    }
}
