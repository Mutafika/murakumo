//! 肌マテリアル
//!
//! サブサーフェススキャタリングによるリアルな肌表現。
//! 光が内部で散乱して赤みを帯びる。

use super::Material;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkinParams {
    /// SSS強度 (default: 1.0)
    pub sss_intensity: f32,
    /// 表面粗さ (default: 0.4)
    pub roughness: f32,
    /// 散乱半径 (default: 0.5)
    pub scatter_radius: f32,
    pub _pad: f32,
    /// 肌の色 (default: [0.85, 0.65, 0.5, 1.0])
    pub skin_color: [f32; 4],
    /// 散乱光の色 (default: [0.9, 0.3, 0.2, 1.0])
    pub scatter_color: [f32; 4],
}

impl Default for SkinParams {
    fn default() -> Self {
        Self {
            sss_intensity: 1.0,
            roughness: 0.4,
            scatter_radius: 0.5,
            _pad: 0.0,
            skin_color: [0.85, 0.65, 0.5, 1.0],
            scatter_color: [0.9, 0.3, 0.2, 1.0],
        }
    }
}

pub struct Skin {
    pub params: SkinParams,
}

impl Default for Skin {
    fn default() -> Self {
        Self { params: SkinParams::default() }
    }
}

impl Material for Skin {
    fn shader_source(&self) -> &str {
        "// TODO: skin shader"
    }
    fn uniform_bytes(&self) -> Vec<u8> {
        bytemuck::bytes_of(&self.params).to_vec()
    }
}
