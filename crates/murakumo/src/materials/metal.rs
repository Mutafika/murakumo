//! 異方性メタルマテリアル
//!
//! ブラシ仕上げ金属の異方性スペキュラーと
//! GGX近似による環境反射を表現する。

use super::Material;

/// 異方性メタルのパラメータ
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MetalParams {
    /// 基本色 RGBA (default: [0.8, 0.78, 0.75, 1.0])
    pub base_color: [f32; 4],
    /// 粗さ (default: 0.3)
    pub roughness: f32,
    /// 異方性の強さ (default: 0.7)
    pub anisotropy: f32,
    /// 異方性の方向 (ラジアン, default: 0.0)
    pub aniso_direction: f32,
    /// 反射率 (default: 0.9)
    pub reflectance: f32,
}

impl Default for MetalParams {
    fn default() -> Self {
        Self {
            base_color: [0.8, 0.78, 0.75, 1.0],
            roughness: 0.3,
            anisotropy: 0.7,
            aniso_direction: 0.0,
            reflectance: 0.9,
        }
    }
}

pub struct Metal {
    pub params: MetalParams,
}

impl Default for Metal {
    fn default() -> Self {
        Self { params: MetalParams::default() }
    }
}

impl Material for Metal {
    fn shader_source(&self) -> &str {
        include_str!("../../shaders/metal.wgsl")
    }

    fn uniform_bytes(&self) -> Vec<u8> {
        bytemuck::bytes_of(&self.params).to_vec()
    }
}
