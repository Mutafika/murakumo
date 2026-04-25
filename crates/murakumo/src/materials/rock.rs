//! 岩マテリアル
//!
//! 3Dノイズベースのリアルな岩肌。
//! 凹凸、粗さ変化、気泡痕、細かいクラック。
//! Lava等の複合マテリアルのベースとして使用。

use super::Material;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RockParams {
    /// 岩のスケール (default: 1.0)
    pub scale: f32,
    /// 粗さの範囲 (default: 0.85)
    pub roughness: f32,
    /// バンプの強さ (default: 0.5)
    pub bump_strength: f32,
    pub _pad: f32,
    /// ベース色 (default: [0.12, 0.1, 0.08, 1.0])
    pub base_color: [f32; 4],
}

impl Default for RockParams {
    fn default() -> Self {
        Self {
            scale: 1.0,
            roughness: 0.85,
            bump_strength: 0.5,
            _pad: 0.0,
            base_color: [0.12, 0.1, 0.08, 1.0],
        }
    }
}

pub struct Rock {
    pub params: RockParams,
}

impl Default for Rock {
    fn default() -> Self {
        Self { params: RockParams::default() }
    }
}

impl Material for Rock {
    fn shader_source(&self) -> &str {
        "// TODO: rock shader"
    }
    fn uniform_bytes(&self) -> Vec<u8> {
        bytemuck::bytes_of(&self.params).to_vec()
    }
}
