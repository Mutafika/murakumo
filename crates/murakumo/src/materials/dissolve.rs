//! ノイズディゾルブマテリアル
//!
//! FBMノイズを使った溶解エフェクト。
//! 溶解境界に明るいグロー、方向バイアス対応。

use super::Material;

/// ディゾルブのパラメータ
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DissolveParams {
    /// 溶解進行度 (0.0 = 完全表示, 1.0 = 完全消失, default: 0.0)
    pub progress: f32,
    pub _pad0: [f32; 3],
    /// エッジの色 (RGBA, default: [1.0, 0.5, 0.1, 1.0])
    pub edge_color: [f32; 4],
    /// エッジの幅 (default: 0.05)
    pub edge_width: f32,
    /// ノイズスケール (default: 4.0)
    pub noise_scale: f32,
    /// 溶解方向バイアス (UV方向, default: [0.0, 1.0] = 下から上)
    pub direction: [f32; 2],
}

impl Default for DissolveParams {
    fn default() -> Self {
        Self {
            progress: 0.0,
            _pad0: [0.0; 3],
            edge_color: [1.0, 0.5, 0.1, 1.0],
            edge_width: 0.05,
            noise_scale: 4.0,
            direction: [0.0, 1.0],
        }
    }
}

pub struct Dissolve {
    pub params: DissolveParams,
}

impl Default for Dissolve {
    fn default() -> Self {
        Self { params: DissolveParams::default() }
    }
}

impl Material for Dissolve {
    fn shader_source(&self) -> &str {
        include_str!("../../shaders/dissolve.wgsl")
    }

    fn uniform_bytes(&self) -> Vec<u8> {
        bytemuck::bytes_of(&self.params).to_vec()
    }
}
