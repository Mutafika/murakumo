//! 電撃アークマテリアル
//!
//! 手続き的に生成されるライトニングボルト。
//! ミッドポイントディスプレイスメントのシェーダー近似で
//! 分岐する電撃を描画する。

use super::Material;

/// 電撃のパラメータ
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightningParams {
    /// 電撃の色 (RGBA, default: [0.6, 0.8, 1.0, 1.0])
    pub color: [f32; 4],
    /// アーク本数 (default: 3.0)
    pub arc_count: f32,
    /// 太さ (default: 0.015)
    pub thickness: f32,
    /// 分岐確率 (0.0-1.0, default: 0.4)
    pub branch_probability: f32,
    /// 明るさ (default: 2.0)
    pub intensity: f32,
    /// アニメーション速度 (default: 8.0)
    pub speed: f32,
    pub _pad: [f32; 3],
}

impl Default for LightningParams {
    fn default() -> Self {
        Self {
            color: [0.6, 0.8, 1.0, 1.0],
            arc_count: 3.0,
            thickness: 0.015,
            branch_probability: 0.4,
            intensity: 2.0,
            speed: 8.0,
            _pad: [0.0; 3],
        }
    }
}

pub struct Lightning {
    pub params: LightningParams,
}

impl Default for Lightning {
    fn default() -> Self {
        Self { params: LightningParams::default() }
    }
}

impl Material for Lightning {
    fn shader_source(&self) -> &str {
        include_str!("../../shaders/lightning.wgsl")
    }

    fn uniform_bytes(&self) -> Vec<u8> {
        bytemuck::bytes_of(&self.params).to_vec()
    }
}
