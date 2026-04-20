//! 薄膜干渉シャボン玉マテリアル
//!
//! ビルボードクワッドの上でレイマーチした球体に
//! 前面+背面の2層干渉パターンを描画する。

use super::Material;

/// シャボン玉のパラメータ
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BubbleParams {
    /// 基本膜厚 (nm) — 色の基調を決める (default: 400.0)
    pub thickness_base: f32,
    /// 重力による膜厚変化量 (default: 300.0)
    pub gravity_strength: f32,
    /// 干渉バンドの表示閾値 (0.0〜1.0, 高いほど帯が細い, default: 0.7)
    pub band_threshold: f32,
    /// シェル（球体輪郭）の透明度 (default: 0.025)
    pub shell_alpha: f32,
    /// ポップ進行度 (0.0 = 通常, 0.0〜1.0 = 破裂中)
    pub pop_progress: f32,
    pub _pad: [f32; 3],
}

impl Default for BubbleParams {
    fn default() -> Self {
        Self {
            thickness_base: 400.0,
            gravity_strength: 300.0,
            band_threshold: 0.7,
            shell_alpha: 0.025,
            pop_progress: 0.0,
            _pad: [0.0; 3],
        }
    }
}

pub struct Bubble {
    pub params: BubbleParams,
}

impl Default for Bubble {
    fn default() -> Self {
        Self { params: BubbleParams::default() }
    }
}

impl Material for Bubble {
    fn shader_source(&self) -> &str {
        include_str!("../../shaders/bubble.wgsl")
    }

    fn uniform_bytes(&self) -> Vec<u8> {
        bytemuck::bytes_of(&self.params).to_vec()
    }
}
