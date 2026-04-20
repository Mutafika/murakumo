//! 炎マテリアル
//!
//! Billboard quad上にFBMノイズベースの炎エフェクトを描画する。
//! 下から上へスクロールするノイズにカラーグラデーションを適用し、
//! エッジと上部でアルファフェードする。

use super::Material;

/// 炎のパラメータ
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FireParams {
    /// 炎の強度 (default: 1.5)
    pub intensity: f32,
    /// アニメーション速度 (default: 2.0)
    pub speed: f32,
    /// 乱流の強さ (default: 1.0)
    pub turbulence: f32,
    pub _pad: f32,
    /// 下部の色 RGBA (default: [1.0, 0.3, 0.0, 1.0] — オレンジ)
    pub color_bottom: [f32; 4],
    /// 上部の色 RGBA (default: [1.0, 0.9, 0.1, 0.0] — 黄)
    pub color_top: [f32; 4],
}

impl Default for FireParams {
    fn default() -> Self {
        Self {
            intensity: 1.5,
            speed: 2.0,
            turbulence: 1.0,
            _pad: 0.0,
            color_bottom: [1.0, 0.3, 0.0, 1.0],
            color_top: [1.0, 0.9, 0.1, 0.0],
        }
    }
}

pub struct Fire {
    pub params: FireParams,
}

impl Default for Fire {
    fn default() -> Self {
        Self { params: FireParams::default() }
    }
}

impl Material for Fire {
    fn shader_source(&self) -> &str {
        include_str!("../../shaders/fire.wgsl")
    }

    fn uniform_bytes(&self) -> Vec<u8> {
        bytemuck::bytes_of(&self.params).to_vec()
    }
}
