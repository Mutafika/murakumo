//! エネルギーシールドマテリアル
//!
//! ビルボードクワッド上でレイマーチした球体に
//! 六角グリッドのエネルギーシールドを描画する。
//! ヒットエフェクトの波紋やフレネルグローを含む。

use super::Material;

/// シールドのパラメータ
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShieldParams {
    /// シールドの色 (RGBA)
    pub color: [f32; 4],
    /// 六角グリッドのスケール (default: 8.0)
    pub hex_scale: f32,
    /// ヒットポイント (UV座標, default: [0.5, 0.5])
    pub hit_point: [f32; 2],
    /// ヒット発生時刻 (default: -10.0 = 非アクティブ)
    pub hit_time: f32,
    /// 全体の不透明度 (default: 0.6)
    pub opacity: f32,
    /// パルス速度 (default: 2.0)
    pub pulse_speed: f32,
    pub _pad: [f32; 2],
}

impl Default for ShieldParams {
    fn default() -> Self {
        Self {
            color: [0.2, 0.6, 1.0, 1.0],
            hex_scale: 8.0,
            hit_point: [0.5, 0.5],
            hit_time: -10.0,
            opacity: 0.6,
            pulse_speed: 2.0,
            _pad: [0.0; 2],
        }
    }
}

pub struct Shield {
    pub params: ShieldParams,
}

impl Default for Shield {
    fn default() -> Self {
        Self { params: ShieldParams::default() }
    }
}

impl Material for Shield {
    fn shader_source(&self) -> &str {
        include_str!("../../shaders/shield.wgsl")
    }

    fn uniform_bytes(&self) -> Vec<u8> {
        bytemuck::bytes_of(&self.params).to_vec()
    }
}
