//! 煙/霧マテリアル
//!
//! Billboard quad上にレイヤードFBMノイズベースの
//! ボリューメトリックな煙エフェクトを描画する。

use super::Material;

/// 煙のパラメータ
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SmokeParams {
    /// 煙の密度 (default: 1.2)
    pub density: f32,
    /// スクロール速度 (default: 0.5)
    pub scroll_speed: f32,
    /// ディテールスケール (default: 4.0)
    pub detail_scale: f32,
    /// 全体の不透明度 (default: 0.8)
    pub opacity: f32,
    /// 煙の色 RGBA (default: [0.7, 0.7, 0.75, 1.0])
    pub color: [f32; 4],
}

impl Default for SmokeParams {
    fn default() -> Self {
        Self {
            density: 1.2,
            scroll_speed: 0.5,
            detail_scale: 4.0,
            opacity: 0.8,
            color: [0.7, 0.7, 0.75, 1.0],
        }
    }
}

pub struct Smoke {
    pub params: SmokeParams,
}

impl Default for Smoke {
    fn default() -> Self {
        Self { params: SmokeParams::default() }
    }
}

impl Material for Smoke {
    fn shader_source(&self) -> &str {
        include_str!("../../shaders/smoke.wgsl")
    }

    fn uniform_bytes(&self) -> Vec<u8> {
        bytemuck::bytes_of(&self.params).to_vec()
    }
}
