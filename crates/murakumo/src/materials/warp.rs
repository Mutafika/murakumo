//! 空間歪みマテリアル
//!
//! ビルボードクワッド上でスパイラルツイストと
//! 色収差を伴う空間歪みエフェクトを描画する。

use super::Material;

/// ワープのパラメータ
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WarpParams {
    /// 歪みの中心 (UV座標, default: [0.5, 0.5])
    pub center: [f32; 2],
    /// 歪みの強さ (default: 1.0)
    pub strength: f32,
    /// 歪み半径 (default: 0.4)
    pub radius: f32,
    /// 回転速度 (default: 1.5)
    pub rotation_speed: f32,
    pub _pad0: [f32; 3],
    /// 歪みの色 (RGBA, default: [0.5, 0.2, 1.0, 1.0])
    pub distortion_color: [f32; 4],
}

impl Default for WarpParams {
    fn default() -> Self {
        Self {
            center: [0.5, 0.5],
            strength: 1.0,
            radius: 0.4,
            rotation_speed: 1.5,
            _pad0: [0.0; 3],
            distortion_color: [0.5, 0.2, 1.0, 1.0],
        }
    }
}

pub struct Warp {
    pub params: WarpParams,
}

impl Default for Warp {
    fn default() -> Self {
        Self { params: WarpParams::default() }
    }
}

impl Material for Warp {
    fn shader_source(&self) -> &str {
        include_str!("../../shaders/warp.wgsl")
    }

    fn uniform_bytes(&self) -> Vec<u8> {
        bytemuck::bytes_of(&self.params).to_vec()
    }
}
