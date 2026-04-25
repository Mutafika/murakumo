//! 雲マテリアル
//!
//! ボリュメトリックレイマーチによるリアルな雲。
//! ライトスキャタリング + シルバーライニング。

use super::Material;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CloudParams {
    /// 雲の密度 (default: 1.0)
    pub density: f32,
    /// 散乱係数 (default: 0.6)
    pub scattering: f32,
    /// 風速 (default: 0.2)
    pub wind_speed: f32,
    pub _pad: f32,
    /// 雲の色 (default: [0.95, 0.95, 0.97, 1.0])
    pub cloud_color: [f32; 4],
}

impl Default for CloudParams {
    fn default() -> Self {
        Self {
            density: 1.0,
            scattering: 0.6,
            wind_speed: 0.2,
            _pad: 0.0,
            cloud_color: [0.95, 0.95, 0.97, 1.0],
        }
    }
}

pub struct Cloud {
    pub params: CloudParams,
}

impl Default for Cloud {
    fn default() -> Self {
        Self { params: CloudParams::default() }
    }
}

impl Material for Cloud {
    fn shader_source(&self) -> &str {
        "// TODO: cloud shader"
    }
    fn uniform_bytes(&self) -> Vec<u8> {
        bytemuck::bytes_of(&self.params).to_vec()
    }
}
