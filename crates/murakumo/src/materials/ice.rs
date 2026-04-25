//! 氷マテリアル
//!
//! サブサーフェススキャタリング + フロストパターン。
//! 内部散乱で青白く光る半透明な氷。

use super::Material;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct IceParams {
    /// SSS強度 (default: 0.8)
    pub sss_intensity: f32,
    /// フロストの密度 (default: 0.5)
    pub frost_density: f32,
    /// 透明度 (default: 0.7)
    pub clarity: f32,
    pub _pad: f32,
    /// 氷の色 (default: [0.7, 0.85, 0.95, 1.0])
    pub ice_color: [f32; 4],
}

impl Default for IceParams {
    fn default() -> Self {
        Self {
            sss_intensity: 0.8,
            frost_density: 0.5,
            clarity: 0.7,
            _pad: 0.0,
            ice_color: [0.7, 0.85, 0.95, 1.0],
        }
    }
}

pub struct Ice {
    pub params: IceParams,
}

impl Default for Ice {
    fn default() -> Self {
        Self { params: IceParams::default() }
    }
}

impl Material for Ice {
    fn shader_source(&self) -> &str {
        "// TODO: ice shader"
    }
    fn uniform_bytes(&self) -> Vec<u8> {
        bytemuck::bytes_of(&self.params).to_vec()
    }
}
