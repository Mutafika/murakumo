//! 竜巻マテリアル
//!
//! ボリュメトリック渦巻き。螺旋状の風と巻き上がる塵。

use super::Material;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TornadoParams {
    /// 渦の回転速度 (default: 3.0)
    pub rotation_speed: f32,
    /// 竜巻の幅 (default: 0.3)
    pub width: f32,
    /// 巻き上げの強さ (default: 1.0)
    pub updraft: f32,
    pub _pad: f32,
    /// 塵の色 (default: [0.35, 0.3, 0.25, 1.0])
    pub dust_color: [f32; 4],
}

impl Default for TornadoParams {
    fn default() -> Self {
        Self {
            rotation_speed: 3.0,
            width: 0.3,
            updraft: 1.0,
            _pad: 0.0,
            dust_color: [0.35, 0.3, 0.25, 1.0],
        }
    }
}

pub struct Tornado {
    pub params: TornadoParams,
}

impl Default for Tornado {
    fn default() -> Self {
        Self { params: TornadoParams::default() }
    }
}

impl Material for Tornado {
    fn shader_source(&self) -> &str {
        "// TODO: tornado shader"
    }
    fn uniform_bytes(&self) -> Vec<u8> {
        bytemuck::bytes_of(&self.params).to_vec()
    }
}
