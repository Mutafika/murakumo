//! 溶岩マテリアル
//!
//! 割れ目から赤熱した光が漏れる溶岩表面。
//! ボロノイベースの亀裂パターン + 発光。

use super::Material;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LavaParams {
    /// 発光強度 (default: 2.0)
    pub glow_intensity: f32,
    /// 亀裂の速度 (default: 0.3)
    pub flow_speed: f32,
    /// 亀裂の幅 (default: 0.15)
    pub crack_width: f32,
    pub _pad: f32,
    /// 溶岩の色 (default: [1.0, 0.3, 0.0, 1.0])
    pub lava_color: [f32; 4],
    /// 岩の色 (default: [0.08, 0.06, 0.05, 1.0])
    pub rock_color: [f32; 4],
}

impl Default for LavaParams {
    fn default() -> Self {
        Self {
            glow_intensity: 2.0,
            flow_speed: 0.3,
            crack_width: 0.15,
            _pad: 0.0,
            lava_color: [1.0, 0.3, 0.0, 1.0],
            rock_color: [0.08, 0.06, 0.05, 1.0],
        }
    }
}

pub struct Lava {
    pub params: LavaParams,
}

impl Default for Lava {
    fn default() -> Self {
        Self { params: LavaParams::default() }
    }
}

impl Material for Lava {
    fn shader_source(&self) -> &str {
        "// TODO: lava shader"
    }
    fn uniform_bytes(&self) -> Vec<u8> {
        bytemuck::bytes_of(&self.params).to_vec()
    }
}
