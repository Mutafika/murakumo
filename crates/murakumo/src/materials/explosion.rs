//! 爆発マテリアル
//!
//! Fire+Smokeの複合ボリュームエフェクト。
//! 中心が白熱 → オレンジ炎 → 外縁に黒煙。

use super::Material;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ExplosionParams {
    /// 爆発の強度 (default: 2.0)
    pub intensity: f32,
    /// 膨張速度 (default: 1.5)
    pub expansion_speed: f32,
    /// 煙の量 (default: 0.6)
    pub smoke_amount: f32,
    pub _pad: f32,
    /// 爆発の核色 (default: [1.0, 0.95, 0.8, 1.0])
    pub core_color: [f32; 4],
}

impl Default for ExplosionParams {
    fn default() -> Self {
        Self {
            intensity: 2.0,
            expansion_speed: 1.5,
            smoke_amount: 0.6,
            _pad: 0.0,
            core_color: [1.0, 0.95, 0.8, 1.0],
        }
    }
}

pub struct Explosion {
    pub params: ExplosionParams,
}

impl Default for Explosion {
    fn default() -> Self {
        Self { params: ExplosionParams::default() }
    }
}

impl Material for Explosion {
    fn shader_source(&self) -> &str {
        "// TODO: explosion shader"
    }
    fn uniform_bytes(&self) -> Vec<u8> {
        bytemuck::bytes_of(&self.params).to_vec()
    }
}
