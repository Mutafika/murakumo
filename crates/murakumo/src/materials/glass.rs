//! ガラスキューブマテリアル — SDF角丸 + リムライト + スペキュラストリーク

use super::Material;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GlassParams {
    pub border_color: [f32; 4],
    pub corner_radius: f32,
    pub opacity: f32,
    pub _pad: [f32; 2],
}

impl Default for GlassParams {
    fn default() -> Self {
        Self {
            border_color: [0.4, 0.6, 1.0, 1.0],
            corner_radius: 0.06,
            opacity: 0.75,
            _pad: [0.0; 2],
        }
    }
}

pub struct Glass {
    pub params: GlassParams,
}

impl Default for Glass {
    fn default() -> Self {
        Self { params: GlassParams::default() }
    }
}

impl Material for Glass {
    fn shader_source(&self) -> &str {
        // TODO: portfolio-3dのGlass Cubeシェーダーから移植
        ""
    }
    fn uniform_bytes(&self) -> Vec<u8> {
        bytemuck::bytes_of(&self.params).to_vec()
    }
}
