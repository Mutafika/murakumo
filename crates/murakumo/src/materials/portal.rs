//! ポータルマテリアル — ノイズ渦 + スキャンライン + フレーム

use super::Material;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PortalParams {
    pub glow_color: [f32; 4],
    pub border_color: [f32; 4],
    pub energy: f32,
    pub _pad: [f32; 3],
}

impl Default for PortalParams {
    fn default() -> Self {
        Self {
            glow_color: [0.3, 0.5, 1.0, 0.5],
            border_color: [0.5, 0.7, 1.0, 1.0],
            energy: 0.3,
            _pad: [0.0; 3],
        }
    }
}

pub struct Portal {
    pub params: PortalParams,
}

impl Default for Portal {
    fn default() -> Self {
        Self { params: PortalParams::default() }
    }
}

impl Material for Portal {
    fn shader_source(&self) -> &str {
        // TODO: portfolio-3dのPortalシェーダーから移植
        ""
    }
    fn uniform_bytes(&self) -> Vec<u8> {
        bytemuck::bytes_of(&self.params).to_vec()
    }
}
