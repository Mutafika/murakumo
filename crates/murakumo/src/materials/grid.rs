//! サイバーグリッドマテリアル — 無限フロア + パースペクティブグリッド

use super::Material;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GridParams {
    pub line_color: [f32; 4],
    pub grid_spacing: f32,
    pub line_width: f32,
    pub fade_start: f32,
    pub fade_end: f32,
}

impl Default for GridParams {
    fn default() -> Self {
        Self {
            line_color: [0.3, 0.35, 0.5, 0.4],
            grid_spacing: 1.0,
            line_width: 0.02,
            fade_start: 4.0,
            fade_end: 18.0,
        }
    }
}

pub struct Grid {
    pub params: GridParams,
}

impl Default for Grid {
    fn default() -> Self {
        Self { params: GridParams::default() }
    }
}

impl Material for Grid {
    fn shader_source(&self) -> &str {
        // TODO: portfolio-3dのFloor Gridシェーダーから移植
        ""
    }
    fn uniform_bytes(&self) -> Vec<u8> {
        bytemuck::bytes_of(&self.params).to_vec()
    }
}
