//! Murakumo（叢雲） — Seimei上のマテリアル/エフェクト集
//!
//! Seimeiが提供する3Dレンダリング基盤の上に、
//! 再利用可能なプロシージャルマテリアルとエフェクトを提供する。
//!
//! ```rust,ignore
//! let mut mat_pass = murakumo::MaterialPass::new(&renderer);
//! mat_pass.update_camera(&queue, &cam_uniform, time);
//! mat_pass.update_lights(&queue, &lights);
//! mat_pass.upload_params(&queue);
//! mat_pass.render(&device, &queue, &mut render_pass, &draws);
//! ```

pub mod materials;
pub mod effect;
pub mod pass;
pub mod params;

pub use materials::Material;
pub use pass::{
    MaterialPass, MaterialDraw, LayerInstance, MatParamsUniform,
    MATERIAL_COUNT, MATERIAL_NAMES, is_transparent,
};
pub use params::{ParamSpec, PARAMS_PER_MATERIAL, material_params, default_values};
