use glam::{Mat4, Vec3};
use sabitori::*;
use sabitori_widgets::SliderState;
use seimei::procedural;
use seimei::quality::{MsaaSamples, QualityPreset, QualitySettings, ShadowQuality};
use seimei::{CameraUniform, InstanceData, Light, LightUniform, Renderer};
use murakumo::{
    MaterialPass, MaterialDraw, LayerInstance, MATERIAL_COUNT, MATERIAL_NAMES,
    material_params, PARAMS_PER_MATERIAL,
};
use web_time::Instant;
use wgpu::util::DeviceExt;

// ── Constants ──

const GRID_COLS: usize = 8;
const GRID_ROWS: usize = 3;
const SPACING: f32 = 1.8;
const FIELD_INDEX: usize = 22;

// ── Background camera uniform ──

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BgCameraUniform {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    eye_pos: [f32; 3],
    time: f32,
}

// PbrCameraUniform, LayerInstance, MatParamsUniform — moved to murakumo crate

// ── Background pass resources ──

struct BackgroundPass {
    pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    lights_bind_group: wgpu::BindGroup,
}

impl BackgroundPass {
    fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let shader_src = include_str!("../shaders/gallery.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bg_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bg_camera_buffer"),
            size: std::mem::size_of::<BgCameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bg_camera_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let camera_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_camera_bg"),
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let lights_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bg_lights_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let dummy_lights = [0u8; 96];
        let lights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bg_lights_buffer"),
            contents: &dummy_lights,
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let lights_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_lights_bg"),
            layout: &lights_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: lights_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bg_pipeline_layout"),
            bind_group_layouts: &[&camera_bgl, &lights_bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("bg_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_bg"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_bg"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            camera_buffer,
            camera_bind_group: camera_bg,
            lights_bind_group: lights_bg,
        }
    }
}

// PbrMaterialPass — replaced by murakumo::MaterialPass

// ── Orbit Camera ──

struct OrbitCamera {
    yaw: f32,
    pitch: f32,
    distance: f32,
    target: Vec3,
    fov: f32,
    aspect: f32,
    dragging: bool,
    last_mouse: (f32, f32),
}

impl OrbitCamera {
    fn new() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.3,
            distance: 8.0,
            target: Vec3::ZERO,
            fov: std::f32::consts::FRAC_PI_3,
            aspect: 16.0 / 9.0,
            dragging: false,
            last_mouse: (0.0, 0.0),
        }
    }

    fn position(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        self.target + Vec3::new(sy * cp, sp, cy * cp) * self.distance
    }

    fn view_proj(&self) -> Mat4 {
        let pos = self.position();
        let view = Mat4::look_at_rh(pos, self.target, Vec3::Y);
        let proj = Mat4::perspective_rh(self.fov, self.aspect, 0.1, 100.0);
        proj * view
    }

    fn view_matrix(&self) -> Mat4 {
        let pos = self.position();
        Mat4::look_at_rh(pos, self.target, Vec3::Y)
    }

    fn on_drag_start(&mut self, x: f32, y: f32) {
        self.dragging = true;
        self.last_mouse = (x, y);
    }

    fn on_drag_move(&mut self, x: f32, y: f32) {
        if !self.dragging {
            return;
        }
        let dx = x - self.last_mouse.0;
        let dy = y - self.last_mouse.1;
        self.last_mouse = (x, y);

        let sensitivity = 0.005;
        self.yaw -= dx * sensitivity;
        self.pitch = (self.pitch + dy * sensitivity)
            .clamp(-std::f32::consts::FRAC_PI_2 * 0.8, std::f32::consts::FRAC_PI_2 * 0.8);
    }

    fn on_drag_end(&mut self) {
        self.dragging = false;
    }

    fn on_scroll(&mut self, delta: f32) {
        self.distance = (self.distance + delta * 0.01).clamp(3.0, 20.0);
    }

    fn resize(&mut self, w: u32, h: u32) {
        if h > 0 {
            self.aspect = w as f32 / h as f32;
        }
    }
}

// ── Helpers ──

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Format a slider value for compact display (3 chars + sign + dot).
fn format_value(v: f32) -> String {
    let abs = v.abs();
    if abs >= 100.0 {
        format!("{:>5.0}", v)
    } else if abs >= 10.0 {
        format!("{:>5.1}", v)
    } else if abs >= 1.0 {
        format!("{:>5.2}", v)
    } else {
        format!("{:>5.3}", v)
    }
}

fn mesh_id_for_material(mat_idx: usize) -> &'static str {
    match mat_idx {
        0 => "icosphere_hd",              // Bubble — high-poly for transparency
        1 | 4 | 9 | 10 | 12 => "icosphere", // Glass, Water, Crystal, Metal, Shield
        2 | 3 | 8 | 13 => "cube",         // Portal, Grid, Hologram, Dissolve
        5 | 6 => "icosphere",             // Fire, Smoke — volumetric ray march
        7 => "plane",                      // Aurora
        11 => "torus",                     // Neon
        14 => "icosphere",                // Lightning — volumetric arcs
        15 => "icosphere",                // Lava
        16 => "icosphere",                // Ice
        17 | 18 | 19 => "icosphere",      // Cloud, Explosion, Tornado — volumetric
        20 => "icosphere",                // Skin
        21 => "icosphere",                // Rock
        22 => "icosphere",                // Field (placeholder for grid view)
        _ => "cube",
    }
}

/// Build per-material instance with PBR properties + material kind
fn build_instance(mat_idx: usize, time: f32) -> InstanceData {
    let col = mat_idx % GRID_COLS;
    let row = mat_idx / GRID_COLS;

    let x = (col as f32 - (GRID_COLS as f32 - 1.0) * 0.5) * SPACING;
    let y = ((GRID_ROWS - 1 - row) as f32 - (GRID_ROWS as f32 - 1.0) * 0.5) * SPACING;
    let bob = (time * 1.0 + mat_idx as f32 * 0.4).sin() * 0.04;

    let (scale, rot_speed, tilt) = match mat_idx {
        // Spheres (natural/physical)
        0 => (Vec3::splat(0.55), 0.15, 0.0),   // Bubble
        1 => (Vec3::splat(0.55), 0.12, 0.0),   // Glass
        4 => (Vec3::splat(0.55), 0.1, 0.0),    // Water
        9 => (Vec3::splat(0.55), 0.18, 0.1),   // Crystal
        10 => (Vec3::splat(0.55), 0.1, 0.0),   // Metal
        12 => (Vec3::splat(0.55), 0.15, 0.05), // Shield
        15 => (Vec3::splat(0.55), 0.1, 0.0),   // Lava
        16 => (Vec3::splat(0.55), 0.12, 0.05), // Ice
        20 => (Vec3::splat(0.55), 0.08, 0.0),  // Skin
        21 => (Vec3::splat(0.55), 0.06, 0.0),  // Rock
        22 => (Vec3::splat(0.55), 0.05, 0.0),  // Field (thumbnail)
        // Cubes
        2 => (Vec3::splat(1.1), 0.35, 0.0),    // Portal
        3 => (Vec3::splat(1.0), 0.1, 0.4),     // Grid
        8 => (Vec3::splat(1.0), 0.4, 0.05),    // Hologram
        13 => (Vec3::splat(1.0), 0.3, 0.1),    // Dissolve
        // Volumetric spheres
        5 => (Vec3::splat(0.7), 0.05, 0.0),    // Fire
        6 => (Vec3::splat(0.7), 0.03, 0.0),    // Smoke
        17 => (Vec3::splat(0.7), 0.02, 0.0),   // Cloud
        18 => (Vec3::splat(0.7), 0.0, 0.0),    // Explosion
        19 => (Vec3::splat(0.7), 0.0, 0.0),    // Tornado
        // Flat/special
        7 => (Vec3::new(1.3, 1.0, 1.3), 0.08, 0.3),  // Aurora
        14 => (Vec3::splat(0.65), 0.0, 0.0),    // Lightning — volumetric sphere
        11 => (Vec3::splat(1.8), 0.2, 0.0),    // Neon
        _ => (Vec3::splat(1.0), 0.2, 0.0),
    };

    let yaw = time * rot_speed + mat_idx as f32 * 0.5;
    let model = Mat4::from_translation(Vec3::new(x, y + bob, 0.0))
        * Mat4::from_rotation_y(yaw)
        * Mat4::from_rotation_x(tilt)
        * Mat4::from_scale(scale);

    let cols = model.to_cols_array_2d();

    let (metallic, roughness) = match mat_idx {
        0 => (0.0, 0.1),   // Bubble
        1 => (0.1, 0.05),  // Glass
        2 => (0.0, 0.3),   // Portal
        3 => (0.3, 0.5),   // Grid
        4 => (0.0, 0.15),  // Water
        5 => (0.0, 0.8),   // Fire
        6 => (0.0, 0.9),   // Smoke
        7 => (0.0, 0.4),   // Aurora
        8 => (0.5, 0.2),   // Hologram
        9 => (0.2, 0.1),   // Crystal
        10 => (1.0, 0.3),  // Metal
        11 => (0.0, 0.1),  // Neon
        12 => (0.3, 0.2),  // Shield
        13 => (0.0, 0.4),  // Dissolve
        14 => (0.0, 0.5),  // Lightning
        15 => (0.0, 0.8),  // Lava
        16 => (0.1, 0.15), // Ice
        17 => (0.0, 0.9),  // Cloud
        18 => (0.0, 0.8),  // Explosion
        19 => (0.0, 0.7),  // Tornado
        20 => (0.0, 0.4),  // Skin
        21 => (0.0, 0.9),  // Rock
        22 => (0.0, 0.8),  // Field
        _ => (0.0, 0.5),
    };

    InstanceData {
        model: cols,
        color: [1.0, 1.0, 1.0, 1.0],
        // material.z = kind (selects which procedural effect)
        material: [metallic, roughness, mat_idx as f32, 0.0],
    }
}

// Light types — now using seimei::LightUniform directly

// ── Gallery App ──

#[derive(Clone)]
enum GalleryState {
    Grid,
    TransitionToDetail { index: usize, progress: f32 },
    Detail { index: usize },
    TransitionToGrid { from_index: usize, progress: f32 },
}

struct GalleryApp {
    camera: OrbitCamera,
    renderer: Option<Renderer>,
    mat_pass: Option<MaterialPass>,
    bg_pass: Option<BackgroundPass>,
    time: f32,
    start: Instant,
    width: f32,
    height: f32,
    drag_distance: f32,
    state: GalleryState,
    // Saved grid camera for transitions
    grid_camera_yaw: f32,
    grid_camera_pitch: f32,
    grid_camera_distance: f32,
    click_pos: (f32, f32),
    // Per-material tunable params + slider widgets (one per material × param)
    material_param_values: Vec<[f32; PARAMS_PER_MATERIAL]>,
    material_sliders: Vec<Vec<SliderState>>,
    /// What secondary material to overlay when this material is shown in detail.
    /// `None` = no second layer.
    material_layer2: Vec<Option<usize>>,
    /// Alpha mix for the secondary layer per material (0..1).
    material_layer2_alpha: Vec<f32>,
    /// Slider state for the layer2 alpha row (one per material).
    material_layer2_alpha_sliders: Vec<SliderState>,
    /// Drag target: identifies which slider is being dragged.
    /// `Slot::Primary(mat_idx, param_idx)` — a primary-material param.
    /// `Slot::Secondary(layer2_kind, param_idx)` — a secondary-layer param (uses layer2 kind's slot).
    /// `Slot::LayerAlpha(mat_idx)` — the layer-2 alpha mix slider.
    dragging_slider: Option<DragSlot>,
}

#[derive(Clone, Copy, Debug)]
enum DragSlot {
    Primary(usize, usize),
    Secondary(usize, usize),
    LayerAlpha(usize),
}

// ── Parameter drawer layout (detail mode) ──

const DRAWER_W: f32 = 280.0;
const DRAWER_RIGHT_MARGIN: f32 = 12.0;
const DRAWER_TOP: f32 = 64.0;
const DRAWER_PAD: f32 = 14.0;
const SLIDER_LABEL_W: f32 = 78.0;
const SLIDER_TRACK_W: f32 = 130.0;
const SLIDER_VALUE_W: f32 = 48.0;
const SLIDER_GAP: f32 = 8.0;
const SLIDER_ROW_H: f32 = 24.0;
const SLIDER_ROW_STRIDE: f32 = 30.0;
const DRAWER_TITLE_H: f32 = 24.0;
const DRAWER_TITLE_GAP: f32 = 10.0;

/// Returns true if this material should be shown as a landscape (huge sphere = ground)
fn is_landscape_material(index: usize) -> bool {
    matches!(index, 4) // Water only — flat surface works for water
}

/// Mesh to use in detail view
fn detail_mesh_id(index: usize) -> &'static str {
    match index {
        15 | 21 => "rock_mesh",
        FIELD_INDEX => "terrain",  // Field: terrain mesh (water drawn separately)
        _ if is_landscape_material(index) => "icosphere_huge",
        _ => mesh_id_for_material(index),
    }
}

/// Per-material detail view settings
fn detail_scale(index: usize) -> Vec3 {
    match index {
        // Landscape: huge sphere below camera
        _ if is_landscape_material(index) => Vec3::splat(20.0),
        // Volumetric: large bounding spheres
        5 => Vec3::new(2.0, 3.0, 2.0),    // Fire
        6 => Vec3::new(2.5, 3.0, 2.5),    // Smoke
        17 => Vec3::new(3.0, 2.0, 3.0),   // Cloud
        18 => Vec3::new(2.5, 2.5, 2.5),   // Explosion
        19 => Vec3::new(1.5, 3.5, 1.5),   // Tornado
        14 => Vec3::new(2.0, 3.0, 2.0),   // Lightning — tall volumetric column
        // Field: terrain already sized at 8.0, just scale 1:1
        FIELD_INDEX => Vec3::splat(1.0),
        // Surface materials: large sphere to fill view
        10 | 15 | 16 | 20 | 21 => Vec3::splat(2.0),
        // Natural shapes
        0 | 1 | 9 | 12 => Vec3::splat(1.5), // Bubble, Glass, Crystal, Shield
        2 => Vec3::splat(2.0),             // Portal
        8 => Vec3::splat(2.0),             // Hologram
        11 => Vec3::splat(2.5),            // Neon
        _ => Vec3::splat(2.0),
    }
}

fn detail_camera_distance(index: usize) -> f32 {
    match index {
        FIELD_INDEX => 6.0,
        _ if is_landscape_material(index) => 2.5,
        5 | 6 | 14 | 19 => 6.0,
        17 | 18 => 6.0,
        _ => 4.0,
    }
}

fn detail_camera_pitch(index: usize) -> f32 {
    if index == FIELD_INDEX {
        0.5 // Look down at the field
    } else if is_landscape_material(index) {
        0.15
    } else {
        0.25
    }
}

/// Where the camera looks at (orbit target)
fn detail_camera_target(index: usize) -> f32 {
    if is_landscape_material(index) {
        0.3 // Look at a point slightly above the ground surface
    } else {
        match index {
            5 => 0.5,   // Fire: look at flame center
            19 => 0.5,  // Tornado
            18 => 0.5,  // Explosion
            _ => 0.0,
        }
    }
}

/// Where the mesh center is placed (Y offset)
fn detail_y_offset(index: usize) -> f32 {
    if is_landscape_material(index) {
        -20.0 // Sphere center far below — top of sphere is at Y=0
    } else {
        match index {
            5 => -0.5,
            19 => 0.5,
            18 => 0.5,
            _ => 0.0,
        }
    }
}

impl GalleryApp {
    fn new() -> Self {
        let mut material_param_values = Vec::with_capacity(MATERIAL_COUNT);
        let mut material_sliders = Vec::with_capacity(MATERIAL_COUNT);
        for k in 0..MATERIAL_COUNT {
            let defaults = murakumo::default_values(k);
            material_param_values.push(defaults);
            let specs = material_params(k);
            let sliders: Vec<SliderState> = specs
                .iter()
                .enumerate()
                .map(|(i, spec)| SliderState::from_ranged(defaults[i], spec.min, spec.max))
                .collect();
            material_sliders.push(sliders);
        }

        Self {
            camera: OrbitCamera::new(),
            renderer: None,
            mat_pass: None,
            bg_pass: None,
            time: 0.0,
            start: Instant::now(),
            width: 960.0,
            height: 640.0,
            drag_distance: 0.0,
            state: GalleryState::Grid,
            grid_camera_yaw: 0.0,
            grid_camera_pitch: 0.3,
            grid_camera_distance: 8.0,
            click_pos: (0.0, 0.0),
            material_param_values,
            material_sliders,
            material_layer2: vec![None; MATERIAL_COUNT],
            material_layer2_alpha: vec![0.7; MATERIAL_COUNT],
            material_layer2_alpha_sliders: (0..MATERIAL_COUNT)
                .map(|_| SliderState::from_ranged(0.7, 0.0, 1.0))
                .collect(),
            dragging_slider: None,
        }
    }

    /// Material being shown in the param drawer (only when fully in detail).
    fn drawer_material(&self) -> Option<usize> {
        match &self.state {
            GalleryState::Detail { index } => Some(*index),
            _ => None,
        }
    }

    /// Storage slot whose params actually drive rendering for `material_idx`.
    /// Most materials map to themselves; Field reuses Rock's slot since the
    /// gallery renders the field terrain with the Rock material kind.
    fn storage_index(material_idx: usize) -> usize {
        if material_idx == FIELD_INDEX { 21 } else { material_idx }
    }

    /// Drawer row layout for the given primary material. Returns the ordered
    /// list of rows (uniform stride). Used by both `view()` and hit-testing
    /// so they stay in sync.
    fn drawer_rows(&self, mat_idx: usize) -> Vec<DrawerRow> {
        let mut rows: Vec<DrawerRow> = Vec::new();
        let primary_specs = material_params(mat_idx);
        for i in 0..primary_specs.len() {
            rows.push(DrawerRow::Primary(i));
        }
        rows.push(DrawerRow::LayerCycle); // toggle / cycle button
        if let Some(kind2) = self.material_layer2[mat_idx] {
            rows.push(DrawerRow::LayerAlpha);
            let secondary_specs = material_params(kind2);
            for i in 0..secondary_specs.len() {
                rows.push(DrawerRow::Secondary(kind2, i));
            }
        }
        rows
    }

    fn drawer_row_y(&self, row_idx: usize) -> f32 {
        DRAWER_TOP + DRAWER_PAD + DRAWER_TITLE_H + DRAWER_TITLE_GAP
            + (row_idx as f32) * SLIDER_ROW_STRIDE
    }

    fn slider_track_x(&self) -> f32 {
        let drawer_x = self.width - DRAWER_W - DRAWER_RIGHT_MARGIN;
        drawer_x + DRAWER_PAD + SLIDER_LABEL_W + SLIDER_GAP
    }

    /// Hit-test the entire drawer; returns what was clicked.
    fn drawer_hit_test(&self, mat_idx: usize, mouse: (f32, f32)) -> Option<DrawerHit> {
        let drawer_x = self.width - DRAWER_W - DRAWER_RIGHT_MARGIN;
        let row_left = drawer_x + DRAWER_PAD;
        let row_w = DRAWER_W - DRAWER_PAD * 2.0;
        if mouse.0 < row_left || mouse.0 > row_left + row_w {
            return None;
        }
        for (i, row) in self.drawer_rows(mat_idx).into_iter().enumerate() {
            let ty = self.drawer_row_y(i);
            if mouse.1 < ty || mouse.1 > ty + SLIDER_ROW_H {
                continue;
            }
            return Some(match row {
                DrawerRow::Primary(p) => DrawerHit::Slider(DragSlot::Primary(mat_idx, p)),
                DrawerRow::LayerCycle => DrawerHit::CycleLayer,
                DrawerRow::LayerAlpha => DrawerHit::Slider(DragSlot::LayerAlpha(mat_idx)),
                DrawerRow::Secondary(k2, p) => DrawerHit::Slider(DragSlot::Secondary(k2, p)),
            });
        }
        None
    }

    /// Track screen-space `(x, y, w, h)` for the slider in this drag slot,
    /// based on its current row position.
    fn slot_track_xywh(&self, slot: DragSlot) -> Option<(f32, f32, f32, f32)> {
        let mat_idx = self.drawer_material()?;
        for (i, row) in self.drawer_rows(mat_idx).into_iter().enumerate() {
            let matches = match (row, slot) {
                (DrawerRow::Primary(p), DragSlot::Primary(_, sp)) => p == sp,
                (DrawerRow::LayerAlpha, DragSlot::LayerAlpha(_)) => true,
                (DrawerRow::Secondary(rk, rp), DragSlot::Secondary(sk, sp)) => rk == sk && rp == sp,
                _ => false,
            };
            if matches {
                return Some((self.slider_track_x(), self.drawer_row_y(i), SLIDER_TRACK_W, SLIDER_ROW_H));
            }
        }
        None
    }

    /// Begin dragging a slider slot, snapping value to the press location.
    fn begin_slot_drag(&mut self, slot: DragSlot, mouse_x: f32) {
        let (tx, _, tw, _) = match self.slot_track_xywh(slot) {
            Some(v) => v,
            None => return,
        };
        match slot {
            DragSlot::Primary(mat_idx, _) => {
                let storage = Self::storage_index(mat_idx);
                if let DragSlot::Primary(_, p) = slot {
                    if let Some(s) = self.material_sliders[storage].get_mut(p) {
                        s.begin_drag(mouse_x, tx, tw);
                    }
                }
            }
            DragSlot::Secondary(k2, p) => {
                let storage = Self::storage_index(k2);
                if let Some(s) = self.material_sliders[storage].get_mut(p) {
                    s.begin_drag(mouse_x, tx, tw);
                }
            }
            DragSlot::LayerAlpha(mat_idx) => {
                self.material_layer2_alpha_sliders[mat_idx].begin_drag(mouse_x, tx, tw);
            }
        }
        self.sync_value_for_slot(slot);
    }

    fn drag_slot_to(&mut self, slot: DragSlot, mouse_x: f32) {
        let (tx, _, tw, _) = match self.slot_track_xywh(slot) {
            Some(v) => v,
            None => return,
        };
        match slot {
            DragSlot::Primary(mat_idx, p) => {
                let storage = Self::storage_index(mat_idx);
                if let Some(s) = self.material_sliders[storage].get_mut(p) {
                    s.drag_to(mouse_x, tx, tw);
                }
            }
            DragSlot::Secondary(k2, p) => {
                let storage = Self::storage_index(k2);
                if let Some(s) = self.material_sliders[storage].get_mut(p) {
                    s.drag_to(mouse_x, tx, tw);
                }
            }
            DragSlot::LayerAlpha(mat_idx) => {
                self.material_layer2_alpha_sliders[mat_idx].drag_to(mouse_x, tx, tw);
            }
        }
        self.sync_value_for_slot(slot);
    }

    fn end_slot_drag(&mut self, slot: DragSlot) {
        match slot {
            DragSlot::Primary(mat_idx, p) => {
                let storage = Self::storage_index(mat_idx);
                if let Some(s) = self.material_sliders[storage].get_mut(p) {
                    s.end_drag();
                }
            }
            DragSlot::Secondary(k2, p) => {
                let storage = Self::storage_index(k2);
                if let Some(s) = self.material_sliders[storage].get_mut(p) {
                    s.end_drag();
                }
            }
            DragSlot::LayerAlpha(mat_idx) => {
                self.material_layer2_alpha_sliders[mat_idx].end_drag();
            }
        }
    }

    fn sync_value_for_slot(&mut self, slot: DragSlot) {
        match slot {
            DragSlot::Primary(mat_idx, p) => {
                let spec = match material_params(mat_idx).get(p) {
                    Some(s) => *s,
                    None => return,
                };
                let storage = Self::storage_index(mat_idx);
                let v = self.material_sliders[storage][p].ranged(spec.min, spec.max);
                self.material_param_values[storage][p] = v;
            }
            DragSlot::Secondary(k2, p) => {
                let spec = match material_params(k2).get(p) {
                    Some(s) => *s,
                    None => return,
                };
                let storage = Self::storage_index(k2);
                let v = self.material_sliders[storage][p].ranged(spec.min, spec.max);
                self.material_param_values[storage][p] = v;
            }
            DragSlot::LayerAlpha(mat_idx) => {
                self.material_layer2_alpha[mat_idx] =
                    self.material_layer2_alpha_sliders[mat_idx].value();
            }
        }
    }

    /// Cycle the layer 2 selection: None → 0 → 1 → ... → MATERIAL_COUNT-1 → None.
    /// Skips the primary material itself (no Self+Self combos).
    fn cycle_layer2(&mut self, mat_idx: usize) {
        let next = match self.material_layer2[mat_idx] {
            None => Some(0),
            Some(k) => {
                let mut next = k + 1;
                if next == mat_idx { next += 1; }
                if next >= MATERIAL_COUNT { None } else { Some(next) }
            }
        };
        // If next == mat_idx after wrap-around, skip
        let next = match next {
            Some(k) if k == mat_idx => {
                if k + 1 >= MATERIAL_COUNT { None } else { Some(k + 1) }
            }
            other => other,
        };
        self.material_layer2[mat_idx] = next;
    }
}

#[derive(Clone, Copy, Debug)]
enum DrawerRow {
    Primary(usize),         // param_idx within primary material
    LayerCycle,             // layer 2 toggle / cycle button
    LayerAlpha,             // alpha slider for layer 2
    Secondary(usize, usize),// (layer2_kind, param_idx)
}

#[derive(Clone, Copy, Debug)]
enum DrawerHit {
    Slider(DragSlot),
    CycleLayer,
}

impl GalleryApp {

    /// Hit-test: find which material grid cell was clicked
    fn hit_test_grid(&self, screen_x: f32, screen_y: f32) -> Option<usize> {
        let vp = self.camera.view_proj();
        let mut best: Option<(usize, f32)> = None;

        for i in 0..MATERIAL_COUNT {
            let col = i % GRID_COLS;
            let row = i / GRID_COLS;
            let x = (col as f32 - (GRID_COLS as f32 - 1.0) * 0.5) * SPACING;
            let y = ((GRID_ROWS - 1 - row) as f32 - (GRID_ROWS as f32 - 1.0) * 0.5) * SPACING;

            let world = glam::Vec4::new(x, y, 0.0, 1.0);
            let clip = vp * world;
            if clip.w <= 0.0 { continue; }
            let ndc = clip.truncate() / clip.w;
            let sx = (ndc.x * 0.5 + 0.5) * self.width;
            let sy = (1.0 - (ndc.y * 0.5 + 0.5)) * self.height;

            let dist = ((screen_x - sx).powi(2) + (screen_y - sy).powi(2)).sqrt();
            let hit_radius = 50.0; // pixels
            if dist < hit_radius {
                if best.is_none() || dist < best.unwrap().1 {
                    best = Some((i, dist));
                }
            }
        }

        best.map(|(i, _)| i)
    }

    fn enter_detail(&mut self, index: usize) {
        self.grid_camera_yaw = self.camera.yaw;
        self.grid_camera_pitch = self.camera.pitch;
        self.grid_camera_distance = self.camera.distance;
        self.state = GalleryState::TransitionToDetail { index, progress: 0.0 };
    }

    fn enter_grid(&mut self) {
        if let GalleryState::Detail { index } = self.state {
            self.state = GalleryState::TransitionToGrid { from_index: index, progress: 0.0 };
        }
    }
}

// ── DeclarativeApp (2D UI Overlay) ──

impl DeclarativeApp for GalleryApp {
    fn title(&self) -> &str {
        "MURAKUMO \u{2014} Material Gallery"
    }
    fn size(&self) -> (f32, f32) {
        (960.0, 640.0)
    }

    fn fonts(&self) -> Vec<Vec<u8>> {
        vec![
            include_bytes!("../../../../sabitori/assets/fonts/Hack-Regular.ttf").to_vec(),
            include_bytes!("../../../../sabitori/assets/fonts/Hack-Bold.ttf").to_vec(),
        ]
    }

    fn tick(&mut self, dt: f32) {
        self.time = self.start.elapsed().as_secs_f32();

        // Animate state transitions
        let transition_speed = 3.0; // ~0.33 seconds
        let new_state = match &self.state {
            GalleryState::TransitionToDetail { index, progress } => {
                let p = (progress + dt * transition_speed).min(1.0);
                let ease = p * p * (3.0 - 2.0 * p); // smoothstep

                // Interpolate camera toward detail view
                let target_dist = detail_camera_distance(*index);
                self.camera.distance = lerp(self.grid_camera_distance, target_dist, ease);
                self.camera.pitch = lerp(self.grid_camera_pitch, detail_camera_pitch(*index), ease);
                let cam_target_y = detail_camera_target(*index);
                self.camera.target = Vec3::new(0.0, cam_target_y * ease, 0.0);

                if p >= 1.0 {
                    Some(GalleryState::Detail { index: *index })
                } else {
                    Some(GalleryState::TransitionToDetail { index: *index, progress: p })
                }
            }
            GalleryState::TransitionToGrid { from_index, progress } => {
                let p = (progress + dt * transition_speed).min(1.0);
                let ease = p * p * (3.0 - 2.0 * p);

                let from_dist = detail_camera_distance(*from_index);
                self.camera.distance = lerp(from_dist, self.grid_camera_distance, ease);
                self.camera.pitch = lerp(detail_camera_pitch(*from_index), self.grid_camera_pitch, ease);
                let cam_target_y = detail_camera_target(*from_index);
                self.camera.target = Vec3::new(0.0, cam_target_y * (1.0 - ease), 0.0);

                if p >= 1.0 {
                    Some(GalleryState::Grid)
                } else {
                    Some(GalleryState::TransitionToGrid { from_index: *from_index, progress: p })
                }
            }
            _ => None,
        };
        if let Some(s) = new_state {
            self.state = s;
        }
    }

    fn view(&self, ctx: &ViewContext) -> Element {
        let mut children = vec![];

        let is_detail = match &self.state {
            GalleryState::Detail { .. } => true,
            GalleryState::TransitionToDetail { progress, .. } => *progress > 0.5,
            GalleryState::TransitionToGrid { progress, .. } => *progress < 0.5,
            GalleryState::Grid => false,
        };

        if is_detail {
            // ── Detail mode UI ──
            let idx = match &self.state {
                GalleryState::Detail { index } => *index,
                GalleryState::TransitionToDetail { index, .. } => *index,
                GalleryState::TransitionToGrid { from_index, .. } => *from_index,
                _ => 0,
            };

            // Back button (top-left)
            let back = text("\u{2190} Back".to_string())
                .mono()
                .font_size(16.0)
                .color(Color::new(0.7, 0.8, 1.0, 0.8));
            children.push(div().pos(20.0, 14.0).child(back));

            // Material name (large, centered top)
            let name = MATERIAL_NAMES[idx];
            let name_label = text(name.to_string())
                .mono()
                .bold()
                .font_size(32.0)
                .color(Color::new(0.8, 0.85, 1.0, 0.9));
            let name_x = ctx.width / 2.0 - name.len() as f32 * 10.0;
            children.push(div().pos(name_x.max(0.0), 14.0).child(name_label));

            // ESC hint
            let esc = text("ESC to return".to_string())
                .mono()
                .font_size(12.0)
                .color(Color::new(0.4, 0.4, 0.5, 0.5));
            children.push(div().pos(20.0, ctx.height - 24.0).child(esc));

            // ── Param drawer (right side) ──
            // Lays out one row per `DrawerRow` returned by `drawer_rows()`,
            // so view + hit-testing stay in lockstep.
            let drawer_rows = self.drawer_rows(idx);
            if !drawer_rows.is_empty() {
                let drawer_x = ctx.width - DRAWER_W - DRAWER_RIGHT_MARGIN;
                let drawer_h = DRAWER_PAD * 2.0
                    + DRAWER_TITLE_H
                    + DRAWER_TITLE_GAP
                    + (drawer_rows.len() as f32) * SLIDER_ROW_STRIDE
                    - (SLIDER_ROW_STRIDE - SLIDER_ROW_H);

                // Background panel
                children.push(
                    div()
                        .pos(drawer_x, DRAWER_TOP)
                        .w(Px(DRAWER_W))
                        .h(Px(drawer_h))
                        .bg(Color::new(0.04, 0.05, 0.09, 0.85))
                        .border(1.0, Color::new(0.4, 0.5, 0.7, 0.4))
                        .rounded_px(8.0),
                );

                // Title
                children.push(
                    div()
                        .pos(drawer_x + DRAWER_PAD, DRAWER_TOP + DRAWER_PAD)
                        .w(Px(DRAWER_W - DRAWER_PAD * 2.0))
                        .h(Px(DRAWER_TITLE_H))
                        .child(
                            text("PARAMETERS".to_string())
                                .mono()
                                .bold()
                                .font_size(13.0)
                                .color(Color::new(0.7, 0.8, 1.0, 0.9)),
                        ),
                );

                let track_color = Color::new(0.2, 0.25, 0.35, 0.85);
                let fill_color = Color::new(0.55, 0.75, 1.0, 1.0);
                let knob_color = Color::new(0.9, 0.95, 1.0, 1.0);
                let text_color = Color::new(0.78, 0.85, 0.95, 0.95);
                let dim_text = Color::new(0.6, 0.7, 0.85, 0.85);
                let row_w = SLIDER_LABEL_W + SLIDER_TRACK_W + SLIDER_VALUE_W + SLIDER_GAP * 2.0;
                let primary_storage = Self::storage_index(idx);

                for (i, row) in drawer_rows.iter().enumerate() {
                    let row_y = self.drawer_row_y(i);
                    let row_x = drawer_x + DRAWER_PAD;
                    match row {
                        DrawerRow::Primary(p) => {
                            let spec = material_params(idx)[*p];
                            let value = self.material_param_values[primary_storage][*p];
                            let norm = self.material_sliders[primary_storage][*p].value();
                            let id = format!("p-{}-{}", idx, p);
                            let r = labeled_slider(
                                &id, spec.name, &format_value(value), norm,
                                SLIDER_LABEL_W, SLIDER_TRACK_W, SLIDER_VALUE_W,
                                text_color, track_color, fill_color, knob_color,
                            );
                            children.push(
                                div().pos(row_x, row_y).w(Px(row_w)).h(Px(SLIDER_ROW_H)).child(r),
                            );
                        }
                        DrawerRow::LayerCycle => {
                            let label = match self.material_layer2[idx] {
                                None => "Layer 2:  none  \u{25B6}".to_string(),
                                Some(k2) => format!("Layer 2:  {}  \u{25B6}", MATERIAL_NAMES[k2]),
                            };
                            children.push(
                                div()
                                    .pos(row_x, row_y)
                                    .w(Px(row_w))
                                    .h(Px(SLIDER_ROW_H))
                                    .bg(Color::new(0.1, 0.13, 0.2, 0.6))
                                    .rounded_px(4.0)
                                    .child(
                                        text(label)
                                            .mono()
                                            .font_size(12.0)
                                            .color(dim_text),
                                    ),
                            );
                        }
                        DrawerRow::LayerAlpha => {
                            let value = self.material_layer2_alpha[idx];
                            let norm = self.material_layer2_alpha_sliders[idx].value();
                            let id = format!("a-{}", idx);
                            let r = labeled_slider(
                                &id, "Mix", &format_value(value), norm,
                                SLIDER_LABEL_W, SLIDER_TRACK_W, SLIDER_VALUE_W,
                                text_color, track_color, fill_color, knob_color,
                            );
                            children.push(
                                div().pos(row_x, row_y).w(Px(row_w)).h(Px(SLIDER_ROW_H)).child(r),
                            );
                        }
                        DrawerRow::Secondary(k2, p) => {
                            let spec = material_params(*k2)[*p];
                            let storage = Self::storage_index(*k2);
                            let value = self.material_param_values[storage][*p];
                            let norm = self.material_sliders[storage][*p].value();
                            let id = format!("s-{}-{}-{}", idx, k2, p);
                            let r = labeled_slider(
                                &id, spec.name, &format_value(value), norm,
                                SLIDER_LABEL_W, SLIDER_TRACK_W, SLIDER_VALUE_W,
                                dim_text, track_color, fill_color, knob_color,
                            );
                            children.push(
                                div().pos(row_x, row_y).w(Px(row_w)).h(Px(SLIDER_ROW_H)).child(r),
                            );
                        }
                    }
                }
            }
        } else {
            // ── Grid mode UI ──
            let title = text("MURAKUMO \u{2014} Material Gallery".to_string())
                .mono()
                .bold()
                .font_size(24.0)
                .color(Color::new(0.7, 0.8, 1.0, 0.9));
            children.push(div().pos(20.0, 14.0).child(title));

            let hint_alpha = if self.time < 5.0 {
                ((5.0 - self.time) / 2.0).min(1.0)
            } else {
                0.0
            };
            if hint_alpha > 0.0 {
                let hint = text("click to view \u{00b7} drag to orbit".to_string())
                    .mono()
                    .font_size(14.0)
                    .color(Color::new(0.5, 0.5, 0.6, hint_alpha * 0.7));
                children.push(div().pos(ctx.width / 2.0 - 80.0, ctx.height - 28.0).child(hint));
            }

            // Material name labels in grid
            let vp = self.camera.view_proj();
            for i in 0..MATERIAL_COUNT {
                let col = i % GRID_COLS;
                let row = i / GRID_COLS;
                let x = (col as f32 - (GRID_COLS as f32 - 1.0) * 0.5) * SPACING;
                let y = ((GRID_ROWS - 1 - row) as f32 - (GRID_ROWS as f32 - 1.0) * 0.5) * SPACING;

                let world = glam::Vec4::new(x, y - 0.72, 0.0, 1.0);
                let clip = vp * world;
                if clip.w <= 0.0 { continue; }
                let ndc = clip.truncate() / clip.w;
                let sx = (ndc.x * 0.5 + 0.5) * ctx.width;
                let sy = (1.0 - (ndc.y * 0.5 + 0.5)) * ctx.height;

                if sx < -100.0 || sx > ctx.width + 100.0 || sy < -50.0 || sy > ctx.height + 50.0 {
                    continue;
                }

                let name = MATERIAL_NAMES[i];
                let label_x = sx - name.len() as f32 * 4.0;
                let label = text(name.to_string())
                    .mono()
                    .font_size(12.0)
                    .color(Color::new(0.6, 0.65, 0.75, 0.85));
                children.push(div().pos(label_x.max(2.0), sy).child(label));
            }

            let count_str = format!("{} materials \u{00b7} Inline PBR + procedural \u{00b7} No UV seams", MATERIAL_COUNT);
            children.push(
                div().pos(20.0, ctx.height - 24.0).child(
                    text(count_str).mono().font_size(12.0).color(Color::new(0.3, 0.3, 0.4, 0.5)),
                ),
            );
        }

        div()
            .w(Px(ctx.width))
            .h(Px(ctx.height))
            .children(children)
    }

    fn on_input(&mut self, event: &InputEvent) -> bool {
        match event {
            InputEvent::PointerPressed { position, button, .. } => {
                if *button == Some(MouseButton::Left) {
                    // Drawer hit (sliders + cycle button) takes priority over orbit camera
                    if let Some(mat_idx) = self.drawer_material() {
                        match self.drawer_hit_test(mat_idx, (position.x, position.y)) {
                            Some(DrawerHit::Slider(slot)) => {
                                self.begin_slot_drag(slot, position.x);
                                self.dragging_slider = Some(slot);
                                return true;
                            }
                            Some(DrawerHit::CycleLayer) => {
                                self.cycle_layer2(mat_idx);
                                return true;
                            }
                            None => {}
                        }
                    }
                    self.drag_distance = 0.0;
                    self.click_pos = (position.x, position.y);
                    self.camera.on_drag_start(position.x, position.y);
                    return true;
                }
            }
            InputEvent::PointerReleased { position, button, .. } => {
                if *button == Some(MouseButton::Left) {
                    if let Some(slot) = self.dragging_slider.take() {
                        self.end_slot_drag(slot);
                        return true;
                    }
                    self.camera.on_drag_end();
                    // Click detection: if drag was very short, treat as click
                    if self.drag_distance < 5.0 {
                        match &self.state {
                            GalleryState::Grid => {
                                if let Some(index) = self.hit_test_grid(position.x, position.y) {
                                    self.enter_detail(index);
                                }
                            }
                            GalleryState::Detail { .. } => {
                                // Check if clicked "← Back" area (top-left)
                                if position.x < 120.0 && position.y < 40.0 {
                                    self.enter_grid();
                                }
                            }
                            _ => {} // During transitions, ignore clicks
                        }
                    }
                    return true;
                }
            }
            InputEvent::KeyInput { key, pressed, .. } => {
                if *pressed && *key == Key::Escape {
                    if matches!(self.state, GalleryState::Detail { .. }) {
                        self.enter_grid();
                        return true;
                    }
                }
            }
            _ => {}
        }
        false
    }

    fn on_pointer_move(&mut self, x: f32, y: f32) {
        if let Some(slot) = self.dragging_slider {
            self.drag_slot_to(slot, x);
            return;
        }
        if self.camera.dragging {
            let dx = x - self.camera.last_mouse.0;
            let dy = y - self.camera.last_mouse.1;
            self.drag_distance += (dx * dx + dy * dy).sqrt();
        }
        self.camera.on_drag_move(x, y);
    }

    fn on_scroll(&mut self, delta_y: f32) {
        self.camera.on_scroll(delta_y);
    }
}

// ── SceneApp (3D Rendering) ──

impl SceneApp for GalleryApp {
    fn setup(&mut self, ctx: &GpuContext) {
        self.camera.resize(ctx.surface_width, ctx.surface_height);
        self.width = ctx.surface_width as f32 / ctx.scale_factor;
        self.height = ctx.surface_height as f32 / ctx.scale_factor;

        let device = ctx.device.clone();
        let queue = ctx.queue.clone();

        // ── Seimei Renderer (for mesh storage only) ──
        let mut renderer = Renderer::new(
            device.clone(),
            queue.clone(),
            ctx.surface_format,
            ctx.surface_width,
            ctx.surface_height,
        )
        .expect("Seimei Renderer init");

        let quality = QualitySettings {
            preset: Some(QualityPreset::High),
            msaa: MsaaSamples::Off,
            shadow: ShadowQuality::Medium,
            normal_mapping: false,
            emissive: true,
            ibl: false,
            skybox: false,
            ssao: false,
            bloom: false,
            ssr: false,
            dof: false,
            edge_bevel: false,
        };
        let _ = renderer.set_quality(quality);

        // Register meshes
        renderer.add_mesh("icosphere", &procedural::icosphere(1.0, 4), None);
        renderer.add_mesh("icosphere_hd", &procedural::icosphere(1.0, 6), None);
        renderer.add_mesh("cube", &procedural::cube(1.0), None);
        renderer.add_mesh("cylinder", &procedural::cylinder(0.5, 1.0, 32), None);
        renderer.add_mesh("plane", &procedural::plane(1.0, 1.0, 8, 8), None);
        renderer.add_mesh("icosphere_huge", &procedural::icosphere(1.0, 5), None);
        renderer.add_mesh("rock_mesh", &procedural::rock(1.0, 4, 0.35, 42.0), None);
        renderer.add_mesh("terrain", &procedural::terrain(8.0, 64, 1.0), None);
        // Water plane: oversized so shader alpha fade shapes it to pond
        renderer.add_mesh("water_surface", &procedural::water_plane(6.0, 0.0, 32), None);
        renderer.add_mesh("torus", &procedural::torus(0.4, 0.15, 32, 16), None);

        // ── Material pass (murakumo) ──
        let mat_pass = MaterialPass::new(&renderer);

        // Write initial lights
        let lights = build_light_uniform();
        mat_pass.update_lights(&queue, &lights);

        // Write identity shadow matrix
        let identity: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        queue.write_buffer(
            mat_pass.shadow_light_vp_buffer(),
            0,
            bytemuck::bytes_of(&identity),
        );

        // ── Background pass ──
        let bg_pass = BackgroundPass::new(&device, ctx.surface_format);

        self.renderer = Some(renderer);
        self.mat_pass = Some(mat_pass);
        self.bg_pass = Some(bg_pass);
    }

    fn on_resize(&mut self, ctx: &GpuContext) {
        self.camera.resize(ctx.surface_width, ctx.surface_height);
        self.width = ctx.surface_width as f32 / ctx.scale_factor;
        self.height = ctx.surface_height as f32 / ctx.scale_factor;

        if let Some(ref mut renderer) = self.renderer {
            renderer.resize(ctx.surface_width, ctx.surface_height);
        }
    }

    fn render_scene(&mut self, ctx: &mut SceneRenderContext) {
        let Some(ref renderer) = self.renderer else { return };
        let Some(ref mut mat_pass) = self.mat_pass else { return };
        let Some(ref bg_pass) = self.bg_pass else { return };

        let cam_pos = self.camera.position();
        let eye_pos = cam_pos.to_array();

        // ════════════════════════════════════════════════════
        //  Step 1: Background pass (stars, nebula, floor)
        // ════════════════════════════════════════════════════

        {
            let vp = self.camera.view_proj();
            let inv_vp = vp.inverse();
            let bg_cam = BgCameraUniform {
                view_proj: vp.to_cols_array_2d(),
                inv_view_proj: inv_vp.to_cols_array_2d(),
                eye_pos,
                time: self.time,
            };
            ctx.queue.write_buffer(
                &bg_pass.camera_buffer,
                0,
                bytemuck::bytes_of(&bg_cam),
            );

            let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bg_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: ctx.surface_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.01, g: 0.01, b: 0.03, a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: ctx.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&bg_pass.pipeline);
            pass.set_bind_group(0, &bg_pass.camera_bind_group, &[]);
            pass.set_bind_group(1, &bg_pass.lights_bind_group, &[]);
            pass.draw(0..6, 0..1);
        }

        // ════════════════════════════════════════════════════
        //  Step 2: PBR Material pass (inline procedural effects)
        // ════════════════════════════════════════════════════

        // Update camera uniform (with time in position.w)
        let vp = self.camera.view_proj();
        let view = self.camera.view_matrix();
        let cam_uniform = CameraUniform {
            view_proj: vp.to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            position: [eye_pos[0], eye_pos[1], eye_pos[2], 1.0],
            clip_min: [0.0; 4],
            clip_max: [0.0; 4],
        };
        mat_pass.update_camera(&ctx.queue, &cam_uniform, self.time);

        // Sync material params from slider values
        for k in 0..MATERIAL_COUNT {
            for p in 0..PARAMS_PER_MATERIAL {
                mat_pass.set_param(k, p, self.material_param_values[k][p]);
            }
        }
        mat_pass.upload_params(&ctx.queue);

        // Determine which materials to draw and how
        let detail_index = match &self.state {
            GalleryState::Detail { index } => Some(*index),
            GalleryState::TransitionToDetail { index, progress } if *progress > 0.5 => Some(*index),
            GalleryState::TransitionToGrid { from_index, progress } if *progress < 0.5 => Some(*from_index),
            _ => None,
        };

        // Build draw list using MaterialDraw API
        let mut draws: Vec<MaterialDraw> = Vec::new();

        if let Some(idx) = detail_index {
            // Detail mode: single material (+ optional water for field)
            let mesh_id = detail_mesh_id(idx);
            if let Some(mesh) = renderer.get_mesh(mesh_id) {
                let scale = detail_scale(idx);
                let y_off = detail_y_offset(idx);
                let is_static = is_landscape_material(idx) || idx == FIELD_INDEX;
                let yaw = if is_static { 0.0 } else { self.time * 0.1 };
                let model = Mat4::from_translation(Vec3::new(0.0, y_off, 0.0))
                    * Mat4::from_rotation_y(yaw)
                    * Mat4::from_scale(scale);

                let mat_kind = if idx == FIELD_INDEX { 21.0 } else { idx as f32 };
                let (metallic, roughness) = if idx == FIELD_INDEX {
                    (0.0, 0.9)
                } else {
                    match idx {
                        0 => (0.0, 0.1), 1 => (0.1, 0.05), 2 => (0.0, 0.3),
                        3 => (0.3, 0.5), 4 => (0.0, 0.15), 5 => (0.0, 0.8),
                        6 => (0.0, 0.9), 7 => (0.0, 0.4), 8 => (0.5, 0.2),
                        9 => (0.2, 0.1), 10 => (1.0, 0.3), 11 => (0.0, 0.1),
                        12 => (0.3, 0.2), 13 => (0.0, 0.4), 14 => (0.0, 0.5),
                        15 => (0.0, 0.8), 16 => (0.1, 0.15), 17 => (0.0, 0.9),
                        18 => (0.0, 0.8), 19 => (0.0, 0.7), 20 => (0.0, 0.4),
                        21 => (0.0, 0.9),
                        _ => (0.0, 0.5),
                    }
                };
                let flat_flag = if is_landscape_material(idx) || idx == FIELD_INDEX { 1.0 } else { 0.0 };

                let layer = if let Some(k2) = self.material_layer2[idx] {
                    LayerInstance::with_layer(k2, self.material_layer2_alpha[idx])
                } else {
                    LayerInstance::none()
                };

                draws.push(MaterialDraw {
                    material_index: idx,
                    mesh,
                    instance: InstanceData {
                        model: model.to_cols_array_2d(),
                        color: [1.0, 1.0, 1.0, 1.0],
                        material: [metallic, roughness, mat_kind, flat_flag],
                    },
                    layer,
                });
            }

            // Field scene: add water surface
            if idx == FIELD_INDEX {
                if let Some(water_mesh) = renderer.get_mesh("water_surface") {
                    draws.push(MaterialDraw {
                        material_index: 4, // Water
                        mesh: water_mesh,
                        instance: InstanceData {
                            model: Mat4::from_translation(Vec3::new(0.4, -0.02, 0.2)).to_cols_array_2d(),
                            color: [1.0, 1.0, 1.0, 1.0],
                            material: [0.0, 0.15, 4.0, 0.0],
                        },
                        layer: LayerInstance::none(),
                    });
                }
            }
        } else {
            // Grid mode: all materials
            for i in 0..MATERIAL_COUNT {
                let mesh_id = mesh_id_for_material(i);
                if let Some(mesh) = renderer.get_mesh(mesh_id) {
                    draws.push(MaterialDraw {
                        material_index: i,
                        mesh,
                        instance: build_instance(i, self.time),
                        layer: LayerInstance::none(),
                    });
                }
            }
        }

        // Draw materials using MaterialPass
        {
            let device = renderer.device();
            let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("pbr_material_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: ctx.surface_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: ctx.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            mat_pass.render(device, &ctx.queue, &mut pass, &draws);
        }
    }
}

/// Build the light uniform data using Seimei's LightUniform
fn build_light_uniform() -> LightUniform {
    LightUniform::from_lights(
        [0.08, 0.08, 0.12],
        &[
            Light::directional([0.5, 0.8, 0.6], [1.0, 0.95, 0.88], 1.2),
            Light::directional([-0.7, 0.3, 0.4], [0.6, 0.7, 1.0], 0.5),
            Light::directional([0.0, 0.3, -0.9], [0.8, 0.85, 1.0], 0.4),
        ],
    )
}

fn main() {
    sabitori::run_scene(GalleryApp::new());
}
