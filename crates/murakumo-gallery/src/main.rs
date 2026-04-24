use glam::{Mat4, Vec3};
use sabitori::*;
use seimei::procedural;
use seimei::quality::{MsaaSamples, QualityPreset, QualitySettings, ShadowQuality};
use seimei::{GpuVertex, InstanceData, Renderer};
use web_time::Instant;
use wgpu::util::DeviceExt;

// ── Constants ──

const MATERIAL_NAMES: [&str; 16] = [
    "Bubble", "Glass", "Portal", "Grid",
    "Water", "Fire", "Smoke", "Aurora",
    "Hologram", "Crystal", "Metal", "Neon",
    "Shield", "Warp", "Dissolve", "Lightning",
];

const MATERIAL_COUNT: usize = 16;
const GRID_COLS: usize = 4;
const GRID_ROWS: usize = 4;
const SPACING: f32 = 1.8;

// ── Background camera uniform ──

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BgCameraUniform {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    eye_pos: [f32; 3],
    time: f32,
}

// ── PBR Camera uniform (matches shader CameraUniform) ──

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PbrCameraUniform {
    view_proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    position: [f32; 4], // xyz = eye, w = time
    clip_min: [f32; 4],
    clip_max: [f32; 4],
}

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

// ── Custom PBR Material Pipeline ──
// Uses our pbr_material.wgsl with Seimei's bind group layout (for reusing buffers)

struct PbrMaterialPass {
    pipeline: wgpu::RenderPipeline,
    // Group 0: Camera (with time in position.w)
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    // Group 1: Lights
    light_buffer: wgpu::Buffer,
    light_bind_group: wgpu::BindGroup,
    // Group 2: Texture (dummy white)
    texture_bind_group: wgpu::BindGroup,
    // Group 3: Shadow
    shadow_bind_group: wgpu::BindGroup,
    shadow_light_vp_buffer: wgpu::Buffer,
    // Instance buffer
    instance_buffer: wgpu::Buffer,
}

impl PbrMaterialPass {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, surface_format: wgpu::TextureFormat) -> Self {
        let shader_src = include_str!("../shaders/pbr_material.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pbr_material_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        // Group 0: Camera
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pbr_camera_buffer"),
            size: std::mem::size_of::<PbrCameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pbr_camera_bgl"),
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
            label: Some("pbr_camera_bg"),
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // Group 1: Lights
        // LightUniform: ambient_and_count(16) + 8 * GpuLight(48) = 400 bytes
        let light_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pbr_light_buffer"),
            size: 400,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let light_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pbr_light_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let light_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pbr_light_bg"),
            layout: &light_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: light_buffer.as_entire_binding(),
            }],
        });

        // Group 2: Texture (1x1 white)
        let white_data = [255u8, 255, 255, 255];
        let tex = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("pbr_white_tex"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &white_data,
        );
        let tex_view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let tex_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("pbr_tex_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let texture_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pbr_texture_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let texture_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pbr_texture_bg"),
            layout: &texture_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&tex_sampler),
                },
            ],
        });

        // Group 3: Shadow
        let shadow_size = 2048u32;
        let shadow_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pbr_shadow_tex"),
            size: wgpu::Extent3d { width: shadow_size, height: shadow_size, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("pbr_shadow_sampler"),
            compare: Some(wgpu::CompareFunction::LessEqual),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let shadow_light_vp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pbr_shadow_lvp"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shadow_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pbr_shadow_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Depth,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let shadow_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pbr_shadow_bg"),
            layout: &shadow_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&shadow_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: shadow_light_vp_buffer.as_entire_binding(),
                },
            ],
        });

        // Pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pbr_material_pipeline_layout"),
            bind_group_layouts: &[&camera_bgl, &light_bgl, &texture_bgl, &shadow_bgl],
            push_constant_ranges: &[],
        });

        // Instance buffer
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pbr_instance_buffer"),
            size: (std::mem::size_of::<InstanceData>() * MATERIAL_COUNT) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pbr_material_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[GpuVertex::layout(), InstanceData::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // No culling for transparency
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
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
            light_buffer,
            light_bind_group: light_bg,
            texture_bind_group: texture_bg,
            shadow_bind_group: shadow_bg,
            shadow_light_vp_buffer,
            instance_buffer,
        }
    }
}

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

fn mesh_id_for_material(mat_idx: usize) -> &'static str {
    match mat_idx {
        0 | 1 | 4 | 9 | 10 | 12 => "icosphere",
        2 | 3 | 8 | 13 | 14 => "cube",
        5 | 6 => "cylinder",
        7 | 15 => "plane",
        11 => "torus",
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
        0 => (Vec3::splat(0.55), 0.15, 0.0),
        1 => (Vec3::splat(0.55), 0.12, 0.0),
        9 => (Vec3::splat(0.55), 0.18, 0.1),
        10 => (Vec3::splat(0.55), 0.1, 0.0),
        12 => (Vec3::splat(0.55), 0.15, 0.05),
        4 => (Vec3::splat(0.55), 0.1, 0.0),
        2 => (Vec3::splat(1.1), 0.35, 0.0),
        3 => (Vec3::splat(1.0), 0.1, 0.4),
        8 => (Vec3::splat(1.0), 0.4, 0.05),
        13 => (Vec3::splat(1.0), 0.3, 0.1),
        14 => (Vec3::splat(1.0), 0.2, 0.0),
        5 => (Vec3::new(1.5, 2.2, 1.5), 0.05, 0.0),
        6 => (Vec3::new(1.4, 1.8, 1.4), 0.03, 0.0),
        7 => (Vec3::new(1.3, 1.0, 1.3), 0.08, 0.3),
        15 => (Vec3::new(1.1, 1.0, 1.1), 0.05, 0.2),
        11 => (Vec3::splat(1.8), 0.2, 0.0),
        _ => (Vec3::splat(1.0), 0.2, 0.0),
    };

    let yaw = time * rot_speed + mat_idx as f32 * 0.5;
    let model = Mat4::from_translation(Vec3::new(x, y + bob, 0.0))
        * Mat4::from_rotation_y(yaw)
        * Mat4::from_rotation_x(tilt)
        * Mat4::from_scale(scale);

    let cols = model.to_cols_array_2d();

    let (metallic, roughness) = match mat_idx {
        0 => (0.0, 0.1),
        1 => (0.1, 0.05),
        2 => (0.0, 0.3),
        3 => (0.3, 0.5),
        4 => (0.0, 0.15),
        5 => (0.0, 0.8),
        6 => (0.0, 0.9),
        7 => (0.0, 0.4),
        8 => (0.5, 0.2),
        9 => (0.2, 0.1),
        10 => (1.0, 0.3),
        11 => (0.0, 0.1),
        12 => (0.3, 0.2),
        13 => (0.0, 0.4),
        14 => (0.0, 0.5),
        15 => (0.0, 0.3),
        _ => (0.0, 0.5),
    };

    InstanceData {
        model: cols,
        color: [1.0, 1.0, 1.0, 1.0],
        // material.z = kind (selects which procedural effect)
        material: [metallic, roughness, mat_idx as f32, 0.0],
    }
}

// ── Light Uniform (matches shader LightUniform) ──

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuLightData {
    direction_or_position_and_type: [f32; 4],
    color_and_intensity: [f32; 4],
    extra: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LightUniformData {
    ambient_and_count: [f32; 4],
    lights: [GpuLightData; 8],
}

// ── Gallery App ──

struct GalleryApp {
    camera: OrbitCamera,
    renderer: Option<Renderer>,
    pbr_pass: Option<PbrMaterialPass>,
    bg_pass: Option<BackgroundPass>,
    time: f32,
    start: Instant,
    width: f32,
    height: f32,
    drag_distance: f32,
}

impl GalleryApp {
    fn new() -> Self {
        Self {
            camera: OrbitCamera::new(),
            renderer: None,
            pbr_pass: None,
            bg_pass: None,
            time: 0.0,
            start: Instant::now(),
            width: 960.0,
            height: 640.0,
            drag_distance: 0.0,
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

    fn tick(&mut self, _dt: f32) {
        self.time = self.start.elapsed().as_secs_f32();
    }

    fn view(&self, ctx: &ViewContext) -> Element {
        let mut children = vec![];

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
            let hint = text("drag to orbit".to_string())
                .mono()
                .font_size(14.0)
                .color(Color::new(0.5, 0.5, 0.6, hint_alpha * 0.7));
            children.push(div().pos(ctx.width / 2.0 - 40.0, ctx.height - 28.0).child(hint));
        }

        let vp = self.camera.view_proj();
        for i in 0..MATERIAL_COUNT {
            let col = i % GRID_COLS;
            let row = i / GRID_COLS;
            let x = (col as f32 - (GRID_COLS as f32 - 1.0) * 0.5) * SPACING;
            let y = ((GRID_ROWS - 1 - row) as f32 - (GRID_ROWS as f32 - 1.0) * 0.5) * SPACING;

            let world = glam::Vec4::new(x, y - 0.72, 0.0, 1.0);
            let clip = vp * world;
            if clip.w <= 0.0 {
                continue;
            }
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

        children.push(
            div()
                .pos(20.0, ctx.height - 24.0)
                .child(
                    text("16 materials \u{00b7} Inline PBR + procedural \u{00b7} No UV seams".to_string())
                        .mono()
                        .font_size(12.0)
                        .color(Color::new(0.3, 0.3, 0.4, 0.5)),
                ),
        );

        div()
            .w(Px(ctx.width))
            .h(Px(ctx.height))
            .children(children)
    }

    fn on_input(&mut self, event: &InputEvent) -> bool {
        match event {
            InputEvent::PointerPressed { position, button } => {
                if *button == MouseButton::Left {
                    self.drag_distance = 0.0;
                    self.camera.on_drag_start(position.x, position.y);
                    return true;
                }
            }
            InputEvent::PointerReleased { button, .. } => {
                if *button == MouseButton::Left {
                    self.camera.on_drag_end();
                    return true;
                }
            }
            _ => {}
        }
        false
    }

    fn on_pointer_move(&mut self, x: f32, y: f32) {
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
        renderer.add_mesh("cube", &procedural::cube(1.0), None);
        renderer.add_mesh("cylinder", &procedural::cylinder(0.5, 1.0, 32), None);
        renderer.add_mesh("plane", &procedural::plane(1.0, 1.0, 8, 8), None);
        renderer.add_mesh("torus", &procedural::torus(0.4, 0.15, 32, 16), None);

        // ── Custom PBR Material Pipeline (replaces prepass) ──
        let pbr_pass = PbrMaterialPass::new(&device, &queue, ctx.surface_format);

        // Write initial lights to our PBR pass
        let lights = build_light_uniform();
        queue.write_buffer(&pbr_pass.light_buffer, 0, bytemuck::bytes_of(&lights));

        // Write identity shadow matrix
        let identity: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        queue.write_buffer(
            &pbr_pass.shadow_light_vp_buffer,
            0,
            bytemuck::bytes_of(&identity),
        );

        // ── Background pass ──
        let bg_pass = BackgroundPass::new(&device, ctx.surface_format);

        self.renderer = Some(renderer);
        self.pbr_pass = Some(pbr_pass);
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
        let Some(ref pbr_pass) = self.pbr_pass else { return };
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
        let pbr_cam = PbrCameraUniform {
            view_proj: vp.to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            position: [eye_pos[0], eye_pos[1], eye_pos[2], self.time],
            clip_min: [0.0; 4],
            clip_max: [0.0; 4],
        };
        ctx.queue.write_buffer(
            &pbr_pass.camera_buffer,
            0,
            bytemuck::bytes_of(&pbr_cam),
        );

        // Build instances and upload
        let instance_size = std::mem::size_of::<InstanceData>();
        let mut instance_data = Vec::with_capacity(MATERIAL_COUNT * instance_size);
        for i in 0..MATERIAL_COUNT {
            let inst = build_instance(i, self.time);
            instance_data.extend_from_slice(bytemuck::bytes_of(&inst));
        }
        ctx.queue.write_buffer(&pbr_pass.instance_buffer, 0, &instance_data);

        // Draw each material with its mesh
        {
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

            pass.set_pipeline(&pbr_pass.pipeline);
            pass.set_bind_group(0, &pbr_pass.camera_bind_group, &[]);
            pass.set_bind_group(1, &pbr_pass.light_bind_group, &[]);
            pass.set_bind_group(2, &pbr_pass.texture_bind_group, &[]);
            pass.set_bind_group(3, &pbr_pass.shadow_bind_group, &[]);

            for i in 0..MATERIAL_COUNT {
                let mesh_id = mesh_id_for_material(i);
                if let Some(mesh) = renderer.get_mesh(mesh_id) {
                    let offset = (i * instance_size) as u64;
                    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    pass.set_vertex_buffer(1, pbr_pass.instance_buffer.slice(offset..));
                    pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                }
            }
        }
    }
}

/// Build the light uniform data for our PBR pass
fn build_light_uniform() -> LightUniformData {
    let mut data = LightUniformData {
        ambient_and_count: [0.08, 0.08, 0.12, 3.0], // 3 lights
        lights: [GpuLightData {
            direction_or_position_and_type: [0.0; 4],
            color_and_intensity: [0.0; 4],
            extra: [0.0; 4],
        }; 8],
    };

    // Key light (directional, type=0)
    data.lights[0] = GpuLightData {
        direction_or_position_and_type: [0.5, 0.8, 0.6, 0.0],
        color_and_intensity: [1.0, 0.95, 0.88, 1.2],
        extra: [0.0; 4],
    };

    // Fill light
    data.lights[1] = GpuLightData {
        direction_or_position_and_type: [-0.7, 0.3, 0.4, 0.0],
        color_and_intensity: [0.6, 0.7, 1.0, 0.5],
        extra: [0.0; 4],
    };

    // Rim light
    data.lights[2] = GpuLightData {
        direction_or_position_and_type: [0.0, 0.3, -0.9, 0.0],
        color_and_intensity: [0.8, 0.85, 1.0, 0.4],
        extra: [0.0; 4],
    };

    data
}

fn main() {
    sabitori::run_scene(GalleryApp::new());
}
