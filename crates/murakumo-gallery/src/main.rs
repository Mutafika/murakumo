use glam::{Mat4, Vec3};
use sabitori::*;
use seimei::procedural;
use web_time::Instant;
use wgpu::util::DeviceExt;

// ── Material names ──

const MATERIAL_NAMES: [&str; 16] = [
    "Bubble", "Glass", "Portal", "Grid",
    "Water", "Fire", "Smoke", "Aurora",
    "Hologram", "Crystal", "Metal", "Neon",
    "Shield", "Warp", "Dissolve", "Lightning",
];

const GRID_COLS: usize = 4;
const GRID_ROWS: usize = 4;
const QUAD_SIZE: f32 = 1.2;
const SPACING: f32 = 1.8;

// ── Mesh Vertex (position + normal + uv) ──

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MeshVertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

impl MeshVertex {
    fn layout() -> wgpu::VertexBufferLayout<'static> {
        const ATTRS: &[wgpu::VertexAttribute] = &[
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 24, shader_location: 2 },
        ];
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MeshVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: ATTRS,
        }
    }
}

fn render_mesh_to_vertices(mesh: &seimei::RenderMesh) -> Vec<MeshVertex> {
    mesh.vertices.iter().map(|v| MeshVertex {
        position: [v.position.x as f32, v.position.y as f32, v.position.z as f32],
        normal: [v.normal.x as f32, v.normal.y as f32, v.normal.z as f32],
        uv: v.uv,
    }).collect()
}

// ── GPU Mesh Group (shared geometry for multiple materials) ──

struct MeshGroup {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    material_indices: Vec<usize>,
}

// ── GPU Instance Data ──

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct QuadInstance {
    model_0: [f32; 4],
    model_1: [f32; 4],
    model_2: [f32; 4],
    model_3: [f32; 4],
    kind: f32,
    _pad: [f32; 3],
}

impl QuadInstance {
    fn layout() -> wgpu::VertexBufferLayout<'static> {
        const ATTRS: &[wgpu::VertexAttribute] = &[
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 0, shader_location: 3 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 16, shader_location: 4 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 32, shader_location: 5 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 48, shader_location: 6 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 64, shader_location: 7 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 68, shader_location: 8 },
        ];
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<QuadInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: ATTRS,
        }
    }
}

// ── Camera Uniform ──

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    eye_pos: [f32; 3],
    time: f32,
}

// ── Orbit Camera ──

struct OrbitCamera {
    yaw: f32,
    pitch: f32,
    distance: f32,
    target: Vec3,
    fov: f32,
    aspect: f32,
    // Drag state
    dragging: bool,
    last_mouse: (f32, f32),
}

impl OrbitCamera {
    fn new() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.3,
            distance: 8.0,
            target: Vec3::new(0.0, 0.0, 0.0),
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

    fn on_drag_start(&mut self, x: f32, y: f32) {
        self.dragging = true;
        self.last_mouse = (x, y);
    }

    fn on_drag_move(&mut self, x: f32, y: f32) {
        if !self.dragging { return; }
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

// ── Bloom Post-Process Resources ──

struct BloomResources {
    // Intermediate textures
    scene_hdr_texture: wgpu::Texture,
    scene_hdr_view: wgpu::TextureView,
    bright_texture: wgpu::Texture,
    bright_view: wgpu::TextureView,
    blur_ping_texture: wgpu::Texture,
    blur_ping_view: wgpu::TextureView,
    blur_pong_texture: wgpu::Texture,
    blur_pong_view: wgpu::TextureView,
    // Pipelines
    extract_pipeline: wgpu::RenderPipeline,
    blur_h_pipeline: wgpu::RenderPipeline,
    blur_v_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,
    // Bind group layouts
    single_tex_bgl: wgpu::BindGroupLayout,
    composite_bgl: wgpu::BindGroupLayout,
    // Sampler
    sampler: wgpu::Sampler,
    // Depth for HDR pass
    hdr_depth_texture: wgpu::Texture,
    hdr_depth_view: wgpu::TextureView,
    // Dimensions
    width: u32,
    height: u32,
}

impl BloomResources {
    fn create_hdr_texture(device: &wgpu::Device, width: u32, height: u32, label: &str) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }

    fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("bloom_hdr_depth"),
            size: wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }

    fn new(device: &wgpu::Device, width: u32, height: u32, surface_format: wgpu::TextureFormat) -> Self {
        let half_w = (width / 2).max(1);
        let half_h = (height / 2).max(1);

        // Scene HDR (full resolution, Rgba16Float)
        let (scene_hdr_texture, scene_hdr_view) = Self::create_hdr_texture(device, width, height, "bloom_scene_hdr");
        // Bright extract (half resolution)
        let (bright_texture, bright_view) = Self::create_hdr_texture(device, half_w, half_h, "bloom_bright");
        // Blur ping-pong (half resolution)
        let (blur_ping_texture, blur_ping_view) = Self::create_hdr_texture(device, half_w, half_h, "bloom_blur_ping");
        let (blur_pong_texture, blur_pong_view) = Self::create_hdr_texture(device, half_w, half_h, "bloom_blur_pong");
        // HDR depth
        let (hdr_depth_texture, hdr_depth_view) = Self::create_depth_texture(device, width, height);

        // Sampler
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bloom_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        // Bind group layout: single texture + sampler (for extract and blur)
        let single_tex_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom_single_tex_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
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

        // Bind group layout: composite (scene + sampler + bloom)
        let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom_composite_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        // Shader
        let bloom_shader_src = include_str!("../shaders/bloom.wgsl");
        let bloom_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bloom_shader"),
            source: wgpu::ShaderSource::Wgsl(bloom_shader_src.into()),
        });

        let bloom_format = wgpu::TextureFormat::Rgba16Float;

        // Extract pipeline
        let extract_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bloom_extract_layout"),
            bind_group_layouts: &[&single_tex_bgl],
            push_constant_ranges: &[],
        });
        let extract_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("bloom_extract_pipeline"),
            layout: Some(&extract_layout),
            vertex: wgpu::VertexState {
                module: &bloom_shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &bloom_shader,
                entry_point: Some("fs_extract"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: bloom_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Blur H pipeline
        let blur_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bloom_blur_layout"),
            bind_group_layouts: &[&single_tex_bgl],
            push_constant_ranges: &[],
        });
        let blur_h_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("bloom_blur_h_pipeline"),
            layout: Some(&blur_layout),
            vertex: wgpu::VertexState {
                module: &bloom_shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &bloom_shader,
                entry_point: Some("fs_blur_h"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: bloom_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Blur V pipeline
        let blur_v_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("bloom_blur_v_pipeline"),
            layout: Some(&blur_layout),
            vertex: wgpu::VertexState {
                module: &bloom_shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &bloom_shader,
                entry_point: Some("fs_blur_v"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: bloom_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Composite pipeline (outputs to surface format)
        let composite_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bloom_composite_layout"),
            bind_group_layouts: &[&composite_bgl],
            push_constant_ranges: &[],
        });
        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("bloom_composite_pipeline"),
            layout: Some(&composite_layout),
            vertex: wgpu::VertexState {
                module: &bloom_shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &bloom_shader,
                entry_point: Some("fs_composite"),
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
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            scene_hdr_texture,
            scene_hdr_view,
            bright_texture,
            bright_view,
            blur_ping_texture,
            blur_ping_view,
            blur_pong_texture,
            blur_pong_view,
            extract_pipeline,
            blur_h_pipeline,
            blur_v_pipeline,
            composite_pipeline,
            single_tex_bgl,
            composite_bgl,
            sampler,
            hdr_depth_texture,
            hdr_depth_view,
            width,
            height,
        }
    }

    fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width == self.width && height == self.height {
            return;
        }
        let half_w = (width / 2).max(1);
        let half_h = (height / 2).max(1);

        let (t, v) = Self::create_hdr_texture(device, width, height, "bloom_scene_hdr");
        self.scene_hdr_texture = t;
        self.scene_hdr_view = v;

        let (t, v) = Self::create_hdr_texture(device, half_w, half_h, "bloom_bright");
        self.bright_texture = t;
        self.bright_view = v;

        let (t, v) = Self::create_hdr_texture(device, half_w, half_h, "bloom_blur_ping");
        self.blur_ping_texture = t;
        self.blur_ping_view = v;

        let (t, v) = Self::create_hdr_texture(device, half_w, half_h, "bloom_blur_pong");
        self.blur_pong_texture = t;
        self.blur_pong_view = v;

        let (t, v) = Self::create_depth_texture(device, width, height);
        self.hdr_depth_texture = t;
        self.hdr_depth_view = v;

        self.width = width;
        self.height = height;
    }

    fn make_single_tex_bg(&self, device: &wgpu::Device, view: &wgpu::TextureView) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom_single_tex_bg"),
            layout: &self.single_tex_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        })
    }

    fn make_composite_bg(&self, device: &wgpu::Device, scene_view: &wgpu::TextureView, bloom_view: &wgpu::TextureView) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom_composite_bg"),
            layout: &self.composite_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(scene_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(bloom_view) },
            ],
        })
    }

    /// Execute the full bloom pipeline: extract -> blur (2 iterations) -> composite
    fn execute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        output_view: &wgpu::TextureView,
    ) {
        // 1. Brightness extract: scene_hdr -> bright (half-res)
        {
            let bg = self.make_single_tex_bg(device, &self.scene_hdr_view);
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom_extract_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bright_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.extract_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // 2. Gaussian blur: 2 iterations (H then V each iteration)
        // Iteration 1: bright -> ping (H), ping -> pong (V)
        // Iteration 2: pong -> ping (H), ping -> pong (V)
        for iter in 0..3u32 {
            let src_h = if iter == 0 { &self.bright_view } else { &self.blur_pong_view };
            // Horizontal blur -> ping
            {
                let bg = self.make_single_tex_bg(device, src_h);
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("bloom_blur_h_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.blur_ping_view,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(&self.blur_h_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.draw(0..3, 0..1);
            }
            // Vertical blur -> pong
            {
                let bg = self.make_single_tex_bg(device, &self.blur_ping_view);
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("bloom_blur_v_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.blur_pong_view,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(&self.blur_v_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.draw(0..3, 0..1);
            }
        }

        // 3. Composite: scene_hdr + blurred bloom -> output surface
        {
            let bg = self.make_composite_bg(device, &self.scene_hdr_view, &self.blur_pong_view);
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom_composite_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

// ── Gallery App ──

struct GalleryApp {
    camera: OrbitCamera,
    // GPU resources
    pipeline: Option<wgpu::RenderPipeline>,
    bg_pipeline: Option<wgpu::RenderPipeline>,
    camera_buffer: Option<wgpu::Buffer>,
    camera_bind_group: Option<wgpu::BindGroup>,
    instance_buffer: Option<wgpu::Buffer>,
    mesh_groups: Vec<MeshGroup>,
    // Post-processing
    bloom: Option<BloomResources>,
    // State
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
            pipeline: None,
            bg_pipeline: None,
            camera_buffer: None,
            camera_bind_group: None,
            instance_buffer: None,
            mesh_groups: Vec::new(),
            bloom: None,
            time: 0.0,
            start: Instant::now(),
            width: 960.0,
            height: 640.0,
            drag_distance: 0.0,
        }
    }

    fn build_instances(&self) -> Vec<QuadInstance> {
        let mut instances = Vec::with_capacity(16);

        for i in 0..16 {
            let col = i % GRID_COLS;
            let row = i / GRID_COLS;

            let x = (col as f32 - (GRID_COLS as f32 - 1.0) * 0.5) * SPACING;
            let y = ((GRID_ROWS - 1 - row) as f32 - (GRID_ROWS as f32 - 1.0) * 0.5) * SPACING;

            let bob = (self.time * 1.0 + i as f32 * 0.4).sin() * 0.04;

            // Per-material shape + rotation
            // Normalize: icosphere r=1 → 0.55, cube 1.0 → 1.1, cylinder → 1.2, plane → 1.1, torus → 1.8
            let (scale, rot_speed, tilt) = match i {
                // Icosphere (raw diameter=2) → scale 0.55 ≈ display size 1.1
                0 => (Vec3::splat(0.55), 0.15, 0.0),       // Bubble
                1 => (Vec3::splat(0.55), 0.12, 0.0),       // Glass
                9 => (Vec3::splat(0.55), 0.18, 0.1),       // Crystal
                10 => (Vec3::splat(0.55), 0.1, 0.0),       // Metal
                12 => (Vec3::splat(0.55), 0.15, 0.05),     // Shield
                4 => (Vec3::splat(0.55), 0.1, 0.0),        // Water

                // Cube (raw size=1) → scale 1.1
                2 => (Vec3::splat(1.1), 0.35, 0.0),       // Portal
                3 => (Vec3::splat(1.0), 0.1, 0.4),        // Grid
                8 => (Vec3::splat(1.0), 0.4, 0.05),       // Hologram
                13 => (Vec3::splat(1.0), 0.3, 0.1),       // Warp
                14 => (Vec3::splat(1.0), 0.2, 0.0),       // Dissolve

                // Cylinder (raw r=0.5, h=1.0) → scale to match
                5 => (Vec3::new(1.5, 2.2, 1.5), 0.05, 0.0),  // Fire (big + tall)
                6 => (Vec3::new(1.4, 1.8, 1.4), 0.03, 0.0),  // Smoke (big)

                // Plane (raw 1x1) → scale up
                7 => (Vec3::new(1.3, 1.0, 1.3), 0.08, 0.3),  // Aurora
                15 => (Vec3::new(1.1, 1.0, 1.1), 0.05, 0.2), // Lightning

                // Torus (raw R=0.4) → scale 1.8
                11 => (Vec3::splat(1.8), 0.2, 0.0),       // Neon

                _ => (Vec3::splat(1.0), 0.2, 0.0),
            };

            let yaw = self.time * rot_speed + i as f32 * 0.5;
            let model = Mat4::from_translation(Vec3::new(x, y + bob, 0.0))
                * Mat4::from_rotation_y(yaw)
                * Mat4::from_rotation_x(tilt)
                * Mat4::from_scale(scale);

            let cols = model.to_cols_array_2d();
            instances.push(QuadInstance {
                model_0: cols[0],
                model_1: cols[1],
                model_2: cols[2],
                model_3: cols[3],
                kind: i as f32,
                _pad: [0.0; 3],
            });
        }

        instances
    }
}

// ── DeclarativeApp (2D UI Overlay) ──

impl DeclarativeApp for GalleryApp {
    fn title(&self) -> &str { "MURAKUMO \u{2014} Material Gallery" }
    fn size(&self) -> (f32, f32) { (960.0, 640.0) }

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

        // Header title
        let title = text("MURAKUMO \u{2014} Material Gallery".to_string())
            .mono().bold().font_size(24.0)
            .color(Color::new(0.7, 0.8, 1.0, 0.9));
        children.push(div().pos(20.0, 14.0).child(title));

        // Hint text
        let hint_alpha = if self.time < 5.0 { ((5.0 - self.time) / 2.0).min(1.0) } else { 0.0 };
        if hint_alpha > 0.0 {
            let hint = text("drag to orbit".to_string())
                .mono().font_size(14.0)
                .color(Color::new(0.5, 0.5, 0.6, hint_alpha * 0.7));
            children.push(div().pos(ctx.width / 2.0 - 40.0, ctx.height - 28.0).child(hint));
        }

        // Material labels projected to screen space
        let vp = self.camera.view_proj();
        for i in 0..16 {
            let col = i % GRID_COLS;
            let row = i / GRID_COLS;
            let x = (col as f32 - (GRID_COLS as f32 - 1.0) * 0.5) * SPACING;
            let y = ((GRID_ROWS - 1 - row) as f32 - (GRID_ROWS as f32 - 1.0) * 0.5) * SPACING;

            let world = glam::Vec4::new(x, y - QUAD_SIZE * 0.6, 0.0, 1.0);
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
                .mono().font_size(12.0)
                .color(Color::new(0.6, 0.65, 0.75, 0.85));
            children.push(div().pos(label_x.max(2.0), sy).child(label));
        }

        // Footer
        children.push(div().pos(20.0, ctx.height - 24.0).child(
            text("16 materials \u{00b7} Rust + wgpu + WGSL".to_string())
                .mono().font_size(12.0)
                .color(Color::new(0.3, 0.3, 0.4, 0.5))
        ));

        div().w(Px(ctx.width)).h(Px(ctx.height)).children(children)
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

        let device = &ctx.device;

        // Camera uniform
        let cam_data = CameraUniform {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            eye_pos: [0.0; 3],
            time: 0.0,
        };
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gallery_camera_uniform"),
            contents: bytemuck::bytes_of(&cam_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gallery_camera_bgl"),
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
            label: Some("gallery_camera_bg"),
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // Shader
        let shader_src = include_str!("../shaders/gallery.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gallery_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gallery_pipeline_layout"),
            bind_group_layouts: &[&camera_bgl],
            push_constant_ranges: &[],
        });

        let target_format = ctx.surface_format;

        // Generate meshes per shape type and create MeshGroups
        let mesh_defs: Vec<(seimei::RenderMesh, Vec<usize>)> = vec![
            // Icosphere: Bubble(0), Glass(1), Crystal(9), Metal(10), Shield(12), Water(4)
            (procedural::icosphere(1.0, 3), vec![0, 1, 9, 10, 12, 4]),
            // Cube: Portal(2), Grid(3), Hologram(8), Warp(13), Dissolve(14)
            (procedural::cube(1.0), vec![2, 3, 8, 13, 14]),
            // Cylinder: Fire(5), Smoke(6)
            (procedural::cylinder(0.5, 1.0, 32), vec![5, 6]),
            // Plane: Aurora(7), Lightning(15)
            (procedural::plane(1.0, 1.0, 8, 8), vec![7, 15]),
            // Torus: Neon(11)
            (procedural::torus(0.4, 0.15, 32, 16), vec![11]),
        ];

        let mesh_groups: Vec<MeshGroup> = mesh_defs.into_iter().map(|(mesh, mat_indices)| {
            let verts = render_mesh_to_vertices(&mesh);
            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh_vertex_buffer"),
                contents: bytemuck::cast_slice(&verts),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh_index_buffer"),
                contents: bytemuck::cast_slice(&mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
            MeshGroup {
                vertex_buffer,
                index_buffer,
                index_count: mesh.indices.len() as u32,
                material_indices: mat_indices,
            }
        }).collect();

        // Main material pipeline (two vertex buffers: mesh + instance)
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("gallery_material_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[MeshVertex::layout(), QuadInstance::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
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

        // Background pipeline
        let bg_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("gallery_bg_pipeline"),
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
                    format: target_format,
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

        // Instance buffer (16 quads)
        let instances = self.build_instances();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gallery_instances"),
            contents: bytemuck::cast_slice(&instances),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        // Bloom post-process
        let bloom = BloomResources::new(
            device,
            ctx.surface_width, ctx.surface_height,
            ctx.surface_format,
        );

        // HDR pipelines (render to Rgba16Float, bloom composites to surface)
        let hdr_format = wgpu::TextureFormat::Rgba16Float;
        let hdr_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("gallery_hdr_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[MeshVertex::layout(), QuadInstance::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: hdr_format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
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

        let hdr_bg_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("gallery_hdr_bg_pipeline"),
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
                    format: hdr_format,
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

        self.pipeline = Some(hdr_pipeline);
        self.bg_pipeline = Some(hdr_bg_pipeline);
        self.camera_buffer = Some(camera_buffer);
        self.camera_bind_group = Some(camera_bg);
        self.instance_buffer = Some(instance_buffer);
        self.mesh_groups = mesh_groups;
        self.bloom = Some(bloom);
    }

    fn on_resize(&mut self, ctx: &GpuContext) {
        self.camera.resize(ctx.surface_width, ctx.surface_height);
        self.width = ctx.surface_width as f32 / ctx.scale_factor;
        self.height = ctx.surface_height as f32 / ctx.scale_factor;
    }

    fn render_scene(&mut self, ctx: &mut SceneRenderContext) {
        let Some(ref pipeline) = self.pipeline else { return };
        let Some(ref bg_pipeline) = self.bg_pipeline else { return };
        let Some(ref camera_buffer) = self.camera_buffer else { return };
        let Some(ref camera_bg) = self.camera_bind_group else { return };
        let Some(ref instance_buffer) = self.instance_buffer else { return };

        // Update camera uniform
        let vp = self.camera.view_proj();
        let cam_pos = self.camera.position();
        let cam_uniform = CameraUniform {
            view_proj: vp.to_cols_array_2d(),
            eye_pos: cam_pos.to_array(),
            time: self.time,
        };
        ctx.queue.write_buffer(camera_buffer, 0, bytemuck::bytes_of(&cam_uniform));

        // Update instances
        let instances = self.build_instances();
        ctx.queue.write_buffer(instance_buffer, 0, bytemuck::cast_slice(&instances));

        if let Some(ref bloom) = self.bloom {
            // 1. Render scene to HDR texture
            {
                let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("gallery_hdr_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &bloom.scene_hdr_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.01, g: 0.01, b: 0.03, a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &bloom.hdr_depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                pass.set_pipeline(bg_pipeline);
                pass.set_bind_group(0, camera_bg, &[]);
                pass.draw(0..6, 0..1);

                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, camera_bg, &[]);
                for group in &self.mesh_groups {
                    pass.set_vertex_buffer(0, group.vertex_buffer.slice(..));
                    pass.set_vertex_buffer(1, instance_buffer.slice(..));
                    pass.set_index_buffer(group.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    for &mat_idx in &group.material_indices {
                        pass.draw_indexed(0..group.index_count, 0, mat_idx as u32..mat_idx as u32 + 1);
                    }
                }
            }

            // 2. Bloom: extract + blur + composite to surface
            bloom.execute(ctx.encoder, &ctx.device, ctx.surface_view);
        } else {
            // Fallback: direct render to surface
            {
                let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("gallery_scene_pass"),
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

                pass.set_pipeline(bg_pipeline);
                pass.set_bind_group(0, camera_bg, &[]);
                pass.draw(0..6, 0..1);

                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, camera_bg, &[]);
                for group in &self.mesh_groups {
                    pass.set_vertex_buffer(0, group.vertex_buffer.slice(..));
                    pass.set_vertex_buffer(1, instance_buffer.slice(..));
                    pass.set_index_buffer(group.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    for &mat_idx in &group.material_indices {
                        pass.draw_indexed(0..group.index_count, 0, mat_idx as u32..mat_idx as u32 + 1);
                    }
                }
            }
        }
    }
}

fn main() {
    sabitori::run_scene(GalleryApp::new());
}
