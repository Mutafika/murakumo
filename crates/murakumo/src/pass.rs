//! MaterialPass — Seimei Renderer 上のプロシージャルマテリアル描画パス
//!
//! ギャラリーや他のアプリから `MaterialPass::new(&renderer)` で生成し、
//! `render()` でカスタムシェーダーマテリアルを描画する。

use seimei::{CameraUniform, GpuVertex, InstanceData, LightUniform, MeshInstance, Renderer};
use wgpu::util::DeviceExt;

use crate::params::{self, PARAMS_PER_MATERIAL};

// ── Constants ──

pub const MATERIAL_COUNT: usize = 23;
const MAT_PARAMS_VEC4S: usize = MATERIAL_COUNT * 2;

pub const MATERIAL_NAMES: [&str; MATERIAL_COUNT] = [
    "Bubble", "Glass", "Portal", "Grid",
    "Water", "Fire", "Smoke", "Aurora",
    "Hologram", "Crystal", "Metal", "Neon",
    "Shield", "Dissolve", "Lightning", "Lava",
    "Ice", "Cloud", "Explosion", "Tornado",
    "Skin", "Rock", "Field",
];

/// 透明マテリアルか判定（depth write 無効で描画）
pub fn is_transparent(material_index: usize) -> bool {
    matches!(material_index, 0 | 1 | 5 | 6 | 14 | 17 | 18 | 19)
}

// ── Secondary layer instance data ──

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LayerInstance {
    pub layer2: [f32; 4],
}

impl LayerInstance {
    pub fn none() -> Self {
        Self { layer2: [-1.0, 0.0, 0.0, 0.0] }
    }

    pub fn with_layer(kind: usize, alpha: f32) -> Self {
        Self { layer2: [kind as f32, alpha.clamp(0.0, 1.0), 0.0, 0.0] }
    }

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 11,
                format: wgpu::VertexFormat::Float32x4,
            }],
        }
    }
}

// ── Material params uniform ──

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MatParamsUniform {
    pub values: [[f32; 4]; MAT_PARAMS_VEC4S],
}

impl MatParamsUniform {
    pub fn from_values(per_mat: &[[f32; PARAMS_PER_MATERIAL]; MATERIAL_COUNT]) -> Self {
        let mut out = Self { values: [[0.0; 4]; MAT_PARAMS_VEC4S] };
        for k in 0..MATERIAL_COUNT {
            let p = &per_mat[k];
            out.values[k * 2]     = [p[0], p[1], p[2], p[3]];
            out.values[k * 2 + 1] = [p[4], p[5], p[6], p[7]];
        }
        out
    }
}

// ── Draw command ──

/// 1 回の描画命令
pub struct MaterialDraw<'a> {
    /// マテリアルインデックス (0..22)
    pub material_index: usize,
    /// 描画メッシュ
    pub mesh: &'a MeshInstance,
    /// インスタンスデータ（transform, color, material.z = kind）
    pub instance: InstanceData,
    /// セカンダリレイヤー（省略時は無効）
    pub layer: LayerInstance,
}

// ── MaterialPass ──

pub struct MaterialPass {
    // Pipelines
    opaque_pipeline: wgpu::RenderPipeline,
    transparent_pipeline: wgpu::RenderPipeline,
    // Group 0: Camera
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    // Group 1: Lights + MatParams
    light_buffer: wgpu::Buffer,
    mat_params_buffer: wgpu::Buffer,
    light_bind_group: wgpu::BindGroup,
    // Group 2: Texture (dummy white)
    texture_bind_group: wgpu::BindGroup,
    // Group 3: Shadow (dummy)
    shadow_bind_group: wgpu::BindGroup,
    shadow_light_vp_buffer: wgpu::Buffer,
    // Instance buffer
    instance_buffer: wgpu::Buffer,
    instance_capacity: usize,
    // Layer instance buffer
    layer_instance_buffer: wgpu::Buffer,
    // State
    material_params: [[f32; PARAMS_PER_MATERIAL]; MATERIAL_COUNT],
}

impl MaterialPass {
    /// Seimei Renderer から必要な情報を取得して生成
    pub fn new(renderer: &Renderer) -> Self {
        let device = renderer.device();
        let queue = renderer.queue();
        let surface_format = renderer.surface_format();
        let initial_capacity = MATERIAL_COUNT;

        let shader_src = include_str!("../shaders/pbr_material.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("murakumo_pbr_material_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        // ── Group 0: Camera ─���
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("murakumo_camera_buffer"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("murakumo_camera_bgl"),
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
            label: Some("murakumo_camera_bg"),
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // ── Group 1: Lights (binding 0) + MatParams (binding 1) ──
        let light_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("murakumo_light_buffer"),
            size: std::mem::size_of::<LightUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mat_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("murakumo_mat_params_buffer"),
            size: std::mem::size_of::<MatParamsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let light_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("murakumo_light_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
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
        let light_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("murakumo_light_bg"),
            layout: &light_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: light_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: mat_params_buffer.as_entire_binding(),
                },
            ],
        });

        // ── Group 2: Texture (1x1 white dummy) ──
        let white_data = [255u8, 255, 255, 255];
        let tex = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("murakumo_white_tex"),
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
            label: Some("murakumo_tex_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let texture_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("murakumo_texture_bgl"),
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
            label: Some("murakumo_texture_bg"),
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

        // ── Group 3: Shadow (dummy) ──
        let shadow_size = 2048u32;
        let shadow_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("murakumo_shadow_tex"),
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
            label: Some("murakumo_shadow_sampler"),
            compare: Some(wgpu::CompareFunction::LessEqual),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let shadow_light_vp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("murakumo_shadow_lvp"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shadow_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("murakumo_shadow_bgl"),
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
            label: Some("murakumo_shadow_bg"),
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

        // ── Pipeline Layout ──
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("murakumo_pipeline_layout"),
            bind_group_layouts: &[&camera_bgl, &light_bgl, &texture_bgl, &shadow_bgl],
            push_constant_ranges: &[],
        });

        // ── Instance Buffers ──
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("murakumo_instance_buffer"),
            size: (std::mem::size_of::<InstanceData>() * initial_capacity) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layer_instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("murakumo_layer_instance_buffer"),
            size: (std::mem::size_of::<LayerInstance>() * initial_capacity) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Opaque Pipeline ──
        let opaque_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("murakumo_opaque_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[GpuVertex::layout(), InstanceData::layout(), LayerInstance::layout()],
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
                cull_mode: Some(wgpu::Face::Back),
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

        // ── Transparent Pipeline ──
        let transparent_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("murakumo_transparent_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[GpuVertex::layout(), InstanceData::layout(), LayerInstance::layout()],
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
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // ── Default params ──
        let mut material_params = [[0.0f32; PARAMS_PER_MATERIAL]; MATERIAL_COUNT];
        for i in 0..MATERIAL_COUNT {
            material_params[i] = params::default_values(i);
        }

        Self {
            opaque_pipeline,
            transparent_pipeline,
            camera_buffer,
            camera_bind_group: camera_bg,
            light_buffer,
            mat_params_buffer,
            light_bind_group: light_bg,
            texture_bind_group: texture_bg,
            shadow_bind_group: shadow_bg,
            shadow_light_vp_buffer,
            instance_buffer,
            instance_capacity: initial_capacity,
            layer_instance_buffer,
            material_params,
        }
    }

    // ── 更新メソッド ──

    /// カメラ + 時刻を更新（position.w に time を格納）
    pub fn update_camera(&self, queue: &wgpu::Queue, camera_uniform: &CameraUniform, time: f32) {
        let mut u = *camera_uniform;
        u.position[3] = time;
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&u));
    }

    /// ライトを��新
    pub fn update_lights(&self, queue: &wgpu::Queue, lights: &LightUniform) {
        queue.write_buffer(&self.light_buffer, 0, bytemuck::bytes_of(lights));
    }

    /// マテリアルパラメータを1つ設定
    pub fn set_param(&mut self, material_index: usize, param_index: usize, value: f32) {
        if material_index < MATERIAL_COUNT && param_index < PARAMS_PER_MATERIAL {
            self.material_params[material_index][param_index] = value;
        }
    }

    /// マテリアルパラメータを取得
    pub fn get_param(&self, material_index: usize, param_index: usize) -> f32 {
        self.material_params[material_index][param_index]
    }

    /// パラメータ配列への直接参照（スライダーUI等で使用）
    pub fn params_mut(&mut self) -> &mut [[f32; PARAMS_PER_MATERIAL]; MATERIAL_COUNT] {
        &mut self.material_params
    }

    /// パラメータ配列への参照
    pub fn params(&self) -> &[[f32; PARAMS_PER_MATERIAL]; MATERIAL_COUNT] {
        &self.material_params
    }

    /// デフォルト値にリセット
    pub fn reset_params(&mut self, material_index: usize) {
        if material_index < MATERIAL_COUNT {
            self.material_params[material_index] = params::default_values(material_index);
        }
    }

    /// 全マテリアルパラメータを GPU にア���プロード
    pub fn upload_params(&self, queue: &wgpu::Queue) {
        let uniform = MatParamsUniform::from_values(&self.material_params);
        queue.write_buffer(&self.mat_params_buffer, 0, bytemuck::bytes_of(&uniform));
    }

    /// インスタンスバッファ
    pub fn instance_buffer(&self) -> &wgpu::Buffer { &self.instance_buffer }

    /// レイヤーインスタンスバッファ
    pub fn layer_instance_buffer(&self) -> &wgpu::Buffer { &self.layer_instance_buffer }

    /// シャドウライト VP バッファ
    pub fn shadow_light_vp_buffer(&self) -> &wgpu::Buffer { &self.shadow_light_vp_buffer }

    // ── 描画 ──

    /// インスタンスバッファを必要に応じて再確保
    fn ensure_capacity(&mut self, device: &wgpu::Device, count: usize) {
        if count <= self.instance_capacity {
            return;
        }
        let new_cap = count.next_power_of_two().max(MATERIAL_COUNT);
        self.instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("murakumo_instance_buffer"),
            size: (std::mem::size_of::<InstanceData>() * new_cap) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.layer_instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("murakumo_layer_instance_buffer"),
            size: (std::mem::size_of::<LayerInstance>() * new_cap) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.instance_capacity = new_cap;
    }

    /// プロシージャルマテリアルを描画する。
    ///
    /// `draws` の各要素が1つの描画コマンド。
    /// opaque → transparent の順で自動ソートして描画。
    pub fn render<'a>(
        &'a mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'a>,
        draws: &[MaterialDraw<'a>],
    ) {
        if draws.is_empty() {
            return;
        }

        // Ensure capacity
        self.ensure_capacity(device, draws.len());

        // Upload instance + layer data
        let instance_data: Vec<u8> = draws.iter()
            .flat_map(|d| bytemuck::bytes_of(&d.instance))
            .copied()
            .collect();
        queue.write_buffer(&self.instance_buffer, 0, &instance_data);

        let layer_data: Vec<u8> = draws.iter()
            .flat_map(|d| bytemuck::bytes_of(&d.layer))
            .copied()
            .collect();
        queue.write_buffer(&self.layer_instance_buffer, 0, &layer_data);

        let inst_stride = std::mem::size_of::<InstanceData>() as u64;
        let layer_stride = std::mem::size_of::<LayerInstance>() as u64;

        // Bind groups (shared across opaque + transparent)
        pass.set_bind_group(0, &self.camera_bind_group, &[]);
        pass.set_bind_group(1, &self.light_bind_group, &[]);
        pass.set_bind_group(2, &self.texture_bind_group, &[]);
        pass.set_bind_group(3, &self.shadow_bind_group, &[]);

        // Opaque pass
        pass.set_pipeline(&self.opaque_pipeline);
        for (idx, draw) in draws.iter().enumerate() {
            if is_transparent(draw.material_index) {
                continue;
            }
            let offset = idx as u64 * inst_stride;
            let layer_offset = idx as u64 * layer_stride;
            pass.set_vertex_buffer(0, draw.mesh.vertex_buffer.slice(..));
            pass.set_vertex_buffer(1, self.instance_buffer.slice(offset..));
            pass.set_vertex_buffer(2, self.layer_instance_buffer.slice(layer_offset..));
            pass.set_index_buffer(draw.mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..draw.mesh.index_count, 0, 0..1);
        }

        // Transparent pass
        pass.set_pipeline(&self.transparent_pipeline);
        for (idx, draw) in draws.iter().enumerate() {
            if !is_transparent(draw.material_index) {
                continue;
            }
            let offset = idx as u64 * inst_stride;
            let layer_offset = idx as u64 * layer_stride;
            pass.set_vertex_buffer(0, draw.mesh.vertex_buffer.slice(..));
            pass.set_vertex_buffer(1, self.instance_buffer.slice(offset..));
            pass.set_vertex_buffer(2, self.layer_instance_buffer.slice(layer_offset..));
            pass.set_index_buffer(draw.mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..draw.mesh.index_count, 0, 0..1);
        }
    }
}
