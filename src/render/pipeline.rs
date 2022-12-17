use ash::vk;

use crate::render::{
	surface::Surface,
	swapchain::Swapchain,
};

//~ pub const MAX_NUMBER_OF_TEXTURES: u32 = 393210;
pub const MAX_NUMBER_OF_TEXTURES: u32 = 3;

// Pipeline
pub struct GraphicsPipeline {
	pub pipeline: vk::Pipeline,
	pub layout: vk::PipelineLayout,
	pub descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
}

impl GraphicsPipeline {
	pub fn init(
		logical_device: &ash::Device,
		swapchain: &Swapchain,
		renderpass: &vk::RenderPass,
	) -> Result<GraphicsPipeline, vk::Result>{
		// Include shaders
		let vertexshader_createinfo = vk::ShaderModuleCreateInfo::builder().code(vk_shader_macros::include_glsl!(
			"./shaders/vertex.glsl", 
			kind: vert,
		));
		let fragmentshader_createinfo = vk::ShaderModuleCreateInfo::builder().code(vk_shader_macros::include_glsl!(
			"./shaders/fragment.glsl",
			kind: frag,
		));
		let vertexshader_module = unsafe { logical_device.create_shader_module(&vertexshader_createinfo, None)? };
		let fragmentshader_module = unsafe { logical_device.create_shader_module(&fragmentshader_createinfo, None)? };
		
		// Set main function's name to `main`
		let main_function = std::ffi::CString::new("main").unwrap();
		
		let vertexshader_stage = vk::PipelineShaderStageCreateInfo::builder()
			.stage(vk::ShaderStageFlags::VERTEX)
			.module(vertexshader_module)
			.name(&main_function);
		let fragmentshader_stage = vk::PipelineShaderStageCreateInfo::builder()
			.stage(vk::ShaderStageFlags::FRAGMENT)
			.module(fragmentshader_module)
			.name(&main_function);
			
		let shader_stages = vec![vertexshader_stage.build(), fragmentshader_stage.build()];
		
		// Vertex Input Info
		// 
		// Attribute description
		let vertex_attrib_descs = [
			vk::VertexInputAttributeDescription {
				binding: 0,
				location: 0,
				offset: 0,
				format: vk::Format::R32G32B32_SFLOAT,
			},
			vk::VertexInputAttributeDescription {
				binding: 0,
				location: 1,
				offset: 12,
				format: vk::Format::R32G32B32_SFLOAT,
			},
			vk::VertexInputAttributeDescription {
				binding: 1,
				location: 2,
				offset: 0,
				format: vk::Format::R32G32B32A32_SFLOAT,
			},
			vk::VertexInputAttributeDescription {
				binding: 1,
				location: 3,
				offset: 16,
				format: vk::Format::R32G32B32A32_SFLOAT,
			},
			vk::VertexInputAttributeDescription {
				binding: 1,
				location: 4,
				offset: 32,
				format: vk::Format::R32G32B32A32_SFLOAT,
			},
			vk::VertexInputAttributeDescription {
				binding: 1,
				location: 5,
				offset: 48,
				format: vk::Format::R32G32B32A32_SFLOAT,
			},
			vk::VertexInputAttributeDescription {
				binding: 1,
				location: 6,
				offset: 64,
				format: vk::Format::R32G32B32A32_SFLOAT,
			},
			vk::VertexInputAttributeDescription {
				binding: 1,
				location: 7,
				offset: 80,
				format: vk::Format::R32G32B32A32_SFLOAT,
			},
			vk::VertexInputAttributeDescription {
				binding: 1,
				location: 8,
				offset: 96,
				format: vk::Format::R32G32B32A32_SFLOAT,
			},
			vk::VertexInputAttributeDescription {
				binding: 1,
				location: 9,
				offset: 112,
				format: vk::Format::R32G32B32A32_SFLOAT,
			},
			vk::VertexInputAttributeDescription {
				binding: 1,
				location: 10,
				offset: 128,
				format: vk::Format::R32G32B32_SFLOAT,
			},
			vk::VertexInputAttributeDescription {
				binding: 1,
				location: 11,
				offset: 140,
				format: vk::Format::R32_SFLOAT,
			},
			vk::VertexInputAttributeDescription {
				binding: 1,
				location: 12,
				offset: 144,
				format: vk::Format::R32_SFLOAT,
			},
		];
		
		// Input Bindings' description
		//
		// stride	  - binding variables' size
		// input_rate - frequency, when the data is changed (per vertex/ per instance)
		let vertex_binding_descs = [
			vk::VertexInputBindingDescription {
				binding: 0,
				stride: 24,
				input_rate: vk::VertexInputRate::VERTEX,
			},
			vk::VertexInputBindingDescription {
				binding: 1,
				stride: 148,
				input_rate: vk::VertexInputRate::INSTANCE,
			},
		];
		
		// Bind vertex inputs
		let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
			.vertex_attribute_descriptions(&vertex_attrib_descs)
			.vertex_binding_descriptions(&vertex_binding_descs);
			
		let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
			.topology(vk::PrimitiveTopology::TRIANGLE_LIST);
		
		// Viewports
		let viewports = [vk::Viewport {
			x: 0.,
			y: 0.,
			width: swapchain.extent.width as f32,
			height: swapchain.extent.height as f32,
			min_depth: 0.,
			max_depth: 1.,
		}];
		
		let scissors = [vk::Rect2D {
			offset: vk::Offset2D { x: 0, y: 0 },
			extent: swapchain.extent,
		}];

		let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
			.viewports(&viewports)
			.scissors(&scissors);
			
		// Rasterizer
		let rasterizer_info = vk::PipelineRasterizationStateCreateInfo::builder()
			.line_width(1.0)
			.front_face(vk::FrontFace::COUNTER_CLOCKWISE)
			.cull_mode(vk::CullModeFlags::BACK)
			.polygon_mode(vk::PolygonMode::FILL);
			
		// Multisampler	
		let multisampler_info = vk::PipelineMultisampleStateCreateInfo::builder()
			.rasterization_samples(vk::SampleCountFlags::TYPE_1);
			
		// Color blend
		let colorblend_attachments = [vk::PipelineColorBlendAttachmentState::builder()
			.blend_enable(true)
			.src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
			.dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
			.color_blend_op(vk::BlendOp::ADD)
			.src_alpha_blend_factor(vk::BlendFactor::SRC_ALPHA)
			.dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
			.alpha_blend_op(vk::BlendOp::ADD)
			.color_write_mask(
				vk::ColorComponentFlags::R
					| vk::ColorComponentFlags::G
					| vk::ColorComponentFlags::B
					| vk::ColorComponentFlags::A,
			)
			.build()];
		let colourblend_info = vk::PipelineColorBlendStateCreateInfo::builder()
			.attachments(&colorblend_attachments);
		let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::builder()
			.depth_test_enable(true)
			.depth_write_enable(true)
			.depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);
		
		// Bind resource descriptor
		//
		// 0
		let descriptorset_layout_binding_descs0 = [
			vk::DescriptorSetLayoutBinding::builder()
				// Binding = 0
				.binding(0)
				// Resource type = uniform buffer
				.descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
				// Descriptor count = 1
				.descriptor_count(1)
				// Use the buffer in vertex shaders only
				.stage_flags(vk::ShaderStageFlags::VERTEX)
				.build()
		];
		let descriptorset_layout_info0 = vk::DescriptorSetLayoutCreateInfo::builder()
			.bindings(&descriptorset_layout_binding_descs0);
		let descriptorsetlayout0 = unsafe {
			logical_device.create_descriptor_set_layout(&descriptorset_layout_info0, None)
		}?;
		// 1
		let descriptorset_layout_binding_descs1 = [
			vk::DescriptorSetLayoutBinding::builder()
				// Binding = 0
				.binding(0)
				// Resource type = uniform buffer
				.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
				// Descriptor count = 1
				.descriptor_count(1)
				// Use the buffer in fragment shaders only
				.stage_flags(vk::ShaderStageFlags::FRAGMENT)
				.build()
		];
		let descriptorset_layout_info1 = vk::DescriptorSetLayoutCreateInfo::builder()
			.bindings(&descriptorset_layout_binding_descs1);
		let descriptorsetlayout1 = unsafe {
			logical_device.create_descriptor_set_layout(&descriptorset_layout_info1, None)
		}?;
		// Push descriptor layouts
		let desclayouts = vec![
			descriptorsetlayout0,
			descriptorsetlayout1,
		];
		
		// Pipeline layout
		let pipelinelayout_info = vk::PipelineLayoutCreateInfo::builder()
			.set_layouts(&desclayouts);
			
		let pipelinelayout = unsafe { logical_device.create_pipeline_layout(&pipelinelayout_info, None) }?;
		
		// Graphics Pipeline
		let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
			.stages(&shader_stages)
			.vertex_input_state(&vertex_input_info)
			.input_assembly_state(&input_assembly_info)
			.viewport_state(&viewport_info)
			.rasterization_state(&rasterizer_info)
			.multisample_state(&multisampler_info)
			.depth_stencil_state(&depth_stencil_info)
			.color_blend_state(&colourblend_info)
			.layout(pipelinelayout)
			.render_pass(*renderpass)
			.subpass(0);
		let graphicspipeline = unsafe {
			logical_device
				.create_graphics_pipelines(
					vk::PipelineCache::null(),
					&[pipeline_info.build()],
					None,
				).expect("Cannot create pipeline")				
		}[0];
		
		// Destroy used shader modules
		unsafe {
			logical_device.destroy_shader_module(fragmentshader_module, None);
			logical_device.destroy_shader_module(vertexshader_module, None);
		}
		
		Ok(GraphicsPipeline {
			pipeline: graphicspipeline,
			layout: pipelinelayout,
			descriptor_set_layouts: desclayouts,
		})
	}
	
	pub fn init_textured(
		logical_device: &ash::Device,
		swapchain: &Swapchain,
		renderpass: &vk::RenderPass,
	) -> Result<GraphicsPipeline, vk::Result>{
		// Include shaders
		let vertexshader_createinfo = vk::ShaderModuleCreateInfo::builder().code(vk_shader_macros::include_glsl!(
			"./shaders/vertex_combined.glsl", 
			kind: vert,
		));
		let fragmentshader_createinfo = vk::ShaderModuleCreateInfo::builder().code(vk_shader_macros::include_glsl!(
			"./shaders/fragment_combined.glsl",
			kind: frag,
		));
		let vertexshader_module = unsafe { logical_device.create_shader_module(&vertexshader_createinfo, None)? };
		let fragmentshader_module = unsafe { logical_device.create_shader_module(&fragmentshader_createinfo, None)? };
		
		// Set main function's name to `main`
		let main_function = std::ffi::CString::new("main").unwrap();
		
		let vertexshader_stage = vk::PipelineShaderStageCreateInfo::builder()
			.stage(vk::ShaderStageFlags::VERTEX)
			.module(vertexshader_module)
			.name(&main_function);
		let fragmentshader_stage = vk::PipelineShaderStageCreateInfo::builder()
			.stage(vk::ShaderStageFlags::FRAGMENT)
			.module(fragmentshader_module)
			.name(&main_function);
			
		let shader_stages = vec![vertexshader_stage.build(), fragmentshader_stage.build()];
		
		// Vertex Input Info
		// 
		// Attribute description
		let vertex_attrib_descs = [
			vk::VertexInputAttributeDescription {
				binding: 0,
				location: 0,
				offset: 0,
				format: vk::Format::R32G32B32_SFLOAT,
			},
			vk::VertexInputAttributeDescription {
				binding: 0,
				location: 1,
				offset: 12,
				format: vk::Format::R32G32B32_SFLOAT,
			},
			vk::VertexInputAttributeDescription {
				binding: 0,
				location: 2,
				offset: 24,
				format: vk::Format::R32G32_SFLOAT,
			},
			vk::VertexInputAttributeDescription {
				binding: 1,
				location: 3,
				offset: 0,
				format: vk::Format::R32G32B32A32_SFLOAT,
			},
			vk::VertexInputAttributeDescription {
				binding: 1,
				location: 4,
				offset: 16,
				format: vk::Format::R32G32B32A32_SFLOAT,
			},
			vk::VertexInputAttributeDescription {
				binding: 1,
				location: 5,
				offset: 32,
				format: vk::Format::R32G32B32A32_SFLOAT,
			},
			vk::VertexInputAttributeDescription {
				binding: 1,
				location: 6,
				offset: 48,
				format: vk::Format::R32G32B32A32_SFLOAT,
			},
			vk::VertexInputAttributeDescription {
				binding: 1,
				location: 7,
				offset: 64,
				format: vk::Format::R32G32B32A32_SFLOAT,
			},
			vk::VertexInputAttributeDescription {
				binding: 1,
				location: 8,
				offset: 80,
				format: vk::Format::R32G32B32A32_SFLOAT,
			},
			vk::VertexInputAttributeDescription {
				binding: 1,
				location: 9,
				offset: 96,
				format: vk::Format::R32G32B32A32_SFLOAT,
			},
			vk::VertexInputAttributeDescription {
				binding: 1,
				location: 10,
				offset: 112,
				format: vk::Format::R32G32B32A32_SFLOAT,
			},
			vk::VertexInputAttributeDescription{
				binding: 1,
				location: 11,
				offset: 128,
				format: vk::Format::R8G8B8A8_UINT,
			},
			vk::VertexInputAttributeDescription{
				binding: 1,
				location: 12,
				offset: 132,
				format: vk::Format::R32_SFLOAT,
			},
			vk::VertexInputAttributeDescription{
				binding: 1,
				location: 13,
				offset: 136,
				format: vk::Format::R32_SFLOAT,
			},
		];
		let vertex_binding_descs = [
			vk::VertexInputBindingDescription {
				binding: 0,
				stride: 32,
				input_rate: vk::VertexInputRate::VERTEX,
			},
			vk::VertexInputBindingDescription {
				binding: 1,
				stride: 140,
				input_rate: vk::VertexInputRate::INSTANCE,
			},
		];
		
		// Bind vertex inputs
		let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
			.vertex_attribute_descriptions(&vertex_attrib_descs)
			.vertex_binding_descriptions(&vertex_binding_descs);
			
		let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
			.topology(vk::PrimitiveTopology::TRIANGLE_LIST);
		
		// Viewports
		let viewports = [vk::Viewport {
			x: 0.,
			y: 0.,
			width: swapchain.extent.width as f32,
			height: swapchain.extent.height as f32,
			min_depth: 0.,
			max_depth: 1.,
		}];
		
		let scissors = [vk::Rect2D {
			offset: vk::Offset2D { x: 0, y: 0 },
			extent: swapchain.extent,
		}];

		let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
			.viewports(&viewports)
			.scissors(&scissors);
			
		// Rasterizer
		let rasterizer_info = vk::PipelineRasterizationStateCreateInfo::builder()
			.line_width(1.0)
			.front_face(vk::FrontFace::COUNTER_CLOCKWISE)
			.cull_mode(vk::CullModeFlags::BACK)
			.polygon_mode(vk::PolygonMode::FILL);
			
		// Multisampler	
		let multisampler_info = vk::PipelineMultisampleStateCreateInfo::builder()
			.rasterization_samples(vk::SampleCountFlags::TYPE_1);
			
		// Color blend
		let colorblend_attachments = [vk::PipelineColorBlendAttachmentState::builder()
			.blend_enable(true)
			.src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
			.dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
			.color_blend_op(vk::BlendOp::ADD)
			.src_alpha_blend_factor(vk::BlendFactor::SRC_ALPHA)
			.dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
			.alpha_blend_op(vk::BlendOp::ADD)
			.color_write_mask(
				vk::ColorComponentFlags::R
					| vk::ColorComponentFlags::G
					| vk::ColorComponentFlags::B
					| vk::ColorComponentFlags::A,
			)
			.build()];
		let colourblend_info = vk::PipelineColorBlendStateCreateInfo::builder()
			.attachments(&colorblend_attachments);
		let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::builder()
			.depth_test_enable(true)
			.depth_write_enable(true)
			.depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);
		
		// Bind resource descriptor
		//
		//
		// 0
		let descriptorset_layout_binding_descs0 = [
			vk::DescriptorSetLayoutBinding::builder()
				.binding(0)
				.descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
				.descriptor_count(1)
				.stage_flags(vk::ShaderStageFlags::VERTEX)
				.build()
		];
		let descriptorset_layout_info0 = vk::DescriptorSetLayoutCreateInfo::builder()
			.bindings(&descriptorset_layout_binding_descs0);
		let descriptorsetlayout0 = unsafe {
			logical_device.create_descriptor_set_layout(&descriptorset_layout_info0, None)
		}?;
		//
		//
		// 1
		let descriptorset_layout_binding_descs1 = [vk::DescriptorSetLayoutBinding::builder()
			.binding(0)
			.descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
			.descriptor_count(MAX_NUMBER_OF_TEXTURES)
			.stage_flags(vk::ShaderStageFlags::FRAGMENT)
			.build()];
		let descriptorset_layout_info1 = vk::DescriptorSetLayoutCreateInfo::builder()
			.bindings(&descriptorset_layout_binding_descs1);
		let descriptorsetlayout1 = unsafe {
			logical_device.create_descriptor_set_layout(&descriptorset_layout_info1, None)
		}?;
		//
		//
		// 2
		let descriptorset_layout_binding_descs2 = [
			vk::DescriptorSetLayoutBinding::builder()
				.binding(0)
				.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
				.descriptor_count(1)
				.stage_flags(vk::ShaderStageFlags::FRAGMENT)
				.build()
		];
		let descriptorset_layout_info2 = vk::DescriptorSetLayoutCreateInfo::builder()
			.bindings(&descriptorset_layout_binding_descs2);
		let descriptorsetlayout2 = unsafe {
			logical_device.create_descriptor_set_layout(&descriptorset_layout_info2, None)
		}?;
		
		let desclayouts = vec![
			descriptorsetlayout0,
			descriptorsetlayout1,
			descriptorsetlayout2,
		];
		
		// Pipeline layout
		let pipelinelayout_info = vk::PipelineLayoutCreateInfo::builder()
			.set_layouts(&desclayouts);
			
		let pipelinelayout = unsafe { logical_device.create_pipeline_layout(&pipelinelayout_info, None) }?;
		
		// Graphics Pipeline
		let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
			.stages(&shader_stages)
			.vertex_input_state(&vertex_input_info)
			.input_assembly_state(&input_assembly_info)
			.viewport_state(&viewport_info)
			.rasterization_state(&rasterizer_info)
			.multisample_state(&multisampler_info)
			.depth_stencil_state(&depth_stencil_info)
			.color_blend_state(&colourblend_info)
			.layout(pipelinelayout)
			.render_pass(*renderpass)
			.subpass(0);
		let graphicspipeline = unsafe {
			logical_device
				.create_graphics_pipelines(
					vk::PipelineCache::null(),
					&[pipeline_info.build()],
					None,
				).expect("Cannot create pipeline")				
		}[0];
		
		// Destroy used shader modules
		unsafe {
			logical_device.destroy_shader_module(fragmentshader_module, None);
			logical_device.destroy_shader_module(vertexshader_module, None);
		}
		
		Ok(GraphicsPipeline {
			pipeline: graphicspipeline,
			layout: pipelinelayout,
			descriptor_set_layouts: desclayouts,
		})
	}
	
	pub fn cleanup(&self, logical_device: &ash::Device) {
		unsafe {
			for dsl in &self.descriptor_set_layouts {
				logical_device.destroy_descriptor_set_layout(*dsl, None);
			}
			logical_device.destroy_pipeline(self.pipeline, None);
			logical_device.destroy_pipeline_layout(self.layout, None);
		}
	}
}

// Create RenderPass
pub fn init_renderpass(
	logical_device: &ash::Device,
	physical_device: vk::PhysicalDevice,
	surfaces: &Surface
) -> Result<vk::RenderPass, vk::Result> {
	let attachments = [
		vk::AttachmentDescription::builder()
			.format(
				surfaces
					.get_formats(physical_device)?
					.first()
					.unwrap()
					.format,
			)
			.load_op(vk::AttachmentLoadOp::CLEAR)
			.store_op(vk::AttachmentStoreOp::STORE)
			.stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
			.stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
			.initial_layout(vk::ImageLayout::UNDEFINED)
			.final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
			.samples(vk::SampleCountFlags::TYPE_1)
			.build(),
		vk::AttachmentDescription::builder()
			.format(vk::Format::D32_SFLOAT)
			.load_op(vk::AttachmentLoadOp::CLEAR)
			.store_op(vk::AttachmentStoreOp::DONT_CARE)
			.stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
			.stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
			.initial_layout(vk::ImageLayout::UNDEFINED)
			.final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
			.samples(vk::SampleCountFlags::TYPE_1)
			.build(),
	];
	
	// Color attachment reference
	let color_attachment_references = [vk::AttachmentReference {
		attachment: 0,
		layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
	}];
	
	// Depth attachment
	let depth_attachment_references = vk::AttachmentReference {
		attachment: 1,
		layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
	};

	let subpasses = [vk::SubpassDescription::builder()
		.color_attachments(&color_attachment_references)
		.depth_stencil_attachment(&depth_attachment_references)
		.pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
		.build()
	];
	
	let subpass_dependencies = [vk::SubpassDependency::builder()
		.src_subpass(vk::SUBPASS_EXTERNAL)
		.src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
		.dst_subpass(0)
		.dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
		.dst_access_mask(
			vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
		)
		.build()
	];
	
	let renderpass_info = vk::RenderPassCreateInfo::builder()
		.attachments(&attachments)
		.subpasses(&subpasses)
		.dependencies(&subpass_dependencies);
	let renderpass = unsafe { logical_device.create_render_pass(&renderpass_info, None)? };
	Ok(renderpass)
}
