//
//   _____							 _____   _									_ 
//  / ____|							  |_   _| | |								  | |
// | (___   ___  _ __  _   _  __ _		| |   | | _____   _____   _   _  ___  _   _| |
//  \___ \ / _ \| '_ \| | | |/ _` |	 | |   | |/ _ \ \ / / _ \ | | | |/ _ \| | | | |
//  ____) | (_) | | | | |_| | (_| |_   _| |_  | | (_) \ V /  __/ | |_| | (_) | |_| |_|
// |_____/ \___/|_| |_|\__, |\__,_( ) |_____| |_|\___/ \_/ \___|  \__, |\___/ \__,_(_)
//					  __/ |	 |/							   __/ |			  
//					   |___/									  |___/		 
//						   
#[cfg(all(feature = "x11", feature = "windows"))]
compile_error!("features \"x11\" and \"windows\" cannot be enabled at the same time");

use ash::vk;
use gpu_allocator::vulkan::*;
use gpu_allocator::MemoryLocation;
use nalgebra as na;

pub mod render;
pub mod engine;
pub mod ecs;
pub mod physics;
pub mod scripting;

use crate::render::{
	surface::Surface,
	swapchain::Swapchain,
	queues::{
		QueueFamilies,
		Queues,
		init_instance,
		init_device_and_queues,
		init_physical_device_and_properties,
	},
	pipeline::{
		GraphicsPipeline,
		init_renderpass,
		MAX_NUMBER_OF_TEXTURES,
	},
	commandbuffers::{
		CommandBufferPools,
		create_commandbuffers,
	},
	buffer::Buffer,
};

use crate::engine::texture::{
	TexturedInstanceData,
	TexturedVertexData,
	TextureStorage,
	Filter,
};

use crate::engine::{
	debug::Debug,
	model::{
		Model,
	},
};

// Main struct
pub struct Despero {
	pub window: winit::window::Window,
	pub entry: ash::Entry,
	pub instance: ash::Instance,
	pub debug: std::mem::ManuallyDrop<Debug>,
	pub surfaces: std::mem::ManuallyDrop<Surface>,
	pub physical_device: vk::PhysicalDevice,
	pub physical_device_properties: vk::PhysicalDeviceProperties,
	pub queue_families: QueueFamilies,
	pub queues: Queues,
	pub device: ash::Device,
	pub swapchain: Swapchain,
	pub renderpass: vk::RenderPass,
	pub pipeline: GraphicsPipeline,
	pub commandbuffer_pools: CommandBufferPools,
	pub commandbuffers: Vec<vk::CommandBuffer>,
	pub allocator: gpu_allocator::vulkan::Allocator,
	pub models: Vec<Model<TexturedVertexData, TexturedInstanceData>>,
	pub texture_storage: TextureStorage,
	pub uniformbuffer: Buffer,
	pub lightbuffer: Buffer,
	pub descriptor_pool: vk::DescriptorPool,
	pub descriptor_sets_camera: Vec<vk::DescriptorSet>, 
	pub descriptor_sets_texture: Vec<vk::DescriptorSet>,
	pub descriptor_sets_light: Vec<vk::DescriptorSet>, 
}

impl Despero {
	pub fn init(
		window: winit::window::Window,
		app_title: &str,
	) -> Result<Despero, Box<dyn std::error::Error>> {
		// Set window title
		window.set_title(app_title);
		// Create Entry
		let entry = unsafe { ash::Entry::load()? };
		// Instance, Debug, Surface
		let layer_names = vec!["VK_LAYER_KHRONOS_validation"];
		let instance = init_instance(&entry, &layer_names, app_title)?;
		let debug = Debug::init(&entry, &instance)?;
		let surfaces = Surface::init(&window, &entry, &instance)?;
		
		// PhysicalDevice and PhysicalDeviceProperties
		let (physical_device, physical_device_properties, _) = init_physical_device_and_properties(&instance)?;
		// QueueFamilies, (Logical) Device, Queues
		let queue_families = QueueFamilies::init(&instance, physical_device, &surfaces)?;
		let (logical_device, queues) = init_device_and_queues(&instance, physical_device, &queue_families, &layer_names)?;
		
		// Create memory allocator
		let mut allocator = Allocator::new(&AllocatorCreateDesc {
			instance: instance.clone(),
			device: logical_device.clone(),
			physical_device,
			debug_settings: Default::default(),
			buffer_device_address: true,
		}).expect("Cannot create allocator");
		
		// Swapchain
		let mut swapchain = Swapchain::init(
			&instance, 
			physical_device, 
			&logical_device, 
			&surfaces, 
			&queue_families,
			&mut allocator
		)?;
		
		// RenderPass, Pipeline
		let renderpass = init_renderpass(&logical_device, physical_device, &surfaces)?;
		swapchain.create_framebuffers(&logical_device, renderpass)?;
		let pipeline = GraphicsPipeline::init_textured(&logical_device, &swapchain, &renderpass)?;
		
		// CommandBufferPools and CommandBuffers
		let commandbuffer_pools = CommandBufferPools::init(&logical_device, &queue_families)?;
		let commandbuffers = create_commandbuffers(&logical_device, &commandbuffer_pools, swapchain.framebuffers.len())?;
		
		// Uniform buffer
		let mut uniformbuffer = Buffer::new(
			&logical_device,
			&mut allocator,
			128,
			vk::BufferUsageFlags::UNIFORM_BUFFER,
			MemoryLocation::CpuToGpu,
			"Uniform buffer"
		)?;
		
		// Light buffer
		let mut lightbuffer = Buffer::new(
			&logical_device,
			&mut allocator,
			8,
			vk::BufferUsageFlags::STORAGE_BUFFER,
			MemoryLocation::CpuToGpu,
			"Light buffer",
		)?;
		lightbuffer.fill(&logical_device, &mut allocator, &[0.,0.])?;
		
		// Camera transform
		let cameratransform: [[[f32; 4]; 4]; 2] = [
			na::Matrix4::identity().into(),
			na::Matrix4::identity().into(),
		];
		uniformbuffer.fill(&logical_device, &mut allocator, &cameratransform)?;
		
		// Descriptor pool
		//
		// Set pool size
		let pool_sizes = [
			vk::DescriptorPoolSize {
				ty: vk::DescriptorType::UNIFORM_BUFFER,
				descriptor_count: swapchain.amount_of_images,
			},
			vk::DescriptorPoolSize {
				ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
				descriptor_count: MAX_NUMBER_OF_TEXTURES * swapchain.amount_of_images,
			},
			vk::DescriptorPoolSize {
				ty: vk::DescriptorType::STORAGE_BUFFER,
				descriptor_count: swapchain.amount_of_images,
			},
		];
		// PoolCreateInfo
		let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
			// Amount of descriptors
			.max_sets(3 * swapchain.amount_of_images)
			// Size of pool
			.pool_sizes(&pool_sizes); 
		let descriptor_pool = unsafe { logical_device.create_descriptor_pool(&descriptor_pool_info, None) }?;
		
		// Descriptor sets (Camera)
		//
		// Descriptor layouts (Camera)
		let desc_layouts_camera = vec![pipeline.descriptor_set_layouts[0]; swapchain.amount_of_images as usize];
		// SetAllocateInfo (Camera)
		let descriptor_set_allocate_info_camera = vk::DescriptorSetAllocateInfo::builder()
			// DescPool
			.descriptor_pool(descriptor_pool)
			// Layouts
			.set_layouts(&desc_layouts_camera);
		let descriptor_sets_camera = unsafe { logical_device.allocate_descriptor_sets(&descriptor_set_allocate_info_camera) }?;

		// Fill descriptor sets (Camera)
		for descset in &descriptor_sets_camera {
			let buffer_infos = [vk::DescriptorBufferInfo {
				buffer: uniformbuffer.buffer,
				offset: 0,
				range: 128,
			}];
			let desc_sets_write = [vk::WriteDescriptorSet::builder()
				.dst_set(*descset)
				.dst_binding(0)
				.descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
				.buffer_info(&buffer_infos)
				.build()
			];
			unsafe { logical_device.update_descriptor_sets(&desc_sets_write, &[]) };
		}
		
		// Descriptor sets (Texture)
		//
		// Descriptor layouts (Texture)
		let desc_layouts_texture = vec![pipeline.descriptor_set_layouts[1]; swapchain.amount_of_images as usize];
		// SetAllocateInfo (Texture)
		let descriptor_set_allocate_info_texture = vk::DescriptorSetAllocateInfo::builder()
			// DescPool
			.descriptor_pool(descriptor_pool)
			// Layouts
			.set_layouts(&desc_layouts_texture);
		let descriptor_sets_texture = unsafe { logical_device.allocate_descriptor_sets(&descriptor_set_allocate_info_texture) }?;
		
		// Descriptor sets (Light)
		//
		// Descriptor layouts (Light)
		let desc_layouts_light = vec![pipeline.descriptor_set_layouts[2]; swapchain.amount_of_images as usize];
		// SetAllocateInfo (Light)
		let descriptor_set_allocate_info_light = vk::DescriptorSetAllocateInfo::builder()
			// DescPool
			.descriptor_pool(descriptor_pool)
			// Layouts
			.set_layouts(&desc_layouts_light);
		let descriptor_sets_light = unsafe { logical_device.allocate_descriptor_sets(&descriptor_set_allocate_info_light) }?;
		// Fill descriptor sets (Light)
		for descset in &descriptor_sets_light {
			let buffer_infos = [vk::DescriptorBufferInfo {
				buffer: lightbuffer.buffer,
				offset: 0,
				range: 8,
			}];
			let desc_sets_write = [vk::WriteDescriptorSet::builder()
				.dst_set(*descset)
				.dst_binding(0)
				.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
				.buffer_info(&buffer_infos)
				.build()
			];
			unsafe { logical_device.update_descriptor_sets(&desc_sets_write, &[]) };
		}
		 
		Ok(Despero {
			window,
			entry,
			instance,
			debug: std::mem::ManuallyDrop::new(debug),
			surfaces: std::mem::ManuallyDrop::new(surfaces),
			physical_device,
			physical_device_properties,
			queue_families,
			queues,
			device: logical_device,
			swapchain,
			renderpass,
			pipeline,
			commandbuffer_pools,
			commandbuffers,
			allocator,
			models: vec![],
			texture_storage: TextureStorage::new(),
			uniformbuffer,
			lightbuffer,
			descriptor_pool,
			descriptor_sets_camera,
			descriptor_sets_texture,
			descriptor_sets_light,
		})
	}
	
	pub fn update_commandbuffer(&mut self, index: usize) -> Result<(), vk::Result> {
		let commandbuffer = self.commandbuffers[index];
		let commandbuffer_begininfo = vk::CommandBufferBeginInfo::builder();
		
		unsafe {
			self.device
				.begin_command_buffer(commandbuffer, &commandbuffer_begininfo)?;
		}
		
		let clearvalues = [
			vk::ClearValue {
				color: vk::ClearColorValue {
					float32: [0.0, 0.0, 0.0, 1.0],
				},
			},
			vk::ClearValue {
				depth_stencil: vk::ClearDepthStencilValue {
					depth: 1.0,
					stencil: 0,
				},
			},
		];
		
		let renderpass_begininfo = vk::RenderPassBeginInfo::builder()
			.render_pass(self.renderpass)
			.framebuffer(self.swapchain.framebuffers[index])
			.render_area(vk::Rect2D {
				offset: vk::Offset2D { x: 0, y: 0 },
				extent: self.swapchain.extent,
			})
			.clear_values(&clearvalues);
		unsafe {
			// Bind RenderPass
			self.device.cmd_begin_render_pass(
				commandbuffer,
				&renderpass_begininfo,
				vk::SubpassContents::INLINE,
			);
			// Bind Pipeline
			self.device.cmd_bind_pipeline(
				commandbuffer,
				vk::PipelineBindPoint::GRAPHICS,
				self.pipeline.pipeline,
			);
			// Bind DescriptorSets
			self.device.cmd_bind_descriptor_sets(
				commandbuffer,
				vk::PipelineBindPoint::GRAPHICS,
				self.pipeline.layout,
				0,
				&[
					self.descriptor_sets_camera[index],
					self.descriptor_sets_texture[index],
					self.descriptor_sets_light[index],
				],
				&[],
			);
			for m in &self.models {
				m.draw(
					&self.device,
					commandbuffer,
				);
			}
			self.device.cmd_end_render_pass(commandbuffer);
			self.device.end_command_buffer(commandbuffer)?;
		}
		Ok(())
	}
	
	pub fn recreate_swapchain(&mut self) -> Result<(), Box<dyn std::error::Error>> {
		unsafe {
			self.device
				.device_wait_idle()
				.expect("something wrong while waiting");
			self.swapchain.cleanup(&self.device, &mut self.allocator);
		}
		// Recreate Swapchain
		self.swapchain = Swapchain::init(
			&self.instance,
			self.physical_device,
			&self.device,
			&self.surfaces,
			&self.queue_families,
			&mut self.allocator,
		)?;
		
		// Recreate FrameBuffers
		self.swapchain.create_framebuffers(&self.device, self.renderpass)?;
		
		// Recreate Pipeline
		self.pipeline.cleanup(&self.device);
		self.pipeline = GraphicsPipeline::init(
			&self.device, 
			&self.swapchain, 
			&self.renderpass
		)?;
		
		Ok(())
	}
	
	pub fn texture_from_file<P: AsRef<std::path::Path>>(
		&mut self,
		path: P,
		filter: Filter,
	) -> Result<usize, Box<dyn std::error::Error>> {
		self.texture_storage.new_texture_from_file(
			path,
			filter,
			&self.device,
			&mut self.allocator,
			&self.commandbuffer_pools.commandpool_graphics,
			&self.queues.graphics_queue,
		)
	}
}

impl Drop for Despero {
	fn drop(&mut self) {
		unsafe {
			self.device
				.device_wait_idle()
				.expect("Error halting device");	
			// Destroy TextureStorage
			self.texture_storage.cleanup(&self.device, &mut self.allocator);
			self.device.destroy_descriptor_pool(self.descriptor_pool, None);
			// Destroy UniformBuffer
			self.device.destroy_buffer(self.uniformbuffer.buffer, None);
			self.device.free_memory(self.uniformbuffer.allocation.as_ref().unwrap().memory(), None);
			// Destroy LightBuffer
			self.device.destroy_buffer(self.lightbuffer.buffer, None);
			// Models clean
			for m in &mut self.models {
				if let Some(vb) = &mut m.vertexbuffer {
					// Reassign VertexBuffer allocation to remove
					let mut alloc: Option<Allocation> = None;
					std::mem::swap(&mut alloc, &mut vb.allocation);
					let alloc = alloc.unwrap();
					self.allocator.free(alloc).unwrap();
					self.device.destroy_buffer(vb.buffer, None);
				}
				
				if let Some(xb) = &mut m.indexbuffer {
					// Reassign IndexBuffer allocation to remove
					let mut alloc: Option<Allocation> = None;
					std::mem::swap(&mut alloc, &mut xb.allocation);
					let alloc = alloc.unwrap();
					self.allocator.free(alloc).unwrap();
					self.device.destroy_buffer(xb.buffer, None);
				}
				
				if let Some(ib) = &mut m.instancebuffer {
					// Reassign IndexBuffer allocation to remove
					let mut alloc: Option<Allocation> = None;
					std::mem::swap(&mut alloc, &mut ib.allocation);
					let alloc = alloc.unwrap();
					self.allocator.free(alloc).unwrap();
					self.device.destroy_buffer(ib.buffer, None);
				}
			}
			self.commandbuffer_pools.cleanup(&self.device);
			self.pipeline.cleanup(&self.device);
			self.device.destroy_render_pass(self.renderpass, None);
			self.swapchain.cleanup(&self.device, &mut self.allocator);
			self.device.destroy_device(None);
			std::mem::ManuallyDrop::drop(&mut self.surfaces);
			std::mem::ManuallyDrop::drop(&mut self.debug);
			self.instance.destroy_instance(None);
		};
	}
}
