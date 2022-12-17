use gpu_allocator::vulkan::*;
use gpu_allocator::MemoryLocation;
use ash::vk;
use nalgebra as na;

use crate::{
	render::buffer::Buffer,
};

pub type Filter = vk::Filter; 

// Textured VertexData
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct TexturedVertexData {
	pub position: [f32; 3],
	pub normal: [f32; 3],
	pub texcoord: [f32; 2],
}

// Textured InstanceData
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct TexturedInstanceData {
	pub modelmatrix: [[f32; 4]; 4],
	pub inverse_modelmatrix: [[f32; 4]; 4],
	pub texture_id: u32,
	pub metallic: f32,
	pub roughness: f32,
}

impl TexturedInstanceData {
	pub fn new(
		modelmatrix: na::Matrix4<f32>,
		texture_id: usize,
		metallic: f32,
		roughness: f32,
	) -> TexturedInstanceData {
		TexturedInstanceData {
			modelmatrix: modelmatrix.into(),
			inverse_modelmatrix: modelmatrix.try_inverse().unwrap().into(),
			texture_id: texture_id as u32,
			metallic,
			roughness,
		}
	}
}

pub struct Texture {
	pub image: image::RgbaImage,
	pub vk_image: vk::Image,
	pub image_allocation: Option<Allocation>,
	pub imageview: vk::ImageView,
	pub sampler: vk::Sampler,
}

impl Texture {
	pub fn from_file<P: AsRef<std::path::Path>>(
		path: P, 
		filter: Filter,
		logical_device: &ash::Device,
		allocator: &mut Allocator,
		commandpool_graphics: &vk::CommandPool,
		graphics_queue: &vk::Queue,
	) -> Result<Self, vk::Result> {
		// Create image
		let image = image::open(path)
			.map(|img| img.to_rgba8())
			.expect("unable to open image");
			
		let (width, height) = image.dimensions();
		
		let img_create_info = vk::ImageCreateInfo::builder()
			.image_type(vk::ImageType::TYPE_2D)
			.extent(vk::Extent3D {
				width,
				height,
				depth: 1,
			})
			.mip_levels(1)
			.array_layers(1)
			.format(vk::Format::R8G8B8A8_SRGB)
			.samples(vk::SampleCountFlags::TYPE_1)
			.usage(
				vk::ImageUsageFlags::TRANSFER_DST |
				vk::ImageUsageFlags::SAMPLED
			);
		let vk_image = unsafe { logical_device.create_image(&img_create_info, None)? };
		// Allocation info
		let allocation_info = &AllocationCreateDesc {
			name: "Texture allocation",
			requirements: unsafe { logical_device.get_image_memory_requirements(vk_image) },
			location: MemoryLocation::GpuOnly,
			linear: true,
		};
		// Create memory allocation
		let allocation = allocator.allocate(allocation_info).unwrap();
		// Bind memory allocation to the vk_image
		unsafe { logical_device.bind_image_memory(
			vk_image, 
			allocation.memory(), 
			allocation.offset()).unwrap()
		};
		
		// Create ImageView
		let view_create_info = vk::ImageViewCreateInfo::builder()
			.image(vk_image)
			.view_type(vk::ImageViewType::TYPE_2D)
			.format(vk::Format::R8G8B8A8_SRGB)
			.subresource_range(vk::ImageSubresourceRange {
				aspect_mask: vk::ImageAspectFlags::COLOR,
				level_count: 1,
				layer_count: 1,
				..Default::default()
			});
		let imageview = unsafe { logical_device.create_image_view(&view_create_info, None)? };
		
		// Create Sampler
		let sampler_info = vk::SamplerCreateInfo::builder()
			.mag_filter(filter)
			.min_filter(filter);
		let sampler = unsafe { logical_device.create_sampler(&sampler_info, None)? };
		
		// Prepare buffer for the texture
		let data = image.clone().into_raw();
		let mut buffer = Buffer::new(
			&logical_device,
			allocator,
			data.len() as u64,
			vk::BufferUsageFlags::TRANSFER_SRC,
			MemoryLocation::CpuToGpu,
			"Texture allocation"
		)?;
		buffer.fill(&logical_device, allocator, &data)?;
		
		// Create CommandBuffer
		let commandbuf_allocate_info = vk::CommandBufferAllocateInfo::builder()
			.command_pool(*commandpool_graphics)
			.command_buffer_count(1);
		let copycmdbuffer = unsafe {
			logical_device.allocate_command_buffers(&commandbuf_allocate_info)
		}
		.unwrap()[0];

		let cmdbegininfo = vk::CommandBufferBeginInfo::builder()
			.flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
			
		// Begin CommandBuffer
		unsafe {
			logical_device.begin_command_buffer(copycmdbuffer, &cmdbegininfo)
		}?;
		
		// Change image layout for transfering
		let barrier = vk::ImageMemoryBarrier::builder()
			.image(vk_image)
			.src_access_mask(vk::AccessFlags::empty())
			.dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
			.old_layout(vk::ImageLayout::UNDEFINED)
			.new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
			.subresource_range(vk::ImageSubresourceRange {
				aspect_mask: vk::ImageAspectFlags::COLOR,
				base_mip_level: 0,
				level_count: 1,
				base_array_layer: 0,
				layer_count: 1,
			})
			.build();
			
		unsafe {
			logical_device.cmd_pipeline_barrier(
				copycmdbuffer,
				vk::PipelineStageFlags::TOP_OF_PIPE,
				vk::PipelineStageFlags::TRANSFER,
				vk::DependencyFlags::empty(),
				&[],
				&[],
				&[barrier],
			)
		};
		
		// Copy data from the buffer to the image
		let image_subresource = vk::ImageSubresourceLayers {
			aspect_mask: vk::ImageAspectFlags::COLOR,
			mip_level: 0,
			base_array_layer: 0,
			layer_count: 1,
		};
		
		let region = vk::BufferImageCopy {
			buffer_offset: 0,
			buffer_row_length: 0,
			buffer_image_height: 0,
			image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
			image_extent: vk::Extent3D {
				width,
				height,
				depth: 1,
			},
			image_subresource,
			..Default::default()
		};
		
		unsafe {
			logical_device.cmd_copy_buffer_to_image(
				copycmdbuffer,
				buffer.buffer,
				vk_image,
				vk::ImageLayout::TRANSFER_DST_OPTIMAL,
				&[region],
			);
		}
		
		// Change image layout for fragment shader
		
		let barrier = vk::ImageMemoryBarrier::builder()
			.image(vk_image)
			.src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
			.dst_access_mask(vk::AccessFlags::SHADER_READ)
			.old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
			.new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
			.subresource_range(vk::ImageSubresourceRange {
				aspect_mask: vk::ImageAspectFlags::COLOR,
				base_mip_level: 0,
				level_count: 1,
				base_array_layer: 0,
				layer_count: 1,
			})
			.build();
			
		unsafe {
			logical_device.cmd_pipeline_barrier(
				copycmdbuffer,
				vk::PipelineStageFlags::TRANSFER,
				vk::PipelineStageFlags::FRAGMENT_SHADER,
				vk::DependencyFlags::empty(),
				&[],
				&[],
				&[barrier],
			)
		};
		
		// End CommandBuffer
		unsafe { logical_device.end_command_buffer(copycmdbuffer) }?;
		let submit_infos = [vk::SubmitInfo::builder()
			.command_buffers(&[copycmdbuffer])
			.build()];
		let fence = unsafe {
			logical_device.create_fence(&vk::FenceCreateInfo::default(), None)
		}?;
		
		unsafe {
			logical_device.queue_submit(*graphics_queue, &submit_infos, fence)
		}?;
		
		// Destroy buffer
		unsafe { logical_device.wait_for_fences(&[fence], true, std::u64::MAX) }?;
		unsafe { logical_device.destroy_fence(fence, None) };
				
		let mut alloc: Option<Allocation> = None;
		std::mem::swap(&mut alloc, &mut buffer.allocation);
		let alloc = alloc.unwrap();
		allocator.free(alloc).unwrap();
		unsafe { logical_device.destroy_buffer(buffer.buffer, None) };
		
		unsafe {
			logical_device.free_command_buffers(
				*commandpool_graphics,
				&[copycmdbuffer]
			)
		};
		
		Ok(Texture {
			image,
			vk_image,
			image_allocation: Some(allocation),
			imageview,
			sampler,
		})
	}
}

// Texture Storage (temporary)
pub struct TextureStorage {
	pub textures: Vec<Texture>,
}

impl TextureStorage {
	pub fn new() -> Self {
		TextureStorage { textures: vec![] }
	}
	
	pub fn cleanup(
		&mut self,
		logical_device: &ash::Device,
		allocator: &mut Allocator,
	){
		for texture in &mut self.textures {
			// Destroy allocation
			let mut alloc: Option<Allocation> = None;
			std::mem::swap(&mut alloc, &mut texture.image_allocation);
			let alloc = alloc.unwrap();
			allocator.free(alloc).unwrap();
			unsafe { 
				// Destroy Sampler
				logical_device.destroy_sampler(texture.sampler, None);
				// Destroy ImageView
				logical_device.destroy_image_view(texture.imageview, None);
				// Destroy Image
				logical_device.destroy_image(texture.vk_image, None);
			}
		}
	}
	
	pub fn new_texture_from_file<P: AsRef<std::path::Path>>(
		&mut self,
		path: P,
		filter: Filter,
		logical_device: &ash::Device,
		allocator: &mut Allocator,
		commandpool_graphics: &vk::CommandPool,
		graphics_queue: &vk::Queue,
	) -> Result<usize, Box<dyn std::error::Error>> {
		let new_texture = Texture::from_file(
			path,
			filter,
			logical_device,
			allocator,
			commandpool_graphics,
			graphics_queue,
		)?;
		let new_id = self.textures.len();
		self.textures.push(new_texture);
		Ok(new_id)
	}
	
	pub fn get(&self, index: usize) -> Option<&Texture> {
		self.textures.get(index)
	}
	
	pub fn get_mut(&mut self, index: usize) -> Option<&mut Texture> {
		self.textures.get_mut(index)
	}
	
	pub fn get_descriptor_image_info(&self) -> Vec<vk::DescriptorImageInfo> {
		self.textures
			.iter()
			.map(|t| vk::DescriptorImageInfo {
				image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
				image_view: t.imageview,
				sampler: t.sampler,
				..Default::default()
			})
			.collect()
	}
}

