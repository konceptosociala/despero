use ash::vk;
use hecs_schedule::*;

use crate::render::{
	renderer::Renderer,
	transform::Transform,
	pbr::{
		camera::Camera,
		model::{
			Model,
			TexturedInstanceData,
			TexturedVertexData,
		},
	},
};

pub(crate) fn rendering_system(
	mut renderer: Write<Renderer>,
	mut model_world: SubWorld<(&mut Model<TexturedVertexData, TexturedInstanceData>, &Transform)>,
	camera_world: SubWorld<(&mut Camera, &Transform)>,
){
	// Get image of swapchain
	let (image_index, _) = unsafe {
		renderer
			.swapchain
			.swapchain_loader
			.acquire_next_image(
				renderer.swapchain.swapchain,
				std::u64::MAX,
				renderer.swapchain.image_available[renderer.swapchain.current_image],
				vk::Fence::null(),
			)
			.expect("Error image acquisition")
	};
				
	// Control fences
	unsafe {
		renderer
			.device
			.wait_for_fences(
				&[renderer.swapchain.may_begin_drawing[renderer.swapchain.current_image]],
				true,
				std::u64::MAX,
			)
			.expect("fence-waiting");
		renderer
			.device
			.reset_fences(&[
				renderer.swapchain.may_begin_drawing[renderer.swapchain.current_image]
			])
			.expect("resetting fences");
	}
	
	// Update active camera's buffer
	for (_, camera) in &mut camera_world.query::<&mut Camera>(){
		if camera.is_active {		
			camera.update_buffer(&mut renderer).expect("Cannot update uniformbuffer");
		}
	}
	
	// Get image descriptor info
	let imageinfos = renderer.texture_storage.get_descriptor_image_info();
	let descriptorwrite_image = vk::WriteDescriptorSet::builder()
		.dst_set(renderer.descriptor_sets_texture[renderer.swapchain.current_image])
		.dst_binding(0)
		.dst_array_element(0)
		.descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
		.image_info(&imageinfos)
		.build();

	// Update descriptors
	unsafe {
		renderer
			.device
			.update_descriptor_sets(&[descriptorwrite_image], &[]);
	}

	// Update CommandBuffer
	renderer.update_commandbuffer(
		&mut model_world,
		image_index as usize,
	).expect("Cannot update CommandBuffer");
	
	// Submit commandbuffers
	let semaphores_available = [renderer.swapchain.image_available[renderer.swapchain.current_image]];
	let waiting_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
	let semaphores_finished = [renderer.swapchain.rendering_finished[renderer.swapchain.current_image]];
	let commandbuffers = [renderer.commandbuffers[image_index as usize]];
	let submit_info = [vk::SubmitInfo::builder()
		.wait_semaphores(&semaphores_available)
		.wait_dst_stage_mask(&waiting_stages)
		.command_buffers(&commandbuffers)
		.signal_semaphores(&semaphores_finished)
		.build()];
	unsafe {
		renderer
			.device
			.queue_submit(
				renderer.queues.graphics_queue,
				&submit_info,
				renderer.swapchain.may_begin_drawing[renderer.swapchain.current_image],
			)
			.expect("queue submission");
	};
	let swapchains = [renderer.swapchain.swapchain];
	let indices = [image_index];
	let present_info = vk::PresentInfoKHR::builder()
		.wait_semaphores(&semaphores_finished)
		.swapchains(&swapchains)
		.image_indices(&indices);
	unsafe {
		if renderer
			.swapchain
			.swapchain_loader
			.queue_present(renderer.queues.graphics_queue, &present_info)
			.expect("queue presentation")
		{
			renderer.recreate_swapchain().expect("Cannot recreate swapchain");
			
			for (_, camera) in &mut camera_world.query::<&mut Camera>(){
				if camera.is_active {
					camera.set_aspect(
						renderer.swapchain.extent.width as f32
							/ renderer.swapchain.extent.height as f32,
					);

					camera.update_buffer(&mut renderer).expect("Cannot update camera buffer");
				}
			}
		}
	};
	// Set swapchain image
	renderer.swapchain.current_image =
		(renderer.swapchain.current_image + 1) % renderer.swapchain.amount_of_images as usize;
}

pub(crate) fn init_models_system(
	mut renderer: Write<Renderer>,
	world: SubWorld<(&mut Model<TexturedVertexData, TexturedInstanceData>, &Transform)>,
){
	for (_, model) in &mut world.query::<&mut Model<TexturedVertexData, TexturedInstanceData>>() {
		model.update_vertexbuffer(&mut renderer).expect("Cannot update vertexbuffer");	
		model.update_instancebuffer(&mut renderer).expect("Cannot update instancebuffer");
		model.update_indexbuffer(&mut renderer).expect("Cannot update indexbuffer");
	}
}
