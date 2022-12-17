use nalgebra as na;
use ash::vk;
use winit::event::{Event, WindowEvent};
use despero::Despero;
use despero::{
	engine::{
		model::{
			Model,
		},
		camera::Camera,
		debug::Debug,
		screenshot::Screenshot,
		light::*,
		texture::{
			Filter,
			TexturedInstanceData,
		},
	},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let eventloop = winit::event_loop::EventLoop::new();
	let window = winit::window::Window::new(&eventloop)?;
	// Main struct
	let mut despero = Despero::init(window, "App Name")?;
	// Models
	//let mut sphere = Model::sphere(3);
	let mut quad = Model::quad();
	
	let texture_id = despero.texture_from_file("assets/image.jpg", Filter::LINEAR)?;
	let second_texture_id = despero.texture_from_file("assets/image2.jpg", Filter::LINEAR)?;
	let third_texture_id = despero.texture_from_file("assets/image.jpg", Filter::NEAREST)?;
	
	quad.insert_visibly(TexturedInstanceData::new(
		na::Matrix4::identity(),
		texture_id,
		0.0,
		1.0,
	));
	
	quad.insert_visibly(TexturedInstanceData::new(
        na::Matrix4::new_translation(&na::Vector3::new(2.0, 0., 0.3)),
        second_texture_id,
        0.0,
		1.0,
    ));
	
	quad.insert_visibly(TexturedInstanceData::new(
        na::Matrix4::new_translation(&na::Vector3::new(5.0, 0., 0.3)),
        third_texture_id,
        0.0,
		1.0,
    ));
	
	/*for i in 0..10 {
		for j in 0..10 {
			sphere.insert_visibly(InstanceData::new(
				na::Matrix4::new_translation(&na::Vector3::new(i as f32 - 5., j as f32 + 5., 10.0))
					* na::Matrix4::new_scaling(0.5),
				[0., 0., 0.8],
				i as f32 * 0.1,
				j as f32 * 0.1,
			));
		}
	}*/
	
	quad.update_vertexbuffer(&despero.device, &mut despero.allocator)?;
	quad.update_instancebuffer(&despero.device, &mut despero.allocator)?;
	quad.update_indexbuffer(&despero.device, &mut despero.allocator)?;
	despero.models = vec![quad];
	
	//Camera
	let mut camera = Camera::builder().build();
	// Lights
	let mut lights = LightManager::default();
	lights.add_light(DirectionalLight {
		direction: na::Vector3::new(-1., -1., 0.),
		illuminance: [0.5, 0.5, 0.5],
	});
	lights.add_light(PointLight {
		position: na::Point3::new(0.1, -3.0, -3.0),
		luminous_flux: [100.0, 100.0, 100.0],
	});
	lights.add_light(PointLight {
		position: na::Point3::new(0.1, -3.0, -3.0),
		luminous_flux: [100.0, 100.0, 100.0],
	});
	lights.add_light(PointLight {
		position: na::Point3::new(0.1, -3.0, -3.0),
		luminous_flux: [100.0, 100.0, 100.0],
	});
	
	lights.update_buffer(
		&despero.device, 
		&mut despero.allocator, 
		&mut despero.lightbuffer, 
		&mut despero.descriptor_sets_light
	)?;
	
	eventloop.run(move |event, _, controlflow| match event {
		Event::WindowEvent {
			event: WindowEvent::CloseRequested,
			..
		} => {
			*controlflow = winit::event_loop::ControlFlow::Exit;
		}
		
		Event::WindowEvent {
			event: WindowEvent::KeyboardInput {input, ..},
			..
		} => match input {
			winit::event::KeyboardInput {
				state: winit::event::ElementState::Pressed,
				virtual_keycode: Some(keycode),
				..
			} => match keycode {
				// System
				winit::event::VirtualKeyCode::F5 => {
					let path = "screenshots";
					let name = "name";
					Screenshot::take_jpg(&mut despero, name, path).expect("Failed to create a screenshot");
					Debug::info(format!("Screenshot \"{}\" saved in \"{}\"", name, path).as_str());
				}
				winit::event::VirtualKeyCode::F11 => {
					despero.texture_storage.textures.swap(0, 1);
				}
				// Rotating
				winit::event::VirtualKeyCode::Right => {
					camera.turn_right(0.05);
				}
				winit::event::VirtualKeyCode::Left => {
					camera.turn_left(0.05);
				}
				winit::event::VirtualKeyCode::Up => {
					camera.turn_up(0.05);
				}
				winit::event::VirtualKeyCode::Down => {
					camera.turn_down(0.05);
				}
				// Movement
				winit::event::VirtualKeyCode::W => {
					camera.move_forward(0.05);
				}
				winit::event::VirtualKeyCode::S => {
					camera.move_backward(0.05);
				}
				winit::event::VirtualKeyCode::A => {
					camera.move_left(0.05);
				}
				winit::event::VirtualKeyCode::D => {
					camera.move_right(0.05);
				}
				_ => {}
			},
			_ => {}
		},
		
		Event::MainEventsCleared => {
			despero.window.request_redraw();
		}
		
		Event::RedrawRequested(_) => {
			// Get image of swapchain
			let (image_index, _) = unsafe {
				despero
					.swapchain
					.swapchain_loader
					.acquire_next_image(
						despero.swapchain.swapchain,
						std::u64::MAX,
						despero.swapchain.image_available[despero.swapchain.current_image],
						vk::Fence::null(),
					)
					.expect("Error image acquisition")
			};
			// Control fences
			unsafe {
				despero
					.device
					.wait_for_fences(
						&[despero.swapchain.may_begin_drawing[despero.swapchain.current_image]],
						true,
						std::u64::MAX,
					)
					.expect("fence-waiting");
				despero
					.device
					.reset_fences(&[
						despero.swapchain.may_begin_drawing[despero.swapchain.current_image]
					])
					.expect("resetting fences");
			}
			
			camera.update_buffer(
				&despero.device, 
				&mut despero.allocator, 
				&mut despero.uniformbuffer
			).expect("Cannot update uniformbuffer");
			
			// Get image descriptor info
			let imageinfos = despero.texture_storage.get_descriptor_image_info();
			let descriptorwrite_image = vk::WriteDescriptorSet::builder()
				.dst_set(despero.descriptor_sets_texture[despero.swapchain.current_image])
				.dst_binding(0)
				.dst_array_element(0)
				.descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&imageinfos)
                .build();

			// Update descriptors
			unsafe {
				despero
					.device
					.update_descriptor_sets(&[descriptorwrite_image], &[]);
			}
			
			despero
				.update_commandbuffer(image_index as usize)
				.expect("Cannot update CommandBuffer");
			
			// Submit commandbuffers
			let semaphores_available = [despero.swapchain.image_available[despero.swapchain.current_image]];
			let waiting_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
			let semaphores_finished = [despero.swapchain.rendering_finished[despero.swapchain.current_image]];
			let commandbuffers = [despero.commandbuffers[image_index as usize]];
			let submit_info = [vk::SubmitInfo::builder()
				.wait_semaphores(&semaphores_available)
				.wait_dst_stage_mask(&waiting_stages)
				.command_buffers(&commandbuffers)
				.signal_semaphores(&semaphores_finished)
				.build()];
			unsafe {
				despero
					.device
					.queue_submit(
						despero.queues.graphics_queue,
						&submit_info,
						despero.swapchain.may_begin_drawing[despero.swapchain.current_image],
					)
					.expect("queue submission");
			};
			let swapchains = [despero.swapchain.swapchain];
			let indices = [image_index];
			let present_info = vk::PresentInfoKHR::builder()
				.wait_semaphores(&semaphores_finished)
				.swapchains(&swapchains)
				.image_indices(&indices);
			unsafe {
				if despero
					.swapchain
					.swapchain_loader
					.queue_present(despero.queues.graphics_queue, &present_info)
					.expect("queue presentation")
				{
					despero.recreate_swapchain().expect("swapchain recreation");
					
					camera.set_aspect(
						despero.swapchain.extent.width as f32
							/ despero.swapchain.extent.height as f32,
					);
					
					camera
						.update_buffer(
							&despero.device, 
							&mut despero.allocator, 
							&mut despero.uniformbuffer
						).expect("camera buffer update");
				}
			};
			// Set swapchain image
			despero.swapchain.current_image =
				(despero.swapchain.current_image + 1) % despero.swapchain.amount_of_images as usize;
		}
		_ => {}
	});
}
