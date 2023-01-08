//! ```text
//!                                                                             
//!          8I                                                                 
//!          8I                                                                 
//!          8I                                                                 
//!          8I                                                                 
//!    ,gggg,8I   ,ggg,     ,g,     gg,gggg,     ,ggg,    ,gggggg,    ,ggggg,   
//!   dP"  "Y8I  i8" "8i   ,8'8,    I8P"  "Yb   i8" "8i   dP""""8I   dP"  "Y8ggg
//!  i8'    ,8I  I8, ,8I  ,8'  Yb   I8'    ,8i  I8, ,8I  ,8'    8I  i8'    ,8I  
//! ,d8,   ,d8b, `YbadP' ,8'_   8) ,I8 _  ,d8'  `YbadP' ,dP     Y8,,d8,   ,d8'  
//! P"Y8888P"`Y8888P"Y888P' "YY8P8PPI8 YY88888P888P"Y8888P      `Y8P"Y8888P"    
//!                                 I8                                          
//!                                 I8                                          
//!                                 I8                                          
//!                                 I8                                          
//!                                 I8                                          
//!                                 I8                                          
//! ```
//! Despero (_esp._ **despair**) is rusty data-driven 3D game engine, 
//! which implements paradigm of ECS and provides developers with
//! appropriate toolkit to develop PBR games with advanced technologies
//!
//! # Simple example
//!
//! ```rust
//! use despero::prelude::*;
//! 
//! fn main(){
//! 	let mut despero = Despero::init(WindowBuilder::new().with_title("The Game"));
//! 
//! 	despero
//! 		.add_setup_system(s1)
//! 		.add_system(s2)
//! 		.run()
//! }
//! 
//! fn s1(){
//! 	Debug::info("I run only once!");
//! }
//! 
//! fn s2(){
//! 	Debug::info("I run in loop!");
//! } 
//! ```
//! 

// 
//   _____                             _____   _                                    _ 
//  / ____|                           |_   _| | |                                  | |
// | (___   ___  _ __  _   _  __ _      | |   | | _____   _____   _   _  ___  _   _| |
//  \___ \ / _ \| '_ \| | | |/ _` |     | |   | |/ _ \ \ / / _ \ | | | |/ _ \| | | | |
//  ____) | (_) | | | | |_| | (_| |_   _| |_  | | (_) \ V /  __/ | |_| | (_) | |_| |_|
// |_____/ \___/|_| |_|\__, |\__,_( ) |_____| |_|\___/ \_/ \___|  \__, |\___/ \__,_(_)
//                      __/ |     |/                               __/ |
//                     |___/                                      |___/	
// 
//

use std::sync::Arc;
use hecs::*;
use hecs_schedule::*;
use winit::{
	event::{
		Event,
		WindowEvent,
		KeyboardInput,
	},
	platform::run_return::EventLoopExtRunReturn,
	window::WindowBuilder,
};

use crate::ecs::{
	systems::*,
	event::*,
};

use crate::render::renderer::Renderer;

/// Contains submodules and structures to work with graphics
pub mod render;
/// Contains ECS implementations
pub mod ecs;
/// Contains [Rapier3D](https://crates.io/crates/rapier3d) implementations
pub mod physics;
/// Contains [mlua](https://crates.io/crates/mlua) scripting implementations
pub mod scripting;
/// Bundle of all essential components of the engine
pub mod prelude;

/// Main engine struct
pub struct Despero {
	world: World,
	systems: ScheduleBuilder,
	setup_systems: ScheduleBuilder,
	event_writer: EventWriter,
	
	renderer: Renderer,
}

impl Despero {	
	/// Initialize Despero application
	pub fn init(window_builder: WindowBuilder) -> Despero {
		let renderer = Renderer::init(window_builder).expect("Cannot create renderer");
		Despero {
			world: World::new(),
			setup_systems: Schedule::builder(),
			systems: Schedule::builder(),
			event_writer: EventWriter::new(),
			renderer,
		}
	}
	
	/// Add cyclical system to schedule
	pub fn add_system<Args, Ret, S>(mut self, system: S) -> Self 
	where
        S: 'static + System<Args, Ret> + Send,
	{
		self.systems.add_system(system);
		self
	}
	
	/// Add setup system to schedule
	pub fn add_setup_system<Args, Ret, S>(mut self, system: S) -> Self 
	where
        S: 'static + System<Args, Ret> + Send,
	{
		self.setup_systems.add_system(system);
		self
	}
	
	pub fn add_event_reader(&mut self) -> EventReader {
		EventReader::new(&mut self.event_writer)
	}
	
	/// Run main event loop
	pub fn run(mut self) {
		// Init setup-systems Schedule
		let mut setup_systems = self.setup_systems
				//
					.build();
		// Init systems Schedule
		let mut systems = self.systems
			.add_system(update_models_system)
			.add_system(rendering_system)
			.add_system(update_lights)
			.build();
		// Execute setup-systems Schedule
		setup_systems
			.execute((&mut self.world, &mut self.renderer, &mut self.event_writer))
			.expect("Cannot execute setup schedule");
		// Extract `EventLoop` from `Renderer`
		let mut eventloop = self.renderer.window.get_event_loop();
		// Run EventLoop
		eventloop.run_return(move |event, _, controlflow| match event {	
			Event::WindowEvent {
				event: WindowEvent::CloseRequested,
				..
			} => {
				*controlflow = winit::event_loop::ControlFlow::Exit;
			}
					
			Event::MainEventsCleared => {
				self.renderer.window.request_redraw();
			}
			
			Event::RedrawRequested(_) => {
				// Execute loop schedule	
				systems
					.execute((&mut self.world, &mut self.renderer, &mut self.event_writer))
					.expect("Cannot execute loop schedule");
			}
			
			Event::WindowEvent {
				event: WindowEvent::KeyboardInput {input, ..},
				..
			} => {
				self.event_writer.send::<KeyboardInput>(
					Arc::new(input)
				).expect("Event send error");
			}
			
			_ => {}
		});
	}
}

impl Drop for Despero {
	fn drop(&mut self) {
		self.renderer.cleanup(&mut self.world);
	}
}
