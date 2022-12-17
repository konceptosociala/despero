use std::collections::HashMap;
use std::mem::size_of;
use gpu_allocator::vulkan::*;
use gpu_allocator::MemoryLocation;
use ash::vk;
use nalgebra as na;

type Handle = usize;

use crate::render::buffer::Buffer;
use crate::engine::texture::{
	TexturedVertexData,
	TexturedInstanceData,
};

// InvalidHandle custom error
#[derive(Debug, Clone)]
pub struct InvalidHandle;

impl std::fmt::Display for InvalidHandle {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		write!(f, "Invalid handle")
	}
}

impl std::error::Error for InvalidHandle {
	fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
		None
	}
}

// InstanceData
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct InstanceData {
	pub modelmatrix: [[f32; 4]; 4],
	pub inverse_modelmatrix: [[f32; 4]; 4],
	pub colour: [f32; 3],
	pub metallic: f32,
	pub roughness: f32,
}

impl InstanceData {
	pub fn new(
		modelmatrix: na::Matrix4<f32>,
		colour: [f32; 3],
		metallic: f32,
		roughness: f32,
	) -> InstanceData {
		InstanceData {
			modelmatrix: modelmatrix.into(),
			inverse_modelmatrix: modelmatrix.try_inverse().unwrap().into(),
			colour,
			metallic,
			roughness,
		}
	}
}

// VertexData
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct VertexData {
	pub position: [f32; 3],
	pub normal: [f32; 3],
}

impl VertexData {
	fn midpoint(a: &VertexData, b: &VertexData) -> VertexData {
		VertexData {
			position: [
				0.5 * (a.position[0] + b.position[0]),
				0.5 * (a.position[1] + b.position[1]),
				0.5 * (a.position[2] + b.position[2]),
			],
			normal: Self::normalize([
				0.5 * (a.normal[0] + b.normal[0]),
				0.5 * (a.normal[1] + b.normal[1]),
				0.5 * (a.normal[2] + b.normal[2]),
			]),
		}
	}
	
	fn normalize(v: [f32; 3]) -> [f32; 3] {
		let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
		[v[0] / l, v[1] / l, v[2] / l]
	}
}

// Model struct
pub struct Model<V, I> {
	// Model vertices
	pub vertexdata: Vec<V>,
	// Vertex indices
	pub indexdata: Vec<u32>,
	// Handle to index of the model instance
	pub handle_to_index: HashMap<usize, Handle>,
	// Vec of the handles
	pub handles: Vec<usize>,
	// Vec of the instances
	pub instances: Vec<I>,
	// Index of first invisible instance
	pub first_invisible: usize,
	// Next handle to use
	pub next_handle: usize,
	// Buffers
	pub vertexbuffer: Option<Buffer>,
	pub instancebuffer: Option<Buffer>,
	pub indexbuffer: Option<Buffer>,
}

impl<V, I: std::fmt::Debug> Model<V, I> {
	pub fn get(&self, handle: Handle) -> Option<&I> {
		if let Some(&index) = self.handle_to_index.get(&handle) {
			self.instances.get(index)
		} else {
			None
		}
	}
	
	pub fn get_mut(&mut self, handle: Handle) -> Option<&mut I> {
		if let Some(&index) = self.handle_to_index.get(&handle) {
			self.instances.get_mut(index)
		} else {
			None
		}
	}
	// Swap instances by handles
	pub fn swap_by_handle(&mut self, handle1: Handle, handle2: Handle) -> Result<(), InvalidHandle> {
		if handle1 == handle2 {
			return Ok(());
		}
		// Get indices of the handles
		if let (Some(&index1), Some(&index2)) = (
			self.handle_to_index.get(&handle1),
			self.handle_to_index.get(&handle2),
		) {
			self.handles.swap(index1, index2);
			self.instances.swap(index1, index2);
			self.handle_to_index.insert(index1, handle2);
			self.handle_to_index.insert(index2, handle1);
			Ok(())
		} else {
			Err(InvalidHandle)
		}
	}
	//Swap instances by index
	pub fn swap_by_index(&mut self, index1: usize, index2: usize) {
		if index1 == index2 {
			return;
		}
		let handle1 = self.handles[index1];
		let handle2 = self.handles[index2];
		self.handles.swap(index1, index2);
		self.instances.swap(index1, index2);
		self.handle_to_index.insert(index1, handle2);
		self.handle_to_index.insert(index2, handle1);
	}
	
	pub fn is_visible(&self, handle: Handle) -> Result<bool, InvalidHandle> {
		if let Some(index) = self.handle_to_index.get(&handle) {
			Ok(index < &self.first_invisible)
		} else {
			Err(InvalidHandle)
		}
	}
	
	pub fn make_visible(&mut self, handle: Handle) -> Result<(), InvalidHandle> {
		// Check if invisible
		if let Some(&index) = self.handle_to_index.get(&handle) {
			if index < self.first_invisible {
				return Ok(());
			}
			// Move to position `first_invisible` and increase value of `first_invisible`
			self.swap_by_index(index, self.first_invisible);
			self.first_invisible += 1;
			Ok(())
		} else {
			Err(InvalidHandle)
		}
	}
	
	pub fn make_invisible(&mut self, handle: Handle) -> Result<(), InvalidHandle> {
		// Check if visible
		if let Some(&index) = self.handle_to_index.get(&handle) {
			if index >= self.first_invisible {
				return Ok(());
			}
			// Move to position before `first_invisible` and decrease value of `first_invisible`
			self.swap_by_index(index, self.first_invisible - 1);
			self.first_invisible -= 1;
			Ok(())
		} else {
			Err(InvalidHandle)
		}
	}
	
	pub fn insert(&mut self, element: I) -> Handle {
		// Make new handle
		let handle = self.next_handle;
		self.next_handle += 1;
		// Put index at the end
		let index = self.instances.len();
		self.instances.push(element);
		self.handles.push(handle);
		// Link handle and index
		self.handle_to_index.insert(handle, index);
		return handle;
	}
	
	pub fn insert_visibly(&mut self, element: I) -> Handle {
		let handle = self.insert(element);
		self.make_visible(handle).ok();
		return handle;
	}
	
	// Remove handle and get the element
	pub fn remove(&mut self, handle: Handle) -> Result<I, InvalidHandle> {
		// Get index of the handle
		if let Some(&index) = self.handle_to_index.get(&handle) {
			if index < self.first_invisible {
				self.swap_by_index(index, self.first_invisible - 1);
				self.first_invisible -= 1;
			}
			self.swap_by_index(self.first_invisible, self.instances.len() - 1);
			self.handles.pop();
			self.handle_to_index.remove(&handle);
			Ok(self.instances.pop().unwrap())
		} else {
			Err(InvalidHandle)
		}
	}
	
	// Update VertexBuffer
	pub fn update_vertexbuffer(
		&mut self,
		logical_device: &ash::Device,
		allocator: &mut Allocator,
	) -> Result<(), vk::Result> {
		// Check whether the buffer exists
		if let Some(buffer) = &mut self.vertexbuffer {
			buffer.fill(
				logical_device,
				allocator,
				&self.vertexdata
			)?;
			Ok(())
		} else {
			// Set buffer size
			let bytes = (self.vertexdata.len() * size_of::<V>()) as u64;		
			let mut buffer = Buffer::new(
				&logical_device,
				allocator,
				bytes,
				vk::BufferUsageFlags::VERTEX_BUFFER,
				MemoryLocation::CpuToGpu,
				"Model vertex buffer"
			)?;
			
			buffer.fill(
				&logical_device,
				allocator,
				&self.vertexdata
			)?;
			self.vertexbuffer = Some(buffer);
			Ok(())
		}
	}
	
	// Update InstanceBuffer
	pub fn update_instancebuffer(
		&mut self,
		logical_device: &ash::Device,
		allocator: &mut Allocator,
	) -> Result<(), vk::Result> {
		if let Some(buffer) = &mut self.instancebuffer {
			buffer.fill(
				logical_device,
				allocator,
				&self.instances[0..self.first_invisible]
			)?;
			Ok(())
		} else {
			let bytes = (self.first_invisible * size_of::<I>()) as u64; 
			let mut buffer = Buffer::new(
				&logical_device,
				allocator,
				bytes,
				vk::BufferUsageFlags::VERTEX_BUFFER,
				MemoryLocation::CpuToGpu,
				"Model instance buffer"
			)?;
			
			buffer.fill(
				logical_device,
				allocator,
				&self.instances[0..self.first_invisible]
			)?;
			self.instancebuffer = Some(buffer);
			Ok(())
		}
	}

	// Update IndexBuffer
	pub fn update_indexbuffer(
		&mut self,
		logical_device: &ash::Device,
		allocator: &mut Allocator,
	) -> Result<(), vk::Result> {
		// Check whether the buffer exists
		if let Some(buffer) = &mut self.indexbuffer {
			buffer.fill(
				logical_device,
				allocator,
				&self.indexdata,
			)?;
			Ok(())
		} else {
			// Set buffer size
			let bytes = (self.indexdata.len() * size_of::<u32>()) as u64;		
			let mut buffer = Buffer::new(
				&logical_device,
				allocator,
				bytes,
				vk::BufferUsageFlags::INDEX_BUFFER,
				MemoryLocation::CpuToGpu,
				"Model buffer of vertex indices"
			)?;
			
			buffer.fill(
				&logical_device,
				allocator,
				&self.indexdata
			)?;
			self.indexbuffer = Some(buffer);
			Ok(())
		}
	}

	pub fn draw(
		&self, 
		logical_device: &ash::Device, 
		commandbuffer: vk::CommandBuffer,
	){
		if let Some(vertexbuffer) = &self.vertexbuffer {
			if let Some(instancebuffer) = &self.instancebuffer {
				if let Some(indexbuffer) = &self.indexbuffer {
					if self.first_invisible > 0 {
						unsafe {
							// Bind position buffer						
							logical_device.cmd_bind_index_buffer(
								commandbuffer,
								indexbuffer.buffer,
								0,
								vk::IndexType::UINT32,
							);
							
							logical_device.cmd_bind_vertex_buffers(
								commandbuffer,
								0,
								&[vertexbuffer.buffer],
								&[0],
							);
							
							logical_device.cmd_bind_vertex_buffers(
								commandbuffer,
								1,
								&[instancebuffer.buffer],
								&[0],
							);
							
							logical_device.cmd_draw_indexed(
								commandbuffer,
								self.indexdata.len() as u32,
								self.first_invisible as u32,
								0,
								0,
								0,
							);
						}
					}
				}
			}
		}
	}
}


//Implement cubic model
impl Model<VertexData, InstanceData> {
	pub fn cube() -> Model<VertexData, InstanceData> {
		let lbf = [-1.0,1.0,0.0]; // Left-bottom-front
		let lbb = [-1.0,1.0,2.0]; // Left-bottom-back
		let ltf = [-1.0,-1.0,0.0];// Left-top-front
		let ltb = [-1.0,-1.0,2.0];// Left-top-back
		let rbf = [1.0,1.0,0.0];  // Right-bottom-front
		let rbb = [1.0,1.0,2.0];  // Right-bottom-back
		let rtf = [1.0,-1.0,0.0]; // Right-top-front
		let rtb = [1.0,-1.0,2.0]; // Right-top-back

		Model {
			vertexdata: vec![
				VertexData {
					position: lbf,
					normal: VertexData::normalize(lbf),
				},
				VertexData {
					position: lbb,
					normal: VertexData::normalize(lbb),
				},
				VertexData {
					position: ltf,
					normal: VertexData::normalize(ltf),
				},
				VertexData {
					position: ltb,
					normal: VertexData::normalize(ltb),
				},
				VertexData {
					position: rbf,
					normal: VertexData::normalize(rbf),
				},
				VertexData {
					position: rbb,
					normal: VertexData::normalize(rbb),
				},
				VertexData {
					position: rtf,
					normal: VertexData::normalize(rtf),
				},
				VertexData {
					position: rtb,
					normal: VertexData::normalize(rtb),
				},
			],
			indexdata: vec![
				0, 1, 5, 0, 5, 4, //bottom
				2, 7, 3, 2, 6, 7, //top
				0, 6, 2, 0, 4, 6, //front
				1, 3, 7, 1, 7, 5, //back
				0, 2, 1, 1, 2, 3, //left
				4, 5, 6, 5, 7, 6, //right
			],
			handle_to_index: HashMap::new(),
			handles: Vec::new(),
			instances: Vec::new(),
			first_invisible: 0,
			next_handle: 0,
			vertexbuffer: None,
			instancebuffer: None,
			indexbuffer: None,
		}
	}
	
	pub fn icosahedron() -> Model<VertexData, InstanceData> {
		let phi = (1.0 + 5.0_f32.sqrt()) / 2.0;
		Model {
			vertexdata: vec![
				VertexData {
					position: [phi, -1.0, 0.0],
					normal: VertexData::normalize([phi, -1.0, 0.0]),
				},
				VertexData {
					position: [phi, 1.0, 0.0],
					normal: VertexData::normalize([phi, 1.0, 0.0]),
				},
				VertexData {
					position: [-phi, -1.0, 0.0],
					normal: VertexData::normalize([-phi, -1.0, 0.0]),
				},
				VertexData {
					position: [-phi, 1.0, 0.0],
					normal: VertexData::normalize([-phi, 1.0, 0.0]),
				},
				VertexData {
					position: [1.0, 0.0, -phi],
					normal: VertexData::normalize([1.0, 0.0, -phi]),
				},
				VertexData {
					position: [-1.0, 0.0, -phi],
					normal: VertexData::normalize([-1.0, 0.0, -phi]),
				},
				VertexData {
					position: [1.0, 0.0, phi],
					normal: VertexData::normalize([1.0, 0.0, phi]),
				},
				VertexData {
					position: [-1.0, 0.0, phi],
					normal: VertexData::normalize([-1.0, 0.0, phi]),
				},
				VertexData {
					position: [0.0, -phi, -1.0],
					normal: VertexData::normalize([0.0, -phi, -1.0]),
				},
				VertexData {
					position: [0.0, -phi, 1.0],
					normal: VertexData::normalize([0.0, -phi, 1.0]),
				},
				VertexData {
					position: [0.0, phi, -1.0],
					normal: VertexData::normalize([0.0, phi, -1.0]),
				},
				VertexData {
					position: [0.0, phi, 1.0],
					normal: VertexData::normalize([0.0, phi, 1.0]),
				},
			],
			indexdata: vec![
				0,9,8,//
				0,8,4,//
				0,4,1,//
				0,1,6,//
				0,6,9,//
				8,9,2,//
				8,2,5,//
				8,5,4,//
				4,5,10,//
				4,10,1,//
				1,10,11,//
				1,11,6,//
				2,3,5,//
				2,7,3,//
				2,9,7,//
				5,3,10,//
				3,11,10,//
				3,7,11,//
				6,7,9,//
				6,11,7//
			],

			handle_to_index: HashMap::new(),
			handles: Vec::new(),
			instances: Vec::new(),
			first_invisible: 0,
			next_handle: 0,
			vertexbuffer: None,
			instancebuffer: None,
			indexbuffer: None,
		}
	}
	
	pub fn sphere(refinements: u32) -> Model<VertexData, InstanceData> {
		// New icosahedron
		let mut model = Model::icosahedron();
		// Subdivide faces
		for _ in 0..refinements{
			model.refine();
		}
		// Align vertices to equal distance to sphere's center
		for v in &mut model.vertexdata {
			v.position = VertexData::normalize(v.position);
		}
		return model;
	}
	
	// Triangle subdividing
	pub fn refine(&mut self) {
		let mut new_indices = vec![];
		let mut midpoints = std::collections::HashMap::<(u32, u32), u32>::new();
		for triangle in self.indexdata.chunks(3) {
			let a = triangle[0];
			let b = triangle[1];
			let c = triangle[2];
			let vertex_a = self.vertexdata[a as usize];
			let vertex_b = self.vertexdata[b as usize];
			let vertex_c = self.vertexdata[c as usize];
			let mab = if let Some(ab) = midpoints.get(&(a, b)) {
				*ab
			} else {
				let vertex_ab = VertexData::midpoint(&vertex_a, &vertex_b);
				let mab = self.vertexdata.len() as u32;
				self.vertexdata.push(vertex_ab);
				midpoints.insert((a, b), mab);
				midpoints.insert((b, a), mab);
				mab
			};
			let mbc = if let Some(bc) = midpoints.get(&(b, c)) {
				*bc
			} else {
				let vertex_bc = VertexData::midpoint(&vertex_b, &vertex_c);
				let mbc = self.vertexdata.len() as u32;
				midpoints.insert((b, c), mbc);
				midpoints.insert((c, b), mbc);
				self.vertexdata.push(vertex_bc);
				mbc
			};
			let mca = if let Some(ca) = midpoints.get(&(c, a)) {
				*ca
			} else {
				let vertex_ca = VertexData::midpoint(&vertex_c, &vertex_a);
				let mca = self.vertexdata.len() as u32;
				midpoints.insert((c, a), mca);
				midpoints.insert((a, c), mca);
				self.vertexdata.push(vertex_ca);
				mca
			};
			new_indices.extend_from_slice(&[mca, a, mab, mab, b, mbc, mbc, c, mca, mab, mbc, mca]);
		}
		self.indexdata = new_indices;
	}
}

impl Model<TexturedVertexData, TexturedInstanceData> {
	pub fn quad() -> Self {
		let lb = TexturedVertexData {
			position: [-1.0, 1.0, 0.0],
			normal: VertexData::normalize([-1.0, 1.0, 0.0]),
			texcoord: [0.0, 1.0],
		};
		let lt = TexturedVertexData {
			position: [-1.0, -1.0, 0.0],
			normal: VertexData::normalize([-1.0, -1.0, 0.0]),
			texcoord: [0.0, 0.0],
		};
		let rb = TexturedVertexData {
			position: [1.0, 1.0, 0.0],
			normal: VertexData::normalize([1.0, 1.0, 0.0]),
			texcoord: [1.0, 1.0],
		};
		let rt = TexturedVertexData {
			position: [1.0, -1.0, 0.0],
			normal: VertexData::normalize([1.0, -1.0, 0.0]),
			texcoord: [1.0, 0.0],
		};
		
		Model {
			vertexdata: vec![lb, lt, rb, rt],
			indexdata: vec![0, 2, 1, 1, 2, 3],
			handle_to_index: std::collections::HashMap::new(),
			handles: Vec::new(),
			instances: Vec::new(),
			first_invisible: 0,
			next_handle: 0,
			vertexbuffer: None,
			indexbuffer: None,
			instancebuffer: None,
		}
	}
}
