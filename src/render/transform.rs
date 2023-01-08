use nalgebra::*;

pub struct Transform {
	pub translation: Vector3<f32>,
	pub rotation: UnitQuaternion<f32>,
	pub scale: f32,
}

impl Default for Transform {
	fn default() -> Self{
		Transform {
			translation: Vector3::new(0.0, 0.0, 0.0),
			rotation: UnitQuaternion::identity(),
			scale: 1.0,
		}
	}
}

impl Transform {
	pub fn to_matrices(&self) -> (Matrix4<f32>, Matrix4<f32>) {
		let new_matrix = Matrix4::new_translation(&self.translation)
			* Matrix4::from(self.rotation)
			* Matrix4::from([
				[self.scale, 0.0, 0.0, 0.0],
				[0.0, self.scale, 0.0, 0.0],
				[0.0, 0.0, self.scale, 0.0],
				[0.0, 0.0, 0.0, 1.0]
			]);
		
		(new_matrix, new_matrix.try_inverse().unwrap())
	}
}
