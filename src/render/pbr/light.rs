use nalgebra as na;

#[derive(Debug, Clone)]
pub struct DirectionalLight {
	pub direction: na::Vector3<f32>,
	pub illuminance: [f32; 3],
}

#[derive(Debug, Clone)]
pub struct PointLight {
	pub position: na::Point3<f32>,
	pub luminous_flux: [f32; 3],
}
