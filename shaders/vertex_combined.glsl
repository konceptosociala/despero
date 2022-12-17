#version 450

// VERTEX
layout (location=0) in vec3 position;
layout (location=1) in vec3 normal;
layout (location=2) in vec2 texcoord;
// INSTANCE
layout (location=3) in mat4 model_matrix;
layout (location=7) in mat4 inverse_model_matrix;
layout (location=11) in uint texture_id;
layout (location=12) in float metallic_in;
layout (location=13) in float roughness_in;

layout (set=0, binding=0) uniform UniformBufferObject {
	mat4 view_matrix;
	mat4 projection_matrix;
} ubo;

layout (location=0) out vec2 uv;
layout (location=1) out uint tex_id;
layout (location=2) out vec3 out_normal;
layout (location=3) out vec3 out_worldpos;
layout (location=4) out vec3 out_camera_coordinates;
layout (location=5) out float out_metallic;
layout (location=6) out float out_roughness;

void main() {
    vec4 worldpos = model_matrix * vec4(position, 1.0);
    gl_Position = ubo.projection_matrix * ubo.view_matrix * worldpos;
    uv = texcoord;//0
    tex_id = texture_id;//1
    out_normal = transpose(mat3(inverse_model_matrix)) * normal;//2
    out_worldpos = vec3(worldpos);//3
    out_camera_coordinates =//4
		- ubo.view_matrix[3][0] * vec3 (ubo.view_matrix[0][0],ubo.view_matrix[1][0],ubo.view_matrix[2][0])
		- ubo.view_matrix[3][1] * vec3 (ubo.view_matrix[0][1],ubo.view_matrix[1][1],ubo.view_matrix[2][1])
		- ubo.view_matrix[3][2] * vec3 (ubo.view_matrix[0][2],ubo.view_matrix[1][2],ubo.view_matrix[2][2]);
	out_metallic = metallic_in;//5
	out_roughness = roughness_in;//6
}
