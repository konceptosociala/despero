#version 450
#extension GL_EXT_nonuniform_qualifier : enable

layout (location=0) out vec4 theColour;

layout (location=0) in vec2 uv;
layout (location=1) flat in uint texture_id;
layout (location=2) in vec3 normal;
layout (location=3) in vec3 worldpos;
layout (location=4) in vec3 camera_coordinates;
layout (location=5) in float metallic;
layout (location=6) in float roughness;

layout (set=1, binding=0) uniform sampler2D texturesamplers[];

readonly layout (set=2, binding=0) buffer StorageBufferObject {
	float num_directional;
	float num_point;
	vec3 data[];
} sbo;

struct DirectionalLight{
	vec3 direction_to_light;
	vec3 irradiance;
};

struct PointLight{
	vec3 position;
	vec3 luminous_flux;
};

const float PI = 3.1415926535897932;

float distribution(vec3 normal,vec3 halfvector,float roughness){
	float NdotH=dot(halfvector,normal);
	if (NdotH>0){
		float r=roughness*roughness;
		return r / (PI* (1 + NdotH*NdotH*(r-1))*(1 + NdotH*NdotH*(r-1)));
	}else{
		return 0.0;
	}
}

float geometry(vec3 light, vec3 normal, vec3 view, float roughness){
	float NdotL=abs(dot(normal,light));
	float NdotV=abs(dot(normal,view));
	return 0.5/max(0.01,mix(2*NdotL*NdotV,NdotL+NdotV,roughness));
}

vec3 compute_radiance(vec3 irradiance, vec3 light_direction, vec3 normal, vec3 camera_direction, vec3 surface_colour){
	float NdotL=	max(dot(normal,light_direction),0);
	
	vec3 irradiance_on_surface=irradiance*NdotL;

	float roughness2 = roughness*roughness;

	vec3 F0 = mix(vec3(0.03),surface_colour,vec3(metallic));
	vec3 reflected_irradiance = (F0 + (1 - F0)*(1-NdotL)*(1-NdotL)*(1-NdotL)*(1-NdotL)*(1-NdotL)) * irradiance_on_surface;
	vec3 refracted_irradiance = irradiance_on_surface - reflected_irradiance; 
	vec3 refracted_not_absorbed_irradiance = refracted_irradiance * (1-metallic);

	vec3 halfvector=normalize(0.5*(camera_direction + light_direction));
	float NdotH=max(dot(normal,halfvector),0);
	vec3 F=(F0 + (1 - F0)*(1-NdotH)*(1-NdotH)*(1-NdotH)*(1-NdotH)*(1-NdotH));
	vec3 relevant_reflection = reflected_irradiance*F*geometry(light_direction,normal,camera_direction,roughness2)*distribution(normal,halfvector,roughness2);

	return refracted_not_absorbed_irradiance*surface_colour/PI + relevant_reflection;
}

void main(){
	vec3 L = vec3(0);
	vec3 direction_to_camera = normalize(camera_coordinates - worldpos);
	vec3 normal = normalize(normal);
	
	int number_directional = int(sbo.num_directional);
	int number_point = int(sbo.num_point);

	// Directional lights
	for (int i = 0; i < number_directional; i++){
		vec3 direction = sbo.data[2 * i];
		vec3 illuminance = sbo.data[2 * i + 1];
		DirectionalLight dlight = DirectionalLight(normalize(direction), illuminance);

		L += compute_radiance(dlight.irradiance, dlight.direction_to_light, normal, direction_to_camera, vec3(texture(texturesamplers[texture_id], uv)));
	}

	// Point lights
	for (int i = 0; i < number_point; i++){
		vec3 position = sbo.data[2 * i + 2 * number_point];
		vec3 luminous_flux = sbo.data[2 * i + 1 + 2 * number_point];
		PointLight light = PointLight(position, luminous_flux);
		
		vec3 direction_to_light = normalize(light.position - worldpos);
		float d = length(worldpos - light.position);
		vec3 irradiance = light.luminous_flux/(4*PI*d*d);

		L += compute_radiance(irradiance, direction_to_light, normal, direction_to_camera, vec3(texture(texturesamplers[texture_id], uv)));
	}

	theColour=vec4(L/(1+L),1.0);
}
