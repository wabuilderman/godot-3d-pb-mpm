shader_type spatial;

render_mode cull_disabled;
//render_mode blend_mix, depth_draw_opaque;

uniform sampler2D mpm_position_data : filter_nearest, repeat_disable;
uniform sampler2D mpm_normal_data : filter_nearest, repeat_disable;
uniform sampler2D mpm_vertex_count : filter_nearest, repeat_disable;

// Enable or disable wireframe view at compile time / development
const bool WIREFRAME_ENABLED = false;

// Wireframe styling
const float wireframe_thickness = 0.3;
const float reference_distance = 10.0; // Distance where lines are 100% full thickness
const vec3 wireframe_color = vec3(0.0, 1.0, 0.0); // Bright Green
const vec3 surface_color = vec3(0.1, 0.1, 0.1);    // Dark Gray
varying vec3 v_barycentric;

// Helper to fetch a vertex position by linear index
vec3 get_pos(int idx, ivec2 tex_size) {
    int x = idx % tex_size.x;
    int y = idx / tex_size.x;
    return texelFetch(mpm_position_data, ivec2(x, y), 0).xyz;
}

vec3 get_norm(int idx, ivec2 tex_size) {
    int x = idx % tex_size.x;
    int y = idx / tex_size.x;
    return texelFetch(mpm_normal_data, ivec2(x, y), 0).xyz;
}

void vertex()
{
    int vertex_id = VERTEX_ID;
	
	// DEBUG: HARDCODE COUNT DIRECTLY IN THE SPATIAL SHADER
    //int active_vertex_count = 3000;
	
    // Fetch active vertex count from 1x1 R32Sfloat texture
    float active_count_raw = texelFetch(mpm_vertex_count, ivec2(0, 0), 0).r;
    int active_vertex_count = int(active_count_raw);

    // If this vertex index exceeds active generated geometry, collapse it.
    // The GPU hardware rasterizer will cull degenerate 0-area triangles automatically.
    if (vertex_id >= active_vertex_count) {
        VERTEX = vec3(0.0);
    } else {
		ivec2 tex_size = textureSize(mpm_position_data, 0);

		vec3 p = get_pos(vertex_id, tex_size);
		VERTEX = p;
		NORMAL = get_norm(vertex_id, tex_size);
	}
	
	// Assign barycentric coordinates based on 3-vertex triangle tuple
	if (WIREFRAME_ENABLED) {
		int local_id = vertex_id % 3;
		if (local_id == 0) {
			v_barycentric = vec3(1.0, 0.0, 0.0);
		} else if (local_id == 1) {
			v_barycentric = vec3(0.0, 1.0, 0.0);
		} else {
			v_barycentric = vec3(0.0, 0.0, 1.0);
		}
	}
}

void fragment()
{
	
	if (WIREFRAME_ENABLED) {
		// 1. Get distance from camera in View Space (-VIEW_VERTEX.z)
        float depth = max(-VERTEX.z, 0.001);

        // 2. Attenuate thickness by distance
        // As depth increases beyond reference_distance, thickness drops below 1.0 pixel
        float effective_thickness = wireframe_thickness * (reference_distance / depth);

        // 3. Barycentric edge calculations
        vec3 unit_derivatives = fwidth(v_barycentric);
        vec3 edge_distance = v_barycentric / unit_derivatives;
        float min_edge_dist = min(min(edge_distance.x, edge_distance.y), edge_distance.z);

        // 4. Anti-aliased line thresholding
        // When effective_thickness drops < 0.5, the line smoothly fades into the surface
        float edge_factor = smoothstep(effective_thickness - 0.5, effective_thickness + 0.5, min_edge_dist);

        ALBEDO = mix(wireframe_color, surface_color, edge_factor);
        ROUGHNESS = 0.5;
		if (edge_factor > 0.5) {
			discard;
		}
	} else {
		// Default solid shading
		ALBEDO = vec3(0.7, 0.7, 0.7);
		//ALPHA = 0.5;                  // 50% transparent
	}
}