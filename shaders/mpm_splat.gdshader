shader_type spatial;
render_mode cull_back;

uniform sampler2D mpm_particle_positions : filter_nearest, repeat_disable;
uniform sampler2D mpm_particle_velocities : filter_nearest, repeat_disable;
uniform int particle_count;

// Adjust this scale factor to control overall particle size in your scene
uniform float point_scale : hint_range(1.0, 500.0) = 15.0;
uniform float velocity_color_scale : hint_range(0.1, 20.0) = 5.0;

varying vec3 particle_velocity;

void vertex()
{
    int particle_id = VERTEX_ID; 
    ivec2 tex_size = textureSize(mpm_particle_positions, 0);

    int x = particle_id % tex_size.x;
    int y = particle_id / tex_size.x;
    vec4 texDat = texelFetch(mpm_particle_positions, ivec2(x, y), 0);
    vec4 velocityDat = texelFetch(mpm_particle_velocities, ivec2(x, y), 0);
    vec3 pos = texDat.xyz;
    float matID = texDat.w;
    particle_velocity = velocityDat.xyz;

    VERTEX = pos;

    // 1. Transform particle position into View Space (Camera Space)
    vec4 view_pos = MODELVIEW_MATRIX * vec4(pos, 1.0);

    // 2. In Godot's View Space, -Z points forward out of the camera lens.
    // -view_pos.z gives the actual distance (in world units) from the camera to the particle.
    float camera_depth = -view_pos.z;

    // 3. Calculate perspective size: Point Size = Scale / Distance
    if (camera_depth > 0.01 && particle_id < particle_count) {
        POINT_SIZE = clamp(point_scale / camera_depth, 1.0, 512.0);
    } else {
        POINT_SIZE = 0.0; // Hide particles behind or inside the camera clip plane
    }

    if (matID == 3.0) {
        COLOR = vec4(0.768627451,0.509803922,0.207843137, 1.0);
    } else if (matID == 2.0) {
        COLOR = vec4(1.0, 1.0, 0.5, 1.0);
    } else if (matID == 4.0) {
		POINT_SIZE = POINT_SIZE * (1.1 - (clamp(pos.y, 0, 1.0)));
        COLOR = vec4(0.15, 0.3, 0.7, 1.0) * (clamp(pos.y, 0, 0.5)*2.0);
    } else {
        COLOR = vec4(0.2, 0.6, 1.0, 1.0);
    }
}

void fragment()
{
    vec2 circ = POINT_COORD * 2.0 - 1.0;
    float r2 = dot(circ, circ);

    if (r2 > 1.0) {
        discard;
    }

    // Reconstruct 3D sphere normal in view space
    vec3 sphere_normal;
    sphere_normal.xy = circ;
    sphere_normal.z = sqrt(1.0 - r2);

    NORMAL = sphere_normal;

    float speed_factor = clamp(length(particle_velocity) / velocity_color_scale, 0.0, 1.0);
    vec3 speed_color = mix(vec3(0.1, 0.35, 1.0), vec3(1.0, 0.2, 0.04), speed_factor);
    ALBEDO = mix(COLOR.xyz, speed_color, speed_factor * 0.75);
    ROUGHNESS = 0.2;
}
