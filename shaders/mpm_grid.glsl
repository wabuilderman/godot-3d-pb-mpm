#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

const vec3 GRAVITY = vec3(0.0, -9.81, 0.0);
const float FIXED_SCALE = 1e6;

struct GridNode {
    int mass_int;
    int mom_x_int;
    int mom_y_int;
    int mom_z_int;
};

struct SolvedGridNode {
    vec4 vel_mass; // xyz: velocity, w: mass
};


// Uniform block updated before every compute dispatch
layout(set = 0, binding = 0) uniform SimulationParams {
    uvec3 grid_dim;      // e.g., (64, 64, 64) or (128, 128, 128)
    float dx;            // Cell size for this chunk (e.g., 0.5 for coarse, 0.25 for fine)
    vec3 world_offset;   // World-space origin (x, y, z) of this chunk's (0,0,0) node
    float dt;            // Time step
    uint total_particles;// Dynamic particle count  [Unused]
    float inv_dx;        // Pre-calculated 1.0 / dx (saves GPU divisions) [prior FIXED_SCALE]
    float padding;       // Maintains 16-byte alignment for std140 layout
    vec4 impulse_position_radius;
    vec4 impulse_direction_strength;
} params;

/*
struct SimParams {
    uvec3 grid_dim;      // e.g., (64, 64, 64) or (128, 128, 128)
    float dx;            // Cell size for this chunk (e.g., 0.5 for coarse, 0.25 for fine)
    vec3 world_offset;   // World-space origin (x, y, z) of this chunk's (0,0,0) node
    float dt;            // Time step
    uint total_particles;// Dynamic particle count  [Unused]
    float inv_dx;        // Pre-calculated 1.0 / dx (saves GPU divisions) [prior FIXED_SCALE]
    float padding;       // Maintains 16-byte alignment for std140 layout
};

const SimParams params = SimParams(
    uvec3(64,64,64), 0.5, vec3(0.0, 0.0, 0.0), 0.002, 200000, 2, 0.0
);*/

layout(std430, set = 0, binding = 1) buffer GridBuffer       { GridNode grid[]; };
layout(std430, set = 0, binding = 2) buffer SolvedGridBuffer { SolvedGridNode solved_grid[]; };


void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= grid.length()) return;

    float mass = float(grid[i].mass_int) / FIXED_SCALE;

    if (mass > 1e-5) {
        vec3 mom = vec3(
            float(grid[i].mom_x_int),
            float(grid[i].mom_y_int),
            float(grid[i].mom_z_int)
        ) / FIXED_SCALE;

        vec3 vel = mom / mass;
        
        // Corrected block in mpm_grid.glsl
        uint grid_xy = uint(params.grid_dim.x * params.grid_dim.y);
        ivec3 coord = ivec3(
            int(i % uint(params.grid_dim.x)),
            int((i % grid_xy) / uint(params.grid_dim.x)),
            int(i / grid_xy)
        );

        if ((uint(params.padding) & 1u) != 0u) {
            vel += GRAVITY * params.dt;
        }

        float impulse_radius = params.impulse_position_radius.w;
        if (impulse_radius > 0.0 && params.impulse_direction_strength.w != 0.0) {
            vec3 node_position = params.world_offset + vec3(coord) * params.dx;
            vec3 impulse_offset = node_position - params.impulse_position_radius.xyz;
            float distance_to_impulse = length(impulse_offset);
            if (distance_to_impulse < impulse_radius) {
                float falloff = 1.0 - distance_to_impulse / impulse_radius;
                vel += params.impulse_direction_strength.xyz
                    * (params.impulse_direction_strength.w * falloff);
            }
        }

        // Apply ground collision ONLY at the bottom boundary (coord.y == 0 or world y <= 0)
        if (coord.y == 0 && vel.y < 0.0) {
            vel.y = 0.0;
        }

        solved_grid[i].vel_mass = vec4(vel, mass);
    } else {
        solved_grid[i].vel_mass = vec4(0.0);
    }
}
