#version 450

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

//const vec3 GRID_DIM = vec3(64.0, 64.0, 64.0);
//const float DX = 0.5;
//const float FIXED_SCALE = 1e4;


// Uniform block updated before every compute dispatch
layout(set = 0, binding = 0) uniform SimulationParams {
    uvec3 grid_dim;      // e.g., (64, 64, 64) or (128, 128, 128)
    float dx;            // Cell size for this chunk (e.g., 0.5 for coarse, 0.25 for fine)
    vec3 world_offset;   // World-space origin (x, y, z) of this chunk's (0,0,0) node
    float dt;            // Time step
    uint total_particles;// Dynamic particle count
    float inv_dx;        // Pre-calculated 1.0 / dx (saves GPU divisions) [prior FIXED_SCALE]
    float padding;       // Maintains 16-byte alignment for std140 layout
} params;

/*
struct SimParams {
    uvec3 grid_dim;      // e.g., (64, 64, 64) or (128, 128, 128)
    float dx;            // Cell size for this chunk (e.g., 0.5 for coarse, 0.25 for fine)
    vec3 world_offset;   // World-space origin (x, y, z) of this chunk's (0,0,0) node
    float dt;            // Time step
    uint total_particles;// Dynamic particle count [Unused]
    float inv_dx;        // Pre-calculated 1.0 / dx (saves GPU divisions) [prior FIXED_SCALE]
    float padding;       // Maintains 16-byte alignment for std140 layout
};

const SimParams params = SimParams(
    uvec3(64,64,64), 0.5, vec3(0.0, 0.0, 0.0), 0.002, 200000, 2, 0.0
);*/


layout(std430, set = 0, binding = 1) buffer PosMassBuffer { vec4 pos_mass[]; };
layout(std430, set = 0, binding = 2) buffer DensityBuffer { int density_int[]; };

uint get_grid_index(ivec3 coord) {
    return uint(coord.x + coord.y * int(params.grid_dim.x) + coord.z * int(params.grid_dim.x * params.grid_dim.y));
}

void main() {
    uint p = gl_GlobalInvocationID.x;
    if (p >= pos_mass.length()) return;

    vec3 x = pos_mass[p].xyz;
    ivec3 base_cell = ivec3(floor(x / params.dx));

    for (int dx = 0; dx <= 1; dx++) {
        for (int dy = 0; dy <= 1; dy++) {
            for (int dz = 0; dz <= 1; dz++) {
                ivec3 coord = base_cell + ivec3(dx, dy, dz);
                if (all(greaterThanEqual(coord, ivec3(0))) && all(lessThan(coord, ivec3(params.grid_dim)))) {
                    vec3 cell_center = (vec3(coord) + 0.5) * params.dx;
                    float dist = length(x - cell_center);
                    float weight = max(0.0, 1.0 - dist / (params.dx * 1.5));
                    
                    uint idx = get_grid_index(coord);
                    atomicAdd(density_int[idx], int(weight * params.inv_dx));
                }
            }
        }
    }
}