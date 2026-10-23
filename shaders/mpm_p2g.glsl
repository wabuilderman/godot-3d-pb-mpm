#version 450
layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

// Fixed-point scaling factor for float-to-int atomic addition
//const float FIXED_SCALE = 1e6;
//const vec3 GRID_DIM = vec3(64.0, 64.0, 64.0);
//const float DX = 0.5; // Voxel size = 0.5 meters
//const float DT = 0.002;

const float FIXED_SCALE = 1e6;

struct GridNode {
    int mass_int;
    int mom_x_int;
    int mom_y_int;
    int mom_z_int;
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



layout(std430, set = 0, binding = 1) buffer PosMassBuffer { vec4 pos_mass[]; };
layout(std430, set = 0, binding = 2) buffer VelVolBuffer  { vec4 vel_vol[]; };
layout(std430, set = 0, binding = 3) buffer AffineBuffer  { mat4 C[]; }; // 4x4 matrix for std430 alignment
layout(std430, set = 0, binding = 4) buffer StressBuffer  { mat4 stress[]; };
layout(std430, set = 0, binding = 5) buffer GridBuffer    { GridNode grid[]; };
layout(std430, set = 0, binding = 6) buffer DeformationBuffer { mat4 D[]; };

uint get_grid_index(ivec3 coord) {
    return uint(coord.x + coord.y * int(params.grid_dim.x) + coord.z * int(params.grid_dim.x * params.grid_dim.y));
}

void get_weights(vec3 pos, out ivec3 base_node, out vec3 w[3]) {
    // 1. Transform world-space position into local chunk space
    vec3 local_pos = pos - params.world_offset;
    
    // 2. Scale by local cell size
    vec3 cell_pos = local_pos * (1.0 / params.dx);
    base_node = ivec3(floor(cell_pos - 0.5));
    vec3 fx = cell_pos - vec3(base_node);

    w[0] = 0.5 * (1.5 - fx) * (1.5 - fx);
    w[1] = 0.75 - (fx - 1.0) * (fx - 1.0);
    w[2] = 0.5 * (fx - 0.5) * (fx - 0.5);
}

void main() {
    uint p = gl_GlobalInvocationID.x;
    if (p >= pos_mass.length()) return;

    vec3 x = pos_mass[p].xyz;
    float mass = pos_mass[p].w;
    vec3 v = vel_vol[p].xyz;
    float vol = vel_vol[p].w;

    mat3 D_p = mat3(D[p]);
    mat3 C_pb = (D_p - mat3(1.0)) / max(params.dt, 1e-8);

    mat3 C_p = mat3(C[p]);
    mat3 stress_p = mat3(stress[p]);

    ivec3 base_node;
    vec3 w[3];
    get_weights(x, base_node, w);

    // MLS-MPM affine momentum + stress force transfer
    mat3 stress_force = -4.0 * vol * (1.0 / (params.dx * params.dx)) * stress_p;

    for (int gx = 0; gx < 3; gx++) {
        for (int gy = 0; gy < 3; gy++) {
            for (int gz = 0; gz < 3; gz++) {
                ivec3 node_coord = base_node + ivec3(gx, gy, gz);
                if (any(lessThan(node_coord, ivec3(0))) || any(greaterThanEqual(node_coord, ivec3(params.grid_dim)))) continue;

                float weight = w[gx].x * w[gy].y * w[gz].z;
                vec3 dpos = vec3(node_coord) * params.dx - x;
                vec3 momentum = mass * (v + C_pb * dpos);
                //vec3 momentum = mass * (v + C_p * dpos) + params.dt * stress_force * dpos;
                //vec3 momentum = mass * (v + C_p * dpos) + stress_force * dpos;
                float node_mass = mass * weight;

                uint node_idx = get_grid_index(node_coord);

                atomicAdd(grid[node_idx].mass_int, int(node_mass * FIXED_SCALE));
                atomicAdd(grid[node_idx].mom_x_int, int(momentum.x * weight * FIXED_SCALE));
                atomicAdd(grid[node_idx].mom_y_int, int(momentum.y * weight * FIXED_SCALE));
                atomicAdd(grid[node_idx].mom_z_int, int(momentum.z * weight * FIXED_SCALE));
            }
        }
    }
}