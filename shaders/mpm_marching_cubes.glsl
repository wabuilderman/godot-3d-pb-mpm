#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

// -----------------------------------------------------------------------------
// Simulation / reconstruction constants
// -----------------------------------------------------------------------------

const float ISO_VALUE = 0.35;


// -----------------------------------------------------------------------------
// GPU mesh output configuration
//
// IMPORTANT:
// These two values must match the values used when creating the RD textures
// on the C# side.
//
// Example:
//     RenderVertexCapacity = 500000
//     MeshDataTextureWidth = 2048
//
// Height is then ceil(500000 / 2048) = 245.
//
// Every vertex occupies exactly one texel.
// -----------------------------------------------------------------------------

const uint OUTPUT_VERTEX_CAPACITY = 4000000u;
const uint OUTPUT_TEXTURE_WIDTH  = 2048u;


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
); */

layout(std430, set = 0, binding = 1) readonly buffer DensityBuffer { int density_int[]; };
layout(std430, set = 0, binding = 2) readonly buffer TriTableBuffer { int tri_table[]; };
layout(rgba32f, set = 0, binding = 3) writeonly uniform image2D position_image;
layout(rgba16f, set = 0, binding = 4) writeonly uniform image2D normal_image;
layout(std430, set = 0, binding = 5) buffer CounterBuffer { uint vertex_count; };

// -----------------------------------------------------------------------------
// Grid indexing
// -----------------------------------------------------------------------------

uint get_grid_index(ivec3 coord) {
    return uint(coord.x + coord.y * int(params.grid_dim.x) + coord.z * int(params.grid_dim.x * params.grid_dim.y));
}

float get_density(ivec3 coord) {
    if (any(lessThan(coord, ivec3(0))) || any(greaterThanEqual(coord, ivec3(params.grid_dim)))) {
        return 0.0;
    }

    return float(density_int[get_grid_index(coord)]);
}

// -----------------------------------------------------------------------------
// Density-gradient normal
// -----------------------------------------------------------------------------

vec3 get_normal(ivec3 coord)
{
    float nx = get_density(coord + ivec3(1, 0, 0)) - get_density(coord - ivec3(1, 0, 0));
    float ny = get_density(coord + ivec3(0, 1, 0)) - get_density(coord - ivec3(0, 1, 0));
    float nz = get_density(coord + ivec3(0, 0, 1)) - get_density(coord - ivec3(0, 0, 1));

    vec3 grad = vec3(nx, ny, nz);

    return length(grad) > 1e-4 ? normalize(grad) : vec3(0.0, 1.0, 0.0);
}

// -----------------------------------------------------------------------------
// Marching-cubes topology
// -----------------------------------------------------------------------------

const ivec3 corner_offsets[8] = ivec3[](
    ivec3(0,0,0), ivec3(1,0,0), ivec3(1,0,1), ivec3(0,0,1),
    ivec3(0,1,0), ivec3(1,1,0), ivec3(1,1,1), ivec3(0,1,1)
);

const int edge_nodes[24] = int[](
    0,1,  1,2,  2,3,  3,0,
    4,5,  5,6,  6,7,  7,4,
    0,4,  1,5,  2,6,  3,7
);

// -----------------------------------------------------------------------------
// Convert a linear vertex index into an (x,y) texel coordinate.
// -----------------------------------------------------------------------------

ivec2 vertex_texel(uint vertex_index)
{
    uint x = vertex_index % OUTPUT_TEXTURE_WIDTH;
    uint y = vertex_index / OUTPUT_TEXTURE_WIDTH;

    return ivec2(int(x), int(y));
}

// -----------------------------------------------------------------------------
// Write one generated vertex to the GPU mesh textures.
// -----------------------------------------------------------------------------

void write_vertex(uint vertex_index, vec3 position, vec3 normal) {
    ivec2 texel = vertex_texel(vertex_index);
    imageStore(position_image, texel, vec4(position, 1.0));
    imageStore(normal_image, texel, vec4(normal, 0.0));
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

void main()
{
    ivec3 cell = ivec3(gl_GlobalInvocationID.xyz);

    // A cell needs a +1 neighbor in each dimension.
    if (any(greaterThanEqual(cell, ivec3(params.grid_dim) - ivec3(1)))){
        return;
    }

    // -------------------------------------------------------------------------
    // Sample the eight cell corners.
    // -------------------------------------------------------------------------

    float corner_densities[8];
    int cube_index = 0;

    for (int i = 0; i < 8; i++) {
        corner_densities[i] = get_density(cell + corner_offsets[i]);

        if (corner_densities[i] >= ISO_VALUE) {
            cube_index |= (1 << i);
        }
    }

    if (cube_index == 0 || cube_index == 255) {
        return;
    }

    // -------------------------------------------------------------------------
    // Calculate interpolated edge positions and normals.
    // -------------------------------------------------------------------------

    vec3 edge_positions[12];
    vec3 edge_normals[12];

    for (int i = 0; i < 12; i++) {
        int c1 = edge_nodes[i * 2];
        int c2 = edge_nodes[i * 2 + 1];

        ivec3 coord1 = cell + corner_offsets[c1];
        ivec3 coord2 = cell + corner_offsets[c2];

        vec3 p1 = params.world_offset + (vec3(coord1) + 0.5) * params.dx;
        vec3 p2 = params.world_offset + (vec3(coord2) + 0.5) * params.dx;

        float d1 = corner_densities[c1];
        float d2 = corner_densities[c2];

        float t = (abs(d1 - d2) > 1e-5) ? clamp((ISO_VALUE - d1) / (d2 - d1), 0.0, 1.0) : 0.5;

        edge_positions[i] = mix(p1, p2, t);
        edge_normals[i] = normalize(mix(get_normal(coord1), get_normal(coord2), t));
    }

    // -------------------------------------------------------------------------
    // Determine how many triangles this cell emits.
    // -------------------------------------------------------------------------

    int tri_table_base = cube_index * 16;

    int tri_count = 0;

    for (int i = 0; i < 16; i += 3) {
        if (tri_table[tri_table_base + i] == -1) {
            break;
        }

        tri_count++;
    }

    if (tri_count == 0) {
        return;
    }

    uint vertices_to_write = uint(tri_count * 3);

    // -------------------------------------------------------------------------
    // Reserve a contiguous range in the global output array.
    //
    // The counter itself is still GPU-only.
    //
    // Multiple MC invocations can safely reserve different ranges.
    // -------------------------------------------------------------------------

    uint base_vert_idx = atomicAdd(vertex_count, vertices_to_write);

    // -------------------------------------------------------------------------
    // If this allocation extends beyond our render capacity, do not write it.
    //
    // We intentionally leave the counter alone. A later finalize pass will
    // clamp the visible vertex count to OUTPUT_VERTEX_CAPACITY.
    // -------------------------------------------------------------------------

    if (base_vert_idx >= OUTPUT_VERTEX_CAPACITY || base_vert_idx + vertices_to_write > OUTPUT_VERTEX_CAPACITY) {
        return;
    }

    // Emit triangle vertices.
    for (int i = 0; i < 16; i += 3)
    {
        int e0 = tri_table[tri_table_base + i];

        if (e0 == -1) {
            break;
        }

        int e1 = tri_table[tri_table_base + i + 1];
        int e2 = tri_table[tri_table_base + i + 2];
        uint idx = base_vert_idx + uint(i);

        write_vertex(idx + 0u, edge_positions[e0], -edge_normals[e0]);
        write_vertex(idx + 1u, edge_positions[e2], -edge_normals[e2]);
        write_vertex(idx + 2u, edge_positions[e1], -edge_normals[e1]);
    }
}