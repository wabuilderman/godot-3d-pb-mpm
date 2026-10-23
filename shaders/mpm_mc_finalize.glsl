#version 450

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

// Must match OUTPUT_VERTEX_CAPACITY from your Marching Cubes shader
const uint OUTPUT_VERTEX_CAPACITY = 500000u;

layout(set = 0, binding = 0, std430) restrict readonly buffer CounterBuffer
{
    uint count;
} counter_buffer;

layout(r32f, set = 0, binding = 1) uniform writeonly image2D count_image;

void main()
{
    // Clamp to capacity before converting to float
    uint clamped_count = min(counter_buffer.count, OUTPUT_VERTEX_CAPACITY);

    imageStore(
        count_image,
        ivec2(0, 0),
        vec4(float(clamped_count), 0.0, 0.0, 1.0)
    );
}