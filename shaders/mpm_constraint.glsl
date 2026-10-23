#version 450

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform SimulationParams {
    uvec3 grid_dim;
    float dx;
    vec3 world_offset;
    float dt;
    uint total_particles;
    float inv_dx;
    float padding;
} params;

layout(std430, set = 0, binding = 1)
buffer DeformationBuffer {
    mat4 D[];
};

layout(std430, set = 0, binding = 2)
buffer ElasticFBuffer {
    mat4 F[];
};

struct MaterialProps {
    float E;
    float nu;
    float friction;
    float cohesion;
    float density;
    uint model_type;
};

layout(std430, set = 0, binding = 3) readonly buffer MatIDsBuffer {
    uint material_id[];
};

layout(std430, set = 0, binding = 4) readonly buffer MaterialBuffer {
    MaterialProps materials[];
};

bool finiteMatrix(mat3 M)
{
    for (int c = 0; c < 3; ++c) {
        for (int r = 0; r < 3; ++r) {
            if (isnan(M[c][r]) || isinf(M[c][r])) return false;
        }
    }
    return true;
}

bool polarRotation(mat3 Fstar, out mat3 R)
{
    float detF = determinant(Fstar);
    if (!finiteMatrix(Fstar) || isnan(detF) || isinf(detF) || abs(detF) < 1e-6) return false;

    R = Fstar;

    for (int i = 0; i < 5; ++i)
    {
        float detR = determinant(R);
        if (!finiteMatrix(R) || isnan(detR) || isinf(detR) || abs(detR) < 1e-6) return false;
        mat3 R_inv_transpose = transpose(inverse(R));
        R = 0.5 * (R + R_inv_transpose);
    }
    return finiteMatrix(R);
}

void main()
{
   uint p = gl_GlobalInvocationID.x;

    if (p >= params.total_particles)
        return;

    mat3 Fp = mat3(F[p]);
    mat3 Dp = mat3(D[p]);
    float detFp = determinant(Fp);
    if (!finiteMatrix(Fp) || !finiteMatrix(Dp) || isnan(detFp) || isinf(detFp) || detFp < 1e-4)
        return;

    // G2P advances deformation on the left: F_next = D * F.
    // Build and constrain the candidate using that same multiplication order.
    mat3 Fstar = Dp * Fp;
    float J = determinant(Fstar);
    if (!finiteMatrix(Fstar) || isnan(J) || isinf(J) || J < 1e-4)
        return;

    uint matId = material_id[p];
    uint model = materials[matId].model_type;
    mat3 target;

    if (model == 2u) {
        // Fluid constraint: correct volume while leaving the candidate shape
        // unconstrained. This is a local deformation-volume proxy.
        target = Fstar * pow(J, -1.0 / 3.0);
    } else {
        mat3 R;
        if (!polarRotation(Fstar, R)) return;

        // Co-rotational shape plus unit-volume target, as in PB-MPM.
        mat3 volumeTarget = Fstar / J;
        const float SHAPE_WEIGHT = 0.5;
        target = SHAPE_WEIGHT * R + (1.0 - SHAPE_WEIGHT) * volumeTarget;
    }

    mat3 D_new = target * inverse(Fp);
    if (finiteMatrix(D_new)) D[p] = mat4(D_new);
}
