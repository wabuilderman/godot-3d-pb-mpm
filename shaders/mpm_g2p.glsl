#version 450

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

//const vec3 GRID_DIM = vec3(64.0, 64.0, 64.0);
//const float DX = 0.5;
//const float DT = 0.002;

struct MaterialProps {
    float E;           // Young's Modulus
    float nu;          // Poisson's Ratio
    float friction;    // Friction angle in degrees (for sand/soil)
    float cohesion;    // Cohesion force
    float density;
    uint model_type;   // 0 = Elastic, 1 = Drucker-Prager, 2 = Fluid
};

struct SolvedGridNode { vec4 vel_mass; };


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

layout(std430, set = 0, binding = 1) buffer PosMassBuffer   { vec4 pos_mass[]; };
layout(std430, set = 0, binding = 2) buffer VelVolBuffer    { vec4 vel_vol[]; };
layout(std430, set = 0, binding = 3) buffer AffineBuffer    { mat4 C[]; };
layout(std430, set = 0, binding = 4) buffer DeformationBuffer    { mat4 D[]; };
layout(std430, set = 0, binding = 5) buffer ElasticFBuffer  { mat4 F_elastic[]; };
layout(std430, set = 0, binding = 6) buffer StressBuffer    { mat4 stress[]; };
layout(std430, set = 0, binding = 7) buffer SolvedGridBuffer{ SolvedGridNode solved_grid[]; };
layout(std430, set = 0, binding = 8) buffer MatIDsBuffer    { uint material_id[]; };
layout(std430, set = 0, binding = 9) buffer MaterialBuffer  { MaterialProps materials[]; };
layout(rgba32f, set = 0, binding = 10) writeonly uniform image2D mpm_particle_positions;
layout(rgba32f, set = 0, binding = 11) writeonly uniform image2D mpm_particle_velocities;


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

// -----------------------------------------------------------------------------
// Helper: 3x3 Jacobi Eigensolver for Symmetric Matrices (B = F * F^T)
// -----------------------------------------------------------------------------
void eigen_decomp_3x3(mat3 A, out vec3 eigenvalues, out mat3 eigenvectors) {
    eigenvectors = mat3(1.0);
    mat3 V = eigenvectors;
    mat3 D = A;

    for (int iter = 0; iter < 5; iter++) {
        int p = 0, q = 1;
        float max_val = abs(D[0][1]);
        if (abs(D[0][2]) > max_val) { p = 0; q = 2; max_val = abs(D[0][2]); }
        if (abs(D[1][2]) > max_val) { p = 1; q = 2; max_val = abs(D[1][2]); }

        if (max_val < 1e-6) break;

        float app = D[p][p];
        float aqq = D[q][q];
        float apq = D[p][q];

        float phi = 0.5 * atan(2.0 * apq, aqq - app);
        float c = cos(phi);
        float s = sin(phi);

        mat3 J = mat3(1.0);
        J[p][p] = c;  J[q][q] = c;
        J[p][q] = s;  J[q][p] = -s;

        D = transpose(J) * D * J;
        V = V * J;
    }

    eigenvalues = vec3(D[0][0], D[1][1], D[2][2]);
    eigenvectors = V;
}

bool finiteMatrix(mat3 M) {
    for (int c = 0; c < 3; ++c) {
        for (int r = 0; r < 3; ++r) {
            if (isnan(M[c][r]) || isinf(M[c][r])) return false;
        }
    }
    return true;
}

float frobeniusNorm(mat3 M) {
    float sum = 0.0;
    for (int c = 0; c < 3; ++c) sum += dot(M[c], M[c]);
    return sqrt(sum);
}

bool usableDeformation(mat3 F) {
    if (!finiteMatrix(F)) return false;

    float J = determinant(F);
    if (isnan(J) || isinf(J) || J < 1e-6) return false;

    mat3 F_inv = inverse(F);
    if (!finiteMatrix(F_inv)) return false;

    // Match the PB-MPM reference's protection against badly conditioned F.
    float conditionEstimate = frobeniusNorm(F) * frobeniusNorm(F_inv);
    return !isnan(conditionEstimate) && !isinf(conditionEstimate)
        && conditionEstimate <= 1e6;
}

// -----------------------------------------------------------------------------
// Main Shader Loop
// -----------------------------------------------------------------------------
void main() {
    
    uint p = gl_GlobalInvocationID.x;
    if (p >= pos_mass.length()) return;

    vec3 x = pos_mass[p].xyz;
    vec3 new_v = vec3(0.0);
    mat3 new_C = mat3(0.0);

    ivec3 base_node;
    vec3 w[3];
    get_weights(x, base_node, w);

    for (int gx = 0; gx < 3; gx++) {
        for (int gy = 0; gy < 3; gy++) {
            for (int gz = 0; gz < 3; gz++) {
                ivec3 node_coord = base_node + ivec3(gx, gy, gz);
                if (any(lessThan(node_coord, ivec3(0))) || any(greaterThanEqual(node_coord, ivec3(params.grid_dim)))) continue;

                float weight = w[gx].x * w[gy].y * w[gz].z;
                
                // Calculate position relative to chunk's world origin
                vec3 node_world_pos = params.world_offset + vec3(node_coord) * params.dx;
                vec3 dpos = node_world_pos - x;
                
                uint node_idx = get_grid_index(node_coord);

                vec3 grid_vel = solved_grid[node_idx].vel_mass.xyz;

                new_v += weight * grid_vel;
                new_C += 4.0 * (params.inv_dx * params.inv_dx) * weight * outerProduct(grid_vel, dpos);
            }
        }
    }

    // Each PB pass refines the candidate velocity/deformation. Commit physical
    // position and constitutive state only after the final constraint iteration.
    bool commit_state = (uint(params.padding) & 2u) != 0u;
    if (commit_state) {
        x += new_v * params.dt;
        pos_mass[p].xyz = x;
    }
    vel_vol[p].xyz = new_v;
    C[p] = mat4(new_C);
    D[p] = mat4(mat3(1.0) + params.dt * new_C);

    if (!commit_state) return;

    uint mat_id = material_id[p];
    MaterialProps mat = materials[mat_id];

    // Store updated positions to particle texture
    const uint TEXTURE_WIDTH = 2048u;
    ivec2 particle_texel = ivec2(int(p % TEXTURE_WIDTH), int(p / TEXTURE_WIDTH));
    imageStore(
        mpm_particle_positions,
        particle_texel,
        vec4(x, float(mat_id))
    );
    imageStore(
        mpm_particle_velocities,
        particle_texel,
        vec4(new_v, length(new_v))
    );

    // Trial deformation gradient: reject singular or badly conditioned updates
    // before any constitutive model can invert or decompose them.
    mat3 F_old = mat3(F_elastic[p]);
    mat3 F_trial = (mat3(1.0) + params.dt * new_C) * F_old;
    if (!usableDeformation(F_trial)) {
        F_trial = usableDeformation(F_old) ? F_old : mat3(1.0);
    }

    // Compute Lamé Parameters
    float mu = mat.E / (2.0 * (1.0 + mat.nu));
    float lambda = (mat.E * mat.nu) / ((1.0 + mat.nu) * (1.0 - 2.0 * mat.nu));

    mat3 sigma = mat3(0.0);

    // =========================================================================
    // MODEL 0: Neo-Hookean Elasticity (Elastic Solids)
    // =========================================================================
    if (mat.model_type == 0u) {
        float J = determinant(F_trial);
        mat3 F_nT = transpose(inverse(F_trial));
        mat3 S = mu * (F_trial - F_nT) + lambda * log(J) * F_nT;
        sigma = (1.0 / J) * S * transpose(F_trial);

        F_elastic[p] = mat4(F_trial);
    } 
    // =========================================================================
    // MODEL 1: Drucker-Prager Plasticity (Sand / Soil / Clay)
    // =========================================================================
    else if (mat.model_type == 1u) {
        // 1. Compute Left Cauchy-Green strain tensor B = F * F^T
        mat3 B = F_trial * transpose(F_trial);
        vec3 evalues;
        mat3 U;
        eigen_decomp_3x3(B, evalues, U);

        // Principal stretches & Hencky strains
        vec3 s_trial = sqrt(max(evalues, vec3(1e-12)));
        vec3 epsilon = log(s_trial); // Hencky strain

        float eps_v = epsilon.x + epsilon.y + epsilon.z; // Volumetric strain
        vec3 e_dev = epsilon - vec3(eps_v / 3.0);         // Deviatoric strain
        float norm_e = length(e_dev);

        // Friction angle & Drucker-Prager yield parameter alpha
        float phi_rad = radians(mat.friction);
        float sin_phi = sin(phi_rad);
        float alpha = sqrt(2.0 / 3.0) * (2.0 * sin_phi / (3.0 - sin_phi));

        // Convert strain magnitudes to Kirchhoff Stress invariants:
        // s_dev_norm = ||tau_dev|| = 2 * mu * norm_e
        // p = - (lambda + 2/3 * mu) * eps_v  (Compressive pressure > 0)
        float mu_2 = 2.0 * mu;
        float bulk_k = lambda + (2.0 / 3.0) * mu;
        
        float s_dev_norm = mu_2 * norm_e;
        float p_val = -bulk_k * eps_v; // Positive in compression
        

        float cohesion = mat.cohesion; // Cohesion in Pascals (e.g., 30000.0)

        vec3 epsilon_new = epsilon;

        if (cohesion > 0.0) {
            // --- COHESIVE MATERIAL (Clay / Mud) ---
            // Yield condition: ||tau_dev|| + alpha * (-p) > cohesion
            // Yields when shear stress exceeds friction-driven pressure support + cohesion
            float yield_val = s_dev_norm - alpha * p_val - cohesion;

            if (yield_val > 0.0) {
                // Yielded: Project deviatoric strain back to the Drucker-Prager cone
                float s_dev_projected = max(0.0, alpha * p_val + cohesion);
                float norm_e_projected = s_dev_projected / max(mu_2, 1e-9);

                vec3 e_new = norm_e_projected * (e_dev / (norm_e + 1e-9));
                
                // Handle tensile apex cap if pulled beyond cohesive tensile strength
                float max_tension = cohesion / max(alpha, 1e-5);
                if (-p_val > max_tension) {
                    // Pure tensile failure (cracking/tearing)
                    eps_v = -max_tension / bulk_k;
                    e_new = vec3(0.0);
                }

                epsilon_new = e_new + vec3(eps_v / 3.0);
            }
        } else {
            // --- NON-COHESIVE MATERIAL (Dry Sand) ---
            if (eps_v > 0.0) {
                // Tensile separation -> complete return to 0 stress
                epsilon_new = vec3(0.0);
            } else {
                float yield_val = s_dev_norm - alpha * p_val;
                if (yield_val > 0.0) {
                    // Return to cone
                    float s_dev_projected = max(0.0, alpha * p_val);
                    float norm_e_projected = s_dev_projected / max(mu_2, 1e-9);
                    vec3 e_new = norm_e_projected * (e_dev / (norm_e + 1e-9));
                    epsilon_new = e_new + vec3(eps_v / 3.0);
                }
            }
        }

        // 2. Reconstruct Elastic Deformation Gradient (F_elastic)
        vec3 s_new = exp(epsilon_new);
        vec3 scale_ratio = s_new / max(s_trial, vec3(1e-9));
        mat3 S_mat = U * mat3(scale_ratio.x, 0.0, 0.0,
                            0.0, scale_ratio.y, 0.0,
                            0.0, 0.0, scale_ratio.z) * transpose(U);
        mat3 F_new = S_mat * F_trial;
        
        // Save the return-mapped state only if reconstruction stayed valid.
        F_elastic[p] = mat4(usableDeformation(F_new) ? F_new : F_trial);

        // 3. Compute Cauchy Stress
        float J_new = s_new.x * s_new.y * s_new.z;
        float eps_v_new = log(J_new);
        vec3 tau = mu_2 * (epsilon_new - vec3(eps_v_new / 3.0)) + vec3(bulk_k * eps_v_new);
        
        sigma = (1.0 / max(J_new, 0.01)) * (U * mat3(tau.x, 0.0, 0.0,
                                                    0.0, tau.y, 0.0,
                                                    0.0, 0.0, tau.z) * transpose(U));
    }
    // =========================================================================
    // MODEL 2: Weakly Compressible Fluid (Tait Equation of State)
    // =========================================================================
    else if (mat.model_type == 2u) {
        float J = determinant(F_trial);

        // 1. Reset F_elastic to isotropic volume expansion so fluid doesn't store shear deformation
        float J_clamped = max(J, 0.2);
        float c_vol = pow(J_clamped, 1.0 / 3.0);
        F_elastic[p] = mat4(c_vol, 0.0, 0.0, 0.0,
                            0.0, c_vol, 0.0, 0.0,
                            0.0, 0.0, c_vol, 0.0,
                            0.0, 0.0, 0.0, 1.0);

        // 2. Bulk Modulus K
        float K = lambda + (2.0 / 3.0) * mu; 
        float gamma = 7.0; // Standard Tait exponent for fluids

        // 3. Correct Tait Equation of State: P = (K / gamma) * (J^-gamma - 1)
        float pressure = (K / gamma) * (pow(max(1.0 / J_clamped, 0.001), gamma) - 1.0);
        pressure = max(pressure, 0.0); // Clamped to avoid negative pressure (fluid tearing)

        // 4. Newtonian Viscosity driven by mat.NU (Dynamic Viscosity), NOT Lamé shear mu!
        mat3 D_rate = 0.5 * (new_C + transpose(new_C));
        float trace_D = D_rate[0][0] + D_rate[1][1] + D_rate[2][2];
        
        float dynamic_viscosity = mat.nu; // 0.001 = Water, 0.05 = Oil
        mat3 viscous_stress = 2.0 * dynamic_viscosity * (D_rate - (1.0 / 3.0) * trace_D * mat3(1.0));

        // 5. Total Cauchy Stress
        sigma = -pressure * mat3(1.0) + viscous_stress;
    }

    stress[p] = mat4(sigma);
}
