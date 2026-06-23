#include "secp256k1.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

#define GPU_N_STEPS     2048
#define GPU_DP_BUF_SIZE 131072

__device__ ECPointJacobian d_stepPoints[GPU_N_STEPS];
__device__ uint64_t        d_stepScalarsA[GPU_N_STEPS * 4];
__device__ uint64_t        d_stepScalarsB[GPU_N_STEPS * 4];
__device__ ECPointJacobian d_G_OFFSET;
__device__ ECPointJacobian d_target_jac;
__device__ uint64_t        d_max_scalar[4];
__device__ int             d_dp_bits;
__device__ int             d_window_size;
__device__ uint32_t        d_found;
__device__ uint32_t        d_dp_count;

struct GPUDPEntry {
    uint64_t x[4], a[4], b[4];
    uint32_t valid;
};

__device__ GPUDPEntry d_dp_buffer[GPU_DP_BUF_SIZE];

__device__ __forceinline__ uint32_t gpu_step_idx(const uint64_t* x, uint32_t n) {
    uint64_t h = x[0] ^ (x[1] * 0xff51afd7ed558ccdULL);
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return (uint32_t)(h % n);
}

__device__ __forceinline__ bool gpu_is_dp(const uint64_t* x, int dp_bits) {
    if (dp_bits >= 64) return x[0] == 0;
    return (x[0] & ((1ULL << dp_bits) - 1)) == 0;
}

__device__ __forceinline__ bool gpu_exceeds_max(const uint64_t* a, const uint64_t* m) {
    for (int i = 3; i >= 0; i--) {
        if (a[i] > m[i]) return true;
        if (a[i] < m[i]) return false;
    }
    return true;
}

__global__ void rho_walk_gpu(
    ECPointJacobian* walkers_R,
    uint64_t*        walkers_a,
    uint64_t*        walkers_b,
    uint64_t*        snapshot_x,
    uint64_t*        snapshot_steps,
    uint32_t*        walker_type,
    uint32_t         num_walkers,
    uint32_t         steps_per_launch
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_walkers || d_found) return;

    ECPointJacobian R = walkers_R[tid];
    uint64_t a[4], b[4], snap_x[4];
    for (int i = 0; i < 4; i++) {
        a[i]      = walkers_a[tid * 4 + i];
        b[i]      = walkers_b[tid * 4 + i];
        snap_x[i] = snapshot_x[tid * 4 + i];
    }
    uint64_t snap_steps = snapshot_steps[tid];
    int dp_bits = d_dp_bits;

    for (uint32_t step = 0; step < steps_per_launch; step++) {
        if (d_found) break;

        // Converte para affine para obter x real
        ECPointAffine aff;
        jacobianToAffine(&aff, &R);

        // Deteccao de ciclo
        if (aff.x[0] == snap_x[0] && aff.x[1] == snap_x[1] &&
            aff.x[2] == snap_x[2] && aff.x[3] == snap_x[3]) {
            uint32_t ni = (uint32_t)((aff.x[0] ^ (uint64_t)tid * 2654435761ULL) % GPU_N_STEPS);
            R = d_stepPoints[ni];
            for (int i = 0; i < 4; i++) {
                a[i] = d_stepScalarsA[ni * 4 + i];
                b[i] = 0;
            }
            for (int i = 0; i < 4; i++) b[i] = 0; if (walker_type[tid] == 1) { b[0] = 1; pointAddJacobian(&R, &R, &d_target_jac); }
            snap_steps = 0;
            for (int i = 0; i < 4; i++) snap_x[i] = 0xFFFFFFFFFFFFFFFFULL;
            continue;
        }

        // Step via x affine
        uint32_t idx = gpu_step_idx(aff.x, GPU_N_STEPS);
        pointAddJacobian(&R, &R, &d_stepPoints[idx]);
        scalarAdd(a, a, &d_stepScalarsA[idx * 4]);
        scalarAdd(b, b, &d_stepScalarsB[idx * 4]);

        // Wrap
        if (gpu_exceeds_max(a, d_max_scalar)) {
            uint64_t diff[4];
            scalarSub(diff, a, d_max_scalar);
            for (int i = 0; i < 4; i++) a[i] = diff[i];
            pointAddJacobian(&R, &R, &d_G_OFFSET);
        }

        // Snapshot
        snap_steps++;
        if ((snap_steps & (snap_steps - 1)) == 0)
            for (int i = 0; i < 4; i++) snap_x[i] = aff.x[i];

        // DP check
        if (!gpu_is_dp(aff.x, dp_bits)) continue;

        // Salva DP — só se buffer tiver espaco
        uint32_t pos = atomicAdd(&d_dp_count, 1);
        if (pos < GPU_DP_BUF_SIZE) {
            for (int i = 0; i < 4; i++) {
                d_dp_buffer[pos].x[i] = aff.x[i];
                d_dp_buffer[pos].a[i] = a[i];
                d_dp_buffer[pos].b[i] = b[i];
            }
            d_dp_buffer[pos].valid = 1;
        }

        // Reinicia walker apos DP
        snap_steps = 0;
        for (int i = 0; i < 4; i++) snap_x[i] = 0xFFFFFFFFFFFFFFFFULL;
        uint32_t ni = (uint32_t)((aff.x[0] ^ (uint64_t)tid * 2654435761ULL) % GPU_N_STEPS);
        R = d_stepPoints[ni];
        for (int i = 0; i < 4; i++) {
            a[i] = d_stepScalarsA[ni * 4 + i];
            b[i] = 0;
        }
        for (int i = 0; i < 4; i++) b[i] = 0; if (walker_type[tid] == 1) { b[0] = 1; pointAddJacobian(&R, &R, &d_target_jac); }
    }

    walkers_R[tid] = R;
    for (int i = 0; i < 4; i++) {
        walkers_a[tid * 4 + i]  = a[i];
        walkers_b[tid * 4 + i]  = b[i];
        snapshot_x[tid * 4 + i] = snap_x[i];
    }
    snapshot_steps[tid] = snap_steps;
}

extern "C" {

void gpu_upload_steps(const ECPointJacobian* s, const uint64_t* sa, const uint64_t* sb, int n) {
    cudaMemcpyToSymbol(d_stepPoints,   s,  n * sizeof(ECPointJacobian));
    cudaMemcpyToSymbol(d_stepScalarsA, sa, n * 4 * sizeof(uint64_t));
    cudaMemcpyToSymbol(d_stepScalarsB, sb, n * 4 * sizeof(uint64_t));
}

void gpu_upload_params(const ECPointJacobian* go, const ECPointJacobian* tj,
                       const uint64_t* ms, int dp, int ws) {
    cudaMemcpyToSymbol(d_G_OFFSET,    go,  sizeof(ECPointJacobian));
    cudaMemcpyToSymbol(d_target_jac,  tj,  sizeof(ECPointJacobian));
    cudaMemcpyToSymbol(d_max_scalar,  ms,  4 * sizeof(uint64_t));
    cudaMemcpyToSymbol(d_dp_bits,     &dp, sizeof(int));
    cudaMemcpyToSymbol(d_window_size, &ws, sizeof(int));
    uint32_t z = 0;
    cudaMemcpyToSymbol(d_found,    &z, sizeof(uint32_t));
    cudaMemcpyToSymbol(d_dp_count, &z, sizeof(uint32_t));
}

void gpu_reset_dp_buffer() {
    uint32_t z = 0;
    cudaMemcpyToSymbol(d_dp_count, &z, sizeof(uint32_t));
    GPUDPEntry* p = nullptr;
    cudaGetSymbolAddress((void**)&p, d_dp_buffer);
    cudaMemset(p, 0, GPU_DP_BUF_SIZE * sizeof(GPUDPEntry));
}

void gpu_signal_found() {
    uint32_t o = 1;
    cudaMemcpyToSymbol(d_found, &o, sizeof(uint32_t));
}

uint32_t gpu_get_dp_count() {
    uint32_t c = 0;
    cudaMemcpyFromSymbol(&c, d_dp_count, sizeof(uint32_t));
    return c;
}

void gpu_fetch_dp_buffer(GPUDPEntry* h, uint32_t n) {
    GPUDPEntry* p = nullptr;
    cudaGetSymbolAddress((void**)&p, d_dp_buffer);
    cudaMemcpy(h, p, n * sizeof(GPUDPEntry), cudaMemcpyDeviceToHost);
}

void gpu_launch_walk(ECPointJacobian* R, uint64_t* a, uint64_t* b,
                     uint64_t* sx, uint64_t* ss, uint32_t* ty,
                     uint32_t nw, uint32_t spl, cudaStream_t st) {
    int th = 256, bl = (nw + th - 1) / th;
    rho_walk_gpu<<<bl, th, 0, st>>>(R, a, b, sx, ss, ty, nw, spl);
}

} // extern "C"
