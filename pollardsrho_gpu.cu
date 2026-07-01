#include "secp256k1.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

extern __device__ __constant__ uint64_t P_CONST_MINUS_2[4];
extern __device__ __constant__ uint64_t ONE_MONT[4];

#define GPU_N_STEPS     2048
#define GPU_DP_BUF_SIZE 131072
#define BLOCK_SIZE      256
#define INNER_STEPS     16

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

struct GPUDPEntry { uint64_t x[4], a[4], b[4]; uint32_t valid; };
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
    int tid     = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_blk = threadIdx.x;
    bool active = (tid < (int)num_walkers);
    int blk_n   = min((int)blockDim.x,
                      (int)num_walkers - (int)(blockIdx.x * blockDim.x));
    if (blk_n <= 0) blk_n = 1;

    // Shared memory layout:
    // smem_src [BLOCK_SIZE*4] — buffer A do Hillis-Steele
    // smem_dst [BLOCK_SIZE*4] — buffer B do Hillis-Steele
    // smem_Z   [BLOCK_SIZE*4] — Z originais (para back-propagation)
    // smem_inv [BLOCK_SIZE*4] — inversos de Z resultantes
    // smem_x   [BLOCK_SIZE*4] — x_affine resultantes
    extern __shared__ uint64_t smem[];
    uint64_t* smem_src = smem;
    uint64_t* smem_dst = smem + BLOCK_SIZE * 4;
    uint64_t* smem_Z   = smem + BLOCK_SIZE * 8;
    uint64_t* smem_inv = smem + BLOCK_SIZE * 12;
    uint64_t* smem_x   = smem + BLOCK_SIZE * 16;

    ECPointJacobian R;
    uint64_t a[4]={0,0,0,0}, b[4]={0,0,0,0}, snap_x[4]={0,0,0,0};
    uint64_t snap_steps = 0;
    int dp_bits  = d_dp_bits;
    uint32_t wty = 0;

    if (active) {
        R = walkers_R[tid];
        for (int i = 0; i < 4; i++) {
            a[i]      = walkers_a[tid * 4 + i];
            b[i]      = walkers_b[tid * 4 + i];
            snap_x[i] = snapshot_x[tid * 4 + i];
        }
        snap_steps = snapshot_steps[tid];
        wty = walker_type[tid];
    } else {
        R = d_stepPoints[0];
    }

    uint32_t outer_iters = steps_per_launch / INNER_STEPS;
    if (outer_iters == 0) outer_iters = 1;

    for (uint32_t outer = 0; outer < outer_iters; outer++) {

        if (d_found) break;

        // ── Fase 1: INNER_STEPS passos jacobianos ────────────────────────────
        for (int s = 0; s < INNER_STEPS; s++) {
            uint32_t idx = gpu_step_idx(R.X, GPU_N_STEPS);
            pointAddJacobian(&R, &R, &d_stepPoints[idx]);
            scalarAdd(a, a, &d_stepScalarsA[idx * 4]);
            scalarAdd(b, b, &d_stepScalarsB[idx * 4]);
            if (gpu_exceeds_max(a, d_max_scalar)) {
                uint64_t diff[4];
                scalarSub(diff, a, d_max_scalar);
                for (int i = 0; i < 4; i++) a[i] = diff[i];
                pointAddJacobian(&R, &R, &d_G_OFFSET);
            }
            snap_steps++;
        }

        // ── Fase 2: Batch normalize com Hillis-Steele paralelo ───────────────
        //
        // CORREÇÃO DO BUG: threads "fantasma" (tid_blk >= blk_n) usam ONE_MONT
        // como elemento neutro da multiplicação — assim não contaminam o resultado.
        //
        // Hillis-Steele prefix scan:
        //   Após log2(N) passos, src[i] = Z[0] * Z[1] * ... * Z[i]
        //   Cada passo: dst[i] = src[i] * src[i-offset]  (se i >= offset)
        //               dst[i] = src[i]                   (se i < offset)
        //   Threads fantasma: dst[i] = ONE_MONT           (elemento neutro)

        // 2a. Inicializa buffer: walkers ativos usam Z, fantasmas usam ONE_MONT
        if (tid_blk < blk_n) {
            for (int j = 0; j < 4; j++) smem_src[tid_blk*4+j] = R.Z[j];
            for (int j = 0; j < 4; j++) smem_Z[tid_blk*4+j]   = R.Z[j]; // salva Z original
        } else {
            for (int j = 0; j < 4; j++) smem_src[tid_blk*4+j] = ONE_MONT[j];
            for (int j = 0; j < 4; j++) smem_Z[tid_blk*4+j]   = ONE_MONT[j];
        }
        __syncthreads();

        // 2b. Hillis-Steele paralelo — TODAS as threads sempre escrevem em dst
        for (int offset = 1; offset < BLOCK_SIZE; offset *= 2) {
            if (tid_blk >= offset) {
                uint64_t tmp[4];
                modMulMontP(tmp, &smem_src[tid_blk*4], &smem_src[(tid_blk-offset)*4]);
                for (int j = 0; j < 4; j++) smem_dst[tid_blk*4+j] = tmp[j];
            } else {
                for (int j = 0; j < 4; j++) smem_dst[tid_blk*4+j] = smem_src[tid_blk*4+j];
            }
            __syncthreads();
            // Troca src e dst
            uint64_t* tmp_ptr = smem_src; smem_src = smem_dst; smem_dst = tmp_ptr;
        }
        // smem_src[i] = Z[0] * Z[1] * ... * Z[i]  para i < blk_n
        // smem_src[i] = Z[0] * ... * Z[blk_n-1]   para i >= blk_n (neutro não altera)

        // 2c. Thread 0: inversão única do produto total
        if (tid_blk == 0) {
            modExpMontP(&smem_inv[(blk_n-1)*4], &smem_src[(blk_n-1)*4], P_CONST_MINUS_2);
        }
        __syncthreads();

        // 2d. Back-propagation sequencial (thread 0)
        // cur = inv_total = 1/acc[blk_n-1]
        // Para i = blk_n-1 downto 1:
        //   inv_Z[i] = cur * acc[i-1]
        //   cur = cur * Z[i]
        // inv_Z[0] = cur
        if (tid_blk == 0) {
            uint64_t cur[4];
            for (int j = 0; j < 4; j++) cur[j] = smem_inv[(blk_n-1)*4+j];
            for (int i = blk_n - 1; i > 0; i--) {
                uint64_t inv_i[4], new_cur[4];
                modMulMontP(inv_i,   cur, &smem_src[(i-1)*4]);
                modMulMontP(new_cur, cur, &smem_Z[i*4]);
                for (int j = 0; j < 4; j++) { smem_inv[i*4+j] = inv_i[j]; cur[j] = new_cur[j]; }
            }
            for (int j = 0; j < 4; j++) smem_inv[j] = cur[j];
        }
        __syncthreads();

        // 2e. Cada thread calcula x_aff e restaura invariante Z=ONE_MONT
        if (tid_blk < blk_n) {
            uint64_t zInv2[4], x_mont[4], y_mont[4], zInv3[4];
            modMulMontP(zInv2,  &smem_inv[tid_blk*4], &smem_inv[tid_blk*4]);
            modMulMontP(x_mont, R.X, zInv2);
            modMulMontP(zInv3,  zInv2, &smem_inv[tid_blk*4]);
            modMulMontP(y_mont, R.Y, zInv3);
            fromMontgomeryP(&smem_x[tid_blk*4], x_mont);
            ECPointAffine aff;
            for (int j = 0; j < 4; j++) aff.x[j] = smem_x[tid_blk*4+j];
            fromMontgomeryP(aff.y, y_mont);
            aff.infinity = 0;
            affineToJacobian(&R, &aff);
        }
        __syncthreads();

        // ── Fase 3: decisões locais (sem sync dentro) ────────────────────────
        uint64_t* x_aff = &smem_x[tid_blk * 4];

        bool is_cycle = (x_aff[0]==snap_x[0] && x_aff[1]==snap_x[1] &&
                         x_aff[2]==snap_x[2] && x_aff[3]==snap_x[3]);
        bool is_dp_now = active && gpu_is_dp(x_aff, dp_bits) && !is_cycle;
        bool do_restart = active && (is_cycle || is_dp_now);

        if (is_dp_now) {
            uint32_t pos = atomicAdd(&d_dp_count, 1);
            if (pos < GPU_DP_BUF_SIZE) {
                for (int i = 0; i < 4; i++) {
                    d_dp_buffer[pos].x[i] = x_aff[i];
                    d_dp_buffer[pos].a[i] = a[i];
                    d_dp_buffer[pos].b[i] = b[i];
                }
                d_dp_buffer[pos].valid = 1;
            }
        }

        if (active && !do_restart) {
            if ((snap_steps & (snap_steps - 1)) == 0)
                for (int i = 0; i < 4; i++) snap_x[i] = x_aff[i];
        }

        if (do_restart) {
            snap_steps = 0;
            for (int i = 0; i < 4; i++) snap_x[i] = 0xFFFFFFFFFFFFFFFFULL;
            uint32_t ni = (uint32_t)((x_aff[0] ^ (uint64_t)tid * 2654435761ULL) % GPU_N_STEPS);
            R = d_stepPoints[ni];
            for (int i = 0; i < 4; i++) { a[i] = d_stepScalarsA[ni*4+i]; b[i] = 0; }
            if (wty == 1) {
                b[0] = 1;
                pointAddJacobian(&R, &R, &d_target_jac);
                ECPointAffine tmp; jacobianToAffine(&tmp, &R); affineToJacobian(&R, &tmp);
            }
        }

        __syncthreads();
    }

    if (active) {
        walkers_R[tid] = R;
        for (int i = 0; i < 4; i++) {
            walkers_a[tid * 4 + i]  = a[i];
            walkers_b[tid * 4 + i]  = b[i];
            snapshot_x[tid * 4 + i] = snap_x[i];
        }
        snapshot_steps[tid] = snap_steps;
    }
}

extern "C" {

void gpu_upload_steps(const ECPointJacobian* s, const uint64_t* sa, const uint64_t* sb, int n) {
    cudaMemcpyToSymbol(d_stepPoints,   s,  n*sizeof(ECPointJacobian));
    cudaMemcpyToSymbol(d_stepScalarsA, sa, n*4*sizeof(uint64_t));
    cudaMemcpyToSymbol(d_stepScalarsB, sb, n*4*sizeof(uint64_t));
}

void gpu_upload_params(const ECPointJacobian* go, const ECPointJacobian* tj,
                       const uint64_t* ms, int dp, int ws) {
    cudaMemcpyToSymbol(d_G_OFFSET,    go,  sizeof(ECPointJacobian));
    cudaMemcpyToSymbol(d_target_jac,  tj,  sizeof(ECPointJacobian));
    cudaMemcpyToSymbol(d_max_scalar,  ms,  4*sizeof(uint64_t));
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
    cudaMemset(p, 0, GPU_DP_BUF_SIZE*sizeof(GPUDPEntry));
}

void gpu_signal_found() { uint32_t o=1; cudaMemcpyToSymbol(d_found,&o,sizeof(uint32_t)); }

uint32_t gpu_get_dp_count() {
    uint32_t c=0; cudaMemcpyFromSymbol(&c,d_dp_count,sizeof(uint32_t)); return c;
}

void gpu_fetch_dp_buffer(GPUDPEntry* h, uint32_t n) {
    GPUDPEntry* p=nullptr;
    cudaGetSymbolAddress((void**)&p, d_dp_buffer);
    cudaMemcpy(h, p, n*sizeof(GPUDPEntry), cudaMemcpyDeviceToHost);
}

void gpu_launch_walk(ECPointJacobian* R, uint64_t* a, uint64_t* b,
                     uint64_t* sx, uint64_t* ss, uint32_t* ty,
                     uint32_t nw, uint32_t spl, cudaStream_t st) {
    int th = BLOCK_SIZE, bl = (nw + th - 1) / th;
    // smem: src + dst + Z + inv + x = 5 * BLOCK_SIZE * 4 * 8 bytes
    size_t smem = 5 * BLOCK_SIZE * 4 * sizeof(uint64_t);
    rho_walk_gpu<<<bl, th, smem, st>>>(R, a, b, sx, ss, ty, nw, spl);
}

} // extern "C"
