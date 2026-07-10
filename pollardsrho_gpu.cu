#include "secp256k1.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

extern __device__ __constant__ uint64_t ONE_MONT[4];

#define GPU_N_STEPS     2048
#define GPU_DP_BUF_SIZE 131072
#define BLOCK_SIZE      256
#define WARP_SIZE       32

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

__device__ __forceinline__ void field_inv(uint64_t* r, const uint64_t* a) {
    uint64_t x2[4],x3[4],x6[4],x9[4],x11[4],x22[4];
    uint64_t x44[4],x88[4],x176[4],x220[4],x223[4],t[4],tmp[4];
    #define S(r,x)     modMulMontP(r,x,x)
    #define M(r,x,y)   modMulMontP(r,x,y)
    #define SN(r,x,n)  {modMulMontP(r,x,x);for(int _=1;_<(n);_++)modMulMontP(r,r,r);}
    #define C(r,x)     for(int _=0;_<4;_++)r[_]=x[_]
    S(tmp,a);       M(x2,tmp,a);
    S(tmp,x2);      M(x3,tmp,a);
    SN(tmp,x3,3);   M(x6,tmp,x3);
    SN(tmp,x6,3);   M(x9,tmp,x3);
    SN(tmp,x9,2);   M(x11,tmp,x2);
    SN(tmp,x11,11); M(x22,tmp,x11);
    SN(tmp,x22,22); M(x44,tmp,x22);
    SN(tmp,x44,44); M(x88,tmp,x44);
    SN(tmp,x88,88); M(x176,tmp,x88);
    SN(tmp,x176,44);M(x220,tmp,x44);
    SN(tmp,x220,3); M(x223,tmp,x3);
    C(t,x223); S(t,t);
    SN(tmp,t,22);   M(t,tmp,x22);
    SN(t,t,4);
    S(tmp,t);       M(t,tmp,a);
    S(t,t);
    SN(tmp,t,2);    M(t,tmp,x2);
    S(t,t);
    S(tmp,t);       M(t,tmp,a);
    C(r,t);
    #undef S
    #undef M
    #undef SN
    #undef C
}

// ── Kernel v19: Prefix scan paralelo + back-prop por warps ───────────────────
//
// Otimizações vs v17:
// 1. Prefix scan: Hillis-Steele paralelo (todas as threads) vs sequencial (thread 0)
// 2. field_inv: ~270 ops vs modExpMontP ~505 ops
// 3. Back-prop: 8 warps em paralelo vs thread 0 sequencial
//    Cada warp processa WARP_SIZE=32 elementos da back-prop independentemente
//
// Back-prop por warps:
//   Warp w processa índices [w*32 .. (w+1)*32-1]
//   Cada warp precisa de inv_partial[w] = cur após processar até o início do warp
//   Thread 0 calcula inv_partial[] para cada warp (8 valores = rápido)
//   Depois cada warp processa seus 32 elementos em paralelo com os outros warps

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

    int warp_id  = tid_blk / WARP_SIZE;
    int lane_id  = tid_blk % WARP_SIZE;
    int n_warps  = (blk_n + WARP_SIZE - 1) / WARP_SIZE; // warps ativos

    // smem layout (5 buffers + warp_inv):
    // smem_A   [B*4] — double buffer A para prefix scan
    // smem_B   [B*4] — double buffer B para prefix scan
    // smem_Z   [B*4] — Z originais
    // smem_inv [B*4] — inversos de Z resultantes
    // smem_x   [B*4] — x_affine resultante
    // smem_wi  [8*4] — inv_partial por warp (máx 8 warps em bloco de 256)
    extern __shared__ uint64_t smem[];
    uint64_t* smem_A  = smem;
    uint64_t* smem_B  = smem + BLOCK_SIZE * 4;
    uint64_t* smem_Z  = smem + BLOCK_SIZE * 8;
    uint64_t* smem_inv= smem + BLOCK_SIZE * 12;
    uint64_t* smem_x  = smem + BLOCK_SIZE * 16;
    uint64_t* smem_wi = smem + BLOCK_SIZE * 20; // [8 warps * 4 limbs]

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

    for (uint32_t step = 0; step < steps_per_launch; step++) {

        if (d_found) break;

        // ── 1. Salva Z ────────────────────────────────────────────────────────
        if (tid_blk < blk_n) {
            for (int j = 0; j < 4; j++) smem_Z[tid_blk*4+j] = R.Z[j];
        } else {
            for (int j = 0; j < 4; j++) smem_Z[tid_blk*4+j] = ONE_MONT[j];
        }
        __syncthreads();

        // ── 2. Prefix scan (Hillis-Steele paralelo) ───────────────────────────
        // Todas as threads participam
        for (int j = 0; j < 4; j++) smem_A[tid_blk*4+j] = smem_Z[tid_blk*4+j];
        __syncthreads();

        for (int offset = 1; offset < blk_n; offset *= 2) {
            if (tid_blk >= offset && tid_blk < blk_n) {
                uint64_t tmp[4];
                modMulMontP(tmp, &smem_A[tid_blk*4], &smem_A[(tid_blk-offset)*4]);
                for (int j = 0; j < 4; j++) smem_B[tid_blk*4+j] = tmp[j];
            } else {
                for (int j = 0; j < 4; j++) smem_B[tid_blk*4+j] = smem_A[tid_blk*4+j];
            }
            __syncthreads();
            for (int j = 0; j < 4; j++) smem_A[tid_blk*4+j] = smem_B[tid_blk*4+j];
            __syncthreads();
        }
        // smem_A[i] = Z[0]*...*Z[i] para i < blk_n

        // ── 3. Inversão única (thread 0) ──────────────────────────────────────
        if (tid_blk == 0) {
            field_inv(&smem_inv[0], &smem_A[(blk_n-1)*4]);
            // smem_inv[0..3] = inv_total = 1/(Z[0]*...*Z[blk_n-1])
        }
        __syncthreads();

        // ── 4. Back-prop por warps ─────────────────────────────────────────────
        // Thread 0 calcula inv_partial[w] = cur no início de cada warp
        // cur começa como inv_total e avança conforme processa Z[blk_n-1], Z[blk_n-2]...
        if (tid_blk == 0) {
            uint64_t cur[4];
            for (int j = 0; j < 4; j++) cur[j] = smem_inv[j]; // inv_total

            // inv_partial[w] = cur quando chegarmos no warp w (de trás para frente)
            // Warp n_warps-1 começa com inv_total
            // Warp n_warps-2 começa após processar warp n_warps-1
            // etc.
            // Salva inv_partial para cada warp
            for (int j = 0; j < 4; j++) smem_wi[(n_warps-1)*4+j] = cur[j];

            for (int w = n_warps - 1; w > 0; w--) {
                int warp_start = w * WARP_SIZE;
                int warp_end   = min(warp_start + WARP_SIZE, blk_n);
                // Avança cur pelo warp w (processa de trás para frente)
                for (int i = warp_end - 1; i >= warp_start; i--) {
                    uint64_t new_cur[4];
                    modMulMontP(new_cur, cur, &smem_Z[i*4]);
                    for (int j = 0; j < 4; j++) cur[j] = new_cur[j];
                }
                for (int j = 0; j < 4; j++) smem_wi[(w-1)*4+j] = cur[j];
            }
        }
        __syncthreads();

        // Cada warp processa seus elementos em paralelo
        if (tid_blk < blk_n) {
            // cur local do warp
            uint64_t cur[4];
            for (int j = 0; j < 4; j++) cur[j] = smem_wi[warp_id*4+j];

            // Processa sequencialmente dentro do warp mas em paralelo entre warps
            int warp_end = min((warp_id + 1) * WARP_SIZE, blk_n);

            // Calcula inv[tid_blk] percorrendo do fim do warp até tid_blk
            // Cada thread do warp faz isso independentemente — O(WARP_SIZE) cada
            // mas todos os warps em paralelo → speedup de n_warps vezes
            uint64_t my_cur[4];
            for (int j = 0; j < 4; j++) my_cur[j] = cur[j];

            for (int i = warp_end - 1; i > tid_blk; i--) {
                uint64_t new_cur[4];
                modMulMontP(new_cur, my_cur, &smem_Z[i*4]);
                for (int j = 0; j < 4; j++) my_cur[j] = new_cur[j];
            }

            // my_cur agora é cur no ponto i = tid_blk
            // inv[tid_blk] = my_cur * smem_A[tid_blk-1]  (se tid_blk > 0)
            // inv[0] = my_cur
            if (tid_blk > 0) {
                uint64_t inv_i[4];
                modMulMontP(inv_i, my_cur, &smem_A[(tid_blk-1)*4]);
                for (int j = 0; j < 4; j++) smem_inv[tid_blk*4+j] = inv_i[j];
            } else {
                // tid_blk == 0: inv[0] = cur * smem_A[-1] = cur * ONE = cur
                // my_cur já está correto pois não entrou no loop acima
                // mas precisamos checar: cur = inv_total * Z[blk_n-1]*...*Z[1]
                // = 1/(Z[0]*...*Z[blk_n-1]) * Z[blk_n-1]*...*Z[1]
                // = 1/Z[0] ✓
                for (int j = 0; j < 4; j++) smem_inv[0*4+j] = my_cur[j];
            }
        }
        __syncthreads();

        // ── 5. Calcula x_aff ──────────────────────────────────────────────────
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

        // ── 6. Decisões locais ────────────────────────────────────────────────
        uint64_t* x_aff = &smem_x[tid_blk * 4];

        bool is_cycle = active && (x_aff[0]==snap_x[0] && x_aff[1]==snap_x[1] &&
                                   x_aff[2]==snap_x[2] && x_aff[3]==snap_x[3]);
        bool is_dp_now = active && gpu_is_dp(x_aff, dp_bits) && !is_cycle;
        bool do_restart = is_cycle || is_dp_now;

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
            snap_steps++;
            if ((snap_steps & (snap_steps - 1)) == 0)
                for (int i = 0; i < 4; i++) snap_x[i] = x_aff[i];
            uint32_t idx = gpu_step_idx(x_aff, GPU_N_STEPS);
            pointAddJacobian(&R, &R, &d_stepPoints[idx]);
            scalarAdd(a, a, &d_stepScalarsA[idx * 4]);
            scalarAdd(b, b, &d_stepScalarsB[idx * 4]);
            if (gpu_exceeds_max(a, d_max_scalar)) {
                uint64_t diff[4];
                scalarSub(diff, a, d_max_scalar);
                for (int i = 0; i < 4; i++) a[i] = diff[i];
                pointAddJacobian(&R, &R, &d_G_OFFSET);
            }
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
    // smem: A + B + Z + inv + x + wi(8 warps)
    size_t smem = (5 * BLOCK_SIZE * 4 + 8 * 4) * sizeof(uint64_t);
    rho_walk_gpu<<<bl, th, smem, st>>>(R, a, b, sx, ss, ty, nw, spl);
}

} // extern "C"
