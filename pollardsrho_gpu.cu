#include "secp256k1.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

extern __device__ __constant__ uint64_t ONE_MONT[4];

#define GPU_N_STEPS     2048
#define GPU_DP_BUF_SIZE 131072
#define BLOCK_SIZE      128
#define WARP_SIZE       32
#define INNER_STEPS     1  // passos jacobianos entre cada batch inversion

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
    uint64_t x2[4],x3[4],x22[4],x44[4],x88[4],x220[4],x223[4],t[4],t2[4];
    #define S(r,x)     modMulMontP(r,x,x)
    #define M(r,x,y)   modMulMontP(r,x,y)
    #define SN(r,x,n)  {modMulMontP(r,x,x);for(int _=1;_<(n);_++)modMulMontP(r,r,r);}
    #define C(r,x)     for(int _=0;_<4;_++)r[_]=x[_]
    S(t,a);        M(x2,t,a);
    S(t,x2);       M(x3,t,a);
    SN(t,x3,3);    M(t,t,x3);
    SN(t,t,3);     M(t,t,x3);
    SN(t,t,2);     M(t,t,x2);
    SN(x22,t,11);  M(x22,x22,t);
    SN(x44,x22,22);M(x44,x44,x22);
    SN(x88,x44,44);M(x88,x88,x44);
    SN(t,x88,88);  M(t,t,x88);
    SN(x220,t,44); M(x220,x220,x44);
    SN(x223,x220,3);M(x223,x223,x3);
    C(t,x223);     S(t,t);
    SN(t2,t,22);   M(t,t2,x22);
    SN(t,t,4);
    S(t2,t);       M(t,t2,a);
    S(t,t);
    SN(t2,t,2);    M(t,t2,x2);
    S(t,t);
    S(t2,t);       M(t,t2,a);
    C(r,t);
    #undef S
    #undef M
    #undef SN
    #undef C
}

// ── Kernel v21: Re-normalização lazy ─────────────────────────────────────────
// Invariante: R.Z = ONE_MONT no início de cada batch
// Durante INNER_STEPS: hash usa R.X (em Montgomery, correto pois Z=ONE_MONT)
// Após INNER_STEPS: batch inversion restaura Z=ONE_MONT
// Custo da inversão: 1 a cada INNER_STEPS steps (ganho de INNER_STEPS vezes)

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

    int warp_id = tid_blk / WARP_SIZE;
    int n_warps = (blk_n + WARP_SIZE - 1) / WARP_SIZE;

    extern __shared__ uint64_t smem[];
    uint64_t* smem_A  = smem;
    uint64_t* smem_B  = smem + BLOCK_SIZE * 4;
    uint64_t* smem_Z  = smem + BLOCK_SIZE * 8;
    uint64_t* smem_inv= smem + BLOCK_SIZE * 12;
    uint64_t* smem_x  = smem + BLOCK_SIZE * 16;
    uint64_t* smem_wi = smem + BLOCK_SIZE * 20;

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

    // Quantos batches executar
    uint32_t batches = steps_per_launch / INNER_STEPS;
    if (batches == 0) batches = 1;

    for (uint32_t batch = 0; batch < batches; batch++) {

        if (d_found) break;

        // ── Fase 1: INNER_STEPS passos jacobianos puros ───────────────────────
        // R.Z = ONE_MONT no início → R.X = toMont(x_aff) → hash correto!
        // Após cada pointAddJacobian, Z muda mas o hash do PRÓXIMO step
        // usará R.X jacobiano — isso introduz inconsistência entre walkers.
        //
        // SOLUÇÃO: usa step_idx baseado no R.X do início do step (antes do add)
        // Isso garante que dois walkers no mesmo ponto affine tomem o mesmo step.
        for (int s = 0; s < INNER_STEPS; s++) {
            // Hash ANTES do add (R.X ainda é toMont(x_aff) do step anterior)
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

        // ── Fase 2: Batch inversion (1 inversão para todo o bloco) ───────────
        // Salva Z e inicializa smem_A simultaneamente
        if (tid_blk < blk_n) {
            for (int j = 0; j < 4; j++) {
                smem_Z[tid_blk*4+j] = R.Z[j];
                smem_A[tid_blk*4+j] = R.Z[j];
            }
        } else {
            for (int j = 0; j < 4; j++) {
                smem_Z[tid_blk*4+j] = ONE_MONT[j];
                smem_A[tid_blk*4+j] = ONE_MONT[j];
            }
        }
        __syncthreads();

        // Prefix scan (Hillis-Steele)
        uint64_t* cur_src = smem_A;
        uint64_t* cur_dst = smem_B;
        for (int offset = 1; offset < blk_n; offset *= 2) {
            if (tid_blk >= offset && tid_blk < blk_n) {
                uint64_t tmp[4];
                modMulMontP(tmp, &cur_src[tid_blk*4], &cur_src[(tid_blk-offset)*4]);
                for (int j = 0; j < 4; j++) cur_dst[tid_blk*4+j] = tmp[j];
            } else {
                for (int j = 0; j < 4; j++) cur_dst[tid_blk*4+j] = cur_src[tid_blk*4+j];
            }
            __syncthreads();
            uint64_t* t = cur_src; cur_src = cur_dst; cur_dst = t;
        }

        // Inversão única (thread 0)
        if (tid_blk == 0) {
            field_inv(&smem_inv[0], &cur_src[(blk_n-1)*4]);
        }
        __syncthreads();

        // Back-prop por warps
        if (tid_blk == 0) {
            uint64_t cur[4];
            for (int j = 0; j < 4; j++) cur[j] = smem_inv[j];
            for (int j = 0; j < 4; j++) smem_wi[(n_warps-1)*4+j] = cur[j];
            for (int w = n_warps - 1; w > 0; w--) {
                int ws = w * WARP_SIZE;
                int we = min(ws + WARP_SIZE, blk_n);
                for (int i = we - 1; i >= ws; i--) {
                    uint64_t nc[4];
                    modMulMontP(nc, cur, &smem_Z[i*4]);
                    for (int j = 0; j < 4; j++) cur[j] = nc[j];
                }
                for (int j = 0; j < 4; j++) smem_wi[(w-1)*4+j] = cur[j];
            }
        }
        __syncthreads();

        if (tid_blk < blk_n) {
            uint64_t my_cur[4];
            for (int j = 0; j < 4; j++) my_cur[j] = smem_wi[warp_id*4+j];
            int we = min((warp_id + 1) * WARP_SIZE, blk_n);
            for (int i = we - 1; i > tid_blk; i--) {
                uint64_t nc[4];
                modMulMontP(nc, my_cur, &smem_Z[i*4]);
                for (int j = 0; j < 4; j++) my_cur[j] = nc[j];
            }
            if (tid_blk > 0) {
                uint64_t inv_i[4];
                modMulMontP(inv_i, my_cur, &cur_src[(tid_blk-1)*4]);
                for (int j = 0; j < 4; j++) smem_inv[tid_blk*4+j] = inv_i[j];
            } else {
                for (int j = 0; j < 4; j++) smem_inv[j] = my_cur[j];
            }
        }
        __syncthreads();

        // Calcula x_aff e restaura Z=ONE_MONT
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
            affineToJacobian(&R, &aff); // restaura Z=ONE_MONT
        }
        __syncthreads();

        // ── Fase 3: Decisões com x_aff real ──────────────────────────────────
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
    size_t smem = (5 * BLOCK_SIZE * 4 + 8 * 4) * sizeof(uint64_t);
    rho_walk_gpu<<<bl, th, smem, st>>>(R, a, b, sx, ss, ty, nw, spl);
}

} // extern "C"
