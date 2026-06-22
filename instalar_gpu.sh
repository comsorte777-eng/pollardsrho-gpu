#!/bin/bash
# Script que gera todos os arquivos do projeto Pollard's Rho GPU v3
# Execute dentro de ~/teste/pollardsrho:
#   bash instalar_gpu.sh

set -e
cd "$(dirname "$0")"

echo "=== Gerando pollardsrho_gpu.cu ==="
cat > pollardsrho_gpu.cu << 'EOF'
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
            if (walker_type[tid] == 1) { b[0] = 1; pointAddJacobian(&R, &R, &d_target_jac); }
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
        if (walker_type[tid] == 1) { b[0] = 1; pointAddJacobian(&R, &R, &d_target_jac); }
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
EOF

echo "=== Gerando pollardsrho_gpu_host.cpp ==="
cat > pollardsrho_gpu_host.cpp << 'EOF'
#include "secp256k1.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>
#include <cmath>
#include <chrono>
#include <random>
#include <openssl/sha.h>
#include <cuda_runtime.h>
#include "parallel_hashmap/phmap.h"

struct GPUDPEntry { uint64_t x[4], a[4], b[4]; uint32_t valid; };

extern "C" {
    void gpu_upload_steps(const ECPointJacobian*, const uint64_t*, const uint64_t*, int);
    void gpu_upload_params(const ECPointJacobian*, const ECPointJacobian*, const uint64_t*, int, int);
    void gpu_reset_dp_buffer();
    void gpu_signal_found();
    uint32_t gpu_get_dp_count();
    void gpu_fetch_dp_buffer(GPUDPEntry*, uint32_t);
    void gpu_launch_walk(ECPointJacobian*, uint64_t*, uint64_t*, uint64_t*, uint64_t*, uint32_t*, uint32_t, uint32_t, cudaStream_t);
}

constexpr uint256_t N_ORDER = {
    0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
};
constexpr uint64_t ZERO[4] = {0,0,0,0};

const std::string GREEN="\033[92m", BLUE="\033[94m",
    CYAN="\033[38;5;39m", PINK="\033[35m", ORANGE="\033[38;2;255;128;0m", RESET="\033[0m";
const char* BASE_58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

int windowSize = 12;
extern ECPointJacobian* preCompG;
extern ECPointJacobian* preCompGphi;
extern ECPointJacobian* preCompH;
extern ECPointJacobian* preCompHphi;
extern ECPointJacobian* jacNorm;
extern ECPointJacobian* jacEndo;
extern ECPointJacobian* jacNormH;
extern ECPointJacobian* jacEndoH;

void getfcw(int key_range) {
    double exp_steps = std::pow(2.0, key_range / 2.0);
    size_t l2 = 0, l3 = 0;
    for (int idx = 0;; idx++) {
        std::string base = "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(idx);
        std::ifstream lf(base + "/level"); if (!lf.is_open()) break;
        int L = 0; lf >> L;
        std::ifstream tf(base + "/type"); std::string t; tf >> t;
        if (t != "Data" && t != "Unified") continue;
        std::ifstream sf(base + "/size"); std::string s; sf >> s;
        if (s.empty()) continue;
        size_t m = (s.back()=='K') ? 1024 : (s.back()=='M') ? 1024*1024 : 1;
        try {
            size_t sz = std::stoul(s.substr(0,s.size()-1)) * m;
            if (L==2) l2=sz; if (L==3) l3=sz;
        } catch(...) {}
    }
    size_t ls = (exp_steps < (double)l3) ? l3 : l2;
    if (ls > 0) {
        size_t mp = ls / 128;
        if (mp > 1) { int w=(int)std::floor(std::log2((double)mp)); if(w>windowSize) windowSize=w; }
    }
}

static uint256_t sub256(const uint256_t& a, const uint256_t& b) {
    uint256_t r{}; uint64_t bw=0;
    for(int i=0;i<4;i++){uint64_t res=a.limbs[i]-b.limbs[i]-bw;bw=(a.limbs[i]<b.limbs[i]+bw)?1:0;r.limbs[i]=res;}
    return r;
}
static uint256_t add256(const uint256_t& a, const uint256_t& b) {
    uint256_t r{}; uint64_t c=0;
    for(int i=0;i<4;i++){unsigned __int128 s=(unsigned __int128)a.limbs[i]+b.limbs[i]+c;r.limbs[i]=(uint64_t)s;c=(uint64_t)(s>>64);}
    return r;
}
static int cmp256(const uint256_t& a, const uint256_t& b) {
    for(int i=3;i>=0;i--){if(a.limbs[i]>b.limbs[i])return 1;if(a.limbs[i]<b.limbs[i])return -1;}return 0;
}
static uint256_t rng256(const uint256_t& lo, const uint256_t& hi, int bits, std::mt19937_64& rng) {
    uint256_t r{}; int ml=(bits-1)/64, rb=bits%64;
    do {
        for(int i=0;i<4;i++) r.limbs[i]=rng();
        for(int i=ml+1;i<4;i++) r.limbs[i]=0;
        if(rb>0) r.limbs[ml]&=(1ULL<<rb)-1;
    } while(cmp256(r,add256(sub256(hi,lo),{1,0,0,0}))>=0);
    return add256(r,lo);
}
static std::string to_hex(const uint256_t& v) {
    std::ostringstream o; o<<std::hex<<std::setfill('0');
    for(int i=3;i>=0;i--) o<<std::setw(16)<<v.limbs[i]; return o.str();
}
static std::vector<uint8_t> from_hex(const std::string& h) {
    std::vector<uint8_t> r(h.size()/2);
    for(size_t i=0;i<r.size();i++) r[i]=(uint8_t)std::stoi(h.substr(2*i,2),nullptr,16);
    return r;
}

struct DPKey { uint64_t x0,x1; bool operator==(const DPKey& o) const{return x0==o.x0&&x1==o.x1;} };
struct DPKeyHash { size_t operator()(const DPKey& k) const{return k.x0^k.x1*0xff51afd7ed558ccdULL;} };
struct DPVal { uint64_t a[4], b[4]; };
using DPMap = phmap::flat_hash_map<DPKey,DPVal,DPKeyHash>;

static bool resolve(const GPUDPEntry& e1, const DPVal& e2,
                    const std::vector<uint8_t>& pub, uint256_t& out) {
    uint256_t a1{},a2{},b1{},b2{};
    for(int i=0;i<4;i++){a1.limbs[i]=e1.a[i];b1.limbs[i]=e1.b[i];a2.limbs[i]=e2.a[i];b2.limbs[i]=e2.b[i];}
    uint256_t db=(cmp256(b2,b1)>=0)?sub256(b2,b1):sub256(N_ORDER,sub256(b1,b2));
    if(scalarIsZero(db.limbs)) return false;
    uint256_t da=(cmp256(a1,a2)>=0)?sub256(a1,a2):sub256(N_ORDER,sub256(a2,a1));
    uint256_t inv=almostinverse(db,N_ORDER);
    uint64_t k[4]; scalarMul(k,da.limbs,inv.limbs);
    uint8_t tp[33]; generatePublicKey(preCompG,preCompGphi,tp,k,windowSize);
    if(memcmp(tp,pub.data(),33)!=0) return false;
    for(int i=0;i<4;i++) out.limbs[i]=k[i]; return true;
}

static std::string to_wif(const std::string& hex) {
    auto b=from_hex(hex);
    std::vector<uint8_t> p; p.push_back(0x80);
    p.insert(p.end(),b.begin(),b.end()); p.push_back(0x01);
    uint8_t h1[32],h2[32];
    SHA256(p.data(),p.size(),h1); SHA256(h1,32,h2);
    p.insert(p.end(),h2,h2+4);
    std::vector<int> d;
    for(uint8_t byte:p){int c=byte;for(auto&x:d){c+=x*256;x=c%58;c/=58;}while(c>0){d.push_back(c%58);c/=58;}}
    std::string r; for(uint8_t x:p){if(x==0)r+='1';else break;}
    for(auto it=d.rbegin();it!=d.rend();++it) r+=BASE_58[*it]; return r;
}

static void init_ec(int kr) {
    getfcw(kr);
    preCompG    = new ECPointJacobian[1ULL<<windowSize];
    preCompGphi = new ECPointJacobian[1ULL<<windowSize];
    preCompH    = new ECPointJacobian[1ULL<<windowSize];
    preCompHphi = new ECPointJacobian[1ULL<<windowSize];
    jacNorm     = new ECPointJacobian[windowSize];
    jacNormH    = new ECPointJacobian[windowSize];
    jacEndo     = new ECPointJacobian[windowSize];
    jacEndoH    = new ECPointJacobian[windowSize];
    initPreCompG(windowSize);
}

int main(int argc, char* argv[]) {
    std::string pub; int kr=0,nw=0,dp=0;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--pubkey")  &&i+1<argc) pub=argv[++i];
        if(!strcmp(argv[i],"--keyrange")&&i+1<argc) kr=atoi(argv[++i]);
        if(!strcmp(argv[i],"--walkers") &&i+1<argc) nw=atoi(argv[++i]);
        if(!strcmp(argv[i],"--dp")      &&i+1<argc) dp=atoi(argv[++i]);
    }
    if(pub.empty()||kr==0||nw==0){
        std::cerr<<"Uso: ./pollardsrho_gpu --pubkey <hex> --keyrange <bits> --walkers <n> [--dp <bits>]\n";
        return 1;
    }

    // dp_bits: min 8, max kr/2, formula original
    if(dp==0) {
        int formula = (int)std::abs(std::round(kr/2.0 - 10.0));
        dp = std::max(8, std::min(kr/2, formula));
    }

    std::cout<<BLUE<<"-----------------------------------------------------------------------\n"<<RESET;
    std::cout<<CYAN<<"Pollard's Rho GPU - RTX 3050\n"<<RESET;
    std::cout<<CYAN<<"Pubkey    : "<<RESET<<PINK<<pub<<"\n"<<RESET;
    std::cout<<CYAN<<"Key range : "<<RESET<<PINK<<"2^"<<kr<<"\n"<<RESET;
    std::cout<<CYAN<<"Walkers   : "<<RESET<<PINK<<nw<<"\n"<<RESET;
    std::cout<<CYAN<<"DP bits   : "<<RESET<<PINK<<dp<<"\n"<<RESET;
    std::cout<<BLUE<<"-----------------------------------------------------------------------\n"<<RESET;

    init_ec(kr);
    auto pubBytes=from_hex(pub);
    ECPointAffine ta{}; ECPointJacobian tj{};
    decompressPublicKey(&ta,pubBytes.data());
    affineToJacobian(&tj,&ta);
    initPreCompH(&tj,windowSize);

    uint256_t lo{},hi{};
    { int lm=(kr-1)/64,bt=(kr-1)%64;
      lo.limbs[lm]=1ULL<<bt;
      for(int i=0;i<lm;i++) hi.limbs[i]=~0ULL;
      hi.limbs[lm]=(bt==63)?~0ULL:(1ULL<<(bt+1))-1; }

    ECPointJacobian GO{};
    jacobianScalarMultPhi(&GO,preCompG,preCompGphi,hi.limbs,windowSize);
    if(!jacobianIsInfinity(&GO)) modSubP(GO.Y,ZERO,GO.Y);

    const int NS=2048;
    struct SE { ECPointJacobian p; uint64_t a[4],b[4]; };
    std::vector<SE> st(NS);
    std::mt19937_64 salt(ta.x[0]);
    uint256_t ss{};
    { int lm=(kr/2)/64,bt=(kr/2)%64; ss.limbs[lm]=1ULL<<bt; }

    std::cout<<ORANGE<<"[INFO] Gerando tabela de steps...\n"<<RESET;
    for(int i=0;i<NS;i++){
        uint256_t sa=rng256({1,0,0,0},ss,kr/2,salt);
        for(int k=0;k<4;k++){st[i].a[k]=sa.limbs[k];st[i].b[k]=0;}
        uint64_t at[4]; for(int k=0;k<4;k++) at[k]=sa.limbs[k];
        jacobianScalarMultPhi(&st[i].p,preCompG,preCompGphi,at,windowSize);
        ECPointAffine af{}; jacobianToAffine(&af,&st[i].p); affineToJacobian(&st[i].p,&af);
    }

    std::cout<<ORANGE<<"[INFO] Enviando para GPU...\n"<<RESET;
    std::vector<ECPointJacobian> gp(NS);
    std::vector<uint64_t> ga(NS*4),gb(NS*4);
    for(int i=0;i<NS;i++){
        gp[i]=st[i].p;
        for(int k=0;k<4;k++){ga[i*4+k]=st[i].a[k];gb[i*4+k]=st[i].b[k];}
    }
    gpu_upload_steps(gp.data(),ga.data(),gb.data(),NS);
    gpu_upload_params(&GO,&tj,hi.limbs,dp,windowSize);

    ECPointJacobian *dR=nullptr;
    uint64_t *da=nullptr,*db=nullptr,*dsx=nullptr,*dss=nullptr;
    uint32_t *dty=nullptr;
    cudaMalloc(&dR, nw*sizeof(ECPointJacobian));
    cudaMalloc(&da, nw*4*sizeof(uint64_t)); cudaMalloc(&db, nw*4*sizeof(uint64_t));
    cudaMalloc(&dsx,nw*4*sizeof(uint64_t)); cudaMalloc(&dss,nw*sizeof(uint64_t));
    cudaMalloc(&dty,nw*sizeof(uint32_t));

    std::cout<<ORANGE<<"[INFO] Inicializando "<<nw<<" walkers...\n"<<RESET;
    {
        std::vector<ECPointJacobian> hR(nw);
        std::vector<uint64_t> ha(nw*4,0),hb(nw*4,0),hsx(nw*4,0xFFFFFFFFFFFFFFFFULL),hss(nw,0);
        std::vector<uint32_t> hty(nw);
        std::mt19937_64 rng(std::random_device{}());
        for(int i=0;i<nw;i++){
            uint256_t a=rng256(lo,hi,kr,rng);
            for(int k=0;k<4;k++) ha[i*4+k]=a.limbs[k];
            hty[i]=i%2; hb[i*4]=hty[i];
            uint64_t at[4]; for(int k=0;k<4;k++) at[k]=a.limbs[k];
            ECPointJacobian Ra{}; jacobianScalarMultPhi(&Ra,preCompG,preCompGphi,at,windowSize);
            if(hty[i]==0) hR[i]=Ra; else pointAddJacobian(&hR[i],&Ra,&tj);
        }
        cudaMemcpy(dR, hR.data(), nw*sizeof(ECPointJacobian),cudaMemcpyHostToDevice);
        cudaMemcpy(da, ha.data(), nw*4*sizeof(uint64_t),cudaMemcpyHostToDevice);
        cudaMemcpy(db, hb.data(), nw*4*sizeof(uint64_t),cudaMemcpyHostToDevice);
        cudaMemcpy(dsx,hsx.data(),nw*4*sizeof(uint64_t),cudaMemcpyHostToDevice);
        cudaMemcpy(dss,hss.data(),nw*sizeof(uint64_t),  cudaMemcpyHostToDevice);
        cudaMemcpy(dty,hty.data(),nw*sizeof(uint32_t),  cudaMemcpyHostToDevice);
    }

    // Pinned buffer para DPs
    const uint32_t DP_BUF = 131072;
    GPUDPEntry* hbuf=nullptr; cudaMallocHost(&hbuf,DP_BUF*sizeof(GPUDPEntry));
    DPMap dpmap; dpmap.reserve(1<<22);
    cudaStream_t stream; cudaStreamCreate(&stream);

    auto t0=std::chrono::steady_clock::now();
    uint64_t total=0; bool found=false; uint256_t fkey{}; double lp=0;
    uint64_t total_dps=0, collisions=0;

    std::cout<<BLUE<<"-----------------------------------------------------------------------\n"<<RESET;
    std::cout<<GREEN<<"[INFO] Iniciando busca GPU...\n"<<RESET;

    while(!found){
        // Lanca kernel
        gpu_launch_walk(dR,da,db,dsx,dss,dty,nw,8192,stream);
        cudaStreamSynchronize(stream);
        total+=(uint64_t)nw*8192;

        // Fetch e reset IMEDIATAMENTE antes de processar
        uint32_t cnt=gpu_get_dp_count();
        if(cnt>0){
            uint32_t fetch=std::min(cnt,DP_BUF);
            gpu_fetch_dp_buffer(hbuf,fetch);
            gpu_reset_dp_buffer();  // reseta ANTES de processar para nao perder DPs novos
            total_dps+=fetch;

            for(uint32_t i=0;i<fetch&&!found;i++){
                if(!hbuf[i].valid) continue;
                DPKey key={hbuf[i].x[0],hbuf[i].x[1]};
                auto it=dpmap.find(key);
                if(it!=dpmap.end()){
                    collisions++;
                    if(resolve(hbuf[i],it->second,pubBytes,fkey)){
                        found=true;
                        gpu_signal_found();
                    }
                } else {
                    DPVal v{}; for(int k=0;k<4;k++){v.a[k]=hbuf[i].a[k];v.b[k]=hbuf[i].b[k];}
                    dpmap[key]=v;
                }
            }
        }

        // Status a cada 10s
        auto now=std::chrono::steady_clock::now();
        double el=std::chrono::duration<double>(now-t0).count();
        if(el-lp>=10.0){
            lp=el; double ms=total/1e6;
            std::cout<<CYAN<<"["<<(int)el<<"s] "
                     <<std::fixed<<std::setprecision(1)
                     <<ms<<"M steps | "<<(ms/el)<<"M/s | "
                     <<"DPs: "<<dpmap.size()<<" | "
                     <<"Colisoes: "<<collisions
                     <<"\n"<<RESET<<std::flush;
        }
    }

    double el=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    std::string kh=to_hex(fkey);
    std::cout<<"\n"<<GREEN
             <<"╔══════════════════════════════════════════╗\n"
             <<"║  CHAVE PRIVADA ENCONTRADA                ║\n"
             <<"╚══════════════════════════════════════════╝\n"<<RESET;
    std::cout<<CYAN<<"Private key (hex): "<<RESET<<PINK<<kh<<"\n"<<RESET;
    std::cout<<CYAN<<"WIF              : "<<RESET<<PINK<<to_wif(kh)<<"\n"<<RESET;
    std::cout<<CYAN<<"Tempo            : "<<RESET<<PINK
             <<std::fixed<<std::setprecision(1)<<el<<"s\n"<<RESET;

    std::ofstream f("DISCRETE_LOGS_SOLVED",std::ios::app);
    f<<pub<<" : "<<kh<<"\n"; f.close();
    std::cout<<ORANGE<<"[INFO] Salvo em DISCRETE_LOGS_SOLVED\n"<<RESET;

    cudaFree(dR);cudaFree(da);cudaFree(db);cudaFree(dsx);cudaFree(dss);cudaFree(dty);
    cudaFreeHost(hbuf); cudaStreamDestroy(stream);
    delete[]preCompG;delete[]preCompGphi;delete[]preCompH;delete[]preCompHphi;
    delete[]jacNorm;delete[]jacNormH;delete[]jacEndo;delete[]jacEndoH;
    return 0;
}
EOF

echo "=== Gerando Makefile_gpu ==="
cat > Makefile_gpu << 'EOF'
TARGET    := pollardsrho_gpu
CXX       := g++
ARCH_NAME := $(shell uname -m)

ifeq ($(ARCH_NAME), x86_64)
    CUDA_DIR_NAME := cuda_x86_64
else ifeq ($(ARCH_NAME), aarch64)
    CUDA_DIR_NAME := cuda_aarch64
else
    $(error Arch $(ARCH_NAME) does not support!)
endif

CUDA_HOME := $(abspath $(CURDIR)/$(CUDA_DIR_NAME))
NVCC      := $(CUDA_HOME)/bin/nvcc
export PATH := $(CUDA_HOME)/bin:$(CUDA_HOME)/nvvm/bin:$(PATH)

INCLUDES  := -I$(CUDA_HOME)/include -I$(CUDA_HOME) -I.
LDFLAGS   := -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)
LDLIBS    := -lcudadevrt -lcudart_static -lpthread -ldl -lrt -lcrypto

SRC_CPP   := almostinverse.cpp pollardsrho_gpu_host.cpp
SRC_CU    := secp256k1.cu pollardsrho_gpu.cu
OBJ_CPP   := $(SRC_CPP:.cpp=.o)
OBJ_CU    := $(SRC_CU:.cu=.o)
OBJ       := $(OBJ_CPP) $(OBJ_CU)

.PHONY: all fresh set_perms recurse gpu_arch clean

all: fresh

fresh: set_perms gpu_arch
	@$(MAKE) -f Makefile_gpu recurse

set_perms:
	@echo "Configuring CUDA Toolkit Permissions..."
	@find $(CUDA_HOME) -type f -path "*/bin/*" -exec chmod +x {} + 2>/dev/null || true

gpu_arch: arch
	@RESULT=$$(./arch 2>/dev/null | grep -E '^[0-9]+$$' || echo "0"); \
	echo "GPU_ARCH := $$RESULT" > gpu_arch_result

arch: arch.cu | set_perms
	$(NVCC) $(INCLUDES) $(LDFLAGS) -ccbin $(CXX) arch.cu -o arch $(LDLIBS)

recurse: $(TARGET)
-include gpu_arch_result

CXXFLAGS  := -O3 -std=c++17 -pthread -I. $(INCLUDES)
NVCCFLAGS := -O3 -std=c++17 -rdc=true -dc -ccbin $(CXX) $(INCLUDES) \
             -Xcompiler "-O3 -pthread -fpermissive -fPIC" \
             --expt-relaxed-constexpr --maxrregcount=96

DLINKFLAGS :=
ifneq ($(filter-out 0,$(strip $(GPU_ARCH))),)
    NVCCFLAGS  += -gencode arch=compute_$(strip $(GPU_ARCH)),code=sm_$(strip $(GPU_ARCH))
    DLINKFLAGS := -gencode arch=compute_$(strip $(GPU_ARCH)),code=sm_$(strip $(GPU_ARCH))
endif

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

dlink.o: $(OBJ_CU)
	$(NVCC) $(DLINKFLAGS) $(INCLUDES) -rdc=true -dlink $(OBJ_CU) -o dlink.o $(LDFLAGS) -ccbin $(CXX)

$(TARGET): $(OBJ) dlink.o
	$(NVCC) $(DLINKFLAGS) $(OBJ) dlink.o -o $@ $(LDFLAGS) $(LDLIBS) -ccbin $(CXX)
	@echo ""
	@echo "Build OK -> ./$(TARGET)"

clean:
	rm -f $(TARGET) arch gpu_arch_result dlink.o
	find . -type f \( -name "*.o" -o -name "*.d" \) -delete
EOF

echo ""
echo "=== Compilando ==="
make -f Makefile_gpu

echo ""
echo "=== Pronto! Para testar puzzle #40: ==="
echo "./pollardsrho_gpu --pubkey 03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4 --keyrange 40 --walkers 2048"
