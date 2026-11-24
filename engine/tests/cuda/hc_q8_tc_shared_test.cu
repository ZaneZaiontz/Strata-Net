// tests/cuda/hc_q8_tc_shared_test.cu - the STRATA_HC_Q8 window read (fused_gr_read_multi with Q8_0 projections) on the
// tensor-core kernels, which stage the weights in shared memory.  Reading those staged blocks through __ldg (a global
// load) faulted on every window: an illegal address on sm_86, "operation not supported on global/shared address space"
// on sm_120.  ctest runs it four times: as is and with STRATA_HC_Q8_TC=2 (the INT8 tensor-core up, then down too), with
// STRATA_HC_Q8_TC=0 and with STRATA_HC_Q8_BF16TC=2 beside it (the BF16 tensor-core up, then down too).
//
// Exit 0 pass, 1 fail, 77 no CUDA device (ctest SKIP_RETURN_CODE).
#include "strata/core/geometry.hpp"
#include "strata/kernels/fused_gr.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr int N = (int) strata::core::kCompiledGeometry.n_embd;
constexpr int HC = (int) strata::core::kCompiledGeometry.hc;
constexpr int LR = (int) strata::core::kCompiledGeometry.hc_lr;
constexpr int D = N * HC;

/// Q8_0 blocks for a rows x cols matrix: an fp16 scale of 1/64 and small random codes.
std::vector<uint8_t> q8_matrix(int64_t rows, int64_t cols, std::mt19937& g) {
    std::uniform_int_distribution<int> code(-20, 20);
    std::vector<uint8_t> out((size_t) (rows * cols / 32 * 34));
    for (size_t b = 0; b < out.size(); b += 34) {
        out[b] = 0x00;
        out[b + 1] = 0x24;
        for (int i = 0; i < 32; ++i) out[b + 2 + i] = (uint8_t) (int8_t) code(g);
    }
    return out;
}

template <typename T>
T* upload(const std::vector<T>& v) {
    T* p = nullptr;
    if (cudaMalloc(&p, v.size() * sizeof(T)) != cudaSuccess) return nullptr;
    cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}

/// One window read of `t` tokens; true when it ran without a CUDA error and every output is finite.
bool read_window(int t, uint32_t seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> val(0.0f, 1.0f);
    std::vector<float> r((size_t) t * D), norm((size_t) D, 1.0f);
    for (auto& v : r) v = val(g);
    uint8_t* q8_down = upload(q8_matrix(LR, D, g));
    uint8_t* q8_up = upload(q8_matrix(D, LR, g));
    float* d_r = upload(r);
    float* d_norm = upload(norm);
    float *lo = nullptr, *rs = nullptr, *mixed = nullptr, *xn = nullptr;
    cudaMalloc(&lo, (size_t) t * LR * 4);
    cudaMalloc(&rs, (size_t) t * HC * 4);
    cudaMalloc(&mixed, (size_t) t * N * 4);
    cudaMalloc(&xn, (size_t) t * HC * N * 4);
    strata::kernels::FusedGrArgs a[strata::kernels::kFusedGrMaxT];
    for (int k = 0; k < t; ++k) {
        a[k].R = d_r + (size_t) k * D;
        a[k].R_out = d_r + (size_t) k * D;
        a[k].w_norm = d_norm;
        a[k].q8_down = q8_down;
        a[k].q8_up = q8_up;
        a[k].lo = lo + (size_t) k * LR;
        a[k].rs = rs + (size_t) k * HC;
        a[k].mixed = mixed + (size_t) k * N;
    }
    strata::kernels::fused_gr_read_multi(a, t, xn, nullptr);
    const cudaError_t e = cudaDeviceSynchronize();
    std::vector<float> got((size_t) t * N);
    bool finite = e == cudaSuccess && cudaMemcpy(got.data(), mixed, got.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
    for (float v : got) finite = finite && std::isfinite(v);
    std::printf("Q8 window read, %d tokens: %s%s\n", t, e == cudaSuccess ? "ran" : cudaGetErrorString(e),
                finite ? ", finite" : ", FAIL");
    for (void* p : {(void*) q8_down, (void*) q8_up, (void*) d_r, (void*) d_norm, (void*) lo, (void*) rs, (void*) mixed, (void*) xn})
        cudaFree(p);
    return finite;
}

}  // namespace

int main() {
    int devs = 0;
    if (cudaGetDeviceCount(&devs) != cudaSuccess || devs < 1) {
        std::printf("no CUDA device\n");
        return 77;
    }
    int fails = 0;
    for (int t = 1; t <= strata::kernels::kFusedGrMaxT && fails == 0; ++t) fails += read_window(t, (uint32_t) t) ? 0 : 1;
    std::printf(fails == 0 ? "PASS\n" : "FAILED\n");
    return fails == 0 ? 0 : 1;
}
