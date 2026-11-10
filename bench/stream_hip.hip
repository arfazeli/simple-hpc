// STREAM-style triad a = b + s*c on the GPU: the device memory-bandwidth ceiling for T24.
// Build: hipcc -O3 --offload-arch=gfx942 bench/stream_hip.hip -o build-hip/stream_hip
// Usage: stream_hip [N]   (default 2^27 doubles = 1 GiB per array)
// Prints "kernel,GB_per_s" for copy (2 arrays) and triad (3 arrays): best of 20, no
// write-allocate traffic counted, as in STREAM.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CHECK(x)                                                                                \
    do {                                                                                        \
        hipError_t e_ = (x);                                                                    \
        if (e_ != hipSuccess) {                                                                 \
            std::fprintf(stderr, "HIP error %s: %s\n", #x, hipGetErrorString(e_));              \
            std::exit(1);                                                                       \
        }                                                                                       \
    } while (0)

__global__ void init(double* a, double* b, double* c, long n) {
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
        a[i] = 0.0, b[i] = 1.0, c[i] = 2.0;
}
__global__ void copy(double* a, const double* b, long n) {
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
        a[i] = b[i];
}
__global__ void triad(double* a, const double* b, const double* c, long n) {
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
        a[i] = b[i] + 3.0 * c[i];
}

int main(int argc, char** argv) {
    const long n = argc > 1 ? std::atol(argv[1]) : 1L << 27;
    double *a, *b, *c;
    CHECK(hipMalloc(&a, n * sizeof(double)));
    CHECK(hipMalloc(&b, n * sizeof(double)));
    CHECK(hipMalloc(&c, n * sizeof(double)));
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, 0));
    const int blocks = prop.multiProcessorCount * 8, threads = 256;
    init<<<blocks, threads>>>(a, b, c, n);
    CHECK(hipDeviceSynchronize());
    hipEvent_t t0, t1;
    CHECK(hipEventCreate(&t0));
    CHECK(hipEventCreate(&t1));
    std::printf("kernel,GB_per_s\n");
    for (int k = 0; k < 2; ++k) {
        float best = 1e30f;
        for (int r = 0; r < 20; ++r) {
            CHECK(hipEventRecord(t0));
            if (k == 0)
                copy<<<blocks, threads>>>(a, b, n);
            else
                triad<<<blocks, threads>>>(a, b, c, n);
            CHECK(hipEventRecord(t1));
            CHECK(hipEventSynchronize(t1));
            float ms = 0.0f;
            CHECK(hipEventElapsedTime(&ms, t0, t1));
            best = ms < best ? ms : best;
        }
        const double bytes = (k == 0 ? 2.0 : 3.0) * 8.0 * n;
        std::printf("%s,%.1f\n", k == 0 ? "copy" : "triad", bytes / (best * 1e-3) / 1e9);
    }
    CHECK(hipFree(a));
    CHECK(hipFree(b));
    CHECK(hipFree(c));
    return 0;
}
