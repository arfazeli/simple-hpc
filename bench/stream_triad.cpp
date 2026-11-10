// STREAM-style triad a = b + s*c: sustainable memory bandwidth vs OpenMP threads.
// Build: g++-15 -O3 -fopenmp bench/stream_triad.cpp -o build-omp/stream_triad
// Prints "threads,GB_per_s" (best of 20 repeats; 3 arrays x 8 bytes x N moved per triad,
// counting no write-allocate traffic, as in STREAM).
#include <omp.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    // Default 128 MB per array (far above the M3 caches); pass a larger N for big-L3 servers.
    const long N = argc > 1 ? std::atol(argv[1]) : 1L << 24;
    std::vector<double> a(N), b(N), c(N);
    const int max_threads = omp_get_max_threads();
    std::printf("threads,GB_per_s\n");
    for (int t = 1; t <= max_threads; ++t) {
        omp_set_num_threads(t);
#pragma omp parallel for schedule(static)
        for (long i = 0; i < N; ++i) {  // first touch with the same partition
            a[i] = 0.0;
            b[i] = 1.0;
            c[i] = 2.0;
        }
        double best = 1e30;
        for (int r = 0; r < 20; ++r) {
            const double t0 = omp_get_wtime();
#pragma omp parallel for schedule(static)
            for (long i = 0; i < N; ++i) a[i] = b[i] + 3.0 * c[i];
            best = std::min(best, omp_get_wtime() - t0);
        }
        std::printf("%d,%.2f\n", t, 3.0 * 8.0 * N / best / 1e9);
    }
    return a[N / 2] == 7.0 ? 0 : 1;  // also keeps the loop from being optimised away
}
