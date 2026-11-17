// OSU-style ping-pong between ranks 0 and 1 (put them on different nodes) for the alpha-beta
// model: one-way time t(n) = alpha + n / bandwidth. Prints CSV: bytes, one-way time (us), MB/s.
// Build: mpicxx -O2 bench/mpi_pingpong.cpp -o build-mpi/mpi_pingpong
// Run:   srun --nodes=2 --ntasks=2 --ntasks-per-node=1 build-mpi/mpi_pingpong
#include <mpi.h>

#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    if (nranks != 2) {
        if (rank == 0)
            std::fprintf(stderr, "mpi_pingpong needs exactly 2 ranks\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<char> buf(1 << 24);
    const int peer = 1 - rank;
    if (rank == 0)
        std::printf("bytes,one_way_us,MBps\n");
    for (int n = 8; n <= (1 << 24); n *= 4) {
        const int reps = n <= (1 << 16) ? 10000 : 200, warm = reps / 10;
        double t0 = 0.0;
        for (int r = 0; r < warm + reps; ++r) {
            if (r == warm) {
                MPI_Barrier(MPI_COMM_WORLD);
                t0 = MPI_Wtime();
            }
            if (rank == 0) {
                MPI_Send(buf.data(), n, MPI_CHAR, peer, 0, MPI_COMM_WORLD);
                MPI_Recv(buf.data(), n, MPI_CHAR, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(buf.data(), n, MPI_CHAR, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buf.data(), n, MPI_CHAR, peer, 0, MPI_COMM_WORLD);
            }
        }
        const double one_way = (MPI_Wtime() - t0) / reps / 2;
        if (rank == 0)
            std::printf("%d,%.3f,%.1f\n", n, one_way * 1e6, n / one_way / 1e6);
    }
    MPI_Finalize();
    return 0;
}
