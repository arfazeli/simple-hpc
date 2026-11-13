#pragma once
#include <cassert>
#include <cstddef>
#include <memory>
#include <new>
#include <utility>
#include <vector>

// Allocator whose value-construction default-initialises (no zeroing), so that a vector of n
// doubles leaves its pages untouched until Field2D's parallel fill below (T37).
template <class T> struct NoInitAlloc : std::allocator<T> {
    template <class U> struct rebind {
        using other = NoInitAlloc<U>;
    };
    NoInitAlloc() = default;
    template <class U> NoInitAlloc(const NoInitAlloc<U>&) noexcept {}
    template <class U> void construct(U* q) noexcept { ::new (static_cast<void*>(q)) U; }
    template <class U, class... A> void construct(U* q, A&&... a) {
        ::new (static_cast<void*>(q)) U(std::forward<A>(a)...);
    }
};

// First touch in parallel (T37): Linux places a page on the NUMA node of the thread that first
// writes it. Rows are split like the kernels' `omp for schedule(static)` loops, which run i (or
// the multigrid row I) descending, so thread 0 gets the top rows. Same values as a serial fill.
// ponytail: ascending-i loops (assemble_p links, correct) still read some remote rows.
inline void first_touch_fill(double* d, int rows, int cols, double v) {
#pragma omp parallel for schedule(static)
    for (int k = 0; k < rows; ++k) {
        double* row = d + static_cast<std::size_t>(rows - 1 - k) * cols;
        for (int j = 0; j < cols; ++j)
            row[j] = v;
    }
}

class Field2D {
  public:
    Field2D() = default;
    // Stores global rows [row0, row0 + rows); indices stay global, so a rank's slab of the grid
    // (owned rows plus halo rows, see decomp.hpp) is addressed exactly like the full grid.
    Field2D(int rows, int cols, double init = 0.0, int row0 = 0)
        : rows_(rows), cols_(cols), row0_(row0), d_(static_cast<std::size_t>(rows) * cols) {
        first_touch_fill(d_.data(), rows, cols, init);
    }

    double& operator()(int i, int j) { return d_[idx(i, j)]; }
    double operator()(int i, int j) const { return d_[idx(i, j)]; }
    int rows() const { return rows_; } // stored rows (the slab), not the global row count
    int row0() const { return row0_; } // first stored global row
    int cols() const { return cols_; }
    double* data() { return d_.data(); }
    const double* data() const { return d_.data(); }
    void fill(double v) { d_.assign(d_.size(), v); }

  private:
    std::size_t idx(int i, int j) const {
        assert(i >= row0_ && i < row0_ + rows_ && j >= 0 && j < cols_);
        return static_cast<std::size_t>(i - row0_) * cols_ + j;
    }
    int rows_ = 0, cols_ = 0, row0_ = 0;
    std::vector<double, NoInitAlloc<double>> d_;
};
