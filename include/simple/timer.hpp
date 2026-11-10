#pragma once
#include <array>
#include <chrono>
#include <cstddef>

enum class Phase {
    AssembleU,
    BcU,
    ResidualU,
    SweepU,
    AssembleV,
    BcV,
    ResidualV,
    SweepV,
    AssembleP,
    SolveP,
    Correct,
    Output,
    Count
};

inline const char* phase_name(Phase ph) {
    switch (ph) {
    case Phase::AssembleU: return "assemble_u";
    case Phase::BcU: return "bc_u";
    case Phase::ResidualU: return "residual_u";
    case Phase::SweepU: return "sweep_u";
    case Phase::AssembleV: return "assemble_v";
    case Phase::BcV: return "bc_v";
    case Phase::ResidualV: return "residual_v";
    case Phase::SweepV: return "sweep_v";
    case Phase::AssembleP: return "assemble_p";
    case Phase::SolveP: return "solve_p";
    case Phase::Correct: return "correct";
    case Phase::Output: return "output";
    case Phase::Count: break;
    }
    return "?";
}

struct PhaseTimers {
    std::array<double, static_cast<std::size_t>(Phase::Count)> seconds{};
};

// RAII: adds the elapsed wall time to timers.seconds[phase] on destruction.
class ScopedTimer {
public:
    ScopedTimer(PhaseTimers& t, Phase ph)
        : t_(t), ph_(ph), t0_(std::chrono::steady_clock::now()) {}
    ~ScopedTimer() {
        t_.seconds[static_cast<std::size_t>(ph_)] +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
    }
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    PhaseTimers& t_;
    Phase ph_;
    std::chrono::steady_clock::time_point t0_;
};
