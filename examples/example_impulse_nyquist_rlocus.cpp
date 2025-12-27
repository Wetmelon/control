#include <numbers>

#include "control.hpp"
#include "fmt/core.h"

using namespace control;

int main() {
    fmt::print("=== Impulse, Nyquist, and Root Locus Example ===\n\n");

    // Create a simple second-order system: G(s) = 1/(s^2 + 0.5*s + 1)
    TransferFunction G({1.0}, {1.0, 0.5, 1.0});

    fmt::print("Transfer Function: G(s) = 1/(s^2 + 0.5*s + 1)\n\n");

    // 1. Impulse Response
    fmt::print("1. Computing Impulse Response...\n");
    auto impulse_resp = G.impulse(0.0, 10.0);
    fmt::print("   Computed {} time points\n", impulse_resp.time.size());
    fmt::print("   First few outputs: ");
    for (size_t i = 0; i < std::min(size_t(5), impulse_resp.output.size()); ++i) {
        fmt::print("{:.4f} ", impulse_resp.output[i](0));
    }
    fmt::print("...\n\n");

    // 2. Nyquist Plot Data
    fmt::print("2. Computing Nyquist Plot Data...\n");
    auto nyquist_resp = G.nyquist(0.1, 100.0, 200);
    fmt::print("   Computed {} frequency points\n", nyquist_resp.freq.size());
    fmt::print("   First few complex responses:\n");
    for (size_t i = 0; i < std::min(size_t(5), nyquist_resp.response.size()); ++i) {
        auto h = nyquist_resp.response[i];
        fmt::print("   f={:.2f} Hz: {:.4f} + {:.4f}j (mag={:.4f}, phase={:.2f}°)\n",
                   nyquist_resp.freq[i],
                   h.real(), h.imag(),
                   std::abs(h),
                   std::arg(h) * 180.0 / std::numbers::pi);
    }
    fmt::print("\n");

    // 3. Root Locus
    fmt::print("3. Computing Root Locus (gain 0 to 10)...\n");
    auto rlocus_resp = G.rlocus(0.0, 10.0, 50);
    fmt::print("   Number of pole branches: {}\n", rlocus_resp.branches.size());
    fmt::print("   Number of gain values: {}\n", rlocus_resp.gains.size());

    if (!rlocus_resp.branches.empty()) {
        fmt::print("   Pole movement for first branch:\n");
        for (size_t i = 0; i < std::min(size_t(5), rlocus_resp.gains.size()); ++i) {
            auto pole = rlocus_resp.branches[0][i];
            fmt::print("   k={:.2f}: {:.4f} + {:.4f}j\n",
                       rlocus_resp.gains[i],
                       pole.real(), pole.imag());
        }
    }
    fmt::print("\n");

    // 4. Comparison: Step vs Impulse
    fmt::print("4. Comparing Step and Impulse Responses...\n");
    auto step_resp = G.step(0.0, 10.0);
    fmt::print("   Step response has {} points\n", step_resp.time.size());
    fmt::print("   Impulse response has {} points\n", impulse_resp.time.size());
    fmt::print("   (Impulse is the derivative of step response)\n\n");

    // 5. System stability check
    fmt::print("5. System Stability Analysis...\n");
    auto poles = G.poles();
    fmt::print("   System poles:\n");
    for (size_t i = 0; i < poles.size(); ++i) {
        fmt::print("   Pole {}: {:.4f} + {:.4f}j\n", i + 1, poles[i].real(), poles[i].imag());
    }
    fmt::print("   System is {}\n", G.is_stable() ? "STABLE" : "UNSTABLE");

    fmt::print("\n=== Example Complete ===\n");

    return 0;
}
