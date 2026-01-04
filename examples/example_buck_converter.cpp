
#include <fmt/core.h>
#include <fmt/ranges.h>

#include "control_design.hpp"
#include "lqr.hpp"
#include "state_space.hpp"

using namespace wetmelon::control;

int main() {
    constexpr double L = 20e-6;           // Inductance in Henrys
    constexpr double C = 14.7e-6;         // Capacitance in Farads
    constexpr double R = 0.01;            // Series resistance in Ohms
    constexpr double Ts = 1.0 / 128000.0; // Sampling time in seconds

    constexpr ColVec x0 = {0.0, 0.0}; // Initial state: [inductor current; capacitor voltage]
    ColVec           u = {0.0};       // Input: [input voltage]

    constexpr StateSpace sys = {
        .A = Matrix<2, 2>{
            {-R / L, -1.0 / L},
            {1.0 / C, 0.0},
        },
        .B = Matrix<2, 1>{
            {1.0 / L},
            {0.0},
        },
        .C = Matrix<1, 2>{
            {1.0, 1.0},
        },
    };

    // State cost matrix (Penalize state deviations)
    constexpr Matrix Q = Matrix<2, 2>::identity();

    constexpr Matrix R_mat = Matrix<1, 1>{{0.01}}; // Control effort cost (smaller = more aggressive)

    // Design discrete-time LQR controller from continuous-time model
    constexpr auto result = design::discrete_lqr_from_continuous(sys, Q, R_mat, Ts).as<float>();

    LQR controller = result;

    // Print closed loop poles
    fmt::println("\nClosed Loop Poles:");
    for (size_t i = 0; i < result.e.size(); ++i) {
        fmt::println("{:.4f} + {:.4f}j", result.e[i].real(), result.e[i].imag());
    }

    if (result.is_stable()) {
        fmt::println("The closed-loop system is stable.");
    } else {
        fmt::println("The closed-loop system is unstable.");
    }

    // Print the resulting gain matrix
    fmt::println("\nLQR Gain Matrix K:");
    for (size_t i = 0; i < result.K.rows(); ++i) {
        for (size_t j = 0; j < result.K.cols(); ++j) {
            fmt::print("{:.4f} ", result.K(i, j));
        }
        fmt::println("");
    }

    return 0;
}