#include <fmt/core.h>
#include <fmt/ranges.h>

#include "control.hpp"

using namespace control;

int main() {
    fmt::print("=== Zero-Pole-Gain (ZPK) Representation Example ===\n\n");

    // Example 1: Create a simple system using ZPK
    // G(s) = 2 * (s + 1) / [(s + 2)(s + 3)]
    // Zeros: -1
    // Poles: -2, -3
    // Gain: 2
    fmt::print("1. Creating system from ZPK representation:\n");
    fmt::print("   G(s) = 2 * (s + 1) / [(s + 2)(s + 3)]\n");

    std::vector<Zero> zeros = {1.0};
    std::vector<Pole> poles = {2.0, 3.0};

    auto G    = zpk(zeros, poles, 2.0);
    auto G_tf = tf(G);  // Convert to TF to print num/den

    fmt::print("   Numerator:   [{}]\n", fmt::join(G_tf.num, ", "));
    fmt::print("   Denominator: [{}]\n", fmt::join(G_tf.den, ", "));
    fmt::print("\n");

    // Example 2: Extract zeros and poles from a transfer function
    fmt::print("2. Creating transfer function and extracting ZPK:\n");
    TransferFunction H({1.0, 5.0, 6.0}, {1.0, 4.0, 3.0});  // (s^2 + 5s + 6) / (s^2 + 4s + 3)

    fmt::print("   H(s) = (s^2 + 5s + 6) / (s^2 + 4s + 3)\n");
    fmt::print("   Numerator: [{}], Denominator: [{}]\n", fmt::join(H.num, ", "), fmt::join(H.den, ", "));

    auto h_poles = H.poles();
    fmt::print("   Poles:\n");
    for (int i = 0; i < static_cast<int>(h_poles.size()); ++i) {
        fmt::print("      p{} = {:.4f} + {:.4f}j\n", i + 1, h_poles[i].real(), h_poles[i].imag());
    }

    auto h_zeros = H.zeros();
    fmt::print("   Zeros:\n");
    for (int i = 0; i < static_cast<int>(h_zeros.size()); ++i) {
        fmt::print("      z{} = {:.4f} + {:.4f}j\n", i + 1, h_zeros[i].real(), h_zeros[i].imag());
    }
    fmt::print("\n");

    // Example 3: Second-order system with complex poles
    fmt::print("3. Second-order system with complex conjugate poles:\n");
    fmt::print("   G(s) = 10 / [(s + 1 + 2j)(s + 1 - 2j)]\n");

    std::vector<Zero> zeros_empty;  // No zeros

    std::vector<Pole> poles_complex = {
        std::complex<double>(-1.0, 2.0),
        std::complex<double>(-1.0, -2.0)};
    auto G2    = zpk(zeros_empty, poles_complex, 10.0);
    auto G2_tf = tf(G2);  // Convert to TF to print num/den

    fmt::print("   Numerator:   [{}]\n", fmt::join(G2_tf.num, ", "));
    fmt::print("   Denominator: [{}]\n", fmt::join(G2_tf.den, ", "));
    fmt::print("   Is stable: {}\n", G2.is_stable() ? "Yes" : "No");
    fmt::print("\n");

    // Example 4: Extracting zeros from state-space
    fmt::print("4. State-space system zeros extraction:\n");
    // Create a simple SISO state-space system
    Matrix A = Matrix{{-3, 1}, {-2, 0}};
    Matrix B = Matrix{{0}, {1}};
    Matrix C = Matrix{{1, 0}};
    Matrix D = Matrix{{0}};

    StateSpace sys(A, B, C, D);

    auto sys_zeros = sys.zeros();
    auto sys_poles = sys.poles();

    fmt::print("   System poles:\n");
    for (int i = 0; i < static_cast<int>(sys_poles.size()); ++i) {
        fmt::print("      p{} = {:.4f} + {:.4f}j\n", i + 1, sys_poles[i].real(), sys_poles[i].imag());
    }

    fmt::print("   System zeros:\n");
    for (int i = 0; i < static_cast<int>(sys_zeros.size()); ++i) {
        fmt::print("      z{} = {:.4f} + {:.4f}j\n", i + 1, sys_zeros[i].real(), sys_zeros[i].imag());
    }
    fmt::print("\n");

    // Example 5: Discrete-time system in ZPK form
    fmt::print("5. Discrete-time system from ZPK:\n");
    fmt::print("   G(z) = 0.5 * (z - 0.5) / [(z - 0.8)(z - 0.9)]\n");

    std::vector<Zero> z_zeros = {0.5};
    std::vector<Pole> z_poles = {0.8, 0.9};

    auto G_discrete    = zpk(z_zeros, z_poles, 0.5, 0.1);  // Ts = 0.1
    auto G_discrete_tf = tf(G_discrete);                   // Convert to TF to print num/den

    fmt::print("   Numerator:   [{}]\n", fmt::join(G_discrete_tf.num, ", "));
    fmt::print("   Denominator: [{}]\n", fmt::join(G_discrete_tf.den, ", "));
    fmt::print("   Sampling time: {:.3f} s\n", *G_discrete.Ts);
    fmt::print("   Is stable: {}\n", G_discrete.is_stable() ? "Yes" : "No");
    fmt::print("\n");

    fmt::print("=== ZPK Example Complete ===\n");
    return 0;
}
