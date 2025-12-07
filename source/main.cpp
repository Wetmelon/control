
#include <print>

#include "LTI.hpp"
#include <matplot/matplot.h>

namespace control {

void plotFrequencyResponse(const FrequencyResponse &response) {
    using namespace matplot;

    // Create a tiled layout with 2 rows: magnitude and phase
    tiledlayout(2, 1);

    // Plot magnitude (top subplot)
    auto ax1 = nexttile();
    semilogx(ax1, response.freq, response.magnitude);
    ax1->xlabel("Frequency (Hz)");
    ax1->ylabel("Magnitude (dB)");
    ax1->title("Frequency Response - Magnitude");
    ax1->grid(on);

    // Plot phase (bottom subplot)
    auto ax2 = nexttile();
    semilogx(ax2, response.freq, response.phase);
    ax2->xlabel("Frequency (Hz)");
    ax2->ylabel("Phase (degrees)");
    ax2->title("Frequency Response - Phase");
    ax2->grid(on);

    show();
}
} // namespace control

int main() {
    using namespace control;
    using namespace std::literals;

    constexpr auto m = 250.0; // system mass
    constexpr auto k = 40.0;  // spring constant
    constexpr auto b = 60.0;  // damping constant

    // Format styled after scipy.signal.StateSpace examples
    const auto A = Matrix{{0, 1.0}, {-k / m, -b / m}};
    const auto B = Matrix{{0}, {1.0 / m}};
    const auto C = Matrix{{1.0, 0}};
    const auto D = Matrix{{0.0}};

    const auto sys = ContinuousStateSpace{A, B, C, D}; // Continuous time system
    std::println("Continuous time matrix sys: \n{}", sys);

    // Test integration methods on continuous system
    Matrix       x0 = Matrix::Zero(2, 1); // initial state
    Matrix       u  = Matrix::Ones(1, 1); // constant input
    const double h  = 0.01;               // time step

    auto x_fe  = sys.evolve(x0, u, h, IntegrationMethod::ForwardEuler);
    auto x_be  = sys.evolve(x0, u, h, IntegrationMethod::BackwardEuler);
    auto x_tr  = sys.evolve(x0, u, h, IntegrationMethod::Trapezoidal);
    auto x_gen = sys.evolve(x0, u, h, IntegrationMethod::Trapezoidal);

    std::println("Initial state:");
    std::cout << x0.transpose() << std::endl;
    std::println("Forward Euler after {}s:", h);
    std::cout << x_fe.transpose() << std::endl;
    std::println("Backward Euler after {}s:", h);
    std::cout << x_be.transpose() << std::endl;
    std::println("Trapezoidal after {}s:", h);
    std::cout << x_tr.transpose() << std::endl;
    std::println("General evolve (Trapezoidal) after {}s:", h);
    std::cout << x_gen.transpose() << std::endl;

    const auto sysd = sys.c2d(0.01, Method::Tustin);
    std::println("Discrete time matrix sysd: \n{}", sysd);

    // Test step method on discrete system
    Matrix x0_d = Matrix::Zero(2, 1); // initial state for discrete system
    Matrix u_d  = Matrix::Ones(1, 1); // input for discrete system

    auto x_next_d = sysd.step(x0_d, u_d);
    std::println("Discrete system step:");
    std::cout << "Initial state: " << x0_d.transpose() << std::endl;
    std::cout << "Next state: " << x_next_d.transpose() << std::endl;

    return 0;
}