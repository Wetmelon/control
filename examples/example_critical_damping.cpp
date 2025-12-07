#include <print>
#include <vector>

#include "LTI.hpp"
#include <matplot/matplot.h>

namespace control {

void plotStepResponse(const std::vector<double> &time, const std::vector<double> &response,
                      const std::string &title = "Step Response") {
    using namespace matplot;

    auto fig = figure();
    plot(time, response);
    xlabel("Time (s)");
    ylabel("Output");
    matplot::title(title);
    grid(on);
    show();
}

} // namespace control

int main() {
    using namespace control;
    using namespace std::literals;

    // Parameters for 2nd order critically damped system
    const double omega_n = 2.0; // Natural frequency (rad/s)
    const double zeta = 1.0;    // Damping ratio (critically damped)

    // State space matrices for 2nd order system: ẋ = Ax + Bu, y = Cx + Du
    // A = [0, 1; -ω_n², -2ζω_n]
    // B = [0; ω_n²]
    // C = [1, 0]
    // D = [0]
    const auto A = Matrix{{0.0, 1.0}, {-omega_n * omega_n, -2.0 * zeta * omega_n}};
    const auto B = Matrix{{0.0}, {omega_n * omega_n}};
    const auto C = Matrix{{1.0, 0.0}};
    const auto D = Matrix{{0.0}};

    // Create continuous-time system
    const auto sys = ContinuousStateSpace{A, B, C, D};

    std::println("2nd Order Critically Damped System:");
    std::println("Natural frequency: {} rad/s", omega_n);
    std::println("Damping ratio: {}", zeta);
    std::println("System matrices:");
    std::println("{}", sys);

    // Simulation parameters
    const double t_start = 0.0;
    const double t_end = 5.0;    // 5 seconds
    const double dt = 0.01;      // Time step
    const int num_steps = static_cast<int>((t_end - t_start) / dt) + 1;

    // Initialize simulation
    std::vector<double> time(num_steps);
    std::vector<double> response(num_steps);
    Matrix state = Matrix::Zero(2, 1); // Initial state [position, velocity] = [0, 0]
    Matrix input = Matrix::Ones(1, 1); // Step input

    // Simulate system response
    for (int i = 0; i < num_steps; ++i) {
        time[i] = i * dt;
        response[i] = sys.output(state, input)(0, 0); // Get scalar output

        // Evolve state using trapezoidal integration (good for stability)
        state = sys.evolve(state, input, dt, IntegrationMethod::Trapezoidal);
    }

    std::println("Simulation completed:");
    std::println("Final time: {} s", time.back());
    std::println("Final output: {:.4f}", response.back());

    // Plot the step response
    plotStepResponse(time, response, "2nd Order Critically Damped System - Step Response");

    return 0;
}