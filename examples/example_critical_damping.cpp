#include <matplot/matplot.h>

#include <print>
#include <vector>

#include "LTI.hpp"

namespace plt = matplot;

void plotStepResponse(const std::vector<double>& time, const std::vector<double>& response,
                      const std::string& title = "Step Response") {
    auto fig = plt::figure();
    plt::plot(time, response);
    plt::xlabel("Time (s)");
    plt::ylabel("Output");
    plt::title(title);
    plt::grid(plt::on);
    plt::show();
}

void plotBodePlot(const control::FrequencyResponse& response, const std::string& title = "Bode Plot") {
    auto fig = plt::figure();
    plt::subplot(2, 1, 1);
    plt::semilogx(response.freq, response.magnitude);
    plt::ylabel("Magnitude (dB)");
    plt::title(title + " - Magnitude");
    plt::grid(plt::on);

    plt::subplot(2, 1, 2);
    plt::semilogx(response.freq, response.phase);
    plt::xlabel("Frequency (Hz)");
    plt::ylabel("Phase (deg)");
    plt::title(title + " - Phase");
    plt::grid(plt::on);

    plt::show();
}

int main() {
    using namespace control;
    using namespace std::literals;

    // Parameters for 2nd order critically damped system
    const double omega_n = 6.0 * 2.0 * std::numbers::pi;  // Natural frequency (rad/s)
    const double zeta    = 0.5;                           // Damping ratio

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
    const double t_start   = 0.0;
    const double t_end     = 3.0 * 5.0 / omega_n;  // View settling plus a bit more
    const double dt        = t_end / 1000.0;       // Time step for ~1000 simulation points
    const int    num_steps = static_cast<int>((t_end - t_start) / dt) + 1;

    // Initialize simulation
    std::vector<double> time(num_steps);
    std::vector<double> response(num_steps);
    Matrix              state = Matrix::Zero(2, 1);  // Initial state [position, velocity] = [0, 0]
    Matrix              input = Matrix::Ones(1, 1);  // Step input

    // Simulate system response
    for (int i = 0; i < num_steps; ++i) {
        time[i]     = i * dt;
        response[i] = sys.output(state, input)(0, 0);  // Get scalar output

        // Evolve state using trapezoidal integration (good for stability)
        state = sys.evolve(state, input, dt, IntegrationMethod::Trapezoidal);
    }

    std::println("Simulation completed:");
    std::println("Final time: {} s", time.back());
    std::println("Final output: {:.4f}", response.back());

    // Generate frequency response
    auto freqResp = sys.generateFrequencyResponse();

    // Plot the step response
    plotStepResponse(time, response, "2nd Order Critically Damped System - Step Response");

    // Plot the Bode plot
    plotBodePlot(freqResp, "2nd Order Critically Damped System - Bode Plot");

    return 0;
}