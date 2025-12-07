#include <matplot/matplot.h>

#include <numbers>
#include <print>
#include <vector>

#include "LTI.hpp"
#include "matplot/freestanding/axes_functions.h"

namespace plt = matplot;

void plotStepResponse(const std::vector<double>& time, const std::vector<std::vector<double>>& responses,
                      const std::vector<std::string>& labels, const std::string& title = "Step Response") {
    auto fig = plt::figure(true);
    fig->size(800, 600);
    plt::hold(plt::on);

    for (size_t i = 0; i < responses.size(); ++i) {
        plt::plot(time, responses[i])->display_name(labels[i]);
    }
    plt::xlabel("Time (s)");
    plt::ylabel("Output");
    plt::title(title);
    plt::grid(plt::on);
    plt::legend()->font_size(10);
    plt::hold(plt::off);
    plt::show();
}

void plotBodePlot(const std::vector<control::FrequencyResponse>& responses, const std::vector<std::string>& labels, const std::string& title = "Bode Plot") {
    auto fig = plt::figure(true);
    fig->size(800, 600);

    plt::subplot(2, 1, 1);
    plt::hold(plt::on);
    for (size_t i = 0; i < responses.size(); ++i) {
        plt::semilogx(responses[i].freq, responses[i].magnitude)->display_name(labels[i]);
    }
    plt::ylabel("Magnitude (dB)");
    plt::title(title + " - Magnitude");
    plt::grid(plt::on);
    plt::legend()->location(plt::legend::general_alignment::bottomleft);
    plt::hold(plt::off);

    plt::subplot(2, 1, 2);
    plt::hold(plt::on);
    for (size_t i = 0; i < responses.size(); ++i) {
        plt::semilogx(responses[i].freq, responses[i].phase)->display_name(labels[i]);
    }
    plt::xlabel("Frequency (Hz)");
    plt::ylabel("Phase (deg)");
    plt::title(title + " - Phase");
    plt::grid(plt::on);
    plt::legend()->location(plt::legend::general_alignment::bottomleft);
    plt::hold(plt::off);
    plt::show();
}

int main() {
    using namespace control;
    using namespace std::literals;

    // Parameters for 2nd order system
    const double              omega_n = 6.0 * 2.0 * std::numbers::pi;  // Natural frequency (rad/s)
    const std::vector<double> zetas   = {0.3, 0.5, std::numbers::sqrt2 / 2.0, 1.0, 2.0};

    // Find minimum zeta for settling time calculation
    double min_zeta = *std::min_element(zetas.begin(), zetas.end());

    // Simulation parameters
    const double t_start   = 0.0;
    const double t_end     = 1.5 * 5.0 / (min_zeta * omega_n);  // Settling time for smallest zeta
    const double dt        = t_end / 1000.0;                    // Time step for ~1000 simulation points
    const int    num_steps = static_cast<int>((t_end - t_start) / dt) + 1;

    // Initialize time vector (will be set from generateStepResponse)
    std::vector<double> time;

    // Simulate for each damping ratio
    std::vector<std::vector<double>>        stepResponses;
    std::vector<control::FrequencyResponse> freqResponses;

    std::vector<std::string> labels;
    for (double zeta : zetas) {
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
        const auto sys = StateSpace{A, B, C, D};

        // Generate step response
        auto stepResp = sys.generateStepResponse(t_start, t_end, num_steps);
        if (stepResponses.empty()) time = stepResp.time;
        stepResponses.push_back(stepResp.output);
        labels.push_back(std::format("ζ = {:.1f}", zeta));
        freqResponses.push_back(sys.generateFrequencyResponse());
    }

    // Plot all step responses and Bode plots
    plotStepResponse(time, stepResponses, labels, "Step Response for Different Damping Ratios");
    plotBodePlot(freqResponses, labels, "Bode Plot for Different Damping Ratios");

    return 0;
}