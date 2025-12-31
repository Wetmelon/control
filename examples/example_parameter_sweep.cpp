#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#include "control.hpp"
#include "matplot/matplot.h"
#include "types.hpp"

using namespace control;
using namespace matplot;

// Create two-mass plant with flexible coupling
// System: Motor mass (m1) connected to load mass (m2) via flexible shaft (spring k, damper c)
// Input: motor force, Output: load position
StateSpace createPlant(double load_mass) {
    // Fixed system parameters
    const double m1 = 1.0;        // Motor mass (kg)
    const double k  = 1000.0;     // Shaft stiffness (N/m)
    const double c  = 10.0;       // Shaft damping (N·s/m)
    const double m2 = load_mass;  // Variable load mass (kg)

    // Open-loop plant state-space
    // States: [x1, v1, x2, v2] where x1,v1 = motor position/velocity, x2,v2 = load position/velocity
    // Input: motor force
    // Output: load position (x2)

    const Matrix A{
        {0, 1, 0, 0},
        {-k / m1, -c / m1, k / m1, c / m1},
        {0, 0, 0, 1},
        {k / m2, c / m2, -k / m2, -c / m2},
    };

    const auto B = ColVec({0, 1.0 / m1, 0, 0});
    const auto C = RowVec({0, 0, 1, 0});
    const auto D = RowVec({0});

    return StateSpace(A, B, C, D);
}

// Create PI controller (PID with proportional on measurement to avoid derivative kick)
StateSpace createController() {
    const double Kp = 500.0;  // Proportional gain
    const double Ki = 50.0;   // Integral gain

    // PI controller state-space
    // State: integral of error
    // Input: error (reference - measurement)
    // Output: control signal

    const auto A = RowVec({0});
    const auto B = RowVec({1});
    const auto C = RowVec({Ki});
    const auto D = RowVec({Kp});

    return StateSpace(A, B, C, D);
}

// Create closed-loop system using LTI operators
StateSpace createClosedLoopSystem(double load_mass) {
    auto plant      = createPlant(load_mass);
    auto controller = createController();
    auto sensor     = StateSpace{
        Matrix::Zero(0, 0),  // A
        Matrix::Zero(0, 1),  // B
        Matrix::Zero(1, 0),  // C
        RowVec({1.0})        // D
    };

    // Closed-loop: feedback(controller * plant, I)
    auto open_loop = controller * plant;
    return feedback(open_loop, sensor, -1);
}

struct SweepData {
    std::vector<double>       param_values;
    std::vector<BodeResponse> bode_responses_closed;
};

void plotSweep(figure_handle fig, const SweepData& data) {
    const double fade_alpha = 0.15;

    figure(fig);

    // Bode Magnitude Plot (top)
    subplot(2, 1, 0);
    auto ax3 = gca();
    ax3->clear();
    hold(on);

    for (size_t i = 0; i < data.bode_responses_closed.size(); ++i) {
        const auto&         bode = data.bode_responses_closed[i];
        std::vector<double> freq_vec, mag_vec;
        for (size_t j = 0; j < bode.freq.size(); ++j) {
            freq_vec.push_back(bode.freq[j]);
            mag_vec.push_back(bode.magnitude[j]);
        }

        auto p = plot(freq_vec, mag_vec, "-b");
        p->line_width(1.5);
        p->color({0.0f, 0.0f, 1.0f, static_cast<float>(fade_alpha)});
    }

    hold(off);
    xlabel("Frequency (Hz)");
    ylabel("Magnitude (dB)");
    title("Closed-Loop Bode - Magnitude");
    grid(on);
    ax3->x_axis().scale(axis_type::axis_scale::log);

    // Bode Phase Plot (bottom)
    subplot(2, 1, 1);
    auto ax4 = gca();
    ax4->clear();
    hold(on);

    for (size_t i = 0; i < data.bode_responses_closed.size(); ++i) {
        const auto&         bode = data.bode_responses_closed[i];
        std::vector<double> freq_vec, phase_vec;
        for (size_t j = 0; j < bode.freq.size(); ++j) {
            freq_vec.push_back(bode.freq[j]);
            phase_vec.push_back(bode.phase[j]);
        }

        auto p = plot(freq_vec, phase_vec, "-b");
        p->line_width(1.5);
        p->color({0.0f, 0.0f, 1.0f, static_cast<float>(fade_alpha)});
    }

    hold(off);
    xlabel("Frequency (Hz)");
    ylabel("Phase (degrees)");
    title("Closed-Loop Bode - Phase");
    grid(on);
    ax4->x_axis().scale(axis_type::axis_scale::log);

    show();
}

int main() {
    fmt::print("=== Parameter Sweep Plotter ===\n\n");
    fmt::print("Closed-loop position control system with variable load mass\n");
    fmt::print("PID controller tuned for nominal mass of 1.0 kg\n\n");

    // Get mass values from user
    fmt::print("Enter mass values (in kg) separated by spaces.\n");
    fmt::print("Example: 0.3 0.5 0.7 1.0 1.5 2.5 5.0 10.0\n");
    fmt::print("Mass values: ");
    std::vector<double> masses;
    std::string         line;
    std::getline(std::cin, line);
    std::istringstream iss(line);
    double             value;
    while (iss >> value) {
        if (value > 0) {
            masses.push_back(value);
        } else {
            fmt::print("Ignoring invalid mass: {}\n", value);
        }
    }

    if (masses.empty()) {
        fmt::print("No valid masses provided. Using default values.\n");
        masses = {0.3, 0.5, 0.7, 1.0, 1.5, 2.5, 5.0, 10.0};
    }

    const size_t num_params = masses.size();
    fmt::print("\nGenerating sweep for {} mass values...\n", num_params);
    SweepData data;
    data.param_values = masses;
    data.bode_responses_closed.resize(num_params);

    auto total_start = std::chrono::high_resolution_clock::now();

    // Launch parallel computations
    std::vector<std::future<void>> futures;
    std::mutex                     cout_mutex;

    for (size_t i = 0; i < num_params; ++i) {
        futures.push_back(std::async(std::launch::async, [&, i]() {
            const double mass       = masses[i];
            auto         iter_start = std::chrono::high_resolution_clock::now();

            auto sys = createClosedLoopSystem(mass);

            // Closed-loop Bode response (adaptive frequency sampling)
            auto bode_start  = std::chrono::high_resolution_clock::now();
            auto bode_closed = sys.bode(0.1, 100.0);
            auto bode_end    = std::chrono::high_resolution_clock::now();

            data.bode_responses_closed[i] = bode_closed;

            auto iter_end      = std::chrono::high_resolution_clock::now();
            auto iter_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(iter_end - iter_start);
            auto bode_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(bode_end - bode_start);

            std::lock_guard<std::mutex> lock(cout_mutex);
            fmt::print("  m = {:5.2f} kg | Points: {:>4} | Bode: {:>8.3f} ms | Total: {:>8.3f} ms\n",
                       mass,
                       bode_closed.freq.size(),
                       bode_duration.count() / 1e6,
                       iter_duration.count() / 1e6);
        }));
    }

    // Wait for all computations to complete
    for (auto& future : futures) {
        future.get();
    }

    auto total_end      = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(total_end - total_start);

    fmt::print("\nTotal computation time: {:.3f} ms ({:.6f} s)\n\n",
               total_duration.count() / 1e6,
               total_duration.count() / 1e9);
    fmt::print("\nDone!\n\n");

    // Create figure
    auto fig = figure(true);
    fig->size(800, 800);
    fig->name("Closed-Loop Bode Diagram");

    // Plot sweep
    plotSweep(fig, data);

    return 0;
}
