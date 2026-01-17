#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <iostream>
#include <mutex>
#include <plotlypp/figure.hpp>
#include <plotlypp/trace.hpp>
#include <plotlypp/traces/scatter.hpp>
#include <sstream>
#include <thread>
#include <vector>

#include "control.hpp"
#include "types.hpp"

using namespace control;
using namespace plotlypp;

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

void plotSweep(Figure& fig, const SweepData& data) {
    const double fade_alpha = 0.15;

    std::vector<Trace> traces;

    // Bode Magnitude Plot (top)
    for (size_t i = 0; i < data.bode_responses_closed.size(); ++i) {
        const auto&         bode = data.bode_responses_closed[i];
        std::vector<double> freq_vec, mag_vec;
        for (size_t j = 0; j < bode.freq.size(); ++j) {
            freq_vec.push_back(bode.freq[j]);
            mag_vec.push_back(bode.magnitude[j]);
        }

        auto trace = Scatter()
                         .x(freq_vec)
                         .y(mag_vec)
                         .mode({Scatter::Mode::Lines})
                         .line(Scatter::Line().width(1.5).color("rgba(0,0,255,0.15)"))
                         .xaxis("x")
                         .yaxis("y")
                         .showlegend(false);
        traces.push_back(trace);
    }

    // Bode Phase Plot (bottom)
    for (size_t i = 0; i < data.bode_responses_closed.size(); ++i) {
        const auto&         bode = data.bode_responses_closed[i];
        std::vector<double> freq_vec, phase_vec;
        for (size_t j = 0; j < bode.freq.size(); ++j) {
            freq_vec.push_back(bode.freq[j]);
            phase_vec.push_back(bode.phase[j]);
        }

        auto trace = Scatter()
                         .x(freq_vec)
                         .y(phase_vec)
                         .mode({Scatter::Mode::Lines})
                         .line(Scatter::Line().width(1.5).color("rgba(0,0,255,0.15)"))
                         .xaxis("x2")
                         .yaxis("y2")
                         .showlegend(false);
        traces.push_back(trace);
    }

    auto layout = Layout()
                      .title([](auto& t) { t.text("Closed-Loop Bode Diagram"); })
                      .height(800)
                      .width(800)
                      .xaxis(1, Layout::Xaxis().type(Layout::Xaxis::Type::Log).title([](auto& t) { t.text("Frequency (Hz)"); }).showgrid(true))
                      .yaxis(1, Layout::Yaxis().title([](auto& t) { t.text("Magnitude (dB)"); }).showgrid(true))
                      .xaxis(2, Layout::Xaxis().type(Layout::Xaxis::Type::Log).title([](auto& t) { t.text("Frequency (Hz)"); }).showgrid(true))
                      .yaxis(2, Layout::Yaxis().title([](auto& t) { t.text("Phase (degrees)"); }).showgrid(true))
                      .grid(Layout::Grid{}
                                .rows(2)
                                .columns(1)
                                .subplots(std::vector<std::vector<std::string>>{{"xy"}, {"x2y2"}})
                                .roworder(Layout::Grid::Roworder::BottomToTop));

    fig.addTraces(traces);
    fig.setLayout(layout);

    fig.writeHtml("parameter_sweep_bode.html");
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
    Figure fig;

    // Plot sweep
    plotSweep(fig, data);

    return 0;
}
