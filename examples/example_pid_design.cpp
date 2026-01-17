#include <fmt/core.h>

#include <plotlypp/figure.hpp>
#include <plotlypp/trace.hpp>
#include <plotlypp/traces/scatter.hpp>

#include "control.hpp"

int main() {
    using namespace control;
    using namespace plotlypp;

    fmt::print("=== Advanced Control System Example ===\n");
    fmt::print("Building a PID controller for a second-order plant\n\n");

    // Create a second-order plant: G(s) = 1/(s^2 + 2s + 1)
    // This represents a critically damped system
    StateSpace plant{
        Matrix{{0.0, 1.0}, {-1.0, -2.0}},  // A
        Matrix{{0.0}, {1.0}},              // B
        Matrix{{1.0, 0.0}},                // C
        Matrix::Zero(1, 1)                 // D
    };
    fmt::print("Plant G(s) = 1/(s^2 + 2s + 1):\n{}", plant);

    // Create PID controller components
    // P: Proportional gain Kp = 10
    StateSpace P{
        Matrix::Zero(0, 0),           // A
        Matrix::Zero(0, 1),           // B
        Matrix::Zero(1, 0),           // C
        Matrix::Constant(1, 1, 10.0)  // D
    };

    // I: Integral gain Ki/s where Ki = 5
    // State-space: A=0, B=1, C=5, D=0
    StateSpace I{
        Matrix::Constant(1, 1, 0.0),  // A
        Matrix::Constant(1, 1, 1.0),  // B
        Matrix::Constant(1, 1, 5.0),  // C
        Matrix::Zero(1, 1)            // D
    };

    // D: Derivative gain Kd*s where Kd = 2
    // For implementation, use: Kd*s/(tau*s + 1) with tau = 0.01 (filtered derivative)
    StateSpace D{
        Matrix::Constant(1, 1, -100.0),  // A = -1/tau
        Matrix::Constant(1, 1, 1.0),     // B
        Matrix::Constant(1, 1, 200.0),   // C = Kd/tau
        Matrix::Zero(1, 1)               // D
    };

    fmt::print("\n=== PID Controller Components ===\n");
    fmt::print("P (Kp = 10):\n{}\n", P);
    fmt::print("I (Ki/s, Ki = 5):\n{}\n", I);
    fmt::print("D (Kd*s/(tau*s+1), Kd = 2, tau = 0.01):\n{}\n", D);

    // Combine PID components: PID = P + I + D
    auto PI  = P + I;
    auto PID = PI + D;

    fmt::print("\n=== Combined PID Controller ===\n");
    fmt::print("PID = P + I + D:\n{}\n", PID);

    // Create open-loop system: L(s) = PID(s) * G(s)
    auto open_loop = PID * plant;
    fmt::print("\n=== Open-Loop System ===\n");
    fmt::print("L(s) = PID(s) * G(s):\n{}\n", open_loop);

    // Unity feedback sensor
    StateSpace sensor{
        Matrix::Zero(0, 0),          // A
        Matrix::Zero(0, 1),          // B
        Matrix::Zero(1, 0),          // C
        Matrix::Constant(1, 1, 1.0)  // D
    };

    // Create closed-loop system with negative feedback
    auto closed_loop = feedback(open_loop, sensor, -1);
    fmt::print("\n=== Closed-Loop System ===\n");
    fmt::print("T(s) = L(s) / (1 + L(s)):\n");
    fmt::print("System order: {}\n", closed_loop.A.rows());
    fmt::print("{}\n", closed_loop);

    // Test step response
    fmt::print("\n=== Step Response Analysis ===\n");
    auto step_resp = closed_loop.step(0.0, 2.0);

    // Find steady-state value and settling time
    double steady_state       = step_resp.output.back()(0, 0);
    double settling_threshold = 0.02 * std::abs(steady_state);  // 2% criterion
    size_t settling_idx       = 0;
    for (size_t i = 0; i < step_resp.output.size(); ++i) {
        if (std::abs(step_resp.output[i](0, 0) - steady_state) > settling_threshold) {
            settling_idx = i;
        }
    }
    double settling_time = step_resp.time[settling_idx];

    // Find peak overshoot
    // Find peak value
    double peak = 0.0;
    for (const auto& y : step_resp.output) {
        peak = std::max(peak, y(0, 0));
    }
    double overshoot = ((peak - steady_state) / steady_state) * 100.0;

    fmt::print("Steady-state value: {:.4f}\n", steady_state);
    fmt::print("Peak overshoot: {:.2f}%\n", overshoot);
    fmt::print("Settling time (2%): {:.3f}s\n", settling_time);

    fmt::print("\nFirst few step response values:\n");
    for (size_t i = 0; i < std::min(size_t(15), step_resp.time.size()); ++i) {
        fmt::print("  t={:.4f}s, y={:.4f}\n", step_resp.time[i], step_resp.output[i](0, 0));
    }

    constexpr double fmin = 0.01;   // Minimum frequency for Bode plot
    constexpr double fmax = 100.0;  // Maximum frequency for Bode plot

    // Test frequency response (uses adaptive sampling)
    fmt::print("\n=== Bode Plot Analysis ===\n");
    const auto bode_open   = open_loop.bode(fmin, fmax);
    const auto bode_closed = closed_loop.bode(fmin, fmax);

    // Find -3dB bandwidth
    double bandwidth = 0.0;
    for (size_t i = 0; i < bode_closed.magnitude.size(); ++i) {
        if (bode_closed.magnitude[i] < -3.0) {
            if (i > 0) {
                // Linear interpolation
                double f1 = bode_closed.freq[i - 1];
                double f2 = bode_closed.freq[i];
                double m1 = bode_closed.magnitude[i - 1];
                double m2 = bode_closed.magnitude[i];
                bandwidth = f1 + (f2 - f1) * (-3.0 - m1) / (m2 - m1);
            } else {
                bandwidth = bode_closed.freq[i];
            }
            break;
        }
    }

    fmt::print("Closed-loop bandwidth (-3dB): {:.2f} Hz\n", bandwidth);
    fmt::print("Closed-loop bandwidth: {:.2f} rad/s\n", bandwidth * 2.0 * 3.14159);

    // Create figure for Bode plots
    Figure fig;

    // Magnitude plot
    auto trace_open_mag = Scatter()
                              .x(bode_open.freq)
                              .y(bode_open.magnitude)
                              .mode({Scatter::Mode::Lines})
                              .name("Open-Loop L(s)")
                              .line(Scatter::Line().width(2).color("blue"))
                              .xaxis("x")
                              .yaxis("y");

    auto trace_closed_mag = Scatter()
                                .x(bode_closed.freq)
                                .y(bode_closed.magnitude)
                                .mode({Scatter::Mode::Lines})
                                .name("Closed-Loop T(s)")
                                .line(Scatter::Line().width(2).color("red"))
                                .xaxis("x")
                                .yaxis("y");

    // Phase plot
    auto trace_open_ph = Scatter()
                             .x(bode_open.freq)
                             .y(bode_open.phase)
                             .mode({Scatter::Mode::Lines})
                             .name("Open-Loop L(s)")
                             .line(Scatter::Line().width(2).color("blue"))
                             .xaxis("x2")
                             .yaxis("y2");

    auto trace_closed_ph = Scatter()
                               .x(bode_closed.freq)
                               .y(bode_closed.phase)
                               .mode({Scatter::Mode::Lines})
                               .name("Closed-Loop T(s)")
                               .line(Scatter::Line().width(2).color("red"))
                               .xaxis("x2")
                               .yaxis("y2");

    auto layout = Layout()
                      .title([](auto& t) { t.text("Bode Plot: Open-Loop vs Closed-Loop"); })
                      .height(800)
                      .width(1200)
                      .xaxis(1, Layout::Xaxis().type(Layout::Xaxis::Type::Log).title([](auto& t) { t.text("Frequency (Hz)"); }).showgrid(true))
                      .yaxis(1, Layout::Yaxis().title([](auto& t) { t.text("Magnitude (dB)"); }).showgrid(true))
                      .xaxis(2, Layout::Xaxis().type(Layout::Xaxis::Type::Log).title([](auto& t) { t.text("Frequency (Hz)"); }).showgrid(true))
                      .yaxis(2, Layout::Yaxis().title([](auto& t) { t.text("Phase (deg)"); }).showgrid(true))
                      .grid(Layout::Grid{}
                                .rows(2)
                                .columns(1)
                                .subplots(std::vector<std::vector<std::string>>{{"xy"}, {"x2y2"}})
                                .roworder(Layout::Grid::Roworder::BottomToTop));

    fig.addTraces(std::vector<Trace>{trace_open_mag, trace_closed_mag, trace_open_ph, trace_closed_ph});
    fig.setLayout(layout);

    fig.writeHtml("pid_design_bode_plots.html");
    fmt::print("Bode plots saved to pid_design_bode_plots.html\n");

    fmt::print("\n=== Analysis Complete ===\n");
    return 0;
}
