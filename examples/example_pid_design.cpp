#include <format>
#include <iostream>

#include "../source/control.hpp"
#include "matplot/matplot.h"

int main() {
    using namespace control;

    std::cout << "=== Advanced Control System Example ===\n";
    std::cout << "Building a PID controller for a second-order plant\n\n";

    // Create a second-order plant: G(s) = 1/(s^2 + 2s + 1)
    // This represents a critically damped system
    StateSpace plant{
        (Matrix(2, 2) << 0, 1, -1, -2).finished(),  // A
        (Matrix(2, 1) << 0, 1).finished(),          // B
        (Matrix(1, 2) << 1, 0).finished(),          // C
        Matrix::Zero(1, 1)                          // D
    };
    std::cout << "Plant G(s) = 1/(s^2 + 2s + 1):\n"
              << std::format("{}\n", plant);

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

    std::cout << "\n=== PID Controller Components ===\n";
    std::cout << "P (Kp = 10):\n"
              << std::format("{}\n", P);
    std::cout << "I (Ki/s, Ki = 5):\n"
              << std::format("{}\n", I);
    std::cout << "D (Kd*s/(tau*s+1), Kd = 2, tau = 0.01):\n"
              << std::format("{}\n", D);

    // Combine PID components: PID = P + I + D
    auto PI  = P + I;
    auto PID = PI + D;

    std::cout << "\n=== Combined PID Controller ===\n";
    std::cout << "PID = P + I + D:\n"
              << std::format("{}\n", PID);

    // Create open-loop system: L(s) = PID(s) * G(s)
    auto open_loop = PID * plant;
    std::cout << "\n=== Open-Loop System ===\n";
    std::cout << "L(s) = PID(s) * G(s):\n"
              << std::format("{}\n", open_loop);

    // Unity feedback sensor
    StateSpace sensor{
        Matrix::Zero(0, 0),          // A
        Matrix::Zero(0, 1),          // B
        Matrix::Zero(1, 0),          // C
        Matrix::Constant(1, 1, 1.0)  // D
    };

    // Create closed-loop system with negative feedback
    auto closed_loop = feedback(open_loop, sensor, -1);
    std::cout << "\n=== Closed-Loop System ===\n";
    std::cout << "T(s) = L(s) / (1 + L(s)):\n";
    std::cout << "System order: " << closed_loop.A.rows() << "\n";
    std::cout << std::format("{}\n", closed_loop);

    // Test step response
    std::cout << "\n=== Step Response Analysis ===\n";
    auto step_resp = closed_loop.step(0.0, 2.0);

    // Find steady-state value and settling time
    double steady_state       = step_resp.output.back();
    double settling_threshold = 0.02 * steady_state;  // 2% criterion
    double settling_time      = 0.0;

    for (size_t i = step_resp.output.size() - 1; i > 0; --i) {
        if (std::abs(step_resp.output[i] - steady_state) > settling_threshold) {
            settling_time = step_resp.time[i];
            break;
        }
    }

    // Find peak overshoot
    double peak      = *std::max_element(step_resp.output.begin(), step_resp.output.end());
    double overshoot = ((peak - steady_state) / steady_state) * 100.0;

    std::cout << "Steady-state value: " << std::format("{:.4f}", steady_state) << "\n";
    std::cout << "Peak overshoot: " << std::format("{:.2f}%", overshoot) << "\n";
    std::cout << "Settling time (2%): " << std::format("{:.3f}s", settling_time) << "\n";

    std::cout << "\nFirst few step response values:\n";
    for (size_t i = 0; i < std::min(size_t(15), step_resp.time.size()); ++i) {
        std::cout << std::format("  t={:.4f}s, y={:.4f}\n", step_resp.time[i], step_resp.output[i]);
    }

    constexpr double fmin       = 0.01;   // Minimum frequency for Bode plot
    constexpr double fmax       = 100.0;  // Maximum frequency for Bode plot
    constexpr double num_points = 1000;   // Number of frequency points

    // Test frequency response
    std::cout << "\n=== Bode Plot Analysis ===\n";
    const auto bode_open   = open_loop.bode(fmin, fmax, num_points);
    const auto bode_closed = closed_loop.bode(fmin, fmax, num_points);

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

    std::cout << "Closed-loop bandwidth (-3dB): " << std::format("{:.2f} Hz", bandwidth) << "\n";
    std::cout << "Closed-loop bandwidth: " << std::format("{:.2f} rad/s", bandwidth * 2.0 * 3.14159) << "\n";

    // Create figure for Bode plots
    auto fig_mag = matplot::figure(true);
    fig_mag->size(1200, 800);

    // Create magnitude plot
    auto ax1 = fig_mag->add_subplot(2, 1, 0);
    ax1->semilogx(bode_open.freq, bode_open.magnitude, "b-")->line_width(2).display_name("Open-Loop L(s)");
    ax1->hold(matplot::on);
    ax1->semilogx(bode_closed.freq, bode_closed.magnitude, "r-")->line_width(2).display_name("Closed-Loop T(s)");
    ax1->grid(matplot::on);
    ax1->ylabel("Magnitude (dB)");
    ax1->title("Bode Plot: Open-Loop vs Closed-Loop");
    ax1->legend();

    // Create phase plot
    auto ax2 = fig_mag->add_subplot(2, 1, 1);
    ax2->semilogx(bode_open.freq, bode_open.phase, "b-")->line_width(2).display_name("Open-Loop L(s)");
    ax2->hold(matplot::on);
    ax2->semilogx(bode_closed.freq, bode_closed.phase, "r-")->line_width(2).display_name("Closed-Loop T(s)");
    ax2->grid(matplot::on);
    ax2->xlabel("Frequency (Hz)");
    ax2->ylabel("Phase (deg)");
    ax2->legend();

    matplot::show();
    std::cout << "Bode plots displayed.\n";

    std::cout << "\n=== Analysis Complete ===\n";
    return 0;
}
