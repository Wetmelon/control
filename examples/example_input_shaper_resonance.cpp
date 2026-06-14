#include <cmath>
#include <numbers>
#include <vector>

#include "fmt/core.h"
#include "wet/filters/spectral.hpp"
#include "wet/plotting/plot_plotly.hpp"
#include "wet/trajectory/input_shaper.hpp"

using namespace wet;

// ===== Klipper-style Input-Shaper Calibration from an Accelerometer =====
//
// A 3D-printer X gantry behaves like a lightly-damped second-order mode. Mount an
// accelerometer on the toolhead, run a resonance test, find the peak, and design
// an input shaper for it — the same flow as Klipper's TEST_RESONANCES, but on the
// embeddable side:
//
//   1. RESONANCE TEST. Excite the axis with a stepped sine sweep and, at each test
//      frequency, measure the steady-state accelerometer amplitude. We use the
//      library's Goertzel single-bin DFT as the detector — it pulls the response
//      at exactly the drive frequency out of a noisy signal with no FFT, the
//      embeddable analogue of Klipper's PSD. The result is a frequency-response
//      magnitude curve.
//
//   2. IDENTIFY. The curve's peak is the resonance fn; its −3 dB (half-power)
//      bandwidth gives the damping ζ ≈ (f2 − f1) / (2·fn).
//
//   3. SHAPE. Feed (fn, ζ) to design::synthesize_input_shaper and run the shaper
//      on the motion command. A point-to-point move, shaped vs raw, shows the
//      residual ringing cancelled.

namespace {

constexpr double pi = std::numbers::pi;

// The *true* gantry mode (unknown to the calibration below).
constexpr double FN_TRUE = 52.0;   // [Hz]
constexpr double ZETA_TRUE = 0.06; // damping ratio

constexpr double FS_ACCEL = 2000.0; // accelerometer sample rate [Hz] (ADXL345-class)

// One step of the mode x'' + 2 zeta wn x' + wn^2 x = wn^2 u; returns the
// accelerometer reading (x'') for input u. Explicit Euler.
struct Mode {
    double x{0.0};
    double v{0.0};
    double step(double u, double wn, double zeta, double dt) {
        const double accel = (wn * wn * (u - x)) - (2.0 * zeta * wn * v);
        v += dt * accel;
        x += dt * v;
        return accel;
    }
    void reset() {
        x = 0.0;
        v = 0.0;
    }
};

// Deterministic broadband accelerometer noise (no <random>).
double accel_noise(double t) {
    return 0.15 * (std::sin(733.0 * t) + std::sin(2111.0 * t) + std::sin(5189.0 * t)) / 3.0;
}

// Drive the mode at frequency f and return the steady-state accel amplitude at f
// via a coherent Goertzel block — one point of the measured FRF magnitude.
double measure_frf(double f) {
    const double     dt = 1.0 / FS_ACCEL;
    const double     wn = 2.0 * pi * FN_TRUE;
    const int        settle = static_cast<int>(15.0 * FS_ACCEL / f); // ~15 cycles to steady state
    const int        block = static_cast<int>(20.0 * FS_ACCEL / f);  // ~20 coherent cycles
    Goertzel<double> det(f, FS_ACCEL, static_cast<size_t>(block));
    Mode             mode;
    for (int k = 0; k < settle + block; ++k) {
        const double t = k * dt;
        const double u = std::sin(2.0 * pi * f * t);
        const double a = mode.step(u, wn, ZETA_TRUE, dt) + accel_noise(t);
        if (k >= settle) {
            det.push(a);
        }
    }
    return det.amplitude();
}

} // namespace

int main() {
    fmt::print("===== Input-Shaper Calibration from an Accelerometer (Klipper-style) =====\n\n");
    fmt::print("True gantry mode: fn = {:.1f} Hz, zeta = {:.3f}  (unknown to the calibration)\n", FN_TRUE, ZETA_TRUE);
    fmt::print("Accelerometer: {:.0f} Hz, stepped-sine sweep, Goertzel detector\n\n", FS_ACCEL);

    // --- 1. Resonance sweep: FRF magnitude over a frequency grid ---
    std::vector<double> freqs, mag;
    for (double f = 20.0; f <= 90.0; f += 0.5) {
        freqs.push_back(f);
        mag.push_back(measure_frf(f));
    }

    // --- 2. Identify: peak frequency + half-power-bandwidth damping ---
    size_t pk = 0;
    for (size_t i = 1; i < mag.size(); ++i) {
        if (mag[i] > mag[pk]) {
            pk = i;
        }
    }
    const double fn_meas = freqs[pk];
    const double half_power = mag[pk] / std::sqrt(2.0);
    // Walk out from the peak to the -3 dB crossings (linear interpolation on the grid).
    auto cross = [&](int dir) {
        size_t i = pk;
        while (i > 0 && i < mag.size() - 1 && mag[i] > half_power) {
            i = static_cast<size_t>(static_cast<int>(i) + dir);
        }
        // interpolate between i and i-dir
        const size_t j = static_cast<size_t>(static_cast<int>(i) - dir);
        const double frac = (half_power - mag[i]) / (mag[j] - mag[i]);
        return freqs[i] + frac * (freqs[j] - freqs[i]);
    };
    const double f1 = cross(-1);
    const double f2 = cross(+1);
    const double zeta_meas = (f2 - f1) / (2.0 * fn_meas);

    fmt::print("Identified:  fn = {:.2f} Hz (true {:.1f}), zeta = {:.3f} (true {:.3f})\n", fn_meas, FN_TRUE, zeta_meas, ZETA_TRUE);
    fmt::print("             (accel-FRF peak sits a touch above fn for light damping — as on real hardware)\n\n");

    // --- 3. Design the shaper from the measured mode and run a move ---
    constexpr double         Ts_plan = 1.0 / 1000.0; // motion-planner setpoint rate
    const auto               art = design::synthesize_input_shaper(fn_meas, zeta_meas, Ts_plan, ShaperType::ZVD);
    InputShaper<128, double> shaper(art);

    fmt::print("ZVD shaper @ {:.1f} Hz: {} impulses, command delay {:.1f} ms\n", fn_meas, art.count, art.times[art.count - 1] * 1e3);

    // Simulate a 1.0 unit point-to-point move on the TRUE mode, shaped vs raw.
    auto run_move = [&](bool shape) {
        Mode                     mode;
        InputShaper<128, double> s(art);
        std::vector<double>      pos;
        const double             wn = 2.0 * pi * FN_TRUE;
        double                   worst_residual = 0.0;
        const int                steps = 600; // 0.6 s
        for (int k = 0; k < steps; ++k) {
            const double cmd = (k > 0) ? 1.0 : 0.0;
            const double u = shape ? s.step(cmd) : cmd;
            mode.step(u, wn, ZETA_TRUE, Ts_plan);
            pos.push_back(mode.x);
            // Residual ringing once the move is nominally complete (past the
            // shaper's command delay), where the unshaped mode is still ringing.
            if (k >= 30) {
                worst_residual = std::max(worst_residual, std::abs(mode.x - 1.0));
            }
        }
        return std::pair<std::vector<double>, double>{pos, worst_residual};
    };
    auto [pos_raw, res_raw] = run_move(false);
    auto [pos_shaped, res_shaped] = run_move(true);

    fmt::print("Residual ringing after the move:  raw {:.4f}  ->  shaped {:.4f}   ({:.0f}x less)\n", res_raw, res_shaped, res_raw / std::max(res_shaped, 1e-9));

    // --- 4. Plots: measured FRF (with detected peak) + the move response ---
    using namespace plotlypp;
    std::vector<double> t_move;
    for (size_t k = 0; k < pos_raw.size(); ++k) {
        t_move.push_back(k * Ts_plan);
    }
    const std::vector<double> peak_x{fn_meas, fn_meas};
    const std::vector<double> peak_y{0.0, mag[pk]};

    Figure fig;
    fig.addTrace(Scatter().x(freqs).y(mag).mode({Scatter::Mode::Lines}).name("accel FRF magnitude").legend("legend"));
    fig.addTrace(Scatter().x(peak_x).y(peak_y).mode({Scatter::Mode::Lines}).name(fmt::format("detected peak {:.1f} Hz", fn_meas)).legend("legend").line(Scatter::Line().dash("dash")));

    fig.addTrace(Scatter().x(t_move).y(pos_raw).mode({Scatter::Mode::Lines}).name("raw command").xaxis("x2").yaxis("y2").legend("legend2"));
    fig.addTrace(Scatter().x(t_move).y(pos_shaped).mode({Scatter::Mode::Lines}).name("input-shaped").xaxis("x2").yaxis("y2").legend("legend2"));

    using Lg = Layout::Legend;
    auto layout = Layout()
                      .title([](auto& t) { t.text("Input-Shaper Calibration: accelerometer FRF -> ZVD shaper -> ring-free move"); })
                      .xaxis(Layout::Xaxis().title([](auto& t) { t.text("Frequency (Hz)"); }).anchor("y"))
                      .yaxis(Layout::Yaxis().title([](auto& t) { t.text("Accel response"); }).domain({0.58, 1.0}))
                      .xaxis(2, Layout::Xaxis().title([](auto& t) { t.text("Time (s)"); }).anchor("y2"))
                      .yaxis(2, Layout::Yaxis().title([](auto& t) { t.text("Gantry position"); }).domain({0.0, 0.42}))
                      .legend(Lg().x(1.02).y(1.0).xanchor(Lg::Xanchor::Left).yanchor(Lg::Yanchor::Top))
                      .legend(2, Lg().x(1.02).y(0.42).xanchor(Lg::Xanchor::Left).yanchor(Lg::Yanchor::Top))
                      .height(820);
    fig.setLayout(wet::move(layout));
    fig.writeHtml("input_shaper_resonance.html");
    fmt::print("\nPlot written to input_shaper_resonance.html\n");
    return 0;
}
