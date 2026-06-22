#include <cmath>
#include <numbers>
#include <vector>

#include "fmt/core.h"
#include "wet/filters/differentiator.hpp"
#include "wet/simulation/plot_plotly.hpp"

using namespace wet;

// ===== Servo Encoder Velocity Estimation =====
//
// An encoder reports *quantized* position. To get velocity you must differentiate
// it, and at low speed (a fraction of a count per sample) the raw finite difference
// is almost pure quantization noise. Four ways to get a usable velocity:
//
//   raw finite difference    : (θ[k] − θ[k−1]) / dt          -> noisy, zero lag
//   LPF'd finite difference   : one-pole LPF of the above      -> quieter, but lags
//   2nd-order PLL observer    : critically-damped tracking loop -> smooth, model-light
//   Levant differentiator     : robust exact differentiator     -> tuning-light, robust
//
// The PLL is the classic encoder tracking observer: predict position from the
// velocity estimate, then correct both states from the position error with gains kp = 2·bw,
// ki = 0.25·kp² (critically-damped repeated pole at −bw). It is a linear observer,
// so it trades noise for phase lag, and its bandwidth must be tuned to the encoder.
// The Levant differentiator is model-free and finite-time exact, needs no bandwidth
// tuning (just an acceleration bound + standard gains), and its error scales with
// the noise *magnitude* not its derivative.
//
// We run two very different encoders to show both estimators are robust:
//   A) 14-bit absolute encoder, bw=1000 — smooth, high resolution.
//   B) 6-state hall on a 7-pole-pair motor (42 counts/rev!), bw=100 — coarse.
// The PLL and the Levant differentiator stay neck-and-neck across both (within
// ~15%, the winner swapping with the regime); raw differencing and an LPF fall
// well behind. Positions are in turns (revolutions) and velocities in turns/s,
// the usual servo-drive units.

constexpr double dt = 1.0 / 8000.0; // 8 kHz current-loop rate [s]

namespace {

constexpr double two_pi = 2.0 * std::numbers::pi;

struct Scenario {
    const char* label;
    double      counts_per_rev;
    double      bandwidth;   // PLL bandwidth [rad/s]
    double      amplitude;   // motion amplitude [rad]
    double      freq;        // motion frequency [Hz]
    double      lpf_tau;     // one-pole LPF time constant [s]
    double      zoom_half_s; // half-width of the position-zoom panel [s]
    const char* plot_file;
};

// 2nd-order critically-damped PLL position/velocity tracking observer.
struct PllObserver {
    double kp;
    double ki;
    double pos{0.0};
    double vel{0.0};
    explicit PllObserver(double bw) : kp(2.0 * bw), ki(0.25 * (2.0 * bw) * (2.0 * bw)) {}
    double update(double meas) {
        pos += dt * vel;
        const double e = meas - pos;
        pos += dt * kp * e;
        vel += dt * ki * e;
        return vel;
    }
};

std::vector<double> decimate(const std::vector<double>& v, int d) {
    std::vector<double> out;
    for (size_t i = 0; i < v.size(); i += static_cast<size_t>(d)) {
        out.push_back(v[i]);
    }
    return out;
}

void run_scenario(const Scenario& s) {
    const double w = two_pi * s.freq;
    const double q = 1.0 / s.counts_per_rev; // turns per count (position is in turns)
    auto         true_pos = [&](double t) { return s.amplitude * std::sin(w * t); };
    auto         true_vel = [&](double t) { return s.amplitude * w * std::cos(w * t); };
    auto         encoder = [&](double t) { return std::round(true_pos(t) / q) * q; };

    const int steps = static_cast<int>(4.0 / s.freq / dt); // 4 full periods
    const int settle = steps / 8;

    RobustExactDifferentiator<double> red(2.0 * s.amplitude * w * w, dt); // L = 2·|θ̈|max
    const double                      lpf_alpha = dt / (s.lpf_tau + dt);
    double                            lpf_vel = 0.0;
    PllObserver                       pll(s.bandwidth);
    double                            pos_prev = encoder(0.0);

    std::vector<double> ts, p_true, p_enc, p_pll, p_red;
    std::vector<double> v_true, v_lpf, v_pll, v_red, e_lpf, e_pll, e_red;

    double sum_fd = 0, sum_lpf = 0, sum_pll = 0, sum_red = 0;
    double rev_fd = 0, rev_lpf = 0, rev_pll = 0, rev_red = 0;
    int    n = 0, n_rev = 0;

    for (int k = 0; k < steps; ++k) {
        const double t = k * dt;
        const double pos = encoder(t);
        const double vt = true_vel(t);

        const double fd = (pos - pos_prev) / dt;
        pos_prev = pos;
        lpf_vel += lpf_alpha * (fd - lpf_vel);
        const double vp = pll.update(pos);
        const double rd = red.update(pos);

        ts.push_back(t);
        p_true.push_back(true_pos(t));
        p_enc.push_back(pos);
        p_pll.push_back(pll.pos);
        p_red.push_back(red.value());
        v_true.push_back(vt);
        v_lpf.push_back(lpf_vel);
        v_pll.push_back(vp);
        v_red.push_back(rd);
        e_lpf.push_back(lpf_vel - vt);
        e_pll.push_back(vp - vt);
        e_red.push_back(rd - vt);

        if (k >= settle) {
            sum_fd += (fd - vt) * (fd - vt);
            sum_lpf += (lpf_vel - vt) * (lpf_vel - vt);
            sum_pll += (vp - vt) * (vp - vt);
            sum_red += (rd - vt) * (rd - vt);
            ++n;
            if (std::abs(vt) < 0.15 * s.amplitude * w) {
                rev_fd += (fd - vt) * (fd - vt);
                rev_lpf += (lpf_vel - vt) * (lpf_vel - vt);
                rev_pll += (vp - vt) * (vp - vt);
                rev_red += (rd - vt) * (rd - vt);
                ++n_rev;
            }
        }
    }

    auto rms = [](double s2, int cnt) { return std::sqrt(s2 / cnt); };
    fmt::print("--- {} ---\n", s.label);
    fmt::print("  {:.0f} counts/rev ({:.2f} deg/count), bw={:.0f}, motion {:.2f} turns @ {:.2f} Hz, ~{:.2f} counts/sample peak\n", s.counts_per_rev, q * 360.0, s.bandwidth, s.amplitude, s.freq, (s.amplitude * w * dt) / q);
    fmt::print("  {:<26} {:>16} {:>18}\n", "estimator", "RMS err [turns/s]", "near-reversal");
    fmt::print("  {:<26} {:>16.4f} {:>18.4f}\n", "raw finite difference", rms(sum_fd, n), rms(rev_fd, n_rev));
    fmt::print("  {:<26} {:>16.4f} {:>18.4f}\n", "LPF'd diff", rms(sum_lpf, n), rms(rev_lpf, n_rev));
    fmt::print("  {:<26} {:>16.4f} {:>18.4f}\n", "PLL observer", rms(sum_pll, n), rms(rev_pll, n_rev));
    fmt::print("  {:<26} {:>16.4f} {:>18.4f}\n\n", "Levant differentiator", rms(sum_red, n), rms(rev_red, n_rev));

    // ----- Plot -----
    using namespace plotlypp;
    const int  rev_idx = static_cast<int>((1.0 / (4.0 * s.freq)) / dt); // first reversal
    const int  half = static_cast<int>(s.zoom_half_s / dt);
    const int  z0 = rev_idx - half, z1 = rev_idx + half;
    auto       slice = [&](const std::vector<double>& v) { return std::vector<double>(v.begin() + z0, v.begin() + z1); };
    const auto tz = slice(ts);
    const int  D = 1 + steps / 2500; // decimate full-run traces to ~2500 points
    const auto td = decimate(ts, D);

    // Each panel's traces go to their own legend ("legend"/"legend2"/"legend3"),
    // positioned beside that panel below.
    Figure fig;
    fig.addTrace(Scatter().x(tz).y(slice(p_true)).mode({Scatter::Mode::Lines}).name("true position").legend("legend"));
    fig.addTrace(Scatter().x(tz).y(slice(p_enc)).mode({Scatter::Mode::Lines}).name("encoder (quantized)").legend("legend").line(Scatter::Line().shape(Scatter::Line::Shape::Hv)));
    fig.addTrace(Scatter().x(tz).y(slice(p_pll)).mode({Scatter::Mode::Lines}).name("PLL estimate").legend("legend"));
    fig.addTrace(Scatter().x(tz).y(slice(p_red)).mode({Scatter::Mode::Lines}).name("Levant value()").legend("legend"));

    fig.addTrace(Scatter().x(td).y(decimate(v_true, D)).mode({Scatter::Mode::Lines}).name("true velocity").xaxis("x2").yaxis("y2").legend("legend2").line(Scatter::Line().dash("dash")));
    fig.addTrace(Scatter().x(td).y(decimate(v_lpf, D)).mode({Scatter::Mode::Lines}).name("LPF'd diff").xaxis("x2").yaxis("y2").legend("legend2"));
    fig.addTrace(Scatter().x(td).y(decimate(v_pll, D)).mode({Scatter::Mode::Lines}).name("PLL observer").xaxis("x2").yaxis("y2").legend("legend2"));
    fig.addTrace(Scatter().x(td).y(decimate(v_red, D)).mode({Scatter::Mode::Lines}).name("Levant").xaxis("x2").yaxis("y2").legend("legend2"));

    fig.addTrace(Scatter().x(td).y(decimate(e_lpf, D)).mode({Scatter::Mode::Lines}).name("LPF'd err").xaxis("x3").yaxis("y3").legend("legend3"));
    fig.addTrace(Scatter().x(td).y(decimate(e_pll, D)).mode({Scatter::Mode::Lines}).name("PLL err").xaxis("x3").yaxis("y3").legend("legend3"));
    fig.addTrace(Scatter().x(td).y(decimate(e_red, D)).mode({Scatter::Mode::Lines}).name("Levant err").xaxis("x3").yaxis("y3").legend("legend3"));

    // Stack the three panels vertically: each y-axis gets its own domain band and
    // each x-axis is anchored to its y-axis (otherwise plotly overlays them all).
    // One legend per panel, pinned to the top of that panel's band.
    using Lg = Layout::Legend;
    auto panel_legend = [](double y_top) {
        return Lg().x(1.02).y(y_top).xanchor(Lg::Xanchor::Left).yanchor(Lg::Yanchor::Top);
    };
    auto layout = Layout()
                      .title([&](auto& t) { t.text(fmt::format("Encoder Velocity Estimation — {}", s.label)); })
                      .xaxis(Layout::Xaxis().title([](auto& t) { t.text("Time (s) — zoom on a reversal"); }).anchor("y"))
                      .yaxis(Layout::Yaxis().title([](auto& t) { t.text("Position (turns)"); }).domain({0.70, 1.0}))
                      .xaxis(2, Layout::Xaxis().title([](auto& t) { t.text("Time (s)"); }).anchor("y2"))
                      .yaxis(2, Layout::Yaxis().title([](auto& t) { t.text("Velocity (turns/s)"); }).domain({0.37, 0.63}))
                      .xaxis(3, Layout::Xaxis().title([](auto& t) { t.text("Time (s)"); }).anchor("y3"))
                      .yaxis(3, Layout::Yaxis().title([](auto& t) { t.text("Velocity error (turns/s)"); }).domain({0.0, 0.30}))
                      .legend(panel_legend(1.0))
                      .legend(2, panel_legend(0.63))
                      .legend(3, panel_legend(0.30))
                      .height(1100);
    fig.setLayout(wet::move(layout));
    fig.writeHtml(s.plot_file);
    fmt::print("  Plot written to {}\n\n", s.plot_file);
}

} // namespace

int main() {
    fmt::print("===== Servo Encoder Velocity Estimation =====\n");
    fmt::print("8 kHz loop. PLL = critically-damped tracking observer.\n\n");

    // A) Smooth, high-resolution: a well-tuned PLL is excellent here.
    run_scenario({"14-bit absolute, bw=1000", 16384.0, 1000.0, 0.25, 1.0, 3.0e-3, 0.02, "encoder_velocity_absolute.html"});

    // B) Coarse: 6 hall states x 7 pole pairs = 42 counts/rev, bw=100.
    run_scenario({"6-state hall, 7 pole pairs (42 cpr), bw=100", 42.0, 100.0, 1.0, 0.5, 8.0e-3, 0.15, "encoder_velocity_hall.html"});

    fmt::print("Across both a fine 14-bit encoder and a coarse 42-count hall, the PLL\n");
    fmt::print("observer and the Levant differentiator stay neck-and-neck (within ~15%;\n");
    fmt::print("which one edges ahead flips with the regime), and both leave raw\n");
    fmt::print("differencing and the LPF well behind. The Levant block gets there\n");
    fmt::print("model-free, with no bandwidth to tune -- just an acceleration bound and\n");
    fmt::print("the standard gains -- and it's the matched rate estimator for the\n");
    fmt::print("super-twisting controller.\n");
    return 0;
}
