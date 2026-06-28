/**
 * @file example_pmac_estimator.cpp
 * @brief Leaf demo: the [theta, omega, tau_load] Kalman estimator in isolation.
 *
 * Runs wet::motor::MechanicalEstimator against a simulated 1-DOF drivetrain with a mid-run
 * load-torque step. Shows the cheap predict + multirate encoder update converging
 * speed and the unknown load torque, and the optional load-accelerometer channel
 * tracking the step faster. The run is recorded and written to an HTML plot. No servo.
 */

#include <vector>

#include "fmt/base.h"
#include "fmt/core.h"
#include "plotlypp/figure.hpp"
#include "plotlypp/layout/layout.hpp"
#include "plotlypp/traces/scatter.hpp"
#include "wet/backend.hpp"
#include "wet/motor/mechanical_estimator.hpp"

using namespace wet;

namespace {

constexpr double J = 2e-4, b = 1e-3, Kt = 0.06; // plant + torque constant
constexpr double Ts = 1.0 / 20000.0;            // 20 kHz predict rate
constexpr int    enc_decim = 20;                // encoder updates at 1 kHz
constexpr int    steps = 20000;                 // 1 s
constexpr double iq = 3.0;                      // known applied current

// The load steps on partway through, unknown to the estimator.
double load_at(int k) { return (k < steps / 2) ? 0.0 : 0.04; }

// Truth: rigid 1-DOF mechanical plant driven by a known iq, with a real load torque.
struct Plant {
    double w{0}, th{0};
    void   step(double tau_load) {
        const double h = Ts / 8.0;
        for (int i = 0; i < 8; ++i) {
            w += h * (((Kt * iq) - tau_load - (b * w)) / J);
            th += h * w;
        }
    }
};

motor::MechanicalEstimator<double> make_estimator() {
    return motor::MechanicalEstimator<double>{
        motor::MechanicalEstimatorConfig<double>{.J = J, .b = b, .Kt = Kt, .Ts = Ts}
    };
}

} // namespace

int main() {
    std::vector<double> t, w_true, w_est, load_true, load_enc, load_accel;

    fmt::print("Encoder-only (1 kHz updates) vs encoder + load accelerometer\n");
    auto  enc = make_estimator();
    auto  acc = make_estimator();
    Plant p_enc;
    Plant p_acc;
    for (int k = 0; k < steps; ++k) {
        const double tau = load_at(k);
        p_enc.step(tau);
        p_acc.step(tau);

        enc.predict(iq);
        acc.predict(iq);
        if (k % enc_decim == 0) {
            enc.update_encoder(p_enc.th);
            acc.update_encoder(p_acc.th);
            const double alpha = ((Kt * iq) - (b * p_acc.w) - tau) / J; // measured load accel
            acc.update_load_accel(alpha, iq);
        }

        t.push_back(k * Ts);
        w_true.push_back(p_enc.w);
        w_est.push_back(enc.omega());
        load_true.push_back(tau);
        load_enc.push_back(enc.load_torque());
        load_accel.push_back(acc.load_torque());

        if (k % 4000 == 0) {
            fmt::print("  t={:.2f}s  w {:7.2f}   tau_load: true {:.4f}  enc {:.4f}  accel {:.4f}\n", k * Ts, p_enc.w, tau, enc.load_torque(), acc.load_torque());
        }
    }

    using namespace plotlypp;
    Figure fig;
    fig.addTrace(Scatter().x(t).y(w_true).mode({Scatter::Mode::Lines}).name("speed true").legend("legend").line(Scatter::Line().dash("dash")));
    fig.addTrace(Scatter().x(t).y(w_est).mode({Scatter::Mode::Lines}).name("speed est").legend("legend"));
    fig.addTrace(Scatter().x(t).y(load_true).mode({Scatter::Mode::Lines}).name("load true").xaxis("x2").yaxis("y2").legend("legend2").line(Scatter::Line().dash("dash")));
    fig.addTrace(Scatter().x(t).y(load_enc).mode({Scatter::Mode::Lines}).name("load est (encoder)").xaxis("x2").yaxis("y2").legend("legend2"));
    fig.addTrace(Scatter().x(t).y(load_accel).mode({Scatter::Mode::Lines}).name("load est (+accel)").xaxis("x2").yaxis("y2").legend("legend2"));

    using Lg = Layout::Legend;
    auto panel_legend = [](double y_top) { return Lg().x(1.02).y(y_top).xanchor(Lg::Xanchor::Left).yanchor(Lg::Yanchor::Top); };
    auto layout = Layout()
                      .title([](auto& tt) { tt.text("Mechanical Estimator — speed and load-torque tracking through a load step"); })
                      .xaxis(Layout::Xaxis().anchor("y"))
                      .yaxis(Layout::Yaxis().title([](auto& tt) { tt.text("Speed (rad/s)"); }).domain({0.55, 1.0}))
                      .xaxis(2, Layout::Xaxis().title([](auto& tt) { tt.text("Time (s)"); }).anchor("y2"))
                      .yaxis(2, Layout::Yaxis().title([](auto& tt) { tt.text("Load torque (Nm)"); }).domain({0.0, 0.45}))
                      .legend(panel_legend(1.0))
                      .legend(2, panel_legend(0.45))
                      .height(800);
    fig.setLayout(wet::move(layout));
    fig.writeHtml("pmac_estimator.html");
    fmt::print("  Plot written to pmac_estimator.html\n");
    return 0;
}
