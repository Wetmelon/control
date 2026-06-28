/**
 * @file example_pmac_calibration.cpp
 * @brief Leaf demo: recover phase R and L by RLS with PRBS injection.
 *
 * Drives wet::PhaseParameterCalibrator against an exact discrete R-L plant (the
 * same ARX model it fits internally), at a few operating points, and prints the
 * recovered parameters versus truth. For the representative motor it also records
 * the RLS estimate each step and writes an HTML plot of the convergence. No motor,
 * no servo — just the commissioning primitive in isolation.
 */

#include <cmath>
#include <vector>

#include "fmt/base.h"
#include "fmt/core.h"
#include "plotlypp/figure.hpp"
#include "plotlypp/layout/layout.hpp"
#include "plotlypp/traces/scatter.hpp"
#include "wet/backend.hpp"
#include "wet/motor/calibration.hpp"

using namespace wet;

namespace {

constexpr double Ts = 1.0 / 20000.0; // 20 kHz

struct Convergence {
    std::vector<double> t, r_est, l_est; // recorded once the fit is valid
};

// Identify one (R, L) pair from the exact discrete d-axis plant the calibrator
// fits: i[k] = a*i[k-1] + b*v[k-1], a = exp(-R*Ts/L), b = (1-a)/R. Optionally
// records the running estimate for plotting.
void identify(double R, double L, Convergence* trace = nullptr) {
    const double a = std::exp(-R * Ts / L);
    const double bd = (1.0 - a) / R;

    PhaseParameterCalibrator<double> cal{
        PhaseCalibrationConfig<double>{.inject_voltage = 2.0, .duration_s = 0.2}
    };

    double i = 0.0, t = 0.0;
    for (;;) {
        const auto cmd = cal.step(i, Ts);
        i = (a * i) + (bd * cmd.v_d); // plant response to the commanded voltage
        if (trace != nullptr && cal.valid()) {
            trace->t.push_back(t);
            trace->r_est.push_back(cal.resistance());
            trace->l_est.push_back(cal.inductance() * 1e6); // uH
        }
        t += Ts;
        if (cmd.done) {
            break;
        }
    }

    fmt::print(
        "  R = {:.4f} ohm (true {:.4f}, {:+.2f}%)   L = {:.2f} uH (true {:.2f}, {:+.2f}%)   valid={}\n",
        cal.resistance(), R, 100.0 * (cal.resistance() - R) / R, cal.inductance() * 1e6, L * 1e6,
        100.0 * (cal.inductance() - L) / L, cal.valid()
    );
}

void plot(const Convergence& c, double R_true, double L_true_uH) {
    using namespace plotlypp;
    const std::vector<double> r_ref(c.t.size(), R_true), l_ref(c.t.size(), L_true_uH);

    Figure fig;
    fig.addTrace(Scatter().x(c.t).y(c.r_est).mode({Scatter::Mode::Lines}).name("R estimate").legend("legend"));
    fig.addTrace(Scatter().x(c.t).y(r_ref).mode({Scatter::Mode::Lines}).name("R true").legend("legend").line(Scatter::Line().dash("dash")));
    fig.addTrace(Scatter().x(c.t).y(c.l_est).mode({Scatter::Mode::Lines}).name("L estimate").xaxis("x2").yaxis("y2").legend("legend2"));
    fig.addTrace(Scatter().x(c.t).y(l_ref).mode({Scatter::Mode::Lines}).name("L true").xaxis("x2").yaxis("y2").legend("legend2").line(Scatter::Line().dash("dash")));

    using Lg = Layout::Legend;
    auto panel_legend = [](double y_top) { return Lg().x(1.02).y(y_top).xanchor(Lg::Xanchor::Left).yanchor(Lg::Yanchor::Top); };
    auto layout = Layout()
                      .title([](auto& t) { t.text("Phase R/L Calibration — RLS convergence (0.5 ohm, 200 uH)"); })
                      .xaxis(Layout::Xaxis().title([](auto& t) { t.text("Time (s)"); }).anchor("y"))
                      .yaxis(Layout::Yaxis().title([](auto& t) { t.text("Resistance (ohm)"); }).domain({0.55, 1.0}))
                      .xaxis(2, Layout::Xaxis().title([](auto& t) { t.text("Time (s)"); }).anchor("y2"))
                      .yaxis(2, Layout::Yaxis().title([](auto& t) { t.text("Inductance (uH)"); }).domain({0.0, 0.45}))
                      .legend(panel_legend(1.0))
                      .legend(2, panel_legend(0.45))
                      .height(800);
    fig.setLayout(wet::move(layout));
    fig.writeHtml("pmac_calibration.html");
    fmt::print("  Plot written to pmac_calibration.html\n");
}

} // namespace

int main() {
    fmt::print("Phase R/L identification (RLS + PRBS, exact discrete-ARX fit)\n");
    Convergence trace;
    identify(0.5, 200e-6, &trace); // small hobby gimbal motor (plotted)
    identify(1.2, 1.5e-3);         // higher-impedance stepper-ish winding
    identify(0.05, 50e-6);         // big low-inductance traction motor
    plot(trace, 0.5, 200.0);
    return 0;
}
