/**
 * @file example_pmac_thermal.cpp
 * @brief Leaf demo: FET junction-temperature estimation and thermal derating.
 *
 * Builds a Foster thermal network as a StateSpace, discretizes it (ZOH), and runs
 * wet::JunctionEstimator under a current profile to show Tj climbing above the
 * measured case temperature and into the derate band, then recovering. A
 * wet::ThermalLimiter turns Tj into a current derate; the run is recorded and
 * written to an HTML plot. Shows both the full FetLossModel and the one-number
 * ResistiveLossModel. No servo.
 */

#include <vector>

#include "fmt/base.h"
#include "fmt/core.h"
#include "plotlypp/figure.hpp"
#include "plotlypp/layout/layout.hpp"
#include "plotlypp/traces/scatter.hpp"
#include "wet/backend.hpp"
#include "wet/power/thermal.hpp"
#include "wet/systems/discretization.hpp"

using namespace wet;

namespace {

constexpr double Ts = 1.0e-3; // 1 ms thermal tick (slow relative to control)

} // namespace

int main() {
    // Two-section Foster Zth from a datasheet: 0.5 K/W @ 5 ms, 1.5 K/W @ 80 ms.
    const auto cont = design::foster_thermal_ss<2, double>({0.5, 1.5}, {5e-3, 80e-3});
    const auto disc = discretize(cont, Ts, DiscretizationMethod::ZOH);

    // Full FET model: 12 mOhm Rds with +0.4%/°C tempco, switching energy, 6 devices.
    const FetLossModel<double> fet{
        .rds_on = 12e-3,
        .rds_on_tempco = 4e-3,
        .t_ref = 25.0,
        .sw_energy = 80e-6,
        .v_ref = 48.0,
        .i_ref = 20.0,
        .f_sw = 20e3,
        .device_count = 6.0
    };

    JunctionEstimator<2, double, FetLossModel<double>> est{fet, disc};

    // Derate from 110 C, zero at 125 C; hard fault at 130 C.
    const ThermalLimiter<2, double> limiter{
        ThermalLimits<2, double>{derate_window(110.0, 125.0), 130.0}
    };

    // Current profile: a hard 14 A pull (into the derate band) then back to 6 A.
    constexpr double Vdc = 48.0;
    constexpr double case_temp = 70.0;

    std::vector<double> t, tj_hist, scale_hist, i_hist;
    fmt::print("FET junction estimate under a 14 A pull then 6 A (case held at 70 C)\n");
    for (int k = 0; k <= 1500; ++k) {
        const double i_rms = (k < 800) ? 14.0 : 6.0;
        const double tj = est.step(i_rms, Vdc, case_temp);
        const auto   th = limiter.evaluate(tj);
        t.push_back(k * Ts);
        tj_hist.push_back(tj);
        scale_hist.push_back(th.scale);
        i_hist.push_back(i_rms);
        if (k % 250 == 0) {
            fmt::print("  t={:5.0f} ms   I={:4.1f} A   Tj={:6.2f} C   derate={:.3f}  ok={}\n", k * Ts * 1e3, i_rms, tj, th.scale, th.ok);
        }
    }

    // Hobbyist path: one lumped conduction resistance, nothing else known.
    JunctionEstimator<2, double, ResistiveLossModel<double>> simple{ResistiveLossModel<double>{0.05}, disc};
    double                                                   tj_simple = 0.0;
    for (int k = 0; k < 800; ++k) {
        tj_simple = simple.step(14.0, Vdc, case_temp);
    }
    fmt::print("ResistiveLossModel (R=50 mOhm) steady Tj at 14 A = {:.2f} C\n", tj_simple);

    // ----- Plot: junction temperature (with the derate band) and the derate scale.
    using namespace plotlypp;
    const std::vector<double> case_line(t.size(), case_temp);
    const std::vector<double> d_start(t.size(), 110.0), d_cut(t.size(), 125.0);

    Figure fig;
    fig.addTrace(Scatter().x(t).y(tj_hist).mode({Scatter::Mode::Lines}).name("Tj (estimated)").legend("legend"));
    fig.addTrace(Scatter().x(t).y(case_line).mode({Scatter::Mode::Lines}).name("case (NTC)").legend("legend").line(Scatter::Line().dash("dot")));
    fig.addTrace(Scatter().x(t).y(d_start).mode({Scatter::Mode::Lines}).name("derate start").legend("legend").line(Scatter::Line().dash("dash")));
    fig.addTrace(Scatter().x(t).y(d_cut).mode({Scatter::Mode::Lines}).name("cutoff").legend("legend").line(Scatter::Line().dash("dash")));
    fig.addTrace(Scatter().x(t).y(scale_hist).mode({Scatter::Mode::Lines}).name("derate scale").xaxis("x2").yaxis("y2").legend("legend2"));
    fig.addTrace(Scatter().x(t).y(i_hist).mode({Scatter::Mode::Lines}).name("phase current").xaxis("x3").yaxis("y3").legend("legend3"));

    using Lg = Layout::Legend;
    auto panel_legend = [](double y_top) { return Lg().x(1.02).y(y_top).xanchor(Lg::Xanchor::Left).yanchor(Lg::Yanchor::Top); };
    auto layout = Layout()
                      .title([](auto& t2) { t2.text("FET Junction Estimation — Tj, derate, and current"); })
                      .xaxis(Layout::Xaxis().anchor("y"))
                      .yaxis(Layout::Yaxis().title([](auto& t2) { t2.text("Temperature (C)"); }).domain({0.70, 1.0}))
                      .xaxis(2, Layout::Xaxis().anchor("y2"))
                      .yaxis(2, Layout::Yaxis().title([](auto& t2) { t2.text("Derate scale"); }).domain({0.37, 0.63}))
                      .xaxis(3, Layout::Xaxis().title([](auto& t2) { t2.text("Time (s)"); }).anchor("y3"))
                      .yaxis(3, Layout::Yaxis().title([](auto& t2) { t2.text("Current (A)"); }).domain({0.0, 0.30}))
                      .legend(panel_legend(1.0))
                      .legend(2, panel_legend(0.63))
                      .legend(3, panel_legend(0.30))
                      .height(1000);
    fig.setLayout(wet::move(layout));
    fig.writeHtml("pmac_thermal.html");
    fmt::print("  Plot written to pmac_thermal.html\n");
    return 0;
}
