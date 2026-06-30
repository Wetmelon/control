/**
 * @file example_pmac_bus_limit.cpp
 * @brief Leaf demo: DC-bus current/power/regen limiting and the UV/OV gate.
 *
 * Evaluates wet::DcBusLimiter at a handful of discrete operating points (printed)
 * and across a torque-current sweep at a fixed speed, writing an HTML plot of the
 * bus power and the resulting derate so the current/power knees are visible. No
 * servo loop — just the limiter primitive.
 */

#include <vector>

#include "fmt/base.h"
#include "fmt/core.h"
#include "plotlypp/figure.hpp"
#include "plotlypp/layout/layout.hpp"
#include "plotlypp/traces/scatter.hpp"
#include "wet/backend.hpp"
#include "wet/motor/limits.hpp"
#include "wet/toolbox/bounds.hpp"
#include "wet/transforms.hpp"

using namespace wet;

namespace {

void report(const DcBusLimiter<double>& bus, const char* label, DirectQuadrature<double> Vdq, DirectQuadrature<double> Idq, double Vdc) {
    const auto s = bus.evaluate(Vdq, Idq, Vdc);
    fmt::print(
        "  {:<22} P_bus={:7.1f} W  I_bus={:6.2f} A  ->  scale={:.3f}  ok={}\n", label, s.bus_power, s.bus_current, s.scale, s.ok
    );
}

// Sweep the commanded q-current at a fixed speed and plot what the limiter does:
// Vq = back-emf + R*iq models the operating point so bus power grows with current.
void sweep(const DcBusLimiter<double>& bus) {
    using namespace plotlypp;

    constexpr double Vdc = 30.0;
    constexpr double Vq_bemf = 18.0;
    constexpr double R = 0.1;
    constexpr double P_max = 250.0;

    std::vector<double> iq_cmd, p_bus, scale, iq_delivered, p_cap;
    for (double iq = 0.0; iq <= 40.0; iq += 0.25) {
        const DirectQuadrature<double> Vdq{.d = 0.0, .q = Vq_bemf + (R * iq)};
        const DirectQuadrature<double> Idq{.d = 0.0, .q = iq};
        const auto                     s = bus.evaluate(Vdq, Idq, Vdc);
        iq_cmd.push_back(iq);
        p_bus.push_back(s.bus_power);
        scale.push_back(s.scale);
        iq_delivered.push_back(iq * s.scale);
        p_cap.push_back(P_max);
    }

    Figure fig;
    fig.addTrace(Scatter().x(iq_cmd).y(p_bus).mode({Scatter::Mode::Lines}).name("bus power").legend("legend"));
    fig.addTrace(Scatter().x(iq_cmd).y(p_cap).mode({Scatter::Mode::Lines}).name("P_max cap").legend("legend").line(Scatter::Line().dash("dash")));
    fig.addTrace(Scatter().x(iq_cmd).y(scale).mode({Scatter::Mode::Lines}).name("derate scale").xaxis("x2").yaxis("y2").legend("legend2"));
    fig.addTrace(Scatter().x(iq_cmd).y(iq_delivered).mode({Scatter::Mode::Lines}).name("delivered iq").xaxis("x3").yaxis("y3").legend("legend3"));
    fig.addTrace(Scatter().x(iq_cmd).y(iq_cmd).mode({Scatter::Mode::Lines}).name("commanded iq").xaxis("x3").yaxis("y3").legend("legend3").line(Scatter::Line().dash("dash")));

    using Lg = Layout::Legend;
    auto panel_legend = [](double y_top) { return Lg().x(1.02).y(y_top).xanchor(Lg::Xanchor::Left).yanchor(Lg::Yanchor::Top); };
    auto layout = Layout()
                      .title([](auto& t) { t.text("DC-Bus Limiter — derate over a torque-current sweep (30 V bus)"); })
                      .xaxis(Layout::Xaxis().anchor("y"))
                      .yaxis(Layout::Yaxis().title([](auto& t) { t.text("Bus power (W)"); }).domain({0.70, 1.0}))
                      .xaxis(2, Layout::Xaxis().anchor("y2"))
                      .yaxis(2, Layout::Yaxis().title([](auto& t) { t.text("Derate scale"); }).domain({0.37, 0.63}))
                      .xaxis(3, Layout::Xaxis().title([](auto& t) { t.text("Commanded q-current (A)"); }).anchor("y3"))
                      .yaxis(3, Layout::Yaxis().title([](auto& t) { t.text("q-current (A)"); }).domain({0.0, 0.30}))
                      .legend(panel_legend(1.0))
                      .legend(2, panel_legend(0.63))
                      .legend(3, panel_legend(0.30))
                      .height(1000);
    fig.setLayout(wet::move(layout));
    fig.writeHtml("pmac_bus_limit.html");
    fmt::print("  Plot written to pmac_bus_limit.html\n");
}

} // namespace

int main() {
    // 30 V bus: 10 A motoring / 5 A regen ceiling, 250 W cap, arm window 20..36 V.
    const DcBusLimiter<double> bus{
        DcBusLimits<double>{
            .bus_current = Bounds<1, double>{-5.0, 10.0},
            .bus_power_max = 250.0,
            .voltage = Bounds<1, double>{20.0, 36.0},
        }
    };

    fmt::print("DC-bus limiter (I in [-5,10] A, P_max 250 W, arm 20..36 V)\n");
    report(bus, "light motoring", {.d = 0.0, .q = 12.0}, {.d = 0.0, .q = 5.0}, 30.0);   // ~90 W, no derate
    report(bus, "current-limited", {.d = 0.0, .q = 24.0}, {.d = 0.0, .q = 20.0}, 30.0); // 360 A-equiv -> clamps
    report(bus, "power-limited", {.d = 0.0, .q = 30.0}, {.d = 0.0, .q = 14.0}, 30.0);   // 630 W -> P cap binds
    report(bus, "regen", {.d = 0.0, .q = -28.0}, {.d = 0.0, .q = 18.0}, 30.0);          // braking into the bus
    report(bus, "undervoltage (disarm)", {.d = 0.0, .q = 5.0}, {.d = 0.0, .q = 2.0}, 18.0);
    report(bus, "overvoltage (disarm)", {.d = 0.0, .q = 5.0}, {.d = 0.0, .q = 2.0}, 40.0);
    sweep(bus);
    return 0;
}
