/**
 * @file example_foc.cpp
 * @brief PMSM current-loop (FOC) electrical sim: PI vs I-P on a step + disturbance
 *
 * Closes wet::FOController around a dq-frame PMSM electrical model and runs the
 * same scenario twice — once as a standard PI current loop (setpoint weight
 * b = 1, P on the error) and once as an I-P loop (b = 0, P on the measurement) —
 * to contrast their behaviour on:
 *
 *   1. Reference step  — the q-axis (torque) current command steps 0 → Iq_ref.
 *   2. Disturbance     — an unmodeled q-axis voltage step (inverter dead-time /
 *                        back-EMF mismatch) is injected.
 *
 * Setpoint weighting moves only the reference→output zero, not the closed-loop
 * poles, so both share the same pole placement (design::current_loop_pi) and the
 * same disturbance rejection. The difference is the step: PI applies a full
 * proportional kick Kp·Δr (overshoot, and the usual trigger of the SVPWM
 * voltage-circle limit), while I-P lets only the integral ramp the command — a
 * smoother current step that stays clear of the voltage circle.
 *
 * Plant (motor convention), integrated with forward Euler sub-steps:
 *   Ld * d(id)/dt = vd - R*id + omega*Lq*iq
 *   Lq * d(iq)/dt = vq - R*iq - omega*Ld*id - omega*lambda
 */

#include <algorithm>
#include <cmath>
#include <vector>

#include "fmt/core.h"
#include "wet/simulation/plot_plotly.hpp"
#include "wet/utility/foc.hpp"

using namespace wet;

namespace {

using T = double;

// ---- Motor nameplate (Turnigy D5065 270KV surface-PMSM, Ld = Lq) -----------
constexpr T Rs = 0.039;      // [ohm] phase resistance (phase-neutral)
constexpr T Ls = 16e-6;      // [H]   phase inductance (Ld = Lq, phase-neutral)
constexpr T Kt_spec = 0.031; // [Nm/A] torque constant (amplitude / peak per-phase)
constexpr T pole_pairs = 7.0;
constexpr T Vdc = 24.0;      // [V]  DC bus
constexpr T omega_e = 600.0; // [erad/s] fixed electrical speed

// ---- Loop timing -----------------------------------------------------------
constexpr T   fsw = 20000.0;  // [Hz] PWM / current-loop rate
constexpr T   Ts = 1.0 / fsw; // [s]  control period
constexpr int substeps = 100; // plant Euler sub-steps per control tick
constexpr T   t_end = 6.0e-3; // [s]

// omega_bw ~ fsw/13 keeps the continuous pole placement valid (see
// design::current_loop_pi's sampling note).
constexpr T bw = 2.0 * numbers::pi_v<T> * fsw / 13.0;

// ---- Scenario --------------------------------------------------------------
constexpr T Iq_ref = 8.0;    // [A]   torque-current step target
constexpr T t_step = 0.5e-3; // [s]   reference step time
constexpr T t_dist = 3.0e-3; // [s]   disturbance onset
constexpr T vq_dist = -1.0;  // [V]   unmodeled q-axis voltage disturbance

// Plant flux linkage recovered from the datasheet Kt.
const T                   lambda_pm = design::flux_from_torque_constant(pole_pairs, Kt_spec);
const DirectQuadrature<T> Ldq{Ls, Ls};
const T                   Vmax = design::voltage_circle_radius(Vdc);

struct RunLog {
    std::vector<double> ts, iq_ref, iq, id, vmag; // sampled at the plant sub-tick rate
    double              rise_ms = 0.0;            // 98% rise after the step
    double              overshoot_pct = 0.0;      // peak iq overshoot in the step window
    double              peak_vmag = 0.0;          // peak |Vdq| in the step window
    double              final_iq = 0.0;           // iq at end (post-disturbance recovery)
};

// Run the full scenario with proportional setpoint weight b (1 = PI, 0 = I-P).
RunLog run(T b) {
    FOController<T> foc{Ldq, Rs, lambda_pm, omega_e, bw};
    foc.tune(bw, T{1}, b); // re-seed with the requested setpoint weight
    foc.reset();
    foc.enable();

    DirectQuadrature<T> Idq{0.0, 0.0};
    RunLog              log;

    double     peak_iq = 0.0;
    bool       rise_found = false;
    const T    dt = Ts / substeps;
    const auto n_ticks = static_cast<int>(std::lround(t_end / Ts));

    for (int k = 0; k <= n_ticks; ++k) {
        const T                   t = k * Ts;
        const T                   iq_ref = (t >= t_step) ? Iq_ref : T{0};
        const DirectQuadrature<T> Idq_ref{T{0}, iq_ref};

        const auto cmd = foc.current_controller(Idq_ref, Idq, Ts, Vmax);
        const T    vd = cmd.Vdq.d;
        const T    vq = cmd.Vdq.q + ((t >= t_dist) ? vq_dist : T{0});
        const T    vmag = cmd.Vdq.abs();

        for (int s = 0; s < substeps; ++s) {
            const T did = (vd - (Rs * Idq.d) + (omega_e * Ldq.q * Idq.q)) / Ldq.d;
            const T diq = (vq - (Rs * Idq.q) - (omega_e * Ldq.d * Idq.d) - (omega_e * lambda_pm)) / Ldq.q;
            Idq.d += did * dt;
            Idq.q += diq * dt;

            log.ts.push_back((t + ((s + 1) * dt)) * 1e3); // [ms]
            log.iq_ref.push_back(iq_ref);
            log.id.push_back(Idq.d);
            log.iq.push_back(Idq.q);
            log.vmag.push_back(vmag);
        }

        // Step-response metrics, measured in the window before the disturbance.
        if (t >= t_step && t < t_dist) {
            if (!rise_found && Idq.q >= 0.98 * Iq_ref) {
                log.rise_ms = (t - t_step) * 1e3;
                rise_found = true;
            }
            peak_iq = std::max(peak_iq, Idq.q);
            log.peak_vmag = std::max(log.peak_vmag, vmag);
        }
    }

    log.overshoot_pct = 100.0 * (peak_iq - Iq_ref) / Iq_ref;
    log.final_iq = Idq.q;
    return log;
}

} // namespace

int main() {
    const T Kt = design::torque_constant_from_flux(pole_pairs, lambda_pm);
    const T Km = design::motor_constant(Kt, Rs);

    fmt::print("===== PMSM FOC current loop: PI vs I-P =====\n");
    fmt::print("Kt = {:.4f} Nm/A   Km = {:.3f} Nm/sqrt(W)   Vmax = {:.2f} V   f_bw = {:.0f} Hz   fsw = {:.0f} Hz\n\n", Kt, Km, Vmax, bw / (2.0 * numbers::pi_v<T>), fsw);

    const RunLog pi = run(T{1}); // standard PI (P on error)
    const RunLog ip = run(T{0}); // I-P        (P on measurement)

    fmt::print("  structure |  98%% rise [ms] | overshoot [%%] | peak |Vdq| [V] | final iq [A]\n");
    fmt::print("  ----------+----------------+---------------+----------------+-------------\n");
    fmt::print("  PI  (b=1) | {:14.3f} | {:13.1f} | {:14.3f} | {:11.3f}\n", pi.rise_ms, pi.overshoot_pct, pi.peak_vmag, pi.final_iq);
    fmt::print("  I-P (b=0) | {:14.3f} | {:13.1f} | {:14.3f} | {:11.3f}\n", ip.rise_ms, ip.overshoot_pct, ip.peak_vmag, ip.final_iq);
    fmt::print("\n  Vmax = {:.2f} V — PI peaks higher into the voltage circle; both reject the\n", Vmax);
    fmt::print("  {:.0f} V disturbance to the same steady iq (identical poles / rejection).\n", vq_dist);

    // ----- Plot: PI vs I-P overlay, currents + |Vdq| ------------------------
    using namespace plotlypp;
    const std::vector<double> vmax_line(pi.ts.size(), Vmax);

    auto event_lines = [&](const char* yref) {
        return std::vector<Layout::Shape>{
            Layout::Shape().type(Layout::Shape::Type::Line).x0(t_step * 1e3).x1(t_step * 1e3).xref("x").y0(0).y1(1).yref(yref).line(Layout::Shape::Line().dash("dot").color("green")),
            Layout::Shape().type(Layout::Shape::Type::Line).x0(t_dist * 1e3).x1(t_dist * 1e3).xref("x").y0(0).y1(1).yref(yref).line(Layout::Shape::Line().dash("dot").color("red"))
        };
    };

    Figure fig;
    // Panel 1: q-axis current — reference vs PI vs I-P.
    fig.addTrace(Scatter().x(pi.ts).y(pi.iq_ref).mode({Scatter::Mode::Lines}).name("iq_ref").legend("legend").line(Scatter::Line().dash("dash").color("gray")));
    fig.addTrace(Scatter().x(pi.ts).y(pi.iq).mode({Scatter::Mode::Lines}).name("iq — PI (b=1)").legend("legend"));
    fig.addTrace(Scatter().x(ip.ts).y(ip.iq).mode({Scatter::Mode::Lines}).name("iq — I-P (b=0)").legend("legend"));

    // Panel 2: voltage command magnitude vs the circle limit.
    fig.addTrace(Scatter().x(pi.ts).y(pi.vmag).mode({Scatter::Mode::Lines}).name("|Vdq| — PI").xaxis("x2").yaxis("y2").legend("legend2"));
    fig.addTrace(Scatter().x(ip.ts).y(ip.vmag).mode({Scatter::Mode::Lines}).name("|Vdq| — I-P").xaxis("x2").yaxis("y2").legend("legend2"));
    fig.addTrace(Scatter().x(pi.ts).y(vmax_line).mode({Scatter::Mode::Lines}).name("Vmax").xaxis("x2").yaxis("y2").legend("legend2").line(Scatter::Line().dash("dot").color("black")));

    using Lg = Layout::Legend;
    auto panel_legend = [](double y_top) {
        return Lg().x(1.02).y(y_top).xanchor(Lg::Xanchor::Left).yanchor(Lg::Yanchor::Top);
    };

    auto shapes = event_lines("y");
    for (auto& sh : event_lines("y2")) {
        shapes.push_back(sh);
    }

    auto layout = Layout()
                      .title([](auto& t) { t.text("PMSM FOC Current Loop — PI vs I-P (step + voltage disturbance)"); })
                      .xaxis(Layout::Xaxis().title([](auto& t) { t.text("Time (ms)"); }).anchor("y"))
                      .yaxis(Layout::Yaxis().title([](auto& t) { t.text("q-axis current (A)"); }).domain({0.55, 1.0}))
                      .xaxis(2, Layout::Xaxis().title([](auto& t) { t.text("Time (ms)"); }).matches("x").anchor("y2"))
                      .yaxis(2, Layout::Yaxis().title([](auto& t) { t.text("|Vdq| (V)"); }).domain({0.0, 0.45}))
                      .legend(panel_legend(1.0))
                      .legend(2, panel_legend(0.45))
                      .shapes(shapes)
                      .height(800);
    fig.setLayout(wet::move(layout));
    fig.writeHtml("foc_current_loop.html");
    fmt::print("\n  Plot written to foc_current_loop.html\n");
    return 0;
}
