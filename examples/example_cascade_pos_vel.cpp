/**
 * @file example_cascade_pos_vel.cpp
 * @brief Position / velocity cascade on a simple mass, demonstrating the
 *        library's "fancy" features end-to-end:
 *
 *  1. Generic Cascade<P, PI> composition (one line via CascadePPI<float>).
 *  2. Inner-reference clamping (the cascade caps `r_inner` -- velocity command
 *     here -- to [v_min, v_max] for slew-rate limiting).
 *  3. Inner-loop back-calculation anti-windup when the actuator (force) hits
 *     its u_min/u_max rails -- prevents integral windup during saturated
 *     transients.
 *  4. Bumpless transfer via PIDRuntimeMode::Tracking -- demonstrates an
 *     operator manual mode mid-simulation: the inner PI is disabled, a
 *     constant manual force is applied to the plant, then the cascade is
 *     re-engaged. The integrator is pre-loaded each Tracking tick so the
 *     first Auto command matches the manual command, with no bump.
 *
 * The control loop is *unrolled* below for plotting visibility (so we can
 * log r_vel_unsat, r_vel, and the mode flag separately). In production code
 * the same three lines collapse to a single `cascade.control(r_pos, pos)`:
 *
 * @code
 * constinit CascadePPI<float> cascade(pos_loop, vel_loop, v_min, v_max);
 * for (...) {
 *     const float u = cascade.control(r_pos, pos);   // does everything below
 *     apply_to_plant(u);
 * }
 * @endcode
 *
 * The Cascade class wraps the same anti-windup propagation we do explicitly
 * here. The mode-switch logic (vel_loop.disable() / enable()) is operator
 * state and lives outside the cascade in either form.
 */

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

#include "fmt/core.h"
#include "plotlypp/figure.hpp"
#include "plotlypp/traces/scatter.hpp"
#include "wet/controllers/cascade.hpp"
#include "wet/controllers/pid.hpp"

using namespace wetmelon::control;

// ===== Position/Velocity Cascade on 1-DOF Mass =====
//
// Demonstrates Cascade<P, PI> composition, inner-reference clamping for
// slew-rate limiting, back-calculation anti-windup, and bumpless transfer
// via PIDRuntimeMode::Tracking.
//
// The control loop is unrolled below for plotting visibility (r_vel_unsat,
// r_vel, and mode are logged separately).  In production the same three
// lines collapse to `cascade.control(r_pos, pos)`.
//
// Plant:
//   x_dot = v
//   v_dot = (u - b*v) / M

// ===== Plant Parameters =====
constexpr float M = 1.0f;      // [kg]   mass
constexpr float b_visc = 0.5f; // [N*s/m] viscous friction
constexpr float Ts = 0.001f;   // [s]    1 kHz control loop

// ===== Saturation Limits =====
constexpr float u_min = -5.0f; // [N]   actuator force clamp
constexpr float u_max = 5.0f;
constexpr float v_min = -2.0f; // [m/s] cascade velocity-reference clamp
constexpr float v_max = 2.0f;
constexpr float i_min = -50.0f; // [N]   inner integrator clamp
constexpr float i_max = 50.0f;

// ===== Simulation =====
constexpr float t_end = 6.0f;

int main() {
    fmt::print("===== Cascade pos/vel example =====\n\n");
    fmt::print("Plant: M={:.1f} kg, b={:.2f} N*s/m, Ts={:.0f} us\n", M, b_visc, Ts * 1e6f);
    fmt::print("Limits: u=[{:+.0f}, {:+.0f}] N, v_cmd=[{:+.0f}, {:+.0f}] m/s\n\n", u_min, u_max, v_min, v_max);

    // Outer: P-only (stateless -- nothing to wind up)
    PController<float> pos_loop(5.0f);

    // Inner: PI with back-calculation anti-windup
    PIController<float> vel_loop{design::pid<float>(
        8.0f, 50.0f, 0.0f, Ts,
        u_min, u_max, i_min, i_max, 0.5f
    )};

    // Production-equivalent single object (not used here; loop is unrolled)
    CascadePPI<float> cascade(pos_loop, vel_loop, v_min, v_max);

    const auto N = static_cast<size_t>(t_end / Ts);

    std::vector<double> t_log, r_pos_log, pos_log, r_vel_unsat_log, r_vel_log, vel_log, u_log, mode_log;
    for (auto* v : {&t_log, &r_pos_log, &pos_log, &r_vel_unsat_log, &r_vel_log, &vel_log, &u_log, &mode_log}) {
        v->reserve(N);
    }

    float pos = 0.0f;
    float vel = 0.0f;

    for (size_t k = 0; k < N; ++k) {
        const auto t = static_cast<float>(k) * Ts;

        // Reference: chirp -> step -> return to zero
        float r_pos = 0.0f;
        if (t < 2.0f) {
            constexpr float f0 = 0.3f;
            constexpr float f1 = 1.0f;
            constexpr float T_chirp = 2.0f;
            const float     phase = 2.0f * std::numbers::pi_v<float>
                              * ((f0 * t) + (0.5f * (f1 - f0) / T_chirp * t * t));
            r_pos = 0.15f * std::sin(phase);
        } else if (t < 4.0f) {
            r_pos = 1.0f;
        }

        // Operator manual mode at t=3.0s, re-engage at t=3.5s
        if (std::abs(t - 3.0f) < Ts / 2.0f) {
            vel_loop.disable(2.0f);
        }
        if (std::abs(t - 3.5f) < Ts / 2.0f) {
            vel_loop.enable();
        }

        // Cascade tick (unrolled equivalent of cascade.control(r_pos, pos))
        const float r_vel_unsat = pos_loop.control(r_pos, pos);
        const float r_vel = std::clamp(r_vel_unsat, v_min, v_max);
        const float u = vel_loop.control(r_vel, vel);

        // Plant integration (Forward Euler)
        const float accel = (u - (b_visc * vel)) / M;
        pos += vel * Ts;
        vel += accel * Ts;

        t_log.push_back(t);
        r_pos_log.push_back(r_pos);
        pos_log.push_back(pos);
        r_vel_unsat_log.push_back(r_vel_unsat);
        r_vel_log.push_back(r_vel);
        vel_log.push_back(vel);
        u_log.push_back(u);
        mode_log.push_back(vel_loop.is_enabled() ? 1.0 : 0.0);
    }

    // Console snapshots at phase boundaries
    auto idx = [&](float t) { return static_cast<size_t>(t / Ts); };

    fmt::print("Phase 1 (chirp tracking, no clamping)\n");
    fmt::print("  t=1.000s  r_pos={:+.3f}  pos={:+.3f}  v_cmd={:+.3f}  u={:+.3f}\n", r_pos_log[idx(1.0f)], pos_log[idx(1.0f)], r_vel_log[idx(1.0f)], u_log[idx(1.0f)]);

    fmt::print("\nPhase 2 (step to 1m, velocity clamped)\n");
    fmt::print("  t=2.005s  r_pos={:+.3f}  pos={:+.3f}  v_cmd_unsat={:+.3f}  v_cmd={:+.3f}  u={:+.3f}\n", r_pos_log[idx(2.005f)], pos_log[idx(2.005f)], r_vel_unsat_log[idx(2.005f)], r_vel_log[idx(2.005f)], u_log[idx(2.005f)]);

    fmt::print("\nPhase 3 (operator manual mode, 2.0 N)\n");
    fmt::print("  t=3.200s  r_pos={:+.3f}  pos={:+.3f}  u={:+.3f}  mode=Tracking\n", r_pos_log[idx(3.2f)], pos_log[idx(3.2f)], u_log[idx(3.2f)]);

    fmt::print("\nBumpless re-engage at t=3.5s\n");
    fmt::print("  t=3.500s  u={:+.3f}  (last Tracking tick)\n", u_log[idx(3.5f) - 1]);
    fmt::print("  t=3.501s  u={:+.3f}  (first Auto tick)\n", u_log[idx(3.5f) + 1]);

    fmt::print("\nPhase 4 (return to zero)\n");
    fmt::print("  t=4.005s  r_pos={:+.3f}  pos={:+.3f}  v_cmd={:+.3f}  u={:+.3f}\n", r_pos_log[idx(4.005f)], pos_log[idx(4.005f)], r_vel_log[idx(4.005f)], u_log[idx(4.005f)]);

    // ===== Plot =====
    using namespace plotlypp;
    Figure fig;

    // Row 1: position
    fig.addTrace(Scatter().x(t_log).y(r_pos_log).mode({Scatter::Mode::Lines}).name("r_pos"));
    fig.addTrace(Scatter().x(t_log).y(pos_log).mode({Scatter::Mode::Lines}).name("pos"));

    // Row 2: velocity
    fig.addTrace(Scatter().x(t_log).y(r_vel_unsat_log).mode({Scatter::Mode::Lines}).name("v_cmd (unsat)").xaxis("x2").yaxis("y2"));
    fig.addTrace(Scatter().x(t_log).y(r_vel_log).mode({Scatter::Mode::Lines}).name("v_cmd (sat)").xaxis("x2").yaxis("y2"));
    fig.addTrace(Scatter().x(t_log).y(vel_log).mode({Scatter::Mode::Lines}).name("vel").xaxis("x2").yaxis("y2"));

    // Row 3: force
    fig.addTrace(Scatter().x(t_log).y(u_log).mode({Scatter::Mode::Lines}).name("u (force)").xaxis("x3").yaxis("y3"));

    // Row 4: mode flag
    fig.addTrace(Scatter().x(t_log).y(mode_log).mode({Scatter::Mode::Lines}).name("enabled").xaxis("x4").yaxis("y4"));

    fig.setLayout(Layout().title([](auto& t) { t.text("Cascade pos/vel: chirp + velocity-clamped step + bumpless manual mode"); }).grid(Layout::Grid().rows(4).columns(1).pattern(Layout::Grid::Pattern::Independent).roworder(Layout::Grid::Roworder::TopToBottom)).xaxis(Layout::Xaxis().title([](auto& t) { t.text(""); })).yaxis(Layout::Yaxis().title([](auto& t) { t.text("Position [m]"); })).xaxis(2, Layout::Xaxis().title([](auto& t) { t.text(""); })).yaxis(2, Layout::Yaxis().title([](auto& t) { t.text("Velocity [m/s]"); })).xaxis(3, Layout::Xaxis().title([](auto& t) { t.text(""); })).yaxis(3, Layout::Yaxis().title([](auto& t) { t.text("Force [N]"); })).xaxis(4, Layout::Xaxis().title([](auto& t) { t.text("Time [s]"); })).yaxis(4, Layout::Yaxis().title([](auto& t) { t.text("Enabled"); })));

    fig.writeHtml("cascade_pos_vel.html");
    fmt::print("\nPlot written to cascade_pos_vel.html\n");

    return 0;
}
