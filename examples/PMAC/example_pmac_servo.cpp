/**
 * @file example_pmac_servo.cpp
 * @brief End-to-end PMAC servo: bandwidth tuning, R/L calibration, and the three
 *        control modes on an average dq + mechanical plant.
 *
 * Drives wet::motor::PmacServo through commissioning and Torque / Velocity / Position
 * modes, plus a thermal-derate event, on a simulated PMSM. Shows the full
 * "set bandwidths → calibrate → run" workflow with no raw control gains.
 */

#include <cmath>
#include <numbers>
#include <vector>

#include "fmt/base.h"
#include "fmt/core.h"
#include "plotlypp/figure.hpp"
#include "plotlypp/layout/layout.hpp"
#include "plotlypp/traces/scatter.hpp"
#include "wet/backend.hpp"
#include "wet/design/pid_design.hpp"
#include "wet/matrix/colvec.hpp"
#include "wet/motor/calibration.hpp"
#include "wet/motor/foc.hpp"
#include "wet/motor/servo.hpp"
#include "wet/simulation/integrator.hpp"
#include "wet/transforms.hpp"

using namespace wet;
using namespace wet::motor;

namespace {
using std::numbers::pi_v;

// True motor parameters (unknown to the controller until calibration / config).
constexpr double Ld = 200e-6, Lq = 200e-6, R = 0.5, lambda = 0.01;
constexpr double pole_pairs = 4.0, J = 2e-4, b = 1e-3;
constexpr float  Vdc = 48.0f;
constexpr double Ts = 1.0 / 20000.0; // 20 kHz
const double     Kt = 1.5 * pole_pairs * lambda;

// Average dq PMSM + 1-DOF mechanical plant. The dq cross-coupling makes this
// nonlinear (not an LTI StateSpace), so it's integrated with the library's generic
// f(t,x) ODE solver — one RK4 step per control tick, no hand-rolled substepping.
struct Plant {
    ColVec<4, double>   x{}; // [id, iq, w, theta]
    double              tau_load{0};
    sim::RK4<4, double> solver{};

    [[nodiscard]] double id() const { return x[0]; }
    [[nodiscard]] double iq() const { return x[1]; }
    [[nodiscard]] double w() const { return x[2]; }
    [[nodiscard]] double th() const { return x[3]; }

    void step(double vd, double vq, double dt) {
        auto dynamics = [&](double, const ColVec<4, double>& s) {
            const double we = pole_pairs * s[2];
            return ColVec<4, double>{
                (vd - (R * s[0]) + (we * Lq * s[1])) / Ld,                        // d(id)/dt
                (vq - (R * s[1]) - (we * Ld * s[0]) - (we * lambda)) / Lq,        // d(iq)/dt
                ((1.5 * pole_pairs * lambda * s[1]) - tau_load - (b * s[2])) / J, // d(w)/dt
                s[2],                                                             // d(theta)/dt = w
            };
        };
        x = solver.evolve(dynamics, x, 0.0, dt).x;
    }
};

// Decimated time series of the closed-loop run, for plotting.
struct Recorder {
    std::vector<double> t, iq, w, th;
    void                sample(double time, const Plant& p) {
        t.push_back(time);
        iq.push_back(p.iq());
        w.push_back(p.w());
        th.push_back(p.th());
    }
};

// One closed-loop tick: plant currents → abc feedback → servo → duties → dq voltage.
void tick(PmacServo<float>& servo, Plant& p, float pos, float vel, float trq) {
    const float theta_e = static_cast<float>(pole_pairs * p.th());
    const auto  Iabc = inverse_park_clarke_transform(
        DirectQuadrature<float>{static_cast<float>(p.id()), static_cast<float>(p.iq())}, theta_e
    );

    const auto res = servo.update(pos, vel, trq, Iabc, Vdc, ((float)p.th() / (2.0f * pi_v<float>)), Ts);

    const ColVec<3, float> Vabc{(res.duties[0] - 0.5f) * Vdc, (res.duties[1] - 0.5f) * Vdc, (res.duties[2] - 0.5f) * Vdc};
    const auto             Vdq = clarke_park_transform(Vabc, theta_e);
    p.step(Vdq.d, Vdq.q, Ts);
}

// Run for `seconds`, optionally logging every `dec`-th tick into `rec` at absolute time t0+k*Ts.
void run(PmacServo<float>& servo, Plant& p, double seconds, float pos, float vel, float trq, Recorder* rec = nullptr, double t0 = 0.0, int dec = 10) {
    const int steps = static_cast<int>(seconds / Ts);
    for (int k = 0; k < steps; ++k) {
        tick(servo, p, pos, vel, trq);
        if (rec != nullptr && k % dec == 0) {
            rec->sample(t0 + (k * Ts), p);
        }
    }
}

} // namespace

int main() {
    // --- 0. Tune by bandwidth (rad/s), never raw gains ----------------------
    const CascadeBandwidths<float> bw{
        .omega_position = 2.0f * std::numbers::pi_v<float> * 5.0f,  // 5 Hz
        .omega_velocity = 2.0f * std::numbers::pi_v<float> * 50.0f, // 50 Hz
        .omega_current = 2.0f * std::numbers::pi_v<float> * 1000.0f
    }; // 1 kHz
    fmt::print("Cascade bandwidths (Hz): pos=5  vel=50  cur=1000   separated={}\n", bw.valid());

    const auto cur = design::current_loop_pi(Ld, R, static_cast<double>(bw.omega_current));
    const auto vel = design::pi_pole_placement_first_order(J, b, static_cast<double>(bw.omega_velocity));
    fmt::print("  -> current PI  Kp={:.4f}  Ki={:.1f}\n", cur.Kp, cur.Ki);
    fmt::print("  -> velocity PI Kp={:.4f}  Ki={:.4f}\n", vel.Kp, vel.Ki);
    fmt::print("  -> position P  Kp={:.2f}\n\n", bw.omega_position);

    // --- 1. Commission phase R and L by RLS (PRBS injection) -----------------
    {
        // Exact discrete d-axis plant the calibrator should recover.
        const double a = std::exp(-R * Ts / Ld);
        const double bd = (1.0 - a) / R;
        double       i = 0.0;

        PhaseParameterCalibrator<double> cal{PhaseCalibrationConfig<double>{.inject_voltage = 2.0, .duration_s = 0.2}};
        for (;;) {
            const auto cmd = cal.step(i, Ts);
            i = (a * i) + (bd * cmd.v_d);
            if (cmd.done) {
                break;
            }
        }
        fmt::print("Calibration:  R = {:.4f} ohm (true {:.4f}),  L = {:.2f} uH (true {:.2f})\n\n", cal.resistance(), R, cal.inductance() * 1e6, Ld * 1e6);
    }

    // --- Build the servo from the (now known) parameters --------------------
    PmacServoConfig<float> cfg{
        .Ldq = {static_cast<float>(Ld), static_cast<float>(Lq)},
        .R = static_cast<float>(R),
        .lambda = static_cast<float>(lambda),
        .pole_pairs = static_cast<float>(pole_pairs),
        .J = static_cast<float>(J),
        .b = static_cast<float>(b),
        .iq_max = 20.0f,
        .bandwidths = bw,
    };

    Recorder rec_torque;
    Recorder rec_vel;
    Recorder rec_pos;

    constexpr float torque_cmd = 0.12f;
    constexpr float vel_cmd = 150.0f;
    constexpr float pos_cmd = 10.0f;

    // --- 2. Torque mode -----------------------------------------------------
    {
        PmacServo<float> servo{cfg};
        Plant            p;
        servo.set_mode(ControlMode::Torque);
        run(servo, p, 0.2, 0.0f, 0.0f, torque_cmd, &rec_torque); // Nm
        fmt::print("Torque mode:   cmd 0.12 Nm -> iq {:.3f} A (target {:.3f}),  speed {:.1f} rad/s\n", p.iq(), 0.12 / Kt, p.w());
    }

    // --- 3. Velocity mode with a mid-run load step --------------------------
    {
        PmacServo<float> servo{cfg};
        Plant            p;
        servo.set_mode(ControlMode::Velocity);
        const float vel_turns = vel_cmd / (2.0f * pi_v<float>); // 150 rad/s -> turns/s
        run(servo, p, 0.4, 0.0f, vel_turns, 0.0f, &rec_vel);
        const double w_noload = p.w();
        p.tau_load = 0.06; // Nm load step
        run(servo, p, 0.4, 0.0f, vel_turns, 0.0f, &rec_vel, 0.4);
        fmt::print(
            "Velocity mode: cmd 150 rad/s -> {:.1f} (no load), {:.1f} (after 0.06 Nm load step, "
            "rejected by the velocity PI)\n",
            w_noload, p.w()
        );
    }

    // --- 4. Position mode ---------------------------------------------------
    {
        PmacServo<float> servo{cfg};
        Plant            p;
        servo.set_mode(ControlMode::Position);
        const float pos_turns = pos_cmd / (2.0f * pi_v<float>); // 10 rad -> turns
        run(servo, p, 2.0, pos_turns, 0.0f, 0.0f, &rec_pos);
        fmt::print("Position mode: cmd 10 rad -> {:.3f} rad,  speed {:.3f} rad/s (settled)\n", p.th(), p.w());
    }

    // --- 5. Thermal derate shrinks the current ceiling ----------------------
    {
        PmacServo<float> servo{cfg};
        Plant            p;
        servo.set_mode(ControlMode::Torque);
        servo.set_thermal_scale(0.2f);        // 20% derate -> ceiling 4 A
        run(servo, p, 0.1, 0.0f, 0.0f, 5.0f); // demands far more current than allowed
        fmt::print("Thermal derate: 20%% scale on iq_max=20 -> iq held to {:.2f} A (ceiling 4 A)\n", p.iq());
    }

    // --- 6. Plot the three modes' tracking, one panel each -------------------
    using namespace plotlypp;
    auto setpoint = [](const std::vector<double>& t2, double v) { return std::vector<double>(t2.size(), v); };

    Figure fig;
    // Position panel (x/y): angle tracking a 10 rad step.
    fig.addTrace(Scatter().x(rec_pos.t).y(rec_pos.th).mode({Scatter::Mode::Lines}).name("theta").legend("legend"));
    fig.addTrace(Scatter().x(rec_pos.t).y(setpoint(rec_pos.t, static_cast<double>(pos_cmd))).mode({Scatter::Mode::Lines}).name("command").legend("legend").line(Scatter::Line().dash("dash")));
    // Velocity panel (x2/y2): speed tracking 150 rad/s with a load step at 0.4 s.
    fig.addTrace(Scatter().x(rec_vel.t).y(rec_vel.w).mode({Scatter::Mode::Lines}).name("speed").xaxis("x2").yaxis("y2").legend("legend2"));
    fig.addTrace(Scatter().x(rec_vel.t).y(setpoint(rec_vel.t, static_cast<double>(vel_cmd))).mode({Scatter::Mode::Lines}).name("command").xaxis("x2").yaxis("y2").legend("legend2").line(Scatter::Line().dash("dash")));
    // Torque panel (x3/y3): iq tracking the 0.12 Nm command.
    fig.addTrace(Scatter().x(rec_torque.t).y(rec_torque.iq).mode({Scatter::Mode::Lines}).name("iq").xaxis("x3").yaxis("y3").legend("legend3"));
    fig.addTrace(Scatter().x(rec_torque.t).y(setpoint(rec_torque.t, static_cast<double>(torque_cmd) / Kt)).mode({Scatter::Mode::Lines}).name("command").xaxis("x3").yaxis("y3").legend("legend3").line(Scatter::Line().dash("dash")));

    using Lg = Layout::Legend;
    auto panel_legend = [](double y_top) { return Lg().x(1.02).y(y_top).xanchor(Lg::Xanchor::Left).yanchor(Lg::Yanchor::Top); };
    auto layout = Layout()
                      .title([](auto& t2) { t2.text("PMAC Servo — Position, Velocity, and Torque tracking"); })
                      .xaxis(Layout::Xaxis().anchor("y"))
                      .yaxis(Layout::Yaxis().title([](auto& t2) { t2.text("Angle (rad)"); }).domain({0.70, 1.0}))
                      .xaxis(2, Layout::Xaxis().anchor("y2"))
                      .yaxis(2, Layout::Yaxis().title([](auto& t2) { t2.text("Speed (rad/s)"); }).domain({0.37, 0.63}))
                      .xaxis(3, Layout::Xaxis().title([](auto& t2) { t2.text("Time (s)"); }).anchor("y3"))
                      .yaxis(3, Layout::Yaxis().title([](auto& t2) { t2.text("iq (A)"); }).domain({0.0, 0.30}))
                      .legend(panel_legend(1.0))
                      .legend(2, panel_legend(0.63))
                      .legend(3, panel_legend(0.30))
                      .height(1000);
    fig.setLayout(wet::move(layout));
    fig.writeHtml("pmac_servo.html");
    fmt::print("  Plot written to pmac_servo.html\n");

    return 0;
}
