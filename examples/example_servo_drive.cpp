#include <cmath>
#include <numbers>

#include "constexpr_math.hpp"
#include "fmt/core.h"
#include "motor_control.hpp"
#include "pid.hpp"
#include "plot_plotly.hpp"
#include "simulate.hpp"
#include "solver.hpp"

using namespace wetmelon::control;

// ===== PMSM 3-Phase Servo Drive with Two-Mass Mechanical Load =====
//
// Electrical model (dq reference frame, surface-mount PMSM: Ld = Lq = Ls):
//   di_d/dt = (v_d - Rs*i_d + omega_e*Ls*i_q) / Ls
//   di_q/dt = (v_q - Rs*i_q - omega_e*Ls*i_d - omega_e*lambda_pm) / Ls
//
// Electromagnetic torque (non-salient, no reluctance torque):
//   Te = 1.5 * P * lambda_pm * i_q
//
// Two-mass mechanical model (motor + flexible coupling + load):
//   Motor: J_m * d(omega_m)/dt = Te - B_m*omega_m - T_shaft
//   Load:  J_L * d(omega_L)/dt = T_shaft - B_L*omega_L - T_coulomb - T_ext
//   Shaft: T_shaft = K_s*(theta_m - theta_L) + B_s*(omega_m - omega_L)
//
// State vector x = [i_d, i_q, omega_m, theta_m, omega_L, theta_L]
// Input vector u = [v_d, v_q]
//
// Control architecture: P-PI-PI cascaded FOC
//   Inner loop:  PI current regulators (d-axis and q-axis), bandwidth-designed
//   Middle loop: PI speed regulator with explicit gains
//   Outer loop:  P position controller with speed limiting
//   Feedforward: dq cross-coupling decoupling and back-EMF compensation

// ===== Motor Parameters (typical 400W surface-mount PMSM servo) =====
constexpr double Rs = 1.2;        // Stator resistance [Ohm]
constexpr double Ls = 4.7e-3;     // Stator inductance [H] (Ld = Lq for surface-mount)
constexpr double lambda_pm = 0.1; // PM flux linkage [Wb]
constexpr int    P = 4;           // Pole pairs
constexpr double Jm = 5e-5;       // Motor rotor inertia [kg*m^2]
constexpr double Bm = 1e-4;       // Motor viscous friction [Nm*s/rad]

// ===== Coupling Parameters (flexible shaft) =====
constexpr double Ks = 5000.0; // Shaft torsional stiffness [Nm/rad]
constexpr double Bs = 0.05;   // Shaft torsional damping [Nm*s/rad]

// ===== Load Parameters =====
constexpr double JL = 5e-4; // Load inertia [kg*m^2] (10x motor)
constexpr double BL = 1e-3; // Load viscous friction [Nm*s/rad]
constexpr double Tc = 0.02; // Coulomb friction [Nm]

// ===== Derived Constants =====
constexpr double KT = 1.5 * P * lambda_pm; // Torque constant [Nm/A]
constexpr double Vdc = 48.0;               // DC bus voltage [V]

// ===== Control Design =====
// Current loop: PI with pole-zero cancellation, bandwidth-based design
//   Plant: G(s) = 1/(sLs + Rs)
//   PI: Kp = wc*Ls, Ki = wc*Rs
constexpr double bw_current = 1000.0; // Current loop bandwidth [Hz]
constexpr double wc_current = 2.0 * std::numbers::pi * bw_current;
constexpr double Kp_i = wc_current * Ls;
constexpr double Ki_i = wc_current * Rs;

// Speed loop: explicit PI gains
constexpr double Kp_speed = 0.3;
constexpr double Ki_speed = 18.0;
constexpr double i_max = 10.0; // Current limit [A]

// Position loop: P controller with speed limiting
constexpr double Kp_position = 60.0; // Position P gain [rad/s per rad]
constexpr double speed_max = 500.0;  // Speed command limit [rad/s]

// Simulation timestep (10 kHz control rate)
constexpr double dt = 100e-6;

// Loop decimation (all loops at same rate for this example)
constexpr int speed_ratio = 1;
constexpr int position_ratio = 1;

int main() {
    fmt::print("===== PMSM Servo Drive Simulation (P-PI-PI) =====\n\n");
    fmt::print("Motor:   Rs={:.2f} Ohm, Ls={:.1f} mH, lambda_pm={:.2f} Wb, P={}\n", Rs, Ls * 1e3, lambda_pm, P);
    fmt::print("         Jm={:.1e} kg*m^2, KT={:.3f} Nm/A\n", Jm, KT);
    fmt::print("Coupling: Ks={:.0f} Nm/rad, Bs={:.3f} Nm*s/rad\n", Ks, Bs);
    fmt::print("Load:    JL={:.1e} kg*m^2, BL={:.1e}, Tc={:.3f} Nm\n", JL, BL, Tc);
    fmt::print("Control: Current PI  Kp={:.2f}, Ki={:.1f}  (BW={:.0f} Hz)\n", Kp_i, Ki_i, bw_current);
    fmt::print("         Speed PI    Kp={:.4f}, Ki={:.2f}\n", Kp_speed, Ki_speed);
    fmt::print("         Position P  Kp={:.1f}, speed_max={:.0f} rad/s\n", Kp_position, speed_max);
    fmt::print("         Vdc={:.0f} V, dt={:.0f} us\n\n", Vdc, dt * 1e6);

    // Create PI controllers
    const double          v_lim = Vdc / 2.0;
    PIDController<double> pi_d{online::pid(Kp_i, Ki_i, 0.0, dt, -v_lim, v_lim, -v_lim / Ki_i, v_lim / Ki_i, Ki_i)};
    PIDController<double> pi_q{online::pid(Kp_i, Ki_i, 0.0, dt, -v_lim, v_lim, -v_lim / Ki_i, v_lim / Ki_i, Ki_i)};
    PIDController<double> pi_spd{online::pid(Kp_speed, Ki_speed, 0.0, dt * speed_ratio, -i_max, i_max, -i_max / Ki_speed, i_max / Ki_speed, Ki_speed)};

    // Controller state
    double theta_ref = 0.0; // Position reference [rad]
    int    speed_decim = 0;
    int    pos_decim = 0;
    double iq_ref = 0.0;
    double omega_ref_cmd = 0.0; // Speed command from position loop

    // Nonlinear plant dynamics: dx/dt = f(t, x, u)
    auto plant = [](double t, const ColVec<6>& x, const ColVec<2>& u) -> ColVec<6> {
        const double id = x(0, 0);
        const double iq = x(1, 0);
        const double omega_m = x(2, 0);
        const double theta_m = x(3, 0);
        const double omega_L = x(4, 0);
        const double theta_L = x(5, 0);
        const double vd = u(0, 0);
        const double vq = u(1, 0);

        // Electrical speed
        const double omega_e = P * omega_m;

        // dq current dynamics (average voltage model)
        const double did_dt = (vd - Rs * id + omega_e * Ls * iq) / Ls;
        const double diq_dt = (vq - Rs * iq - omega_e * Ls * id - omega_e * lambda_pm) / Ls;

        // Electromagnetic torque (non-salient)
        const double Te = 1.5 * P * lambda_pm * iq;

        // Shaft coupling torque
        const double twist = theta_m - theta_L;
        const double T_shaft = Ks * twist + Bs * (omega_m - omega_L);

        // Motor rotor dynamics
        const double domega_m = (Te - Bm * omega_m - T_shaft) / Jm;

        // Load dynamics with Coulomb friction (smooth tanh approximation)
        const double T_coulomb = Tc * std::tanh(omega_L * 100.0);

        // External load torque disturbance: 0.3 Nm step at t=1.0s
        const double T_ext = (t > 1.0) ? 0.3 : 0.0;

        const double domega_L = (T_shaft - BL * omega_L - T_coulomb - T_ext) / JL;

        return ColVec<6>{{did_dt}, {diq_dt}, {domega_m}, {omega_m}, {domega_L}, {omega_L}};
    };

    // Output: full state for recording
    auto output = [](const ColVec<6>& x) -> ColVec<6> { return x; };

    // P-PI-PI cascaded FOC controller
    auto controller = [&](const ColVec<6>& x) -> ColVec<2> {
        const double id = x(0, 0);
        const double iq = x(1, 0);
        const double omega_m = x(2, 0);
        const double theta_L = x(5, 0);

        // Position loop (P controller with speed limiting)
        if (++pos_decim >= position_ratio) {
            pos_decim = 0;
            double pos_err = theta_ref - theta_L;
            double cmd = Kp_position * pos_err;
            if (cmd > speed_max) {
                cmd = speed_max;
            }
            if (cmd < -speed_max) {
                cmd = -speed_max;
            }
            omega_ref_cmd = cmd;
        }

        // Speed loop (PI, decimated)
        if (++speed_decim >= speed_ratio) {
            speed_decim = 0;
            iq_ref = pi_spd.control(omega_ref_cmd - omega_m);
        }

        // Current loops
        double vd = pi_d.control(0.0 - id);
        double vq = pi_q.control(iq_ref - iq);

        // Cross-coupling decoupling and back-EMF feedforward
        const double omega_e = P * omega_m;
        vd -= omega_e * Ls * iq;
        vq += omega_e * Ls * id + omega_e * lambda_pm;

        return ColVec<2>{{vd}, {vq}};
    };

    // RK45 adaptive solver
    RK45<6>            rk45;
    AdaptiveStepSolver solver(rk45, dt, 1e-8, 1e-8, 200e-6, 1000000);

    // Initial condition: motor at rest
    ColVec<6> x0{};

    // Scenario: position step to 2π rad, hold, then load disturbance at t=1.0s
    theta_ref = 2.0 * std::numbers::pi;

    fmt::print("Simulating 2 seconds (position step to {:.2f} rad, load disturbance at t=1.0s)...\n", theta_ref);
    auto sim = simulate_state_feedback<6, 2, 6>(plant, output, controller, solver, x0, {0.0, 2.0});

    fmt::print("Simulated {} time steps\n", sim.t.size());
    fmt::print("Final load position: {:.3f} rad (ref: {:.3f} rad)\n", sim.x.back()(5, 0), theta_ref);
    fmt::print("Final motor speed:   {:.1f} rad/s\n", sim.x.back()(2, 0));
    fmt::print("Final load speed:    {:.1f} rad/s\n", sim.x.back()(4, 0));

    // ===== Custom plot with descriptive trace names =====
    using namespace plotlypp;

    auto   t = plot::to_double_vector(sim.t);
    Figure fig;

    // Row 1: Position tracking
    fig.addTrace(Scatter().x(t).y(plot::extract_channel(sim.x, 5)).mode({Scatter::Mode::Lines}).name("Load position θ_L"));
    fig.addTrace(Scatter().x(t).y(plot::extract_channel(sim.x, 3)).mode({Scatter::Mode::Lines}).name("Motor position θ_m"));
    {
        std::vector<double> ref(t.size(), theta_ref);
        fig.addTrace(Scatter().x(t).y(ref).mode({Scatter::Mode::Lines}).name("Reference θ*").line(Scatter::Line().dash("dash")));
    }

    // Row 2: Speed
    fig.addTrace(Scatter().x(t).y(plot::extract_channel(sim.x, 2)).mode({Scatter::Mode::Lines}).name("Motor speed ω_m").xaxis("x2").yaxis("y2"));
    fig.addTrace(Scatter().x(t).y(plot::extract_channel(sim.x, 4)).mode({Scatter::Mode::Lines}).name("Load speed ω_L").xaxis("x2").yaxis("y2"));

    // Row 3: dq currents
    fig.addTrace(Scatter().x(t).y(plot::extract_channel(sim.x, 0)).mode({Scatter::Mode::Lines}).name("i_d").xaxis("x3").yaxis("y3"));
    fig.addTrace(Scatter().x(t).y(plot::extract_channel(sim.x, 1)).mode({Scatter::Mode::Lines}).name("i_q").xaxis("x3").yaxis("y3"));

    // Row 4: Applied voltages
    fig.addTrace(Scatter().x(t).y(plot::extract_channel(sim.u, 0)).mode({Scatter::Mode::Lines}).name("v_d").xaxis("x4").yaxis("y4"));
    fig.addTrace(Scatter().x(t).y(plot::extract_channel(sim.u, 1)).mode({Scatter::Mode::Lines}).name("v_q").xaxis("x4").yaxis("y4"));

    // Row 5: Shaft twist angle
    {
        auto                theta_m = plot::extract_channel(sim.x, 3);
        auto                theta_L = plot::extract_channel(sim.x, 5);
        std::vector<double> twist(t.size());
        for (size_t i = 0; i < t.size(); ++i) {
            twist[i] = theta_m[i] - theta_L[i];
        }
        fig.addTrace(Scatter().x(t).y(twist).mode({Scatter::Mode::Lines}).name("Shaft twist (θ_m − θ_L)").xaxis("x5").yaxis("y5"));
    }

    auto layout = Layout()
                      .title([](auto& t) { t.text("PMSM Servo Drive — P-PI-PI Position Control + Load Disturbance @ t=1.0s"); })
                      .grid(Layout::Grid().rows(5).columns(1).pattern(Layout::Grid::Pattern::Independent).roworder(Layout::Grid::Roworder::TopToBottom))
                      .xaxis(Layout::Xaxis().title([](auto& t) { t.text("Time (s)"); }))
                      .yaxis(Layout::Yaxis().title([](auto& t) { t.text("Position (rad)"); }))
                      .xaxis(2, Layout::Xaxis().title([](auto& t) { t.text("Time (s)"); }))
                      .yaxis(2, Layout::Yaxis().title([](auto& t) { t.text("Speed (rad/s)"); }))
                      .xaxis(3, Layout::Xaxis().title([](auto& t) { t.text("Time (s)"); }))
                      .yaxis(3, Layout::Yaxis().title([](auto& t) { t.text("Current (A)"); }))
                      .xaxis(4, Layout::Xaxis().title([](auto& t) { t.text("Time (s)"); }))
                      .yaxis(4, Layout::Yaxis().title([](auto& t) { t.text("Voltage (V)"); }))
                      .xaxis(5, Layout::Xaxis().title([](auto& t) { t.text("Time (s)"); }))
                      .yaxis(5, Layout::Yaxis().title([](auto& t) { t.text("Twist (rad)"); }))
                      .height(1200);

    fig.setLayout(std::move(layout));
    fig.writeHtml("servo_drive.html");
    fmt::print("\nPlot written to servo_drive.html\n");

    return 0;
}
