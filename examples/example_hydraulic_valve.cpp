#include <fmt/core.h>

#include <algorithm>
#include <plotlypp/figure.hpp>
#include <plotlypp/trace.hpp>
#include <plotlypp/traces/scatter.hpp>

#include "control.hpp"
#include "integrator.hpp"
#include "types.hpp"

// Monolithic simulation example for the hydraulic pressure-reducing valve.
// Replace `IntegratorT` with a stiff/implicit integrator available in this repo
// (e.g. an implicit Euler / Rosenbrock / CVODE wrapper) for realistic stiff solves.

using namespace control;
using namespace plotlypp;

// Combined state vector:
// x = [ i, z, x_spool, v_spool, P ]
//  i       : solenoid current [A] (electrical LTI)
//  z       : integrator state for PI controller
//  x_spool : spool position [m]
//  v_spool : spool velocity [m/s]
//  P       : chamber pressure [Pa]

template <typename IntegratorT>
SolveResult simulate_valve_monolithic(double t0, double tf, double dt) {
    // Mechanical and hydraulic parameters
    const double m = 0.1;   // kg
    const double b = 10.0;  // structural viscous damping [N/(m/s)]
    // Additional hydraulic damping (flow-induced / fluid damping) to increase
    // spool damping for stability. Tuned larger to suppress slamming.
    const double b_hydraulic = 2000.0;  // N/(m/s)

    // Quadratic (velocity^2) damping coeff (flow separation / nonlinear drag)
    const double b_quad = 1e3;  // N/(m^2) multiplier used as b_quad * |v| * v

    // Coulomb friction (static friction) [N]
    const double F_coulomb = 0.0;     // N
    const double k         = 1000.0;  // N/m

    // Spool face area (for hydraulic force/volume) and orifice flow area
    const double area_spool = 1e-4;  // m^2  (spool pressure face area)

    // Orifice geometry: supply-side and tank-side maximum areas
    const double A_supply_open = 1e-4;   // m^2 (max open to supply)
    const double A_tank_open   = 1e-4;   // m^2 (max open to tank)
    const double spool_stroke  = 1e-3;   // m    (opening travel from closed->open)
    const double beta          = 1.5e9;  // Pa
    const double V0            = 5e-5;   // m^3
    const double Cd            = 0.7;
    const double rho           = 850.0;  // kg/m^3
    const double Ps            = 50e6;   // Pa (supply)
    const double P_atm         = 1e5;    // Pa

    // Electrical solenoid parameters (simple RL)
    const double L = 10e-3;  // H
    const double R = 2.4;    // Ohm

    // Force per ampere (N/A) — choose realistic actuator constant so required
    // current fits within voltage/current limits. With V_max=12V and R=1Ω,
    // 12A * Kf should produce ≈ required hydraulic force. Set Kf≈250 N/A.
    const double Kf = 250.0;  // N/A (realistic actuator constant)

    // Solenoid voltage limits (V)
    const double V_min = 0.0;   // 0 V
    const double V_max = 12.0;  // 12 V

    // Controller gains (PI) — choose for critical damping of the electrical loop
    // Characteristic polynomial: L s^2 + (R+Kp) s + Ki
    // For critical damping: R+Kp = 2*sqrt(L*Ki)
    // Choose closed-loop natural frequency based on electrical time-constant tau_e = L/R
    double tau_e = L / R;
    double wn    = 1.0 / tau_e;  // target natural frequency (rad/s)
    double Ki    = L * wn * wn;
    double Kp    = 2.0 * std::sqrt(L * Ki) - R;
    fmt::print("PI gains set for critical damping: Kp={:.6g}, Ki={:.6g}\n", Kp, Ki);

    // Initial states
    double i0       = 0.0;  // initial current [A]
    double iterm0   = 0.0;  // initial integrator state
    double x_spool0 = 0.0;  // initial spool position [m]
    double v_spool0 = 0.0;  // initial spool velocity [m/s]
    double P0       = 0.0;  // initial chamber pressure [Pa]
    ColVec x0{i0, iterm0, x_spool0, v_spool0, P0};

    // Reference spool position (step)
    const double i_ref = 800e-3;  // 800 mA

    // Right-hand side of the combined system
    auto rhs = [&](double /*t*/, const ColVec& x) -> ColVec {
        double i     = x(0);
        double iterm = x(1);
        double xs    = x(2);
        double vs    = x(3);
        double P     = x(4);

        // PI controller (voltage output)
        double err        = i_ref - i;
        double Va         = Kp * err + Ki * iterm;  // controller output voltage
        double Va_limited = std::clamp(Va, V_min, V_max);

        // Electrical LTI: L di/dt + R*i = Va_limited  --> di/dt = (-R/L)*i + (1/L)*Va_limited
        double di = (-R / L) * i + (1.0 / L) * Va_limited;

        // Solenoid force (nonlinear coupling: proportional to current)
        double Fs = Kf * i;

        // Hydraulic flow through orifice (nonlinear): Q = Cd * A * sqrt(2/ρ * (Ps - P))
        // ensure non-negative inside sqrt
        // Clamp spool position to physical stroke [0, spool_stroke]
        double xs_clamped = std::clamp(xs, 0.0, spool_stroke);

        // Orifice area varies with spool position (0..spool_stroke)
        double f = xs_clamped / spool_stroke;  // normalized 0..1

        // Map position to supply/tank orifice fractions so that:
        // f=0   -> tank fully open (1.0), supply closed (0.0)
        // f=0.5 -> both slightly open (0.05)
        // f=1   -> supply fully open (1.0), tank closed (0.0)
        double tank_frac, supply_frac;
        if (f <= 0.5) {
            tank_frac   = 1.0 - 1.9 * f;  // linear from 1 -> 0.05 over [0,0.5]
            supply_frac = 0.1 * f;        // linear from 0 -> 0.05 over [0,0.5]
        } else {
            tank_frac   = 0.1 * (1.0 - f);  // linear from 0.05 -> 0 over [0.5,1]
            supply_frac = 1.9 * f - 0.9;    // linear from 0.05 -> 1 over [0.5,1]
        }

        double A_supply = A_supply_open * supply_frac;
        double A_tank   = A_tank_open * tank_frac;

        // Spool dynamics: m*dv/dt = Fs - b*v - k*x - area*(P - P_atm)
        double F_quad = b_quad * vs * std::abs(vs);
        double sgn_vs = (vs > 0.0) ? 1.0 : ((vs < 0.0) ? -1.0 : 0.0);
        double F_fric = (sgn_vs == 0.0) ? 0.0 : (F_coulomb * sgn_vs);

        double dv = (Fs - (b + b_hydraulic) * vs - F_quad - F_fric - k * xs_clamped - area_spool * (P - P_atm)) / m;

        // Chamber pressure dynamics (mass balance approximation):
        // V = V0 + area * x (assume small change in volume with spool position)
        double V = V0 + area_spool * xs_clamped;
        if (V <= V0) V = V0;

        // Flow continuity: dP/dt = (beta / V) * (Q_in - Q_out)
        double Qin  = Cd * A_supply * std::sqrt(2.0 / rho * std::max(Ps - P, 0.0));
        double Qout = Cd * A_tank * std::sqrt(std::max(2.0 / rho * (P - P_atm), 0.0));
        double dPdt = (beta / V) * (Qin - Qout);

        // Integrator state (z_dot = error)
        // Integrator state (z_dot = error) with simple anti-windup:
        // if the controller output is saturated and the error would drive
        // further into saturation, stop integrating.
        double dz = 0.0;
        if (!((Va > V_max && err > 0.0) || (Va < V_min && err < 0.0))) {
            dz = err;
        }

        // Prevent motion beyond limits: if at a bound and velocity would drive further out, stop it
        double dxdt = vs;
        if ((xs <= 0.0 && vs <= 0.0) || (xs >= spool_stroke && vs >= 0.0)) dxdt = 0.0;
        if ((xs <= 0.0 && dv < 0.0) || (xs >= spool_stroke && dv > 0.0)) dv = 0.0;

        ColVec xdot{di, dz, dxdt, dv, dPdt};
        return xdot;
    };

    // Solve monolithically (user must pick an appropriate integrator type)
    FixedStepSolver<IntegratorT>  solver{dt};
    auto                          start   = std::chrono::high_resolution_clock::now();
    SolveResult                   res     = solver.solve(rhs, x0, {t0, tf});
    auto                          end     = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    fmt::print("Simulation time: {:.6g} seconds\n", elapsed.count());
    return res;
}

int main() {
    fmt::print("Monolithic hydraulic valve simulation example\n");

    // NOTE: Replace `/*YourStiffIntegrator*/` with an actual integrator type from this repo
    // that implements `AdaptiveStepIntegrator` or `FixedStepIntegrator`. For example,
    // an implicit Euler or Rosenbrock integrator is appropriate for stiff problems.

    // Sim
    auto result = simulate_valve_monolithic<BDF2>(0.0, 0.05, 1e-5);

    if (!result.success) {
        fmt::print("Simulation failed! {}\n", result.message);
        return -1;
    }

    // Plot results
    Figure fig;

    // Extract time and states
    std::vector<double> t      = result.t;
    std::vector<ColVec> states = result.x;

    // Extract individual state components
    std::vector<double> current(t.size());
    std::vector<double> integrator_state(t.size());
    std::vector<double> spool_position(t.size());
    std::vector<double> spool_velocity(t.size());
    std::vector<double> pressure(t.size());
    for (size_t i = 0; i < t.size(); ++i) {
        current[i]          = states[i](0) * 1000.0;  // convert to mA
        integrator_state[i] = states[i](1);
        spool_position[i]   = states[i](2);
        spool_velocity[i]   = states[i](3);
        pressure[i]         = states[i](4) / 1e5;  // convert to bar
    }

    // Create traces for each subplot
    auto trace_current = Scatter()
                             .x(t)
                             .y(current)
                             .mode({Scatter::Mode::Lines})
                             .name("Solenoid Current")
                             .xaxis("x")
                             .yaxis("y");

    auto trace_integrator = Scatter()
                                .x(t)
                                .y(integrator_state)
                                .mode({Scatter::Mode::Lines})
                                .name("PI Integrator State")
                                .xaxis("x2")
                                .yaxis("y2");

    auto trace_spool_pos = Scatter()
                               .x(t)
                               .y(spool_position)
                               .mode({Scatter::Mode::Lines})
                               .name("Spool Position")
                               .xaxis("x3")
                               .yaxis("y3");

    auto trace_spool_vel = Scatter()
                               .x(t)
                               .y(spool_velocity)
                               .mode({Scatter::Mode::Lines})
                               .name("Spool Velocity")
                               .xaxis("x4")
                               .yaxis("y4");

    auto trace_pressure = Scatter()
                              .x(t)
                              .y(pressure)
                              .mode({Scatter::Mode::Lines})
                              .name("Chamber Pressure")
                              .xaxis("x5")
                              .yaxis("y5");

    auto layout = Layout()
                      .title([](auto& t) { t.text("Hydraulic Valve Simulation"); })
                      .height(1200)
                      .width(1000)
                      .xaxis(1, Layout::Xaxis().title([](auto& t) { t.text("Time [s]"); }).showgrid(true))
                      .yaxis(1, Layout::Yaxis().title([](auto& t) { t.text("i [mA]"); }).showgrid(true))
                      .xaxis(2, Layout::Xaxis().title([](auto& t) { t.text("Time [s]"); }).showgrid(true))
                      .yaxis(2, Layout::Yaxis().title([](auto& t) { t.text("z"); }).showgrid(true))
                      .xaxis(3, Layout::Xaxis().title([](auto& t) { t.text("Time [s]"); }).showgrid(true))
                      .yaxis(3, Layout::Yaxis().title([](auto& t) { t.text("x_spool [m]"); }).showgrid(true))
                      .xaxis(4, Layout::Xaxis().title([](auto& t) { t.text("Time [s]"); }).showgrid(true))
                      .yaxis(4, Layout::Yaxis().title([](auto& t) { t.text("v_spool [m/s]"); }).showgrid(true))
                      .xaxis(5, Layout::Xaxis().title([](auto& t) { t.text("Time [s]"); }).showgrid(true))
                      .yaxis(5, Layout::Yaxis().title([](auto& t) { t.text("P [bar]"); }).showgrid(true).range(std::vector<double>{-2, 32}))
                      .grid(Layout::Grid{}
                                .rows(5)
                                .columns(1)
                                .subplots(std::vector<std::vector<std::string>>{{"xy"}, {"x2y2"}, {"x3y3"}, {"x4y4"}, {"x5y5"}})
                                .roworder(Layout::Grid::Roworder::BottomToTop));

    fig.addTraces(std::vector<Trace>{trace_current, trace_integrator, trace_spool_pos, trace_spool_vel, trace_pressure});
    fig.setLayout(layout);

    fig.writeHtml("hydraulic_valve_simulation.html");

    fmt::print("Plots saved to hydraulic_valve_simulation.html\n");
    fmt::print("Add your stiff integrator type to run the simulation (see comment).\n");
    return 0;
}
