#include "constexpr_math.hpp"
#include "fmt/core.h"
#include "lqr.hpp"
#include "plot_plotly.hpp"
#include "simulate.hpp"
#include "solver.hpp"

using namespace wetmelon::control;

// ===== Nonlinear Pendulum System =====
// States: [theta, theta_dot]  (angle from vertical, angular velocity)
// Dynamics: theta_ddot = (g/L)*sin(theta) - (b/m*L^2)*theta_dot + (1/m*L^2)*u
constexpr double g = 9.81;
constexpr double L = 1.0;
constexpr double m = 1.0;
constexpr double b_damp = 0.1;

// Linearize at upright equilibrium (theta=0)
constexpr auto linearize_pendulum() {
    const double a21 = g / L;
    const double a22 = -b_damp / (m * L * L);

    Matrix<2, 2> A{{0.0, 1.0}, {a21, a22}};
    Matrix<2, 1> B{{0.0}, {1.0 / (m * L * L)}};
    Matrix<1, 2> C{{1.0, 0.0}};

    return StateSpace{.A = A, .B = B, .C = C};
}

// ===== Compile-time LQR design =====
constexpr auto sys = linearize_pendulum();
constexpr auto Q = Matrix<2, 2>::identity() * 10.0;
constexpr auto R = Matrix<1, 1>{{1.0}};
constexpr auto Ts = 0.01;
constexpr auto lqr_d = design::lqrd(sys.A, sys.B, Q, R, Ts);

int main() {
    fmt::print("===== Pendulum Simulation Example =====\n\n");
    fmt::print("LQR Gain K = [{:.4f}, {:.4f}]\n\n", lqr_d.K(0, 0), lqr_d.K(0, 1));

    // Create runtime controller from compile-time design
    LQR controller{lqr_d};

    // Nonlinear plant: dx/dt = f(t, x, u)
    auto plant = [](double /*t*/, const ColVec<2>& x, const ColVec<1>& u) -> ColVec<2> {
        double theta = x(0, 0);
        double theta_dot = x(1, 0);
        double torque = u(0, 0);

        double theta_ddot = (g / L) * wet::sin(theta)
                          - (b_damp / (m * L * L)) * theta_dot
                          + (1.0 / (m * L * L)) * torque;

        return ColVec<2>{{theta_dot}, {theta_ddot}};
    };

    // Output: measure full state (state feedback)
    auto output = [](const ColVec<2>& x) -> ColVec<2> { return x; };

    // State-feedback controller: u = -K * x
    auto ctrl = [&](const ColVec<2>& x) -> ColVec<1> {
        return controller.control(x);
    };

    // Solver: RK4 with 1ms steps
    RK4<2>          rk4;
    FixedStepSolver solver(rk4, 0.001);

    // Initial condition: 30 degrees from vertical, zero velocity
    ColVec<2> x0{{0.5236}, {0.0}};

    // Simulate for 5 seconds
    auto sim = simulate_state_feedback<2, 1, 2>(plant, output, ctrl, solver, x0, {0.0, 5.0});

    fmt::print("Simulated {} time steps\n", sim.t.size());
    fmt::print("Final state: theta = {:.4f} rad, theta_dot = {:.4f} rad/s\n", sim.x.back()(0, 0), sim.x.back()(1, 0));

    // Plot results
    auto fig = plot::plot_simulation(sim, "Pendulum LQR Simulation");
    fig.writeHtml("pendulum_sim.html");
    fmt::print("\nPlot written to pendulum_sim.html\n");

    // ===== Demonstrate runtime redesign with online:: =====
    fmt::print("\n===== Runtime LQR Redesign =====\n");
    auto Q2 = Matrix<2, 2>::identity() * 50.0; // Higher state penalty
    auto lqr_d2 = online::lqrd(sys.A, sys.B, Q2, R, Ts);
    LQR  controller2{lqr_d2};
    fmt::print("New LQR Gain K = [{:.4f}, {:.4f}]\n", lqr_d2.K(0, 0), lqr_d2.K(0, 1));

    auto ctrl2 = [&](const ColVec<2>& x) -> ColVec<1> {
        return controller2.control(x);
    };

    auto sim2 = simulate_state_feedback<2, 1, 2>(plant, output, ctrl2, solver, x0, {0.0, 5.0});
    fmt::print("Final state: theta = {:.4f} rad, theta_dot = {:.4f} rad/s\n", sim2.x.back()(0, 0), sim2.x.back()(1, 0));

    auto fig2 = plot::plot_simulation(sim2, "Pendulum LQR (Higher Q) Simulation");
    fig2.writeHtml("pendulum_sim_high_q.html");
    fmt::print("Plot written to pendulum_sim_high_q.html\n");

    return 0;
}
