#include <fmt/core.h>

#include "control.hpp"
#include "matplot/matplot.h"

int main() {
    using namespace control;
    namespace plt = matplot;

    fmt::print("=== Inverted Pendulum on Cart Example ===\n");
    fmt::print("Balancing a pendulum using LQR control\n\n");

    // System parameters
    const double M = 0.5;   // Cart mass [kg]
    const double m = 0.2;   // Pendulum mass [kg]
    const double l = 0.3;   // Pendulum length [m]
    const double g = 9.81;  // Gravity [m/s^2]
    const double b = 0.1;   // Cart friction coefficient [N/(m/s)]

    fmt::print("System parameters:\n");
    fmt::print("  Cart mass M = {:.1f} kg\n", M);
    fmt::print("  Pendulum mass m = {:.1f} kg\n", m);
    fmt::print("  Pendulum length l = {:.1f} m\n", l);
    fmt::print("  Gravity g = {:.2f} m/s²\n", g);
    fmt::print("  Friction b = {:.1f} N/(m/s)\n\n", b);

    // State-space model linearized around θ = π (upright position)
    // States: [x, x_dot, θ, θ_dot]^T
    // Input: Force F on cart
    // Output: [x, θ]^T (cart position and pendulum angle)

    const double p = m / (M + m);  // Mass ratio

    Matrix A_sys = Matrix{{0, 1, 0, 0},
                          {0, -b * (1 - p) / M, -m * g * p / M, 0},
                          {0, 0, 0, 1},
                          {0, -b * p / (M * l), g * (M + m) / (M * l), 0}};

    Matrix B_sys = Matrix{{0}, {(1 - p) / M}, {0}, {p / (M * l)}};

    Matrix C_sys = Matrix{{1, 0, 0, 0},   // Cart position
                          {0, 0, 1, 0}};  // Pendulum angle

    Matrix D_sys = Matrix::Zero(2, 1);

    StateSpace pendulum_system(A_sys, B_sys, C_sys, D_sys);

    // Check stability (should be unstable)
    bool stable = is_stable(pendulum_system);
    fmt::print("Open-loop system is {}stable\n\n", stable ? "" : "un");

    // Design LQR controller
    // State weighting Q: penalize angle and angular velocity heavily
    Matrix Q = Matrix{{1, 0, 0, 0},    // Cart position
                      {0, 1, 0, 0},    // Cart velocity
                      {0, 0, 100, 0},  // Pendulum angle (high penalty)
                      {0, 0, 0, 10}};  // Pendulum angular velocity

    // Design LQR controller
    Matrix R       = Matrix::Constant(1, 1, 0.1);
    auto   lqr_res = lqr(A_sys, B_sys, Q, R);
    Matrix K       = lqr_res.K;  // 1x4 gain

    // fmt::print("LQR gain K = \n{}\n", K);

    // Closed-loop system: A_cl = A - B*K
    Matrix     A_cl = A_sys - B_sys * K;
    StateSpace cl_system(A_cl, B_sys, C_sys, D_sys);

    // Check closed-loop stability
    bool cl_stable = is_stable(cl_system);
    fmt::print("Closed-loop system is {}stable\n\n", cl_stable ? "" : "un");

    // Simulate response to initial condition (pendulum slightly off vertical)
    Matrix x0 = Matrix{{0.0}, {0.0}, {0.1}, {0.0}};  // Small angle perturbation

    fmt::print("Simulating response to initial condition: θ₀ = 0.1 rad\n");

    // Use ExactSolver on closed-loop A_cl with zero input (B_zero) to obtain state trajectories
    const double        dt = 0.01;
    const double        t0 = 0.0;
    const double        tf = 5.0;
    std::vector<double> t_eval;
    for (double t = t0; t <= tf + 1e-12; t += dt) t_eval.push_back(t);

    Matrix B_zero = Matrix::Zero(4, 1);
    ColVec u0     = ColVec::Zero(1);
    auto   sol    = ExactSolver().solve(A_cl, B_zero, x0, u0, {t0, tf}, t_eval);

    // Extract outputs and control history
    std::vector<double> time = sol.t;
    std::vector<double> cart_pos, pend_angle, control_u;
    cart_pos.reserve(sol.x.size());
    pend_angle.reserve(sol.x.size());
    control_u.reserve(sol.x.size());
    for (const auto& x : sol.x) {
        ColVec y = C_sys * x + D_sys * u0;
        cart_pos.push_back(y(0, 0));
        pend_angle.push_back(y(1, 0));
        control_u.push_back((-K * x)(0, 0));
    }

    // Plot results (use explicit axes to ensure labels map to correct subplot)
    auto fig = plt::figure(true);
    fig->size(1200, 800);

    // Use a figure-level title so it doesn't conflict with subplot layout
    plt::sgtitle("Inverted Pendulum - LQR Control");
    auto ax1 = plt::subplot(3, 1, 1);
    ax1->plot(time, cart_pos);
    ax1->ylabel("Cart Position [m]");
    ax1->grid(true);

    auto ax2 = plt::subplot(3, 1, 2);
    ax2->plot(time, pend_angle);
    ax2->ylabel("Pendulum Angle [rad]");
    ax2->grid(true);

    auto ax3 = plt::subplot(3, 1, 3);
    ax3->plot(time, control_u);
    ax3->xlabel("Time [s]");
    ax3->ylabel("Control Force [N]");
    ax3->grid(true);

    plt::show();

    // Analyze closed-loop poles
    auto poles = cl_system.poles();
    fmt::print("Closed-loop poles:\n");
    for (size_t i = 0; i < poles.size(); ++i) {
        fmt::print("  λ_{} = {:.3f} ± {:.3f}j\n", i + 1, poles[i].real(), poles[i].imag());
    }

    return 0;
}