#include "constexpr_math.hpp"
#include "control_design.hpp"
#include "fmt/core.h"
#include "lqr.hpp"

using namespace wetmelon::control;

// ===== Nonlinear Pendulum System =====
// States: [theta, theta_dot]  (angle from vertical, angular velocity)
// Dynamics: theta_ddot = (g/L)*sin(theta) - (b/m*L^2)*theta_dot + (1/m*L^2)*u
constexpr double g = 9.81; // gravity (m/s^2)
constexpr double L = 1.0;  // length (m)
constexpr double m = 1.0;  // mass (kg)
constexpr double b = 0.1;  // damping (N*m*s)

// Linearize pendulum dynamics at operating point (theta_0, theta_dot_0)
constexpr auto linearize_pendulum(double theta_0, double /*theta_dot_0*/) {
    // Linearized A matrix: df/dx evaluated at (theta_0, theta_dot_0)
    // x' = [theta_dot, (g/L)*cos(theta_0)*theta - (b/m*L^2)*theta_dot]
    const double a21 = (g / L) * wet::cos(theta_0); // df2/dtheta
    const double a22 = -b / (m * L * L);            // df2/dtheta_dot

    Matrix<2, 2> A{{0.0, 1.0}, {a21, a22}};

    // Linearized B matrix: df/du (constant for this system)
    Matrix<2, 1> B{{0.0}, {1.0 / (m * L * L)}};

    // Output: measure angle
    Matrix<1, 2> C{{1.0, 0.0}};

    return StateSpace{.A = A, .B = B, .C = C};
}

// ===== A) COMPILE-TIME LQR: Linearize at equilibrium (theta=0) =====
constexpr auto sys_eq = linearize_pendulum(0.0, 0.0); // Upright equilibrium
constexpr auto Q = Matrix<2, 2>::identity() * 10.0;   // Penalize angle and velocity
constexpr auto R = Matrix<1, 1>{{1.0}};               // Penalize torque
constexpr auto Ts = 0.01;                             // 100Hz control loop

// Design at compile time - guaranteed zero runtime overhead
constexpr auto res_eq = design::lqrd(sys_eq.A, sys_eq.B, Q, R, Ts);
LQR            controller_eq = res_eq.as<float>();

// ===== B) RUNTIME LQR: Linearize about current state =====
static LQR<2, 1> design_lqr_at_state(double theta, double theta_dot) {
    auto sys = linearize_pendulum(theta, theta_dot);
    auto res = online::lqrd(sys.A, sys.B, Q, R, Ts);
    return LQR{res};
}

int main() {
    fmt::print("===== Nonlinear Pendulum LQR Example =====\n\n");

    // A) Compile-time controller (linearized at upright equilibrium)
    fmt::print("A) Compile-Time LQR (linearized at theta=0, upright equilibrium):\n");
    fmt::print("   Gain K = [{:.4f}, {:.4f}]\n", controller_eq.K(0, 0), controller_eq.K(0, 1));

    ColVec<2> x_eq{0.1, 0.0}; // Small angle, zero velocity
    double    u_eq = controller_eq.control(x_eq)(0, 0);
    fmt::print("   Control for state [{:.2f}, {:.2f}]: u = {:.4f}\n\n", x_eq(0, 0), x_eq(1, 0), u_eq);

    // B) Runtime controllers (linearized at different operating points)
    fmt::print("B) Runtime LQR (linearized about current state):\n");

    // Test at different angles
    const double test_angles[] = {0.0, 0.5, 1.0, 1.5}; // radians
    for (double theta : test_angles) {
        auto controller_rt = design_lqr_at_state(theta, 0.0);

        ColVec<2> x{theta, 0.0};
        double    u = controller_rt.control(x)(0, 0);

        fmt::print("   theta = {:.2f} rad ({:.1f} deg):\n", theta, theta * 180.0 / 3.14159);
        fmt::print("     Gain K = [{:.4f}, {:.4f}]\n", controller_rt.K(0, 0), controller_rt.K(0, 1));
        fmt::print("     Control: u = {:.4f}\n\n", u);
    }

    fmt::print("Note: Gain adapts based on linearization point!\n");
    fmt::print("      Higher angles → stronger feedback needed\n");

    return 0;
}
