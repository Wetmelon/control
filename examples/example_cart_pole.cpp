/**
 * @file example_cart_pole.cpp
 * @brief Inverted pendulum on cart (cart-pole) LQR control example
 *
 * Demonstrates compile-time linearization and LQR design for the classic
 * cart-pole system - an inherently unstable nonlinear system.
 */

#include "constexpr_math.hpp"
#include "control_design.hpp"
#include "fmt/core.h"
#include "lqr.hpp"

using namespace wetmelon::control;

/**
 * @brief Inverted Pendulum on Cart (Cart-Pole) System
 *
 * States: [x, x_dot, theta, theta_dot]  (cart pos, cart vel, pole angle, pole angular vel)
 * Control: u (horizontal force on cart)
 *
 * Nonlinear dynamics:
 *   (M+m)*x_ddot + m*L*theta_ddot*cos(theta) - m*L*theta_dot^2*sin(theta) = u
 *   L*theta_ddot - g*sin(theta) = -x_ddot*cos(theta)
 */

// Physical parameters
constexpr double M = 1.0;  /// Cart Mass (kg)
constexpr double m = 0.1;  /// Pole Mass (kg)
constexpr double L = 0.5;  /// Pole Length (m)
constexpr double g = 9.81; /// Gravity (m/s^2)
constexpr double b = 0.1;  /// Cart friction coefficient (N/m/s)

/**
 * @brief Linearize cart-pole dynamics at operating point
 *
 * @param x_0 Cart position operating point (unused)
 * @param x_dot_0 Cart velocity operating point (unused)
 * @param theta_0 Pole angle operating point
 * @param theta_dot_0 Pole angular velocity operating point (unused)
 * @return Linearized StateSpace system
 */
constexpr auto linearize_cart_pole(double /*x_0*/, double /*x_dot_0*/, double theta_0, double /*theta_dot_0*/) {
    /// Linearized equations at arbitrary theta_0:
    ///
    /// For inverted pendulum (theta=0 is upright), the linearization gives:
    ///      d/dt [x, x_dot, theta, theta_dot]' = A*[x, x_dot, theta, theta_dot]' + B*u

    const double s = wet::sin(theta_0);
    const double c = wet::cos(theta_0);
    const double denom = M + m * (1.0 - c * c); /// = M + m*cos^2(theta)

    /// A matrix elements
    /// The (3,2) element is the key: pendulum angular acceleration due to angle deviation
    /// At upright: df_theta_dot_dot/dtheta = (M+m)*g*cos(theta_0) / (L*denom)
    const double a22 = -b / denom;
    const double a23 = m * g * s * c / denom;
    const double a42 = -b * c / (L * denom);
    const double a43 = (M + m) * g * c / (L * denom);

    Matrix<4, 4> A{
        {0.0, 1.0, 0.0, 0.0}, /// x' = x_dot
        {0.0, a22, a23, 0.0}, /// x_dot' = ...
        {0.0, 0.0, 0.0, 1.0}, /// theta' = theta_dot
        {0.0, a42, a43, 0.0}  /// theta_dot' = ... (a43 > 0 makes it unstable!)
    };

    /// B matrix (force input)
    const double b2 = 1.0 / denom;
    const double b4 = c / (L * denom);

    Matrix<4, 1> B{
        {0.0}, /// x' doesn't directly depend on u
        {b2},  /// x_dot' depends on u
        {0.0}, /// theta' doesn't directly depend on u
        {b4}   /// theta_dot' depends on u
    };

    /// Output: measure cart position and pole angle
    Matrix<2, 4> C{
        {1.0, 0.0, 0.0, 0.0}, /// measure x
        {0.0, 0.0, 1.0, 0.0}  /// measure theta
    };

    return StateSpace{.A = A, .B = B, .C = C};
}

/// Note: 4x4 DARE solution at compile-time requires many operations.  You may need to increase -fconstexpr-loop-limit
constexpr auto sys_eq = linearize_cart_pole(0.0, 0.0, 0.0, 0.0); /// Upright, centered

/// Cost matrices: heavily penalize pole angle, moderately penalize position
constexpr auto Q = Matrix<4, 4>{
    {10.0, 0.0, 0.0, 0.0},  /// Cart position
    {0.0, 0.1, 0.0, 0.0},   /// Cart velocity
    {0.0, 0.0, 100.0, 0.0}, /// Pole angle (critical!)
    {0.0, 0.0, 0.0, 10.0}   /// Pole angular velocity
};

constexpr auto Ts = 0.01;               // 100Hz control loop
constexpr auto R = Matrix<1, 1>{{1.0}}; // Control effort penalty

/// Design compile-time discrete LQR controller for cart-pole
LQR controller_eq = design::lqrd(sys_eq.A, sys_eq.B, Q, R, Ts).as<float>();

/**
 * @brief Main function demonstrating cart-pole LQR control
 *
 * Shows compile-time linearization and LQR design for the cart-pole system,
 * then tests the controller against various disturbances.
 */
int main() {
    fmt::print("===== Inverted Pendulum on Cart (Cart-Pole) LQR Example =====\n\n");
    fmt::print("System: Cart mass M={:.1f} kg, Pole mass m={:.1f} kg, Length L={:.1f} m\n\n", M, m, L);

    /// Compile-time controller (linearized at upright equilibrium)
    fmt::print("Compile-Time LQR (linearized at upright equilibrium):\n");
    fmt::print("  Gain K = [{:.4f}, {:.4f}, {:.4f}, {:.4f}]\n", controller_eq.K(0, 0), controller_eq.K(0, 1), controller_eq.K(0, 2), controller_eq.K(0, 3));
    fmt::print("           [x,       x_dot,    theta,     theta_dot]\n\n");

    /// Test controller with several disturbances
    fmt::print("Controller response to disturbances:\n");

    /// Create an array of test cases: {x, x_dot, theta, theta_dot, description}
    const struct {
        float       x, x_dot, theta, theta_dot;
        const char* desc;
    } test_cases[] = {
        {0.05, 0.0, 0.0, 0.0, "Cart displaced 5cm right"},
        {0.0, 0.0, 0.05, 0.0, "Pole tilted 2.9° from vertical"},
        {0.0, 0.1, 0.0, 0.0, "Cart moving 0.1 m/s right"},
        {0.0, 0.0, 0.0, 0.1, "Pole rotating 0.1 rad/s"},
        {0.02, 0.0, 0.03, 0.0, "Combined: cart+pole offset"}
    };

    for (const auto& test : test_cases) {
        ColVec<4, float> x{test.x, test.x_dot, test.theta, test.theta_dot};
        float            u = controller_eq.control(x)(0, 0);

        fmt::print("  {:35s} → u = {:7.3f} N\n", test.desc, u);
    }

    return 0;
}
