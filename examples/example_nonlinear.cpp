#include <iostream>
#include <numbers>

#include "control.hpp"

int main() {
    // Example: Simple nonlinear pendulum
    // dx/dt = [x2, -sin(x1) - 0.1*x2 + u]
    // y = x1

    auto f = [](const control::ColVec& x, const control::ColVec& u) -> control::ColVec {
        return control::ColVec{
            x(1),
            -std::sin(x(0)) - 0.1 * x(1) + u(0)};
    };

    auto h = [](const control::ColVec& x, [[maybe_unused]] const control::ColVec& u) -> control::ColVec {
        return control::ColVec{x(0)};
    };

    control::NonlinearSystem pendulum(f, h, 2, 1, 1);

    // Linearize around upright position (x = [pi, 0], u = 0)
    control::ColVec x0 = {std::numbers::pi, 0.0};
    control::ColVec u0 = {0.0};

    control::StateSpace linearized = pendulum.linearize(x0, u0);

    std::cout << "Linearized pendulum around upright position:\n";
    std::cout << "A = \n"
              << linearized.A << "\n\n";
    std::cout << "B = \n"
              << linearized.B << "\n\n";
    std::cout << "C = \n"
              << linearized.C << "\n\n";
    std::cout << "D = \n"
              << linearized.D << "\n\n";

    // Design LQR controller for the linearized system
    control::Matrix Q = control::Matrix::Identity(2, 2);
    Q(0, 0)           = 10.0;  // Penalize angle error
    control::Matrix R = control::Matrix{{1.0}};

    auto lqr_result = control::lqr(pendulum, x0, u0, Q, R);

    std::cout << "LQR gain K = " << lqr_result.K << "\n";

    // Design LQG controller
    control::Matrix Qn = 0.01 * control::Matrix::Identity(2, 2);  // Process noise
    control::Matrix Rn = control::Matrix{{0.1}};                  // Measurement noise

    auto lqg_result = control::lqg(pendulum, x0, u0, Q, R, Qn, Rn);

    std::cout << "LQG designed successfully!\n";
    std::cout << "LQG gain K = " << lqg_result.K << "\n";

    // Demonstrate Extended Kalman Filter with NonlinearSystem
    control::ExtendedKalmanFilter ekf(pendulum, x0, Qn, Rn);

    // Simulate one step
    control::ColVec u = {0.0};  // No control input
    ekf.predict(u);

    // Simulate measurement (true angle with noise)
    double          true_angle  = x0(0) + 0.01;         // Small perturbation
    control::ColVec measurement = {true_angle + 0.01};  // Measurement with noise

    ekf.update(measurement);

    std::cout << "EKF state estimate: " << ekf.state().transpose() << "\n";
    std::cout << "EKF covariance trace: " << ekf.covariance().trace() << "\n";

    return 0;
}