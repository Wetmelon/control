#include "doctest.h"
#include "observer.hpp"
#include "types.hpp"

TEST_CASE("LuenbergerObserver basic functionality") {
    using namespace control;

    // Define a simple discrete-time state-space model
    StateSpace sys{
        Matrix{{1.0, 0.1}, {0.0, 1.0}},  // A
        Matrix{{0.0}, {0.1}},            // B
        Matrix{{1.0, 0.0}},              // C
        Matrix{{0.0}},                   // D
        0.1                              // Ts
    };

    // Observer gain
    Matrix L{{0.5}, {1.0}};

    // Create Luenberger Observer object
    auto observer = LuenbergerObserver{sys, L};

    // Initial state
    ColVec x0 = {0.0, 0.0};

    // Simulate a few steps
    ColVec u = {1.0};  // Constant input
    ColVec y;          // Measurement

    // Predict and update loop
    for (int i = 0; i < 10; ++i) {
        observer.predict(u);  // Predict step, dt is ignored for discrete systems

        y = sys.C * x0 + ColVec::Random(1) * 0.01;  // Simulated measurement with noise
        observer.update(y);
    }

    ColVec x_est = observer.state();
    CHECK(x_est.size() == 2);
}

TEST_CASE("KalmanFilter basic functionality") {
    using namespace control;

    // Define a simple discrete-time state-space model
    StateSpace sys = {
        Matrix{{1.0, 0.1}, {0.0, 1.0}},  // A
        Matrix{{0.0}, {0.1}},            // B
        Matrix{{1.0, 0.0}},              // C
        Matrix{{0.0}},                   // D
        0.1                              // Ts
    };

    // Process and measurement noise covariances
    Matrix Q = Matrix::Identity(2, 2) * 0.01;
    Matrix R = Matrix::Identity(1, 1) * 0.1;

    // Create Kalman Filter object
    auto kf = KalmanFilter{sys, Q, R};

    // Initial state
    ColVec x0 = {0.0, 0.0};

    // Simulate a few steps
    ColVec u = {1.0};  // Constant input
    ColVec y;          // Measurement

    // Predict and update loop
    for (int i = 0; i < 10; ++i) {
        kf.predict(u);  // Predict step, dt is ignored for discrete systems

        y = sys.C * x0 + ColVec::Random(1) * 0.1;  // Simulated measurement with noise
        kf.update(y);
    }

    ColVec x_est = kf.state();
    CHECK(x_est.size() == 2);
}

TEST_CASE("ExtendedKalmanFilter basic functionality") {
    using namespace control;

    double dt = 0.1;

    // Define nonlinear state and measurement functions
    auto f = [=](const ColVec& x, const ColVec& u) {
        ColVec x_next(2);
        x_next[0] = x[0] + dt * x[1] + 0.5 * dt * dt * u[0];
        x_next[1] = x[1] + dt * u[0];
        return x_next;
    };

    auto h = [=](const ColVec& x) {
        ColVec y(1);
        y[0] = x[0] * x[0];
        return y;
    };

    // Initial state
    ColVec x0 = {0.0, 1.0};

    // Create Extended Kalman Filter object
    auto ekf = ExtendedKalmanFilter{f, h, x0};

    // Simulate a few steps
    ColVec u = ColVec{0.5};  // Constant input

    // Predict and update loop
    for (int i = 0; i < 10; ++i) {
        ekf.predict(u);  // Predict step

        auto z = h(x0) + ColVec::Random(1) * 0.1;  // Simulated measurement with noise
        ekf.update(z);
    }

    ColVec x_est = ekf.state();
    CHECK(x_est.size() == 2);
}

// Manually constructed Jacobians
TEST_CASE("ExtendedKalmanFilter with user-provided Jacobians") {
    using namespace control;

    // Define nonlinear state and measurement functions
    auto f = [](const ColVec& x, const ColVec& u) {
        ColVec x_next(2);
        x_next[0] = x[0] + 0.1 * x[1] + 0.05 * u[0];
        x_next[1] = x[1] + 0.1 * std::sin(x[0]);
        return x_next;
    };

    auto h = [](const ColVec& x) {
        ColVec y(1);
        y[0] = x[0] * x[0];
        return y;
    };

    // Define Jacobians
    auto F = [](const ColVec& x, const ColVec& /*u*/) {
        Matrix J(2, 2);
        J << 1.0, 0.1,
            0.1 * std::cos(x[0]), 1.0;
        return J;
    };

    auto H = [](const ColVec& x) {
        Matrix J(1, 2);
        J << 2.0 * x[0], 0.0;
        return J;
    };

    // Initial state
    ColVec x0  = ColVec{0.0, 1.0};
    auto   ekf = ExtendedKalmanFilter{f, h, F, H, x0};

    // Simulate a few steps
    ColVec u = ColVec{0.5};  // Constant input

    // Predict and update loop
    for (int i = 0; i < 10; ++i) {
        ekf.predict(u);

        auto z = h(x0) + ColVec::Random(1) * 0.1;  // Simulated measurement with noise
        ekf.update(z);
    }

    ColVec x_est = ekf.state();
    CHECK(x_est.size() == 2);
}