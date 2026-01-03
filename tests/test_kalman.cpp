#include <cmath>

#include "control_design.hpp"
#include "doctest.h"
#include "kalman.hpp"
#include "lqr.hpp"
#include "state_space.hpp"

using namespace wetmelon::control;

TEST_CASE("State-space discrete propagation with control") {
    constexpr double dt = 1.0;
    Matrix<2, 2>     A = {{1.0, dt}, {0.0, 1.0}};
    Matrix<2, 1>     B = {{0.5 * dt * dt}, {dt}};
    ColVec<2>        x0 = {0.0, 1.0};
    ColVec<1>        u = {1.0};

    auto x1 = ColVec<2>(A * x0 + B * u);
    CHECK(x1[0] == doctest::Approx(1.5));
    CHECK(x1[1] == doctest::Approx(2.0));
}

TEST_CASE("Kalman predict/update 1D") {
    StateSpace<1, 1, 1> sys{};
    sys.A(0, 0) = 1.0;
    sys.B(0, 0) = 1.0;
    sys.C(0, 0) = 1.0;
    sys.G(0, 0) = 1.0;
    sys.H(0, 0) = 1.0;
    sys.Ts = 1.0; // discrete system

    KalmanFilter<1, 1, 1> kf(sys, Matrix<1, 1, double>{{0.1}}, Matrix<1, 1, double>{{0.25}}, ColVec<1>{0.0}, Matrix<1, 1, double>::identity());

    ColVec<1> u = {0.0};
    kf.predict(u);

    ColVec<1> z = {1.2};
    bool      ok = kf.update(z, u);
    CHECK(ok);
    CHECK(kf.state()[0] == doctest::Approx(0.978).epsilon(1e-3));
    CHECK(kf.covariance()(0, 0) == doctest::Approx(0.204).epsilon(1e-3));
}

TEST_CASE("EKF range-only measurement") {
    ColVec<2>    x0 = {1.0, 0.5};
    Matrix<2, 2> P0 = Matrix<2, 2, double>::identity();

    Matrix<2, 2, double> Q{};
    Q.fill(0.01);

    Matrix<1, 1, double> R{};
    R(0, 0) = 0.04;

    ExtendedKalmanFilter<2, 0, 1> ekf(x0, P0, Q, R);

    auto state_fn = [](const ColVec<2, double>& x_in, const ColVec<0, double>&) {
        StateJacobian<double, 2> sj;
        sj.x_pred = x_in;
        sj.F = Matrix<2, 2, double>::identity();
        sj.G = Matrix<2, 2, double>::identity();
        return sj;
    };

    ekf.predict(state_fn, ColVec<0>{});

    auto meas_fn = [](const ColVec<2, double>& x_in, const ColVec<0, double>&) {
        double               r = std::sqrt(x_in[0] * x_in[0] + x_in[1] * x_in[1]);
        ColVec<1, double>    z_pred{r};
        Matrix<1, 2, double> H{};
        if (r > 1e-9) {
            H(0, 0) = x_in[0] / r;
            H(0, 1) = x_in[1] / r;
        }
        MeasJacobian<double, 1, 2> mj;
        mj.z_pred = z_pred;
        mj.H = H;
        mj.M = Matrix<1, 1, double>::identity();
        return mj;
    };

    ColVec<1> z = {std::sqrt(1.0 * 1.0 + 1.0 * 1.0)}; // target at (1,1)

    bool ok = ekf.update(meas_fn, z, ColVec<0>{});
    CHECK(ok);

    // Updated state should be closer (in range space) to the measured radius
    const auto& x_hat = ekf.state();
    double      r_est = std::sqrt(x_hat[0] * x_hat[0] + x_hat[1] * x_hat[1]);
    CHECK(r_est == doctest::Approx(z[0]).epsilon(2e-2));

    // Covariance should contract after incorporating the measurement
    CHECK(ekf.covariance()(0, 0) < 1.01);
    CHECK(ekf.covariance()(1, 1) < 1.01);
}

TEST_CASE("LQR scalar DARE solution") {
    Matrix<1, 1> A = {{1.0}};
    Matrix<1, 1> B = {{1.0}};
    Matrix<1, 1> Q = {{1.0}};
    Matrix<1, 1> R = {{1.0}};

    auto result = online::dlqr(A, B, Q, R);

    // Known solution: P = (1+sqrt(5))/2, K = P/(1+P)
    double P_expected = (1.0 + std::sqrt(5.0)) * 0.5;
    double K_expected = P_expected / (1.0 + P_expected);
    CHECK(result.S(0, 0) == doctest::Approx(P_expected).epsilon(1e-3));
    CHECK(result.K(0, 0) == doctest::Approx(K_expected).epsilon(1e-3));

    // Test control via LQR struct
    LQR<1, 1> lqr(result);
    ColVec<1> x = {2.0};
    auto      u = lqr.control(x);
    CHECK(u[0] == doctest::Approx(-K_expected * 2.0).epsilon(1e-3));
}

TEST_CASE("LQG combines LQR and KF") {
    StateSpace<1, 1, 1> sys{};
    sys.A(0, 0) = 1.0;
    sys.B(0, 0) = 1.0;
    sys.C(0, 0) = 1.0;
    sys.Ts = 1.0; // discrete

    Matrix<1, 1> Qlqr = {{1.0}};
    Matrix<1, 1> Rlqr = {{1.0}};
    Matrix<1, 1> Qkf = {{0.1}};
    Matrix<1, 1> Rkf = {{0.25}};

    LQG lqg = online::lqg(sys, Qlqr, Rlqr, Qkf, Rkf);

    ColVec<1> z = {1.0};
    bool      ok = lqg.update(z);
    CHECK(ok);

    auto u = lqg.control();
    CHECK(u[0] < 0.0); // drives error toward zero
}

TEST_CASE("LQG servo (LQI + KF) tracks reference") {
    StateSpace<1, 1, 1> sys{};
    sys.A(0, 0) = 1.0;
    sys.B(0, 0) = 1.0;
    sys.C(0, 0) = 1.0;
    sys.Ts = 1.0; // discrete

    Matrix<2, 2, double> Q_aug{};
    Q_aug(0, 0) = 1.0; // state weight
    Q_aug(1, 1) = 5.0; // integral weight
    Matrix<1, 1> R = {{1.0}};
    Matrix<1, 1> Qkf = {{0.1}};
    Matrix<1, 1> Rkf = {{0.25}};

    LQGI lqgs = online::lqgtrack(sys, Q_aug, R, Qkf, Rkf);

    ColVec<1> z = {0.0};
    lqgs.update(z);

    ColVec<1> r = {1.0};
    auto      u = lqgs.control(r);
    CHECK(u[0] < 0.0); // LQI uses u = -Ki*xi, so positive tracking error gives negative control in regulator form
}
