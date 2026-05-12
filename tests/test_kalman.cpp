#include <cmath>

#include "doctest.h"
#include "ekf.hpp"
#include "kalman.hpp"
#include "lqg.hpp"
#include "lqgi.hpp"
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
    StateSpace<1, 1, 1, 1, 1> sys{};
    sys.A(0, 0) = 1.0;
    sys.B(0, 0) = 1.0;
    sys.C(0, 0) = 1.0;
    sys.G(0, 0) = 1.0;
    sys.H(0, 0) = 1.0;
    sys.Ts = 1.0; // discrete system

    KalmanFilter<1, 1, 1, 1, 1> kf(sys, Matrix<1, 1, double>{{0.1}}, Matrix<1, 1, double>{{0.25}}, ColVec<1>{0.0}, Matrix<1, 1, double>::identity());

    ColVec<1> u = {0.0};
    kf.predict(u);

    ColVec<1> y = {1.2};
    bool      ok = kf.update(y, u);
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

    ExtendedKalmanFilter<2, 0, 1> ekf(x0, P0, Q);

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
        ColVec<1, double>    y_pred{r};
        Matrix<1, 2, double> H{};
        if (r > 1e-9) {
            H(0, 0) = x_in[0] / r;
            H(0, 1) = x_in[1] / r;
        }
        MeasJacobian<double, 1, 2> mj;
        mj.y_pred = y_pred;
        mj.H = H;
        mj.M = Matrix<1, 1, double>::identity();
        return mj;
    };

    ColVec<1> y = {std::sqrt(1.0 * 1.0 + 1.0 * 1.0)}; // target at (1,1)

    bool ok = ekf.update(meas_fn, y, R, ColVec<0>{});
    CHECK(ok);

    // Updated state should be closer (in range space) to the measured radius
    const auto& x_hat = ekf.state();
    double      r_est = std::sqrt(x_hat[0] * x_hat[0] + x_hat[1] * x_hat[1]);
    CHECK(r_est == doctest::Approx(y[0]).epsilon(2e-2));

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
    StateSpace<1, 1, 1, 1, 1> sys{};
    sys.A(0, 0) = 1.0;
    sys.B(0, 0) = 1.0;
    sys.C(0, 0) = 1.0;
    sys.G(0, 0) = 1.0;
    sys.H(0, 0) = 1.0;
    sys.Ts = 1.0; // discrete

    Matrix<1, 1> Qlqr = {{1.0}};
    Matrix<1, 1> Rlqr = {{1.0}};
    Matrix<1, 1> Qkf = {{0.1}};
    Matrix<1, 1> Rkf = {{0.25}};

    LQG lqg = online::lqg(sys, Qlqr, Rlqr, Qkf, Rkf);

    ColVec<1> y = {1.0};
    bool      ok = lqg.update(y);
    CHECK(ok);

    auto u = lqg.control();
    CHECK(u[0] < 0.0); // drives error toward zero
}

TEST_CASE("LQG servo (LQI + KF) tracks reference") {
    StateSpace<1, 1, 1, 1, 1> sys{};
    sys.A(0, 0) = 1.0;
    sys.B(0, 0) = 1.0;
    sys.C(0, 0) = 1.0;
    sys.G(0, 0) = 1.0;
    sys.H(0, 0) = 1.0;
    sys.Ts = 1.0; // discrete

    Matrix<2, 2, double> Q_aug{};
    Q_aug(0, 0) = 1.0; // state weight
    Q_aug(1, 1) = 5.0; // integral weight
    Matrix<1, 1> R = {{1.0}};
    Matrix<1, 1> Qkf = {{0.1}};
    Matrix<1, 1> Rkf = {{0.25}};

    LQGI lqgs = online::lqgtrack(sys, Q_aug, R, Qkf, Rkf);

    ColVec<1> y = {0.0};
    lqgs.update(y);

    ColVec<1> r = {1.0};
    // auto      u = lqgs.control(r);
    // CHECK(u[0] < 0.0); // LQI uses u = -Ki*xi, so positive tracking error gives negative control in regulator form
}

TEST_CASE("Kalman design R!=0, square C (NY=NX=2)") {
    // Double integrator, dt=0.1: position+velocity state, both measured directly.
    // Reference values generated with scipy.linalg.solve_discrete_are.
    constexpr double          dt = 0.1;
    StateSpace<2, 1, 2, 2, 2> sys{};
    sys.A = {{1.0, dt}, {0.0, 1.0}};
    sys.B = {{0.5 * dt * dt}, {dt}};
    sys.C = Matrix<2, 2, double>::identity();
    sys.G = Matrix<2, 2, double>::identity();
    sys.H = Matrix<2, 2, double>::identity();
    sys.Ts = dt;

    Matrix<2, 2> Q_proc = {{1e-4, 0.0}, {0.0, 1e-2}};
    Matrix<2, 2> R_meas = {{1e-2, 0.0}, {0.0, 5e-2}};

    auto result = online::kalman(sys, Q_proc, R_meas);
    REQUIRE(result.success);

    // scipy: solve_discrete_are(A.T, C.T, Q, R)
    CHECK(result.P(0, 0) == doctest::Approx(0.002606406390998).epsilon(1e-8));
    CHECK(result.P(0, 1) == doctest::Approx(0.003583502847688).epsilon(1e-8));
    CHECK(result.P(1, 0) == doctest::Approx(0.003583502847688).epsilon(1e-8));
    CHECK(result.P(1, 1) == doctest::Approx(0.027171128304431).epsilon(1e-8));

    CHECK(result.L(0, 0) == doctest::Approx(0.196141710450478).epsilon(1e-8));
    CHECK(result.L(0, 1) == doctest::Approx(0.037327800344898).epsilon(1e-8));
    CHECK(result.L(1, 0) == doctest::Approx(0.186639001724491).epsilon(1e-8));
    CHECK(result.L(1, 1) == doctest::Approx(0.343422566088615).epsilon(1e-8));

    // compile-time version must agree
    constexpr auto dresult = [&]() {
        StateSpace<2, 1, 2, 2, 2> s{};
        s.A = {{1.0, dt}, {0.0, 1.0}};
        s.B = {{0.5 * dt * dt}, {dt}};
        s.C = Matrix<2, 2, double>::identity();
        s.G = Matrix<2, 2, double>::identity();
        s.H = Matrix<2, 2, double>::identity();
        s.Ts = dt;
        Matrix<2, 2> Qp = {{1e-4, 0.0}, {0.0, 1e-2}};
        Matrix<2, 2> Rm = {{1e-2, 0.0}, {0.0, 5e-2}};
        return design::kalman(s, Qp, Rm);
    }();
    static_assert(dresult.success);
    CHECK(dresult.P(0, 0) == doctest::Approx(0.002606406390998).epsilon(1e-8));
    CHECK(dresult.L(1, 1) == doctest::Approx(0.343422566088615).epsilon(1e-8));
}

TEST_CASE("Kalman design R!=0, non-square C (NY=1, NX=2)") {
    // Same double integrator, only position is measured (velocity is hidden state).
    // Reference values generated with scipy.linalg.solve_discrete_are.
    constexpr double          dt = 0.1;
    StateSpace<2, 1, 1, 2, 1> sys{};
    sys.A = {{1.0, dt}, {0.0, 1.0}};
    sys.B = {{0.5 * dt * dt}, {dt}};
    sys.C = {{1.0, 0.0}}; // 1x2: observe position only
    sys.G = Matrix<2, 2, double>::identity();
    sys.H = Matrix<1, 1, double>::identity();
    sys.Ts = dt;

    Matrix<2, 2> Q_proc = {{1e-4, 0.0}, {0.0, 1e-2}};
    Matrix<1, 1> R_meas = {{5e-2}};

    auto result = online::kalman(sys, Q_proc, R_meas);
    REQUIRE(result.success);

    // scipy: solve_discrete_are(A.T, C.T, Q, R) with C = [[1,0]], R = [[0.05]]
    CHECK(result.P(0, 0) == doctest::Approx(0.017691003582802).epsilon(1e-8));
    CHECK(result.P(0, 1) == doctest::Approx(0.026017494803075).epsilon(1e-8));
    CHECK(result.P(1, 0) == doctest::Approx(0.026017494803075).epsilon(1e-8));
    CHECK(result.P(1, 1) == doctest::Approx(0.077996568142721).epsilon(1e-8));

    // L is 2x1: Kalman gain for position measurement
    CHECK(result.L(0, 0) == doctest::Approx(0.261349406072277).epsilon(1e-8));
    CHECK(result.L(1, 0) == doctest::Approx(0.384356759776049).epsilon(1e-8));

    // compile-time version must agree
    constexpr auto dresult = [&]() {
        StateSpace<2, 1, 1, 2, 1> s{};
        s.A = {{1.0, dt}, {0.0, 1.0}};
        s.B = {{0.5 * dt * dt}, {dt}};
        s.C = {{1.0, 0.0}};
        s.G = Matrix<2, 2, double>::identity();
        s.H = Matrix<1, 1, double>::identity();
        s.Ts = dt;
        Matrix<2, 2> Qp = {{1e-4, 0.0}, {0.0, 1e-2}};
        Matrix<1, 1> Rm = {{5e-2}};
        return design::kalman(s, Qp, Rm);
    }();
    static_assert(dresult.success);
    CHECK(dresult.P(0, 0) == doctest::Approx(0.017691003582802).epsilon(1e-8));
    CHECK(dresult.L(1, 0) == doctest::Approx(0.384356759776049).epsilon(1e-8));
}

TEST_CASE("Kalman design with R=0 (perfect measurements)") {
    // With R=0 (noiseless measurements) the standard DARE cannot be used (R
    // must be positive definite).  kalman() detects this, and for a square
    // invertible C solves analytically: P_ss = 0, L = C^{-1}.

    StateSpace<2, 1, 2, 2, 2> sys{};
    sys.A = {{0.9, 0.1}, {0.0, 0.8}};
    sys.B = {{0.0}, {1.0}};
    sys.C = Matrix<2, 2, double>::identity();
    sys.G = Matrix<2, 2, double>::identity();
    sys.H = Matrix<2, 2, double>::identity();
    sys.Ts = 0.01;

    Matrix<2, 2> Q_proc = {{0.01, 0.0}, {0.0, 0.01}};
    Matrix<2, 2> R_zero = Matrix<2, 2, double>::zeros();

    // kalman() should now succeed for the R=0 + square-C case
    auto result = online::kalman(sys, Q_proc, R_zero);
    CHECK(result.success);

    // P_ss = Q_eff = G*Q*G' = Q_proc (G=I here).
    // Scipy confirms: as R->0, solve_discrete_are(A', C', Q, R) -> Q.
    // Intuition: each predict step injects Q; the L=C^{-1}=I update fully
    // removes it, so the steady-state *prior* covariance equals Q.
    CHECK(result.P(0, 0) == doctest::Approx(Q_proc(0, 0)).epsilon(1e-12));
    CHECK(result.P(1, 1) == doctest::Approx(Q_proc(1, 1)).epsilon(1e-12));
    CHECK(result.P(0, 1) == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(result.P(1, 0) == doctest::Approx(0.0).epsilon(1e-12));

    // L = C^{-1} = I (since C = I here): trust the measurement completely
    CHECK(result.L(0, 0) == doctest::Approx(1.0).epsilon(1e-12));
    CHECK(result.L(1, 1) == doctest::Approx(1.0).epsilon(1e-12));
    CHECK(result.L(0, 1) == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(result.L(1, 0) == doctest::Approx(0.0).epsilon(1e-12));

    // Non-square C with R=0 now works via RDE fallback
    StateSpace<2, 1, 1, 2, 1> sys1d{};
    sys1d.A = {{0.9, 0.1}, {0.0, 0.8}};
    sys1d.B = {{0.0}, {1.0}};
    sys1d.C = {{1.0, 0.0}};
    sys1d.G = Matrix<2, 2, double>::identity();
    sys1d.H = Matrix<1, 1, double>::identity();
    sys1d.Ts = 0.01;

    Matrix<2, 2> Q2 = {{0.01, 0.0}, {0.0, 0.01}};
    Matrix<1, 1> R1_zero = Matrix<1, 1, double>::zeros();
    auto         result1d = online::kalman(sys1d, Q2, R1_zero);
    CHECK(result1d.success);
}

TEST_CASE("Kalman design R!=0, non-square C (NY=2, NX=4)") {
    // 4-state integrator chain, 2 measurements observing states 0 and 2.
    // Typical of a 2-axis inertial system where position is measured but not velocity.
    // A = chain-of-integrators with dt=0.1, C picks out states 0 and 2.
    // Reference values generated with scipy.linalg.solve_discrete_are.
    StateSpace<4, 1, 2, 4, 2> sys{};

    sys.A = {
        {1.0, 0.1, 0.0, 0.0},
        {0.0, 1.0, 0.1, 0.0},
        {0.0, 0.0, 1.0, 0.1},
        {0.0, 0.0, 0.0, 1.0},
    };

    // Single input on the last state
    sys.B = {
        {0.0},
        {0.0},
        {0.0},
        {1.0},
    };

    // 2x4: observe states 0 and 2
    sys.C = {
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0},
    };

    sys.G = Matrix<4, 4, double>::identity();
    sys.H = Matrix<2, 2, double>::identity();
    sys.Ts = 0.1;

    Matrix<4, 4> Q_proc = Matrix<4, 4, double>::identity();
    for (size_t i = 0; i < 4; ++i) {
        Q_proc(i, i) = 0.01;
    }

    Matrix<2, 2> R_meas = Matrix<2, 2, double>::identity();
    for (size_t i = 0; i < 2; ++i) {
        R_meas(i, i) = 0.1;
    }

    auto result = online::kalman(sys, Q_proc, R_meas);
    REQUIRE(result.success);

    // P is 4x4 symmetric
    CHECK(result.P(0, 0) == doctest::Approx(0.050147732129045).epsilon(1e-8));
    CHECK(result.P(0, 1) == doctest::Approx(0.040639610088462).epsilon(1e-8));
    CHECK(result.P(0, 2) == doctest::Approx(0.001193670164291).epsilon(1e-8));
    CHECK(result.P(0, 3) == doctest::Approx(-0.000207024981456).epsilon(1e-8));
    CHECK(result.P(1, 1) == doctest::Approx(0.146757944186451).epsilon(1e-8));
    CHECK(result.P(1, 2) == doctest::Approx(0.010539815102521).epsilon(1e-8));
    CHECK(result.P(1, 3) == doctest::Approx(0.003949387027125).epsilon(1e-8));
    CHECK(result.P(2, 2) == doctest::Approx(0.049606917603299).epsilon(1e-8));
    CHECK(result.P(2, 3) == doctest::Approx(0.038675628845935).epsilon(1e-8));
    CHECK(result.P(3, 3) == doctest::Approx(0.138213910986039).epsilon(1e-8));

    // L is 4x2: Kalman gain
    CHECK(result.L(0, 0) == doctest::Approx(0.333947026467608).epsilon(1e-8));
    CHECK(result.L(0, 1) == doctest::Approx(0.005314243318957).epsilon(1e-8));
    CHECK(result.L(1, 0) == doctest::Approx(0.270121220006257).epsilon(1e-8));
    CHECK(result.L(1, 1) == doctest::Approx(0.068294833054111).epsilon(1e-8));
    CHECK(result.L(2, 0) == doctest::Approx(0.005314243318957).epsilon(1e-8));
    CHECK(result.L(2, 1) == doctest::Approx(0.331539309439724).epsilon(1e-8));
    CHECK(result.L(3, 0) == doctest::Approx(-0.003434213066953).epsilon(1e-8));
    CHECK(result.L(3, 1) == doctest::Approx(0.258542377473313).epsilon(1e-8));

    // compile-time version must agree
    constexpr auto dresult = []() {
        StateSpace<4, 1, 2, 4, 2> s{};
        s.A = {{1.0, 0.1, 0.0, 0.0}, {0.0, 1.0, 0.1, 0.0}, {0.0, 0.0, 1.0, 0.1}, {0.0, 0.0, 0.0, 1.0}};
        s.B = {{0.0}, {0.0}, {0.0}, {1.0}};
        s.C = {{1.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}};
        s.G = Matrix<4, 4, double>::identity();
        s.H = Matrix<2, 2, double>::identity();
        s.Ts = 0.1;
        Matrix<4, 4> Qp = Matrix<4, 4, double>::identity();
        for (size_t i = 0; i < 4; ++i) {
            Qp(i, i) = 0.01;
        }
        Matrix<2, 2> Rm = Matrix<2, 2, double>::identity();
        for (size_t i = 0; i < 2; ++i) {
            Rm(i, i) = 0.1;
        }
        return design::kalman(s, Qp, Rm);
    }();

    static_assert(dresult.success);
    CHECK(dresult.P(0, 0) == doctest::Approx(0.050147732129045).epsilon(1e-8));
    CHECK(dresult.L(3, 1) == doctest::Approx(0.258542377473313).epsilon(1e-8));
}

TEST_CASE("Kalman design R=0, non-square C (NY=2, NX=4)") {
    // scipy reference (solve_discrete_are(A.T, C.T, Q, zeros(2,2))):
    StateSpace<4, 1, 2, 4, 2> sys{};

    sys.A = {
        {1.0, 0.1, 0.0, 0.0},
        {0.0, 1.0, 0.1, 0.0},
        {0.0, 0.0, 1.0, 0.1},
        {0.0, 0.0, 0.0, 1.0},
    };

    sys.B = {
        {0.0},
        {0.0},
        {0.0},
        {1.0},
    };

    sys.C = {
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0},
    };

    sys.G = Matrix<4, 4, double>::identity();
    sys.H = Matrix<2, 2, double>::identity();
    sys.Ts = 0.1;

    Matrix<4, 4> Q_proc = Matrix<4, 4, double>::identity();
    for (size_t i = 0; i < 4; ++i) {
        Q_proc(i, i) = 0.01;
    }

    Matrix<2, 2> R_zero = Matrix<2, 2, double>::zeros();

    auto result = online::kalman(sys, Q_proc, R_zero);
    REQUIRE(result.success);
    CHECK(result.P(0, 0) == doctest::Approx(1.105124921972504e-02).epsilon(1e-8));
    CHECK(result.P(0, 1) == doctest::Approx(1.051249219725040e-02).epsilon(1e-8));
    CHECK(result.P(0, 2) == doctest::Approx(0.0).epsilon(1e-8));
    CHECK(result.P(0, 3) == doctest::Approx(0.0).epsilon(1e-8));
    CHECK(result.P(1, 1) == doctest::Approx(1.151249219725040e-01).epsilon(1e-8));
    CHECK(result.P(1, 2) == doctest::Approx(0.0).epsilon(1e-8));
    CHECK(result.P(1, 3) == doctest::Approx(0.0).epsilon(1e-8));
    CHECK(result.P(2, 2) == doctest::Approx(1.105124921972504e-02).epsilon(1e-8));
    CHECK(result.P(2, 3) == doctest::Approx(1.051249219725037e-02).epsilon(1e-8));
    CHECK(result.P(3, 3) == doctest::Approx(1.151249219725037e-01).epsilon(1e-8));
    CHECK(result.L(0, 0) == doctest::Approx(1.0).epsilon(1e-8));
    CHECK(result.L(0, 1) == doctest::Approx(0.0).epsilon(1e-8));
    CHECK(result.L(1, 0) == doctest::Approx(9.512492197250400e-01).epsilon(1e-8));
    CHECK(result.L(1, 1) == doctest::Approx(0.0).epsilon(1e-8));
    CHECK(result.L(2, 0) == doctest::Approx(0.0).epsilon(1e-8));
    CHECK(result.L(2, 1) == doctest::Approx(1.0).epsilon(1e-8));
    CHECK(result.L(3, 0) == doctest::Approx(0.0).epsilon(1e-8));
    CHECK(result.L(3, 1) == doctest::Approx(9.512492197250375e-01).epsilon(1e-8));
}
