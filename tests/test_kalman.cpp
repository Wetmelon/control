#include <cmath>

#include "wet/controllers/lqg.hpp"
#include "wet/controllers/lqgi.hpp"
#include "wet/controllers/lqr.hpp"
#include "wet/estimation/ekf.hpp"
#include "wet/estimation/kalman.hpp"
#include "wet/systems/state_space.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wet;

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

    auto result = design::discrete_lqr(A, B, Q, R);

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

    LQG lqg = design::discrete_lqg(sys, Qlqr, Rlqr, Qkf, Rkf);

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

    LQGI lqgs = design::discrete_lqgi(sys, Q_aug, R, Qkf, Rkf);

    ColVec<1> y = {0.0};
    lqgs.update(y);

    // ColVec<1> r = {1.0};
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

    auto result = design::kalman(sys, Q_proc, R_meas);
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

    auto result = design::kalman(sys, Q_proc, R_meas);
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
    auto result = design::kalman(sys, Q_proc, R_zero);
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
    auto         result1d = design::kalman(sys1d, Q2, R1_zero);
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

    auto result = design::kalman(sys, Q_proc, R_meas);
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

    auto result = design::kalman(sys, Q_proc, R_zero);
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

TEST_CASE("Kalman design R=0 (NY=1, NX=4)") {
    StateSpace<4, 1, 1, 4, 1> sys{};

    sys.A = {
        {1.0000000000000000e+00, 1.2500000000000003e-04, 0.0, 0.0},
        {0.0, 1.0000000000000000e+00, 0.0, 0.0},
        {1.2044286656514691e-01, 7.6886479655468900e-06, 8.7955713343485309e-01, 0.0},
        {7.5633576541031576e-03, 3.2193761045041167e-07, 1.1287950891104376e-01, 8.7955713343485309e-01},
    };

    sys.B = {
        {7.8125000000000030e-09},
        {1.2500000000000003e-04},
        {3.2375688155733029e-10},
        {1.0189648978629236e-11},
    };

    sys.C = {{0.0, 0.0, 2.0, -1.0}};

    sys.G = Matrix<4, 4, double>::identity();
    sys.H = Matrix<1, 1, double>::identity();
    sys.Ts = 1.25e-4;

    // Qd from Van Loan discretization of continuous-time noise model
    Matrix<4, 4> Qd = {
        {6.5104166666666704e-13, 7.8125000000000046e-09, 3.0287979498172528e-14, 1.0167935310197761e-15},
        {7.8125000000000030e-09, 1.2500000000000003e-04, 3.2375688155733029e-10, 1.0189648978629238e-11},
        {3.0287979498172516e-14, 3.2375688155733013e-10, 2.7948205317517660e-10, 1.7167518071887303e-11},
        {1.0167935310197763e-15, 1.0189648978629233e-11, 1.7167518071887290e-11, 1.4368706442600565e-12},
    };

    Matrix<1, 1> R_zero = Matrix<1, 1, double>::zeros();

    auto result = design::kalman(sys, Qd, R_zero);
    REQUIRE(result.success);

    // Kalman gain (4x1): scipy reference
    CHECK(result.L(0, 0) == doctest::Approx(5.860848699642386e-01).epsilon(1e-4));
    CHECK(result.L(1, 0) == doctest::Approx(3.213133860108700e+02).epsilon(1e-4));
    CHECK(result.L(2, 0) == doctest::Approx(5.158533783244187e-01).epsilon(1e-4));
    CHECK(result.L(3, 0) == doctest::Approx(3.170675664883725e-02).epsilon(1e-4));

    // Covariance P (spot checks on diagonal and key off-diagonals)
    CHECK(result.P(0, 0) == doctest::Approx(3.1493750780952886e-09).epsilon(1e-4));
    CHECK(result.P(1, 1) == doctest::Approx(1.8865288001709108e-03).epsilon(1e-4));
    CHECK(result.P(2, 2) == doctest::Approx(3.2303061178439298e-10).epsilon(1e-4));
    CHECK(result.P(3, 3) == doctest::Approx(4.6007682900265547e-12).epsilon(1e-4));
    CHECK(result.P(0, 1) == doctest::Approx(1.7788444152705028e-06).epsilon(1e-4));
}

TEST_CASE("Kalman set_state clamps the estimate to a physical bound") {
    // 1D filter; the state represents a physically non-negative quantity (e.g.
    // a concentration / SoC). A measurement pulls the estimate negative; the
    // caller clamps it back to 0 via set_state, and the next predict/update
    // proceeds from the constrained value rather than the non-physical one.
    StateSpace<1, 1, 1, 1, 1> sys{};
    sys.A(0, 0) = 1.0;
    sys.C(0, 0) = 1.0;
    sys.G(0, 0) = 1.0;
    sys.H(0, 0) = 1.0;
    sys.Ts = 1.0;

    KalmanFilter<1, 1, 1, 1, 1> kf(sys, Matrix<1, 1>{{0.1}}, Matrix<1, 1>{{0.25}}, ColVec<1>{0.0}, Matrix<1, 1>::identity());

    kf.predict();
    REQUIRE(kf.update(ColVec<1>{-5.0})); // measurement drives estimate negative
    REQUIRE(kf.state()[0] < 0.0);

    if (kf.state()[0] < 0.0) {
        kf.set_state(0, 0.0); // clamp to physical floor
    }
    CHECK(kf.state()[0] == doctest::Approx(0.0));

    // Covariance setter likewise lets the caller re-inflate uncertainty after a
    // hard reset of the estimate.
    kf.set_covariance(Matrix<1, 1>{{2.0}});
    CHECK(kf.covariance()(0, 0) == doctest::Approx(2.0));

    // Vector setter overload.
    kf.set_state(ColVec<1>{0.7});
    CHECK(kf.state()[0] == doctest::Approx(0.7));
}
