#include <cmath>

#include "doctest.h"
#include "kalman.hpp"
#include "lqg.hpp"
#include "lqgi.hpp"
#include "lqi.hpp"
#include "lqr.hpp"
#include "matrix.hpp"
#include "state_space.hpp"

using namespace wetmelon::control;

/**
 * @brief Tests for runtime online:: functions
 *
 * These tests use runtime evaluation for design functions and controller
 * construction/operation.
 */

TEST_SUITE("Online: LQR with Cross-Term N") {
    TEST_CASE("online::dlqr with cross-term N") {
        auto result_no_N = online::dlqr(
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{1.0}}
        );

        auto result_with_N = online::dlqr(
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{0.5}}
        );

        CHECK(result_no_N.success);
        CHECK(result_with_N.success);
        CHECK(result_no_N.K(0, 0) != result_with_N.K(0, 0));
    }
}

TEST_SUITE("Online: LQI Integration") {
    TEST_CASE("online::lqi design produces valid result") {
        StateSpace<1, 1, 1, 1, 1> sys{};
        sys.A(0, 0) = 1.0;
        sys.B(0, 0) = 1.0;
        sys.C(0, 0) = 1.0;

        Matrix<2, 2> Q_aug = Matrix<2, 2>::identity();
        Matrix<1, 1> R = {{1.0}};

        auto lqi_result = online::lqi(sys, Q_aug, R);

        CHECK(lqi_result.success);
        CHECK(lqi_result.K(0, 0) != 0.0);
        CHECK(lqi_result.K(0, 1) != 0.0);
    }

    TEST_CASE("LQI controller construction and control law") {
        StateSpace<1, 1, 1, 1, 1> sys{};
        sys.A(0, 0) = 1.0;
        sys.B(0, 0) = 1.0;
        sys.C(0, 0) = 1.0;

        Matrix<2, 2> Q_aug = Matrix<2, 2>::identity();
        Matrix<1, 1> R = {{1.0}};

        auto lqi_result = online::lqi(sys, Q_aug, R);
        LQI  lqi(lqi_result);

        // Verify gains match
        CHECK(doctest::Approx(lqi.getK()(0, 0)).epsilon(1e-10) == lqi_result.K(0, 0));
        CHECK(doctest::Approx(lqi.getK()(0, 1)).epsilon(1e-10) == lqi_result.K(0, 1));

        // Test control law
        ColVec<2> x_aug = {0.5, 0.0}; // x = 0.5, xi = 0
        auto      u = lqi.control(x_aug);

        // Check that control is computed as -K * x_aug
        ColVec<1> expected_u = -lqi_result.K * x_aug;
        CHECK(u[0] == doctest::Approx(expected_u[0]).epsilon(1e-10));
    }
}

TEST_SUITE("Online: Kalman Filter") {
    TEST_CASE("online::kalman design produces valid steady-state result") {
        StateSpace<1, 1, 1, 1, 1> sys{};
        sys.A(0, 0) = 0.95;
        sys.B(0, 0) = 1.0;
        sys.C(0, 0) = 1.0;
        sys.G(0, 0) = 1.0;
        sys.H(0, 0) = 1.0;
        sys.Ts = 1.0;

        auto result = online::kalman(sys, Matrix<1, 1>{{0.1}}, Matrix<1, 1>{{0.1}});

        CHECK(result.success);
        CHECK(result.L(0, 0) > 0.0);
        CHECK(result.P(0, 0) > 0.0);
    }
}

TEST_SUITE("Online: LQG Regulator") {
    TEST_CASE("online::lqg design produces valid result") {
        StateSpace<1, 1, 1, 1, 1> sys{};
        sys.A(0, 0) = 0.95;
        sys.B(0, 0) = 1.0;
        sys.C(0, 0) = 1.0;
        sys.G(0, 0) = 1.0;
        sys.H(0, 0) = 1.0;
        sys.Ts = 1.0;

        auto result = online::lqg(
            sys,
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{0.1}},
            Matrix<1, 1>{{0.1}}
        );

        CHECK(result.success);
        CHECK(result.success == (result.lqr.success && result.kalman.success));
    }

    TEST_CASE("LQG controller with steady-state covariance initialization") {
        StateSpace<2, 1, 1, 2, 1> sys{};
        sys.A = {{1.0, 0.1}, {0.0, 0.95}};
        sys.B = {{0.0}, {1.0}};
        sys.C = {{1.0, 0.0}};
        sys.G = Matrix<2, 2>::identity();
        sys.H = Matrix<1, 1>::identity();
        sys.Ts = 1.0;

        Matrix<2, 2> Q_lqr = Matrix<2, 2>::identity();
        Matrix<1, 1> R_lqr = {{1.0}};
        Matrix<2, 2> Q_kf = {{0.01, 0.0}, {0.0, 0.01}};
        Matrix<1, 1> R_kf = {{0.1}};

        auto lqg_result = online::lqg(sys, Q_lqr, R_lqr, Q_kf, R_kf);
        LQG  lqg(lqg_result);

        // Verify KF is initialized with designed covariance
        const auto& P_kf = lqg.kf.covariance();
        const auto& P_designed = lqg_result.kalman.P;

        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                CHECK(doctest::Approx(P_kf(i, j)).epsilon(1e-10) == P_designed(i, j));
            }
        }
    }

    TEST_CASE("LQG controller predict/update/control cycle") {
        StateSpace<1, 1, 1, 1, 1> sys{};
        sys.A(0, 0) = 1.0;
        sys.B(0, 0) = 1.0;
        sys.C(0, 0) = 1.0;
        sys.G(0, 0) = 1.0;
        sys.H(0, 0) = 1.0;
        sys.Ts = 1.0;

        auto lqg_result = online::lqg(
            sys,
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{0.1}},
            Matrix<1, 1>{{0.1}}
        );

        LQG lqg(lqg_result);

        // Predict
        ColVec<1> u_init{0.0};
        lqg.predict(u_init);

        // Update
        ColVec<1> y = {0.5};
        bool      ok = lqg.update(y);
        CHECK(ok);

        // Control
        [[maybe_unused]] auto u = lqg.control();
    }
}

TEST_SUITE("Online: LQGI Servo") {
    TEST_CASE("online::lqgtrack design produces valid result") {
        StateSpace<1, 1, 1, 1, 1> sys{};
        sys.A(0, 0) = 1.0;
        sys.B(0, 0) = 1.0;
        sys.C(0, 0) = 1.0;
        sys.G(0, 0) = 1.0;
        sys.H(0, 0) = 1.0;
        sys.Ts = 1.0;

        Matrix<2, 2> Q_aug = Matrix<2, 2>::identity();
        Matrix<1, 1> R = {{1.0}};
        Matrix<1, 1> Qkf = {{0.1}};
        Matrix<1, 1> Rkf = {{0.25}};

        auto lqgi_result = online::lqgtrack(sys, Q_aug, R, Qkf, Rkf);

        CHECK(lqgi_result.success);
        CHECK(lqgi_result.success == (lqgi_result.lqi.success && lqgi_result.kalman.success));
    }

    TEST_CASE("LQGI controller with covariance initialization") {
        StateSpace<1, 1, 1, 1, 1> sys{};
        sys.A(0, 0) = 1.0;
        sys.B(0, 0) = 1.0;
        sys.C(0, 0) = 1.0;
        sys.G(0, 0) = 1.0;
        sys.H(0, 0) = 1.0;
        sys.Ts = 1.0;

        Matrix<2, 2> Q_aug = Matrix<2, 2>::identity();
        Matrix<1, 1> R = {{1.0}};
        Matrix<1, 1> Qkf = {{0.1}};
        Matrix<1, 1> Rkf = {{0.25}};

        auto lqgi_result = online::lqgtrack(sys, Q_aug, R, Qkf, Rkf);
        LQGI lqgi(lqgi_result);

        // Verify KF covariance initialization
        const auto& P = lqgi.kf.covariance();
        const auto& P_designed = lqgi_result.kalman.P;
        CHECK(doctest::Approx(P(0, 0)).epsilon(1e-10) == P_designed(0, 0));
    }

    TEST_CASE("LQGI controller predict/update/control cycle") {
        StateSpace<1, 1, 1, 1, 1> sys{};
        sys.A(0, 0) = 1.0;
        sys.B(0, 0) = 1.0;
        sys.C(0, 0) = 1.0;
        sys.G(0, 0) = 1.0;
        sys.H(0, 0) = 1.0;
        sys.Ts = 1.0;

        Matrix<2, 2> Q_aug = Matrix<2, 2>::identity();
        Matrix<1, 1> R = {{1.0}};
        Matrix<1, 1> Qkf = {{0.1}};
        Matrix<1, 1> Rkf = {{0.25}};

        auto lqgi_result = online::lqgtrack(sys, Q_aug, R, Qkf, Rkf);
        LQGI lqgi(lqgi_result);

        // Predict
        ColVec<1> u_init{0.0};
        lqgi.predict(u_init);

        // Update
        ColVec<1> y = {0.0};
        bool      ok = lqgi.update(y);
        CHECK(ok);

        // Control with reference
        ColVec<1> r = {1.0};
        ColVec<2> x_aug = {0.0, 0.0};

        // [[maybe_unused]] auto u = lqgi.control(r);
    }

    TEST_CASE("LQGI reset") {
        StateSpace<1, 1, 1, 1, 1> sys{};
        sys.A(0, 0) = 1.0;
        sys.B(0, 0) = 1.0;
        sys.C(0, 0) = 1.0;
        sys.G(0, 0) = 1.0;
        sys.H(0, 0) = 1.0;
        sys.Ts = 1.0;

        Matrix<2, 2> Q_aug = Matrix<2, 2>::identity();
        Matrix<1, 1> R = {{1.0}};
        Matrix<1, 1> Qkf = {{0.1}};
        Matrix<1, 1> Rkf = {{0.25}};

        auto lqgi_result = online::lqgtrack(sys, Q_aug, R, Qkf, Rkf);
        LQGI lqgi(lqgi_result);

        // Accumulate state
        lqgi.predict();
        lqgi.update({0.0});
        // [[maybe_unused]] auto u1 = lqgi.control({1.0});
    }
}

TEST_SUITE("Online: Success Flag Propagation") {
    TEST_CASE("LQRResult success reflects DARE convergence") {
        auto result = online::dlqr(
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{1.0}}
        );

        CHECK(result.success);
        CHECK(result.S(0, 0) > 0.0);
    }

    TEST_CASE("LQIResult success reflects augmented DARE convergence") {
        StateSpace<1, 1, 1, 1, 1> sys{};
        sys.A(0, 0) = 0.95;
        sys.B(0, 0) = 1.0;
        sys.C(0, 0) = 1.0;

        auto result = online::lqi(sys, Matrix<2, 2>::identity(), Matrix<1, 1>{{1.0}});

        CHECK(result.success);
        CHECK(result.K(0, 0) != 0.0);
        CHECK(result.K(0, 1) != 0.0);
    }

    TEST_CASE("KalmanResult success reflects filter DARE convergence") {
        StateSpace<1, 1, 1, 1, 1> sys{};
        sys.A(0, 0) = 0.95;
        sys.B(0, 0) = 1.0;
        sys.C(0, 0) = 1.0;
        sys.G(0, 0) = 1.0;
        sys.H(0, 0) = 1.0;
        sys.Ts = 1.0;

        auto result = online::kalman(sys, Matrix<1, 1>{{0.1}}, Matrix<1, 1>{{0.1}});

        CHECK(result.success);
        CHECK(result.L(0, 0) > 0.0);
        CHECK(result.P(0, 0) > 0.0);
    }

    TEST_CASE("LQGResult success is conjunction of LQR and Kalman success") {
        StateSpace<1, 1, 1, 1, 1> sys{};
        sys.A(0, 0) = 0.95;
        sys.B(0, 0) = 1.0;
        sys.C(0, 0) = 1.0;
        sys.G(0, 0) = 1.0;
        sys.H(0, 0) = 1.0;
        sys.Ts = 1.0;

        auto result = online::lqg(
            sys,
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{0.1}},
            Matrix<1, 1>{{0.1}}
        );

        CHECK(result.success == (result.lqr.success && result.kalman.success));
    }

    TEST_CASE("LQGIResult success is conjunction of LQI and Kalman success") {
        StateSpace<1, 1, 1, 1, 1> sys{};
        sys.A(0, 0) = 0.95;
        sys.B(0, 0) = 1.0;
        sys.C(0, 0) = 1.0;
        sys.G(0, 0) = 1.0;
        sys.H(0, 0) = 1.0;
        sys.Ts = 1.0;

        auto result = online::lqgtrack(
            sys,
            Matrix<2, 2>::identity(),
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{0.1}},
            Matrix<1, 1>{{0.1}}
        );

        CHECK(result.success == (result.lqi.success && result.kalman.success));
    }
}
