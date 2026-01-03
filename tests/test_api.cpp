#include <cmath>

#include "control_design.hpp"
#include "doctest.h"

using namespace wetmelon::control;

/**
 * @brief Tests for MATLAB®-style control design API functions
 *
 * Tests the high-level API: dlqr, lqrd, lqi, lqg, lqgtrack, lqgreg
 */

TEST_SUITE("MATLAB®-Style Control Design API") {
    // Test dlqr: Discrete LQR from A, B, Q, R matrices
    TEST_CASE("dlqr: discrete LQR design") {
        // Double integrator (discrete): x[k+1] = [1 0.1; 0 1]*x[k] + [0.005; 0.1]*u[k]
        constexpr double Ts = 0.1;
        Matrix<2, 2>     Ad{{1.0, Ts}, {0.0, 1.0}};
        Matrix<2, 1>     Bd{{Ts * Ts / 2}, {Ts}};
        Matrix<2, 2>     Q{{1.0, 0.0}, {0.0, 1.0}};
        Matrix<1, 1>     R{{0.1}};

        auto result = online::dlqr(Ad, Bd, Q, R);

        // Should produce non-zero gain
        CHECK(result.K(0, 0) != 0.0);
        CHECK(result.K(0, 1) != 0.0);

        // S should be positive definite (check diagonal)
        CHECK(result.S(0, 0) > 0.0);
        CHECK(result.S(1, 1) > 0.0);
    }

    // Test lqrd: Discrete LQR from continuous plant
    TEST_CASE("lqrd: discrete LQR for continuous plant") {
        Matrix<2, 2> A{{0.0, 1.0}, {-2.0, -3.0}};
        Matrix<2, 1> B{{0.0}, {1.0}};
        Matrix<2, 2> Q = Matrix<2, 2>::identity();
        Matrix<1, 1> R{{0.1}};
        double       Ts = 0.01;

        auto result = online::lqrd(A, B, Q, R, Ts);

        // Should compute valid discrete gains
        CHECK(result.K(0, 0) != 0.0);
        CHECK(result.K(0, 1) != 0.0);
    }

    // Test lqi: LQI controller design
    TEST_CASE("lqi: LQI controller design") {
        StateSpace<2, 1, 1> sys{
            Matrix<2, 2>{{0.0, 1.0}, {0.0, 0.0}}, // double integrator
            Matrix<2, 1>{{0.0}, {1.0}},
            Matrix<1, 2>{{1.0, 0.0}} // position output
        };

        // Augmented cost: [state; integral]
        Matrix<3, 3> Q_aug{};
        Q_aug(0, 0) = 1.0;  // position
        Q_aug(1, 1) = 1.0;  // velocity
        Q_aug(2, 2) = 10.0; // integral (high weight for tracking)
        Matrix<1, 1> R{{0.1}};

        auto lqi_result = online::lqi(sys, Q_aug, R);

        // Should have computed gains
        CHECK(lqi_result.Kx(0, 0) != 0.0);
        CHECK(lqi_result.Ki(0, 0) != 0.0);
    }

    // Test lqg: LQG regulator design
    TEST_CASE("lqg: LQG regulator design") {
        StateSpace<2, 1, 1> sys{
            Matrix<2, 2>{{1.0, 0.1}, {0.0, 1.0}}, // discrete double integrator
            Matrix<2, 1>{{0.005}, {0.1}},
            Matrix<1, 2>{{1.0, 0.0}}
        };

        Matrix<2, 2> Q_lqr = Matrix<2, 2>::identity();
        Matrix<1, 1> R_lqr{{0.1}};
        Matrix<2, 2> Q_kf{{0.01, 0.0}, {0.0, 0.01}};
        Matrix<1, 1> R_kf{{0.1}};

        auto lqg_result = online::lqg(sys, Q_lqr, R_lqr, Q_kf, R_kf);

        // LQR gain should exist
        CHECK(lqg_result.lqr.K(0, 0) != 0.0);
        CHECK(lqg_result.lqr.K(0, 1) != 0.0);
    }

    // Test lqgtrack: LQG servo controller
    TEST_CASE("lqgtrack: LQG servo design") {
        StateSpace<2, 1, 1> sys{
            Matrix<2, 2>{{1.0, 0.1}, {0.0, 1.0}},
            Matrix<2, 1>{{0.005}, {0.1}},
            Matrix<1, 2>{{1.0, 0.0}}
        };

        Matrix<3, 3> Q_aug{};
        Q_aug(0, 0) = 1.0;
        Q_aug(1, 1) = 1.0;
        Q_aug(2, 2) = 10.0;
        Matrix<1, 1> R{{0.1}};
        Matrix<2, 2> Q_kf{{0.01, 0.0}, {0.0, 0.01}};
        Matrix<1, 1> R_kf{{0.1}};

        auto servo_result = online::lqgtrack(sys, Q_aug, R, Q_kf, R_kf);

        // Should have computed gains
        CHECK(servo_result.lqi.Kx(0, 0) != 0.0);
        CHECK(servo_result.lqi.Ki(0, 0) != 0.0);
    }

    // Test lqgreg: Form LQG from Kalman and LQR results
    TEST_CASE("lqgreg: combine Kalman result and LQR result") {
        StateSpace<2, 1, 1> sys{
            Matrix<2, 2>{{1.0, 0.1}, {0.0, 1.0}},
            Matrix<2, 1>{{0.005}, {0.1}},
            Matrix<1, 2>{{1.0, 0.0}}
        };

        // Create Kalman result
        Matrix<2, 2> Q_kf{{0.01, 0.0}, {0.0, 0.01}};
        Matrix<1, 1> R_kf{{0.1}};
        auto         kf_result = online::kalman(sys, Q_kf, R_kf);

        // Create LQR result using dlqr free function
        Matrix<2, 2> Q_lqr = Matrix<2, 2>::identity();
        Matrix<1, 1> R_lqr{{0.1}};
        auto         lqr_result = online::dlqr(sys.A, sys.B, Q_lqr, R_lqr);

        // Combine into LQGResult
        auto lqg_result = online::lqgreg(kf_result, lqr_result);

        // Should preserve the LQR gain
        CHECK(lqg_result.lqr.K(0, 0) == lqr_result.K(0, 0));
        CHECK(lqg_result.lqr.K(0, 1) == lqr_result.K(0, 1));
    }

    // Test compile-time design (consteval)
    TEST_CASE("design:: consteval functions compile") {
        // This test verifies that design:: functions can be evaluated at compile time
        constexpr Matrix<2, 2> Ad{{1.0, 0.1}, {0.0, 1.0}};
        constexpr Matrix<2, 1> Bd{{0.005}, {0.1}};
        constexpr Matrix<2, 2> Q{{1.0, 0.0}, {0.0, 1.0}};
        constexpr Matrix<1, 1> R{{0.1}};

        // Note: We can't use consteval directly in runtime CHECK, but we can
        // verify the design functions work at compile time by using constexpr
        constexpr auto result = design::dlqr(Ad, Bd, Q, R);

        // At runtime, verify the compile-time result
        CHECK(result.K(0, 0) != 0.0);
        CHECK(result.K(0, 1) != 0.0);
        CHECK(result.S(0, 0) > 0.0);
    }
}
