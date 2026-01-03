#include <cmath>

#include "control_design.hpp"
#include "doctest.h"

using namespace wetmelon::control;

/**
 * @brief Tests for discretization methods (ZOH, Tustin)
 *
 * These tests verify the accuracy and correctness of continuous-to-discrete
 * system transformations.
 */

TEST_SUITE("Control Design: Discretization") {
    // Test 1: Simple 1st-order system discretization via ZOH
    // System: dx/dt = -x + u (time constant = 1 second)
    // Analytical solution: x[k+1] = exp(-Ts)*x[k] + (1-exp(-Ts))*u[k]
    TEST_CASE("ZOH Discretization: 1st-order system") {
        double     Ts = 0.1; // 100ms sampling time
        StateSpace sys{
            Matrix<1, 1>{{-1.0}}, // A: dx/dt = -x
            Matrix<1, 1>{{1.0}},  // B: input gain
            Matrix<1, 1>{{1.0}}   // C: output is state
        };

        // Use new unified discretize() function
        StateSpace sys_d = discretize(sys, Ts, DiscretizationMethod::ZOH);

        // Verify A_d ≈ exp(-Ts)
        double A_d_expected = std::exp(-Ts);
        double A_d_actual = sys_d.A(0, 0);
        CHECK(doctest::Approx(A_d_actual).epsilon(1e-4) == A_d_expected);

        // Verify B_d ≈ 1 - exp(-Ts)
        double B_d_expected = 1.0 - std::exp(-Ts);
        double B_d_actual = sys_d.B(0, 0);
        CHECK(doctest::Approx(B_d_actual).epsilon(1e-4) == B_d_expected);
    }

    // Test 2: Tustin Discretization
    TEST_CASE("Tustin Discretization: 1st-order system") {
        double     Ts = 0.1;
        StateSpace sys{
            Matrix<1, 1>{{-1.0}},
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{1.0}}
        };

        // Use new unified discretize() function
        StateSpace sys_d = discretize(sys, Ts, DiscretizationMethod::Tustin);

        // For Tustin with A=-1:
        // A_d = (I + A*Ts/2)^{-1} * (I - A*Ts/2)
        //     = (1 + (-1)*0.05)^{-1} * (1 - (-1)*0.05)
        //     = (0.95)^{-1} * (1.05)
        //     = 1.10526...
        double expected_A_d = (2.0 + Ts) / (2.0 - Ts);
        CHECK(doctest::Approx(sys_d.A(0, 0)).epsilon(1e-4) == expected_A_d);
    }

    // Test 3: ZOH vs Tustin comparison using unified API
    TEST_CASE("ZOH and Tustin produce different discretizations") {
        double     Ts = 0.5;
        StateSpace sys{
            Matrix<1, 1>{{-2.0}},
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{1.0}}
        };

        StateSpace sys_d_zoh = discretize(sys, Ts, DiscretizationMethod::ZOH);
        StateSpace sys_d_tustin = discretize(sys, Ts, DiscretizationMethod::Tustin);

        // They should produce different results
        CHECK_NE(sys_d_zoh.A(0, 0), doctest::Approx(sys_d_tustin.A(0, 0)));
    }

    // Test 4: Default method is ZOH
    TEST_CASE("Default discretization method is ZOH") {
        double     Ts = 0.1;
        StateSpace sys{
            Matrix<1, 1>{{-1.0}},
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{1.0}}
        };

        StateSpace sys_d_default = discretize(sys, Ts);
        StateSpace sys_d_zoh = discretize(sys, Ts, DiscretizationMethod::ZOH);

        CHECK(doctest::Approx(sys_d_default.A(0, 0)) == sys_d_zoh.A(0, 0));
        CHECK(doctest::Approx(sys_d_default.B(0, 0)) == sys_d_zoh.B(0, 0));
    }
}

TEST_SUITE("Control Design: Discrete LQR from Continuous") {
    // Test 1: Design discrete controller from continuous system using lqrd
    TEST_CASE("Discrete LQR from Continuous") {
        double Ts = 0.1;

        StateSpace<1, 1, 1> sys{
            Matrix<1, 1>{{-1.0}},
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{1.0}}
        };

        Matrix<1, 1, double> Q{{1.0}};
        Matrix<1, 1, double> R{{1.0}};

        auto controller = online::discrete_lqr_from_continuous(sys, Q, R, Ts);

        // Gain should be non-zero and reasonable
        CHECK(controller.K(0, 0) >= 0.0);
        CHECK(controller.K(0, 0) <= 3.0); // Relaxed upper bound for discretized controllers
        CHECK(controller.S(0, 0) > 0.0);  // DARE solution should be positive
    }

    // Test 2: Test with larger system (2x2)
    TEST_CASE("Discrete LQR from Continuous (2D system)") {
        // Double integrator: [x1; x2] with x1' = x2, x2' = u
        double Ts = 0.1;

        StateSpace<2, 1, 2> sys{
            Matrix<2, 2>{{0.0, 1.0}, {0.0, 0.0}}, // A: [0 1; 0 0]
            Matrix<2, 1>{{0.0}, {1.0}},           // B: [0; 1]
            Matrix<2, 2>{{1.0, 0.0}, {0.0, 1.0}}  // C: identity
        };

        Matrix<2, 2>         Q = Matrix<2, 2, double>::identity();
        Matrix<1, 1, double> R{{1.0}};

        auto controller = online::discrete_lqr_from_continuous(sys, Q, R, Ts);

        // Both gains should be non-zero
        CHECK(controller.K(0, 0) > 0.0); // Position gain
        CHECK(controller.K(0, 1) > 0.0); // Velocity gain
    }
}
