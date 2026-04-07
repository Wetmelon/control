#include <cmath>

#include "discretization.hpp"
#include "doctest.h"
#include "lqr.hpp"
#include "matrix.hpp"
#include "state_space.hpp"

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

        // For Tustin with A=-1, Ts=0.1:
        // M = (I - A*Ts/2)^{-1} = (1 + 0.05)^{-1} = 1/1.05
        // A_d = M * (I + A*Ts/2) = (1/1.05) * 0.95 = (2-Ts)/(2+Ts)
        // B_d = Ts * M * B = 0.1/1.05
        // C_d = C * M = 1/1.05
        // D_d = D + (Ts/2) * C_d * B = 0 + 0.05/1.05
        double expected_A_d = (2.0 - Ts) / (2.0 + Ts);       // 0.904762...
        double expected_B_d = Ts / (1.0 + Ts / 2.0);         // 0.095238...
        double expected_C_d = 1.0 / (1.0 + Ts / 2.0);        // 0.952381...
        double expected_D_d = (Ts / 2.0) / (1.0 + Ts / 2.0); // 0.047619...

        CHECK(sys_d.A(0, 0) == doctest::Approx(expected_A_d).epsilon(1e-12));
        CHECK(sys_d.B(0, 0) == doctest::Approx(expected_B_d).epsilon(1e-12));
        CHECK(sys_d.C(0, 0) == doctest::Approx(expected_C_d).epsilon(1e-12));
        CHECK(sys_d.D(0, 0) == doctest::Approx(expected_D_d).epsilon(1e-12));

        // Discrete pole should be inside unit circle (stable system stays stable)
        CHECK(std::abs(sys_d.A(0, 0)) < 1.0);
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
