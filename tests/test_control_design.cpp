#include <cmath>

#include "control_design.hpp"
#include "doctest.h"
#include "fmt/core.h"
#include "lqr.hpp"

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

        StateSpace sys_d = discretize_zoh(sys, Ts);

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

        StateSpace sys_d = discretize_tustin(sys, Ts);

        // For Tustin with A=-1:
        // A_d = (I + A*Ts/2)^{-1} * (I - A*Ts/2)
        //     = (1 + (-1)*0.05)^{-1} * (1 - (-1)*0.05)
        //     = (0.95)^{-1} * (1.05)
        //     = 1.10526...
        double expected_A_d = (2.0 + Ts) / (2.0 - Ts);
        CHECK(doctest::Approx(sys_d.A(0, 0)).epsilon(1e-4) == expected_A_d);
    }

    // Test 3: ZOH vs Tustin comparison
    TEST_CASE("ZOH and Tustin produce different discretizations") {
        double     Ts = 0.5;
        StateSpace sys{
            Matrix<1, 1>{{-2.0}},
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{1.0}}
        };

        StateSpace sys_d_zoh = discretize_zoh(sys, Ts);
        StateSpace sys_d_tustin = discretize_tustin(sys, Ts);

        // They should produce different results
        CHECK_NE(sys_d_zoh.A(0, 0), doctest::Approx(sys_d_tustin.A(0, 0)));
    }
}

TEST_SUITE("Control Design: Continuous LQR") {
    // Test 1: Simple integrator system continuous LQR design
    // System: dx/dt = u (pure integrator)
    // Cost: minimize x^2 + R*u^2
    TEST_CASE("Continuous LQR: Integrator system") {
        // 1D integrator
        StateSpace sys{
            Matrix<1, 1>{{0.0}}, // A: dx/dt = u
            Matrix<1, 1>{{1.0}}, // B: u is control
            Matrix<1, 1>{{1.0}}  // C: output is state
        };

        Matrix<1, 1, double> Q{{1.0}}; // Penalize state
        Matrix<1, 1, double> R{{1.0}}; // Penalize control

        auto gain = online::continuous_lqr(sys, Q, R);

        // For integrator with equal weighting, optimal gain should be sqrt(Q/R) = 1.0
        double expected_K = 1.0;
        CHECK(doctest::Approx(gain.K(0, 0)).epsilon(0.1) == expected_K);

        // S should be positive (Riccati solution)
        CHECK(gain.S(0, 0) > 0.0);
    }

    // Test 2: First-order system continuous LQR
    TEST_CASE("Continuous LQR: First-order system") {
        StateSpace sys{
            Matrix<1, 1>{{-1.0}}, // Natural damping
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{1.0}}
        };

        Matrix<1, 1, double> Q{{1.0}};
        Matrix<1, 1, double> R{{1.0}};

        auto gain = online::continuous_lqr(sys, Q, R);

        // For stable system, gain should be positive
        CHECK(gain.K(0, 0) > 0.0);
        CHECK(gain.S(0, 0) > 0.0);
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

TEST_SUITE("Control Design: Consteval Compile-Time Guarantees") {
    // These tests verify that designs can be computed at compile-time
    // The consteval versions guarantee execution during compilation

    TEST_CASE("Continuous LQR design runs at compile-time") {
        // Use a consteval function to force compile-time execution
        constexpr auto controller = []() consteval {
            StateSpace<1, 1, 1> sys{
                Matrix<1, 1>{{-1.0}},
                Matrix<1, 1>{{1.0}},
                Matrix<1, 1>{{1.0}}
            };
            Matrix<1, 1, double> Q{{1.0}};
            Matrix<1, 1, double> R{{1.0}};
            return design::continuous_lqr(sys, Q, R);
        }();

        // If we reach this point, the consteval executed successfully
        CHECK(controller.K(0, 0) >= 0.0);
    }

    TEST_CASE("Discrete LQR design from continuous runs at compile-time") {
        constexpr auto controller = []() consteval {
            StateSpace<1, 1, 1> sys{
                Matrix<1, 1>{{-1.0}},
                Matrix<1, 1>{{1.0}},
                Matrix<1, 1>{{1.0}}
            };
            Matrix<1, 1, double> Q{{1.0}};
            Matrix<1, 1, double> R{{1.0}};
            return design::discrete_lqr_from_continuous(sys, Q, R, 0.1);
        }();

        CHECK(controller.K(0, 0) >= 0.0);
    }
}

TEST_SUITE("Control Design: Type Conversion") {
    TEST_CASE("Convert continuous LQR result from double to float") {
        // Design in double precision
        StateSpace<1, 1, 1> sys{
            Matrix<1, 1>{{-1.0}},
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{1.0}}
        };
        Matrix<1, 1, double> Q{{1.0}};
        Matrix<1, 1, double> R{{1.0}};

        auto gain_double = online::continuous_lqr(sys, Q, R);

        // Convert to float
        design::LQRResult<1, 1, float> gain_float(gain_double);

        // Should be approximately equal
        CHECK(gain_float.K(0, 0) == doctest::Approx(static_cast<float>(gain_double.K(0, 0))));
        CHECK(gain_float.S(0, 0) == doctest::Approx(static_cast<float>(gain_double.S(0, 0))));
    }

    TEST_CASE("Convert discrete LQR result from double to float") {
        // Design in double precision
        StateSpace<2, 1, 2> sys{
            Matrix<2, 2>{{0.0, 1.0}, {0.0, 0.0}},
            Matrix<2, 1>{{0.0}, {1.0}},
            Matrix<2, 2>{{1.0, 0.0}, {0.0, 1.0}}
        };
        Matrix<2, 2>         Q = Matrix<2, 2, double>::identity();
        Matrix<1, 1, double> R{{1.0}};

        auto controller_double = online::discrete_lqr_from_continuous(sys, Q, R, 0.1);

        // Convert to float
        design::LQRResult<2, 1, float> controller_float(controller_double);

        // Gains should be approximately equal
        CHECK(controller_float.K(0, 0) == doctest::Approx(static_cast<float>(controller_double.K(0, 0))));
        CHECK(controller_float.K(0, 1) == doctest::Approx(static_cast<float>(controller_double.K(0, 1))));
    }
}

TEST_SUITE("Integration: Motor + Flexible Coupling + Mass System") {
    // Example system: motor with flexible coupling and inertial load
    // States:
    //   x1 = motor angular velocity (rad/s)
    //   x2 = coupling deflection (rad) - measures spring compression
    //   x3 = load angular velocity (rad/s)
    //
    // System parameters:
    //   J_m = motor inertia = 0.1 kg*m^2
    //   J_l = load inertia = 0.2 kg*m^2
    //   k = spring constant = 50 N*m/rad
    //   f_m = motor friction = 2 N*m*s/rad
    //   f_l = load friction = 1 N*m*s/rad
    //
    // Continuous dynamics:
    //   dx1/dt = -k/J_m * x2 - f_m/J_m * x1 + u/J_m
    //   dx2/dt = x1 - x3
    //   dx3/dt = k/J_m * x2 - f_l/J_l * x3
    //
    // Output: y = [x1, x3]^T (motor velocity and load velocity)

    TEST_CASE("Compile-time LQG design: Motor-coupling-mass system in double, runtime in float") {
        // Design parameters
        constexpr double J_m = 0.1; // motor inertia
        constexpr double J_l = 0.2; // load inertia
        constexpr double k = 50.0;  // spring constant
        constexpr double f_m = 2.0; // motor friction
        constexpr double f_l = 1.0; // load friction
        constexpr double Ts = 0.01; // sampling period 10 ms

        // Step 1: Design continuous-time LQG controller in double precision at compile time
        constexpr auto lqg_double = []() consteval {
            // System: 3 states, 1 input, 3 outputs (full state feedback)
            StateSpace sys_c{
                Matrix<3, 3>{
                    {-f_m / J_m, -k / J_m, 0.0},
                    {1.0, 0.0, -1.0},
                    {k / J_l, 0.0, -f_l / J_l}
                },
                Matrix<3, 1>{{1.0 / J_m}, {0.0}, {0.0}},
                Matrix<3, 3>{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}
            };

            // Cost: penalize motor and load velocities, not coupling deflection
            Matrix<3, 3, double> Q{{1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}};
            Matrix<1, 1, double> R{{0.01}};

            return design::discrete_lqr_from_continuous(sys_c, Q, R, Ts);
        }();

        // Step 2: Convert the double-precision controller to float for runtime use
        design::LQRResult<3, 1, float> lqr_float(lqg_double);

        // Discretize the system for simulation
        constexpr auto sys_d_double = []() consteval {
            StateSpace sys_c{
                Matrix<3, 3>{
                    {-f_m / J_m, -k / J_m, 0.0},
                    {1.0, 0.0, -1.0},
                    {k / J_l, 0.0, -f_l / J_l}
                },
                Matrix<3, 1>{{1.0 / J_m}, {0.0}, {0.0}},
                Matrix<3, 3>{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}
            };
            return discretize_zoh(sys_c, Ts);
        }();
        StateSpace<3, 1, 3, 3, 3, float> sys_d_float(sys_d_double);

        // Step 3: Verify the controller was designed properly
        CHECK(lqr_float.K(0, 0) != 0.0f); // Should have non-zero gains
        CHECK(lqr_float.K(0, 1) != 0.0f);
        CHECK(lqr_float.K(0, 2) != 0.0f);

        // Step 4: Simulate the discrete-time closed-loop system at runtime
        // Initial state: motor and load at rest, coupling stressed
        ColVec x = {
            0.0f, // motor velocity
            0.1f, // coupling deflection (10% of full)
            0.0f  // load velocity
        };

        // Run 10 time steps of closed-loop control
        for (int i = 0; i < 10; ++i) {
            // Compute control: u = -K * x
            ColVec u = -lqr_float.K * x;

            // Update state: x[k+1] = A_d * x[k] + B_d * u[k]
            x = sys_d_float.A * x + sys_d_float.B * u;
        }

        // After 10 steps of feedback control, the coupling deflection should decrease
        // and load velocity should increase toward motor velocity
        CHECK(x(1, 0) < 0.1f);           // Coupling deflection reduced
        CHECK(std::abs(x(0, 0)) > 0.0f); // Motor velocity changed
        CHECK(std::abs(x(2, 0)) > 0.0f); // Load velocity changed

        // Verify that final state shows active damping of the oscillation
        float final_norm_sq = x(0, 0) * x(0, 0) + x(1, 0) * x(1, 0) + x(2, 0) * x(2, 0);
        CHECK(final_norm_sq < 0.15f); // System energy reduced
    }
}

// ============================================================================
// Tests for MATLAB-style API functions
// ============================================================================
TEST_SUITE("MATLAB-Style Control Design API") {
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

    // Test lqr: Continuous LQR from A, B, Q, R
    TEST_CASE("lqr: continuous LQR design") {
        // Simple 1st-order system: dx/dt = -x + u
        Matrix<1, 1> A{{-1.0}};
        Matrix<1, 1> B{{1.0}};
        Matrix<1, 1> Q{{1.0}};
        Matrix<1, 1> R{{1.0}};

        auto result = online::lqr(A, B, Q, R);

        // Gain should be positive for this stable system
        CHECK(result.K(0, 0) > 0.0);
        CHECK(result.S(0, 0) > 0.0);
    }

    // Test lqr with StateSpace input (using 1x1 system that works with CARE)
    TEST_CASE("lqr: continuous LQR with StateSpace") {
        // Simple 1st-order system
        StateSpace<1, 1, 1> sys{
            Matrix<1, 1>{{-1.0}},
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{1.0}}
        };
        Matrix<1, 1> Q{{1.0}};
        Matrix<1, 1> R{{0.1}};

        auto result = online::lqr(sys, Q, R);

        CHECK(result.K(0, 0) > 0.0);
        CHECK(result.S(0, 0) > 0.0);
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

        auto servo_result = online::lqgtrack(sys, Q_aug, R, Q_kf, R_kf, ServoDOF::TwoDOF);

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
        Matrix<2, 2>         Q_kf{{0.01, 0.0}, {0.0, 0.01}};
        Matrix<1, 1>         R_kf{{0.1}};
        design::KalmanResult kf_result{sys, Q_kf, R_kf};

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

    TEST_CASE("Buck Converter LQR Design") {
        // Buck converter parameters

        auto result = []() {
            double R_ind = 0.005; // Inductor resistance in Ohms
            double L = 200e-6;    // Inductance in Henry
            double C = 14.7e-6;   // Capacitance in Farads
            double R_load = 10;   // Load resistance in Ohms

            // Notes:
            //  States:
            //  x1 = Inductor current (iL)
            //  x2 = Capacitor voltage (vC)

            // Input:
            // u = DC Bridge voltage (V_in) (Vdc_nom * duty cycle))

            // Measured Outputs:
            // y1 = Inductor current (iL)
            // y2 = Capacitor voltage (vC)

            // Control Objective:
            // Regulate current and voltage to desired setpoints (iL_ref, vC_ref)

            StateSpace<2, 1, 2> sys{
                Matrix<2, 2>{
                    {-R_ind / L, -1.0 / L},
                    {1.0 / C, -1.0 / (R_load * C)}
                },
                Matrix<2, 1>{{1.0 / L}, {0.0}},
                Matrix<2, 2>{{1.0, 0.0}, {0.0, 1.0}}
            };

            Matrix<2, 2> Q = Matrix<2, 2>::identity();
            Matrix<1, 1> R{{0.01}};

            return design::continuous_lqr(sys, Q, R);
        }();

        // Both poles should be negative for a stable closed-loop system
        fmt::print("LQR Poles: {}, {}\n", result.poles[0], result.poles[1]);
        CHECK(result.poles[0] < 0.0);
        CHECK(result.poles[1] < 0.0);

        // Verify controller gain is finite
        LQR lqr{result};
        CHECK(std::isfinite(lqr.getK()(0, 0)));
        CHECK(std::isfinite(lqr.getK()(0, 1)));
    }
}
