#include "control.hpp"
#include "doctest.h"

using namespace control;

TEST_CASE("Linear Quadratic Regulator (LQR)") {
    SUBCASE("Simple SISO system") {
        // System: dx/dt = -x + u, y = x
        Matrix A = Matrix::Constant(1, 1, -1.0);
        Matrix B = Matrix::Constant(1, 1, 1.0);
        Matrix Q = Matrix::Constant(1, 1, 1.0);
        Matrix R = Matrix::Constant(1, 1, 0.1);

        auto result = lqr(A, B, Q, R);

        // Check dimensions
        CHECK(result.K.rows() == 1);
        CHECK(result.K.cols() == 1);
        CHECK(result.S.rows() == 1);
        CHECK(result.S.cols() == 1);
        CHECK(result.P.size() == 1);

        // Check values against Python control.lqr
        CHECK(result.K(0, 0) == doctest::Approx(2.3166).epsilon(1e-3));
        CHECK(result.S(0, 0) == doctest::Approx(0.2317).epsilon(1e-3));
        CHECK(result.P[0].real() == doctest::Approx(-3.3166).epsilon(1e-3));

        // Check that gain is positive (since we want to stabilize)
        CHECK(result.K(0, 0) > 0);

        // Check closed-loop stability
        Matrix A_cl = A - B * result.K;
        CHECK(A_cl(0, 0) < 0);  // Should be stable

        // Check that Riccati solution is positive definite
        Eigen::SelfAdjointEigenSolver<Matrix> eigen(result.S);
        CHECK(eigen.eigenvalues().minCoeff() > -1e-10);
    }

    SUBCASE("MIMO system") {
        // 2-state, 1-input system
        Matrix A(2, 2);
        A << -1, 1, 0, -2;
        Matrix B(2, 1);
        B << 0, 1;
        Matrix Q = Matrix::Identity(2, 2);
        Matrix R = Matrix::Constant(1, 1, 0.1);

        auto result = lqr(A, B, Q, R);

        // Check dimensions
        CHECK(result.K.rows() == 1);
        CHECK(result.K.cols() == 2);
        CHECK(result.S.rows() == 2);
        CHECK(result.S.cols() == 2);
        CHECK(result.P.size() == 2);

        // Check values against Python control.lqr
        CHECK(result.K(0, 0) == doctest::Approx(0.9192).epsilon(1e-3));
        CHECK(result.K(0, 1) == doctest::Approx(1.9798).epsilon(1e-3));
        CHECK(result.S(0, 0) == doctest::Approx(0.4578).epsilon(1e-3));
        CHECK(result.S(1, 1) == doctest::Approx(0.1980).epsilon(1e-3));
        CHECK(result.P[0].real() == doctest::Approx(-1.3495).epsilon(1e-3));
        CHECK(result.P[1].real() == doctest::Approx(-3.6303).epsilon(1e-3));

        // Check closed-loop stability
        Matrix                     A_cl = A - B * result.K;
        Eigen::EigenSolver<Matrix> solver(A_cl);
        for (int i = 0; i < A_cl.rows(); ++i) {
            CHECK(solver.eigenvalues()(i).real() < 1e-10);
        }
    }

    SUBCASE("With cross-coupling term N") {
        Matrix A = Matrix::Constant(1, 1, -1.0);
        Matrix B = Matrix::Constant(1, 1, 1.0);
        Matrix Q = Matrix::Constant(1, 1, 1.0);
        Matrix R = Matrix::Constant(1, 1, 0.1);
        Matrix N = Matrix::Constant(1, 1, 0.5);

        auto result = lqr(A, B, Q, R, N);

        CHECK(result.K.rows() == 1);
        CHECK(result.K.cols() == 1);
    }

    SUBCASE("Error cases") {
        Matrix A = Matrix::Constant(1, 1, -1.0);
        Matrix B = Matrix::Constant(1, 1, 1.0);
        Matrix Q = Matrix::Constant(1, 1, 1.0);
        Matrix R = Matrix::Constant(1, 1, -0.1);  // Negative definite

        CHECK_THROWS_AS(lqr(A, B, Q, R), std::invalid_argument);
    }
}

// TEST_CASE("Linear Quadratic Integrator (LQI)") {
//     SUBCASE("SISO system with integral action") {
//         // Plant: dx/dt = -x + u, y = x
//         StateSpace plant{
//             Matrix::Constant(1, 1, -1.0),  // A
//             Matrix::Constant(1, 1, 1.0),   // B
//             Matrix::Constant(1, 1, 1.0),   // C
//             Matrix::Constant(1, 1, 0.0)    // D
//         };

//         Matrix Q = Matrix::Identity(2, 2);  // State + integral state
//         Matrix R = Matrix::Constant(1, 1, 0.1);

//         auto result = lqi(plant, Q, R);

//         // Augmented system has 2 states (original + integral)
//         CHECK(result.K.rows() == 1);
//         CHECK(result.K.cols() == 2);
//         CHECK(result.S.rows() == 2);
//         CHECK(result.S.cols() == 2);
//         CHECK(result.P.size() == 2);
//     }
// }

TEST_CASE("Discrete Linear Quadratic Regulator (DLQR)") {
    SUBCASE("Simple discrete system") {
        // Discrete system: x[k+1] = 0.9*x[k] + u[k]
        Matrix Ad = Matrix::Constant(1, 1, 0.9);
        Matrix Bd = Matrix::Constant(1, 1, 1.0);
        Matrix Q  = Matrix::Constant(1, 1, 1.0);
        Matrix R  = Matrix::Constant(1, 1, 0.1);

        auto result = dlqr(Ad, Bd, Q, R);

        CHECK(result.K.rows() == 1);
        CHECK(result.K.cols() == 1);
        CHECK(result.S.rows() == 1);
        CHECK(result.S.cols() == 1);
        CHECK(result.P.size() == 1);

        // Check values against Python control.dlqr
        CHECK(result.K(0, 0) == doctest::Approx(0.8233).epsilon(1e-3));
        CHECK(result.S(0, 0) == doctest::Approx(1.0741).epsilon(1e-3));
        CHECK(result.P[0].real() == doctest::Approx(0.0767).epsilon(1e-3));

        // Check closed-loop stability (|λ| < 1)
        Matrix A_cl = Ad - Bd * result.K;
        CHECK(std::abs(A_cl(0, 0)) < 1.0);
    }
}

TEST_CASE("LQR with discretization (LQRD)") {
    SUBCASE("Continuous system discretized") {
        Matrix A  = Matrix::Constant(1, 1, -1.0);
        Matrix B  = Matrix::Constant(1, 1, 1.0);
        Matrix Q  = Matrix::Constant(1, 1, 1.0);
        Matrix R  = Matrix::Constant(1, 1, 0.1);
        double Ts = 0.1;

        auto result = lqrd(A, B, Q, R, Ts);

        CHECK(result.K.rows() == 1);
        CHECK(result.K.cols() == 1);
        CHECK(result.S.rows() == 1);
        CHECK(result.S.cols() == 1);
        CHECK(result.P.size() == 1);
    }
}

TEST_CASE("Kalman Filter Synthesis") {
    SUBCASE("Continuous system") {
        StateSpace model{
            Matrix::Constant(1, 1, -1.0),  // A
            Matrix::Constant(1, 1, 1.0),   // B
            Matrix::Constant(1, 1, 1.0),   // C
            Matrix::Constant(1, 1, 0.0)    // D
        };

        Matrix Qn = Matrix::Constant(1, 1, 0.1);   // Process noise
        Matrix Rn = Matrix::Constant(1, 1, 0.01);  // Measurement noise

        auto result = kalman(model, Qn, Rn);

        CHECK(result.filter.getModel().A == model.A);
        CHECK(result.L.rows() == 1);
        CHECK(result.L.cols() == 1);
        CHECK(result.P.rows() == 1);
        CHECK(result.P.cols() == 1);
        // For continuous systems, Mx, Z, My should be nullopt
        CHECK(!result.Mx.has_value());
        CHECK(!result.Z.has_value());
        CHECK(!result.My.has_value());

        // Kalman gain should be positive
        CHECK(result.L(0, 0) > 0);
    }

    SUBCASE("Discrete system") {
        StateSpace model{
            Matrix::Constant(1, 1, 0.9),  // A
            Matrix::Constant(1, 1, 1.0),  // B
            Matrix::Constant(1, 1, 1.0),  // C
            Matrix::Constant(1, 1, 0.0),  // D
            0.1                           // Ts
        };

        Matrix Qn = Matrix::Constant(1, 1, 0.1);
        Matrix Rn = Matrix::Constant(1, 1, 0.01);

        auto result = kalman(model, Qn, Rn);

        CHECK(result.L.rows() == 1);
        CHECK(result.L.cols() == 1);
        CHECK(result.P.rows() == 1);
        CHECK(result.P.cols() == 1);
        // For discrete systems, Mx, Z, My should have values
        CHECK(result.Mx.has_value());
        CHECK(result.Z.has_value());
        CHECK(result.My.has_value());
        CHECK(result.Mx.value().rows() == 1);
        CHECK(result.Mx.value().cols() == 1);
        CHECK(result.Z.value().rows() == 1);
        CHECK(result.Z.value().cols() == 1);
        CHECK(result.My.value().rows() == 1);
        CHECK(result.My.value().cols() == 1);

        // Kalman gain should be positive
        CHECK(result.L(0, 0) > 0);
        // Innovation covariance should be positive definite
        CHECK(result.Mx.value()(0, 0) > 0);
    }

    SUBCASE("System with noise input matrices G and H") {
        // Create a system with G and H matrices
        StateSpace model{
            Matrix::Constant(2, 2, -1.0),  // A
            Matrix::Constant(2, 1, 1.0),   // B
            Matrix::Constant(1, 2, 1.0),   // C
            Matrix::Constant(1, 1, 0.0),   // D
            Matrix::Identity(2, 2),        // G (default)
            Matrix::Zero(1, 2)             // H (default)
        };

        // Set non-default G and H
        model.G = Matrix::Constant(2, 1, 0.5);  // G affects how process noise enters
        model.H = Matrix::Constant(1, 1, 0.2);  // H affects how process noise enters measurement

        Matrix Qn = Matrix::Constant(1, 1, 0.1);   // Process noise covariance
        Matrix Rn = Matrix::Constant(1, 1, 0.01);  // Measurement noise covariance

        auto result = kalman(model, Qn, Rn);

        CHECK(result.filter.getModel().A == model.A);
        CHECK(result.L.rows() == 2);
        CHECK(result.L.cols() == 1);
        CHECK(result.P.rows() == 2);
        CHECK(result.P.cols() == 2);

        // Kalman gain should be computed correctly with transformed covariances
        // The actual process noise covariance used is G*Qn*G^T
        // The actual measurement noise covariance used is H*Qn*H^T + Rn
        Matrix expected_Q = model.G * Qn * model.G.transpose();
        Matrix expected_R = model.H * Qn * model.H.transpose() + Rn;

        // Check that the filter was created with the transformed covariances
        CHECK(result.filter.getQ().isApprox(expected_Q, 1e-10));
        CHECK(result.filter.getR().isApprox(expected_R, 1e-10));
    }
}

TEST_CASE("Continuous Algebraic Riccati Equation (CARE)") {
    SUBCASE("Simple case") {
        Matrix A = Matrix::Constant(1, 1, -1.0);
        Matrix B = Matrix::Constant(1, 1, 1.0);
        Matrix Q = Matrix::Constant(1, 1, 1.0);
        Matrix R = Matrix::Constant(1, 1, 0.1);

        Matrix P = care(A, B, Q, R);

        CHECK(P.rows() == 1);
        CHECK(P.cols() == 1);

        // Check value against scipy.linalg.solve_continuous_are
        CHECK(P(0, 0) == doctest::Approx(0.2317).epsilon(1e-3));

        // Check that P satisfies the CARE
        Matrix lhs = A.transpose() * P + P * A - P * B * R.inverse() * B.transpose() * P + Q;
        CHECK(lhs.norm() < 1e-10);
    }
}

TEST_CASE("Discrete Algebraic Riccati Equation (DARE)") {
    SUBCASE("Simple case") {
        Matrix A = Matrix::Constant(1, 1, 0.9);
        Matrix B = Matrix::Constant(1, 1, 1.0);
        Matrix Q = Matrix::Constant(1, 1, 1.0);
        Matrix R = Matrix::Constant(1, 1, 0.1);

        Matrix P = dare(A, B, Q, R);

        CHECK(P.rows() == 1);
        CHECK(P.cols() == 1);

        // Check value against scipy.linalg.solve_discrete_are
        CHECK(P(0, 0) == doctest::Approx(1.0741).epsilon(1e-3));

        // Check that P satisfies the DARE
        Matrix lhs = A.transpose() * P * A - P - A.transpose() * P * B * (R + B.transpose() * P * B).inverse() * B.transpose() * P * A + Q;
        CHECK(lhs.norm() < 1e-10);
    }
}

TEST_CASE("Lyapunov Equation Solvers") {
    SUBCASE("Continuous Lyapunov (lyap)") {
        Matrix A = Matrix::Identity(2, 2) * -1.0;
        Matrix Q = Matrix::Identity(2, 2);

        Matrix P = lyap(A, Q);

        CHECK(P.rows() == 2);
        CHECK(P.cols() == 2);

        // Check values against scipy.linalg.solve_lyapunov
        CHECK(P(0, 0) == doctest::Approx(0.5).epsilon(1e-3));
        CHECK(P(1, 1) == doctest::Approx(0.5).epsilon(1e-3));
        CHECK(P(0, 1) == doctest::Approx(0.0).epsilon(1e-3));

        // Check that P satisfies A^T P + P A = -Q
        Matrix lhs = A.transpose() * P + P * A + Q;
        CHECK(lhs.norm() < 1e-10);
    }

    SUBCASE("Discrete Lyapunov (dlyap)") {
        Matrix A = Matrix::Identity(2, 2) * 0.9;
        Matrix Q = Matrix::Identity(2, 2);

        Matrix P = dlyap(A, Q);

        CHECK(P.rows() == 2);
        CHECK(P.cols() == 2);

        // Check values against scipy.linalg.solve_discrete_lyapunov
        CHECK(P(0, 0) == doctest::Approx(5.2632).epsilon(1e-3));
        CHECK(P(1, 1) == doctest::Approx(5.2632).epsilon(1e-3));
        CHECK(P(0, 1) == doctest::Approx(0.0).epsilon(1e-3));

        // Check that P satisfies A^T P A - P = -Q
        Matrix lhs = A.transpose() * P * A - P + Q;
        CHECK(lhs.norm() < 1e-10);
    }
}

TEST_CASE("Controllability and Observability") {
    SUBCASE("Controllability matrix (ctrb)") {
        Matrix A(2, 2);
        A << 0, 1, -1, -1;
        Matrix B(2, 1);
        B << 0, 1;

        Matrix C_mat = ctrb(A, B);
        CHECK(C_mat.rows() == 2);
        CHECK(C_mat.cols() == 2);

        // Check rank (should be full rank for controllable system)
        Eigen::FullPivLU<Matrix> lu(C_mat);
        CHECK(lu.rank() == 2);
    }

    SUBCASE("Observability matrix (obsv)") {
        Matrix C(1, 2);
        C << 1, 0;
        Matrix A(2, 2);
        A << 0, 1, -1, -1;

        Matrix O_mat = obsv(C, A);
        CHECK(O_mat.rows() == 2);
        CHECK(O_mat.cols() == 2);

        // Check rank
        Eigen::FullPivLU<Matrix> lu(O_mat);
        CHECK(lu.rank() == 2);
    }

    SUBCASE("StateSpace versions") {
        StateSpace sys{
            Matrix::Identity(2, 2),
            Matrix::Identity(2, 1),
            Matrix::Identity(1, 2),
            Matrix::Zero(1, 1)};

        Matrix C_mat = ctrb(sys);
        Matrix O_mat = obsv(sys);

        CHECK(C_mat.rows() == 2);
        CHECK(C_mat.cols() == 2);
        CHECK(O_mat.rows() == 2);
        CHECK(O_mat.cols() == 2);
    }
}

TEST_CASE("System Norms") {
    SUBCASE("H-infinity norm") {
        StateSpace sys{
            Matrix::Constant(1, 1, -1.0),
            Matrix::Constant(1, 1, 1.0),
            Matrix::Constant(1, 1, 1.0),
            Matrix::Constant(1, 1, 0.0)};

        double n_inf = norm(sys, "inf");
        CHECK(n_inf > 0);

        // For this simple system, H_inf norm should be 1
        CHECK(n_inf == doctest::Approx(1.0).epsilon(1e-3));
    }

    SUBCASE("Matrix version") {
        Matrix A = Matrix::Constant(1, 1, -1.0);
        Matrix B = Matrix::Constant(1, 1, 1.0);
        Matrix C = Matrix::Constant(1, 1, 1.0);
        Matrix D = Matrix::Constant(1, 1, 0.0);

        double n_inf = norm(A, B, C, D, "inf");
        CHECK(n_inf == doctest::Approx(1.0).epsilon(1e-3));
    }
}

TEST_CASE("Solver Event Handling") {
    SUBCASE("Fixed Step Solver with Callbacks") {
        FixedStepSolver<ForwardEuler> solver(0.1);

        // Simple system: dx/dt = -x
        auto ode = [](double /*t*/, const ColVec& x) -> ColVec {
            return -x;
        };

        ColVec                    x0     = {1.0};
        std::pair<double, double> t_span = {0.0, 1.0};

        // Track callbacks
        std::vector<double> callback_times;
        std::vector<ColVec> callback_states;

        solver.set_on_step_callback([&](double t, const ColVec& x) {
            callback_times.push_back(t);
            callback_states.push_back(x);
        });

        // Stop when x < 0.5
        solver.set_stop_condition([](double /*t*/, const ColVec& x) {
            return x(0) < 0.5;
        });

        auto result = solver.solve(ode, x0, t_span);

        // Should have stopped early
        CHECK(result.message == "Stopped by user-defined condition");
        CHECK(result.t.back() < 1.0);

        // Check callbacks were called
        CHECK(callback_times.size() > 0);
        CHECK(callback_states.size() > 0);
        CHECK(callback_states.back()(0) < 0.5);
    }

    SUBCASE("Adaptive Step Solver with Callbacks") {
        AdaptiveStepSolver<RK45> solver;

        // Same system
        auto ode = [](double /*t*/, const ColVec& x) -> ColVec {
            return -x;
        };

        ColVec                    x0     = {1.0};
        std::pair<double, double> t_span = {0.0, 1.0};

        // Track callbacks
        std::vector<double> callback_times;

        solver.set_on_step_callback([&](double t, const ColVec& /*x*/) {
            callback_times.push_back(t);
        });

        auto result = solver.solve(ode, x0, t_span);

        // Should complete normally
        CHECK(result.success);
        CHECK(result.t.back() == doctest::Approx(1.0).epsilon(1e-6));

        // Check callbacks were called
        CHECK(callback_times.size() > 0);
    }

    SUBCASE("Event Detection for Mechanical Endstops") {
        FixedStepSolver<ForwardEuler> solver(0.1);

        // Mechanical system: mass-spring-damper with endstop
        // States: [position, velocity]
        // Endstop at position = ±1.0
        auto ode = [](double /*t*/, const ColVec& x) -> ColVec {
            const double m     = 1.0;  // mass
            const double k     = 1.0;  // spring constant
            const double c     = 0.1;  // damping
            const double force = 2.0;  // constant force

            double pos = x(0);
            double vel = x(1);

            // Spring force + damping + external force
            double accel = (-k * pos - c * vel + force) / m;

            return ColVec{{vel, accel}};
        };

        ColVec                    x0     = {0.0, 0.0};  // start at rest
        std::pair<double, double> t_span = {0.0, 5.0};

        // Event detector for endstop impact
        solver.set_event_detector([](double /*t*/, const ColVec& x) -> EventResult {
            double pos = x(0);
            double vel = x(1);

            // Check for endstop contact (position limits)
            if (pos >= 1.0 && vel > 0) {
                // Hit upper endstop while moving up - bounce back
                return {true, ColVec{{1.0, -0.5 * vel}}, false, false, "Hit upper endstop"};
            } else if (pos <= -1.0 && vel < 0) {
                // Hit lower endstop while moving down - bounce back
                return {true, ColVec{{-1.0, -0.5 * vel}}, false, false, "Hit lower endstop"};
            }

            return {false, x, false, false, ""};
        });

        auto result = solver.solve(ode, x0, t_span);

        // Should complete normally (events handled, not stopped)
        CHECK(result.success);
        CHECK(result.t.back() == doctest::Approx(5.0).epsilon(1e-6));

        // Check that position stayed within bounds
        for (const auto& state : result.x) {
            CHECK(state(0) >= -1.0 - 1e-10);
            CHECK(state(0) <= 1.0 + 1e-10);
        }
    }
}

TEST_CASE("Sampling and Holding Blocks") {
    SUBCASE("Zero-Order Hold (ZOH)") {
        auto zoh_sys = zoh();

        // ZOH should be an integrator: 1/s
        CHECK(zoh_sys.A.rows() == 1);
        CHECK(zoh_sys.A.cols() == 1);
        CHECK(zoh_sys.A(0, 0) == 0.0);
        CHECK(zoh_sys.B(0, 0) == 1.0);
        CHECK(zoh_sys.C(0, 0) == 1.0);
        CHECK(zoh_sys.D(0, 0) == 0.0);

        // Check transfer function
        auto tf_zoh = zoh_sys.toTransferFunction();
        CHECK(tf_zoh.num.size() == 1);
        CHECK(tf_zoh.den.size() == 2);
        CHECK(tf_zoh.num[0] == doctest::Approx(1.0));
        CHECK(tf_zoh.den[0] == doctest::Approx(1.0));
        CHECK(tf_zoh.den[1] == doctest::Approx(0.0));
    }

    SUBCASE("First-Order Hold (FOH)") {
        auto foh_sys = foh();

        // FOH should be a 2nd order system
        CHECK(foh_sys.A.rows() == 2);
        CHECK(foh_sys.A.cols() == 2);
        CHECK(foh_sys.B.rows() == 2);
        CHECK(foh_sys.B.cols() == 1);
        CHECK(foh_sys.C.rows() == 1);
        CHECK(foh_sys.C.cols() == 2);

        // Check matrices
        CHECK(foh_sys.A(0, 0) == 0.0);
        CHECK(foh_sys.A(0, 1) == 1.0);
        CHECK(foh_sys.A(1, 0) == 0.0);
        CHECK(foh_sys.A(1, 1) == 0.0);
        CHECK(foh_sys.B(0, 0) == 0.0);
        CHECK(foh_sys.B(1, 0) == 1.0);
        CHECK(foh_sys.C(0, 0) == 1.0);
        CHECK(foh_sys.C(0, 1) == 0.0);
        CHECK(foh_sys.D(0, 0) == 0.0);
    }
}

TEST_CASE("Pole Placement (place)") {
    SUBCASE("Simple SISO system") {
        // System: dx/dt = -x + u, y = x
        Matrix A = Matrix::Constant(1, 1, -1.0);
        Matrix B = Matrix::Constant(1, 1, 1.0);

        // Place pole at -2
        std::vector<Pole> poles = {-2.0};

        Matrix K = place(A, B, poles);

        // Check dimensions
        CHECK(K.rows() == 1);
        CHECK(K.cols() == 1);

        // Check closed-loop pole
        Matrix A_cl = A - B * K;
        CHECK(A_cl(0, 0) == doctest::Approx(-2.0).epsilon(1e-3));
    }

    SUBCASE("Second order system") {
        // System: dx/dt = [0 1; -1 -1] x + [0; 1] u
        Matrix A(2, 2);
        A << 0, 1, -1, -1;
        Matrix B(2, 1);
        B << 0, 1;

        // Place poles at -1 ± j
        std::vector<Pole> poles = {std::complex<double>(-1, 1), std::complex<double>(-1, -1)};

        Matrix K = place(A, B, poles);

        // Check dimensions
        CHECK(K.rows() == 1);
        CHECK(K.cols() == 2);

        // Check closed-loop eigenvalues
        Matrix                     A_cl = A - B * K;
        Eigen::EigenSolver<Matrix> solver(A_cl);
        auto                       eigenvals = solver.eigenvalues();

        // Should have eigenvalues at -1 ± j
        bool found_neg1_plus_j  = false;
        bool found_neg1_minus_j = false;
        for (int i = 0; i < eigenvals.size(); ++i) {
            if (std::abs(eigenvals[i] - std::complex<double>(-1, 1)) < 1e-3) {
                found_neg1_plus_j = true;
            }
            if (std::abs(eigenvals[i] - std::complex<double>(-1, -1)) < 1e-3) {
                found_neg1_minus_j = true;
            }
        }
        CHECK(found_neg1_plus_j);
        CHECK(found_neg1_minus_j);
    }
}

TEST_CASE("Discrete Kalman Design (kalmd)") {
    SUBCASE("Simple continuous system") {
        // Create a continuous system
        Matrix A = Matrix::Constant(1, 1, -1.0);
        Matrix B = Matrix::Constant(1, 1, 1.0);
        Matrix C = Matrix::Constant(1, 1, 1.0);
        Matrix D = Matrix::Zero(1, 1);

        StateSpace sys(A, B, C, D);  // Continuous system

        // Process and measurement noise
        Matrix Q  = Matrix::Constant(1, 1, 0.1);   // Process noise
        Matrix R  = Matrix::Constant(1, 1, 0.01);  // Measurement noise
        double Ts = 0.1;                           // Sampling time

        auto result = kalmd(sys, Q, R, Ts);

        // Check that we get a discrete Kalman filter
        auto model = result.filter.getModel();
        CHECK(model.A.rows() == 1);
        CHECK(model.A.cols() == 1);
        CHECK(model.Ts.has_value());  // Should be discrete
        CHECK(model.Ts.value() == doctest::Approx(Ts).epsilon(1e-10));
        CHECK(result.L.rows() == 1);
        CHECK(result.L.cols() == 1);
        CHECK(result.P.rows() == 1);
        CHECK(result.P.cols() == 1);
        // For discrete systems (kalmd), Mx, Z, My should have values
        CHECK(result.Mx.has_value());
        CHECK(result.Z.has_value());
        CHECK(result.My.has_value());
        CHECK(result.Mx.value().rows() == 1);
        CHECK(result.Mx.value().cols() == 1);
        CHECK(result.Z.value().rows() == 1);
        CHECK(result.Z.value().cols() == 1);
        CHECK(result.My.value().rows() == 1);
        CHECK(result.My.value().cols() == 1);

        // Check that gain is positive
        CHECK(result.L(0, 0) > 0);
        // Innovation covariance should be positive definite
        CHECK(result.Mx.value()(0, 0) > 0);
    }
}

TEST_CASE("Regulator Design (reg)") {
    SUBCASE("Simple SISO system") {
        // Create a simple system
        Matrix A = Matrix::Constant(1, 1, -1.0);
        Matrix B = Matrix::Constant(1, 1, 1.0);
        Matrix C = Matrix::Constant(1, 1, 1.0);
        Matrix D = Matrix::Zero(1, 1);

        StateSpace sys(A, B, C, D);

        // State feedback gain
        Matrix K = Matrix::Constant(1, 1, 2.0);

        // Observer gain
        Matrix L = Matrix::Constant(1, 1, 5.0);

        StateSpace regulator = reg(sys, K, L);

        // Check dimensions
        CHECK(regulator.A.rows() == 1);
        CHECK(regulator.A.cols() == 1);
        CHECK(regulator.B.rows() == 1);
        CHECK(regulator.B.cols() == 1);
        CHECK(regulator.C.rows() == 1);
        CHECK(regulator.C.cols() == 1);
        CHECK(regulator.D.rows() == 1);
        CHECK(regulator.D.cols() == 1);

        // Check regulator dynamics: A_reg = A - L*C - B*K
        Matrix expected_A = A - L * C - B * K;
        CHECK(regulator.A(0, 0) == doctest::Approx(expected_A(0, 0)).epsilon(1e-10));

        // Check input matrix: B_reg = L
        CHECK(regulator.B(0, 0) == doctest::Approx(L(0, 0)).epsilon(1e-10));

        // Check output matrix: C_reg = -K
        CHECK(regulator.C(0, 0) == doctest::Approx(-K(0, 0)).epsilon(1e-10));

        // Check feedthrough: D_reg = 0
        CHECK(regulator.D(0, 0) == doctest::Approx(0.0).epsilon(1e-10));
    }
}

TEST_CASE("System Augmentation with Disturbance States") {
    SUBCASE("SISO system") {
        StateSpace sys{
            Matrix::Constant(1, 1, -1.0),  // A
            Matrix::Constant(1, 1, 1.0),   // B
            Matrix::Constant(1, 1, 1.0),   // C
            Matrix::Constant(1, 1, 0.0)    // D
        };

        StateSpace aug_sys = augd(sys);

        // Augmented system should have 2 states (original + disturbance)
        CHECK(aug_sys.A.rows() == 2);
        CHECK(aug_sys.A.cols() == 2);
        CHECK(aug_sys.B.rows() == 2);
        CHECK(aug_sys.B.cols() == 1);
        CHECK(aug_sys.C.rows() == 1);
        CHECK(aug_sys.C.cols() == 2);
        CHECK(aug_sys.D.rows() == 1);
        CHECK(aug_sys.D.cols() == 1);

        // Check A matrix structure: [A, 0; 0, 0]
        CHECK(aug_sys.A(0, 0) == doctest::Approx(-1.0).epsilon(1e-10));
        CHECK(aug_sys.A(0, 1) == doctest::Approx(0.0).epsilon(1e-10));
        CHECK(aug_sys.A(1, 0) == doctest::Approx(0.0).epsilon(1e-10));
        CHECK(aug_sys.A(1, 1) == doctest::Approx(0.0).epsilon(1e-10));

        // Check B matrix: [B; 0]
        CHECK(aug_sys.B(0, 0) == doctest::Approx(1.0).epsilon(1e-10));
        CHECK(aug_sys.B(1, 0) == doctest::Approx(0.0).epsilon(1e-10));

        // Check C matrix: [C, I]
        CHECK(aug_sys.C(0, 0) == doctest::Approx(1.0).epsilon(1e-10));
        CHECK(aug_sys.C(0, 1) == doctest::Approx(1.0).epsilon(1e-10));

        // Check D matrix: unchanged
        CHECK(aug_sys.D(0, 0) == doctest::Approx(0.0).epsilon(1e-10));
    }

    SUBCASE("MIMO system") {
        StateSpace sys{
            Matrix::Constant(2, 2, -1.0),  // A
            Matrix::Constant(2, 1, 1.0),   // B
            Matrix::Constant(1, 2, 1.0),   // C
            Matrix::Constant(1, 1, 0.0)    // D
        };

        StateSpace aug_sys = augd(sys);

        // Augmented system should have 3 states (2 original + 1 disturbance)
        CHECK(aug_sys.A.rows() == 3);
        CHECK(aug_sys.A.cols() == 3);
        CHECK(aug_sys.B.rows() == 3);
        CHECK(aug_sys.B.cols() == 1);
        CHECK(aug_sys.C.rows() == 1);
        CHECK(aug_sys.C.cols() == 3);
        CHECK(aug_sys.D.rows() == 1);
        CHECK(aug_sys.D.cols() == 1);

        // Check C matrix: [C, I] where I is 1x1 for single output
        CHECK(aug_sys.C(0, 0) == doctest::Approx(1.0).epsilon(1e-10));
        CHECK(aug_sys.C(0, 1) == doctest::Approx(1.0).epsilon(1e-10));
        CHECK(aug_sys.C(0, 2) == doctest::Approx(1.0).epsilon(1e-10));  // Disturbance state
    }
}