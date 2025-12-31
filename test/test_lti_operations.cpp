#include "control.hpp"
#include "doctest.h"

using namespace control;

TEST_CASE("LTI System Arithmetic Operations") {
    // Create test systems
    const StateSpace plant{
        Matrix::Constant(1, 1, -1.0),  // A
        Matrix::Constant(1, 1, 1.0),   // B
        Matrix::Constant(1, 1, 1.0),   // C
        Matrix::Constant(1, 1, 0.0)    // D
    };

    const StateSpace controller{
        Matrix::Zero(0, 0),          // A - No states (pure gain)
        Matrix::Zero(0, 1),          // B
        Matrix::Zero(1, 0),          // C
        Matrix::Constant(1, 1, 2.0)  // D
    };

    const StateSpace sensor{
        Matrix::Zero(0, 0),          // A
        Matrix::Zero(0, 1),          // B
        Matrix::Zero(1, 0),          // C
        Matrix::Constant(1, 1, 1.0)  // D
    };

    SUBCASE("Series Connection: Controller * Plant") {
        // Open-loop: L(s) = C(s) * G(s) = 2/(s+1)
        auto open_loop = controller * plant;

        // Check dimensions
        CHECK(open_loop.A.rows() == 1);
        CHECK(open_loop.A.cols() == 1);
        CHECK(open_loop.B.rows() == 1);
        CHECK(open_loop.B.cols() == 1);
        CHECK(open_loop.C.rows() == 1);
        CHECK(open_loop.C.cols() == 1);
        CHECK(open_loop.D.rows() == 1);
        CHECK(open_loop.D.cols() == 1);

        // Check values (2/(s+1) should have A=-1, B=2, C=1, D=0)
        CHECK(open_loop.A(0, 0) == doctest::Approx(-1.0));
        CHECK(open_loop.B(0, 0) == doctest::Approx(2.0));
        CHECK(open_loop.C(0, 0) == doctest::Approx(1.0));
        CHECK(open_loop.D(0, 0) == doctest::Approx(0.0));
    }

    SUBCASE("Parallel Connection: Plant + Plant") {
        // Sum: G1(s) + G2(s) should give 2/(s+1)
        auto parallel_sum = plant + plant;

        // Check dimensions (2 states from 2 systems)
        CHECK(parallel_sum.A.rows() == 2);
        CHECK(parallel_sum.A.cols() == 2);
        CHECK(parallel_sum.B.rows() == 2);
        CHECK(parallel_sum.C.cols() == 2);

        // Check A is block diagonal
        CHECK(parallel_sum.A(0, 0) == doctest::Approx(-1.0));
        CHECK(parallel_sum.A(1, 1) == doctest::Approx(-1.0));
        CHECK(parallel_sum.A(0, 1) == doctest::Approx(0.0));
        CHECK(parallel_sum.A(1, 0) == doctest::Approx(0.0));

        // Check C combines both outputs
        CHECK(parallel_sum.C(0, 0) == doctest::Approx(1.0));
        CHECK(parallel_sum.C(0, 1) == doctest::Approx(1.0));

        // D should be sum (0 + 0 = 0)
        CHECK(parallel_sum.D(0, 0) == doctest::Approx(0.0));
    }

    SUBCASE("Parallel Subtraction: Plant - Plant") {
        // Difference: G1(s) - G2(s) should give 0
        auto parallel_diff = plant - plant;

        // Check dimensions
        CHECK(parallel_diff.A.rows() == 2);
        CHECK(parallel_diff.A.cols() == 2);

        // Check A is block diagonal
        CHECK(parallel_diff.A(0, 0) == doctest::Approx(-1.0));
        CHECK(parallel_diff.A(1, 1) == doctest::Approx(-1.0));
        CHECK(parallel_diff.A(0, 1) == doctest::Approx(0.0));
        CHECK(parallel_diff.A(1, 0) == doctest::Approx(0.0));

        // Check C has negation on second system
        CHECK(parallel_diff.C(0, 0) == doctest::Approx(1.0));
        CHECK(parallel_diff.C(0, 1) == doctest::Approx(-1.0));

        // D should be difference (0 - 0 = 0)
        CHECK(parallel_diff.D(0, 0) == doctest::Approx(0.0));
    }

    SUBCASE("Negative Feedback: Unity Feedback on Open Loop") {
        // First create open loop
        auto open_loop = controller * plant;

        // Closed-loop with unity feedback: T(s) = L(s) / (1 + L(s))
        // For L(s) = 2/(s+1), T(s) = 2/(s+3)
        auto closed_loop = feedback(open_loop, sensor, -1);

        // Check dimensions (1 state from open_loop, 0 from sensor)
        CHECK(closed_loop.A.rows() == 1);
        CHECK(closed_loop.A.cols() == 1);

        // Check values for 2/(s+3): A=-3, B=2, C=1, D=0
        CHECK(closed_loop.A(0, 0) == doctest::Approx(-3.0));
        CHECK(closed_loop.B(0, 0) == doctest::Approx(2.0));
        CHECK(closed_loop.C(0, 0) == doctest::Approx(1.0));
        CHECK(closed_loop.D(0, 0) == doctest::Approx(0.0));
    }

    SUBCASE("Complete Control System: Feedback(Controller * Plant, Sensor)") {
        // T(s) = C(s)*G(s) / (1 + C(s)*G(s)*H(s))
        auto control_system = feedback(controller * plant, sensor, -1);

        // Check dimensions
        CHECK(control_system.A.rows() == 1);
        CHECK(control_system.A.cols() == 1);

        // Should be same as previous test (2/(s+3))
        CHECK(control_system.A(0, 0) == doctest::Approx(-3.0));
        CHECK(control_system.B(0, 0) == doctest::Approx(2.0));
        CHECK(control_system.C(0, 0) == doctest::Approx(1.0));
        CHECK(control_system.D(0, 0) == doctest::Approx(0.0));

        // Test step response properties
        auto step_resp = control_system.step(0.0, 5.0);
        CHECK(step_resp.time.size() > 0);
        CHECK(step_resp.output.size() == step_resp.time.size());

        // Check steady-state is approximately 2/3
        CHECK(step_resp.output.back()(0, 0) == doctest::Approx(2.0 / 3.0).epsilon(0.01));
    }

    SUBCASE("Feedback: StateSpace plant with TF PID") {
        // Plant: 1/(s^2 + 2s + 1)
        TransferFunction plant_tf{{1.0}, {1.0, 2.0, 1.0}};
        StateSpace       G = plant_tf.toStateSpace();

        // PID controller: (Kd*s^2 + Kp*s + Ki) / s
        double           Kd = 0.1;
        double           Kp = 1.0;
        double           Ki = 1.0;
        TransferFunction pid({Kd, Kp, Ki}, {1.0, 0.0});

        StateSpace closed = feedback(G, pid, -1);
        CHECK(closed.A.rows() >= 1);
        CHECK(G / pid == closed);
    }
}

TEST_CASE("LTI System Type Safety") {
    const StateSpace continuous{
        Matrix::Constant(1, 1, -1.0),
        Matrix::Constant(1, 1, 1.0),
        Matrix::Constant(1, 1, 1.0),
        Matrix::Constant(1, 1, 0.0)};

    const StateSpace discrete{
        Matrix::Constant(1, 1, 0.9),
        Matrix::Constant(1, 1, 0.1),
        Matrix::Constant(1, 1, 1.0),
        Matrix::Constant(1, 1, 0.0),
        0.1  // Ts
    };

    SUBCASE("Same type operations compile") {
        // These should compile fine
        auto result1 = continuous * continuous;
        auto result2 = continuous + continuous;
        auto result3 = continuous - continuous;
        auto result4 = feedback(continuous, continuous);

        CHECK(result1.A.rows() > 0);  // Just verify it compiled
        CHECK(result2.A.rows() > 0);
        CHECK(result3.A.rows() > 0);
        CHECK(result4.A.rows() > 0);
    }

    SUBCASE("Discrete type operations compile") {
        // These should compile fine
        auto result1 = discrete * discrete;
        auto result2 = discrete + discrete;
        auto result3 = discrete - discrete;
        auto result4 = feedback(discrete, discrete);

        CHECK(result1.A.rows() > 0);  // Just verify it compiled
        CHECK(result2.A.rows() > 0);
        CHECK(result3.A.rows() > 0);
        CHECK(result4.A.rows() > 0);
    }

    // Note: Mixed type operations (continuous * discrete) should fail at compile time
    // due to static_assert, so we can't test them here
}
TEST_CASE("Integrators with LTI Systems") {
    // Simple decaying system: x' = -x, u = 1 -> solution is e^(-t)
    const Matrix A  = Matrix::Constant(1, 1, -1.0);
    const Matrix B  = Matrix::Constant(1, 1, 1.0);
    const Matrix x0 = Matrix::Zero(1, 1);
    const Matrix u  = Matrix::Constant(1, 1, 1.0);
    const double h  = 0.01;

    SUBCASE("ForwardEuler integrator") {
        ForwardEuler integrator;
        auto         result = integrator.evolve(A, B, x0, u, h);
        CHECK(result.x(0, 0) == doctest::Approx(0.01));  // x + h*(A*x + B*u) = 0 + 0.01*1
    }

    SUBCASE("BackwardEuler integrator") {
        BackwardEuler integrator;
        auto          result = integrator.evolve(A, B, x0, u, h);
        // (I - h*A)^-1 * (x + h*B*u) = (1 + 0.01)^-1 * 0.01 ≈ 0.0099
        CHECK(result.x(0, 0) == doctest::Approx(0.0099).epsilon(0.001));
    }

    SUBCASE("Trapezoidal integrator") {
        Trapezoidal integrator;
        auto        result = integrator.evolve(A, B, x0, u, h);
        // (I - h/2*A)^-1 * ((I + h/2*A)*x + h/2*B*u) ≈ 0.0099
        CHECK(result.x(0, 0) == doctest::Approx(0.0099).epsilon(0.001));
    }

    SUBCASE("RK45 integrator") {
        RK45 integrator;
        auto result = integrator.evolve(A, B, x0, u, h);
        // RK45 with smaller h should give more accurate result
        CHECK(result.x(0, 0) == doctest::Approx(0.00995).epsilon(0.001));
    }

    SUBCASE("Discrete integrator for LTI") {
        Discrete integrator;
        auto     result = integrator.evolve(A, B, x0, u);
        // Simply: A*x + B*u = -1*0 + 1*1 = 1
        CHECK(result.x(0, 0) == doctest::Approx(1.0));
    }
}

TEST_CASE("Integrators with Nonlinear ODEs") {
    // Simple test ODE: x' = -2*x, solution is e^(-2*t)
    auto f = [](double /*t*/, const Matrix& x) -> Matrix {
        return -2.0 * x;
    };

    const Matrix x0 = Matrix::Constant(1, 1, 1.0);
    const double t  = 0.0;
    const double h  = 0.1;

    SUBCASE("ForwardEuler on nonlinear ODE") {
        ForwardEuler integrator;
        auto         result = integrator.evolve(f, x0, t, h);
        // x' = -2*x -> x_new = x + h*f(t,x) = 1 + 0.1*(-2) = 0.8
        CHECK(result.x(0, 0) == doctest::Approx(0.8));
    }

    SUBCASE("RK45 on nonlinear ODE") {
        RK45 integrator;
        auto result = integrator.evolve(f, x0, t, h);
        // RK45 should be more accurate than ForwardEuler
        // Exact: e^(-2*h) ≈ 0.8187
        CHECK(result.x(0, 0) == doctest::Approx(std::exp(-2.0 * h)).epsilon(0.01));
    }

    SUBCASE("BackwardEuler on nonlinear ODE") {
        BackwardEuler integrator;
        auto          result = integrator.evolve(f, x0, t, h);
        // y = x + h*f(t+h, y) = 1 + 0.1*f(0.1, y)
        // Since f(t, x) = -2*x: y = 1 - 0.2*y -> y*(1 + 0.2) = 1 -> y ≈ 0.833
        CHECK(result.x(0, 0) == doctest::Approx(1.0 / 1.2).epsilon(0.001));
    }
}

TEST_CASE("solve() with generic nonlinear ODE") {
    // Van der Pol oscillator: x1' = x2, x2' = (1-x1^2)*x2 - x1
    auto fun = [](double /*t*/, const Matrix& x) -> Matrix {
        const double mu = 1.0;
        Matrix       dx = Matrix::Zero(2, 1);
        dx(0, 0)        = x(1, 0);
        dx(1, 0)        = mu * (1.0 - x(0, 0) * x(0, 0)) * x(1, 0) - x(0, 0);
        return dx;
    };

    const Matrix                    x0          = Matrix::Zero(2, 1);
    const auto                      x0_modified = [&x0]() { auto tmp = x0; tmp(0, 0) = 2.0; return tmp; }();
    const std::pair<double, double> t_span{0.0, 1.0};
    const std::vector<double>       t_eval = {0.0, 0.25, 0.5, 0.75, 1.0};

    SUBCASE("solve with RK45") {
        FixedStepSolver<RK45> solver(0.01);

        auto result = solver.solve(fun, x0_modified, t_span, t_eval);
        CHECK(result.success);
        CHECK(result.t.size() >= 5);
        CHECK(result.x.size() == result.t.size());
        CHECK(result.t.back() <= t_span.second);
    }

    SUBCASE("solve with ForwardEuler") {
        FixedStepSolver<ForwardEuler> solver(0.01);

        auto result = solver.solve(fun, x0_modified, t_span, t_eval);
        CHECK(result.success);
        CHECK(result.t.size() >= 5);
        CHECK(result.x.size() == result.t.size());
    }

    SUBCASE("solve with BackwardEuler") {
        FixedStepSolver<BackwardEuler> solver(0.01);

        auto result = solver.solve(fun, x0_modified, t_span, t_eval);
        CHECK(result.success);
        CHECK(result.t.size() >= 5);
    }
}

TEST_CASE("solve() with LTI constant input") {
    // Simple system: x' = -x, u = 1
    const Matrix                    A       = Matrix::Constant(1, 1, -1.0);
    const Matrix                    B       = Matrix::Constant(1, 1, 1.0);
    const ColVec                    x0      = ColVec::Zero(1);
    const ColVec                    u_const = ColVec::Constant(1, 1.0);
    const std::pair<double, double> t_span{0.0, 1.0};
    const std::vector<double>       t_eval = {0.0, 0.5, 1.0};

    SUBCASE("solve LTI with constant input (using Exact)") {
        ExactSolver solver;

        auto result = solver.solve(A, B, x0, u_const, t_span, t_eval);
        CHECK(result.success);
        CHECK(result.t.size() == 3);
        CHECK(result.x.size() == 3);
        // At t=0: x(0) = 0
        CHECK(result.x[0](0, 0) == doctest::Approx(0.0));
        // At t=1: x(1) = 1 - e^(-1) ≈ 0.632
        CHECK(result.x.back()(0, 0) == doctest::Approx(1.0 - std::exp(-1.0)).epsilon(0.001));
    }

    SUBCASE("solve LTI with Exact integrator") {
        ExactSolver solver;

        auto result = solver.solve(A, B, x0, u_const, t_span, t_eval);
        CHECK(result.success);
        CHECK(result.t.size() == 3);
    }
}

TEST_CASE("solve() with LTI time-varying input") {
    // Simple system: x' = -x + u(t), where u(t) = sin(t)
    const Matrix A  = Matrix::Constant(1, 1, -1.0);
    const Matrix B  = Matrix::Constant(1, 1, 1.0);
    const ColVec x0 = ColVec::Zero(1);

    auto u_func = [](double t) -> ColVec {
        return ColVec::Constant(1, std::sin(t));
    };

    const std::pair<double, double> t_span{0.0, 1.0};
    const std::vector<double>       t_eval = {0.0, 0.5, 1.0};

    SUBCASE("solve LTI with time-varying input and RK45") {
        FixedStepSolver<RK45> solver(0.01);

        auto ode_func = [&A, &B, &u_func](double t, const ColVec& x) -> ColVec {
            return A * x + B * u_func(t);
        };
        auto result = solver.solve(ode_func, x0, t_span, t_eval);
        CHECK(result.success);
        CHECK(result.t.size() >= 3);
        CHECK(result.x.size() == result.t.size());
    }

    SUBCASE("solve LTI with time-varying input and BackwardEuler") {
        FixedStepSolver<BackwardEuler> solver(0.01);

        auto ode_func = [&A, &B, &u_func](double t, const ColVec& x) -> ColVec {
            return A * x + B * u_func(t);
        };
        auto result = solver.solve(ode_func, x0, t_span, t_eval);
        CHECK(result.success);
        CHECK(result.t.size() >= 3);
    }

    SUBCASE("solve LTI with lambda for time-varying input") {
        FixedStepSolver<RK45> solver(0.01);

        auto u_lambda = [](double t) { return Matrix::Constant(1, 1, std::sin(t)); };
        auto ode_func = [&A, &B, &u_lambda](double t, const Matrix& x) -> Matrix {
            return A * x + B * u_lambda(t);
        };
        auto result = solver.solve(ode_func, x0, t_span, t_eval);
        CHECK(result.success);
        CHECK(result.t.size() >= 3);
    }
}

TEST_CASE("Solver API overload disambiguation") {
    // Test that all overloads can be called without ambiguity
    const Matrix A       = Matrix::Constant(1, 1, -1.0);
    const Matrix B       = Matrix::Constant(1, 1, 1.0);
    const Matrix x0      = Matrix::Zero(1, 1);
    const Matrix u_const = Matrix::Constant(1, 1, 1.0);

    const std::pair<double, double> t_span{0.0, 0.1};
    const std::vector<double>       t_eval;

    SUBCASE("LTI with const Matrix input") {
        // Matrix input - use ExactSolver with A, B matrices
        ExactSolver solver;

        auto result = solver.solve(A, B, x0, u_const, t_span, t_eval);
        CHECK(result.success);
    }

    SUBCASE("LTI with function input") {
        // std::function input
        auto u_func = [](double /*t*/) -> ColVec {
            return ColVec::Constant(1, 1.0);
        };
        FixedStepSolver<RK45> solver(0.01);

        auto ode_func = [&A, &B, &u_func](double t, const Matrix& x) -> Matrix {
            return A * x + B * u_func(t);
        };
        auto result = solver.solve(ode_func, x0, t_span, t_eval);
        CHECK(result.success);
    }

    SUBCASE("Generic ODE with explicit integrator and h") {
        // Generic nonlinear ODE solver with required parameters
        FixedStepSolver<RK45> solver(0.01);

        auto result = solver.solve(
            [](double /*t*/, const Matrix& x) -> Matrix { return -x; },
            x0, t_span, t_eval);
        CHECK(result.success);
    }
}

TEST_CASE("StateSpace to TransferFunction Conversion") {
    SUBCASE("SISO system conversion") {
        // Create a simple SISO system: G(s) = 1/(s+1)
        StateSpace sys{
            Matrix::Constant(1, 1, -1.0),  // A = -1
            Matrix::Constant(1, 1, 1.0),   // B = 1
            Matrix::Constant(1, 1, 1.0),   // C = 1
            Matrix::Constant(1, 1, 0.0)    // D = 0
        };

        // Convert to transfer function
        auto tf_sys = tf(sys);

        // Should get num=[1], den=[1, 1] representing 1/(s+1)
        CHECK(tf_sys.num.size() == 1);
        CHECK(tf_sys.den.size() == 2);
        CHECK(tf_sys.num[0] == doctest::Approx(1.0));
        CHECK(tf_sys.den[0] == doctest::Approx(1.0));
        CHECK(tf_sys.den[1] == doctest::Approx(1.0));
    }

    SUBCASE("SISO system with D term") {
        // Create a system with feedthrough: G(s) = (s+2)/(s+1)
        StateSpace sys{
            Matrix::Constant(1, 1, -1.0),  // A = -1
            Matrix::Constant(1, 1, 1.0),   // B = 1
            Matrix::Constant(1, 1, 1.0),   // C = 1
            Matrix::Constant(1, 1, 1.0)    // D = 1
        };

        // Convert to transfer function
        auto tf_sys = tf(sys);

        // Check dimensions
        CHECK(tf_sys.num.size() >= 1);
        CHECK(tf_sys.den.size() >= 1);

        // Verify it's normalized
        CHECK(tf_sys.den[0] == doctest::Approx(1.0));
    }

    SUBCASE("Extract SISO from MIMO using indices") {
        // Create a 2x2 MIMO system
        StateSpace mimo_sys{
            (Matrix(2, 2) << -1.0, 0.0, 0.0, -2.0).finished(),  // A (diagonal)
            (Matrix(2, 2) << 1.0, 0.0, 0.0, 1.0).finished(),    // B
            (Matrix(2, 2) << 1.0, 0.0, 0.0, 2.0).finished(),    // C
            Matrix::Zero(2, 2)                                  // D
        };

        // Extract G_00: output 0, input 0
        auto tf_00 = tf(mimo_sys, 0, 0);
        CHECK(tf_00.num.size() >= 1);
        CHECK(tf_00.den.size() >= 1);

        // Extract G_11: output 1, input 1
        auto tf_11 = tf(mimo_sys, 1, 1);
        CHECK(tf_11.num.size() >= 1);
        CHECK(tf_11.den.size() >= 1);

        // The system has characteristic polynomial (s+1)(s+2) = s^2 + 3s + 2
        // For G_00, the unobservable/uncontrollable mode at s=-2 should cancel:
        // G_00 = 1/(s+1) ideally, but from 2nd order state-space we get (s+2)/(s^2+3s+2)
        // which simplifies to 1/(s+1) after cancellation
        // Check that denominator has both poles before cancellation
        CHECK(tf_00.den[0] == doctest::Approx(1.0));  // s^2 coefficient
        CHECK(tf_00.den.size() >= 2);

        // The system should have both eigenvalues in denominator: (s+1)(s+2)
        // After pole-zero cancellation, G_00 should reduce to 1/(s+1)
        // but our algorithm doesn't do cancellation, so we get the full form
    }

    SUBCASE("Invalid indices throw out_of_range") {
        StateSpace sys{
            Matrix::Constant(1, 1, -1.0),  // A
            Matrix::Constant(1, 1, 1.0),   // B
            Matrix::Constant(1, 1, 1.0),   // C
            Matrix::Constant(1, 1, 0.0)    // D
        };

        // Output index out of range
        CHECK_THROWS_AS(tf(sys, 1, 0), std::out_of_range);

        // Input index out of range
        CHECK_THROWS_AS(tf(sys, 0, 1), std::out_of_range);

        // Both out of range
        CHECK_THROWS_AS(tf(sys, 1, 1), std::out_of_range);

        // Negative indices
        CHECK_THROWS_AS(tf(sys, -1, 0), std::out_of_range);
        CHECK_THROWS_AS(tf(sys, 0, -1), std::out_of_range);
    }

    SUBCASE("Round-trip: TF -> SS -> TF") {
        // Create a transfer function
        TransferFunction original({1.0}, {1.0, 1.0});  // 1/(s+1)

        // Convert to state-space
        StateSpace ss_sys = ss(original);

        // Convert back to transfer function
        TransferFunction recovered = tf(ss_sys);

        // Should match the original (within numerical precision)
        CHECK(recovered.num.size() == original.num.size());
        CHECK(recovered.den.size() == original.den.size());

        for (size_t i = 0; i < recovered.num.size(); ++i) {
            CHECK(recovered.num[i] == doctest::Approx(original.num[i]).epsilon(1e-6));
        }
        for (size_t i = 0; i < recovered.den.size(); ++i) {
            CHECK(recovered.den[i] == doctest::Approx(original.den[i]).epsilon(1e-6));
        }
    }
}

TEST_CASE("TransferFunction Poles and Zeros") {
    SUBCASE("First-order system: 1/(s+2)") {
        // G(s) = 1/(s+2)
        // Pole at s = -2, no zeros
        TransferFunction sys({1.0}, {1.0, 2.0});

        auto poles_vec = sys.poles();
        auto zeros_vec = sys.zeros();

        // Should have 1 pole
        CHECK(poles_vec.size() == 1);
        CHECK(poles_vec[0].real() == doctest::Approx(-2.0).epsilon(1e-6));
        CHECK(std::abs(poles_vec[0].imag()) < 1e-10);

        // Should have no zeros
        CHECK(zeros_vec.size() == 0);
    }

    SUBCASE("First-order with zero: (s+1)/(s+2)") {
        // G(s) = (s+1)/(s+2)
        // Pole at s = -2, zero at s = -1
        TransferFunction sys({1.0, 1.0}, {1.0, 2.0});

        auto poles_vec = sys.poles();
        auto zeros_vec = sys.zeros();

        // Should have 1 pole at -2
        CHECK(poles_vec.size() == 1);
        CHECK(poles_vec[0].real() == doctest::Approx(-2.0).epsilon(1e-6));

        // Should have 1 zero at -1
        CHECK(zeros_vec.size() == 1);
        CHECK(zeros_vec[0].real() == doctest::Approx(-1.0).epsilon(1e-6));
    }

    SUBCASE("Second-order: 1/(s^2 + 3s + 2)") {
        // G(s) = 1/(s^2 + 3s + 2) = 1/((s+1)(s+2))
        // Poles at s = -1 and s = -2, no zeros
        TransferFunction sys({1.0}, {1.0, 3.0, 2.0});

        auto poles_vec = sys.poles();
        auto zeros_vec = sys.zeros();

        // Should have 2 poles
        CHECK(poles_vec.size() == 2);

        // Check poles are -1 and -2 (order may vary)
        std::vector<double> pole_reals = {poles_vec[0].real(), poles_vec[1].real()};
        std::sort(pole_reals.begin(), pole_reals.end());
        CHECK(pole_reals[0] == doctest::Approx(-2.0).epsilon(1e-6));
        CHECK(pole_reals[1] == doctest::Approx(-1.0).epsilon(1e-6));

        // Should have no zeros
        CHECK(zeros_vec.size() == 0);
    }

    SUBCASE("Complex conjugate poles: 1/(s^2 + 2s + 5)") {
        // G(s) = 1/(s^2 + 2s + 5)
        // Poles at s = -1 ± 2j
        TransferFunction sys({1.0}, {1.0, 2.0, 5.0});

        auto poles_vec = sys.poles();

        // Should have 2 poles
        CHECK(poles_vec.size() == 2);

        // Both poles should have real part -1
        CHECK(poles_vec[0].real() == doctest::Approx(-1.0).epsilon(1e-6));
        CHECK(poles_vec[1].real() == doctest::Approx(-1.0).epsilon(1e-6));

        // Imaginary parts should be ±2
        double imag_sum = std::abs(poles_vec[0].imag()) + std::abs(poles_vec[1].imag());
        CHECK(imag_sum == doctest::Approx(4.0).epsilon(1e-6));
    }

    SUBCASE("Complex zeros: (s^2 + 2s + 5)/(s+1)") {
        // G(s) = (s^2 + 2s + 5)/(s+1)
        // Pole at s = -1, zeros at s = -1 ± 2j
        TransferFunction sys({1.0, 2.0, 5.0}, {1.0, 1.0});

        auto zeros_vec = sys.zeros();
        auto poles_vec = sys.poles();

        // Should have 2 zeros
        CHECK(zeros_vec.size() == 2);
        CHECK(zeros_vec[0].real() == doctest::Approx(-1.0).epsilon(1e-6));
        CHECK(zeros_vec[1].real() == doctest::Approx(-1.0).epsilon(1e-6));

        // Should have 1 pole
        CHECK(poles_vec.size() == 1);
        CHECK(poles_vec[0].real() == doctest::Approx(-1.0).epsilon(1e-6));
    }

    SUBCASE("Pure gain (no dynamics)") {
        // G(s) = 5
        // No poles, no zeros
        TransferFunction sys({5.0}, {1.0});

        auto poles_vec = sys.poles();
        auto zeros_vec = sys.zeros();

        // Should have no poles or zeros
        CHECK(poles_vec.size() == 0);
        CHECK(zeros_vec.size() == 0);
    }

    SUBCASE("Invalid construction throws") {
        // Zero leading coefficient in denominator
        CHECK_THROWS_AS(TransferFunction({1.0}, {0.0, 1.0}), std::invalid_argument);

        // Empty denominator
        CHECK_THROWS_AS(TransferFunction({1.0}, {}), std::invalid_argument);

        // Empty numerator
        CHECK_THROWS_AS(TransferFunction({}, {1.0, 1.0}), std::invalid_argument);

        // Leading zero in numerator
        CHECK_THROWS_AS(TransferFunction({0.0, 1.0}, {1.0, 1.0}), std::invalid_argument);
    }
}

TEST_CASE("StateSpace Poles and Zeros") {
    SUBCASE("Simple SISO system poles") {
        // System with A = -2 has pole at s = -2
        StateSpace sys{
            Matrix::Constant(1, 1, -2.0),  // A
            Matrix::Constant(1, 1, 1.0),   // B
            Matrix::Constant(1, 1, 1.0),   // C
            Matrix::Constant(1, 1, 0.0)    // D
        };

        auto poles_vec = sys.poles();

        CHECK(poles_vec.size() == 1);
        CHECK(poles_vec[0].real() == doctest::Approx(-2.0).epsilon(1e-6));
    }

    SUBCASE("Second-order system poles") {
        // System with diagonal A = [-1, 0; 0, -2]
        StateSpace sys{
            (Matrix(2, 2) << -1.0, 0.0, 0.0, -2.0).finished(),  // A
            Matrix::Constant(2, 1, 1.0),                        // B
            Matrix::Constant(1, 2, 1.0),                        // C
            Matrix::Zero(1, 1)                                  // D
        };

        auto poles_vec = sys.poles();

        CHECK(poles_vec.size() == 2);

        // Sort poles by real part
        std::vector<double> pole_reals = {poles_vec[0].real(), poles_vec[1].real()};
        std::sort(pole_reals.begin(), pole_reals.end());

        CHECK(pole_reals[0] == doctest::Approx(-2.0).epsilon(1e-6));
        CHECK(pole_reals[1] == doctest::Approx(-1.0).epsilon(1e-6));
    }

    SUBCASE("SISO system zeros (converts to TF internally)") {
        // System G(s) = C(sI-A)^(-1)B + D
        // With A=-2, B=1, C=-1, D=1: G(s) = -1/(s+2) + 1 = (s+1)/(s+2)
        // This has zero at s = -1, pole at s = -2
        StateSpace sys{
            Matrix::Constant(1, 1, -2.0),  // A = -2
            Matrix::Constant(1, 1, 1.0),   // B = 1
            Matrix::Constant(1, 1, -1.0),  // C = -1 (creates zero at -1)
            Matrix::Constant(1, 1, 1.0)    // D = 1
        };

        auto zeros_vec = sys.zeros();
        auto poles_vec = sys.poles();

        // Should have 1 zero at -1
        CHECK(zeros_vec.size() == 1);
        CHECK(zeros_vec[0].real() == doctest::Approx(-1.0).epsilon(1e-6));

        // Should have 1 pole at -2
        CHECK(poles_vec.size() == 1);
        CHECK(poles_vec[0].real() == doctest::Approx(-2.0).epsilon(1e-6));
    }

    SUBCASE("MIMO system throws on zeros()") {
        // 2x2 MIMO system
        StateSpace mimo_sys{
            Matrix::Identity(2, 2),  // A
            Matrix::Identity(2, 2),  // B (2 inputs)
            Matrix::Identity(2, 2),  // C (2 outputs)
            Matrix::Zero(2, 2)       // D
        };

        // zeros() only works for SISO
        CHECK_THROWS_AS(mimo_sys.zeros(), std::invalid_argument);
    }

    SUBCASE("Oscillator poles (complex conjugate)") {
        // Simple harmonic oscillator: x'' + ω²x = 0
        // A = [0, 1; -ω², 0], poles at ±jω
        const double omega = 2.0;
        StateSpace   sys{
            (Matrix(2, 2) << 0.0, 1.0, -omega * omega, 0.0).finished(),
            Matrix::Constant(2, 1, 0.0),
            Matrix::Constant(1, 2, 1.0),
            Matrix::Zero(1, 1)};

        auto poles_vec = sys.poles();

        CHECK(poles_vec.size() == 2);

        // Poles should be purely imaginary (±2j)
        CHECK(std::abs(poles_vec[0].real()) < 1e-10);
        CHECK(std::abs(poles_vec[1].real()) < 1e-10);

        // Magnitude of imaginary parts should sum to 2*omega
        double imag_sum = std::abs(poles_vec[0].imag()) + std::abs(poles_vec[1].imag());
        CHECK(imag_sum == doctest::Approx(2.0 * omega).epsilon(1e-6));
    }

    SUBCASE("Stability check using poles") {
        // Stable system: all poles have negative real parts
        StateSpace stable_sys{
            Matrix::Constant(1, 1, -1.0),
            Matrix::Constant(1, 1, 1.0),
            Matrix::Constant(1, 1, 1.0),
            Matrix::Zero(1, 1)};

        CHECK(stable_sys.is_stable());

        auto poles = stable_sys.poles();
        for (const auto& pole : poles) {
            CHECK(pole.real() < 0.0);
        }
    }

    SUBCASE("Unstable system poles") {
        // Unstable system: at least one pole with non-negative real part
        StateSpace unstable_sys{
            Matrix::Constant(1, 1, 1.0),  // Positive eigenvalue
            Matrix::Constant(1, 1, 1.0),
            Matrix::Constant(1, 1, 1.0),
            Matrix::Zero(1, 1)};

        CHECK_FALSE(unstable_sys.is_stable());

        auto poles = unstable_sys.poles();
        CHECK(poles[0].real() > 0.0);
    }
}

TEST_CASE("Direct Arithmetic Function Calls") {
    SUBCASE("series function") {
        TransferFunction sys1({1.0}, {1.0, 1.0});  // 1/(s+1)
        TransferFunction sys2({1.0}, {1.0, 2.0});  // 1/(s+2)

        StateSpace result = series(sys1, sys2);

        CHECK(result.A.rows() == 2);
        CHECK(result.A.cols() == 2);
        CHECK(result.B.rows() == 2);
        CHECK(result.B.cols() == 1);
        CHECK(result.C.rows() == 1);
        CHECK(result.C.cols() == 2);
        CHECK(result.D.rows() == 1);
        CHECK(result.D.cols() == 1);
    }

    SUBCASE("parallel function") {
        TransferFunction sys1({1.0}, {1.0, 1.0});  // 1/(s+1)
        TransferFunction sys2({1.0}, {1.0, 2.0});  // 1/(s+2)

        StateSpace result = parallel(sys1, sys2);

        CHECK(result.A.rows() == 2);
        CHECK(result.A.cols() == 2);
        CHECK(result.B.rows() == 2);
        CHECK(result.B.cols() == 1);
        CHECK(result.C.rows() == 1);
        CHECK(result.C.cols() == 2);
        CHECK(result.D.rows() == 1);
        CHECK(result.D.cols() == 1);
    }

    SUBCASE("feedback function - negative feedback") {
        TransferFunction forward({1.0}, {1.0, 1.0});   // 1/(s+1)
        TransferFunction feedback({1.0}, {1.0, 0.1});  // 1/(s+0.1)

        StateSpace result = control::feedback(forward, feedback, -1);

        CHECK(result.A.rows() == 2);
        CHECK(result.A.cols() == 2);
        CHECK(result.B.rows() == 2);
        CHECK(result.B.cols() == 1);
        CHECK(result.C.rows() == 1);
        CHECK(result.C.cols() == 2);
        CHECK(result.D.rows() == 1);
        CHECK(result.D.cols() == 1);
    }

    SUBCASE("feedback function - positive feedback") {
        TransferFunction forward({1.0}, {1.0, 1.0});   // 1/(s+1)
        TransferFunction feedback({1.0}, {1.0, 0.1});  // 1/(s+0.1)

        StateSpace result = control::feedback(forward, feedback, 1);

        CHECK(result.A.rows() == 2);
        CHECK(result.A.cols() == 2);
        CHECK(result.B.rows() == 2);
        CHECK(result.B.cols() == 1);
        CHECK(result.C.rows() == 1);
        CHECK(result.C.cols() == 2);
        CHECK(result.D.rows() == 1);
        CHECK(result.D.cols() == 1);
    }
}

TEST_CASE("Mixed-type arithmetic operations") {
    SUBCASE("Series: TransferFunction * StateSpace") {
        TransferFunction G({1.0}, {1.0, 1.0});  // 1/(s+1)
        StateSpace       ssys = ss(G);

        // Series: TF * SS (should return StateSpace via template)
        StateSpace result = G * ssys;
        CHECK(result.B.cols() == 1);
        CHECK(result.C.rows() == 1);
    }

    SUBCASE("Parallel: StateSpace + TransferFunction") {
        StateSpace       sA = ss(TransferFunction({1.0}, {1.0, 2.0}));
        TransferFunction H({1.0}, {1.0, 0.5});
        StateSpace       result = sA + H;
        CHECK(result.A.rows() >= 1);
    }

    SUBCASE("Feedback: TransferFunction / ZeroPoleGain") {
        TransferFunction G({1.0}, {1.0, 1.0});  // 1/(s+1)
        ZeroPoleGain     zpk_sys = zpk(G);

        StateSpace closed = G / zpk_sys;
        CHECK(closed.A.rows() >= 1);
    }

    SUBCASE("Mixed operator overloads maintain Ts compatibility") {
        TransferFunction td({1.0}, {1.0, -0.5}, 0.1);
        StateSpace       sd = ss(td);

        // different Ts should throw
        TransferFunction other({1.0}, {1.0, -0.5}, std::nullopt);
        CHECK_THROWS_AS(sd + other, std::runtime_error);
    }
}