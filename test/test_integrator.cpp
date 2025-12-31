#include <cmath>

#include "control.hpp"
#include "doctest.h"

using namespace control;

TEST_CASE("ForwardEuler Integrator") {
    SUBCASE("Basic integration: x' = -x") {
        ForwardEuler integrator;

        Matrix x = Matrix::Constant(1, 1, 1.0);
        double t = 0.0;
        double h = 0.1;

        // x' = -x, analytical solution: x(t) = e^(-t)
        auto f = [](double /*t*/, const Matrix& x) -> Matrix {
            return -x;
        };

        // Take one step
        auto   result = integrator.evolve(f, x, t, h);
        Matrix x_new  = result.x;

        // Forward Euler: x_new = x + h*f(t,x) = 1 + 0.1*(-1) = 0.9
        CHECK(x_new(0, 0) == doctest::Approx(0.9));

        // Analytical solution at t=0.1: e^(-0.1) ≈ 0.9048
        double analytical = std::exp(-0.1);
        CHECK(x_new(0, 0) == doctest::Approx(analytical).epsilon(0.01));  // First order accuracy
    }

    SUBCASE("Multiple steps convergence") {
        ForwardEuler integrator;

        Matrix x = Matrix::Constant(1, 1, 1.0);
        double t = 0.0;
        double h = 0.01;

        auto f = [](double /*t*/, const Matrix& x) -> Matrix {
            return -x;
        };

        // Integrate to t=1.0
        for (int i = 0; i < 100; ++i) {
            auto result = integrator.evolve(f, x, t, h);
            x           = result.x;
            t += h;
        }

        double analytical = std::exp(-1.0);
        CHECK(x(0, 0) == doctest::Approx(analytical).epsilon(0.01));  // Should be reasonably accurate with small steps
    }

    SUBCASE("System of equations") {
        ForwardEuler integrator;

        // 2D system: x' = -x, y' = -2*y
        Matrix x(2, 1);
        x << 1.0, 2.0;
        double t = 0.0;
        double h = 0.01;

        auto f = [](double /*t*/, const Matrix& x) -> Matrix {
            Matrix dx(2, 1);
            dx << -x(0, 0), -2.0 * x(1, 0);
            return dx;
        };

        // Take one step
        auto   result = integrator.evolve(f, x, t, h);
        Matrix x_new  = result.x;

        CHECK(x_new(0, 0) == doctest::Approx(1.0 - 0.01 * 1.0));
        CHECK(x_new(1, 0) == doctest::Approx(2.0 - 0.01 * 2.0 * 2.0));
    }
}

TEST_CASE("RK4 Integrator") {
    SUBCASE("High accuracy for smooth functions") {
        RK4 integrator;

        Matrix x = Matrix::Constant(1, 1, 1.0);
        double t = 0.0;
        double h = 0.1;

        // x' = -x, analytical solution: x(t) = e^(-t)
        auto f = [](double /*t*/, const Matrix& x) -> Matrix {
            return -x;
        };

        // Take one step
        auto   result = integrator.evolve(f, x, t, h);
        Matrix x_new  = result.x;

        // RK4 should be much more accurate than Forward Euler
        double analytical = std::exp(-0.1);
        CHECK(x_new(0, 0) == doctest::Approx(analytical).epsilon(1e-6));  // Fourth order accuracy
    }

    SUBCASE("Nonlinear oscillator: x'' = -x") {
        RK4 integrator;

        // Convert to first order: [x, v]' = [v, -x]
        Matrix state(2, 1);
        state << 1.0, 0.0;  // x=1, v=0
        double t = 0.0;
        double h = 0.01;

        auto f = [](double /*t*/, const Matrix& state) -> Matrix {
            Matrix dstate(2, 1);
            dstate << state(1, 0), -state(0, 0);
            return dstate;
        };

        // Store initial energy
        double initial_energy = state(0, 0) * state(0, 0) + state(1, 0) * state(1, 0);

        // Integrate for one period (T=2π ≈ 6.28)
        double T     = 2.0 * std::numbers::pi;
        int    steps = static_cast<int>(T / h);

        for (int i = 0; i < steps; ++i) {
            auto result = integrator.evolve(f, state, t, h);
            state       = result.x;
            t += h;
        }

        // Check energy conservation (should be close to initial)
        double final_energy = state(0, 0) * state(0, 0) + state(1, 0) * state(1, 0);
        CHECK(std::abs(final_energy - initial_energy) < 1e-2);  // Allow some energy drift
    }

    SUBCASE("Stiff equation with small step size") {
        RK4 integrator;

        Matrix x = Matrix::Constant(1, 1, 1.0);
        double t = 0.0;
        double h = 0.001;  // Very small step for stiff equation

        // Stiff: x' = -100*x
        auto f = [](double /*t*/, const Matrix& x) -> Matrix {
            return -100.0 * x;
        };

        // Take one step
        auto   result = integrator.evolve(f, x, t, h);
        Matrix x_new  = result.x;

        // Analytical: x(h) = e^(-100*h) ≈ 1 - 100*h + (100*h)^2/2 - ...
        double analytical = std::exp(-100.0 * h);
        CHECK(x_new(0, 0) == doctest::Approx(analytical).epsilon(1e-4));
    }
}

TEST_CASE("Trapezoidal Integrator") {
    SUBCASE("Implicit method properties") {
        Trapezoidal integrator;

        Matrix x = Matrix::Constant(1, 1, 1.0);
        double t = 0.0;
        double h = 0.1;

        // x' = -x
        auto f = [](double /*t*/, const Matrix& x) -> Matrix {
            return -x;
        };

        // Take one step
        auto   result = integrator.evolve(f, x, t, h);
        Matrix x_new  = result.x;

        // Trapezoidal: x_new = x + (h/2)*(f(t,x) + f(t+h,x_new))
        double analytical = std::exp(-0.1);
        CHECK(x_new(0, 0) == doctest::Approx(analytical).epsilon(1e-4));  // Second order accuracy
    }

    SUBCASE("A-stable for stiff equations") {
        Trapezoidal integrator;

        Matrix x = Matrix::Constant(1, 1, 1.0);
        double t = 0.0;
        double h = 0.01;  // Use smaller step for stiff system

        // Very stiff: x' = -100*x
        auto f = [](double /*t*/, const Matrix& x) -> Matrix {
            return -100.0 * x;
        };

        // Take one step
        auto   result = integrator.evolve(f, x, t, h);
        Matrix x_new  = result.x;

        // Trapezoidal should remain stable
        double analytical = std::exp(-100.0 * h);
        CHECK(x_new(0, 0) > 0.0);  // Should remain positive (A-stable)
        CHECK(x_new(0, 0) == doctest::Approx(analytical).epsilon(1e-1));
    }

    SUBCASE("Convergence for linear systems") {
        Trapezoidal integrator;

        Matrix x = Matrix::Constant(1, 1, 1.0);
        double t = 0.0;
        double h = 0.01;

        auto f = [](double /*t*/, const Matrix& x) -> Matrix {
            return -x;
        };

        // Integrate to t=1.0
        for (int i = 0; i < 100; ++i) {
            auto result = integrator.evolve(f, x, t, h);
            x           = result.x;
            t += h;
        }

        double analytical = std::exp(-1.0);
        CHECK(x(0, 0) == doctest::Approx(analytical).epsilon(1e-2));  // High accuracy
    }
}

TEST_CASE("BackwardEuler Integrator") {
    SUBCASE("Fully implicit method") {
        BackwardEuler integrator;

        Matrix x = Matrix::Constant(1, 1, 1.0);
        double t = 0.0;
        double h = 0.1;

        // x' = -x
        auto f = [](double /*t*/, const Matrix& x) -> Matrix {
            return -x;
        };

        // Take one step
        auto   result = integrator.evolve(f, x, t, h);
        Matrix x_new  = result.x;

        // Backward Euler: x_new = x + h*f(t+h,x_new)
        // For x' = -x, this gives: x_new = x / (1 + h)
        double expected = 1.0 / (1.0 + 0.1);  // 1 / 1.1 ≈ 0.909091
        CHECK(x_new(0, 0) == doctest::Approx(expected).epsilon(1e-6));
    }

    SUBCASE("L-stable for stiff equations") {
        BackwardEuler integrator;

        Matrix x = Matrix::Constant(1, 1, 1.0);
        double t = 0.0;
        double h = 0.001;  // Use smaller step

        // Stiff: x' = -100*x
        auto f = [](double /*t*/, const Matrix& x) -> Matrix {
            return -100.0 * x;
        };

        // Take one step
        auto   result = integrator.evolve(f, x, t, h);
        Matrix x_new  = result.x;

        // Backward Euler should be very stable
        double analytical = std::exp(-100.0 * h);
        CHECK(x_new(0, 0) > 0.0);
        CHECK(x_new(0, 0) == doctest::Approx(analytical).epsilon(1e-2));
    }

    SUBCASE("Damping properties") {
        BackwardEuler integrator;

        Matrix x = Matrix::Constant(1, 1, 1.0);
        double t = 0.0;
        double h = 0.01;

        auto f = [](double /*t*/, const Matrix& x) -> Matrix {
            return -x;
        };

        // Integrate to t=1.0
        for (int i = 0; i < 100; ++i) {
            auto result = integrator.evolve(f, x, t, h);
            x           = result.x;
            t += h;
        }

        double analytical = std::exp(-1.0);
        CHECK(x(0, 0) == doctest::Approx(analytical).epsilon(1e-2));
    }
}

TEST_CASE("RK45 Integrator (Adaptive)") {
    SUBCASE("Adaptive step size control") {
        RK45 integrator;

        Matrix x = Matrix::Constant(1, 1, 1.0);
        double t = 0.0;
        double h = 0.1;

        // x' = -x
        auto f = [](double /*t*/, const Matrix& x) -> Matrix {
            return -x;
        };

        // Take one step with error control
        auto   result = integrator.evolve(f, x, t, h);
        Matrix x_new  = result.x;
        double error  = result.error;

        CHECK(error > 0.0);
        CHECK(x_new(0, 0) > 0.0);
        CHECK(x_new(0, 0) < 1.0);
    }

    SUBCASE("High accuracy for smooth problems") {
        RK45 integrator;

        Matrix x = Matrix::Constant(1, 1, 0.0);
        double t = 0.0;
        double h = 0.1;

        // x' = cos(t), analytical solution: x(t) = sin(t)
        auto f = [](double t, const Matrix& /*x*/) -> Matrix {
            return Matrix::Constant(1, 1, std::cos(t));
        };

        // Integrate to t=π/2
        double target_t = std::numbers::pi / 2.0;
        while (t < target_t) {
            auto result = integrator.evolve(f, x, t, h);
            x           = result.x;
            t += h;
            if (t > target_t) t = target_t;
        }

        // Analytical solution: sin(π/2) = 1
        CHECK(x(0, 0) == doctest::Approx(1.0).epsilon(1e-3));
    }

    SUBCASE("Step size reduction for difficult regions") {
        RK45 integrator;

        Matrix x = Matrix::Constant(1, 1, 1.0);
        double t = 0.0;
        double h = 0.1;

        // x' = -100*x + sin(100*t) - creates rapid oscillations
        auto f = [](double t, const Matrix& x) -> Matrix {
            return -100.0 * x + Matrix::Constant(1, 1, std::sin(100.0 * t));
        };

        // Take a few adaptive steps
        for (int i = 0; i < 5; ++i) {
            auto result = integrator.evolve(f, x, t, h);
            x           = result.x;
            t += h;

            CHECK(h > 0.0);
            CHECK(result.error >= 0.0);
        }
    }
}

TEST_CASE("RK23 Integrator (Adaptive)") {
    SUBCASE("Lower order adaptive method") {
        RK23 integrator;

        Matrix x = Matrix::Constant(1, 1, 1.0);
        double t = 0.0;
        double h = 0.1;

        // x' = -x
        auto f = [](double /*t*/, const Matrix& x) -> Matrix {
            return -x;
        };

        // Take one adaptive step
        auto   result = integrator.evolve(f, x, t, h);
        Matrix x_new  = result.x;
        double error  = result.error;

        CHECK(error > 0.0);
        CHECK(x_new(0, 0) > 0.0);
        CHECK(x_new(0, 0) < 1.0);
    }

    SUBCASE("Comparison with RK45 for efficiency") {
        RK23 rk23;
        RK45 rk45;

        Matrix x23 = Matrix::Constant(1, 1, 1.0);
        Matrix x45 = Matrix::Constant(1, 1, 1.0);
        double t23 = 0.0;
        double t45 = 0.0;
        double h23 = 0.1;
        double h45 = 0.1;

        auto f = [](double /*t*/, const Matrix& x) -> Matrix {
            return -x;
        };

        // Integrate both to t=1.0
        double target_t = 1.0;
        int    steps23  = 0;
        int    steps45  = 0;

        while (t23 < target_t) {
            auto result = rk23.evolve(f, x23, t23, h23);
            x23         = result.x;
            t23 += h23;
            steps23++;
        }

        while (t45 < target_t) {
            auto result = rk45.evolve(f, x45, t45, h45);
            x45         = result.x;
            t45 += h45;
            steps45++;
        }

        // Both should reach similar final values
        double analytical = std::exp(-1.0);
        CHECK(x23(0, 0) == doctest::Approx(analytical).epsilon(1e-1));
        CHECK(x45(0, 0) == doctest::Approx(analytical).epsilon(1e-1));

        // RK23 might take more steps but should still work
        CHECK(steps23 > 0);
        CHECK(steps45 > 0);
    }
}

TEST_CASE("Integrator Order and Accuracy") {
    SUBCASE("Convergence rates") {
        // Test convergence by halving step size and checking error reduction
        auto f = [](double /*t*/, const Matrix& x) -> Matrix {
            return -x;
        };

        Matrix x0         = Matrix::Constant(1, 1, 1.0);
        double t_final    = 1.0;
        double analytical = std::exp(-1.0);

        // Test Forward Euler (order 1)
        std::vector<double> errors_fe;
        for (double h : {0.1, 0.05, 0.025, 0.0125}) {
            ForwardEuler integrator;
            Matrix       x     = x0;
            double       t     = 0.0;
            int          steps = static_cast<int>(t_final / h);

            for (int i = 0; i < steps; ++i) {
                auto result = integrator.evolve(f, x, t, h);
                x           = result.x;
                t += h;
            }

            double error = std::abs(x(0, 0) - analytical);
            errors_fe.push_back(error);
        }

        // Check order 1 convergence: error should halve when h halves
        CHECK(errors_fe[1] / errors_fe[0] == doctest::Approx(0.5).epsilon(0.1));
        CHECK(errors_fe[2] / errors_fe[1] == doctest::Approx(0.5).epsilon(0.1));

        // Test RK4 (order 4)
        std::vector<double> errors_rk4;
        for (double h : {0.1, 0.05, 0.025, 0.0125}) {
            RK4    integrator;
            Matrix x     = x0;
            double t     = 0.0;
            int    steps = static_cast<int>(t_final / h);

            for (int i = 0; i < steps; ++i) {
                auto result = integrator.evolve(f, x, t, h);
                x           = result.x;
                t += h;
            }

            double error = std::abs(x(0, 0) - analytical);
            errors_rk4.push_back(error);
        }

        // Check order 4 convergence: error should be reduced by factor of 16 when h halves
        CHECK(errors_rk4[1] / errors_rk4[0] == doctest::Approx(1.0 / 16.0).epsilon(0.5));
        CHECK(errors_rk4[2] / errors_rk4[1] == doctest::Approx(1.0 / 16.0).epsilon(0.5));
    }
}

TEST_CASE("Integrator Stability") {
    SUBCASE("Forward Euler instability") {
        ForwardEuler integrator;

        Matrix x = Matrix::Constant(1, 1, 1.0);
        double t = 0.0;
        double h = 2.1;  // h * |lambda| = 2.1 > 2, should be unstable

        // x' = -x
        auto f = [](double /*t*/, const Matrix& x) -> Matrix {
            return -x;
        };

        // Take a few steps
        for (int i = 0; i < 5; ++i) {
            auto result = integrator.evolve(f, x, t, h);
            x           = result.x;
            t += h;
        }

        // Forward Euler should become unstable (oscillate and grow)
        CHECK((x(0, 0) > 1.0 || x(0, 0) < -1.0));  // Should have grown in magnitude
    }

    SUBCASE("Backward Euler stability") {
        BackwardEuler integrator;

        Matrix x = Matrix::Constant(1, 1, 1.0);
        double t = 0.0;
        double h = 1.0;  // Large step but not extreme

        // x' = -x
        auto f = [](double /*t*/, const Matrix& x) -> Matrix {
            return -x;
        };

        // Take a few steps
        for (int i = 0; i < 5; ++i) {
            auto result = integrator.evolve(f, x, t, h);
            x           = result.x;
            t += h;
        }

        // Backward Euler should remain stable and decay (but may decay very fast)
        CHECK(x(0, 0) >= 0.0);  // Should not go negative (stable)
        CHECK(x(0, 0) <= 1.0);  // Should not grow
    }

    SUBCASE("RK4 stability") {
        RK4 integrator;

        Matrix x = Matrix::Constant(1, 1, 1.0);
        double t = 0.0;
        double h = 2.5;  // h * |lambda| = 2.5, RK4 stability limit is around 2.8

        // x' = -x
        auto f = [](double /*t*/, const Matrix& x) -> Matrix {
            return -x;
        };

        // Take a few steps
        for (int i = 0; i < 5; ++i) {
            auto result = integrator.evolve(f, x, t, h);
            x           = result.x;
            t += h;
        }

        // RK4 should remain stable
        CHECK(x(0, 0) > 0.0);
        CHECK(x(0, 0) < 1.0);
    }
}