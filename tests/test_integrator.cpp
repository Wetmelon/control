#include <cmath>

#include "wet/simulation/integrator.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wet;

// Exercises the fixed-step integrators in wet::sim, with emphasis on the
// implicit LTI paths (BackwardEuler / BDF2 / Trapezoidal), which solve
// (I − αhA)·x_next = rhs each step. These overloads are templated and were
// previously never instantiated, so this suite both compiles and numerically
// validates them.
TEST_SUITE("integrator") {
    TEST_CASE("implicit LTI one-step matches the closed form (dx/dt = -x)") {
        // Scalar decay: A = -1, B = 0. Exact one-step factors are known.
        const Matrix<1, 1, double> A{{-1.0}};
        const Matrix<1, 1, double> B{{0.0}};
        const ColVec<1, double>    x0{1.0};
        const ColVec<1, double>    u{0.0};
        const double               h = 0.1;

        sim::BackwardEuler<1, double> be;
        // (1 + h)·x_next = x  ->  x_next = 1 / (1 + h).
        CHECK(be.evolve(A, B, x0, u, h).x[0] == doctest::Approx(1.0 / (1.0 + h)));

        sim::Trapezoidal<1, double> tr;
        // (1 + h/2)·x_next = (1 - h/2)·x.
        CHECK(tr.evolve(A, B, x0, u, h).x[0] == doctest::Approx((1.0 - (0.5 * h)) / (1.0 + 0.5 * h)));

        sim::BDF2<1, double> bdf;
        // First BDF2 step falls back to Backward Euler.
        CHECK(bdf.evolve(A, B, x0, u, h).x[0] == doctest::Approx(1.0 / (1.0 + h)));
    }

    TEST_CASE("implicit LTI integrators converge to e^{-t}") {
        const Matrix<1, 1, double> A{{-1.0}};
        const Matrix<1, 1, double> B{{0.0}};
        const ColVec<1, double>    u{0.0};
        const double               h = 0.001;
        const int                  N = 1000; // integrate to t = 1
        const double               exact = std::exp(-1.0);

        SUBCASE("Backward Euler (1st order)") {
            sim::BackwardEuler<1, double> be;
            ColVec<1, double>             x{1.0};
            for (int k = 0; k < N; ++k) {
                x = be.evolve(A, B, x, u, h).x;
            }
            CHECK(x[0] == doctest::Approx(exact).epsilon(1e-3));
        }
        SUBCASE("Trapezoidal (2nd order)") {
            sim::Trapezoidal<1, double> tr;
            ColVec<1, double>           x{1.0};
            for (int k = 0; k < N; ++k) {
                x = tr.evolve(A, B, x, u, h).x;
            }
            CHECK(x[0] == doctest::Approx(exact).epsilon(1e-5));
        }
        SUBCASE("BDF2 (2nd order, multistep)") {
            sim::BDF2<1, double> bdf;
            ColVec<1, double>    x{1.0};
            for (int k = 0; k < N; ++k) {
                x = bdf.evolve(A, B, x, u, h).x;
            }
            CHECK(x[0] == doctest::Approx(exact).epsilon(1e-4));
        }
    }

    TEST_CASE("implicit LTI integrators track a forced steady state (x_ss = u)") {
        // dx/dt = -x + u  ->  steady state x = u.
        const Matrix<1, 1, double> A{{-1.0}};
        const Matrix<1, 1, double> B{{1.0}};
        const ColVec<1, double>    u{2.0};
        const double               h = 0.01;

        sim::BackwardEuler<1, double> be;
        ColVec<1, double>             x{0.0};
        for (int k = 0; k < 3000; ++k) {
            x = be.evolve(A, B, x, u, h).x;
        }
        CHECK(x[0] == doctest::Approx(2.0).epsilon(1e-3));
    }

    TEST_CASE("implicit 2-state solve agrees with the exact (matrix-exponential) integrator") {
        // Damped oscillator: stable, non-diagonal A exercises the 2x2 linear solve.
        const Matrix<2, 2, double> A{{0.0, 1.0}, {-4.0, -0.5}};
        const Matrix<2, 1, double> B{{0.0}, {0.0}};
        const ColVec<1, double>    u{0.0};
        const ColVec<2, double>    x0{1.0, 0.0};
        const double               h = 0.0005;
        const int                  N = 2000; // t = 1

        // Reference: exact discrete update via the matrix exponential.
        sim::Exact<2, double> ex;
        ColVec<2, double>     xref = x0;
        for (int k = 0; k < N; ++k) {
            xref = ex.evolve(A, B, xref, u, h).x;
        }

        sim::BackwardEuler<2, double> be;
        ColVec<2, double>             xbe = x0;
        for (int k = 0; k < N; ++k) {
            xbe = be.evolve(A, B, xbe, u, h).x;
        }
        CHECK(xbe[0] == doctest::Approx(xref[0]).epsilon(2e-2));
        CHECK(xbe[1] == doctest::Approx(xref[1]).epsilon(2e-2));

        sim::Trapezoidal<2, double> tr;
        ColVec<2, double>           xtr = x0;
        for (int k = 0; k < N; ++k) {
            xtr = tr.evolve(A, B, xtr, u, h).x;
        }
        CHECK(xtr[0] == doctest::Approx(xref[0]).epsilon(1e-3));
        CHECK(xtr[1] == doctest::Approx(xref[1]).epsilon(1e-3));
    }

    TEST_CASE("implicit LTI integrator is constexpr-evaluable") {
        constexpr double y = []() consteval {
            Matrix<1, 1, double>          A{{-1.0}};
            Matrix<1, 1, double>          B{{0.0}};
            ColVec<1, double>             x{1.0};
            ColVec<1, double>             u{0.0};
            sim::BackwardEuler<1, double> be;
            for (int k = 0; k < 10; ++k) {
                x = be.evolve(A, B, x, u, 0.1).x;
            }
            return x[0];
        }();
        // 10 Backward-Euler steps of h = 0.1: x = 1 / 1.1^10.
        static_assert(y > 0.38 && y < 0.39, "implicit integrator must work at compile time");
        CHECK(y == doctest::Approx(1.0 / std::pow(1.1, 10)));
    }

    // -----------------------------------------------------------------------
    // Explicit integrators: ForwardEuler, Heun, RK3, RK23, DP5
    // All tested on scalar decay dx/dt = -x, x(0) = 1, exact solution e^{-1}.
    // -----------------------------------------------------------------------

    TEST_CASE("ForwardEuler LTI converges to e^{-1}") {
        const Matrix<1, 1, double> A{{-1.0}};
        const Matrix<1, 1, double> B{{0.0}};
        const ColVec<1, double>    u{0.0};
        const double               h = 0.001;
        const int                  N = 1000;
        const double               exact = std::exp(-1.0);

        sim::ForwardEuler<1, double> fe;
        ColVec<1, double>            x{1.0};
        for (int k = 0; k < N; ++k) {
            x = fe.evolve(A, B, x, u, h).x;
        }
        // Forward Euler is 1st-order; with h = 0.001 expect ~1e-3 relative error.
        CHECK(x[0] == doctest::Approx(exact).epsilon(1e-2));
    }

    TEST_CASE("ForwardEuler nonlinear converges to e^{-1}") {
        // f(t, x) = -x, same decay, nonlinear overload.
        auto f = [](double, const ColVec<1, double>& x) {
            return ColVec<1, double>{-x[0]};
        };
        const double h = 0.001;
        const int    N = 1000;
        const double exact = std::exp(-1.0);

        sim::ForwardEuler<1, double> fe;
        ColVec<1, double>            x{1.0};
        for (int k = 0; k < N; ++k) {
            x = fe.evolve(f, x, k * h, h).x;
        }
        CHECK(x[0] == doctest::Approx(exact).epsilon(1e-2));
    }

    TEST_CASE("Heun LTI converges to e^{-1}") {
        const Matrix<1, 1, double> A{{-1.0}};
        const Matrix<1, 1, double> B{{0.0}};
        const ColVec<1, double>    u{0.0};
        const double               h = 0.01;
        const int                  N = 100;
        const double               exact = std::exp(-1.0);

        sim::Heun<1, double> heun;
        ColVec<1, double>    x{1.0};
        for (int k = 0; k < N; ++k) {
            x = heun.evolve(A, B, x, u, h).x;
        }
        // Heun is 2nd-order; with h = 0.01 expect ~1e-4 error.
        CHECK(x[0] == doctest::Approx(exact).epsilon(1e-4));
    }

    TEST_CASE("Heun nonlinear: dx/dt = cos(t) -> sin(t)") {
        // Integrating dx/dt = cos(t) from t=0 to t=1 gives x(1) = sin(1).
        auto f = [](double t, const ColVec<1, double>&) {
            return ColVec<1, double>{std::cos(t)};
        };
        const double h = 0.001;
        const int    N = 1000;

        sim::Heun<1, double> heun;
        ColVec<1, double>    x{0.0};
        for (int k = 0; k < N; ++k) {
            x = heun.evolve(f, x, k * h, h).x;
        }
        CHECK(x[0] == doctest::Approx(std::sin(1.0)).epsilon(1e-5));
    }

    TEST_CASE("RK3 LTI converges to e^{-1}") {
        const Matrix<1, 1, double> A{{-1.0}};
        const Matrix<1, 1, double> B{{0.0}};
        const ColVec<1, double>    u{0.0};
        const double               h = 0.01;
        const int                  N = 100;
        const double               exact = std::exp(-1.0);

        sim::RK3<1, double> rk3;
        ColVec<1, double>   x{1.0};
        for (int k = 0; k < N; ++k) {
            x = rk3.evolve(A, B, x, u, h).x;
        }
        // RK3 is 3rd-order; with h = 0.01 expect ~1e-6 error.
        CHECK(x[0] == doctest::Approx(exact).epsilon(1e-6));
    }

    TEST_CASE("RK3 nonlinear: dx/dt = -x") {
        auto f = [](double, const ColVec<1, double>& x) {
            return ColVec<1, double>{-x[0]};
        };
        const double h = 0.01;
        const int    N = 100;
        const double exact = std::exp(-1.0);

        sim::RK3<1, double> rk3;
        ColVec<1, double>   x{1.0};
        for (int k = 0; k < N; ++k) {
            x = rk3.evolve(f, x, k * h, h).x;
        }
        CHECK(x[0] == doctest::Approx(exact).epsilon(1e-6));
    }

    TEST_CASE("RK23 LTI converges to e^{-1} and populates error estimate") {
        const Matrix<1, 1, double> A{{-1.0}};
        const Matrix<1, 1, double> B{{0.0}};
        const ColVec<1, double>    u{0.0};
        const double               h = 0.01;
        const int                  N = 100;
        const double               exact = std::exp(-1.0);

        sim::RK23<1, double> rk23;
        ColVec<1, double>    x{1.0};
        double               last_error = 0.0;
        for (int k = 0; k < N; ++k) {
            auto res = rk23.evolve(A, B, x, u, h);
            x = res.x;
            last_error = res.error;
        }
        // RK23 returns the 3rd-order solution; with h = 0.01 expect ~1e-6.
        CHECK(x[0] == doctest::Approx(exact).epsilon(1e-6));
        // Error estimate must be non-negative (it is a norm).
        CHECK(last_error >= 0.0);
    }

    TEST_CASE("RK23 nonlinear: dx/dt = -x") {
        auto f = [](double, const ColVec<1, double>& x) {
            return ColVec<1, double>{-x[0]};
        };
        const double h = 0.01;
        const int    N = 100;
        const double exact = std::exp(-1.0);

        sim::RK23<1, double> rk23;
        ColVec<1, double>    x{1.0};
        for (int k = 0; k < N; ++k) {
            x = rk23.evolve(f, x, k * h, h).x;
        }
        CHECK(x[0] == doctest::Approx(exact).epsilon(1e-6));
    }

    TEST_CASE("DP5 LTI converges to e^{-1} with high accuracy") {
        const Matrix<1, 1, double> A{{-1.0}};
        const Matrix<1, 1, double> B{{0.0}};
        const ColVec<1, double>    u{0.0};
        const double               h = 0.1;
        const int                  N = 10;
        const double               exact = std::exp(-1.0);

        sim::DP5<1, double> dp5;
        ColVec<1, double>   x{1.0};
        for (int k = 0; k < N; ++k) {
            x = dp5.evolve(A, B, x, u, h).x;
        }
        // DP5 is 5th-order; with h = 0.1 (10 steps to t = 1) expect ~1e-9.
        CHECK(x[0] == doctest::Approx(exact).epsilon(1e-9));
    }

    TEST_CASE("DP5 nonlinear: dx/dt = cos(t) -> sin(t)") {
        auto f = [](double t, const ColVec<1, double>&) {
            return ColVec<1, double>{std::cos(t)};
        };
        const double h = 0.1;
        const int    N = 10;

        sim::DP5<1, double> dp5;
        ColVec<1, double>   x{0.0};
        for (int k = 0; k < N; ++k) {
            x = dp5.evolve(f, x, k * h, h).x;
        }
        CHECK(x[0] == doctest::Approx(std::sin(1.0)).epsilon(1e-9));
    }

    TEST_CASE("explicit integrators are constexpr-evaluable (ForwardEuler)") {
        constexpr double y = []() consteval {
            Matrix<1, 1, double>         A{{-1.0}};
            Matrix<1, 1, double>         B{{0.0}};
            ColVec<1, double>            x{1.0};
            ColVec<1, double>            u{0.0};
            sim::ForwardEuler<1, double> fe;
            for (int k = 0; k < 10; ++k) {
                x = fe.evolve(A, B, x, u, 0.1).x;
            }
            return x[0];
        }();
        // 10 Forward-Euler steps of h = 0.1: x = (1 - 0.1)^10 = 0.9^10.
        static_assert(y > 0.34 && y < 0.36, "ForwardEuler must work at compile time");
        CHECK(y == doctest::Approx(std::pow(0.9, 10)));
    }

    TEST_CASE("Integrator variant wrapper holds an alternative and is indexable") {
        // Construct a variant holding RK4 and confirm the index matches.
        sim::Integrator<1, double> intgr{sim::RK4<1, double>{}};

        // Index 8 corresponds to RK4<NX,T> in the variant list:
        // Discrete(0), ForwardEuler(1), BackwardEuler(2), Trapezoidal(3),
        // BDF2(4), Heun(5), RK3(6), RK23(7), RK4(8), RK45(9), DP5(10), Exact(11).
        CHECK(intgr.index() == 8);

        // std::visit can extract the contained integrator and call evolve.
        const Matrix<1, 1, double> A{{-1.0}};
        const Matrix<1, 1, double> B{{0.0}};
        const ColVec<1, double>    u{0.0};
        const double               h = 0.1;
        const int                  N = 10;
        const double               exact = std::exp(-1.0);

        // std::get extracts the RK4 alternative directly and calls evolve.
        ColVec<1, double>    x{1.0};
        sim::RK4<1, double>& rk4 = std::get<sim::RK4<1, double>>(intgr);
        for (int k = 0; k < N; ++k) {
            x = rk4.evolve(A, B, x, u, h).x;
        }
        CHECK(x[0] == doctest::Approx(exact).epsilon(1e-6));
    }
}
