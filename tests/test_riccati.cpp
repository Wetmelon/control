#include "doctest.h"
#include "wet/analysis/riccati.hpp"
#include "wet/controllers/lqr.hpp"
#include "wet/matrix/matrix.hpp"

using namespace wet;

// The Riccati solvers (care/dare, in namespace wet) are
// foundational: LQR, LQI, LQG and LQGI all stand on top of dare(), yet they had
// no direct test of their own. These tests validate by residual (plug the
// returned P back into the defining equation and check the residual is ~0) so
// they don't depend on a hand-derived closed-form solution, plus a cross-check
// that design::discrete_lqr's Riccati solution S matches dare() on the same
// system.
//
// @see "Optimal Control" (Anderson & Moore, 1990), §3.3 (CARE) and Ch. 4 (DARE).

TEST_SUITE("Riccati Solvers") {
    // ---- Continuous-time algebraic Riccati equation: AᵀP + PA − PBR⁻¹BᵀP + Q = 0 ----

    TEST_CASE("CARE scalar has known solution P = 1") {
        // a = 0, b = 1, q = 1, r = 1  ⇒  −P² + 1 = 0  ⇒  P = 1.
        constexpr Matrix<1, 1> A{{0.0}};
        constexpr Matrix<1, 1> B{{1.0}};
        constexpr Matrix<1, 1> Q{{1.0}};
        constexpr Matrix<1, 1> R{{1.0}};

        const auto P = care(A, B, Q, R);
        REQUIRE(P.has_value());
        CHECK(P.value()(0, 0) == doctest::Approx(1.0));
    }

    TEST_CASE("CARE residual is ~0 for a double integrator") {
        // Double integrator: position/velocity, acceleration input.
        constexpr Matrix<2, 2> A{{0.0, 1.0}, {0.0, 0.0}};
        constexpr Matrix<2, 1> B{{0.0}, {1.0}};
        constexpr auto         Q = Matrix<2, 2>::identity();
        constexpr auto         R = Matrix<1, 1>::identity(); // R = I ⇒ R⁻¹ = I, no inverse needed

        const auto P_opt = care(A, B, Q, R);
        REQUIRE(P_opt.has_value());
        const Matrix<2, 2>& P = P_opt.value();

        // Residual: AᵀP + PA − PB(R⁻¹)BᵀP + Q  (R⁻¹ = I)
        const Matrix<2, 2> residual = A.t() * P + P * A - P * B * B.t() * P + Q;
        CHECK(residual.norm() == doctest::Approx(0.0).epsilon(1e-6));

        // P of a CARE solution is symmetric positive definite.
        CHECK(P(0, 1) == doctest::Approx(P(1, 0)));
        CHECK(P(0, 0) > 0.0);
    }

    TEST_CASE("CARE is usable at compile time (constexpr)") {
        constexpr Matrix<1, 1> A{{0.0}};
        constexpr Matrix<1, 1> B{{1.0}};
        constexpr Matrix<1, 1> Q{{1.0}};
        constexpr Matrix<1, 1> R{{1.0}};
        constexpr auto         P = care(A, B, Q, R);
        static_assert(P.has_value(), "scalar CARE must converge at compile time");
    }

    // ---- Discrete-time algebraic Riccati equation ----
    // AᵀPA − P − AᵀPB(R + BᵀPB)⁻¹BᵀPA + Q = 0

    TEST_CASE("DARE residual is ~0 for a scalar system") {
        constexpr Matrix<1, 1> A{{1.0}};
        constexpr Matrix<1, 1> B{{1.0}};
        constexpr Matrix<1, 1> Q{{1.0}};
        constexpr Matrix<1, 1> R{{1.0}};

        const auto P_opt = dare(A, B, Q, R);
        REQUIRE(P_opt.has_value());
        const Matrix<1, 1>& P = P_opt.value();

        const Matrix<1, 1> S = R + B.t() * P * B; // scalar, positive
        const Matrix<1, 1> Sinv{{1.0 / S(0, 0)}};
        const Matrix<1, 1> residual = A.t() * P * A - P - A.t() * P * B * Sinv * B.t() * P * A + Q;
        CHECK(residual.norm() == doctest::Approx(0.0).epsilon(1e-6));
        CHECK(P(0, 0) > 0.0);
    }

    TEST_CASE("DARE residual is ~0 for a discretized double integrator") {
        // x[k+1] = A x[k] + B u[k], Ts = 0.1 s.
        constexpr Matrix<2, 2> A{{1.0, 0.1}, {0.0, 1.0}};
        constexpr Matrix<2, 1> B{{0.005}, {0.1}};
        constexpr auto         Q = Matrix<2, 2>::identity();
        constexpr auto         R = Matrix<1, 1>::identity();

        const auto P_opt = dare(A, B, Q, R);
        REQUIRE(P_opt.has_value());
        const Matrix<2, 2>& P = P_opt.value();

        const Matrix<1, 1> S = R + B.t() * P * B;
        const Matrix<1, 1> Sinv{{1.0 / S(0, 0)}};
        const Matrix<2, 2> residual = A.t() * P * A - P - A.t() * P * B * Sinv * B.t() * P * A + Q;
        CHECK(residual.norm() == doctest::Approx(0.0).epsilon(1e-6));

        // Symmetric, positive definite.
        CHECK(P(0, 1) == doctest::Approx(P(1, 0)));
        CHECK(P(0, 0) > 0.0);
    }

    TEST_CASE("DARE solution matches discrete_lqr's Riccati solution S") {
        // discrete_lqr solves the same DARE internally; its S field must agree
        // with calling design::dare directly on the same (A, B, Q, R).
        constexpr Matrix<2, 2> A{{1.0, 0.1}, {0.0, 1.0}};
        constexpr Matrix<2, 1> B{{0.005}, {0.1}};
        constexpr auto         Q = Matrix<2, 2>::identity();
        constexpr auto         R = Matrix<1, 1>::identity();

        const auto P_opt = dare(A, B, Q, R);
        REQUIRE(P_opt.has_value());

        const auto lqr = design::discrete_lqr(A, B, Q, R);
        REQUIRE(lqr.success);

        const Matrix<2, 2> diff = P_opt.value() - lqr.S;
        CHECK(diff.norm() == doctest::Approx(0.0).epsilon(1e-6));
    }

    TEST_CASE("DARE is usable at compile time (constexpr)") {
        constexpr Matrix<1, 1> A{{1.0}};
        constexpr Matrix<1, 1> B{{1.0}};
        constexpr Matrix<1, 1> Q{{1.0}};
        constexpr Matrix<1, 1> R{{1.0}};
        constexpr auto         P = wet::dare(A, B, Q, R);
        static_assert(P.has_value(), "scalar DARE must converge at compile time");
    }
}
