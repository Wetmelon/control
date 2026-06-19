#include <cmath>

#include "wet/controllers/lqr.hpp"
#include "wet/matrix/matrix.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wet;

TEST_CASE("discrete_lqr scalar gain matches hand solution") {
    //! A=1,B=1,Q=1,R=1: DARE gives S²−S−1=0 → S=φ=1.618…, K=S/(1+S)=0.618…
    const auto res = design::discrete_lqr(
        Matrix<1, 1>{{1.0}}, Matrix<1, 1>{{1.0}}, Matrix<1, 1>{{1.0}}, Matrix<1, 1>{{1.0}}
    );

    REQUIRE(res.success);
    const double phi = (1.0 + std::sqrt(5.0)) / 2.0;
    CHECK(res.S(0, 0) == doctest::Approx(phi));
    CHECK(res.K(0, 0) == doctest::Approx(phi / (1.0 + phi))); // 0.6180…
    CHECK(res.is_stable());                                   // pole A−BK = 0.382
    CHECK(res.e[0].abs() == doctest::Approx(1.0 - phi / (1.0 + phi)));
}

TEST_CASE("LQR runtime regulation and tracking laws") {
    const design::LQRResult<2, 1> res = design::discrete_lqr(
        Matrix<2, 2>{{1.0, 0.1}, {0.0, 1.0}}, Matrix<2, 1>{{0.0}, {0.1}},
        Matrix<2, 2>{{1.0, 0.0}, {0.0, 1.0}}, Matrix<1, 1>{{1.0}}
    );
    REQUIRE(res.success);

    const LQR<2, 1> ctrl{res}; // implicit from result

    SUBCASE("u = -K x") {
        const ColVec<2> x{{1.0, 2.0}};
        const auto      u = ctrl.control(x);
        CHECK(u[0] == doctest::Approx(-(res.K(0, 0) * 1.0 + res.K(0, 1) * 2.0)));
    }

    SUBCASE("tracking error is zero at the reference") {
        const ColVec<2> x{{3.0, -1.0}};
        const auto      u = ctrl.control(x, x); // x == x_ref
        CHECK(u[0] == doctest::Approx(0.0));
    }
}

TEST_CASE("LQRResult::as<float>() round-trips every field") {
    const auto res_d = design::discrete_lqr(
        Matrix<1, 1>{{1.0}}, Matrix<1, 1>{{1.0}}, Matrix<1, 1>{{1.0}}, Matrix<1, 1>{{1.0}}
    );
    const auto res_f = res_d.as<float>();

    CHECK(res_f.success);
    CHECK(res_f.K(0, 0) == doctest::Approx(static_cast<float>(res_d.K(0, 0))));
    CHECK(res_f.S(0, 0) == doctest::Approx(static_cast<float>(res_d.S(0, 0))));
    CHECK(res_f.e[0].abs() == doctest::Approx(static_cast<float>(res_d.e[0].abs())));
}

TEST_CASE("discretize_lqr_cost matches closed-form for A=0 integrator") {
    //! ẋ = u (A=0,B=1), x(τ)=x0+uτ over a sample h gives exact weights:
    //!   Qd = h,  Nd = h²/2,  Rd = h + h³/3
    const double h = 0.1;
    const auto   cost = design::discretize_lqr_cost(
        Matrix<1, 1>{{0.0}}, Matrix<1, 1>{{1.0}}, Matrix<1, 1>{{1.0}}, Matrix<1, 1>{{1.0}}, h
    );

    CHECK(cost.Q(0, 0) == doctest::Approx(h));
    CHECK(cost.N(0, 0) == doctest::Approx(h * h / 2.0));
    CHECK(cost.R(0, 0) == doctest::Approx(h + h * h * h / 3.0));
}

TEST_CASE("discrete_lqr_from_continuous discretizes cost (differs from naive)") {
    const Matrix<2, 2> A{{0.0, 1.0}, {0.0, 0.0}}; // double integrator
    const Matrix<2, 1> B{{0.0}, {1.0}};
    const Matrix<2, 2> Q{{1.0, 0.0}, {0.0, 1.0}};
    const Matrix<1, 1> R{{1.0}};
    const double       Ts = 0.2;

    const auto proper = design::discrete_lqr_from_continuous(A, B, Q, R, Ts);
    REQUIRE(proper.success);
    CHECK(proper.is_stable());

    //! Golden gain from an independent scipy implementation (Van Loan cost
    //! discretization + dlqr), see AGENTS.md "golden reference data".
    CHECK(proper.K(0, 0) == doctest::Approx(0.843695465061778));
    CHECK(proper.K(0, 1) == doctest::Approx(1.548939304133434));

    //! And it must differ from the naive path (ZOH dynamics, continuous cost) —
    //! otherwise the cost discretization is a silent no-op.
    StateSpace<2, 1, 2, 2, 2> sys_c{A, B, Matrix<2, 2>::identity()};
    const auto                sys_d = discretize(sys_c, Ts, DiscretizationMethod::ZOH);
    const auto                naive = design::discrete_lqr(sys_d.A, sys_d.B, Q, R);
    REQUIRE(naive.success);
    CHECK(std::abs(proper.K(0, 0) - naive.K(0, 0)) > 1e-4);
}
