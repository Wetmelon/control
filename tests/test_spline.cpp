#include <cmath>

#include "wet/trajectory/trajectory.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wet;

namespace {

// The d-th derivative of one segment's polynomial at local time tau.
template<typename Coeffs>
double seg_deriv(const Coeffs& c, double tau, size_t d) {
    double sum = 0.0;
    for (size_t j = d; j < c.size(); ++j) {
        double ff = 1.0; // falling factorial j·(j−1)···(j−d+1)
        for (size_t m = 0; m < d; ++m) {
            ff *= static_cast<double>(j - m);
        }
        sum += static_cast<double>(c[j]) * ff * std::pow(tau, static_cast<double>(j - d));
    }
    return sum;
}

// Largest jump in the d-th derivative across the interior knots, measured as the
// exact one-sided limits (left segment at its end vs right segment at 0).
template<typename Profile, size_t NPts>
double max_discontinuity(const Profile& sp, size_t d) {
    double worst = 0.0;
    for (size_t k = 1; k + 1 < NPts; ++k) {
        const double hL = static_cast<double>(sp.knots[k] - sp.knots[k - 1]);
        const double left = seg_deriv(sp.coeffs[k - 1], hL, d);
        const double right = seg_deriv(sp.coeffs[k], 0.0, d);
        worst = std::max(worst, std::abs(left - right));
    }
    return worst;
}

} // namespace

TEST_SUITE("spline") {

    TEST_CASE("cubic spline: interpolates every waypoint and is C² at interior knots") {
        const wet::array<double, 5> t{0.0, 1.0, 2.5, 3.0, 4.5};
        const wet::array<double, 5> p{0.0, 2.0, -1.0, 0.5, 3.0};
        const auto                  sp = design::cubic_spline<5>(t, p);
        REQUIRE(sp.success);

        for (size_t i = 0; i < 5; ++i) {
            CHECK(sp.eval(t[i]).position == doctest::Approx(p[i]).epsilon(1e-9));
        }
        // Position, velocity, acceleration continuous (cubic ⇒ C²).
        CHECK(max_discontinuity<decltype(sp), 5>(sp, 0) < 1e-9);
        CHECK(max_discontinuity<decltype(sp), 5>(sp, 1) < 1e-9);
        CHECK(max_discontinuity<decltype(sp), 5>(sp, 2) < 1e-9);
        CHECK(sp.duration == doctest::Approx(4.5));
    }

    TEST_CASE("cubic spline: clamped end velocities are honoured") {
        const wet::array<double, 4> t{0.0, 1.0, 2.0, 3.0};
        const wet::array<double, 4> p{0.0, 1.0, 1.0, 0.0};
        const auto                  sp = design::cubic_spline<4>(t, p, 0.5, -0.7);
        REQUIRE(sp.success);
        CHECK(sp.eval(0.0).velocity == doctest::Approx(0.5));
        CHECK(sp.eval(3.0).velocity == doctest::Approx(-0.7));
    }

    TEST_CASE("quintic spline: interpolates and is jerk-continuous (C³+)") {
        const wet::array<double, 4> t{0.0, 1.2, 2.0, 3.4};
        const wet::array<double, 4> p{0.0, 1.5, -0.5, 2.0};
        const auto                  sp = design::quintic_spline<4>(t, p);
        REQUIRE(sp.success);

        for (size_t i = 0; i < 4; ++i) {
            CHECK(sp.eval(t[i]).position == doctest::Approx(p[i]).epsilon(1e-9));
        }
        // Quintic ⇒ velocity, acceleration AND jerk continuous.
        CHECK(max_discontinuity<decltype(sp), 4>(sp, 1) < 1e-9);
        CHECK(max_discontinuity<decltype(sp), 4>(sp, 2) < 1e-9);
        CHECK(max_discontinuity<decltype(sp), 4>(sp, 3) < 1e-8);
    }

    TEST_CASE("quintic spline: clamped end velocity and acceleration are honoured") {
        const wet::array<double, 3> t{0.0, 1.0, 2.0};
        const wet::array<double, 3> p{0.0, 1.0, 0.0};
        const auto                  sp = design::quintic_spline<3>(t, p, 0.2, 0.3, -0.4, -0.6);
        REQUIRE(sp.success);
        CHECK(sp.eval(0.0).velocity == doctest::Approx(0.2));
        CHECK(sp.eval(0.0).acceleration == doctest::Approx(0.3));
        CHECK(sp.eval(2.0).velocity == doctest::Approx(-0.4));
        CHECK(sp.eval(2.0).acceleration == doctest::Approx(-0.6));
    }

    TEST_CASE("single-segment spline reduces to the boundary-value polynomial") {
        // NPts == 2 is exactly synthesize_poly_trajectory's BVP.
        const wet::array<double, 2> t{0.0, 2.0};
        const wet::array<double, 2> p{1.0, 4.0};
        const auto                  sp = design::cubic_spline<2>(t, p, 0.3, -0.5);
        const auto                  bvp = design::synthesize_poly_trajectory<3>(
            TrajectoryBoundary<double>{1.0, 0.3}, TrajectoryBoundary<double>{4.0, -0.5}, 2.0
        );
        REQUIRE(sp.success);
        REQUIRE(bvp.success);
        for (double tt = 0.0; tt <= 2.0; tt += 0.25) {
            CHECK(sp.eval(tt).position == doctest::Approx(bvp.eval(tt).position).epsilon(1e-9));
            CHECK(sp.eval(tt).velocity == doctest::Approx(bvp.eval(tt).velocity).epsilon(1e-9));
        }
    }

    TEST_CASE("runtime SplineTrajectory plays the profile and reports done") {
        const wet::array<double, 3>    t{0.0, 1.0, 2.0};
        const wet::array<double, 3>    p{0.0, 1.0, 0.0};
        SplineTrajectory<3, 3, double> traj(design::cubic_spline<3>(t, p));
        REQUIRE(traj.valid());
        CHECK(traj.duration() == doctest::Approx(2.0));

        TrajectoryState<double> s{};
        for (int i = 0; i < 20; ++i) {
            s = traj.step(0.1);
        }
        CHECK(traj.done());
        CHECK(s.position == doctest::Approx(0.0).epsilon(1e-9)); // last waypoint
    }

    TEST_CASE("constexpr construction and as<float> rebind") {
        constexpr wet::array<double, 3> t{0.0, 1.0, 2.0};
        constexpr wet::array<double, 3> p{0.0, 2.0, 0.0};
        constexpr auto                  sp = design::cubic_spline<3>(t, p);
        static_assert(sp.success, "cubic spline must solve at compile time");

        const auto spf = sp.as<float>();
        CHECK(spf.success);
        CHECK(static_cast<double>(spf.eval(1.0F).position) == doctest::Approx(2.0).epsilon(1e-5));
    }

    TEST_CASE("invalid times are rejected") {
        const wet::array<double, 3> t_bad{0.0, 1.0, 1.0}; // not strictly increasing
        const wet::array<double, 3> p{0.0, 1.0, 2.0};
        CHECK_FALSE(design::cubic_spline<3>(t_bad, p).success);
        CHECK_FALSE(design::quintic_spline<3>(t_bad, p).success);
    }
}
