#include <cmath>

#include "wet/matrix/colvec.hpp" // Vec3
#include "wet/trajectory/cartesian_move.hpp"
#include "wet/trajectory/topp.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wet;

namespace {

// Task point is just a 3-vector; IK is the identity onto the first NJoints axes,
// so a "task path" is also the joint path and the geometry is transparent.

// Straight 1-joint path q(s) = s along x.
struct Path1 {
    [[nodiscard]] Vec3<double> operator()(double s) const { return Vec3<double>{s, 0.0, 0.0}; }
};
struct Ik1 {
    [[nodiscard]] wet::pair<wet::array<double, 1>, bool> operator()(const Vec3<double>& p) const {
        return {wet::array<double, 1>{p[0]}, true};
    }
};

// Straight 2-joint diagonal path.
struct Path2 {
    [[nodiscard]] Vec3<double> operator()(double s) const { return Vec3<double>{s, 0.5 * s, 0.0}; }
};
struct Ik2 {
    [[nodiscard]] wet::pair<wet::array<double, 2>, bool> operator()(const Vec3<double>& p) const {
        return {wet::array<double, 2>{p[0], p[1]}, true};
    }
};

// Curved 2-joint path: a circular arc, q = (r cos θ, r sin θ), θ = s/r. Exercises
// the curvature (acceleration-MVC) handling — q'' ≠ 0.
struct ArcPath {
    double                     r{1.0};
    [[nodiscard]] Vec3<double> operator()(double s) const {
        const double th = s / r;
        return Vec3<double>{r * std::cos(th), r * std::sin(th), 0.0};
    }
};
using IkArc = Ik2;

// Largest joint velocity / acceleration ratio against the limits over the move.
template<typename Move, size_t NJoints>
void check_within_limits(const Move& m, const JointLimits<NJoints, double>& lim, double v_tol, double a_tol) {
    const int    M = 600;
    const double T = m.duration();
    double       worst_v = 0.0;
    double       worst_a = 0.0;
    for (int k = 0; k <= M; ++k) {
        const double t = (T * k) / M;
        const auto   st = m.eval(t);
        for (size_t i = 0; i < NJoints; ++i) {
            worst_v = std::max(worst_v, std::abs(st[i].velocity) / lim.max_velocity[i]);
            worst_a = std::max(worst_a, std::abs(st[i].acceleration) / lim.max_acceleration[i]);
        }
    }
    CHECK(worst_v <= 1.0 + v_tol);
    CHECK(worst_a <= 1.0 + a_tol);
}

} // namespace

TEST_SUITE("topp") {

    TEST_CASE("single-joint straight path matches the analytic trapezoid time") {
        JointLimits<1, double> lim{};
        lim.max_velocity = {1.0};
        lim.max_acceleration = {2.0};

        // L = 2, vmax = 1, amax = 2 ⇒ trapezoidal (cruise reached): t = L/v + v/a.
        const double L = 2.0;
        const auto   m = make_topp_move<256>(Path1{}, Ik1{}, L, lim);
        REQUIRE(m.valid());
        const double t_expected = (L / 1.0) + (1.0 / 2.0); // 2.5
        CHECK(m.duration() == doctest::Approx(t_expected).epsilon(0.01));
        check_within_limits(m, lim, 0.01, 0.02);
    }

    TEST_CASE("single-joint short path matches the analytic triangular time") {
        JointLimits<1, double> lim{};
        lim.max_velocity = {1.0};
        lim.max_acceleration = {2.0};

        // L = 0.2 < vmax²/amax = 0.5 ⇒ triangular: t = 2·sqrt(L/amax).
        const double L = 0.2;
        const auto   m = make_topp_move<256>(Path1{}, Ik1{}, L, lim);
        REQUIRE(m.valid());
        const double t_expected = 2.0 * std::sqrt(L / 2.0);
        CHECK(m.duration() == doctest::Approx(t_expected).epsilon(0.02));
        // The cruise speed is never reached.
        CHECK(m.profile().sdot[128] < 1.0);
    }

    TEST_CASE("profile is a valid schedule: monotone s and strictly increasing time") {
        JointLimits<2, double> lim{};
        lim.max_velocity = {1.0, 1.0};
        lim.max_acceleration = {3.0, 3.0};
        const auto m = make_topp_move<128>(Path2{}, Ik2{}, 1.5, lim);
        REQUIRE(m.valid());
        const auto& pr = m.profile();
        for (size_t i = 0; i + 1 < 128; ++i) {
            CHECK(pr.s[i + 1] > pr.s[i] - 1e-12);
            CHECK(pr.time[i + 1] > pr.time[i]);
        }
        CHECK(pr.time[127] == doctest::Approx(pr.duration));
        // y-axis joint moves at half the rate ⇒ x-joint velocity is binding.
        check_within_limits(m, lim, 0.01, 0.03);
    }

    TEST_CASE("curved path: acceleration-MVC keeps joints within limits through the arc") {
        JointLimits<2, double> lim{};
        lim.max_velocity = {2.0, 2.0};
        lim.max_acceleration = {1.5, 1.5};
        ArcPath      arc{0.8};
        const double arc_len = 0.8 * (3.14159265358979 / 2.0); // quarter circle
        const auto   m = make_topp_move<400>(arc, IkArc{}, arc_len, lim);
        REQUIRE(m.valid());
        // Centripetal term q'' ≠ 0 makes acceleration the binding constraint.
        check_within_limits(m, lim, 0.01, 0.04);
    }

    TEST_CASE("TOPP is faster than the global-K CartesianMove on the same path") {
        JointLimits<2, double> lim{};
        lim.max_velocity = {1.0, 1.0};
        lim.max_acceleration = {2.0, 2.0};
        const double L = 1.5;

        TrajectoryLimits<double> path_limits{};
        path_limits.max_velocity = 1.0;
        path_limits.max_acceleration = 2.0;
        path_limits.max_deceleration = 2.0;
        path_limits.max_jerk = 20.0;

        const auto topp = make_topp_move<256>(Path2{}, Ik2{}, L, lim);
        const auto cart = make_cartesian_move(Path2{}, Ik2{}, L, path_limits, lim);
        REQUIRE(topp.valid());
        REQUIRE(cart.valid());

        // Pointwise-optimal bang-bang beats the single-global-scale jerk-limited move.
        CHECK(topp.duration() < cart.duration());
        // Both must respect the joint limits.
        check_within_limits(topp, lim, 0.01, 0.03);
    }

    TEST_CASE("boundary path speeds are honoured") {
        JointLimits<1, double> lim{};
        lim.max_velocity = {2.0};
        lim.max_acceleration = {3.0};

        // Start already moving; end at rest. The start speed shortens the move
        // versus rest-to-rest.
        const auto moving = make_topp_move<256>(Path1{}, Ik1{}, 2.0, lim, 1.0, 0.0);
        const auto rest = make_topp_move<256>(Path1{}, Ik1{}, 2.0, lim, 0.0, 0.0);
        REQUIRE(moving.valid());
        REQUIRE(rest.valid());

        CHECK(moving.profile().sdot[0] == doctest::Approx(1.0).epsilon(1e-6));
        CHECK(moving.eval(0.0)[0].velocity == doctest::Approx(1.0).epsilon(1e-6));
        CHECK(moving.duration() < rest.duration());
        check_within_limits(moving, lim, 0.01, 0.03);
    }

    TEST_CASE("invalid configuration is reported, not silently accepted") {
        JointLimits<1, double> bad{};
        bad.max_velocity = {0.0}; // non-positive limit
        bad.max_acceleration = {1.0};
        const auto m = make_topp_move<64>(Path1{}, Ik1{}, 1.0, bad);
        CHECK_FALSE(m.valid());

        JointLimits<1, double> ok{};
        ok.max_velocity = {1.0};
        ok.max_acceleration = {1.0};
        const auto zero_len = make_topp_move<64>(Path1{}, Ik1{}, 0.0, ok);
        CHECK_FALSE(zero_len.valid());
    }
}
