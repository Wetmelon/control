#include <algorithm>
#include <cmath>

#include "wet/kinematics/motion_maps.hpp"
#include "wet/trajectory/cartesian_move.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wet;

namespace {

// IK adapters: wrap a motion map as p -> {array<T,N>, reachable}.
auto corexy_ik = [](const Vec3<double>& p) {
    const auto m = CoreXY<double>::inverse(p[0], p[1]);
    return wet::pair<wet::array<double, 2>, bool>{wet::array<double, 2>{m.a, m.b}, true};
};
auto polar_ik = [](const Vec3<double>& p) {
    const auto a = PolarMap<double>::inverse(p[0], p[1]);
    return wet::pair<wet::array<double, 2>, bool>{wet::array<double, 2>{a.r, a.theta}, true};
};
// Polar IK that declares points outside a radius unreachable (a circular
// workspace), to exercise the move's reachability propagation.
auto polar_ik_bounded = [](const Vec3<double>& p) {
    const auto a = PolarMap<double>::inverse(p[0], p[1]);
    const bool ok = a.r <= 100.0;
    return wet::pair<wet::array<double, 2>, bool>{wet::array<double, 2>{a.r, a.theta}, ok};
};

// Largest deviation of a reconstructed tool point from the straight line A->B
// (i.e. is the path preserved?). fwd maps joints back to the (x, y) tool point.
template<typename Move, typename Fwd>
double max_path_deviation(const Move& mv, const Vec3<double>& A, const Vec3<double>& B, Fwd fwd) {
    Vec3<double>       d = B - A;
    const double       L = d.norm();
    const Vec3<double> u = d / L;
    double             worst = 0.0;
    const int          N = 400;
    for (int k = 0; k <= N; ++k) {
        const auto   st = mv.eval(mv.duration() * k / N);
        const auto   xy = fwd(st); // {x, y}
        const double ex = xy.first - A[0], ey = xy.second - A[1];
        // perpendicular distance to the line = |cross((ex,ey), u)|
        worst = std::max(worst, std::abs((ex * u[1]) - (ey * u[0])));
    }
    return worst;
}

} // namespace

TEST_SUITE("cartesian_move") {

    TEST_CASE("CoreXY straight line: path preserved, motors derated to their limit") {
        const Vec3<double>       A{-50.0, 30.0, 0.0};
        const Vec3<double>       B{50.0, -30.0, 0.0};
        const LinearPath<double> path(A, B);
        JointLimits<2, double>   jl;
        jl.max_velocity = {120.0, 120.0};
        jl.max_acceleration = {2000.0, 2000.0};
        const auto mv = make_cartesian_move<2>(path, corexy_ik, path.length(), TrajectoryLimits<double>{200.0, 800.0, 800.0, 8000.0}, jl);
        REQUIRE(mv.valid());
        REQUIRE(mv.reachable());

        // Endpoints are exactly IK(A) and IK(B).
        const auto s0 = mv.eval(0.0);
        CHECK(s0[0].position == doctest::Approx(-20.0)); // a = x+y = -50+30
        CHECK(s0[1].position == doctest::Approx(-80.0)); // b = x-y = -50-30
        const auto sf = mv.eval(mv.duration());
        CHECK(sf[0].position == doctest::Approx(20.0));
        CHECK(sf[1].position == doctest::Approx(80.0));

        // Path preserved: tool stays exactly on the line.
        const double dev = max_path_deviation(mv, A, B, [](const auto& st) {
            const auto pt = CoreXY<double>::forward(st[0].position, st[1].position);
            return std::pair<double, double>{pt.x, pt.y};
        });
        CHECK(dev < 1e-9);

        // Motor velocities respect the cap, and the binding motor (b) hits it.
        double max_va = 0.0, max_vb = 0.0;
        for (int k = 0; k <= 1000; ++k) {
            const auto st = mv.eval(mv.duration() * k / 1000.0);
            max_va = std::max(max_va, std::abs(st[0].velocity));
            max_vb = std::max(max_vb, std::abs(st[1].velocity));
        }
        CHECK(max_va <= 120.0 + 1e-6);
        CHECK(max_vb <= 120.0 + 1e-6);
        CHECK(max_vb == doctest::Approx(120.0).epsilon(2e-3)); // b is the constraint
        CHECK(max_va < 120.0);                                 // a has headroom
        CHECK(mv.scale() > 1.0);                               // genuinely derated
    }

    TEST_CASE("CoreXY: generous joint limits => no derate (K = 1, feed-limited)") {
        const Vec3<double>       A{0.0, 0.0, 0.0};
        const Vec3<double>       B{40.0, 30.0, 0.0};
        const LinearPath<double> path(A, B);
        JointLimits<2, double>   jl;
        jl.max_velocity = {1e4, 1e4};
        jl.max_acceleration = {1e6, 1e6};
        const auto mv = make_cartesian_move(path, corexy_ik, path.length(), TrajectoryLimits<double>{50.0, 400.0, 400.0, 4000.0}, jl);
        REQUIRE(mv.valid());
        CHECK(mv.scale() == doctest::Approx(1.0)); // joints never bind -> path feed governs
    }

    TEST_CASE("Polar straight line near the origin: huge derate, path still exact") {
        // Skims the origin (closest approach ~2.5 mm) — θ swings fast but stays
        // continuous (a line *through* the origin is a genuine atan2 branch cut).
        const Vec3<double>       A{-50.0, 20.0, 0.0};
        const Vec3<double>       B{50.0, -15.0, 0.0};
        const LinearPath<double> path(A, B);
        JointLimits<2, double>   jl;    // r [mm], theta [rad]
        jl.max_velocity = {500.0, 4.0}; // theta-rate is the tight one near r->0
        jl.max_acceleration = {5000.0, 80.0};
        // Oversample the K-sweep: the θ̇ peak near the skim is narrow, so a coarse
        // sweep underestimates it and under-derates (a real global-K limitation).
        const auto mv = make_cartesian_move<2>(path, polar_ik, path.length(), TrajectoryLimits<double>{300.0, 2000.0, 2000.0, 20000.0}, jl, 4000);
        REQUIRE(mv.valid());

        // Path preserved despite the polar nonlinearity.
        const double dev = max_path_deviation(mv, A, B, [](const auto& st) {
            const auto pt = PolarMap<double>::forward(st[0].position, st[1].position);
            return std::pair<double, double>{pt.x, pt.y};
        });
        CHECK(dev < 1e-6);

        // theta-rate honored everywhere; the near-origin pass forces a big slow-down.
        double max_w = 0.0;
        for (int k = 0; k <= 4000; ++k) {
            const auto st = mv.eval(mv.duration() * k / 4000.0);
            max_w = std::max(max_w, std::abs(st[1].velocity));
        }
        CHECK(max_w <= 4.0 + 1e-2); // small residual: discrete sweep vs the narrow peak
        CHECK(mv.scale() > 2.0);    // strongly derated by the singularity skim
    }

    TEST_CASE("analytic joint velocity matches finite-difference of position") {
        const Vec3<double>       A{-40.0, 25.0, 0.0};
        const Vec3<double>       B{40.0, -20.0, 0.0}; // skims, doesn't cross, the origin
        const LinearPath<double> path(A, B);
        JointLimits<2, double>   jl;
        jl.max_velocity = {80.0, 3.0};
        jl.max_acceleration = {1500.0, 60.0};
        const auto mv = make_cartesian_move<2>(path, polar_ik, path.length(), TrajectoryLimits<double>{250.0, 1500.0, 1500.0, 15000.0}, jl);
        REQUIRE(mv.valid());
        const double h = 1e-6;
        for (int k = 1; k < 40; ++k) {
            const double t = mv.duration() * k / 40.0;
            const auto   sp = mv.eval(t + h);
            const auto   sm = mv.eval(t - h);
            const auto   s = mv.eval(t);
            for (size_t i = 0; i < 2; ++i) {
                const double fd = (sp[i].position - sm[i].position) / (2 * h);
                CHECK(s[i].velocity == doctest::Approx(fd).epsilon(1e-4));
            }
        }
    }

    TEST_CASE("invalid inputs report failure") {
        const LinearPath<double> path({0.0, 0.0, 0.0}, {10.0, 0.0, 0.0});
        JointLimits<2, double>   bad; // zero limits
        const auto               mv = make_cartesian_move<2>(path, corexy_ik, path.length(), TrajectoryLimits<double>{100.0, 400.0, 400.0, 4000.0}, bad);
        CHECK_FALSE(mv.valid());
    }

    TEST_CASE("acceleration is the binding constraint: motors derate to a_max") {
        // Same geometry as the velocity test, but velocity limits are made huge so
        // the acceleration cap binds instead. The slow-down then comes from the
        // sqrt(r_a) branch of K, and accel (∝ 1/K²) is what hits the limit.
        const Vec3<double>       A{-50.0, 30.0, 0.0};
        const Vec3<double>       B{50.0, -30.0, 0.0};
        const LinearPath<double> path(A, B);
        JointLimits<2, double>   jl;
        jl.max_velocity = {1e5, 1e5}; // velocity never binds
        jl.max_acceleration = {500.0, 500.0};
        const auto mv = make_cartesian_move<2>(path, corexy_ik, path.length(), TrajectoryLimits<double>{200.0, 800.0, 800.0, 8000.0}, jl);
        REQUIRE(mv.valid());
        REQUIRE(mv.reachable());
        CHECK(mv.scale() > 1.0); // genuinely derated, by acceleration

        double max_aa = 0.0, max_ab = 0.0, max_va = 0.0, max_vb = 0.0;
        for (int k = 0; k <= 2000; ++k) {
            const auto st = mv.eval(mv.duration() * k / 2000.0);
            max_aa = std::max(max_aa, std::abs(st[0].acceleration));
            max_ab = std::max(max_ab, std::abs(st[1].acceleration));
            max_va = std::max(max_va, std::abs(st[0].velocity));
            max_vb = std::max(max_vb, std::abs(st[1].velocity));
        }
        // Acceleration cap honored; the binding motor (b) hits it, a has headroom.
        CHECK(max_aa <= 500.0 + 1e-3);
        CHECK(max_ab <= 500.0 + 1e-3);
        CHECK(max_ab == doctest::Approx(500.0).epsilon(5e-3)); // b is the constraint
        CHECK(max_aa < 500.0);                                 // a has headroom
        // Velocity is nowhere near its (huge) cap -> it really is accel-bound.
        CHECK(max_va < 1.0e3);
        CHECK(max_vb < 1.0e3);
    }

    TEST_CASE("reachability propagates from the IK over the swept path") {
        JointLimits<2, double> jl;
        jl.max_velocity = {500.0, 8.0};
        jl.max_acceleration = {5000.0, 160.0};
        const TrajectoryLimits<double> pl{300.0, 2000.0, 2000.0, 20000.0};

        // A path that leaves the r <= 100 workspace: the move is well-formed
        // (valid) but flagged unreachable because some swept points fail IK.
        const LinearPath<double> outside({0.0, 0.0, 0.0}, {200.0, 0.0, 0.0});
        const auto               mv_out = make_cartesian_move<2>(outside, polar_ik_bounded, outside.length(), pl, jl);
        CHECK(mv_out.valid());
        CHECK_FALSE(mv_out.reachable());

        // A path that stays inside the workspace is reachable.
        const LinearPath<double> inside({10.0, 0.0, 0.0}, {80.0, 0.0, 0.0});
        const auto               mv_in = make_cartesian_move<2>(inside, polar_ik_bounded, inside.length(), pl, jl);
        CHECK(mv_in.valid());
        CHECK(mv_in.reachable());
    }

    TEST_CASE("rest-to-rest: joints start and end at zero velocity and acceleration") {
        const Vec3<double>       A{-50.0, 30.0, 0.0};
        const Vec3<double>       B{50.0, -30.0, 0.0};
        const LinearPath<double> path(A, B);
        JointLimits<2, double>   jl;
        jl.max_velocity = {120.0, 120.0};
        jl.max_acceleration = {2000.0, 2000.0};
        const auto mv = make_cartesian_move<2>(path, corexy_ik, path.length(), TrajectoryLimits<double>{200.0, 800.0, 800.0, 8000.0}, jl);
        REQUIRE(mv.valid());

        const auto s0 = mv.eval(0.0);
        const auto sf = mv.eval(mv.duration());
        for (size_t i = 0; i < 2; ++i) {
            CHECK(s0[i].velocity == doctest::Approx(0.0));
            CHECK(s0[i].acceleration == doctest::Approx(0.0));
            CHECK(sf[i].velocity == doctest::Approx(0.0));
            CHECK(sf[i].acceleration == doctest::Approx(0.0));
        }
    }

    TEST_CASE("analytic joint acceleration matches finite-difference of velocity") {
        // A vertical line well away from the origin: the polar IK is curved here
        // (q''(s) != 0), so this exercises *both* terms of q̈ = q''·ṡ² + q'·s̈.
        const Vec3<double>       A{60.0, -25.0, 0.0};
        const Vec3<double>       B{60.0, 35.0, 0.0};
        const LinearPath<double> path(A, B);
        JointLimits<2, double>   jl;
        jl.max_velocity = {120.0, 5.0};
        jl.max_acceleration = {2000.0, 80.0};
        const auto mv = make_cartesian_move<2>(path, polar_ik, path.length(), TrajectoryLimits<double>{200.0, 1200.0, 1200.0, 15000.0}, jl);
        REQUIRE(mv.valid());
        const double h = 1e-6;
        for (int k = 1; k < 40; ++k) {
            const double t = mv.duration() * k / 40.0;
            const auto   sp = mv.eval(t + h);
            const auto   sm = mv.eval(t - h);
            const auto   s = mv.eval(t);
            for (size_t i = 0; i < 2; ++i) {
                const double fd = (sp[i].velocity - sm[i].velocity) / (2 * h);
                // Absolute slack covers the jerk corners (acceleration is only C⁰);
                // relative term covers the larger-magnitude ramp samples.
                CHECK(std::abs(s[i].acceleration - fd) <= 5e-2 + (2e-3 * std::abs(s[i].acceleration)));
            }
        }
    }

    TEST_CASE("clock interface: step accumulates, done() latches, reset() rewinds") {
        const Vec3<double>       A{0.0, 0.0, 0.0};
        const Vec3<double>       B{40.0, 30.0, 0.0};
        const LinearPath<double> path(A, B);
        JointLimits<2, double>   jl;
        jl.max_velocity = {120.0, 120.0};
        jl.max_acceleration = {2000.0, 2000.0};
        auto mv = make_cartesian_move<2>(path, corexy_ik, path.length(), TrajectoryLimits<double>{200.0, 800.0, 800.0, 8000.0}, jl);
        REQUIRE(mv.valid());
        const double D = mv.duration();

        CHECK(mv.time() == doctest::Approx(0.0));
        CHECK_FALSE(mv.done());

        mv.step(0.5 * D);
        CHECK(mv.time() == doctest::Approx(0.5 * D));
        CHECK_FALSE(mv.done());

        const auto last = mv.step(0.6 * D); // now past the end
        CHECK(mv.time() == doctest::Approx(1.1 * D));
        CHECK(mv.done());
        // step() is just eval() at the accumulated clock.
        const auto at_now = mv.eval(mv.time());
        for (size_t i = 0; i < 2; ++i) {
            CHECK(last[i].position == doctest::Approx(at_now[i].position));
        }

        mv.reset();
        CHECK(mv.time() == doctest::Approx(0.0));
        CHECK_FALSE(mv.done());
        // A fresh step matches a direct eval at the same time.
        const double dt = 0.25 * D;
        const auto   stepped = mv.step(dt);
        const auto   evaled = mv.eval(dt);
        for (size_t i = 0; i < 2; ++i) {
            CHECK(stepped[i].position == doctest::Approx(evaled[i].position));
            CHECK(stepped[i].velocity == doctest::Approx(evaled[i].velocity));
        }
    }
}
