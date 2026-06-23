#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>

#include "wet/backend.hpp"
#include "wet/math/math.hpp"
#include "wet/trajectory/trajectory.hpp"
#include "wet/trajectory/trajectory_types.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wet;

namespace {

// Numerically verify a planned profile: sample eval(t) finely and check
// boundary conditions, position/velocity continuity, that the endpoint is
// reached, and that the kinematic limits are respected.
struct Report {
    double pos_err_start{0}, vel_err_start{0};
    double pos_err_end{0}, vel_err_end{0};
    double max_pos_jump{0}, max_vel_jump{0};
    double max_speed{0}, max_accel_phase{0}, max_decel_phase{0};
    bool   monotonic_through{true};
};

template<typename Profile>
Report verify(const Profile& p) {
    using S = decltype(p.Xi); // scalar type of the profile (float/double), no value_type alias needed
    Report       r{};
    const double Tf = p.Tf;
    const int    N = 4000;
    const double dt = (Tf > 0.0) ? Tf / N : 0.0;

    auto s0 = p.eval(static_cast<S>(0));
    r.pos_err_start = std::abs(s0.position - p.Xi);
    r.vel_err_start = std::abs(s0.velocity - p.Vi);

    auto sf = p.eval(static_cast<S>(Tf));
    r.pos_err_end = std::abs(sf.position - p.Xf);
    r.vel_err_end = std::abs(sf.velocity - p.Vf);

    // Classify |a| by phase (time window), not by sign: with asymmetric limits
    // the first (Amax-bounded) ramp may itself be a deceleration, so accel sign
    // does not identify the phase.
    const double Ta = p.Ta;
    const double Tav = p.Ta + p.Tv;

    double prev_p = s0.position;
    double prev_v = s0.velocity;
    for (int k = 1; k <= N; ++k) {
        const double t = k * dt;
        const auto   st = p.eval(static_cast<S>(t));
        const double pp = st.position;
        const double vv = st.velocity;
        const double aa = std::abs(st.acceleration);
        r.max_pos_jump = std::max(r.max_pos_jump, std::abs(pp - prev_p));
        r.max_vel_jump = std::max(r.max_vel_jump, std::abs(vv - prev_v));
        r.max_speed = std::max(r.max_speed, std::abs(vv));
        if (t < Ta) {
            r.max_accel_phase = std::max(r.max_accel_phase, aa);
        } else if (t >= Tav) {
            r.max_decel_phase = std::max(r.max_decel_phase, aa);
        }
        prev_p = pp;
        prev_v = vv;
    }
    return r;
}

} // namespace

TEST_SUITE("trajectory") {
    using Limits = TrajectoryLimits<double>;

    TEST_CASE("rest-to-rest symmetric long move: trapezoid reaches cruise") {
        const Limits lim{2.0, 5.0, 5.0};
        const auto   p = design::synthesize_trapezoidal(0.0, 10.0, lim);
        REQUIRE(p.success);
        CHECK(p.Tv > 0.0);                   // long move has a cruise phase
        CHECK(p.Vr == doctest::Approx(2.0)); // reaches Vmax
        const auto r = verify(p);
        CHECK(r.pos_err_start < 1e-9);
        CHECK(r.pos_err_end < 1e-6);
        CHECK(r.vel_err_end < 1e-6);
        CHECK(r.max_speed <= 2.0 + 1e-6);
        CHECK(r.max_accel_phase <= 5.0 + 1e-6);
    }

    TEST_CASE("rest-to-rest short move: triangle never reaches Vmax") {
        const Limits lim{100.0, 5.0, 5.0}; // Vmax huge -> no cruise
        const auto   p = design::synthesize_trapezoidal(0.0, 1.0, lim);
        REQUIRE(p.success);
        CHECK(p.Tv == doctest::Approx(0.0));
        CHECK(std::abs(p.Vr) < 100.0);
        const auto r = verify(p);
        CHECK(r.pos_err_end < 1e-6);
        CHECK(r.vel_err_end < 1e-6);
        // symmetric: Ta == Td
        CHECK(p.Ta == doctest::Approx(p.Td));
    }

    TEST_CASE("asymmetric limits: gentle accel, hard decel") {
        const Limits lim{2.0, 1.0, 8.0}; // Amax=1, Dmax=8
        const auto   p = design::synthesize_trapezoidal(0.0, 10.0, lim);
        REQUIRE(p.success);
        CHECK(p.Ta > p.Td); // slow accel takes longer than fast decel
        CHECK(p.Ar == doctest::Approx(1.0));
        CHECK(p.Dr == doctest::Approx(-8.0));
        const auto r = verify(p);
        CHECK(r.pos_err_end < 1e-6);
        CHECK(r.vel_err_end < 1e-6);
        CHECK(r.max_accel_phase <= 1.0 + 1e-6);
        CHECK(r.max_decel_phase <= 8.0 + 1e-6);
        CHECK(r.max_speed <= 2.0 + 1e-6);
    }

    TEST_CASE("negative direction move") {
        const Limits lim{2.0, 5.0, 5.0};
        const auto   p = design::synthesize_trapezoidal(5.0, -5.0, lim);
        REQUIRE(p.success);
        CHECK(p.Vr < 0.0);
        const auto r = verify(p);
        CHECK(r.pos_err_end < 1e-6);
        CHECK(r.vel_err_end < 1e-6);
        CHECK(r.max_speed <= 2.0 + 1e-6);
    }

    TEST_CASE("arbitrary nonzero final velocity") {
        const Limits lim{5.0, 4.0, 4.0};
        const auto   p = design::synthesize_trapezoidal(0.0, 0.0, 8.0, 3.0, lim);
        REQUIRE(p.success);
        const auto r = verify(p);
        CHECK(r.pos_err_start < 1e-9);
        CHECK(r.vel_err_start < 1e-9);
        CHECK(r.pos_err_end < 1e-6);
        CHECK(r.vel_err_end < 1e-6); // ends at Vf = 3, not 0
        CHECK(r.max_speed <= 5.0 + 1e-6);
    }

    TEST_CASE("nonzero initial velocity, both ends moving") {
        const Limits lim{6.0, 3.0, 5.0};
        const auto   p = design::synthesize_trapezoidal(-2.0, 1.5, 9.0, -1.0, lim);
        REQUIRE(p.success);
        const auto r = verify(p);
        CHECK(r.pos_err_start < 1e-9);
        CHECK(r.vel_err_start < 1e-9);
        CHECK(r.pos_err_end < 1e-6);
        CHECK(r.vel_err_end < 1e-6);
        CHECK(r.max_speed <= 6.0 + 1e-6);
    }

    TEST_CASE("handbrake: initial speed exceeds Vmax toward target") {
        const Limits lim{2.0, 5.0, 5.0};
        const auto   p = design::synthesize_trapezoidal(0.0, 4.0, 10.0, 0.0, lim); // Vi=4 > Vmax=2
        REQUIRE(p.success);
        const auto r = verify(p);
        CHECK(r.pos_err_end < 1e-6);
        CHECK(r.vel_err_end < 1e-6);
        // it must brake from 4 down to the cruise speed: first phase decelerates
        CHECK(p.Ar < 0.0);
    }

    TEST_CASE("invalid limits and out-of-range Vf report failure") {
        CHECK_FALSE(design::synthesize_trapezoidal(0.0, 10.0, Limits{0.0, 5.0, 5.0}).success);
        CHECK_FALSE(design::synthesize_trapezoidal(0.0, 10.0, Limits{2.0, -1.0, 5.0}).success);
        CHECK_FALSE(design::synthesize_trapezoidal(0.0, 0.0, 1.0, 99.0, Limits{2.0, 5.0, 5.0}).success);
    }

    TEST_CASE("trivial move: already at target") {
        const Limits lim{2.0, 5.0, 5.0};
        const auto   p = design::synthesize_trapezoidal(3.0, 3.0, lim);
        REQUIRE(p.success);
        CHECK(p.Tf == doctest::Approx(0.0));
        CHECK(p.eval(0.0).position == doctest::Approx(3.0));
    }

    TEST_CASE("runtime step() walks the profile and reports done") {
        const Limits                 lim{2.0, 5.0, 5.0};
        const auto                   p = design::synthesize_trapezoidal(0.0, 10.0, lim).as<float>();
        TrapezoidalTrajectory<float> traj(p);
        REQUIRE(traj.valid());
        CHECK_FALSE(traj.done());
        const float dt = 1.0e-3f;
        for (int k = 0; k < 10000 && !traj.done(); ++k) {
            traj.step(dt);
        }
        CHECK(traj.done());
        CHECK(traj.eval(traj.duration()).position == doctest::Approx(10.0f).epsilon(0.001));
    }

    TEST_CASE("fuzz: random plans satisfy BCs, continuity, and limits") {
        // Simple LCG for reproducibility (no <random> dependence on platform).
        uint64_t seed = 0x1234567ull;
        auto     rnd = [&seed](double lo, double hi) {
            seed = (seed * 6364136223846793005ull) + 1442695040888963407ull;
            const double u = static_cast<double>(seed >> 11) / static_cast<double>(1ull << 53);
            return lo + ((hi - lo) * u);
        };
        for (int i = 0; i < 500; ++i) {
            const double Vmax = rnd(0.5, 10.0);
            const double Amax = rnd(0.5, 10.0);
            const double Dmax = rnd(0.5, 10.0);
            const Limits lim{Vmax, Amax, Dmax};
            const double Xi = rnd(-20.0, 20.0);
            const double Xf = rnd(-20.0, 20.0);
            const double Vi = rnd(-Vmax, Vmax);
            const double Vf = rnd(-Vmax, Vmax);
            const auto   p = design::synthesize_trapezoidal(Xi, Vi, Xf, Vf, lim);
            REQUIRE_MESSAGE(p.success, "i=" << i);
            const auto   r = verify(p);
            const double pos_tol = 1e-4 * (1.0 + std::abs(Xf - Xi));
            const double vel_tol = 1e-4 * (1.0 + Vmax);
            CHECK_MESSAGE(r.pos_err_start < 1e-7, "i=" << i);
            CHECK_MESSAGE(r.vel_err_start < 1e-7, "i=" << i);
            CHECK_MESSAGE(r.pos_err_end < pos_tol, "i=" << i);
            CHECK_MESSAGE(r.vel_err_end < vel_tol, "i=" << i);
            // continuity: per-step jumps stay small relative to the profile scale
            CHECK_MESSAGE(r.max_vel_jump < ((Amax + Dmax) * (p.Tf / 4000.0)) + 1e-6, "i=" << i);
            CHECK_MESSAGE(r.max_speed <= Vmax + 1e-4, "i=" << i);
            CHECK_MESSAGE(r.max_accel_phase <= Amax + 1e-4, "i=" << i);
            CHECK_MESSAGE(r.max_decel_phase <= Dmax + 1e-4, "i=" << i);
        }
    }

    // ---- Polynomial families (fixed-time, derivative-optimal) ----------------

    TEST_CASE("min-jerk quintic matches the Flash-Hogan closed form") {
        const double p0 = 2.0, pf = 9.0, T = 3.0;
        const auto   p = design::min_jerk(p0, pf, T);
        REQUIRE(p.success);
        // Boundary conditions: rest-to-rest.
        CHECK(p.eval(0.0).position == doctest::Approx(p0));
        CHECK(p.eval(0.0).velocity == doctest::Approx(0.0));
        CHECK(p.eval(0.0).acceleration == doctest::Approx(0.0));
        CHECK(p.eval(T).position == doctest::Approx(pf));
        CHECK(p.eval(T).velocity == doctest::Approx(0.0));
        CHECK(p.eval(T).acceleration == doctest::Approx(0.0));
        // Closed form: p0 + (pf-p0)(10τ³ - 15τ⁴ + 6τ⁵), τ = t/T.
        for (int k = 0; k <= 20; ++k) {
            const double t = T * k / 20.0;
            const double tau = t / T;
            const double ref = p0 + ((pf - p0) * ((10 * tau * tau * tau) - (15 * tau * tau * tau * tau) + (6 * tau * tau * tau * tau * tau)));
            CHECK(p.eval(t).position == doctest::Approx(ref).epsilon(1e-9));
        }
    }

    TEST_CASE("min-accel cubic matches the closed form and has zero end velocities") {
        const double p0 = -1.0, pf = 4.0, T = 2.0;
        const auto   p = design::min_accel(p0, pf, T);
        REQUIRE(p.success);
        CHECK(p.eval(0.0).velocity == doctest::Approx(0.0));
        CHECK(p.eval(T).velocity == doctest::Approx(0.0));
        CHECK(p.eval(T).position == doctest::Approx(pf));
        for (int k = 0; k <= 20; ++k) {
            const double t = T * k / 20.0;
            const double tau = t / T;
            const double ref = p0 + ((pf - p0) * ((3 * tau * tau) - (2 * tau * tau * tau)));
            CHECK(p.eval(t).position == doctest::Approx(ref).epsilon(1e-9));
        }
    }

    TEST_CASE("min-snap septic zeroes velocity, acceleration and jerk at both ends") {
        const double p0 = 0.0, pf = 5.0, T = 4.0;
        const auto   p = design::min_snap(p0, pf, T);
        REQUIRE(p.success);
        for (double t : {0.0, T}) {
            CHECK(p.eval(t).velocity == doctest::Approx(0.0).epsilon(1e-9));
            CHECK(p.eval(t).acceleration == doctest::Approx(0.0).epsilon(1e-9));
            CHECK(p.eval(t).jerk == doctest::Approx(0.0).epsilon(1e-9));
        }
        CHECK(p.eval(0.0).position == doctest::Approx(p0));
        CHECK(p.eval(T).position == doctest::Approx(pf));
    }

    TEST_CASE("general BVP honors arbitrary nonzero boundary derivatives") {
        using B = TrajectoryBoundary<double>;
        const B      start{1.0, 2.0, -0.5, 0.0}; // p, v, a, (j unused for quintic)
        const B      end{6.0, -1.0, 0.25, 0.0};
        const double T = 2.5;
        const auto   p = design::synthesize_poly_trajectory<5>(start, end, T);
        REQUIRE(p.success);
        const auto s0 = p.eval(0.0);
        const auto sf = p.eval(T);
        CHECK(s0.position == doctest::Approx(start.position));
        CHECK(s0.velocity == doctest::Approx(start.velocity));
        CHECK(s0.acceleration == doctest::Approx(start.acceleration));
        CHECK(sf.position == doctest::Approx(end.position));
        CHECK(sf.velocity == doctest::Approx(end.velocity));
        CHECK(sf.acceleration == doctest::Approx(end.acceleration));
    }

    TEST_CASE("velocity is the analytic derivative of position (finite difference)") {
        const auto p = design::min_jerk(0.0, 1.0, 1.5);
        REQUIRE(p.success);
        const double h = 1e-6;
        for (int k = 1; k < 20; ++k) {
            const double t = 1.5 * k / 20.0;
            const double fd = (p.eval(t + h).position - p.eval(t - h).position) / (2 * h);
            CHECK(p.eval(t).velocity == doctest::Approx(fd).epsilon(1e-5));
            const double fda = (p.eval(t + h).velocity - p.eval(t - h).velocity) / (2 * h);
            CHECK(p.eval(t).acceleration == doctest::Approx(fda).epsilon(1e-5));
        }
    }

    TEST_CASE("polynomial runtime walks the profile and clamps outside [0,T]") {
        const auto                     p = design::min_jerk(0.0, 10.0, 2.0).as<float>();
        PolynomialTrajectory<5, float> traj(p);
        REQUIRE(traj.valid());
        CHECK(traj.eval(-1.0f).position == doctest::Approx(0.0f));  // clamped to start
        CHECK(traj.eval(99.0f).position == doctest::Approx(10.0f)); // clamped to end
        const float dt = 1.0e-3f;
        for (int k = 0; k < 4000 && !traj.done(); ++k) {
            traj.step(dt);
        }
        CHECK(traj.done());
        CHECK(traj.eval(traj.duration()).position == doctest::Approx(10.0f).epsilon(0.001));
    }

    TEST_CASE("degenerate duration reports failure") {
        CHECK_FALSE(design::min_jerk(0.0, 1.0, 0.0).success);
        CHECK_FALSE(design::synthesize_poly_trajectory<3>(TrajectoryBoundary<double>{}, TrajectoryBoundary<double>{1.0}, -1.0).success);
    }

    // ---- Jerk-limited double-S (S-curve) -------------------------------------

    // Verify an S-curve profile by fine sampling: boundary conditions, that the
    // endpoint is reached, C² continuity (continuous acceleration, so the per-step
    // accel jump is bounded by Jmax·dt), and the v/a/jerk limits.
    struct ScReport {
        double pos_err_start{0}, vel_err_start{0}, pos_err_end{0}, vel_err_end{0};
        double accel_start{0}, accel_end{0};
        double max_speed{0}, max_accel{0}, max_jerk{0}, max_accel_jump{0};
    };
    auto verify_scurve = [](const auto& p) {
        ScReport     r{};
        const double Tf = p.duration;
        const int    N = 6000;
        const double dt = (Tf > 0.0) ? Tf / N : 0.0;
        const auto   s0 = p.eval(0.0);
        const auto   sf = p.eval(p.duration);
        r.pos_err_start = std::abs(s0.position - p.Xi);
        r.vel_err_start = std::abs(s0.velocity - p.Vi);
        r.pos_err_end = std::abs(sf.position - p.Xf);
        r.vel_err_end = std::abs(sf.velocity - p.Vf);
        double prev_a = p.eval(0.0).acceleration;
        for (int k = 1; k <= N; ++k) {
            const auto   st = p.eval(static_cast<double>(k * dt));
            const double aa = st.acceleration;
            r.max_speed = std::max(r.max_speed, std::abs(st.velocity));
            r.max_accel = std::max(r.max_accel, std::abs(aa));
            r.max_jerk = std::max(r.max_jerk, std::abs(st.jerk));
            r.max_accel_jump = std::max(r.max_accel_jump, std::abs(aa - prev_a));
            prev_a = aa;
        }
        r.accel_start = std::abs(s0.acceleration);
        r.accel_end = std::abs(static_cast<double>(p.eval(p.duration - 1e-9).acceleration));
        return r;
    };

    TEST_CASE("S-curve long move: reaches Vmax and Amax, C²") {
        const TrajectoryLimits<double> lim{2.0, 5.0, 5.0, 30.0};
        const auto                     p = design::synthesize_scurve(0.0, 10.0, lim);
        REQUIRE(p.success);
        const auto r = verify_scurve(p);
        CHECK(r.pos_err_end < 1e-5);
        CHECK(r.vel_err_end < 1e-5);
        CHECK(r.accel_start < 1e-6);
        CHECK(r.accel_end < 1e-4);
        CHECK(r.max_speed <= 2.0 + 1e-4);
        CHECK(r.max_accel <= 5.0 + 1e-4);
        CHECK(r.max_jerk <= 30.0 + 1e-4);
        CHECK(r.max_accel_jump < (30.0 * (p.duration / 6000.0)) + 1e-6);
    }

    TEST_CASE("S-curve converges to the trapezoidal as Jmax grows") {
        const TrajectoryLimits<double> lim_s{2.0, 5.0, 5.0, 5.0e5};
        const TrajectoryLimits<double> lim_t{2.0, 5.0, 5.0};
        const auto                     ps = design::synthesize_scurve(0.0, 10.0, lim_s);
        const auto                     pt = design::synthesize_trapezoidal(0.0, 10.0, lim_t);
        REQUIRE(ps.success);
        REQUIRE(pt.success);
        CHECK(ps.duration == doctest::Approx(pt.Tf).epsilon(0.005));
    }

    TEST_CASE("S-curve respects asymmetric accel vs decel limits") {
        const TrajectoryLimits<double> lim{3.0, 2.0, 8.0, 40.0}; // Amax=2, Dmax=8
        const auto                     p = design::synthesize_scurve(0.0, 20.0, lim);
        REQUIRE(p.success);
        double max_a_accel = 0.0, max_a_decel = 0.0;
        bool   peaked = false;
        double prev_v = 0.0;
        for (int k = 0; k <= 4000; ++k) {
            const auto   st = p.eval(p.duration * k / 4000.0);
            const double v = st.velocity;
            if (v + 1e-9 < prev_v) {
                peaked = true;
            }
            if (!peaked) {
                max_a_accel = std::max(max_a_accel, std::abs(st.acceleration));
            } else {
                max_a_decel = std::max(max_a_decel, std::abs(st.acceleration));
            }
            prev_v = v;
        }
        CHECK(max_a_accel <= 2.0 + 1e-3);
        CHECK(max_a_decel <= 8.0 + 1e-3);
        CHECK(max_a_decel > 2.0 + 1e-3); // decel really exceeds the accel cap
    }

    TEST_CASE("S-curve very short move never reaches Amax (triangular accel)") {
        const TrajectoryLimits<double> lim{10.0, 5.0, 5.0, 20.0};
        const auto                     p = design::synthesize_scurve(0.0, 0.05, lim);
        REQUIRE(p.success);
        const auto r = verify_scurve(p);
        CHECK(r.pos_err_end < 1e-5);
        CHECK(r.vel_err_end < 1e-5);
        CHECK(r.max_accel < 5.0); // plateau never reached
        CHECK(r.max_jerk <= 20.0 + 1e-4);
    }

    TEST_CASE("S-curve runtime walks the profile") {
        const TrajectoryLimits<double> lim{2.0, 5.0, 5.0, 30.0};
        const auto                     p = design::synthesize_scurve(0.0, 10.0, lim).as<float>();
        ScurveTrajectory<float>        traj(p);
        REQUIRE(traj.valid());
        CHECK_FALSE(traj.done());
        for (int k = 0; k < 20000 && !traj.done(); ++k) {
            traj.step(1.0e-3f);
        }
        CHECK(traj.done());
        CHECK(traj.eval(traj.duration()).position == doctest::Approx(10.0f).epsilon(0.001));
    }

    TEST_CASE("S-curve invalid limits report failure") {
        CHECK_FALSE(design::synthesize_scurve(0.0, 10.0, TrajectoryLimits<double>{2.0, 5.0, 5.0, 0.0}).success);
        CHECK_FALSE(
            design::synthesize_scurve(0.0, 0.0, 1.0, 99.0, TrajectoryLimits<double>{2.0, 5.0, 5.0, 30.0}).success
        );
    }

    TEST_CASE("S-curve fuzz: arbitrary Vi/Vf, asymmetric limits, C² and bounded") {
        uint64_t seed = 0xC0FFEEull;
        auto     rnd = [&seed](double lo, double hi) {
            seed = (seed * 6364136223846793005ull) + 1442695040888963407ull;
            const double u = static_cast<double>(seed >> 11) / static_cast<double>(1ull << 53);
            return lo + ((hi - lo) * u);
        };
        for (int i = 0; i < 400; ++i) {
            const double                   Vmax = rnd(0.5, 10.0);
            const double                   Amax = rnd(0.5, 10.0);
            const double                   Dmax = rnd(0.5, 10.0);
            const double                   Jmax = rnd(1.0, 50.0);
            const TrajectoryLimits<double> lim{Vmax, Amax, Dmax, Jmax};
            const double                   Xi = rnd(-20.0, 20.0);
            const double                   Xf = rnd(-20.0, 20.0);
            const double                   Vi = rnd(-Vmax, Vmax);
            const double                   Vf = rnd(-Vmax, Vmax);
            const auto                     p = design::synthesize_scurve(Xi, Vi, Xf, Vf, lim);
            REQUIRE_MESSAGE(p.success, "i=" << i);
            const auto   r = verify_scurve(p);
            const double pos_tol = 1e-3 * (1.0 + std::abs(Xf - Xi));
            CHECK_MESSAGE(r.pos_err_start < 1e-7, "i=" << i);
            CHECK_MESSAGE(r.vel_err_start < 1e-7, "i=" << i);
            CHECK_MESSAGE(r.pos_err_end < pos_tol, "i=" << i);
            CHECK_MESSAGE(r.vel_err_end < 1e-3 * (1.0 + Vmax), "i=" << i);
            CHECK_MESSAGE(r.max_speed <= Vmax + 1e-3, "i=" << i);
            CHECK_MESSAGE(r.max_accel <= std::max(Amax, Dmax) + 1e-3, "i=" << i);
            CHECK_MESSAGE(r.max_jerk <= Jmax + 1e-3, "i=" << i);
            CHECK_MESSAGE(r.max_accel_jump < (Jmax * (p.duration / 6000.0)) + 1e-4, "i=" << i);
        }
    }

    // ---- Multi-axis coordination bank ----------------------------------------

    TEST_CASE("bank: axes with different move lengths arrive synchronized") {
        const Limits lim{2.0, 5.0, 5.0};
        // Three rest-to-rest moves of increasing distance -> increasing min-time.
        wet::array<TrapezoidalTrajectory<double>, 3> axes{
            TrapezoidalTrajectory<double>(design::synthesize_trapezoidal(0.0, 2.0, lim)),
            TrapezoidalTrajectory<double>(design::synthesize_trapezoidal(0.0, 6.0, lim)),
            TrapezoidalTrajectory<double>(design::synthesize_trapezoidal(0.0, 12.0, lim)),
        };
        TrajectoryBank<3, TrapezoidalTrajectory<double>> bank(axes);
        REQUIRE(bank.valid());

        // T_sync is the slowest axis's native duration; that axis runs unscaled.
        const double slowest = axes[2].duration();
        CHECK(bank.duration() == doctest::Approx(slowest));
        CHECK(bank.scale(2) == doctest::Approx(1.0));
        CHECK(bank.scale(0) < bank.scale(1));
        CHECK(bank.scale(1) < bank.scale(2));

        // All axes reach their endpoints exactly at T_sync, and none before.
        const auto   mid = bank.eval(bank.duration() * 0.5);
        const auto   end = bank.eval(bank.duration());
        const double targets[3] = {2.0, 6.0, 12.0};
        for (size_t i = 0; i < 3; ++i) {
            CHECK(end[i].position == doctest::Approx(targets[i]));
            CHECK(end[i].velocity == doctest::Approx(0.0));
            CHECK(std::abs(mid[i].position - targets[i]) > 1e-3); // still moving at the midpoint
        }
    }

    TEST_CASE("bank: time-scaling keeps every axis within its own limits") {
        const Limits                                 lim{2.0, 5.0, 5.0};
        wet::array<TrapezoidalTrajectory<double>, 2> axes{
            TrapezoidalTrajectory<double>(design::synthesize_trapezoidal(0.0, 1.0, lim)),  // fast (short)
            TrapezoidalTrajectory<double>(design::synthesize_trapezoidal(0.0, 20.0, lim)), // slow (long)
        };
        TrajectoryBank<2, TrapezoidalTrajectory<double>> bank(axes);
        REQUIRE(bank.valid());

        double    max_v0 = 0.0, max_a0 = 0.0;
        const int N = 2000;
        for (int k = 0; k <= N; ++k) {
            const auto st = bank.eval(bank.duration() * k / N);
            max_v0 = std::max(max_v0, std::abs(st[0].velocity));
            max_a0 = std::max(max_a0, std::abs(st[0].acceleration));
        }
        // The short axis is heavily slowed -> peak v/a well under the limits.
        CHECK(max_v0 <= 2.0 + 1e-9);
        CHECK(max_a0 <= 5.0 + 1e-9);
        CHECK(max_v0 < 2.0); // genuinely throttled, not just at the cap
    }

    TEST_CASE("bank: scaled velocity is the analytic derivative of scaled position") {
        const Limits                                 lim{3.0, 4.0, 4.0};
        wet::array<TrapezoidalTrajectory<double>, 2> axes{
            TrapezoidalTrajectory<double>(design::synthesize_trapezoidal(0.0, 5.0, lim)),
            TrapezoidalTrajectory<double>(design::synthesize_trapezoidal(0.0, 15.0, lim)),
        };
        TrajectoryBank<2, TrapezoidalTrajectory<double>> bank(axes);
        const double                                     h = 1e-6;
        for (int k = 1; k < 40; ++k) {
            const double t = bank.duration() * k / 40.0;
            const auto   sp = bank.eval(t + h);
            const auto   sm = bank.eval(t - h);
            const auto   s = bank.eval(t);
            for (size_t i = 0; i < 2; ++i) {
                const double fd = (sp[i].position - sm[i].position) / (2 * h);
                CHECK(s[i].velocity == doctest::Approx(fd).epsilon(1e-4));
            }
        }
    }

    TEST_CASE("bank: a stationary axis holds while the others move") {
        const Limits                                 lim{2.0, 5.0, 5.0};
        wet::array<TrapezoidalTrajectory<double>, 2> axes{
            TrapezoidalTrajectory<double>(design::synthesize_trapezoidal(3.0, 3.0, lim)), // no move
            TrapezoidalTrajectory<double>(design::synthesize_trapezoidal(0.0, 8.0, lim)), // moves
        };
        TrajectoryBank<2, TrapezoidalTrajectory<double>> bank(axes);
        REQUIRE(bank.valid());
        CHECK(bank.scale(0) == doctest::Approx(0.0)); // zero-duration axis
        for (int k = 0; k <= 10; ++k) {
            const auto st = bank.eval(bank.duration() * k / 10.0);
            CHECK(st[0].position == doctest::Approx(3.0)); // held
            CHECK(st[0].velocity == doctest::Approx(0.0));
        }
        CHECK(bank.eval(bank.duration())[1].position == doctest::Approx(8.0));
    }

    TEST_CASE("bank: S-curve axes synchronize and step() walks to done") {
        const TrajectoryLimits<double>          lim{2.0, 5.0, 5.0, 30.0};
        wet::array<ScurveTrajectory<double>, 2> axes{
            ScurveTrajectory<double>(design::synthesize_scurve(0.0, 4.0, lim)),
            ScurveTrajectory<double>(design::synthesize_scurve(0.0, 13.0, lim)),
        };
        TrajectoryBank<2, ScurveTrajectory<double>> bank(axes);
        REQUIRE(bank.valid());
        CHECK(bank.duration() == doctest::Approx(axes[1].duration())); // slowest

        int guard = 0;
        while (!bank.done() && guard++ < 100000) {
            bank.step(1.0e-3);
        }
        CHECK(bank.done());
        const auto end = bank.eval(bank.duration());
        CHECK(end[0].position == doctest::Approx(4.0).epsilon(1e-4));
        CHECK(end[1].position == doctest::Approx(13.0).epsilon(1e-4));
    }

    TEST_CASE("bank: invalid if any axis failed to plan") {
        const Limits                                 lim{2.0, 5.0, 5.0};
        wet::array<TrapezoidalTrajectory<double>, 2> axes{
            TrapezoidalTrajectory<double>(design::synthesize_trapezoidal(0.0, 5.0, lim)),
            TrapezoidalTrajectory<double>(design::synthesize_trapezoidal(0.0, 5.0, Limits{0.0, 5.0, 5.0})), // bad Vmax
        };
        TrajectoryBank<2, TrapezoidalTrajectory<double>> bank(axes);
        CHECK_FALSE(bank.valid());
    }
}
