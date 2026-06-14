#include <cmath>
#include <cstdint>

#include "wet/kinematics/stewart.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wet;

namespace {

// A reachable symmetric 6-6 rig used across the cases. The base and platform
// half-angles differ (0.25 vs 0.55) so the two hexagons are *not* similar —
// similar polygons make the home pose a Stewart singularity (rank-deficient
// Jacobian), which the dedicated singular-Jacobian case below exercises.
constexpr StewartConfig<double> make_rig() {
    return design::stewart_symmetric<double>(
        1.0,
        0.6,
        0.25,
        0.55,
        1.2,
        0.5,
        2.5
    );
}

Pose<double> make_pose(double x, double y, double z, double roll, double pitch, double yaw) {
    Pose<double> p;
    p.translation = Translation3<double>(x, y, z);
    const EulerXYZ<double> e{roll, pitch, yaw};
    p.orientation = Quaternion<double>::from_euler(e);
    return p;
}

// Reproducible LCG.
struct Rng {
    uint64_t s{0x243F6A8885A308D3ull};
    double   operator()(double lo, double hi) {
        s = (s * 6364136223846793005ull) + 1442695040888963407ull;
        const double u = static_cast<double>(s >> 11) / static_cast<double>(1ull << 53);
        return lo + ((hi - lo) * u);
    }
};

} // namespace

TEST_SUITE("stewart") {

    TEST_CASE("synthesize: valid symmetric rig reports success, home is reachable") {
        constexpr auto cfg = make_rig();
        static_assert(cfg.success, "symmetric home pose must be reachable at compile time");
        CHECK(cfg.success);

        const StewartPlatform<double> rig(cfg);

        Pose<double> home;
        home.translation = Translation3<double>(0.0, 0.0, cfg.geometry.home_height);
        const auto inv = rig.inverse(home);
        CHECK(inv.reachable);

        // The layout has 3-fold symmetry, so the legs come in two alternating
        // groups: {0,2,4} share one length and {1,3,5} share another.
        for (size_t i = 2; i < kStewartLegs; i += 2) {
            CHECK(inv.lengths[i] == doctest::Approx(inv.lengths[0]).epsilon(1e-12));
        }
        for (size_t i = 3; i < kStewartLegs; i += 2) {
            CHECK(inv.lengths[i] == doctest::Approx(inv.lengths[1]).epsilon(1e-12));
        }
    }

    TEST_CASE("synthesize: rejects degenerate / mis-ordered geometry") {
        StewartGeometry<double> g{};
        g.stroke_min = 2.0;
        g.stroke_max = 1.0; // inverted window
        CHECK_FALSE(design::synthesize_stewart(g).success);

        // Coincident base anchors.
        auto good = make_rig().geometry;
        good.base[3] = good.base[2];
        CHECK_FALSE(design::synthesize_stewart(good).success);

        // Valid geometry but home unreachable (stroke window too tight).
        auto tight = make_rig().geometry;
        tight.stroke_min = 1.19;
        tight.stroke_max = 1.21;
        CHECK_FALSE(design::synthesize_stewart(tight).success);
    }

    TEST_CASE("inverse: reproduces leg lengths by direct formula Lᵢ = ‖t + R·pᵢ − bᵢ‖") {
        const auto                    cfg = make_rig();
        const StewartPlatform<double> rig(cfg);
        const Pose<double>            pose = make_pose(0.05, -0.03, 1.25, 0.04, -0.02, 0.06);
        const auto                    inv = rig.inverse(pose);

        for (size_t i = 0; i < kStewartLegs; ++i) {
            const Vec3<double> lvec = static_cast<const Vec3<double>&>(pose.translation) + pose.orientation.rotate(cfg.geometry.platform[i]) - cfg.geometry.base[i];
            CHECK(inv.lengths[i] == doctest::Approx(lvec.norm()).epsilon(1e-12));
        }
        CHECK(inv.reachable);
    }

    TEST_CASE("inverse: flags an out-of-stroke pose") {
        const StewartPlatform<double> rig(make_rig());
        const Pose<double>            high = make_pose(0.0, 0.0, 3.0, 0.0, 0.0, 0.0);
        CHECK_FALSE(rig.inverse(high).reachable);
    }

    TEST_CASE("forward: Newton round-trips inverse across the workspace") {
        const auto                    cfg = make_rig();
        const StewartPlatform<double> rig(cfg);

        Pose<double> home;
        home.translation = Translation3<double>(0.0, 0.0, cfg.geometry.home_height);

        Rng rng;
        int converged = 0;
        for (int trial = 0; trial < 200; ++trial) {
            const Pose<double> pose = make_pose(rng(-0.1, 0.1), rng(-0.1, 0.1), rng(1.1, 1.35), rng(-0.08, 0.08), rng(-0.08, 0.08), rng(-0.1, 0.1));
            const auto         inv = rig.inverse(pose);
            if (!inv.reachable) {
                continue;
            }
            const auto fwd = rig.forward(inv.lengths, home, 50, 1e-9);
            REQUIRE(fwd.converged);
            ++converged;

            // Pose recovered to numerical tolerance.
            CHECK(fwd.pose.translation.distance(pose.translation) < 1e-7);
            // Inverse of the recovered pose reproduces the measured lengths.
            const auto check = rig.inverse(fwd.pose);
            for (size_t i = 0; i < kStewartLegs; ++i) {
                CHECK(check.lengths[i] == doctest::Approx(inv.lengths[i]).epsilon(1e-9));
            }
        }
        CHECK(converged > 150); // the vast majority of sampled poses are reachable
    }

    TEST_CASE("forward: warm-started from the previous solution converges fast") {
        const StewartPlatform<double> rig(make_rig());
        const Pose<double>            pose = make_pose(0.02, 0.01, 1.22, 0.01, 0.01, 0.02);
        const auto                    inv = rig.inverse(pose);
        // Seed at the true pose: it is already converged on the first residual check.
        const auto fwd = rig.forward(inv.lengths, pose, 3, 1e-9);
        CHECK(fwd.converged);
        CHECK(fwd.residual < 1e-9);
    }

    TEST_CASE("jacobian: matches a finite-difference of leg lengths") {
        const StewartPlatform<double> rig(make_rig());
        const Pose<double>            pose = make_pose(0.03, -0.02, 1.23, 0.03, -0.02, 0.04);
        const auto                    J = rig.jacobian(pose);
        const auto                    base_len = rig.inverse(pose).lengths;

        const double h = 1e-6;
        // Translation columns (0..2): perturb t along each axis.
        for (size_t axis = 0; axis < 3; ++axis) {
            Pose<double> pp = pose;
            Vec3<double> dt{};
            dt[axis] = h;
            pp.translation = Translation3<double>(static_cast<const Vec3<double>&>(pose.translation) + dt);
            const auto len = rig.inverse(pp).lengths;
            for (size_t i = 0; i < kStewartLegs; ++i) {
                const double fd = (len[i] - base_len[i]) / h;
                CHECK(J(i, axis) == doctest::Approx(fd).epsilon(1e-4));
            }
        }
        // Rotation columns (3..5): central difference about a small world-frame
        // rotation (a single-sided 1e-6 step underflows from_axis_angle's eps).
        const double hr = 1e-4;
        for (size_t axis = 0; axis < 3; ++axis) {
            Vec3<double> w{};
            w[axis] = hr;
            const auto dq_p = Quaternion<double>::from_axis_angle(w, hr).value();
            w[axis] = -hr;
            const auto   dq_m = Quaternion<double>::from_axis_angle(w, hr).value();
            Pose<double> pp = pose;
            Pose<double> pm = pose;
            pp.orientation = (dq_p * pose.orientation).normalized();
            pm.orientation = (dq_m * pose.orientation).normalized();
            const auto lp = rig.inverse(pp).lengths;
            const auto lm = rig.inverse(pm).lengths;
            for (size_t i = 0; i < kStewartLegs; ++i) {
                const double fd = (lp[i] - lm[i]) / (2.0 * hr);
                CHECK(J(i, 3 + axis) == doctest::Approx(fd).epsilon(1e-4));
            }
        }
    }

    TEST_CASE("forward: reports failure on a singular Jacobian") {
        // Similar base/platform hexagons (equal half-angles) make the neutral pose
        // a Stewart singularity: the actuator Jacobian is rank-deficient there, so
        // a Newton solve seeded exactly at home must report converged == false.
        const auto cfg = design::stewart_symmetric<double>(1.0, 0.6, 0.3, 0.3, 1.2, 0.5, 2.5);
        REQUIRE(cfg.success); // geometry/home are still *reachable* — only the Jacobian is singular
        const StewartPlatform<double> rig(cfg);

        Pose<double> home;
        home.translation = Translation3<double>(0.0, 0.0, 1.2);
        auto target = rig.inverse(home).lengths;
        // Nudge the targets so the residual at the seed is nonzero: the very first
        // Newton step must factor the singular home Jacobian and bail out.
        for (auto& L : target) {
            L += 0.01;
        }
        const auto fwd = rig.forward(target, home, 10, 1e-12);
        CHECK_FALSE(fwd.converged);
    }

    TEST_CASE("as<float>: rebinds the configuration") {
        const auto cfg = make_rig();
        const auto cfgf = cfg.as<float>();
        CHECK(cfgf.success == cfg.success);
        CHECK(cfgf.geometry.home_height == doctest::Approx(static_cast<float>(cfg.geometry.home_height)));

        const StewartPlatform<float> rigf(cfgf);
        Pose<float>                  home;
        home.translation = Translation3<float>(0.0F, 0.0F, cfgf.geometry.home_height);
        CHECK(rigf.inverse(home).reachable);
    }
}
