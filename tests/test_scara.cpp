#include <cmath>
#include <cstddef>
#include <cstdint>

#include "wet/backend.hpp"
#include "wet/kinematics/pose.hpp"
#include "wet/kinematics/scara.hpp"
#include "wet/kinematics/serial_arm.hpp"
#include "wet/math/geometry.hpp"
#include "wet/matrix/matrix.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wet;

namespace {

constexpr double pi = 3.14159265358979323846;

struct Rng {
    uint64_t s{0xC3A5C85C97CB3127ull};
    double   operator()(double lo, double hi) {
        s = (s * 6364136223846793005ull) + 1442695040888963407ull;
        const double u = static_cast<double>(s >> 11) / static_cast<double>(1ull << 53);
        return lo + ((hi - lo) * u);
    }
};

} // namespace

TEST_SUITE("scara") {

    // ---- Prismatic joint support in SerialArm --------------------------------

    TEST_CASE("prismatic joint: forward kinematics translates along its axis") {
        // A single prismatic joint along z: q maps directly to a z-translation.
        DhChain<1, double> c{};
        c.joints[0] = {0.0, 0.0, 0.2, 0.0, 0.0, 1.0, JointType::Prismatic};
        const SerialArm<1, double> arm(c);

        const Pose<double> p0 = arm.forward(wet::array<double, 1>{0.0});
        const Pose<double> p1 = arm.forward(wet::array<double, 1>{0.5});
        CHECK(p0.translation[2] == doctest::Approx(0.2)); // base d only
        CHECK(p1.translation[2] == doctest::Approx(0.7)); // d + q
        // A prismatic joint adds no rotation.
        CHECK(std::abs(p1.orientation.w()) == doctest::Approx(1.0));
    }

    TEST_CASE("prismatic joint: Jacobian column is pure translation [z; 0]") {
        DhChain<2, double> c{};
        c.joints[0] = {0.3, 0.0, 0.0, 0.0, -pi, pi, JointType::Revolute};
        c.joints[1] = {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, JointType::Prismatic};
        const SerialArm<2, double>  arm(c);
        const wet::array<double, 2> q{0.4, 0.25};
        const auto                  J = arm.jacobian(q);

        // Column 1 (prismatic, axis z aligned with base z): [0,0,1, 0,0,0].
        CHECK(J(0, 1) == doctest::Approx(0.0));
        CHECK(J(1, 1) == doctest::Approx(0.0));
        CHECK(J(2, 1) == doctest::Approx(1.0));
        CHECK(J(3, 1) == doctest::Approx(0.0));
        CHECK(J(4, 1) == doctest::Approx(0.0));
        CHECK(J(5, 1) == doctest::Approx(0.0));

        // Cross-check the whole Jacobian against a finite difference.
        const double h = 1e-6;
        for (size_t j = 0; j < 2; ++j) {
            wet::array<double, 2> qp = q;
            wet::array<double, 2> qm = q;
            qp[j] += h;
            qm[j] -= h;
            const auto pp = arm.forward(qp);
            const auto pm = arm.forward(qm);
            CHECK(J(0, j) == doctest::Approx((pp.translation[0] - pm.translation[0]) / (2 * h)).epsilon(1e-4));
            CHECK(J(1, j) == doctest::Approx((pp.translation[1] - pm.translation[1]) / (2 * h)).epsilon(1e-4));
            CHECK(J(2, j) == doctest::Approx((pp.translation[2] - pm.translation[2]) / (2 * h)).epsilon(1e-4));
        }
    }

    TEST_CASE("regression: DhJoint defaults to revolute (6-field aggregate init)") {
        // Adding the `type` field must not change existing all-revolute chains:
        // a 6-field aggregate init (no type) stays JointType::Revolute.
        constexpr DhJoint<double> def{};
        static_assert(def.type == JointType::Revolute, "default joint must be revolute");
        constexpr DhJoint<double> legacy{0.3, 0.0, 0.1, 0.0, -3.0, 3.0}; // pre-`type` form
        static_assert(legacy.type == JointType::Revolute, "6-field init must be revolute");

        // A revolute joint drives θ (rotation), not d (translation).
        DhChain<1, double> c{};
        c.joints[0] = {0.0, 0.0, 0.0, 0.0, -3.0, 3.0}; // legacy 6-field
        const SerialArm<1, double> arm(c);
        const Pose<double>         p = arm.forward(wet::array<double, 1>{0.7});
        CHECK(p.translation[2] == doctest::Approx(0.0)); // d unchanged by q
        const auto e = p.orientation.template to_euler<EulerOrder::ZYX>();
        CHECK(e.angle1 == doctest::Approx(0.7)); // q drove yaw
        // .as<U>() carries the joint type across a rebind.
        CHECK(c.as<float>().joints[0].type == JointType::Revolute);
    }

    // ---- Series SCARA (RRPR) -------------------------------------------------

    TEST_CASE("scara_arm builder: forward kinematics matches the closed form") {
        constexpr auto cfg = design::scara_arm<double>(0.4, 0.3, 0.5, 0.2, 0.05);
        static_assert(cfg.success, "SCARA chain must validate at compile time");
        CHECK(cfg.success);
        CHECK_FALSE(cfg.spherical_wrist); // N = 4, not a 6-axis spherical wrist

        const SerialArm<4, double>  arm(cfg.chain);
        const wet::array<double, 4> q{0.3, -0.5, 0.1, 0.2}; // shoulder, elbow, Z, wrist
        const Pose<double>          p = arm.forward(q);

        CHECK(p.translation[0] == doctest::Approx((0.4 * std::cos(0.3)) + (0.3 * std::cos(0.3 - 0.5))));
        CHECK(p.translation[1] == doctest::Approx((0.4 * std::sin(0.3)) + (0.3 * std::sin(0.3 - 0.5))));
        CHECK(p.translation[2] == doctest::Approx(0.5 + 0.1 + 0.05)); // base + stroke + tool
        // End-effector yaw is the sum of the three revolute angles.
        const auto e = p.orientation.template to_euler<EulerOrder::ZYX>();
        CHECK(e.angle1 == doctest::Approx(0.3 - 0.5 + 0.2));
    }

    TEST_CASE("scara_arm: position+yaw IK round-trips forward") {
        const SerialArm<4, double> arm(design::scara_arm<double>(0.4, 0.3, 0.5, 0.3, 0.0).chain);
        Rng                        rng;
        int                        ok = 0;
        for (int trial = 0; trial < 80; ++trial) {
            const wet::array<double, 4> q{rng(-1.2, 1.2), rng(-1.5, 1.5), rng(0.05, 0.25), rng(-1.0, 1.0)};
            const Pose<double>          target = arm.forward(q);
            wet::array<double, 4>       seed = q;
            for (auto& v : seed) {
                v += rng(-0.2, 0.2);
            }
            const auto sol = arm.inverse(target, seed, task_position_yaw, 200, 0.03, 1e-10);
            if (!sol.converged) {
                continue;
            }
            ++ok;
            const Pose<double> got = arm.forward(sol.joints);
            CHECK(got.translation.distance(target.translation) < 1e-6);
        }
        CHECK(ok > 60);
    }

    // ---- Parallel SCARA (five-bar) ------------------------------------------

    TEST_CASE("five_bar: forward round-trips inverse across the workspace") {
        const auto fb = design::five_bar_symmetric<double>(0.2, 0.15, 0.2);
        REQUIRE(fb.valid());

        int    n = 0;
        double worst = 0.0;
        for (double x = -0.08; x <= 0.08 + 1e-9; x += 0.02) {
            for (double y = 0.16; y <= 0.26 + 1e-9; y += 0.02) {
                const auto inv = fb.inverse(x, y);
                if (!inv.reachable) {
                    continue;
                }
                const auto fwd = fb.forward(inv.angles[0], inv.angles[1]);
                REQUIRE(fwd.valid);
                worst = std::max(worst, std::hypot(fwd.point[0] - x, fwd.point[1] - y));
                ++n;
            }
        }
        CHECK(n > 20);
        CHECK(worst < 1e-12);
    }

    TEST_CASE("five_bar: out-of-reach target is flagged unreachable") {
        const auto fb = design::five_bar_symmetric<double>(0.2, 0.15, 0.2);
        // Far outside the reach of either arm (max ≈ proximal + distal = 0.35).
        const auto inv = fb.inverse(0.0, 2.0);
        CHECK_FALSE(inv.reachable);
    }

    TEST_CASE("five_bar: velocity Jacobian matches a finite difference") {
        const auto fb = design::five_bar_symmetric<double>(0.2, 0.15, 0.2);
        const auto inv = fb.inverse(0.02, 0.22);
        REQUIRE(inv.reachable);
        const double a1 = inv.angles[0];
        const double a2 = inv.angles[1];
        const auto   J = fb.jacobian(a1, a2);

        const double h = 1e-6;
        // Column j: ∂P/∂θ_j.
        const auto p_a1p = fb.forward(a1 + h, a2).point;
        const auto p_a1m = fb.forward(a1 - h, a2).point;
        const auto p_a2p = fb.forward(a1, a2 + h).point;
        const auto p_a2m = fb.forward(a1, a2 - h).point;
        CHECK(J(0, 0) == doctest::Approx((p_a1p[0] - p_a1m[0]) / (2 * h)).epsilon(1e-4));
        CHECK(J(1, 0) == doctest::Approx((p_a1p[1] - p_a1m[1]) / (2 * h)).epsilon(1e-4));
        CHECK(J(0, 1) == doctest::Approx((p_a2p[0] - p_a2m[0]) / (2 * h)).epsilon(1e-4));
        CHECK(J(1, 1) == doctest::Approx((p_a2p[1] - p_a2m[1]) / (2 * h)).epsilon(1e-4));
    }

    TEST_CASE("five_bar: as<float> geometry and a degenerate build") {
        const auto fb = design::five_bar_symmetric<float>(0.2F, 0.15F, 0.2F);
        CHECK(fb.valid());
        const auto inv = fb.inverse(0.0F, 0.22F);
        CHECK(inv.reachable);

        const FiveBar<double> bad(FiveBarGeometry<double>{0.0, 0.15, 0.2});
        CHECK_FALSE(bad.valid());
    }
}
