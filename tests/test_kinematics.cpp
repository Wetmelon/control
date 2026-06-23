#include <cmath>
#include <cstddef>
#include <cstdint>

#include "wet/backend.hpp"
#include "wet/kinematics/motion_maps.hpp"
#include "wet/kinematics/pose.hpp"
#include "wet/math/geometry.hpp"
#include "wet/matrix/colvec.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wet;

namespace {

// Reproducible LCG (no <random> dependence on platform).
struct Rng {
    uint64_t s{0x9E3779B97F4A7C15ull};
    double   operator()(double lo, double hi) {
        s = (s * 6364136223846793005ull) + 1442695040888963407ull;
        const double u = static_cast<double>(s >> 11) / static_cast<double>(1ull << 53);
        return lo + ((hi - lo) * u);
    }
};

constexpr double pi = 3.14159265358979323846;

} // namespace

TEST_SUITE("kinematics") {

    // ---- Pose foundation -----------------------------------------------------

    TEST_CASE("Pose: identity, compose, and inverse round-trip") {
        const auto   q = Quaternion<double>::from_axis_angle(Vec3<double>{0.0, 0.0, 1.0}, 0.7).value();
        Pose<double> p;
        p.translation = Translation3<double>(1.0, -2.0, 3.0);
        p.orientation = q;

        const Pose<double> id = p * p.inverse();
        CHECK(id.translation.norm() < 1e-12);
        CHECK(std::abs(std::abs(id.orientation.w()) - 1.0) < 1e-12);

        // identity is the neutral element
        const Pose<double> p2 = p * Pose<double>::identity();
        CHECK(p2.translation.distance(p.translation) < 1e-12);
    }

    TEST_CASE("Pose: transform_point matches translate-then-rotate, both ways") {
        const auto   q = Quaternion<double>::from_axis_angle(Vec3<double>{1.0, 0.0, 0.0}, pi / 2.0).value();
        Pose<double> p;
        p.translation = Translation3<double>(0.0, 0.0, 5.0);
        p.orientation = q;
        // 90° about x: (0,1,0) -> (0,0,1); then + translation
        const Vec3<double> out = p.transform_point(Vec3<double>{0.0, 1.0, 0.0});
        CHECK(out[0] == doctest::Approx(0.0));
        CHECK(out[1] == doctest::Approx(0.0));
        CHECK(out[2] == doctest::Approx(6.0));
        // round-trip through the inverse pose
        const Vec3<double> back = p.inverse().transform_point(out);
        CHECK(back[0] == doctest::Approx(0.0));
        CHECK(back[1] == doctest::Approx(1.0));
        CHECK(back[2] == doctest::Approx(0.0));
    }

    TEST_CASE("Pose: Transform4 export/import round-trips") {
        const auto   q = Quaternion<double>::from_axis_angle(Vec3<double>{0.3, 0.5, 0.8}, 1.1).value().normalized();
        Pose<double> p;
        p.translation = Translation3<double>(2.0, 3.0, -1.0);
        p.orientation = q;
        const Pose<double> r = Pose<double>::from_transform4(p.to_transform4());
        CHECK(r.translation.distance(p.translation) < 1e-9);
        // quaternion equal up to sign
        const double dotq = (r.orientation.w() * q.w()) + (r.orientation.x() * q.x()) + (r.orientation.y() * q.y()) + (r.orientation.z() * q.z());
        CHECK(std::abs(std::abs(dotq) - 1.0) < 1e-9);
    }

    // ---- CoreXY / polar / Cartesian ------------------------------------------

    TEST_CASE("CoreXY: belt sum/difference and round-trip") {
        const auto m = CoreXY<double>::inverse(3.0, 1.0);
        CHECK(m.a == doctest::Approx(4.0));
        CHECK(m.b == doctest::Approx(2.0));
        const auto pt = CoreXY<double>::forward(m.a, m.b);
        CHECK(pt.x == doctest::Approx(3.0));
        CHECK(pt.y == doctest::Approx(1.0));
    }

    TEST_CASE("Polar: known point and round-trip") {
        const auto ax = PolarMap<double>::inverse(0.0, 2.0);
        CHECK(ax.r == doctest::Approx(2.0));
        CHECK(ax.theta == doctest::Approx(pi / 2.0));
        const auto pt = PolarMap<double>::forward(ax.r, ax.theta);
        CHECK(pt.x == doctest::Approx(0.0));
        CHECK(pt.y == doctest::Approx(2.0));
    }

    TEST_CASE("Cartesian: per-axis affine round-trip") {
        CartesianMap<3, double> m{};
        m.scale = {2.0, 0.5, -1.0};
        m.offset = {10.0, -3.0, 0.0};
        const wet::array<double, 3> act{4.0, 8.0, 5.0};
        const auto                  task = m.forward(act);
        CHECK(task[0] == doctest::Approx(18.0));
        CHECK(task[1] == doctest::Approx(1.0));
        CHECK(task[2] == doctest::Approx(-5.0));
        const auto back = m.inverse(task);
        for (size_t i = 0; i < 3; ++i) {
            CHECK(back[i] == doctest::Approx(act[i]));
        }
    }

    // ---- Rotary delta --------------------------------------------------------

    TEST_CASE("rotary delta: forward->inverse round-trips over reachable angles") {
        const RotaryDelta<double> d(RotaryDeltaGeometry<double>{200.0, 80.0, 80.0, 280.0});
        REQUIRE(d.valid());
        Rng rng;
        int tested = 0;
        for (int i = 0; i < 2000; ++i) {
            const double t1 = rng(0.15, 0.9);
            const double t2 = rng(0.15, 0.9);
            const double t3 = rng(0.15, 0.9);
            const auto   fk = d.forward(t1, t2, t3);
            if (!fk.valid) {
                continue;
            }
            const auto ik = d.inverse(fk.pose.translation[0], fk.pose.translation[1], fk.pose.translation[2]);
            if (!ik.reachable) {
                continue;
            }
            CHECK(ik.actuators[0] == doctest::Approx(t1).epsilon(1e-6));
            CHECK(ik.actuators[1] == doctest::Approx(t2).epsilon(1e-6));
            CHECK(ik.actuators[2] == doctest::Approx(t3).epsilon(1e-6));
            ++tested;
        }
        REQUIRE(tested > 100); // the chosen geometry/range is mostly reachable
    }

    TEST_CASE("rotary delta: symmetric angles give an on-axis point") {
        const RotaryDelta<double> d(RotaryDeltaGeometry<double>{200.0, 80.0, 80.0, 280.0});
        const auto                fk = d.forward(0.5, 0.5, 0.5);
        REQUIRE(fk.valid);
        CHECK(fk.pose.translation[0] == doctest::Approx(0.0).epsilon(1e-9));
        CHECK(fk.pose.translation[1] == doctest::Approx(0.0).epsilon(1e-9));
        CHECK(fk.pose.translation[2] < 0.0); // platform hangs below the base
    }

    TEST_CASE("rotary delta: far-out target is unreachable") {
        const RotaryDelta<double> d(RotaryDeltaGeometry<double>{200.0, 80.0, 80.0, 280.0});
        CHECK_FALSE(d.inverse(10000.0, 0.0, -100.0).reachable);
    }

    // ---- Linear delta --------------------------------------------------------

    TEST_CASE("linear delta: inverse->forward round-trips over the workspace") {
        const LinearDelta<double> d(LinearDeltaGeometry<double>{120.0, 30.0, 250.0});
        REQUIRE(d.valid());
        Rng rng;
        int tested = 0;
        for (int i = 0; i < 2000; ++i) {
            const double x = rng(-50.0, 50.0);
            const double y = rng(-50.0, 50.0);
            const double z = rng(-230.0, -170.0);
            const auto   ik = d.inverse(x, y, z);
            if (!ik.reachable) {
                continue;
            }
            const auto fk = d.forward(ik.actuators);
            REQUIRE(fk.valid);
            CHECK(fk.pose.translation[0] == doctest::Approx(x).epsilon(1e-6));
            CHECK(fk.pose.translation[1] == doctest::Approx(y).epsilon(1e-6));
            CHECK(fk.pose.translation[2] == doctest::Approx(z).epsilon(1e-6));
            ++tested;
        }
        REQUIRE(tested > 100);
    }

    TEST_CASE("linear delta: centered point lifts all carriages equally") {
        const LinearDelta<double> d(LinearDeltaGeometry<double>{120.0, 30.0, 250.0});
        const auto                ik = d.inverse(0.0, 0.0, -200.0);
        REQUIRE(ik.reachable);
        CHECK(ik.actuators[0] == doctest::Approx(ik.actuators[1]));
        CHECK(ik.actuators[1] == doctest::Approx(ik.actuators[2]));
    }
}
