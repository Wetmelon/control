#include <cmath>
#include <cstddef>
#include <cstdint>

#include "wet/backend.hpp"
#include "wet/kinematics/pose.hpp"
#include "wet/kinematics/serial_arm.hpp"
#include "wet/math/geometry.hpp"
#include "wet/matrix/matrix.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wet;

namespace {

constexpr double pi = 3.14159265358979323846;

// Reproducible LCG.
struct Rng {
    uint64_t s{0xD1B54A32D192ED03ull};
    double   operator()(double lo, double hi) {
        s = (s * 6364136223846793005ull) + 1442695040888963407ull;
        const double u = static_cast<double>(s >> 11) / static_cast<double>(1ull << 53);
        return lo + ((hi - lo) * u);
    }
};

// Independent textbook standard-DH 4x4 transform.
Matrix<4, 4, double> dh_mat(double a, double alpha, double d, double theta) {
    const double         ct = std::cos(theta);
    const double         st = std::sin(theta);
    const double         ca = std::cos(alpha);
    const double         sa = std::sin(alpha);
    Matrix<4, 4, double> m{};
    m(0, 0) = ct;
    m(0, 1) = -st * ca;
    m(0, 2) = st * sa;
    m(0, 3) = a * ct;
    m(1, 0) = st;
    m(1, 1) = ct * ca;
    m(1, 2) = -ct * sa;
    m(1, 3) = a * st;
    m(2, 0) = 0;
    m(2, 1) = sa;
    m(2, 2) = ca;
    m(2, 3) = d;
    m(3, 0) = 0;
    m(3, 1) = 0;
    m(3, 2) = 0;
    m(3, 3) = 1;
    return m;
}

// Build a generic (non-spherical) 6R chain for stress-testing the numerics.
DhChain<6, double> generic_chain() {
    DhChain<6, double> c{};
    c.joints[0] = {0.10, pi / 2, 0.30, 0.0, -pi, pi};
    c.joints[1] = {0.45, 0.0, 0.05, -pi / 2, -pi, pi};
    c.joints[2] = {0.08, pi / 2, 0.0, 0.0, -pi, pi};
    c.joints[3] = {0.0, -pi / 2, 0.42, 0.0, -pi, pi};
    c.joints[4] = {0.03, pi / 2, 0.0, 0.0, -pi, pi};
    c.joints[5] = {0.0, 0.0, 0.10, 0.0, -pi, pi};
    return c;
}

} // namespace

TEST_SUITE("serial_arm") {

    TEST_CASE("synthesize: valid chain succeeds; spherical-wrist builder is flagged") {
        constexpr auto cfg = design::arm_spherical_wrist<double>(0.4, 0.5, 0.4, 0.1);
        static_assert(cfg.success, "builder must produce a valid chain");
        static_assert(cfg.spherical_wrist, "builder must be a spherical wrist");
        CHECK(cfg.success);
        CHECK(cfg.spherical_wrist);

        // A generic chain is valid but not a spherical wrist.
        const auto g = design::synthesize_serial_arm(generic_chain());
        CHECK(g.success);
        CHECK_FALSE(g.spherical_wrist);

        // Inverted joint limits fail validation.
        auto bad = generic_chain();
        bad.joints[2].q_min = 1.0;
        bad.joints[2].q_max = -1.0;
        CHECK_FALSE(design::synthesize_serial_arm(bad).success);
    }

    TEST_CASE("forward: matches the textbook DH 4x4 transform chain") {
        const SerialArm<6, double> arm(generic_chain());
        const auto&                c = arm.chain();
        Rng                        rng;

        for (int trial = 0; trial < 50; ++trial) {
            wet::array<double, 6> q{};
            for (auto& v : q) {
                v = rng(-pi, pi);
            }
            // Independent 4x4 product.
            Matrix<4, 4, double> T = Matrix<4, 4, double>::identity();
            for (size_t i = 0; i < 6; ++i) {
                T = T * dh_mat(c.joints[i].a, c.joints[i].alpha, c.joints[i].d, q[i] + c.joints[i].theta_offset);
            }
            const Pose<double> p = arm.forward(q);

            // Translation.
            CHECK(p.translation[0] == doctest::Approx(T(0, 3)).epsilon(1e-10));
            CHECK(p.translation[1] == doctest::Approx(T(1, 3)).epsilon(1e-10));
            CHECK(p.translation[2] == doctest::Approx(T(2, 3)).epsilon(1e-10));
            // Rotation block.
            const auto R = p.orientation.to_dcm();
            for (size_t r = 0; r < 3; ++r) {
                for (size_t col = 0; col < 3; ++col) {
                    CHECK(R(r, col) == doctest::Approx(T(r, col)).epsilon(1e-9));
                }
            }
        }
    }

    TEST_CASE("forward: 2R planar arm matches the closed-form position") {
        DhChain<2, double> c{};
        c.joints[0] = {0.5, 0.0, 0.0, 0.0, -pi, pi};
        c.joints[1] = {0.3, 0.0, 0.0, 0.0, -pi, pi};
        const SerialArm<2, double> arm(c);

        const double                q1 = 0.4;
        const double                q2 = -0.7;
        const wet::array<double, 2> q{q1, q2};
        const Pose<double>          p = arm.forward(q);
        CHECK(p.translation[0] == doctest::Approx((0.5 * std::cos(q1)) + (0.3 * std::cos(q1 + q2))));
        CHECK(p.translation[1] == doctest::Approx((0.5 * std::sin(q1)) + (0.3 * std::sin(q1 + q2))));
        CHECK(p.translation[2] == doctest::Approx(0.0));
    }

    TEST_CASE("jacobian: matches a finite difference of the end-effector twist") {
        const SerialArm<6, double>  arm(generic_chain());
        const wet::array<double, 6> q{0.3, -0.5, 0.8, 0.2, -0.6, 0.4};
        const auto                  J = arm.jacobian(q);

        const double h = 1e-6;
        for (size_t j = 0; j < 6; ++j) {
            wet::array<double, 6> qp = q;
            wet::array<double, 6> qm = q;
            qp[j] += h;
            qm[j] -= h;
            const Pose<double> pp = arm.forward(qp);
            const Pose<double> pm = arm.forward(qm);
            // Linear velocity column.
            CHECK(J(0, j) == doctest::Approx((pp.translation[0] - pm.translation[0]) / (2 * h)).epsilon(1e-4));
            CHECK(J(1, j) == doctest::Approx((pp.translation[1] - pm.translation[1]) / (2 * h)).epsilon(1e-4));
            CHECK(J(2, j) == doctest::Approx((pp.translation[2] - pm.translation[2]) / (2 * h)).epsilon(1e-4));
            // Angular velocity column: rotvec(Rp * Rm^T) / (2h).
            const auto         dq = pp.orientation * pm.orientation.conjugate();
            Quaternion<double> qd = dq;
            if (qd.w() < 0) {
                qd = Quaternion<double>{-qd.w(), -qd.x(), -qd.y(), -qd.z()};
            }
            const Vec3<double> v{qd.x(), qd.y(), qd.z()};
            const double       vn = v.norm();
            const double       ang = (vn > 0) ? 2 * std::atan2(vn, qd.w()) : 0.0;
            const Vec3<double> w = (vn > 0) ? Vec3<double>(v * (ang / vn)) : Vec3<double>{};
            CHECK(J(3, j) == doctest::Approx(w[0] / (2 * h)).epsilon(1e-4));
            CHECK(J(4, j) == doctest::Approx(w[1] / (2 * h)).epsilon(1e-4));
            CHECK(J(5, j) == doctest::Approx(w[2] / (2 * h)).epsilon(1e-4));
        }
    }

    TEST_CASE("inverse: full-pose DLS round-trips forward across the workspace") {
        const SerialArm<6, double> arm(design::arm_spherical_wrist<double>(0.4, 0.5, 0.4, 0.1).chain);
        Rng                        rng;
        int                        ok = 0;

        for (int trial = 0; trial < 150; ++trial) {
            wet::array<double, 6> q{};
            for (auto& v : q) {
                v = rng(-2.0, 2.0);
            }
            const Pose<double> target = arm.forward(q);
            // Seed away from the true configuration.
            wet::array<double, 6> seed = q;
            for (auto& v : seed) {
                v += rng(-0.3, 0.3);
            }
            const auto sol = arm.inverse(target, seed, task_full, 200, 0.02, 1e-10);
            if (!sol.converged) {
                continue;
            }
            ++ok;
            // The recovered configuration reproduces the target pose (branch may differ).
            const Pose<double> got = arm.forward(sol.joints);
            CHECK(got.translation.distance(target.translation) < 1e-6);
            const auto   qd = (target.orientation * got.orientation.conjugate());
            const double ang = 2 * std::acos(std::min(1.0, std::abs(qd.w())));
            CHECK(ang < 1e-5);
        }
        CHECK(ok > 110); // most seeds converge from a nearby start
    }

    TEST_CASE("inverse: position-only task mask on a 3-DOF arm") {
        DhChain<3, double> c{};
        c.joints[0] = {0.0, pi / 2, 0.3, 0.0, -pi, pi};
        c.joints[1] = {0.4, 0.0, 0.0, 0.0, -pi, pi};
        c.joints[2] = {0.4, 0.0, 0.0, 0.0, -pi, pi};
        const SerialArm<3, double> arm(c);

        const wet::array<double, 3> q{0.5, -0.6, 0.7};
        const Pose<double>          target = arm.forward(q);
        const wet::array<double, 3> seed{0.3, -0.4, 0.5};

        const auto sol = arm.inverse(target, seed, task_position, 200, 0.02, 1e-10);
        REQUIRE(sol.converged);
        // Position is reproduced; orientation is left free by the mask.
        const Pose<double> got = arm.forward(sol.joints);
        CHECK(got.translation.distance(target.translation) < 1e-6);
    }

    TEST_CASE("manipulability: positive in general, vanishes at a wrist singularity") {
        const SerialArm<6, double>  arm(design::arm_spherical_wrist<double>(0.4, 0.5, 0.4, 0.1).chain);
        const wet::array<double, 6> generic{0.3, -0.5, 0.8, 0.2, 0.6, 0.4};
        // Wrist pitch (joint 5, index 4) at zero aligns the two wrist roll axes —
        // the classic roll-pitch-roll wrist singularity, where the full 6×6
        // Jacobian loses rank.
        const wet::array<double, 6> wrist_sing{0.3, -0.5, 0.8, 0.2, 0.0, 0.4};

        CHECK(arm.manipulability(generic) > 1e-3);
        CHECK(arm.manipulability(wrist_sing) < 1e-9);
        CHECK(arm.near_singular(wrist_sing));
        CHECK_FALSE(arm.near_singular(generic));
    }

    TEST_CASE("select_nearest: picks the branch closest in wrapped joint space") {
        wet::array<wet::array<double, 3>, 4> sols{};
        sols[0] = {3.0, 0.0, 0.0};
        sols[1] = {0.1, -0.1, 0.05}; // closest to the reference
        sols[2] = {-3.0, 2.0, 1.0};
        sols[3] = {2.0, 2.0, 2.0};
        const wet::array<double, 3> ref{0.0, 0.0, 0.0};
        CHECK(select_nearest(sols, 4u, ref) == 1u);

        // Wrapping: an angle near +π is "close" to one near −π.
        wet::array<wet::array<double, 1>, 2> w{};
        w[0] = {pi - 0.05};
        w[1] = {0.5};
        const wet::array<double, 1> rw{-pi + 0.05};
        CHECK(select_nearest(w, 2u, rw) == 0u);

        // Empty candidate set returns the invalid index.
        CHECK(select_nearest(sols, 0u, ref) == 0u);
    }

    TEST_CASE("as<float>: rebinds the chain and config") {
        const auto cfg = design::arm_spherical_wrist<double>(0.4, 0.5, 0.4, 0.1);
        const auto cfgf = cfg.as<float>();
        CHECK(cfgf.success == cfg.success);
        CHECK(cfgf.spherical_wrist == cfg.spherical_wrist);

        const SerialArm<6, float>  armf(cfgf);
        const wet::array<float, 6> q{0.1F, 0.2F, -0.3F, 0.4F, -0.5F, 0.6F};
        const Pose<float>          p = armf.forward(q);
        // Cross-check the float FK position against the double arm.
        const SerialArm<6, double>  armd(cfg.chain);
        const wet::array<double, 6> qd{0.1, 0.2, -0.3, 0.4, -0.5, 0.6};
        const Pose<double>          pd = armd.forward(qd);
        CHECK(p.translation[0] == doctest::Approx(pd.translation[0]).epsilon(1e-4));
        CHECK(p.translation[2] == doctest::Approx(pd.translation[2]).epsilon(1e-4));
    }
}
