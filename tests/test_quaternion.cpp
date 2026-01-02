#include <cmath>
#include <numbers>

#include "doctest.h"
#include "rotation.hpp"

constexpr float kPi = std::numbers::pi_v<float>;

TEST_SUITE("Quaternion") {
    TEST_CASE("Identity and normalization") {
        Quaternion<float> q_id = Quaternion<float>::identity();
        CHECK(q_id.w() == 1.0f);
        CHECK(q_id.x() == 0.0f);
        CHECK(q_id.y() == 0.0f);
        CHECK(q_id.z() == 0.0f);

        auto q_norm = q_id.normalized();
        CHECK(q_norm.w() == 1.0f);
        CHECK(q_norm.x() == 0.0f);

        Quaternion<float> q_zero{0.0f, 0.0f, 0.0f, 0.0f};
        auto              maybe_norm = q_zero.normalized_safe();
        CHECK_FALSE(maybe_norm.has_value());
    }

    TEST_CASE("Conjugate and inverse") {
        auto q_opt = Quaternion<float>::from_axis_angle(Vec3<float>{0.0f, 0.0f, 1.0f}, kPi * 0.5f);
        REQUIRE(q_opt.has_value());
        auto q = q_opt.value();

        Vec3<float> v{1.0f, 0.0f, 0.0f};
        auto        v_rot = q.rotate(v);
        CHECK(v_rot[0] == doctest::Approx(0.0f).epsilon(1e-5));
        CHECK(v_rot[1] == doctest::Approx(1.0f).epsilon(1e-5));
        CHECK(v_rot[2] == doctest::Approx(0.0f).epsilon(1e-5));

        auto q_inv = q.inverse();
        REQUIRE(q_inv.has_value());
        auto v_restored = q_inv->rotate(v_rot);
        CHECK(v_restored[0] == doctest::Approx(v[0]).epsilon(1e-5));
        CHECK(v_restored[1] == doctest::Approx(v[1]).epsilon(1e-5));
        CHECK(v_restored[2] == doctest::Approx(v[2]).epsilon(1e-5));
    }

    TEST_CASE("Rotation matrix conversion round-trip") {
        // Create quaternion from Euler ZYX (yaw, pitch, roll)
        EulerZYX<float> e_in(kPi * 0.5f, 0.0f, 0.0f); // 90 deg yaw only
        auto            q = e_in.to_quaternion();
        auto            R = q.to_dcm();

        // Expected 90 deg about Z
        CHECK(R(0, 0) == doctest::Approx(0.0f).epsilon(1e-6));
        CHECK(R(0, 1) == doctest::Approx(-1.0f).epsilon(1e-6));
        CHECK(R(1, 0) == doctest::Approx(1.0f).epsilon(1e-6));
        CHECK(R(1, 1) == doctest::Approx(0.0f).epsilon(1e-6));

        auto q_rt = R.to_quaternion();
        REQUIRE(q_rt.has_value());

        Vec3<float> v{1.0f, 0.0f, 0.0f};
        auto        v_rot_mat = R * v;
        auto        v_rot_q = q_rt->rotate(v);
        CHECK(v_rot_mat[0] == doctest::Approx(v_rot_q[0]).epsilon(1e-5));
        CHECK(v_rot_mat[1] == doctest::Approx(v_rot_q[1]).epsilon(1e-5));
        CHECK(v_rot_mat[2] == doctest::Approx(v_rot_q[2]).epsilon(1e-5));
    }

    TEST_CASE("Integrate body rates (small step)") {
        Quaternion<float> q0 = Quaternion<float>::identity();
        Vec3<float>       omega{0.0f, 0.0f, kPi}; // rad/s about Z
        float             dt = 0.05f;             // small step for first-order approx (~9 deg)
        auto              q1 = q0.integrate_body_rates(omega, dt);

        Vec3<float> v{1.0f, 0.0f, 0.0f};
        auto        v_rot = q1.rotate(v);

        // Expected small-angle rotation using exact trig for reference
        float angle = omega[2] * dt;
        float c = std::cos(angle);
        float s = std::sin(angle);
        CHECK(v_rot[0] == doctest::Approx(c).epsilon(1e-3));
        CHECK(v_rot[1] == doctest::Approx(s).epsilon(1e-3));
        CHECK(v_rot[2] == doctest::Approx(0.0f).epsilon(1e-4));
    }

    TEST_CASE("Slerp mid-rotation about Z") {
        auto qa = Quaternion<float>::identity();
        auto qb_opt = Quaternion<float>::from_axis_angle(Vec3<float>{0.0f, 0.0f, 1.0f}, kPi * 0.5f); // 90 deg
        REQUIRE(qb_opt.has_value());
        auto qb = qb_opt.value();

        auto        qm = Quaternion<float>::slerp(qa, qb, 0.5f); // ~45 deg about Z
        Vec3<float> v{1.0f, 0.0f, 0.0f};
        auto        v_rot = qm.rotate(v);
        CHECK(v_rot[0] == doctest::Approx(std::cos(kPi * 0.25f)).epsilon(1e-5));
        CHECK(v_rot[1] == doctest::Approx(std::sin(kPi * 0.25f)).epsilon(1e-5));
        CHECK(v_rot[2] == doctest::Approx(0.0f).epsilon(1e-5));
    }

    TEST_CASE("Euler <-> DCM conversion") {
        EulerZYX<float> e(0.5f, -0.4f, 0.3f); // yaw, pitch, roll
        auto            R = e.to_dcm();
        auto            e_back = EulerZYX<float>::from_dcm(R);

        CHECK(e_back.roll() == doctest::Approx(e.roll()).epsilon(1e-5));
        CHECK(e_back.pitch() == doctest::Approx(e.pitch()).epsilon(1e-5));
        CHECK(e_back.yaw() == doctest::Approx(e.yaw()).epsilon(1e-5));

        // Spot-check a couple matrix entries against direct trig reference
        float cr = std::cos(e.roll()), sr = std::sin(e.roll());
        float cp = std::cos(e.pitch()), sp = std::sin(e.pitch());
        float cy = std::cos(e.yaw()), sy = std::sin(e.yaw());
        CHECK(R(0, 0) == doctest::Approx(cy * cp).epsilon(1e-6));
        CHECK(R(2, 0) == doctest::Approx(-sp).epsilon(1e-6));
        CHECK(R(0, 1) == doctest::Approx(cy * sp * sr - sy * cr).epsilon(1e-6));
    }

    TEST_CASE("Quat <-> Euler conversion") {
        EulerZYX<float> e(0.4f, -0.1f, 0.2f); // yaw, pitch, roll
        auto            q = e.to_quaternion();
        auto            e_back = q.to_euler<EulerOrder::ZYX>();

        CHECK(e_back.roll() == doctest::Approx(e.roll()).epsilon(1e-5));
        CHECK(e_back.pitch() == doctest::Approx(e.pitch()).epsilon(1e-5));
        CHECK(e_back.yaw() == doctest::Approx(e.yaw()).epsilon(1e-5));
    }

    TEST_CASE("DCM <-> Quaternion conversion") {
        EulerZYX<float> e(-0.45f, 0.35f, -0.25f); // yaw, pitch, roll
        auto            R = e.to_dcm();
        auto            q_opt = R.to_quaternion();
        REQUIRE(q_opt.has_value());
        auto R_back = q_opt->to_dcm();

        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                CHECK(R_back(r, c) == doctest::Approx(R(r, c)).epsilon(1e-5));
            }
        }
    }
}
