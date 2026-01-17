
#include "doctest.h"
#include "sensor_fusion.hpp"

using namespace wetmelon::control;

TEST_SUITE("Sensor Fusion Filters") {
    TEST_CASE("ComplementaryFilter basic functionality") {
        ComplementaryFilter<float> filter(0.98f);

        // Test initial orientation is identity
        auto q_init = filter.getOrientation();
        CHECK(q_init.w() == doctest::Approx(1.0));
        CHECK(q_init.x() == doctest::Approx(0.0));
        CHECK(q_init.y() == doctest::Approx(0.0));
        CHECK(q_init.z() == doctest::Approx(0.0));

        // Test update with stationary IMU (gravity aligned with -Z)
        Vec3<float> accel{0.0f, 0.0f, -9.81f}; // gravity down
        Vec3<float> gyro{0.0f, 0.0f, 0.0f};    // no rotation
        float       dt = 0.01f;

        filter.update(accel, gyro, dt);
        auto q = filter.getOrientation();

        // Should remain close to identity (normalized quaternion)
        CHECK(q.norm() == doctest::Approx(1.0).epsilon(1e-6));

        // Test with rotation around Z axis
        Vec3<float> accel_rotated{9.81f, 0.0f, 0.0f}; // gravity along +X (90 deg rotation)
        Vec3<float> gyro_z{0.0f, 0.0f, 1.57f};        // 90 deg/s around Z

        filter.update(accel_rotated, gyro_z, dt);
        q = filter.getOrientation();

        // Should have some rotation component
        CHECK(q.norm() == doctest::Approx(1.0).epsilon(1e-6));
        CHECK(std::abs(q.w()) < 1.0f); // Not identity anymore
    }

    TEST_CASE("MadgwickFilter basic functionality") {
        MadgwickFilter<float> filter(0.1f);

        // Test initial orientation
        auto q_init = filter.getOrientation();
        CHECK(q_init.w() == doctest::Approx(1.0));
        CHECK(q_init.x() == doctest::Approx(0.0));
        CHECK(q_init.y() == doctest::Approx(0.0));
        CHECK(q_init.z() == doctest::Approx(0.0));

        // Test update with accelerometer and magnetometer
        Vec3<float> accel{0.0f, 0.0f, -9.81f}; // gravity down
        Vec3<float> gyro{0.0f, 0.0f, 0.0f};    // no rotation
        Vec3<float> mag{0.0f, 1.0f, 0.0f};     // magnetic north along +Y
        float       dt = 0.01f;

        filter.update(accel, gyro, mag, dt);
        auto q = filter.getOrientation();

        CHECK(q.norm() == doctest::Approx(1.0).epsilon(1e-6));
    }

    TEST_CASE("MahonyFilter basic functionality") {
        MahonyFilter<float> filter(0.5f, 0.0f);

        // Test initial orientation
        auto q_init = filter.getOrientation();
        CHECK(q_init.w() == doctest::Approx(1.0));
        CHECK(q_init.x() == doctest::Approx(0.0));
        CHECK(q_init.y() == doctest::Approx(0.0));
        CHECK(q_init.z() == doctest::Approx(0.0));

        // Test update with accelerometer and magnetometer
        Vec3<float> accel{0.0f, 0.0f, -9.81f}; // gravity down
        Vec3<float> gyro{0.0f, 0.0f, 0.0f};    // no rotation
        Vec3<float> mag{0.0f, 1.0f, 0.0f};     // magnetic north along +Y
        float       dt = 0.01f;

        filter.update(accel, gyro, mag, dt);
        auto q = filter.getOrientation();

        CHECK(q.norm() == doctest::Approx(1.0).epsilon(1e-6));
    }

    TEST_CASE("ESKFOrientationFilter basic functionality") {
        ESKFOrientationFilter filter;

        // Test initial orientation
        auto q_init = filter.getOrientation();
        CHECK(q_init.w() == doctest::Approx(1.0));
        CHECK(q_init.x() == doctest::Approx(0.0));
        CHECK(q_init.y() == doctest::Approx(0.0));
        CHECK(q_init.z() == doctest::Approx(0.0));

        // Test gyro bias initialization
        auto bias_init = filter.getGyroBias();
        CHECK(bias_init[0] == doctest::Approx(0.0));
        CHECK(bias_init[1] == doctest::Approx(0.0));
        CHECK(bias_init[2] == doctest::Approx(0.0));

        // Test update with accelerometer and magnetometer
        Vec3<float> accel{0.0f, 0.0f, -9.81f}; // gravity down
        Vec3<float> gyro{0.0f, 0.0f, 0.0f};    // no rotation
        Vec3<float> mag{0.0f, 1.0f, 0.0f};     // magnetic north along +Y
        float       dt = 0.01f;

        filter.update(accel, gyro, mag, dt);
        auto q = filter.getOrientation();
        auto bias = filter.getGyroBias();

        CHECK(q.norm() == doctest::Approx(1.0).epsilon(1e-6));
        // Bias should remain close to zero for stationary case
        CHECK(std::abs(bias[0]) < 0.1f);
        CHECK(std::abs(bias[1]) < 0.1f);
        CHECK(std::abs(bias[2]) < 0.1f);
    }

    TEST_CASE("ComplementaryFilter convergence test") {
        ComplementaryFilter<float> filter(0.98f);

        // Start with identity orientation
        Vec3<float> accel{0.0f, 0.0f, -9.81f}; // gravity down
        Vec3<float> gyro{0.0f, 0.0f, 0.0f};    // no rotation
        float       dt = 0.01f;

        // Run multiple updates to test convergence
        for (int i = 0; i < 100; ++i) {
            filter.update(accel, gyro, dt);
        }

        auto q = filter.getOrientation();
        CHECK(q.norm() == doctest::Approx(1.0).epsilon(1e-6));

        // For stationary case, should converge to orientation that aligns gravity with -Z
        // This means the quaternion should represent minimal rotation
        CHECK(std::abs(q.w()) > 0.9f); // Should be close to identity
    }

    TEST_CASE("Filter orientation consistency") {
        // Test that all filters produce valid quaternions (unit norm)

        ComplementaryFilter<float> comp_filter(0.95f);
        MadgwickFilter<float>      madgwick_filter(0.1f);
        MahonyFilter<float>        mahony_filter(0.5f, 0.0f);
        ESKFOrientationFilter      eskf_filter;

        Vec3<float> accel{0.0f, 0.0f, -9.81f};
        Vec3<float> gyro{0.1f, 0.05f, -0.08f}; // Small rotation
        Vec3<float> mag{0.0f, 1.0f, 0.0f};
        float       dt = 0.01f;

        // Update all filters
        comp_filter.update(accel, gyro, dt);
        madgwick_filter.update(accel, gyro, mag, dt);
        mahony_filter.update(accel, gyro, mag, dt);
        eskf_filter.update(accel, gyro, mag, dt);

        // Check all produce valid unit quaternions
        CHECK(comp_filter.getOrientation().norm() == doctest::Approx(1.0).epsilon(1e-6));
        CHECK(madgwick_filter.getOrientation().norm() == doctest::Approx(1.0).epsilon(1e-6));
        CHECK(mahony_filter.getOrientation().norm() == doctest::Approx(1.0).epsilon(1e-6));
        CHECK(eskf_filter.getOrientation().norm() == doctest::Approx(1.0).epsilon(1e-6));
    }

    TEST_CASE("ESKF basic operation") {
        ESKFOrientationFilter filter;

        // Test basic operation with stationary inputs
        Vec3<float> accel{0.0f, 0.0f, -9.81f};
        Vec3<float> gyro{0.0f, 0.0f, 0.0f};
        Vec3<float> mag{0.0f, 1.0f, 0.0f};
        float       dt = 0.01f;

        // Run several updates
        for (int i = 0; i < 10; ++i) {
            filter.update(accel, gyro, mag, dt);
        }

        auto q = filter.getOrientation();
        auto bias = filter.getGyroBias();

        // Check that orientation remains valid
        CHECK(q.norm() == doctest::Approx(1.0).epsilon(1e-6));
        // For stationary case, should stay close to identity
        CHECK(std::abs(q.w()) > 0.99f);
        // Bias should remain small
        CHECK(std::abs(bias[0]) < 0.01f);
        CHECK(std::abs(bias[1]) < 0.01f);
        CHECK(std::abs(bias[2]) < 0.01f);
    }

    TEST_CASE("Filter robustness to noisy measurements") {
        ComplementaryFilter<float> filter(0.9f);

        Vec3<float> true_accel{0.0f, 0.0f, -9.81f};
        Vec3<float> true_gyro{0.0f, 0.0f, 0.0f};
        float       dt = 0.01f;

        // Add noise to measurements
        for (int i = 0; i < 100; ++i) {
            Vec3<float> noisy_accel = true_accel + Vec3<float>{
                                          0.1f * (rand() / float(RAND_MAX) - 0.5f),
                                          0.1f * (rand() / float(RAND_MAX) - 0.5f),
                                          0.1f * (rand() / float(RAND_MAX) - 0.5f),
                                      };

            Vec3<float> noisy_gyro = true_gyro + Vec3<float>{
                                         0.01f * (rand() / float(RAND_MAX) - 0.5f),
                                         0.01f * (rand() / float(RAND_MAX) - 0.5f),
                                         0.01f * (rand() / float(RAND_MAX) - 0.5f),
                                     };

            filter.update(noisy_accel, noisy_gyro, dt);

            // Check quaternion remains valid
            auto q = filter.getOrientation();
            CHECK(q.norm() == doctest::Approx(1.0).epsilon(1e-5));
        }
    }

} // TEST_SUITE