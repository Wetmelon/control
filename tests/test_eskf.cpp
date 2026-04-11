#include <cmath>
#include <numbers>

#include "doctest.h"
#include "eskf.hpp"
#include "rotation.hpp"

using namespace wetmelon::control;

constexpr double kPi = std::numbers::pi_v<double>;

// Type aliases for ESKF with 6-state error [δθ (3), δb_gyro (3)]
using Mat6d = Matrix<6, 6, double>;
using Mat3x6d = Matrix<3, 6, double>;

// Convenient aliases for the explicit Jacobian types
using PredictJac6 = ErrorStateJacobian<double, 6>;
using MeasJac6x6 = MeasJacobian<double, 6, 6>;

TEST_SUITE("Error-State Kalman Filter (ESKF)") {
    // ESKF with 6D error state: [attitude(3), gyro_bias(3)]
    // Measurements: accelerometer (3D) + magnetometer (3D)

    TEST_CASE("ESKF initialization") {
        // Initialize with moderate uncertainty
        auto P0 = Mat6d::identity() * 0.1; // Initial covariance
        auto Q = Mat6d::identity() * 1e-6; // Process noise (gyro drift)
        auto R = Mat6d::identity() * 0.01; // Measurement noise (accel/mag sensors)

        ErrorStateKalmanFilter<6, 6> eskf(P0, Q, R);

        CHECK(static_cast<double>(eskf.covariance()(0, 0)) == doctest::Approx(0.1));
        CHECK(static_cast<double>(eskf.process_noise_covariance()(0, 0)) == doctest::Approx(1e-6));
        CHECK(static_cast<double>(eskf.measurement_noise_covariance()(0, 0)) == doctest::Approx(0.01));
    }

    TEST_CASE("ESKF runtime tuning via setters") {
        auto P0 = Mat6d::identity() * 0.1;
        auto Q = Mat6d::identity() * 1e-6;
        auto R = Mat6d::identity() * 0.01;

        ErrorStateKalmanFilter<6, 6> eskf(P0, Q, R);

        // Adjust Q at runtime (e.g., increase when vibrations detected)
        auto Q_new = Mat6d::identity() * 1e-4;
        eskf.set_process_noise_covariance(Q_new);
        CHECK(static_cast<double>(eskf.process_noise_covariance()(0, 0)) == doctest::Approx(1e-4));

        // Adjust R at runtime (e.g., trust sensors less during high dynamics)
        auto R_new = Mat6d::identity() * 0.5;
        eskf.set_measurement_noise_covariance(R_new);
        CHECK(static_cast<double>(eskf.measurement_noise_covariance()(0, 0)) == doctest::Approx(0.5));

        // Reset covariance (e.g., after significant disturbance)
        auto P_reset = Mat6d::identity() * 0.2;
        eskf.set_covariance(P_reset);
        CHECK(static_cast<double>(eskf.covariance()(0, 0)) == doctest::Approx(0.2));
    }

    TEST_CASE("ESKF predict step with proper gyro-bias coupling") {
        auto P0 = Mat6d::identity() * 0.1;
        auto Q = Mat6d::identity() * 1e-6;
        auto R = Mat6d::identity() * 0.01;

        ErrorStateKalmanFilter<6, 6> eskf(P0, Q, R);

        double dt = 0.01; // 10 ms

        // Proper F matrix for ESKF error dynamics:
        // δθ_dot ≈ -δb_g  (gyro bias error causes attitude error to grow)
        // δb_g_dot ≈ 0    (bias is approximately constant)
        //
        // Discrete-time: F = I + dt * A, where A has -I in the (attitude, gyro_bias) block
        auto F = Mat6d::identity();
        // Gyro bias error propagates into attitude error
        F(0, 3) = -dt; // δb_gx → δθ_x
        F(1, 4) = -dt; // δb_gy → δθ_y
        F(2, 5) = -dt; // δb_gz → δθ_z

        auto G = Mat6d::identity() * 0.01;

        // Return explicit ErrorStateJacobian type (concept-checked)
        auto propagate = [F, G](double) -> PredictJac6 {
            return {.F = F, .G = G};
        };

        double P_before = static_cast<double>(eskf.covariance()(0, 0));
        eskf.predict(propagate, dt);
        double P_after = static_cast<double>(eskf.covariance()(0, 0));

        // Covariance should grow due to process noise
        CHECK(P_after > P_before * 0.99);

        // Cross-covariance between attitude and gyro bias should be non-zero
        // This shows the coupling is working
        double P_03 = static_cast<double>(eskf.covariance()(0, 3));
        CHECK(P_03 != doctest::Approx(0.0).epsilon(1e-10));
    }

    TEST_CASE("ESKF measurement update (accel + mag correction)") {
        auto P0 = Mat6d::identity() * 0.1;
        auto Q = Mat6d::identity() * 1e-6;
        auto R = Mat6d::identity() * 0.01;

        ErrorStateKalmanFilter<6, 6> eskf(P0, Q, R);

        // Measurements: accelerometer and magnetometer
        constexpr double g = 9.81;
        Vec3d            accel_meas{0.0, 0.0, g}; // Gravity pointing down (Z)
        Vec3d            mag_meas{0.0, 1.0, 0.0}; // Magnetic field along Y

        // Measurement Jacobian H (6x6)
        auto H = Mat6d{};
        // Accelerometer part (first 3 rows)
        H(0, 1) = -g; // δθ_y affects accel_x
        H(1, 0) = g;  // δθ_x affects accel_y
        // Magnetometer part (last 3 rows)
        H(3, 1) = -1.0; // δθ_y affects mag_x
        H(4, 0) = 1.0;  // δθ_x affects mag_y

        auto M = Mat6d::identity(); // Noise coupling

        // Predicted measurements (from nominal state - assuming upright)
        Vec3d     accel_pred{0.0, 0.0, 9.81};
        Vec3d     mag_pred{0.0, 1.0, 0.0};
        ColVec<6> y_pred{accel_pred[0], accel_pred[1], accel_pred[2], mag_pred[0], mag_pred[1], mag_pred[2]};

        // Return explicit MeasJacobian type (concept-checked)
        auto meas_fn = [H, M, y_pred]() -> MeasJac6x6 {
            return {.y_pred = y_pred, .H = H, .M = M};
        };

        ColVec<6> y{accel_meas[0], accel_meas[1], accel_meas[2], mag_meas[0], mag_meas[1], mag_meas[2]};
        bool      success = eskf.update(meas_fn, y);
        REQUIRE(success);

        // Error state should remain small (no significant error detected)
        auto err_state = eskf.error_state();
        CHECK(std::abs(static_cast<double>(err_state[0])) < 0.1); // Attitude errors small
        CHECK(std::abs(static_cast<double>(err_state[3])) < 0.1); // Gyro bias errors small

        // Covariance should shrink (measurement reduces uncertainty)
        CHECK(static_cast<double>(eskf.covariance()(0, 0)) < static_cast<double>(P0(0, 0)));
    }

    TEST_CASE("ESKF with attitude error (detects misalignment)") {
        auto P0 = Mat6d::identity() * 0.1;
        auto Q = Mat6d::identity() * 1e-6;
        auto R = Mat6d::identity() * 0.01;

        ErrorStateKalmanFilter<6, 6> eskf(P0, Q, R);

        // Simulated scenario: nominal attitude is tilted, sensors say upright
        // This simulates a gyro drift that caused attitude error
        constexpr double g = 9.81;

        // Correct H matrix for accel + mag
        auto H = Mat6d{};
        H(0, 1) = -g;   // δθ_y affects accel_x
        H(1, 0) = g;    // δθ_x affects accel_y
        H(3, 1) = -1.0; // δθ_y affects mag_x
        H(4, 0) = 1.0;  // δθ_x affects mag_y

        auto M = Mat6d::identity();

        // Measurements: upright
        Vec3d accel_meas{0.0, 0.0, 9.81};
        Vec3d mag_meas{0.0, 1.0, 0.0};

        // Predicted: tilted (wrong)
        double tilt_angle = 5.0 * kPi / 180.0; // 5 degrees
        Vec3d  accel_pred{
            9.81 * std::sin(tilt_angle),
            0.0,
            9.81 * std::cos(tilt_angle)
        };
        Vec3d mag_pred{
            -std::sin(tilt_angle),
            1.0,
            0.0
        };
        ColVec<6> y_pred{accel_pred[0], accel_pred[1], accel_pred[2], mag_pred[0], mag_pred[1], mag_pred[2]};

        // Return explicit MeasJacobian type (concept-checked)
        auto meas_fn = [H, M, y_pred]() -> MeasJac6x6 {
            return {.y_pred = y_pred, .H = H, .M = M};
        };

        ColVec<6> y{accel_meas[0], accel_meas[1], accel_meas[2], mag_meas[0], mag_meas[1], mag_meas[2]};
        bool      success = eskf.update(meas_fn, y);
        REQUIRE(success);

        // Error state should detect the tilt error
        auto err_state = eskf.error_state();
        CHECK(std::abs(static_cast<double>(err_state[1])) > 0.001); // Should detect pitch/roll error
    }

    TEST_CASE("ESKF cycle: predict then update") {
        auto P0 = Mat6d::identity() * 0.1;
        auto Q = Mat6d::identity() * 1e-7;
        auto R = Mat6d::identity() * 0.001;

        ErrorStateKalmanFilter<6, 6> eskf(P0, Q, R);

        double dt = 0.01;

        // Propagation Jacobians (simple model)
        auto F = Mat6d::identity();
        F(0, 0) = 0.99;
        F(1, 1) = 0.99;
        F(2, 2) = 0.99;
        F(3, 3) = 0.999;
        F(4, 4) = 0.999;
        F(5, 5) = 0.999;

        auto G = Mat6d::identity() * 0.1;

        // Return explicit ErrorStateJacobian type (concept-checked)
        auto propagate = [F, G](double) -> PredictJac6 {
            return {.F = F, .G = G};
        };

        // Measurement - correct H for accel + mag
        constexpr double g = 9.81;
        auto             H = Mat6d{};
        H(0, 1) = -g;   // δθ_y affects accel_x
        H(1, 0) = g;    // δθ_x affects accel_y
        H(3, 1) = -1.0; // δθ_y affects mag_x
        H(4, 0) = 1.0;  // δθ_x affects mag_y

        auto      M = Mat6d::identity();
        Vec3d     accel_meas{0.0, 0.0, g};
        Vec3d     mag_meas{0.0, 1.0, 0.0};
        Vec3d     accel_pred{0.0, 0.0, g};
        Vec3d     mag_pred{0.0, 1.0, 0.0};
        ColVec<6> y_pred{accel_pred[0], accel_pred[1], accel_pred[2], mag_pred[0], mag_pred[1], mag_pred[2]};

        // Return explicit MeasJacobian type (concept-checked)
        auto meas_fn = [H, M, y_pred]() -> MeasJac6x6 {
            return {.y_pred = y_pred, .H = H, .M = M};
        };

        ColVec<6> y{accel_meas[0], accel_meas[1], accel_meas[2], mag_meas[0], mag_meas[1], mag_meas[2]};

        // Run several iterations
        for (int i = 0; i < 5; ++i) {
            eskf.predict(propagate, dt);
            [[maybe_unused]] bool success = eskf.update(meas_fn, y);
            REQUIRE(success);

            // Error state should stay bounded
            auto err = eskf.error_state();
            for (size_t j = 0; j < 6; ++j) {
                CHECK(std::abs(static_cast<double>(err[j])) < 0.5);
            }
        }

        // Covariance should converge to stable value
        double final_P_00 = static_cast<double>(eskf.covariance()(0, 0));
        CHECK(final_P_00 > 0.0);
        CHECK(final_P_00 < static_cast<double>(P0(0, 0))); // Should have reduced uncertainty
    }
}
