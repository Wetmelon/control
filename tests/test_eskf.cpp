#include <cmath>
#include <numbers>

#include "doctest.h"
#include "kalman.hpp"
#include "rotation.hpp"

constexpr double kPi = std::numbers::pi_v<double>;

// Type aliases for ESKF with 9-state error [δθ (3), δb_gyro (3), δb_accel (3)]
using Mat9d = Matrix<9, 9, double>;
using Mat3x9d = Matrix<3, 9, double>;

// Convenient aliases for the explicit Jacobian types
using PredictJac9 = ErrorStateJacobian<double, 9>;
using MeasJac3x9 = MeasJacobian<double, 3, 9>;

TEST_SUITE("Error-State Kalman Filter (ESKF)") {
    // ESKF with 9D error state: [attitude(3), gyro_bias(3), accel_bias(3)]
    // Measurements: accelerometer (3D)

    TEST_CASE("ESKF initialization") {
        // Initialize with moderate uncertainty
        auto P0 = Mat9d::identity() * 0.1;        // Initial covariance
        auto Q = Mat9d::identity() * 1e-6;        // Process noise (gyro/accel drift)
        auto R = Mat3<double>::identity() * 0.01; // Measurement noise (accel sensor)

        ErrorStateKalmanFilter<9, 3> eskf(P0, Q, R);

        CHECK(static_cast<double>(eskf.covariance()(0, 0)) == doctest::Approx(0.1));
        CHECK(static_cast<double>(eskf.process_noise_covariance()(0, 0)) == doctest::Approx(1e-6));
        CHECK(static_cast<double>(eskf.measurement_noise_covariance()(0, 0)) == doctest::Approx(0.01));
    }

    TEST_CASE("ESKF runtime tuning via setters") {
        auto P0 = Mat9d::identity() * 0.1;
        auto Q = Mat9d::identity() * 1e-6;
        auto R = Mat3<double>::identity() * 0.01;

        ErrorStateKalmanFilter<9, 3> eskf(P0, Q, R);

        // Adjust Q at runtime (e.g., increase when vibrations detected)
        auto Q_new = Mat9d::identity() * 1e-4;
        eskf.set_process_noise_covariance(Q_new);
        CHECK(static_cast<double>(eskf.process_noise_covariance()(0, 0)) == doctest::Approx(1e-4));

        // Adjust R at runtime (e.g., trust accel less during high dynamics)
        auto R_new = Mat3<double>::identity() * 0.5;
        eskf.set_measurement_noise_covariance(R_new);
        CHECK(static_cast<double>(eskf.measurement_noise_covariance()(0, 0)) == doctest::Approx(0.5));

        // Reset covariance (e.g., after significant disturbance)
        auto P_reset = Mat9d::identity() * 0.2;
        eskf.set_covariance(P_reset);
        CHECK(static_cast<double>(eskf.covariance()(0, 0)) == doctest::Approx(0.2));
    }

    TEST_CASE("ESKF predict step with proper gyro-bias coupling") {
        auto P0 = Mat9d::identity() * 0.1;
        auto Q = Mat9d::identity() * 1e-6;
        auto R = Mat3<double>::identity() * 0.01;

        ErrorStateKalmanFilter<9, 3> eskf(P0, Q, R);

        double dt = 0.01; // 10 ms

        // Proper F matrix for ESKF error dynamics:
        // δθ_dot ≈ -δb_g  (gyro bias error causes attitude error to grow)
        // δb_g_dot ≈ 0    (bias is approximately constant)
        // δb_a_dot ≈ 0    (bias is approximately constant)
        //
        // Discrete-time: F = I + dt * A, where A has -I in the (attitude, gyro_bias) block
        auto F = Mat9d::identity();
        // Gyro bias error propagates into attitude error
        F(0, 3) = -dt; // δb_gx → δθ_x
        F(1, 4) = -dt; // δb_gy → δθ_y
        F(2, 5) = -dt; // δb_gz → δθ_z

        auto G = Mat9d::identity() * 0.01;

        // Return explicit ErrorStateJacobian type (concept-checked)
        auto propagate = [F, G](double) -> PredictJac9 {
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

    TEST_CASE("ESKF measurement update (accel correction)") {
        auto P0 = Mat9d::identity() * 0.1;
        auto Q = Mat9d::identity() * 1e-6;
        auto R = Mat3<double>::identity() * 0.01;

        ErrorStateKalmanFilter<9, 3> eskf(P0, Q, R);

        // Measurement: accelerometer reading (should measure gravity + accel bias)
        constexpr double g = 9.81;
        Vec3d            accel_meas{0.0, 0.0, g}; // Gravity pointing down (Z)

        // Measurement function: H matrix (relates error state to measurement)
        // For gravity g = [0, 0, g], the Jacobian ∂(R*g)/∂δθ ≈ -[g]× (skew symmetric)
        // [g]× = |  0  -gz  gy |   with g=[0,0,g] gives:
        //        | gz   0  -gx |   |  0  -g   0 |
        //        |-gy  gx   0  |   |  g   0   0 |
        //                          |  0   0   0 |
        auto H = Mat3x9d{};
        // Attitude error affects acceleration measurement (skew of gravity)
        H(0, 1) = -g; // δθ_y affects accel_x: -gz
        H(1, 0) = g;  // δθ_x affects accel_y: +gz
        // Note: δθ_z doesn't affect accel when gravity is along Z axis
        // Accel bias directly affects measurement (negative because we subtract bias)
        H(0, 6) = -1.0;
        H(1, 7) = -1.0;
        H(2, 8) = -1.0;

        auto M = Mat3<double>::identity(); // Noise coupling

        // Predicted acceleration (from nominal state - assuming upright)
        Vec3d accel_pred{0.0, 0.0, 9.81};

        // Return explicit MeasJacobian type (concept-checked)
        auto h = [H, M, accel_pred]() -> MeasJac3x9 {
            return {.z_pred = accel_pred, .H = H, .M = M};
        };

        bool success = eskf.update(h, accel_meas);
        REQUIRE(success);

        // Error state should remain small (no significant error detected)
        auto err_state = eskf.error_state();
        CHECK(std::abs(static_cast<double>(err_state[0])) < 0.1); // Attitude errors small
        CHECK(std::abs(static_cast<double>(err_state[3])) < 0.1); // Gyro bias errors small

        // Covariance should shrink (measurement reduces uncertainty)
        CHECK(static_cast<double>(eskf.covariance()(0, 0)) < static_cast<double>(P0(0, 0)));
    }

    TEST_CASE("ESKF with attitude error (detects misalignment)") {
        auto P0 = Mat9d::identity() * 0.1;
        auto Q = Mat9d::identity() * 1e-6;
        auto R = Mat3<double>::identity() * 0.01;

        ErrorStateKalmanFilter<9, 3> eskf(P0, Q, R);

        // Simulated scenario: nominal attitude is tilted, accel says upright
        // This simulates a gyro drift that caused attitude error
        constexpr double g = 9.81;

        // Correct H matrix: skew-symmetric of gravity [0, 0, g]
        auto H = Mat3x9d{};
        H(0, 1) = -g;   // δθ_y affects accel_x
        H(1, 0) = g;    // δθ_x affects accel_y
        H(0, 6) = -1.0; // Accel bias x
        H(1, 7) = -1.0; // Accel bias y
        H(2, 8) = -1.0; // Accel bias z

        auto M = Mat3<double>::identity();

        // Accel predicts upright, but nominal state thinks tilted ~5 degrees
        Vec3d accel_meas{0.0, 0.0, 9.81}; // True: vertical

        // Predicted: tilted (wrong)
        double tilt_angle = 5.0 * kPi / 180.0; // 5 degrees
        Vec3d  accel_pred{
            9.81 * std::sin(tilt_angle),
            0.0,
            9.81 * std::cos(tilt_angle)
        };

        // Return explicit MeasJacobian type (concept-checked)
        auto h = [H, M, accel_pred]() -> MeasJac3x9 {
            return {.z_pred = accel_pred, .H = H, .M = M};
        };

        bool success = eskf.update(h, accel_meas);
        REQUIRE(success);

        // Error state should detect the tilt error
        auto err_state = eskf.error_state();
        CHECK(std::abs(static_cast<double>(err_state[1])) > 0.001); // Should detect pitch/roll error
    }

    TEST_CASE("ESKF cycle: predict then update") {
        auto P0 = Mat9d::identity() * 0.1;
        auto Q = Mat9d::identity() * 1e-7;
        auto R = Mat3<double>::identity() * 0.001;

        ErrorStateKalmanFilter<9, 3> eskf(P0, Q, R);

        double dt = 0.01;

        // Propagation Jacobians (simple model)
        auto F = Mat9d::identity();
        F(0, 0) = 0.99;
        F(1, 1) = 0.99;
        F(2, 2) = 0.99;
        F(3, 3) = 0.999;
        F(4, 4) = 0.999;
        F(5, 5) = 0.999;
        F(6, 6) = 0.999;
        F(7, 7) = 0.999;
        F(8, 8) = 0.999;

        auto G = Mat9d::identity() * 0.1;

        // Return explicit ErrorStateJacobian type (concept-checked)
        auto propagate = [F, G](double) -> PredictJac9 {
            return {.F = F, .G = G};
        };

        // Measurement - correct skew-symmetric H for gravity [0, 0, g]
        constexpr double g = 9.81;
        auto             H = Mat3x9d{};
        H(0, 1) = -g;   // δθ_y affects accel_x
        H(1, 0) = g;    // δθ_x affects accel_y
        H(0, 6) = -1.0; // Accel bias x
        H(1, 7) = -1.0; // Accel bias y
        H(2, 8) = -1.0; // Accel bias z

        auto  M = Mat3<double>::identity();
        Vec3d accel_meas{0.0, 0.0, g};
        Vec3d accel_pred{0.0, 0.0, g};

        // Return explicit MeasJacobian type (concept-checked)
        auto h = [H, M, accel_pred]() -> MeasJac3x9 {
            return {.z_pred = accel_pred, .H = H, .M = M};
        };

        // Run several iterations
        for (int i = 0; i < 5; ++i) {
            eskf.predict(propagate, dt);
            [[maybe_unused]] bool success = eskf.update(h, accel_meas);
            REQUIRE(success);

            // Error state should stay bounded
            auto err = eskf.error_state();
            for (size_t j = 0; j < 9; ++j) {
                CHECK(std::abs(static_cast<double>(err[j])) < 0.5);
            }
        }

        // Covariance should converge to stable value
        double final_P_00 = static_cast<double>(eskf.covariance()(0, 0));
        CHECK(final_P_00 > 0.0);
        CHECK(final_P_00 < static_cast<double>(P0(0, 0))); // Should have reduced uncertainty
    }
}
