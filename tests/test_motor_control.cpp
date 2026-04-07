#include <cmath>
#include <numbers>

#include "doctest.h"
#include "motor_control.hpp"

using namespace wetmelon::control;

/**
 * @brief Tests for motor control transforms
 */

TEST_SUITE("Motor Control Transforms") {
    TEST_CASE("Clarke transform") {
        // Test balanced three-phase
        const float a = 1.0f;
        const float b = -0.5f;
        const float c = -0.5f;

        const auto [alpha, beta] = clarke_transform(a, b, c);

        // For balanced system: α = (2a - b - c)/3, β = (b - c)/√3
        CHECK(alpha == doctest::Approx(1.0f));
        CHECK(beta == doctest::Approx(0.0f).epsilon(1e-6f));
    }

    TEST_CASE("Inverse Clarke transform") {
        // Test round-trip
        const float a_orig = 1.0f;
        const float b_orig = -0.5f;
        const float c_orig = -0.5f;

        const auto [alpha, beta] = clarke_transform(a_orig, b_orig, c_orig);
        const auto [a, b, c] = inverse_clarke_transform(alpha, beta);

        CHECK(a == doctest::Approx(a_orig).epsilon(1e-6f));
        CHECK(b == doctest::Approx(b_orig).epsilon(1e-6f));
        CHECK(c == doctest::Approx(c_orig).epsilon(1e-6f));
    }

    TEST_CASE("Park transform") {
        // Test with θ = 0 (should be identity)
        const float alpha = 1.0f;
        const float beta = 0.5f;
        const float theta = 0.0f;

        const auto [d, q] = park_transform(alpha, beta, theta);

        CHECK(d == doctest::Approx(alpha));
        CHECK(q == doctest::Approx(beta));
    }

    TEST_CASE("Park transform with rotation") {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        const float theta = std::numbers::pi_v<float> / 4.0f; // 45°

        const auto [d, q] = park_transform(alpha, beta, theta);

        const float expected_d = alpha * std::cos(theta);
        const float expected_q = -alpha * std::sin(theta);

        CHECK(d == doctest::Approx(expected_d).epsilon(1e-6f));
        CHECK(q == doctest::Approx(expected_q).epsilon(1e-6f));
    }

    TEST_CASE("Inverse Park transform") {
        // Test round-trip
        const float d_orig = 0.8f;
        const float q_orig = 0.3f;
        const float theta = std::numbers::pi_v<float> / 6.0f; // 30°

        const auto [alpha, beta] = inverse_park_transform(d_orig, q_orig, theta);
        const auto [d, q] = park_transform(alpha, beta, theta);

        CHECK(d == doctest::Approx(d_orig).epsilon(1e-6f));
        CHECK(q == doctest::Approx(q_orig).epsilon(1e-6f));
    }

    TEST_CASE("Clarke-Park combined transform") {
        // Test three-phase to dq
        const float a = std::cos(0.0f);
        const float b = std::cos(2.0f * std::numbers::pi_v<float> / 3.0f);
        const float c = std::cos(4.0f * std::numbers::pi_v<float> / 3.0f);
        const float theta = 0.0f;

        const auto [d, q] = clarke_park_transform(a, b, c, theta);

        // At θ = 0, d should be the amplitude, q should be 0
        CHECK(d == doctest::Approx(1.0f).epsilon(1e-6f)); // Clarke transform normalizes to 1.0
        CHECK(q == doctest::Approx(0.0f).epsilon(1e-6f));
    }

    TEST_CASE("Inverse Park-Clarke combined transform") {
        // Test dq to three-phase round-trip
        const float d = 1.0f;
        const float q = 0.5f;
        const float theta = std::numbers::pi_v<float> / 4.0f;

        const auto [a, b, c] = inverse_park_clarke_transform(d, q, theta);
        const auto [d2, q2] = clarke_park_transform(a, b, c, theta);

        CHECK(d2 == doctest::Approx(d).epsilon(1e-6f));
        CHECK(q2 == doctest::Approx(q).epsilon(1e-6f));
    }

    TEST_CASE("SVM duty cycles") {
        // Test zero voltage
        const auto [duty_a, duty_b, duty_c] = svm_duty_cycles(0.0f, 0.0f, 100.0f);

        CHECK(duty_a == doctest::Approx(0.5f));
        CHECK(duty_b == doctest::Approx(0.5f));
        CHECK(duty_c == doctest::Approx(0.5f));

        // Test maximum voltage (should be clamped)
        const float v_max = 100.0f / std::numbers::sqrt3_v<float>;
        const auto [duty_a_max, duty_b_max, duty_c_max] = svm_duty_cycles(v_max, 0.0f, 100.0f);

        CHECK(duty_a_max >= 0.0f);
        CHECK(duty_a_max <= 1.0f);
        CHECK(duty_b_max >= 0.0f);
        CHECK(duty_b_max <= 1.0f);
        CHECK(duty_c_max >= 0.0f);
        CHECK(duty_c_max <= 1.0f);
    }

} // TEST_SUITE