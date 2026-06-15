#include <cmath>
#include <numbers>

#include "wet/utility/motor_control.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wet;

/**
 * @brief Tests for motor control transforms
 */

TEST_SUITE("Motor Control Transforms") {
    TEST_CASE("Clarke transform") {
        // Test balanced three-phase
        const ColVec<3, float> abc = {1.0f, -0.5f, -0.5f};

        const auto [alpha, beta] = clarke_transform(abc);

        // For balanced system: α = (2a - b - c)/3, β = (b - c)/√3
        CHECK(alpha == doctest::Approx(1.0f));
        CHECK(beta == doctest::Approx(0.0f).epsilon(1e-6f));
    }

    TEST_CASE("Inverse Clarke transform") {
        // Test round-trip
        const ColVec<3, float> abc_orig = {1.0f, -0.5f, -0.5f};

        const auto ab = clarke_transform(abc_orig);
        const auto abc = inverse_clarke_transform(ab);

        CHECK(abc[0] == doctest::Approx(abc_orig[0]).epsilon(1e-6f));
        CHECK(abc[1] == doctest::Approx(abc_orig[1]).epsilon(1e-6f));
        CHECK(abc[2] == doctest::Approx(abc_orig[2]).epsilon(1e-6f));
    }

    TEST_CASE("Park transform") {
        // Test with θ = 0 (should be identity)
        const AlphaBeta<float> ab = {.alpha = 1.0f, .beta = 0.5f};
        const float            theta = 0.0f;

        const auto [d, q] = park_transform(ab, theta);

        CHECK(d == doctest::Approx(ab.alpha));
        CHECK(q == doctest::Approx(ab.beta));
    }

    TEST_CASE("Park transform with rotation") {
        const AlphaBeta<float> ab = {.alpha = 1.0f, .beta = 0.0f};
        const float            theta = std::numbers::pi_v<float> / 4.0f; // 45°

        const auto [d, q] = park_transform(ab, theta);

        const float expected_d = ab.alpha * std::cos(theta);
        const float expected_q = -ab.alpha * std::sin(theta);

        CHECK(d == doctest::Approx(expected_d).epsilon(1e-6f));
        CHECK(q == doctest::Approx(expected_q).epsilon(1e-6f));
    }

    TEST_CASE("Inverse Park transform") {
        // Test round-trip
        const DirectQuadrature<float> dq_orig = {.d = 0.8f, .q = 0.3f};
        const float                   theta = std::numbers::pi_v<float> / 6.0f; // 30°

        const auto ab = inverse_park_transform(dq_orig, theta);
        const auto [d, q] = park_transform(ab, theta);

        CHECK(d == doctest::Approx(dq_orig.d).epsilon(1e-6f));
        CHECK(q == doctest::Approx(dq_orig.q).epsilon(1e-6f));
    }

    TEST_CASE("Clarke-Park combined transform") {
        // Test three-phase to dq
        const ColVec<3, float> abc = {
            std::cos(0.0f),
            std::cos(2.0f * std::numbers::pi_v<float> / 3.0f),
            std::cos(4.0f * std::numbers::pi_v<float> / 3.0f),
        };
        const float theta = 0.0f;

        const auto [d, q] = clarke_park_transform(abc, theta);

        // At θ = 0, d should be the amplitude, q should be 0
        CHECK(d == doctest::Approx(1.0f).epsilon(1e-6f)); // Clarke transform normalizes to 1.0
        CHECK(q == doctest::Approx(0.0f).epsilon(1e-6f));
    }

    TEST_CASE("Inverse Park-Clarke combined transform") {
        // Test dq to three-phase round-trip
        const DirectQuadrature<float> dq = {.d = 1.0f, .q = 0.5f};
        const float                   theta = std::numbers::pi_v<float> / 4.0f;

        const auto abc = inverse_park_clarke_transform(dq, theta);
        const auto [d2, q2] = clarke_park_transform(abc, theta);

        CHECK(d2 == doctest::Approx(dq.d).epsilon(1e-6f));
        CHECK(q2 == doctest::Approx(dq.q).epsilon(1e-6f));
    }

    TEST_CASE("SVM duty cycles") {
        // Test zero voltage
        const auto svm = svm_duty_cycles<float>({.alpha = 0.0f, .beta = 0.0f}, 100.0f);

        CHECK(svm.duties[0] == doctest::Approx(0.5f));
        CHECK(svm.duties[1] == doctest::Approx(0.5f));
        CHECK(svm.duties[2] == doctest::Approx(0.5f));
        CHECK_FALSE(svm.is_clipped);

        // Test maximum linear voltage (peak phase = Vdc/√3 with SVPWM injection)
        const float v_max = 100.0f / std::numbers::sqrt3_v<float>;
        const auto  svm_max = svm_duty_cycles<float>({.alpha = v_max, .beta = 0.0f}, 100.0f);

        CHECK(svm_max.duties[0] >= 0.0f);
        CHECK(svm_max.duties[0] <= 1.0f);
        CHECK(svm_max.duties[1] >= 0.0f);
        CHECK(svm_max.duties[1] <= 1.0f);
        CHECK(svm_max.duties[2] >= 0.0f);
        CHECK(svm_max.duties[2] <= 1.0f);
        CHECK_FALSE(svm_max.is_clipped); // exactly on the inscribed circle, not clipped
    }

} // TEST_SUITE
