#include <cmath>

#include "filters.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wetmelon::control;

/**
 * @brief Tests for filter design and runtime implementations
 */

TEST_SUITE("Filter Design") {
    TEST_CASE("First-order low-pass design") {
        const auto coeffs = design::lowpass_1st<double>(10.0, 0.001); // 10 Hz cutoff, 1kHz sample rate
        // Check that coefficients are reasonable (non-zero)
        CHECK(coeffs.b0 != doctest::Approx(0.0));
        CHECK(coeffs.b1 != doctest::Approx(0.0));
        CHECK(coeffs.a1 != doctest::Approx(0.0));

        // DC gain should be 1: b0 + b1 / (1 + a1) ≈ 1
        const double dc_gain = (coeffs.b0 + coeffs.b1) / (1.0 + coeffs.a1);
        CHECK(dc_gain == doctest::Approx(1.0).epsilon(0.01));
    }

    TEST_CASE("Second-order low-pass design") {
        const auto coeffs = design::lowpass_2nd<double>(10.0, 0.001, 0.707); // 10 Hz, 1kHz sample rate, Butterworth
        // Check that coefficients are reasonable (non-zero)
        CHECK(coeffs.b0 != doctest::Approx(0.0));
        CHECK(coeffs.b1 != doctest::Approx(0.0));
        CHECK(coeffs.b2 != doctest::Approx(0.0));
        CHECK(coeffs.a1 != doctest::Approx(0.0));
        CHECK(coeffs.a2 != doctest::Approx(0.0));

        // DC gain should be 1: (b0 + b1 + b2) / (1 + a1 + a2) ≈ 1
        const double dc_gain = (coeffs.b0 + coeffs.b1 + coeffs.b2) / (1.0 + coeffs.a1 + coeffs.a2);
        CHECK(dc_gain == doctest::Approx(1.0).epsilon(0.01));
    }

    TEST_CASE("First-order Pade delay") {
        constexpr double T_delay = 0.01; // 10ms delay
        constexpr auto   tf = design::pade_delay_1st<double>(T_delay);

        // First-order Pade: H(s) = (1 - sT/2) / (1 + sT/2)
        const double half_T = T_delay / 2.0;

        // Numerator: 1, -T/2
        CHECK(tf.num[0] == doctest::Approx(1.0));
        CHECK(tf.num[1] == doctest::Approx(-half_T));

        // Denominator: 1, T/2
        CHECK(tf.den[0] == doctest::Approx(1.0));
        CHECK(tf.den[1] == doctest::Approx(half_T));
    }

    TEST_CASE("Second-order Pade delay") {
        constexpr double T_delay = 0.01; // 10ms delay
        constexpr auto   tf = design::pade_delay_2nd<double>(T_delay);

        // Second-order Pade: H(s) = (1 - sT/2 + (sT)²/12) / (1 + sT/2 + (sT)²/12)
        const double half_T = T_delay / 2.0;
        const double T_sq_12 = T_delay * T_delay / 12.0;

        // Numerator: 1, -T/2, T²/12
        CHECK(tf.num[0] == doctest::Approx(1.0));
        CHECK(tf.num[1] == doctest::Approx(-half_T));
        CHECK(tf.num[2] == doctest::Approx(T_sq_12));

        // Denominator: 1, T/2, T²/12
        CHECK(tf.den[0] == doctest::Approx(1.0));
        CHECK(tf.den[1] == doctest::Approx(half_T));
        CHECK(tf.den[2] == doctest::Approx(T_sq_12));
    }

} // TEST_SUITE

TEST_SUITE("Runtime Filters") {
    TEST_CASE("LowPass runtime") {
        // Hardcoded coeffs for 10 Hz cutoff, 1kHz sample rate
        design::FirstOrderCoeffs<float> coeffs{.b0 = 0.030418f, .b1 = 0.030418f, .a1 = -0.938874f};
        LowPass<1, float>               lpf(std::array<float, 2>{coeffs.b0, coeffs.b1}, std::array<float, 1>{coeffs.a1});

        // Test step response
        lpf.reset();
        float output = lpf(1.0f); // Step input

        // Should start responding
        CHECK(output > 0.0f);
        CHECK(output < 1.0f);

        // Let it settle
        for (int i = 0; i < 1000; ++i) {
            output = lpf(1.0f);
        }

        // Should be close to 1.0 (DC gain = 1)
        CHECK(output == doctest::Approx(1.0).epsilon(0.01));
    }

    TEST_CASE("LowPass2nd runtime") {
        auto              coeffs = design::lowpass_2nd(10.0f, 0.001f, 0.707f).as<float>(); // 10 Hz cutoff, Butterworth, 1kHz sample rate
        LowPass<2, float> lpf({coeffs.b0, coeffs.b1, coeffs.b2}, {coeffs.a1, coeffs.a2});

        lpf.reset();
        float output = lpf(1.0f); // Step input

        // Should start responding
        CHECK(output > 0.0f);
        CHECK(output == doctest::Approx(1.0).epsilon(0.01));

        // Let it settle
        for (int i = 0; i < 2000; ++i) {
            output = lpf(1.0f);
        }

        // Should be close to 1.0
        CHECK(output == doctest::Approx(1.0).epsilon(0.01));
    }

    TEST_CASE("Discrete delay runtime") {
        Delay<10, float> delay; // Max 10 sample delay
        delay.init(3);          // 3 sample delay

        delay.reset();

        // Test impulse response
        float output = delay(1.0f);            // Input impulse at t=0
        CHECK(output == doctest::Approx(0.0)); // Should output 0 (no history)

        output = delay(0.0f);                  // t=1
        CHECK(output == doctest::Approx(0.0)); // Still no output

        output = delay(0.0f);                  // t=2
        CHECK(output == doctest::Approx(0.0)); // Still no output

        output = delay(0.0f);                  // t=3
        CHECK(output == doctest::Approx(1.0)); // Should output the impulse from t=0

        output = delay(0.0f);                  // t=4
        CHECK(output == doctest::Approx(0.0)); // Back to zero

        // Test reset
        delay.reset();
        output = delay(2.0f); // New impulse
        CHECK(output == doctest::Approx(0.0));

        output = delay(0.0f);
        CHECK(output == doctest::Approx(0.0));

        output = delay(0.0f);
        CHECK(output == doctest::Approx(0.0));

        output = delay(0.0f);
        CHECK(output == doctest::Approx(2.0)); // Should output the reset impulse
    }

} // TEST_SUITE

TEST_CASE("LowPass general") {
    constexpr TransferFunction<2, 2, float> tf1{
        .num = {0.0f, 10.0f},
        .den = {1.0f, 10.0f}
    };

    LowPass<1, float> lpf(tf1, 0.001f);
}