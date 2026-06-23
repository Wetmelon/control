#include "wet/toolbox/conditioning.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wet;

/**
 * @brief Tests for nonlinear signal-conditioning blocks (conditioning.hpp):
 *        deadband, inverse_deadband, SlewLimiter, Hysteresis.
 */

TEST_SUITE("Signal conditioning") {

    TEST_CASE("deadband matches Simulink Dead Zone (symmetric and [lower, upper])") {
        // Symmetric overload: [-0.5, 0.5].
        CHECK(deadband(0.3, 0.5) == doctest::Approx(0.0));       // inside
        CHECK(deadband(-0.5, 0.5) == doctest::Approx(0.0));      // on edge
        CHECK(deadband(0.6, 0.5) == doctest::Approx(0.1));       // x - upper
        CHECK(deadband(-0.6, 0.5) == doctest::Approx(-0.1));     // x - lower
        CHECK(deadband(0.5001, 0.5) == doctest::Approx(0.0001)); // continuous at edge

        // Asymmetric [lower, upper] like the Dead Zone block.
        CHECK(deadband(2.0, 1.5, 2.5) == doctest::Approx(0.0));  // inside
        CHECK(deadband(2.7, 1.5, 2.5) == doctest::Approx(0.2));  // x - upper
        CHECK(deadband(1.0, 1.5, 2.5) == doctest::Approx(-0.5)); // x - lower (below band)
    }

    TEST_CASE("inverse_deadband boosts past a physical dead zone (symmetric + asymmetric)") {
        // Symmetric overload.
        CHECK(inverse_deadband(0.0, 0.2) == doctest::Approx(0.0)); // exact zero held
        CHECK(inverse_deadband(1.0, 0.2) == doctest::Approx(1.2)); // +band
        CHECK(inverse_deadband(-1.0, 0.2) == doctest::Approx(-1.2));

        // Asymmetric [lower, upper] with independent offsets + threshold.
        CHECK(inverse_deadband(0.5, -0.1, 0.3) == doctest::Approx(0.8));       // x>0 -> +upper
        CHECK(inverse_deadband(-0.5, -0.1, 0.3) == doctest::Approx(-0.6));     // x<0 -> +lower
        CHECK(inverse_deadband(0.05, -0.1, 0.3, 0.1) == doctest::Approx(0.0)); // within threshold
    }

    TEST_CASE("scaled_deadband rescales surviving range to full span") {
        CHECK(scaled_deadband(0.05, 0.1) == doctest::Approx(0.0)); // inside dead zone
        CHECK(scaled_deadband(1.0, 0.1) == doctest::Approx(1.0));  // extreme stays at 1
        CHECK(scaled_deadband(-1.0, 0.1) == doctest::Approx(-1.0));
        CHECK(scaled_deadband(0.1, 0.1) == doctest::Approx(0.0)); // on edge
        // Just past the edge starts from 0: (0.55-0.1)/(1-0.1) = 0.5.
        CHECK(scaled_deadband(0.55, 0.1) == doctest::Approx(0.5));
    }

    TEST_CASE("expo softens center, preserves endpoints and sign") {
        CHECK(expo(0.0, 0.5) == doctest::Approx(0.0));
        CHECK(expo(1.0, 0.5) == doctest::Approx(1.0)); // endpoint preserved
        CHECK(expo(-1.0, 0.5) == doctest::Approx(-1.0));
        CHECK(expo(0.5, 0.0) == doctest::Approx(0.5));   // k=0 linear
        CHECK(expo(0.5, 1.0) == doctest::Approx(0.125)); // k=1 cubic: 0.5^3
        // Mid expo: (1-0.5)*0.5 + 0.5*0.125 = 0.3125, softer than linear.
        CHECK(expo(0.5, 0.5) == doctest::Approx(0.3125));
    }

    TEST_CASE("SlewLimiter caps the per-step change, asymmetric up/down") {
        SlewLimiter<double> sl{10.0, 100.0}; // up 10/s, down 100/s
        const double        dt = 0.1;        // max +1.0 / −10.0 per step

        CHECK(sl(5.0, dt) == doctest::Approx(5.0));      // first sample seeds to target
        CHECK(sl(100.0, dt) == doctest::Approx(6.0));    // limited rise: +1.0
        CHECK(sl(100.0, dt) == doctest::Approx(7.0));    // +1.0 again
        CHECK(sl(-100.0, dt) == doctest::Approx(-3.0));  // fast fall: −10.0 (7 → −3)
        CHECK(sl(-100.0, dt) == doctest::Approx(-13.0)); // −10.0 again
        // A small move within the step limit snaps straight to target.
        CHECK(sl(-13.5, dt) == doctest::Approx(-13.5));

        // Small moves within the limit pass through exactly.
        SlewLimiter<double> s2{1000.0};
        (void)s2(0.0, dt);
        CHECK(s2(0.5, dt) == doctest::Approx(0.5));
    }

    TEST_CASE("classify_range / RangeMonitor bands per NE43 [0.25 (0.5 4.5) 4.75]") {
        // Pure kernel.
        CHECK(classify_range(0.1, 0.25, 0.5, 4.5, 4.75) == SignalStatus::FaultLow);
        CHECK(classify_range(0.3, 0.25, 0.5, 4.5, 4.75) == SignalStatus::UnderRange);
        CHECK(classify_range(2.5, 0.25, 0.5, 4.5, 4.75) == SignalStatus::Valid);
        CHECK(classify_range(4.6, 0.25, 0.5, 4.5, 4.75) == SignalStatus::OverRange);
        CHECK(classify_range(4.9, 0.25, 0.5, 4.5, 4.75) == SignalStatus::FaultHigh);
        // Band edges: valid span is inclusive, fault floor belongs to UnderRange.
        CHECK(classify_range(0.5, 0.25, 0.5, 4.5, 4.75) == SignalStatus::Valid);
        CHECK(classify_range(0.25, 0.25, 0.5, 4.5, 4.75) == SignalStatus::UnderRange);

        RangeMonitor<double> ai{0.25, 0.5, 4.5, 4.75};
        CHECK(ai.faulted(0.1) == true);  // broken wire
        CHECK(ai.faulted(0.3) == false); // saturated low is NOT a wire fault
        CHECK(ai.valid(2.5) == true);
        CHECK(is_fault(ai(4.9)) == true);
    }

    TEST_CASE("RangeMonitor hysteresis suppresses boundary chatter") {
        RangeMonitor<double> ai{0.25, 0.5, 4.5, 4.75, /*hysteresis=*/0.05};
        CHECK(ai(2.5) == SignalStatus::Valid);
        // Just below valid_lo: without hysteresis this would flip to UnderRange,
        // but it must drop a further 0.05 (below 0.45) to leave Valid.
        CHECK(ai(0.48) == SignalStatus::Valid);
        CHECK(ai(0.44) == SignalStatus::UnderRange); // crossed valid_lo - hyst
        // And to recover it must rise back above valid_lo + hyst = 0.55.
        CHECK(ai(0.52) == SignalStatus::UnderRange);
        CHECK(ai(0.56) == SignalStatus::Valid);
    }

    TEST_CASE("Hysteresis latches with separate trip/release thresholds") {
        Hysteresis<double> h{1.0, 2.0}; // release < 1, trip > 2

        CHECK(h(0.0) == false);
        CHECK(h(1.5) == false); // in the band, still low
        CHECK(h(2.5) == true);  // trips above high
        CHECK(h(1.5) == true);  // holds in the band
        CHECK(h(0.5) == false); // releases below low
        CHECK(h(1.5) == false); // holds low again
    }
}
