#include <cmath>
#include <numbers>

#include "wet/backend.hpp"
#include "wet/filters/pll.hpp"
#include "wet/matrix/matrix.hpp"
#include "wet/transforms.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wet;

namespace {

// Build a balanced three-phase set at the given amplitude/phase and Clarke it.
AlphaBeta<float> grid_ab(float amp, float wt) {
    const float            tp = 2.0f * std::numbers::pi_v<float> / 3.0f;
    const ColVec<3, float> abc = {amp * std::cos(wt), amp * std::cos(wt - tp), amp * std::cos(wt + tp)};
    return clarke_transform(abc);
}

} // namespace

TEST_SUITE("DSOGI-PLL") {

    TEST_CASE("Instantaneous sequence calculator: balanced input is pure positive") {
        // v = (1, 0); its 90° lag qv = (0, -1) for a positive-sequence rotation.
        const AlphaBeta<float> v = {1.0f, 0.0f};
        const AlphaBeta<float> qv = {0.0f, -1.0f};

        const auto pos = positive_sequence_ab(v, qv);
        const auto neg = negative_sequence_ab(v, qv);

        CHECK(pos.alpha == doctest::Approx(1.0f));
        CHECK(pos.beta == doctest::Approx(0.0f));
        CHECK(neg.abs() == doctest::Approx(0.0f).epsilon(1e-6f));
    }

    TEST_CASE("Locks frequency and phase on a balanced 50 Hz grid") {
        const float f0 = 50.0f;
        const float Ts = 1.0f / 10000.0f; // 10 kHz
        const float w = 2.0f * std::numbers::pi_v<float> * f0;

        DsogiPll<float> pll(f0);

        // Run for ~0.5 s to settle.
        float wt = 0.0f;
        for (int i = 0; i < 5000; ++i) {
            pll.step(grid_ab(1.0f, wt), Ts);
            wt += w * Ts;
        }

        CHECK(pll.frequency() == doctest::Approx(f0).epsilon(0.01f));
        // At lock the positive-sequence dq aligns: d → amplitude, q → 0.
        CHECK(pll.positive_dq().d == doctest::Approx(1.0f).epsilon(0.02f));
        CHECK(pll.positive_dq().q == doctest::Approx(0.0f).epsilon(0.02f));
        // Balanced input ⇒ negligible negative sequence.
        CHECK(pll.negative_sequence().abs() == doctest::Approx(0.0f).epsilon(0.02f));
    }

    TEST_CASE("Extracts positive sequence from an unbalanced grid") {
        const float f0 = 50.0f;
        const float Ts = 1.0f / 10000.0f;
        const float w = 2.0f * std::numbers::pi_v<float> * f0;

        DsogiPll<float> pll(f0);

        // Unbalanced: inject a negative-sequence component by scaling one phase.
        const float tp = 2.0f * std::numbers::pi_v<float> / 3.0f;
        float       wt = 0.0f;
        for (int i = 0; i < 6000; ++i) {
            const ColVec<3, float> abc = {std::cos(wt), 0.6f * std::cos(wt - tp), std::cos(wt + tp)};
            pll.step(clarke_transform(abc), Ts);
            wt += w * Ts;
        }

        // Still locks to the line frequency despite unbalance.
        CHECK(pll.frequency() == doctest::Approx(f0).epsilon(0.02f));
        // Negative sequence is non-trivial (unbalance is detected).
        CHECK(pll.negative_sequence().abs() > 0.05f);
    }

} // TEST_SUITE
