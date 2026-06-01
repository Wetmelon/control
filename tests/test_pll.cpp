#include <cmath>
#include <numbers>

#include "wet/controllers/pll.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wetmelon::control;

TEST_SUITE("Single-Phase PLL") {
    TEST_CASE("Constructor initializes limits and outputs") {
        constexpr float Fnom = 50.0f;
        constexpr float alpha = 1.414f;
        constexpr float Ts = 0.0001f;

        constexpr SinglePhasePLL<float> pll(Fnom, alpha, Ts);

        static_assert(pll.params.integrator_max == Fnom * 0.5f);
        static_assert(pll.params.integrator_min == -Fnom * 0.5f);
        static_assert(pll.params.output_max == Fnom * 1.5f);
        static_assert(pll.params.output_min == Fnom * 0.5f);
        static_assert(pll.frequency() == Fnom);
        static_assert(pll.phase() == 0.0f);

        CHECK(pll.params.Kp == doctest::Approx(10.0f * Ts));
        CHECK(pll.params.Ki == doctest::Approx(100.0f * Ts));
        CHECK(pll.frequency() == doctest::Approx(Fnom));
        CHECK(pll.phase() == doctest::Approx(0.0f));
    }

    TEST_CASE("Step keeps estimates finite and within configured bounds") {
        constexpr float Fnom = 50.0f;
        constexpr float alpha = 1.414f;
        constexpr float Ts = 0.0001f;

        SinglePhasePLL<float> pll(Fnom, alpha, Ts);
        const float           two_pi = 2.0f * std::numbers::pi_v<float>;

        for (int i = 0; i < 5000; ++i) {
            const float t = static_cast<float>(i) * Ts;
            const float input = std::sin(two_pi * Fnom * t);
            pll.step(input, Ts);
        }

        CHECK(std::isfinite(pll.frequency()));
        CHECK(std::isfinite(pll.phase()));
        CHECK(pll.frequency() <= pll.params.output_max);
        CHECK(pll.frequency() >= pll.params.output_min);
        CHECK(pll.phase() >= 0.0f);
        CHECK(pll.phase() < two_pi);
        CHECK(pll.frequency() == doctest::Approx(Fnom).epsilon(0.25f));
    }

    TEST_CASE("Integrator leak defaults to zero (pure integrator)") {
        constexpr float Fnom = 50.0f;
        constexpr float alpha = 1.414f;
        constexpr float Ts = 0.0001f;

        constexpr SinglePhasePLL<float> pll(Fnom, alpha, Ts);
        static_assert(pll.params.integrator_leak == 0.0f);
    }

    TEST_CASE("Integrator leak bleeds a parked offset back toward zero") {
        // Drive the integrator to a non-zero state with a constant phase-error
        // surrogate, then with leak enabled and zero excitation it must decay.
        constexpr float Fnom = 50.0f;
        constexpr float alpha = 1.414f;
        constexpr float Ts = 0.001f;

        SinglePhasePLL<float> pll(Fnom, alpha, Ts);
        pll.params.integrator_leak = 50.0f; // [1/s] → time constant 20 ms

        // Park an offset in the integrator by hand (white-box: it's a public-ish
        // state via the step dynamics). Use a constant input that biases the
        // phase detector so the integrator charges up.
        for (int i = 0; i < 2000; ++i) {
            pll.step(1.0f, Ts);
        }
        const float freq_charged = pll.frequency();

        // Now feed zero input: phase_error = input*quadrature = 0, so the only
        // dynamics on the integrator are the leak, which pulls it toward 0
        // (estimate toward nominal).
        for (int i = 0; i < 2000; ++i) {
            pll.step(0.0f, Ts);
        }
        const float freq_bled = pll.frequency();

        // After ~100 time constants of leak with no excitation, the estimate
        // must have moved back toward nominal relative to the charged state.
        CHECK(wet::abs(freq_bled - Fnom) < wet::abs(freq_charged - Fnom));
        CHECK(freq_bled == doctest::Approx(Fnom).epsilon(0.05f));
    }

    TEST_CASE("Reset restores nominal frequency and zero phase") {
        constexpr float Fnom = 60.0f;
        constexpr float alpha = 1.2f;
        constexpr float Ts = 0.0001f;

        SinglePhasePLL<float> pll(Fnom, alpha, Ts);

        // Drive the estimator away from initial state.
        for (int i = 0; i < 1000; ++i) {
            pll.step(0.5f, Ts);
        }

        pll.reset();

        CHECK(pll.frequency() == doctest::Approx(Fnom));
        CHECK(pll.phase() == doctest::Approx(0.0f));
    }
}
