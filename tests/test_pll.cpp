#include <cmath>
#include <numbers>

#include "wet/backend.hpp"
#include "wet/filters/pll.hpp"
#include "wet/math/math.hpp"
#include "wet/matrix/matrix.hpp"
#include "wet/transforms.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wet;

TEST_SUITE("Single-Phase PLL") {
    TEST_CASE("Constructor initializes limits and outputs") {
        constexpr float Fnom = 50.0f;

        constexpr SinglePhasePLL<float> pll(Fnom);

        // Loop-filter output (frequency offset about nominal) is clamped to ±50%.
        static_assert(pll.loop_filter.u_max == Fnom * 0.5f);
        static_assert(pll.loop_filter.u_min == -Fnom * 0.5f);
        static_assert(pll.loop_filter.Kbc == 10.0f); // Kbc = Kp anti-windup constant
        static_assert(pll.frequency() == Fnom);
        static_assert(pll.phase() == 0.0f);

        CHECK(pll.loop_filter.Kp == doctest::Approx(10.0f));
        CHECK(pll.loop_filter.Ki == doctest::Approx(100.0f));
        CHECK(pll.frequency() == doctest::Approx(Fnom));
        CHECK(pll.phase() == doctest::Approx(0.0f));
    }

    TEST_CASE("Step keeps estimates finite and within configured bounds") {
        constexpr float Fnom = 50.0f;
        constexpr float Ts = 0.0001f;

        SinglePhasePLL<float> pll(Fnom);
        const float           two_pi = 2.0f * std::numbers::pi_v<float>;

        for (int i = 0; i < 5000; ++i) {
            const float t = static_cast<float>(i) * Ts;
            const float input = std::sin(two_pi * Fnom * t);
            pll.step(input, Ts);
        }

        CHECK(std::isfinite(pll.frequency()));
        CHECK(std::isfinite(pll.phase()));
        CHECK(pll.frequency() <= Fnom * 1.5f);                             // nominal + max offset
        CHECK(pll.frequency() >= Fnom * 0.5f);                             // nominal − max offset
        CHECK(std::abs(pll.phase()) <= std::numbers::pi_v<float> + 1e-4f); // wrapped to [-π, π)
        CHECK(pll.frequency() == doctest::Approx(Fnom).epsilon(0.25f));
    }

    TEST_CASE("Tracks an off-nominal tone in the correct direction") {
        // Regression guard for the SOGI-mixer phase-detector sign: a faster input
        // must drive the estimate UP toward it, not down to the lower frequency
        // rail (the bug the +input·quadrature sign produced).
        constexpr float Fnom = 50.0f;
        constexpr float Fin = 55.0f; // above nominal
        constexpr float Ts = 0.001f;
        const float     two_pi = 2.0f * std::numbers::pi_v<float>;

        SinglePhasePLL<float> pll(Fnom);

        float phase = 0.0f;
        for (int i = 0; i < 30000; ++i) {
            phase += two_pi * Fin * Ts;
            pll.step(std::sin(phase), Ts);
        }

        CHECK(pll.frequency() > Fnom + 1.0f); // moved toward the input, not the rail
        CHECK(pll.frequency() == doctest::Approx(Fin).epsilon(0.1f));
    }

    TEST_CASE("Integrator leak defaults to zero (pure integrator)") {
        constexpr float Fnom = 50.0f;

        constexpr SinglePhasePLL<float> pll(Fnom);
        static_assert(pll.integrator_leak == 0.0f);
    }

    TEST_CASE("Integrator leak bleeds a parked offset back toward zero") {
        constexpr float Fnom = 50.0f;
        constexpr float Fin = 55.0f; // off-nominal drive
        constexpr float Ts = 0.001f;
        const float     two_pi = 2.0f * std::numbers::pi_v<float>;

        SinglePhasePLL<float> pll(Fnom); // gentle default gains (stable for 1-phase)

        // Phase 1 (no leak): an off-nominal tone drives the estimator away from
        // nominal, parking a frequency offset in the integrator.
        float phase = 0.0f;
        for (int i = 0; i < 15000; ++i) {
            phase += two_pi * Fin * Ts;
            pll.step(std::sin(phase), Ts);
        }
        const float freq_charged = pll.frequency();
        CHECK(freq_charged > Fnom + 0.5f); // offset parked, and in the correct direction

        // Phase 2: enable the leak and remove excitation. phase_error → 0, so the
        // only dynamics are the leak, which pulls the estimate back to nominal.
        pll.integrator_leak = 50.0f; // [1/s] → time constant 20 ms
        for (int i = 0; i < 15000; ++i) {
            pll.step(0.0f, Ts);
        }
        const float freq_bled = pll.frequency();

        CHECK(wet::abs(freq_bled - Fnom) < wet::abs(freq_charged - Fnom));
        CHECK(freq_bled == doctest::Approx(Fnom).epsilon(0.05f));
    }

    TEST_CASE("Reset restores nominal frequency and zero phase") {
        constexpr float Fnom = 60.0f;
        constexpr float Ts = 0.0001f;

        SinglePhasePLL<float> pll(Fnom);

        // Drive the estimator away from initial state.
        for (int i = 0; i < 1000; ++i) {
            pll.step(0.5f, Ts);
        }

        pll.reset();

        CHECK(pll.frequency() == doctest::Approx(Fnom));
        CHECK(pll.phase() == doctest::Approx(0.0f));
    }
}

TEST_SUITE("Three-Phase PLL") {
    TEST_CASE("Locks frequency and phase on a balanced set") {
        constexpr float Fnom = 50.0f;
        constexpr float Ts = 0.0001f;
        constexpr float two_pi = 2.0f * std::numbers::pi_v<float>;
        constexpr float Fin = 55.0f; // off-nominal, within ±50% lock range

        ThreePhasePLL<float> pll(Fnom);
        pll.loop_filter.Kp = 40.0f;   // PI loop tuned for ~5 Hz natural frequency,
        pll.loop_filter.Ki = 1000.0f; // ζ≈0.6 (defaults are too slow to pull 5 Hz)

        float phase = 0.7f; // arbitrary input phase offset
        for (int i = 0; i < 20000; ++i) {
            phase += two_pi * Fin * Ts;
            const ColVec<3, float> abc = {
                std::cos(phase),
                std::cos(phase - (two_pi / 3.0f)),
                std::cos(phase + (two_pi / 3.0f)),
            };
            pll.step(abc, Ts);
        }

        CHECK(pll.frequency() == doctest::Approx(Fin).epsilon(0.01f));

        // Phase estimate tracks the input phase (compare on the circle so the
        // [-π, π) wrap boundary doesn't trip the check).
        CHECK(std::cos(pll.phase() - phase) == doctest::Approx(1.0f).epsilon(0.02f));
    }

    TEST_CASE("Reset restores nominal frequency and zero phase") {
        constexpr float Fnom = 60.0f;
        constexpr float Ts = 0.0001f;

        ThreePhasePLL<float> pll(Fnom);
        pll.step({1.0f, -0.5f, -0.5f}, Ts);
        pll.reset();

        CHECK(pll.frequency() == doctest::Approx(Fnom));
        CHECK(pll.phase() == doctest::Approx(0.0f));
    }
}

TEST_SUITE("Sensorless Estimator") {
    TEST_CASE("Locks angle and speed from back-EMF (open-circuit, zero current)") {
        constexpr float Ts = 1e-5f;
        constexpr float lambda = 1.6e-3f;
        constexpr float two_pi = 2.0f * std::numbers::pi_v<float>;
        constexpr float w_e = two_pi * 50.0f; // electrical rad/s

        SensorlessEstimator<float> est({
            .phase_resistance = 0.05f,
            .phase_inductance = 20e-6f,
            .pm_flux_linkage = lambda,
            .observer_gain = 1000.0f,
            .pll_bandwidth = 1000.0f,
            .pole_pairs = 7.0f,
        });

        float theta = 0.0f;
        bool  ok = true;
        for (int i = 0; i < 20000; ++i) {
            // Open-circuit SPMSM: i = 0, so the terminal voltage is the back-EMF.
            const AlphaBeta<float> i_ab = {0.0f, 0.0f};
            const AlphaBeta<float> v_ab = {-w_e * lambda * std::sin(theta), w_e * lambda * std::cos(theta)};
            ok = est.update(i_ab, v_ab, Ts) && ok;
            theta = std::fmod(theta + (w_e * Ts), two_pi);
        }

        CHECK(ok);
        CHECK(est.electrical_velocity() == doctest::Approx(w_e).epsilon(0.02f));
        CHECK(est.mechanical_velocity() == doctest::Approx(w_e / 7.0f).epsilon(0.02f));
        CHECK(est.pm_flux().abs() == doctest::Approx(lambda).epsilon(0.05f));
        // Angle tracks the rotor (compare on the circle to dodge wrap edges).
        const float ref = std::remainder(theta, two_pi);
        CHECK(std::cos(est.phase() - ref) == doctest::Approx(1.0f).epsilon(0.01f));
    }

    TEST_CASE("Unstable PLL gain for the sample time fails") {
        SensorlessEstimator<float> est({.pm_flux_linkage = 1.0f, .pll_bandwidth = 1000.0f});
        // Ts·(2·bw) = 0.01·2000 = 20 ≥ 1 → discrete loop unstable.
        CHECK_FALSE(est.update({0.0f, 0.0f}, {0.0f, 0.0f}, 0.01f));
    }

    TEST_CASE("Fuses a sensor angle at standstill (no back-EMF)") {
        constexpr float Ts = 1e-4f;
        constexpr float theta0 = 1.2f;

        SensorlessEstimator<float> est({
            .pm_flux_linkage = 1.6e-3f,
            .observer_gain = 1000.0f,
            .pll_bandwidth = 500.0f,
            .pole_pairs = 7.0f,
            .fusion_blend_speed = 50.0f, // trust the sensor below 50 elec rad/s
        });

        // Rotor held: no current, no back-EMF — only the sensor carries the angle.
        for (int i = 0; i < 5000; ++i) {
            est.update({0.0f, 0.0f}, {0.0f, 0.0f}, Ts, theta0);
        }
        CHECK(std::cos(est.phase() - theta0) == doctest::Approx(1.0f).epsilon(0.01f));
    }
}
