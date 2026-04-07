#include <cmath>
#include <numbers>

#include "doctest.h"
#include "pr.hpp"

using namespace wetmelon::control;

/**
 * @brief Tests for Proportional-Resonant controller (pr.hpp)
 *
 * Validates PR design, discretization, and harmonic tracking.
 */

TEST_SUITE("PR Controller") {
    TEST_CASE("PR design result") {
        constexpr double w0 = 2.0 * std::numbers::pi * 50.0; // 50 Hz
        constexpr double wc = 10.0;
        constexpr double Ts = 1.0 / 10000.0; // 10 kHz

        constexpr auto result = design::pr(1.0, 100.0, w0, wc, Ts);
        static_assert(result.Kp == 1.0);
        static_assert(result.Ki == 100.0);
        CHECK(result.w0 == doctest::Approx(w0));
        CHECK(result.wc == doctest::Approx(wc));
    }

    TEST_CASE("PR controller tracks sinusoidal reference") {
        // PR at 50 Hz should achieve zero steady-state error for 50 Hz sine
        double w0 = 2.0 * std::numbers::pi * 50.0;
        double wc = 10.0;
        double Ts = 1.0 / 10000.0;

        auto                 result = online::pr(1.0, 200.0, w0, wc, Ts);
        PRController<double> ctrl(result);

        // Simulate: reference = sin(w0*t), plant = unit gain (identity)
        double y = 0.0;
        double error_sum = 0.0;
        size_t n_steps = 20000; // 2 seconds at 10 kHz

        for (size_t k = 0; k < n_steps; ++k) {
            double t = k * Ts;
            double ref = std::sin(w0 * t);
            double error = ref - y;
            double u = ctrl.control(error);
            y = u; // Unit gain plant

            // Track error in last 0.5 seconds (settled)
            if (k > 15000) {
                error_sum += std::abs(error);
            }
        }

        double avg_error = error_sum / 5000.0;
        // Should have very small tracking error at steady state
        CHECK(avg_error < 0.05);
    }

    TEST_CASE("PR controller type conversion") {
        constexpr auto      result = design::pr(1.0, 100.0, 314.0, 10.0, 0.0001);
        PRController<float> ctrl(result.as<float>());
        CHECK(ctrl.Kp == doctest::Approx(1.0f));
        CHECK(ctrl.w0 == doctest::Approx(314.0f).epsilon(0.1));
    }

    TEST_CASE("PR controller reset") {
        auto                 result = online::pr(1.0, 100.0, 314.0, 10.0, 0.0001);
        PRController<double> ctrl(result);

        // Run some steps
        (void)ctrl.control(1.0);
        (void)ctrl.control(0.5);
        (void)ctrl.control(0.3);

        ctrl.reset();
        // After reset, output should be just Kp * error (no resonant history)
        double u = ctrl.control(1.0);
        // First sample after reset: resonant contribution is b0_r * 1.0
        // The Kp contribution is 1.0
        CHECK(u != 0.0);
    }

    TEST_CASE("PR frequency update") {
        auto                 result = online::pr(1.0, 100.0, 314.0, 10.0, 0.0001);
        PRController<double> ctrl(result);

        // Change to 60 Hz
        double w0_60 = 2.0 * std::numbers::pi * 60.0;
        ctrl.set_frequency(w0_60);
        CHECK(ctrl.w0 == doctest::Approx(w0_60));
    }

    TEST_CASE("Multi-harmonic PR design") {
        constexpr double                w_fund = 2.0 * std::numbers::pi * 50.0;
        constexpr double                wc = 10.0;
        constexpr double                Ts = 1.0 / 10000.0;
        constexpr std::array<size_t, 4> harmonics = {1, 3, 5, 7};

        constexpr auto results = design::pr_harmonics(1.0, 100.0, w_fund, wc, Ts, harmonics);

        // First harmonic should have Kp and Ki_fund
        CHECK(results[0].Kp == doctest::Approx(1.0));
        CHECK(results[0].Ki == doctest::Approx(100.0));
        CHECK(results[0].w0 == doctest::Approx(w_fund));

        // Third harmonic: Ki/3, w0*3
        CHECK(results[1].Kp == doctest::Approx(0.0));
        CHECK(results[1].Ki == doctest::Approx(100.0 / 3.0));
        CHECK(results[1].w0 == doctest::Approx(3.0 * w_fund));

        // 7th harmonic
        CHECK(results[3].w0 == doctest::Approx(7.0 * w_fund));
    }
}
