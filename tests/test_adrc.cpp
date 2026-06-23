#include <array>
#include <cstddef>

#include "wet/controllers/adrc.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wet;

TEST_SUITE("Active Disturbance Rejection Control (ADRC)") {
    TEST_CASE("1st-order ADRC gain computation") {
        constexpr size_t NX = 1;

        double wc = 5.0;  // Controller bandwidth
        double wo = 20.0; // Observer bandwidth
        double b0 = 1.0;  // Plant gain

        auto adrc_result = design::adrc<NX>(wc, wo, b0);

        // Expected ESO gains for 1st-order system with poles at -wo
        std::array<double, NX + 1> expected_beta = {
            2 * wo,
            wo * wo,
        };

        CHECK(adrc_result.beta.size() == expected_beta.size());
        for (size_t i = 0; i < expected_beta.size(); ++i) {
            CHECK(adrc_result.beta[i] == doctest::Approx(expected_beta[i]).epsilon(1e-6));
        }

        // Check Kp and Kd gains for 1st order
        double expected_Kp = wc / b0;
        double expected_Kd = 0.0;

        CHECK(adrc_result.Kp == doctest::Approx(expected_Kp).epsilon(1e-6));
        CHECK(adrc_result.Kd == doctest::Approx(expected_Kd).epsilon(1e-6));
    }

    TEST_CASE("2nd-order ADRC gain computation") {
        constexpr size_t NX = 2;

        double wc = 5.0;  // Controller bandwidth
        double wo = 20.0; // Observer bandwidth
        double b0 = 1.0;  // Plant gain

        auto adrc_result = design::adrc<NX>(wc, wo, b0);

        // Expected ESO gains for 2nd-order system with poles at -wo
        std::array<double, NX + 1> expected_beta = {
            3 * wo,
            3 * wo * wo,
            wo * wo * wo,
        };

        CHECK(adrc_result.beta.size() == expected_beta.size());
        for (size_t i = 0; i < expected_beta.size(); ++i) {
            CHECK(adrc_result.beta[i] == doctest::Approx(expected_beta[i]).epsilon(1e-6));
        }

        // Check Kp and Kd gains
        double expected_Kp = (wc * wc) / b0;
        double expected_Kd = (2 * wc) / b0;

        CHECK(adrc_result.Kp == doctest::Approx(expected_Kp).epsilon(1e-6));
        CHECK(adrc_result.Kd == doctest::Approx(expected_Kd).epsilon(1e-6));
    }

    TEST_CASE("1st-order runtime: tracks setpoint and rejects a constant disturbance") {
        const double Ts = 1e-3;
        auto         ctrl = ADRCController<1, double>(design::adrc<1>(20.0, 100.0, 1.0));

        // Plant: ẏ = −a·y + b·u + d. The −a·y term and the constant d are both
        // unmodeled, lumped into the ESO's total-disturbance estimate.
        const double a = 2.0, b = 1.0, d = 0.5, r = 1.0;
        double       y = 0.0;
        for (int k = 0; k < 20000; ++k) {
            const double u = ctrl.control(r, y, Ts);
            y += (-a * y + b * u + d) * Ts;
        }
        CHECK(y == doctest::Approx(r).epsilon(0.01)); // zero steady-state error
    }

    TEST_CASE("2nd-order runtime: double integrator tracks with disturbance rejection") {
        const double Ts = 1e-3;
        auto         ctrl = ADRCController<2, double>(design::adrc<2>(10.0, 50.0, 1.0));

        // Plant: ÿ = b·u + d (double integrator + constant load disturbance).
        const double b = 1.0, d = 0.3, r = 1.0;
        double       y = 0.0, v = 0.0;
        for (int k = 0; k < 30000; ++k) {
            const double u = ctrl.control(r, y, Ts);
            v += (b * u + d) * Ts;
            y += v * Ts;
        }
        CHECK(y == doctest::Approx(r).epsilon(0.01));   // converged to setpoint
        CHECK(v == doctest::Approx(0.0).epsilon(1e-3)); // and settled (no residual velocity)
    }

    TEST_CASE("reset clears the observer state") {
        const double Ts = 1e-3;
        auto         ctrl = ADRCController<1, double>(design::adrc<1>(20.0, 100.0, 1.0));
        for (int k = 0; k < 100; ++k) {
            (void)ctrl.control(1.0, 0.0, Ts);
        }
        ctrl.reset();
        // After reset the ESO state is zero, so the first command equals the
        // fresh-controller command for the same inputs.
        ADRCController<1, double> fresh(design::adrc<1>(20.0, 100.0, 1.0));
        CHECK(ctrl.control(1.0, 0.0, Ts) == doctest::Approx(fresh.control(1.0, 0.0, Ts)));
    }
}
