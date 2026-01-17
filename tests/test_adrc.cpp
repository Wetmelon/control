#include "adrc.hpp"
#include "doctest.h"

using namespace wetmelon::control;

TEST_SUITE("Active Disturbance Rejection Control (ADRC)") {
    TEST_CASE("1st-order ADRC gain computation") {
        constexpr size_t NX = 1;

        double wc = 5.0;  // Controller bandwidth
        double wo = 20.0; // Observer bandwidth
        double b0 = 1.0;  // Plant gain
        double Ts = 0.01; // Sampling time

        auto adrc_result = online::adrc<NX>(wc, wo, b0, Ts);

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
        double Ts = 0.01; // Sampling time

        auto adrc_result = online::adrc<NX>(wc, wo, b0, Ts);

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
}
