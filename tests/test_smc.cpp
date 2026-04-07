#include "doctest.h"
#include "smc.hpp"

using namespace wetmelon::control;

TEST_SUITE("Sliding Mode Control (SMC)") {
    TEST_CASE("SMC gain computation") {
        float lambda = 10.0f; // Sliding surface parameter
        float k = 5.0f;       // Switching gain
        float b0 = 1.0f;      // Plant gain
        float Ts = 0.01f;     // Sampling time

        auto smc_result = online::smc(lambda, k, b0, Ts);

        CHECK(smc_result.lambda == doctest::Approx(lambda).epsilon(1e-6));
        CHECK(smc_result.k == doctest::Approx(k).epsilon(1e-6));
        CHECK(smc_result.b0 == doctest::Approx(b0).epsilon(1e-6));
        CHECK(smc_result.Ts == doctest::Approx(Ts).epsilon(1e-6));
    }

    TEST_CASE("SMC controller basic functionality") {
        float lambda = 10.0f;
        float k = 5.0f;
        float b0 = 1.0f;
        float Ts = 0.01f;

        auto          smc_result = online::smc(lambda, k, b0, Ts);
        SMCController controller(smc_result);

        // Test control with zero error
        float r = 0.0f;
        float y = 0.0f;
        float u = controller.control(r, y);
        CHECK(u == 0.0f); // s = 0, sign(0) = 0, u = 0

        // Test control with positive error
        r = 1.0f;
        y = 0.0f;
        u = controller.control(r, y);
        CHECK(u == doctest::Approx(-k / b0).epsilon(1e-6)); // s > 0, u = -k/b0

        // Test control with negative error
        r = 0.0f;
        y = 1.0f;
        u = controller.control(r, y);
        CHECK(u == doctest::Approx(k / b0).epsilon(1e-6)); // s < 0, u = k/b0
    }
}
