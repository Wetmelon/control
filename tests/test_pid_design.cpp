#include <cmath>
#include <numbers>

#include "doctest.h"
#include "pid_design.hpp"

using namespace wetmelon::control;

/**
 * @brief Tests for modelless PID design methods (pid_design.hpp)
 *
 * Validates Ziegler-Nichols, Cohen-Coon, SIMC, Lambda, bandwidth,
 * Tyreus-Luyben, and pole placement PID tuning methods.
 */

TEST_SUITE("PID Design - Ziegler-Nichols Ultimate Gain") {
    TEST_CASE("ZN PID tuning from Ku and Tu") {
        constexpr double Ku = 10.0;
        constexpr double Tu = 2.0;
        constexpr double Ts = 0.01;

        constexpr auto result = design::ziegler_nichols(Ku, Tu, Ts);
        static_assert(result.Kp == 0.6 * Ku);             // 6.0
        static_assert(result.Ki == result.Kp / 1.0);      // Kp / (Tu/2) = 6.0
        static_assert(result.Kd == result.Kp * Tu / 8.0); // 6.0 * 0.25 = 1.5

        CHECK(result.Kp == doctest::Approx(6.0));
        CHECK(result.Ki == doctest::Approx(6.0));
        CHECK(result.Kd == doctest::Approx(1.5));
    }

    TEST_CASE("ZN PI tuning") {
        constexpr double Ku = 10.0;
        constexpr double Tu = 2.0;
        constexpr double Ts = 0.01;

        constexpr auto result = design::ziegler_nichols(Ku, Tu, Ts, design::PIDType::PI);
        CHECK(result.Kp == doctest::Approx(4.5)); // 0.45 * 10
        CHECK(result.Ki == doctest::Approx(4.5 / (2.0 / 1.2)));
        CHECK(result.Kd == doctest::Approx(0.0));
    }

    TEST_CASE("ZN P-only tuning") {
        constexpr auto result = design::ziegler_nichols(10.0, 2.0, 0.01, design::PIDType::P);
        CHECK(result.Kp == doctest::Approx(5.0));
        CHECK(result.Ki == doctest::Approx(0.0));
        CHECK(result.Kd == doctest::Approx(0.0));
    }

    TEST_CASE("Runtime ZN matches compile-time") {
        auto ct = design::ziegler_nichols(10.0, 2.0, 0.01, design::PIDType::PID);
        auto rt = online::ziegler_nichols(10.0, 2.0, 0.01, design::PIDType::PID);
        CHECK(ct.Kp == doctest::Approx(rt.Kp));
        CHECK(ct.Ki == doctest::Approx(rt.Ki));
        CHECK(ct.Kd == doctest::Approx(rt.Kd));
    }
}

TEST_SUITE("PID Design - Ziegler-Nichols Step Response") {
    TEST_CASE("ZN step response PID") {
        constexpr double K = 1.0;
        constexpr double L = 0.5;
        constexpr double tau = 2.0;
        constexpr double Ts = 0.01;

        constexpr auto result = design::ziegler_nichols_step(K, L, tau, Ts);
        // Kp = 1.2 * tau/(K*L) = 1.2 * 4.0 = 4.8
        CHECK(result.Kp == doctest::Approx(4.8));
        // Ki = Kp / (2*L) = 4.8 / 1.0 = 4.8
        CHECK(result.Ki == doctest::Approx(4.8));
        // Kd = Kp * 0.5 * L = 4.8 * 0.25 = 1.2
        CHECK(result.Kd == doctest::Approx(1.2));
    }
}

TEST_SUITE("PID Design - Tyreus-Luyben") {
    TEST_CASE("TL PID tuning") {
        constexpr double Ku = 10.0;
        constexpr double Tu = 2.0;
        constexpr double Ts = 0.01;

        constexpr auto result = design::tyreus_luyben(Ku, Tu, Ts);
        CHECK(result.Kp == doctest::Approx(Ku / 2.2));
        CHECK(result.Ki == doctest::Approx(result.Kp / (2.2 * Tu)));
        CHECK(result.Kd == doctest::Approx(result.Kp * Tu / 6.3));
    }

    TEST_CASE("TL PI tuning") {
        constexpr auto result = design::tyreus_luyben(10.0, 2.0, 0.01, design::PIDType::PI);
        CHECK(result.Kp == doctest::Approx(10.0 / 3.2));
    }

    TEST_CASE("TL more conservative than ZN") {
        constexpr auto zn = design::ziegler_nichols(10.0, 2.0, 0.01);
        constexpr auto tl = design::tyreus_luyben(10.0, 2.0, 0.01);
        // Tyreus-Luyben should have lower Kp (more conservative)
        CHECK(tl.Kp < zn.Kp);
    }
}

TEST_SUITE("PID Design - Cohen-Coon") {
    TEST_CASE("Cohen-Coon PID for FOPDT") {
        constexpr double K = 1.0;
        constexpr double L = 1.0;
        constexpr double tau = 4.0;
        constexpr double Ts = 0.01;

        constexpr auto result = design::cohen_coon(K, L, tau, Ts);
        // r = L/tau = 0.25
        // a = tau/(K*L) = 4.0
        // Kp = a * (4/3 + r/4) = 4.0 * (1.333 + 0.0625) = 5.583
        double r = L / tau;
        double a = tau / (K * L);
        double expected_Kp = a * (4.0 / 3.0 + r / 4.0);
        CHECK(result.Kp == doctest::Approx(expected_Kp).epsilon(1e-10));
        CHECK(result.Ki > 0.0);
        CHECK(result.Kd > 0.0);
    }

    TEST_CASE("Runtime Cohen-Coon matches compile-time") {
        auto ct = design::cohen_coon(1.0, 1.0, 4.0, 0.01);
        auto rt = online::cohen_coon(1.0, 1.0, 4.0, 0.01);
        CHECK(ct.Kp == doctest::Approx(rt.Kp).epsilon(1e-12));
        CHECK(ct.Ki == doctest::Approx(rt.Ki).epsilon(1e-12));
        CHECK(ct.Kd == doctest::Approx(rt.Kd).epsilon(1e-12));
    }
}

TEST_SUITE("PID Design - SIMC") {
    TEST_CASE("SIMC PI tuning") {
        constexpr double K = 2.0;
        constexpr double L = 0.5;
        constexpr double tau = 3.0;
        constexpr double tau_c = 1.0;
        constexpr double Ts = 0.01;

        constexpr auto result = design::simc(K, L, tau, tau_c, Ts, design::PIDType::PI);
        // Kp = tau / (K * (tau_c + L)) = 3.0 / (2.0 * 1.5) = 1.0
        CHECK(result.Kp == doctest::Approx(1.0).epsilon(1e-12));
        CHECK(result.Ki > 0.0);
        CHECK(result.Kd == doctest::Approx(0.0));
    }
}

TEST_SUITE("PID Design - Lambda Tuning") {
    TEST_CASE("Lambda PI tuning") {
        constexpr double K = 1.0;
        constexpr double L = 0.5;
        constexpr double tau = 2.0;
        constexpr double lambda = 2.0; // Desired closed-loop time constant
        constexpr double Ts = 0.01;

        constexpr auto result = design::lambda_tuning(K, L, tau, lambda, Ts);
        // Kp = tau / (K * (lambda + L)) = 2.0 / (1.0 * 2.5) = 0.8
        CHECK(result.Kp == doctest::Approx(0.8).epsilon(1e-12));
        // Ki = Kp / tau = 0.8 / 2.0 = 0.4
        CHECK(result.Ki == doctest::Approx(0.4).epsilon(1e-12));
        CHECK(result.Kd == doctest::Approx(0.0));
    }
}

TEST_SUITE("PID Design - Bandwidth") {
    TEST_CASE("PI from bandwidth and phase margin") {
        // PI controller phase range: [0, -90°), so max phase margin < 90°
        // With 30° phase margin: desired_phase = 30° - 180° = -150°
        // But controller phase = atan2(-Ki/ω, Kp) is in [-90°, 0°]
        // Achievable range means we set the controller to contribute phase
        // such that phase margin target is met assuming unit-gain plant.
        constexpr auto result = design::pid_from_bandwidth(
            10.0, 30.0, 0.001, design::PIDType::PI
        );
        CHECK(result.Kp > 0.0);
        CHECK(result.Ki > 0.0);
        CHECK(result.Kd == doctest::Approx(0.0));
    }

    TEST_CASE("PID from bandwidth") {
        constexpr auto result = design::pid_from_bandwidth(
            50.0, 45.0, 0.001, design::PIDType::PID
        );
        CHECK(result.Kp > 0.0);
        CHECK(result.Ki > 0.0);
        CHECK(result.Kd > 0.0);
    }
}

TEST_SUITE("PID Design - Pole Placement") {
    TEST_CASE("PI pole placement for first-order plant") {
        // Plant: G(s) = 1/(s+1), K=1, tau=1
        // Desired poles at z=0.5 and z=0.3 (well inside unit circle)
        constexpr double K = 1.0;
        constexpr double tau = 1.0;
        constexpr double Ts = 0.1;
        constexpr double p1 = 0.5;
        constexpr double p2 = 0.3;

        constexpr auto result = design::pid_pole_placement(K, tau, p1, p2, Ts);
        CHECK(result.Kp > 0.0);
        CHECK(result.Ki > 0.0);
        CHECK(result.Kd == doctest::Approx(0.0)); // PI only

        // Verify: closed-loop poles should be at p1, p2
        // Discretize plant: a = exp(-Ts/tau), b = K*(1-a)
        double a = std::exp(-Ts / tau);
        double b = K * (1.0 - a);

        // Closed-loop char poly: z² + (b*Kp - a - 1)*z + (a - b*Kp + b*Ki*Ts)
        double c1 = b * result.Kp - a - 1.0;
        double c0 = a - b * result.Kp + b * result.Ki * Ts;

        // Should equal desired: z² - (p1+p2)*z + p1*p2
        CHECK(c1 == doctest::Approx(-(p1 + p2)).epsilon(1e-10));
        CHECK(c0 == doctest::Approx(p1 * p2).epsilon(1e-10));
    }

    TEST_CASE("PID pole placement for first-order plant (3 poles)") {
        constexpr double K = 2.0;
        constexpr double tau = 0.5;
        constexpr double Ts = 0.01;
        constexpr double p1 = 0.8;
        constexpr double p2 = 0.7;
        constexpr double p3 = 0.6;

        constexpr auto result = design::pid_pole_placement(K, tau, p1, p2, p3, Ts);
        CHECK(result.Kp != 0.0);
        CHECK(result.Ki != 0.0);
        CHECK(result.Kd != 0.0);
    }

    TEST_CASE("Pole placement at origin gives deadbeat") {
        // All poles at z=0 → deadbeat response
        constexpr double K = 1.0;
        constexpr double tau = 1.0;
        constexpr double Ts = 0.1;

        constexpr auto result = design::pid_pole_placement(K, tau, 0.0, 0.0, Ts);

        double a = std::exp(-Ts / tau);
        double b = K * (1.0 - a);
        double c1 = b * result.Kp - a - 1.0;
        double c0 = a - b * result.Kp + b * result.Ki * Ts;

        // Both poles at 0: z² - 0*z + 0 → c1 = 0, c0 = 0
        CHECK(c1 == doctest::Approx(0.0).epsilon(1e-10));
        CHECK(c0 == doctest::Approx(0.0).epsilon(1e-10));
    }
}

TEST_SUITE("PID Design - Type Conversion") {
    TEST_CASE("Design result converts to PIDController") {
        constexpr auto result = design::ziegler_nichols(10.0, 2.0, 0.01);
        PIDController  controller(result.as<float>());
        CHECK(controller.Kp == doctest::Approx(6.0f).epsilon(1e-4));
        CHECK(controller.Ki == doctest::Approx(6.0f).epsilon(1e-4));
        CHECK(controller.Kd == doctest::Approx(1.5f).epsilon(1e-4));
    }

    TEST_CASE("Runtime design feeds into PIDController") {
        auto          result = online::simc(1.0, 0.5, 2.0, 1.0, 0.01);
        PIDController controller(result);
        float         u = controller.control(1.0);
        CHECK(u != 0.0f); // Non-zero output for non-zero error
    }
}
