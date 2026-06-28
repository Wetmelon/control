
#include <cmath>

#include "doctest.h"
#include "wet/matrix/colvec.hpp"
#include "wet/motor/thermal.hpp"
#include "wet/systems/discretization.hpp"

using namespace wet;

TEST_SUITE("Thermal Derate") {

    TEST_CASE("derate_window ramps linearly between start and cutoff") {
        const auto d = derate_window(100.0, 120.0); // Lut1D, clamp extrapolation
        CHECK(d(80.0) == doctest::Approx(1.0));     // below start
        CHECK(d(100.0) == doctest::Approx(1.0));    // at start
        CHECK(d(110.0) == doctest::Approx(0.5));    // midpoint
        CHECK(d(120.0) == doctest::Approx(0.0));    // at cutoff
        CHECK(d(130.0) == doctest::Approx(0.0));    // above cutoff
    }

    TEST_CASE("a multi-segment derating curve interpolates each segment") {
        // Hold full current to 80°C, half-derate by 100°C, off by 120°C.
        const Lut1D<3, double> curve{.xs = {80.0, 100.0, 120.0}, .ys = {1.0, 0.5, 0.0}};
        CHECK(curve(90.0) == doctest::Approx(0.75));  // first segment midpoint
        CHECK(curve(110.0) == doctest::Approx(0.25)); // second segment midpoint
    }
}

TEST_SUITE("Thermal State-Space Models") {

    // Step a discrete junction-rise model under constant power P.
    template<std::size_t N>
    static double rise_after(const StateSpace<N, 1, 1, 0, 0, double>& sys, double P, int steps) {
        ColVec<N, double> x{};
        for (int k = 0; k < steps; ++k) {
            x = sys.A * x + sys.B * P;
        }
        return (sys.C * x)(0, 0);
    }

    TEST_CASE("Cauer single stage: ZOH samples the exact first-order response") {
        constexpr double R = 0.5;  // K/W
        constexpr double C = 0.01; // J/K -> tau = R*C = 5 ms
        constexpr double P = 100.0;
        constexpr double Ts = 1e-4;

        const auto sys = discretize(design::cauer_thermal_ss<1, double>({R}, {C}), Ts, DiscretizationMethod::ZOH);

        // ZOH is exact for constant input: at t = 1 tau (50 steps) the rise is
        // exactly P*R*(1 - e^-1), not an Euler approximation.
        CHECK(rise_after(sys, P, 50) == doctest::Approx(P * R * (1.0 - std::exp(-1.0))).epsilon(1e-6));
        CHECK(rise_after(sys, P, 5000) == doctest::Approx(P * R).epsilon(1e-9)); // steady state
    }

    TEST_CASE("Cauer multi-stage steady state is P times total resistance") {
        const auto sys = discretize(
            design::cauer_thermal_ss<3, double>({0.2, 0.3, 0.1}, {0.005, 0.02, 0.05}), 1e-4, DiscretizationMethod::ZOH
        );
        CHECK(rise_after(sys, 50.0, 200000) == doctest::Approx(50.0 * (0.2 + 0.3 + 0.1)).epsilon(1e-6));
    }

    TEST_CASE("Foster network steady state is P times total resistance") {
        // Datasheet form: (R_i, tau_i) pairs.
        const auto sys = discretize(
            design::foster_thermal_ss<2, double>({0.4, 0.2}, {0.01, 0.1}), 1e-4, DiscretizationMethod::ZOH
        );
        CHECK(rise_after(sys, 30.0, 500000) == doctest::Approx(30.0 * (0.4 + 0.2)).epsilon(1e-6));
    }

    TEST_CASE("models are constexpr (constinit-able)") {
        constexpr auto sys = design::cauer_thermal_ss<1, double>({0.5}, {0.01});
        static_assert(sys.is_continuous());
        static_assert(sys.B(0, 0) == 100.0); // 1/C
    }
}

TEST_SUITE("FET Loss Model") {

    TEST_CASE("conduction loss scales with I^2 * Rds * device_count") {
        FetLossModel<double> m{.rds_on = 0.005, .device_count = 6.0};
        // 6 * 10^2 * 0.005 = 3 W; no switching term configured.
        CHECK(m.loss(10.0, 48.0, 25.0) == doctest::Approx(6.0 * 100.0 * 0.005));
    }

    TEST_CASE("rds_on tempco raises conduction loss with junction temperature") {
        FetLossModel<double> m{.rds_on = 0.005, .rds_on_tempco = 0.005, .t_ref = 25.0, .device_count = 1.0};
        // At 125 C: rds = 0.005 * (1 + 0.005*100) = 0.0075.
        CHECK(m.loss(10.0, 48.0, 125.0) == doctest::Approx(100.0 * 0.0075));
    }

    TEST_CASE("switching loss scales with f_sw, energy, and operating point") {
        FetLossModel<double> m{
            .sw_energy = 1e-4,
            .v_ref = 48.0,
            .i_ref = 10.0,
            .f_sw = 20000.0,
            .device_count = 1.0
        };
        // f_sw * Esw * (Vdc/vref) * (i/iref) = 20000 * 1e-4 * 1 * 1 = 2 W (rds_on = 0).
        CHECK(m.loss(10.0, 48.0, 25.0) == doctest::Approx(2.0));
    }
}

TEST_SUITE("Junction Estimator") {

    TEST_CASE("Tj settles to case temperature plus loss times thermal resistance") {
        FetLossModel<double>         loss{.rds_on = 0.01, .device_count = 1.0}; // P = i^2 * 0.01
        const auto                   sys = discretize(design::cauer_thermal_ss<1, double>({0.8}, {0.02}), 1e-4, DiscretizationMethod::ZOH);
        JunctionEstimator<1, double> est{loss, sys};

        for (int k = 0; k < 100000; ++k) {
            est.step(20.0, 48.0, 40.0); // i_rms=20 -> P = 400 * 0.01 = 4 W, case = 40 °C
        }
        // Tj -> case + P * Rth = 40 + 4 * 0.8 = 43.2 °C.
        CHECK(est.junction_temperature() == doctest::Approx(43.2).epsilon(1e-6));
    }

    TEST_CASE("Tj exceeds the measured case under load") {
        FetLossModel<double> loss{.rds_on = 0.02, .device_count = 6.0};
        const auto           sys = discretize(
            design::cauer_thermal_ss<2, double>({0.3, 0.4}, {0.01, 0.05}), 1e-3, DiscretizationMethod::ZOH
        );
        JunctionEstimator<2, double> est{loss, sys};
        for (int k = 0; k < 50; ++k) {
            est.step(15.0, 48.0, 60.0);
        }
        CHECK(est.junction_temperature() > 60.0); // hotter than the case NTC reads
    }

    TEST_CASE("accepts a simple resistive loss model (weak datasheet)") {
        ResistiveLossModel<double>                               loss{.loss_resistance = 0.01}; // P = i^2 * 0.01
        const auto                                               sys = discretize(design::cauer_thermal_ss<1, double>({0.8}, {0.02}), 1e-4, DiscretizationMethod::ZOH);
        JunctionEstimator<1, double, ResistiveLossModel<double>> est{loss, sys};
        for (int k = 0; k < 100000; ++k) {
            est.step(20.0, 48.0, 40.0); // P = 400 * 0.01 = 4 W
        }
        CHECK(est.junction_temperature() == doctest::Approx(43.2).epsilon(1e-6)); // 40 + 4*0.8
    }
}

TEST_SUITE("Thermal Limiter") {

    TEST_CASE("derates over the curve and faults past fault_temp") {
        // Window 100->120 °C, hard fault at 125 °C.
        ThermalLimiter<2, double> lim{ThermalLimits<2, double>{derate_window(100.0, 120.0), 125.0}};

        CHECK(lim.evaluate(90.0).scale == doctest::Approx(1.0));
        CHECK(lim.evaluate(110.0).scale == doctest::Approx(0.5));
        CHECK(lim.evaluate(120.0).scale == doctest::Approx(0.0));

        CHECK(lim.evaluate(120.0).ok);       // below fault_temp
        CHECK(lim.evaluate(125.0).ok);       // exactly fault_temp
        CHECK_FALSE(lim.evaluate(126.0).ok); // past fault_temp
    }

    TEST_CASE("default-constructed limiter never derates or faults") {
        ThermalLimiter<2, double> lim{};
        CHECK(lim.evaluate(1000.0).scale == doctest::Approx(1.0));
        CHECK(lim.evaluate(1000.0).ok);
    }
}
