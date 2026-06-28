
#include <cmath>
#include <cstdint>

#include "doctest.h"
#include "wet/motor/calibration.hpp"

using namespace wet;

namespace {

// Exact ZOH discrete d-axis R–L plant: i[k] = a·i[k-1] + b·v[k-1].
struct RLPlant {
    double a;
    double b;
    double i{0.0};

    RLPlant(double R, double L, double Ts) : a(std::exp(-R * Ts / L)), b((1.0 - a) / R) {}

    // Apply v over the next interval; returns the resulting current.
    double apply(double v) {
        i = (a * i) + (b * v);
        return i;
    }
};

// Drive the calibrator to completion against a plant, optional measurement noise.
template<typename Plant>
void run(PhaseParameterCalibrator<double>& cal, Plant& plant, double Ts, double noise_amp = 0.0) {
    std::uint32_t rng = 2463534242u;
    auto          noise = [&]() {
        rng = (rng * 1664525u) + 1013904223u;
        return ((static_cast<double>(rng >> 8) / static_cast<double>(1u << 24)) - 0.5) * 2.0 * noise_amp;
    };

    double measured = 0.0;
    for (int k = 0; k < 200000; ++k) {
        const auto cmd = cal.step(measured + noise(), Ts);
        measured = plant.apply(cmd.v_d);
        if (cmd.done) {
            break;
        }
    }
}

} // namespace

TEST_SUITE("Phase R/L Calibration") {

    TEST_CASE("recovers R and L from a clean synthetic plant") {
        constexpr double R_true = 0.45;      // ohm
        constexpr double L_true = 220e-6;    // H
        constexpr double Ts = 1.0 / 20000.0; // 20 kHz

        RLPlant                          plant{R_true, L_true, Ts};
        PhaseParameterCalibrator<double> cal{PhaseCalibrationConfig<double>{
            .inject_voltage = 2.0,
            .duration_s = 0.2,
            .prbs_clock_s = 5e-4
        }};

        CHECK_FALSE(cal.valid()); // nothing fitted yet
        run(cal, plant, Ts);

        REQUIRE(cal.valid());
        CHECK(cal.resistance() == doctest::Approx(R_true).epsilon(1e-3));
        CHECK(cal.inductance() == doctest::Approx(L_true).epsilon(1e-3));
    }

    TEST_CASE("stays within tolerance under measurement noise") {
        constexpr double R_true = 0.9;
        constexpr double L_true = 600e-6;
        constexpr double Ts = 1.0 / 16000.0;

        RLPlant                          plant{R_true, L_true, Ts};
        PhaseParameterCalibrator<double> cal{PhaseCalibrationConfig<double>{
            .inject_voltage = 3.0,
            .duration_s = 0.5,
            .prbs_clock_s = 6e-4
        }};

        run(cal, plant, Ts, /*noise_amp=*/0.02); // 20 mA RMS-ish current noise

        REQUIRE(cal.valid());
        CHECK(cal.resistance() == doctest::Approx(R_true).epsilon(0.05)); // within 5%
        CHECK(cal.inductance() == doctest::Approx(L_true).epsilon(0.05));
    }

    TEST_CASE("a different operating point recovers different parameters") {
        constexpr double R_true = 0.12;
        constexpr double L_true = 35e-6; // low-inductance, high-current motor
        constexpr double Ts = 1.0 / 40000.0;

        RLPlant                          plant{R_true, L_true, Ts};
        PhaseParameterCalibrator<double> cal{PhaseCalibrationConfig<double>{
            .inject_voltage = 1.0,
            .duration_s = 0.1,
            .prbs_clock_s = 2e-4
        }};
        run(cal, plant, Ts);

        REQUIRE(cal.valid());
        CHECK(cal.resistance() == doctest::Approx(R_true).epsilon(2e-3));
        CHECK(cal.inductance() == doctest::Approx(L_true).epsilon(2e-3));
    }

    TEST_CASE("reset clears the fit") {
        RLPlant                          plant{0.5, 200e-6, 1.0 / 20000.0};
        PhaseParameterCalibrator<double> cal{PhaseCalibrationConfig<double>{.duration_s = 0.05}};
        run(cal, plant, 1.0 / 20000.0);
        REQUIRE(cal.valid());
        cal.reset();
        CHECK_FALSE(cal.valid());
    }

    TEST_CASE("config validation") {
        CHECK(PhaseCalibrationConfig<double>{}.valid());
        CHECK_FALSE(PhaseCalibrationConfig<double>{.inject_voltage = 0.0}.valid());
        CHECK_FALSE(PhaseCalibrationConfig<double>{.duration_s = -1.0}.valid());
        CHECK_FALSE(PhaseCalibrationConfig<double>{.lambda = 1.5}.valid());
    }
}
