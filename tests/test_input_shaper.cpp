#include <cmath>

#include "wet/trajectory/input_shaper.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wet;

namespace {
// Simulate a lightly-damped 2nd-order mode driven by a command stream, and
// return the worst residual |position - command_final| after the move settles.
// x'' + 2 zeta wn x' + wn^2 x = wn^2 u  (unity DC gain), explicit Euler.
template<typename Shaper>
double residual(Shaper& shaper, double fn, double zeta, double dt, bool shape) {
    const double wn = 2.0 * 3.14159265358979323846 * fn;
    double       x = 0.0, v = 0.0;
    double       worst = 0.0;
    const int    steps = 8000;
    for (int k = 0; k < steps; ++k) {
        const double cmd = (k > 0) ? 1.0 : 0.0;          // unit step at k=1
        const double u = shape ? shaper.step(cmd) : cmd; // shaped or raw
        const double a = wn * wn * (u - x) - 2.0 * zeta * wn * v;
        v += dt * a;
        x += dt * v;
        if (k > steps / 2) { // residual after the move has had time to settle
            worst = std::max(worst, std::abs(x - 1.0));
        }
    }
    return worst;
}
} // namespace

TEST_SUITE("input_shaper") {
    TEST_CASE("ZV shaper: amplitudes [1, K]/(1+K), normalized, correct delay") {
        const double fn = 2.0, zeta = 0.1, Ts = 1.0e-3;
        const auto   r = design::synthesize_input_shaper(fn, zeta, Ts, ShaperType::ZV);
        REQUIRE(r.success);
        REQUIRE(r.count == 2);

        const double K = std::exp(-zeta * 3.14159265358979323846 / std::sqrt(1.0 - zeta * zeta));
        CHECK(r.amplitudes[0] == doctest::Approx(1.0 / (1.0 + K)));
        CHECK(r.amplitudes[1] == doctest::Approx(K / (1.0 + K)));
        CHECK(r.amplitudes[0] + r.amplitudes[1] == doctest::Approx(1.0)); // unity DC gain

        // 2nd impulse at Td/2 = pi/wd.
        const double wd = 2.0 * 3.14159265358979323846 * fn * std::sqrt(1.0 - zeta * zeta);
        CHECK(r.times[1] == doctest::Approx(3.14159265358979323846 / wd));
        CHECK(r.delays[1] == static_cast<size_t>(r.times[1] / Ts + 0.5));
    }

    TEST_CASE("ZVD: binomial [1,2K,K^2] normalized; ZVDD: [1,3K,3K^2,K^3]") {
        const double fn = 5.0, zeta = 0.05, Ts = 1.0e-3;
        const double K = std::exp(-zeta * 3.14159265358979323846 / std::sqrt(1.0 - zeta * zeta));

        const auto zvd = design::synthesize_input_shaper(fn, zeta, Ts, ShaperType::ZVD);
        REQUIRE(zvd.count == 3);
        const double s3 = 1.0 + 2.0 * K + K * K;
        CHECK(zvd.amplitudes[0] == doctest::Approx(1.0 / s3));
        CHECK(zvd.amplitudes[1] == doctest::Approx(2.0 * K / s3));
        CHECK(zvd.amplitudes[2] == doctest::Approx(K * K / s3));

        const auto zvdd = design::synthesize_input_shaper(fn, zeta, Ts, ShaperType::ZVDD);
        REQUIRE(zvdd.count == 4);
        double sum = 0.0;
        for (size_t i = 0; i < 4; ++i) {
            sum += zvdd.amplitudes[i];
        }
        CHECK(sum == doctest::Approx(1.0));
    }

    TEST_CASE("rejects bad specs") {
        CHECK_FALSE(design::synthesize_input_shaper(0.0, 0.1, 1e-3, ShaperType::ZV).success);      // fn <= 0
        CHECK_FALSE(design::synthesize_input_shaper(2.0, 1.0, 1e-3, ShaperType::ZV).success);      // zeta >= 1
        CHECK_FALSE(design::synthesize_input_shaper(2.0, 0.1, 0.0, ShaperType::ZV).success);       // Ts <= 0
        CHECK_FALSE(design::synthesize_input_shaper(2.0, 0.1, 1e-3, ShaperType::EI, 0.0).success); // V <= 0
    }

    TEST_CASE("shaping strongly attenuates residual vibration") {
        const double fn = 2.0, zeta = 0.02, dt = 1.0e-3;
        // Buffer must hold the longest delay: ZVD -> Td = 1/(fn*sqrt(1-z^2)) ~ 0.5 s -> 500 samples.
        const auto art = design::synthesize_input_shaper(fn, zeta, dt, ShaperType::ZVD);
        REQUIRE(art.success);
        InputShaper<800, double> shaper(art);
        REQUIRE(shaper.valid());

        InputShaper<800, double> dummy(art);
        const double             raw = residual(dummy, fn, zeta, dt, /*shape=*/false);
        const double             shaped = residual(shaper, fn, zeta, dt, /*shape=*/true);

        CHECK(raw > 0.1);           // unshaped step rings hard
        CHECK(shaped < 0.02 * raw); // shaped kills the residual (>50x)
    }

    TEST_CASE("ZVD is more robust to a detuned mode than ZV") {
        const double fn = 2.0, zeta = 0.02, dt = 1.0e-3;
        const double fn_actual = fn * 1.15; // 15% frequency error

        InputShaper<800, double> zv(design::synthesize_input_shaper(fn, zeta, dt, ShaperType::ZV));
        InputShaper<800, double> zvd(design::synthesize_input_shaper(fn, zeta, dt, ShaperType::ZVD));

        const double zv_res = residual(zv, fn_actual, zeta, dt, true);
        const double zvd_res = residual(zvd, fn_actual, zeta, dt, true);
        CHECK(zvd_res < zv_res); // ZVD's extra impulse buys robustness to detuning
    }

    TEST_CASE("unity DC gain: a constant command passes through unchanged") {
        const auto               art = design::synthesize_input_shaper(3.0, 0.1, 1e-3, ShaperType::ZVD);
        InputShaper<800, double> shaper(art);
        double                   y = 0.0;
        for (int k = 0; k < 2000; ++k) {
            y = shaper.step(2.5);
        }
        CHECK(y == doctest::Approx(2.5)); // steady command unaffected
    }

    TEST_CASE("undersized buffer -> invalid -> pass-through") {
        const auto              art = design::synthesize_input_shaper(0.5, 0.0, 1e-3, ShaperType::ZVDD); // long delays
        InputShaper<10, double> tiny(art);                                                               // 10 samples << needed
        CHECK_FALSE(tiny.valid());
        CHECK(tiny.step(7.0) == doctest::Approx(7.0));
    }

    TEST_CASE("multi-axis bank shapes each axis with its own mode") {
        InputShaperBank<2, 800, double> bank;
        bank.set_axis(0, design::synthesize_input_shaper(2.0, 0.05, 1e-3, ShaperType::ZVD));
        bank.set_axis(1, design::synthesize_input_shaper(4.0, 0.05, 1e-3, ShaperType::ZV));
        REQUIRE(bank.axis(0).valid());
        REQUIRE(bank.axis(1).valid());
        // Steady command passes through on both axes (unity DC gain).
        wet::array<double, 2> y{};
        for (int k = 0; k < 2000; ++k) {
            y = bank.step(wet::array<double, 2>{1.0, -2.0});
        }
        CHECK(y[0] == doctest::Approx(1.0));
        CHECK(y[1] == doctest::Approx(-2.0));
    }

    TEST_CASE("float deployment via as<float>()") {
        const auto              art = design::synthesize_input_shaper(2.0, 0.05, 1e-3, ShaperType::ZVD);
        InputShaper<800, float> shaper(art.as<float>());
        CHECK(shaper.valid());
        const float u = shaper.step(1.0f);
        CHECK(std::isfinite(u));
        CHECK(u < 1.0f); // first output is only the first impulse's share
    }

    TEST_CASE("input shaper is constexpr-evaluable") {
        constexpr bool ok = []() consteval {
            auto                     art = design::synthesize_input_shaper(2.0, 0.05, 1e-3, ShaperType::ZVD);
            InputShaper<800, double> shaper(art);
            double                   last = 0.0;
            for (int k = 0; k < 1000; ++k) {
                last = shaper.step(1.0);
            }
            return art.success && shaper.valid() && wet::abs(last - 1.0) < 1e-9;
        }();
        static_assert(ok, "input shaper must work at compile time");
        CHECK(ok);
    }
}
