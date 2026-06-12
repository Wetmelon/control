#include <cmath>

#include "wet/filters/sogi.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wet;

namespace {
constexpr double pi = 3.14159265358979323846;
} // namespace

TEST_SUITE("sogi_fll") {
    TEST_CASE("locks onto a tone offset from the initial guess") {
        const double    fs = 2000.0, dt = 1.0 / fs;
        const double    f_true = 57.0;
        SogiFll<double> fll(50.0, dt); // start 7 Hz low

        for (int k = 0; k < 6000; ++k) { // 3 s
            fll.update(std::sin(2.0 * pi * f_true * (k * dt)));
        }
        CHECK(fll.frequency_hz() == doctest::Approx(f_true).epsilon(0.01)); // locked within 1%
    }

    TEST_CASE("recovers the tone amplitude after lock") {
        const double fs = 2000.0, dt = 1.0 / fs;
        const double f_true = 45.0, A = 2.5;

        SogiFll<double> fll(48.0, dt);
        for (int k = 0; k < 8000; ++k) {
            fll.update(A * std::sin(2.0 * pi * f_true * (k * dt)));
        }
        CHECK(fll.frequency_hz() == doctest::Approx(f_true).epsilon(0.02));
        CHECK(fll.amplitude() == doctest::Approx(A).epsilon(0.05));
    }

    TEST_CASE("tracks a frequency step") {
        const double    fs = 2000.0, dt = 1.0 / fs;
        SogiFll<double> fll(50.0, dt);

        // Phase-continuous tone that steps 50 -> 62 Hz halfway.
        double phase = 0.0;
        double f_final = 0.0;
        for (int k = 0; k < 12000; ++k) {
            const double f = (k < 6000) ? 50.0 : 62.0;
            phase += 2.0 * pi * f * dt;
            fll.update(std::sin(phase));
            f_final = f;
        }
        CHECK(f_final == 62.0);
        CHECK(fll.frequency_hz() == doctest::Approx(62.0).epsilon(0.02)); // re-locked
    }

    TEST_CASE("locks through broadband noise (it is a narrow band-pass)") {
        const double    fs = 2000.0, dt = 1.0 / fs;
        const double    f_true = 53.0;
        SogiFll<double> fll(50.0, dt);
        auto            noise = [](double t) {
            return 0.3 * (std::sin(617.0 * t) + std::sin(1303.0 * t) + std::sin(2999.0 * t)) / 3.0;
        };
        for (int k = 0; k < 8000; ++k) {
            const double t = k * dt;
            fll.update(std::sin(2.0 * pi * f_true * t) + noise(t));
        }
        CHECK(fll.frequency_hz() == doctest::Approx(f_true).epsilon(0.02));
    }

    TEST_CASE("clamps the tracked frequency to the configured band") {
        const double fs = 2000.0, dt = 1.0 / fs;
        // Drive at 90 Hz but clamp the search to [40, 70] Hz.
        SogiFll<double> fll(55.0, dt, wet::numbers::sqrt2_v<double>, 2.0, 40.0, 70.0);
        for (int k = 0; k < 6000; ++k) {
            fll.update(std::sin(2.0 * pi * 90.0 * (k * dt)));
        }
        CHECK(fll.frequency_hz() <= 70.0 + 1e-6);
        CHECK(fll.frequency_hz() >= 40.0 - 1e-6);
    }

    TEST_CASE("invalid config is inert; reset re-seeds the guess") {
        SogiFll<double> bad(50.0, -1.0); // Ts <= 0
        CHECK_FALSE(bad.valid());
        bad.update(1.0);
        CHECK(bad.frequency_hz() == doctest::Approx(50.0)); // unchanged

        SogiFll<double> fll(50.0, 1.0 / 2000.0);
        for (int k = 0; k < 3000; ++k) {
            fll.update(std::sin(2.0 * pi * 58.0 * (k / 2000.0)));
        }
        fll.reset(50.0);
        CHECK(fll.frequency_hz() == doctest::Approx(50.0));
        CHECK(fll.amplitude() == doctest::Approx(0.0));
    }

    TEST_CASE("float specialization locks") {
        SogiFll<float> fll(50.0f, 1.0f / 2000.0f);
        for (int k = 0; k < 8000; ++k) {
            fll.update(std::sin(2.0f * 3.14159265f * 56.0f * (k / 2000.0f)));
        }
        CHECK(fll.frequency_hz() == doctest::Approx(56.0f).epsilon(0.03));
    }

    TEST_CASE("SOGI-FLL is constexpr-evaluable") {
        constexpr double f = []() consteval {
            SogiFll<double> fll(50.0, 1.0 / 2000.0);
            for (int k = 0; k < 6000; ++k) {
                fll.update(wet::sin(2.0 * 3.14159265358979323846 * 55.0 * (k / 2000.0)));
            }
            return fll.frequency_hz();
        }();
        static_assert(f > 54.0 && f < 56.0, "SOGI-FLL must lock at compile time");
        CHECK(f == doctest::Approx(55.0).epsilon(0.02));
    }
}
