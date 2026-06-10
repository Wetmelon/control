#include <cmath>

#include "wet/controllers/repetitive.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wet;

namespace {
constexpr double pi = 3.14159265358979323846;
} // namespace

TEST_SUITE("repetitive") {
    TEST_CASE("synthesize_repetitive computes the period and validates the spec") {
        SUBCASE("valid: fs=1000, f0=50 -> N=20") {
            const auto r = design::synthesize_repetitive(1000.0, 50.0, 1.0, 0.98, 1);
            CHECK(r.success);
            CHECK(r.config.period == 20);
            CHECK(r.config.gain == doctest::Approx(1.0));
            CHECK(r.config.q_filter == doctest::Approx(0.98));
            CHECK(r.config.lead == 1);
        }
        SUBCASE("rounds to nearest period") {
            CHECK(design::synthesize_repetitive(1000.0, 49.0).config.period == 20); // 1000/49 = 20.4
            CHECK(design::synthesize_repetitive(1000.0, 47.0).config.period == 21); // 1000/47 = 21.3
        }
        SUBCASE("rejects bad specs") {
            CHECK_FALSE(design::synthesize_repetitive(1000.0, 1000.0).success);         // f0 >= fs
            CHECK_FALSE(design::synthesize_repetitive(1000.0, -50.0).success);          // f0 < 0
            CHECK_FALSE(design::synthesize_repetitive(1000.0, 50.0, 3.0).success);      // gain > 2
            CHECK_FALSE(design::synthesize_repetitive(1000.0, 50.0, 1.0, 1.5).success); // q > 1
        }
    }

    TEST_CASE("drives a periodic tracking error to zero (internal-model principle)") {
        // Plant y[k] = u[k-1] (unit transport delay); repetitive controller alone.
        // With a perfect 1-sample model (lead m = 1, Q = 1, k_rc = 1) the periodic
        // tracking error must vanish after one period.
        constexpr size_t N = 20;
        const auto       design = design::synthesize_repetitive(1000.0, 50.0, 1.0, 1.0, 1);
        REQUIRE(design.success);
        REQUIRE(design.config.period == N);

        RepetitiveController<64, double> rc(design);

        auto reference = [](size_t k) { return std::sin(2.0 * pi * static_cast<double>(k) / N); };

        double       u_prev = 0.0;
        double       worst_first_period = 0.0;
        double       worst_last_period = 0.0;
        const size_t periods = 8;
        for (size_t k = 0; k < periods * N; ++k) {
            const double y = u_prev;           // y[k] = u[k-1]
            const double e = reference(k) - y; // tracking error
            const double u = rc.step(e);
            u_prev = u;

            if (k < N) {
                worst_first_period = std::max(worst_first_period, std::abs(e));
            }
            if (k >= (periods - 1) * N) {
                worst_last_period = std::max(worst_last_period, std::abs(e));
            }
        }
        // The first period still has error (model not yet learned); the last is ~0.
        CHECK(worst_first_period > 0.1);
        CHECK(worst_last_period < 1e-9);
    }

    TEST_CASE("rejects a periodic disturbance") {
        // Plant y[k] = u[k-1] + d[k], periodic disturbance d, reference r = 0.
        constexpr size_t N = 24;
        const auto       design = design::synthesize_repetitive(1200.0, 50.0, 1.0, 1.0, 1);
        REQUIRE(design.success);
        REQUIRE(design.config.period == N);

        RepetitiveController<64, double> rc(design);
        auto                             disturbance = [](size_t k) {
            return 0.5 * std::sin(2.0 * pi * static_cast<double>(k) / N)
                 + 0.2 * std::sin(2.0 * pi * 3.0 * static_cast<double>(k) / N); // fundamental + 3rd
        };

        double       u_prev = 0.0;
        double       worst_last_period = 0.0;
        const size_t periods = 10;
        for (size_t k = 0; k < periods * N; ++k) {
            const double y = u_prev + disturbance(k);
            const double e = 0.0 - y;
            const double u = rc.step(e);
            u_prev = u;
            if (k >= (periods - 1) * N) {
                worst_last_period = std::max(worst_last_period, std::abs(y));
            }
        }
        CHECK(worst_last_period < 1e-9); // both the fundamental and the 3rd are rejected
    }

    TEST_CASE("Q < 1 stays bounded (robustness roll-off) and reset clears state") {
        const auto                       design = design::synthesize_repetitive(1000.0, 50.0, 1.0, 0.9, 1);
        RepetitiveController<64, double> rc(design);
        // Feed a unit-impulse error; with Q<1 the internal model decays, output bounded.
        double maxout = std::abs(rc.step(1.0));
        for (int k = 0; k < 2000; ++k) {
            maxout = std::max(maxout, std::abs(rc.step(0.0)));
        }
        CHECK(maxout < 2.0); // bounded (no marginal-stability blow-up)
        CHECK(maxout > 0.0); // and it actually produced corrections

        rc.reset();
        // After reset, a zero error gives zero output (empty internal model).
        CHECK(rc.step(0.0) == doctest::Approx(0.0));
    }

    TEST_CASE("invalid design yields a pass-through (zero correction)") {
        RepetitiveController<64, double> rc(design::synthesize_repetitive(1000.0, 1000.0)); // invalid
        CHECK_FALSE(rc.valid());
        CHECK(rc.step(1.0) == doctest::Approx(0.0));
    }

    TEST_CASE("repetitive controller is constexpr-evaluable") {
        constexpr bool ok = []() consteval {
            auto                             d = design::synthesize_repetitive(1000.0, 100.0, 1.0, 1.0, 0);
            RepetitiveController<32, double> rc(d);
            // Period N=10. Feed a constant error of 1: the first period emits no
            // correction (empty internal model), then the model from one period
            // ago surfaces, so the correction at step N (the 11th) equals 1.
            double last = 0.0;
            for (size_t k = 0; k < 11; ++k) {
                last = rc.step(1.0);
            }
            return d.success && d.config.period == 10 && wet::abs(last - 1.0) < 1e-9;
        }();
        static_assert(ok, "repetitive controller must work at compile time");
        CHECK(ok);
    }
}
