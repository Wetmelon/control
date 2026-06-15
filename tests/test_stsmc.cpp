#include <cmath>

#include "wet/controllers/stsmc.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wet;

namespace {

// Closed-loop sim of the super-twisting algorithm on the canonical relative-
// degree-1 plant  ṡ = u + d(t),  integrated with explicit Euler at step Ts.
// Returns {worst |s| over the final `tail` fraction, worst |Δu| between
// consecutive samples over that tail}. The second number is the chattering
// metric: a *continuous* controller keeps it small; a sign() controller jumps
// by ~2k every crossing.
template<typename Ctrl, typename Dist>
wet::pair<double, double> run(Ctrl& c, Dist d, double Ts, int steps, double s0, double tail = 0.25) {
    double    s = s0;
    double    u_prev = 0.0;
    double    worst_s = 0.0;
    double    worst_du = 0.0;
    const int tail_start = static_cast<int>(steps * (1.0 - tail));
    for (int k = 0; k < steps; ++k) {
        const double u = c.control(s);
        if (k >= tail_start) {
            worst_s = std::max(worst_s, std::abs(s));
            worst_du = std::max(worst_du, std::abs(u - u_prev));
        }
        u_prev = u;
        s += Ts * (u + d(k * Ts)); // plant: ṡ = u + d
    }
    return {worst_s, worst_du};
}

} // namespace

TEST_SUITE("stsmc") {
    TEST_CASE("synthesize_stsmc computes Levant/Moreno gains and validates") {
        SUBCASE("valid: k1 = 1.5*sqrt(L), k2 = 1.1*L") {
            const auto r = design::synthesize_stsmc(4.0, 1e-3);
            CHECK(r.success);
            CHECK(r.k1 == doctest::Approx(1.5 * std::sqrt(4.0)));
            CHECK(r.k2 == doctest::Approx(1.1 * 4.0));
            CHECK(r.Ts == doctest::Approx(1e-3));
        }
        SUBCASE("gain_margin scales both gains") {
            const auto r = design::synthesize_stsmc(4.0, 1e-3, 0.0, 0.0, 0.0, 2.0);
            CHECK(r.k1 == doctest::Approx(2.0 * 1.5 * std::sqrt(4.0)));
            CHECK(r.k2 == doctest::Approx(2.0 * 1.1 * 4.0));
        }
        SUBCASE("rejects bad specs") {
            CHECK_FALSE(design::synthesize_stsmc(0.0, 1e-3).success);                     // L <= 0
            CHECK_FALSE(design::synthesize_stsmc(4.0, 0.0).success);                      // Ts <= 0
            CHECK_FALSE(design::synthesize_stsmc(4.0, 1e-3, 0.0, -1.0).success);          // kl < 0
            CHECK_FALSE(design::synthesize_stsmc(4.0, 1e-3, 0.0, 0.0, 0.0, 0.5).success); // margin < 1
        }
        SUBCASE("direct-gain factory validates") {
            CHECK(design::stsmc(2.0, 2.2, 1e-3).success);
            CHECK_FALSE(design::stsmc(-1.0, 2.2, 1e-3).success);
            CHECK_FALSE(design::stsmc(2.0, 0.0, 1e-3).success);
        }
    }

    TEST_CASE("classic super-twisting rejects a Lipschitz disturbance, continuously") {
        // d(t) = sin(2t) -> |d_dot| <= 2 = L. 1 kHz loop.
        const double Ts = 1e-3;
        const double L = 2.0;
        const auto   art = design::synthesize_stsmc(L, Ts);
        REQUIRE(art.success);
        SuperTwistingController<double> c(art);

        auto d = [](double t) { return std::sin(2.0 * t); };
        const auto [worst_s, worst_du] = run(c, d, Ts, 8000, 1.0); // 8 s

        CHECK(worst_s < 1e-2);  // s driven into a tight band despite the disturbance
        CHECK(worst_du < 0.05); // control is continuous (no sign()-style jumps)
        // The integral state tracks -d: at the end of an 8 s run it is near -sin(2*8).
        CHECK(c.integral_state() == doctest::Approx(-std::sin(2.0 * 8.0)).epsilon(0.1));
    }

    TEST_CASE("first-order SMC chatters where super-twisting does not (the whole point)") {
        // Same plant/disturbance; compare the steady-state control jumpiness.
        const double Ts = 1e-3;
        const double L = 2.0;
        auto         d = [](double t) { return std::sin(2.0 * t); };

        SuperTwistingController<double> st(design::synthesize_stsmc(L, Ts));
        const auto [st_s, st_du] = run(st, d, Ts, 8000, 1.0);

        // A bare sign()-based controller of comparable authority: u = -k*sign(s).
        struct SignSMC {
            double k;
            double control(double s) { return -k * static_cast<double>(wet::sgn(s)); }
        } sign_smc{3.0};
        const auto [sg_s, sg_du] = run(sign_smc, d, Ts, 8000, 1.0);

        // Super-twisting's steady-state control steps are far smaller than the
        // sign controller's ~2k bang-bang jumps.
        CHECK(st_du < 0.1);
        CHECK(sg_du > 1.0);
        CHECK(st_du < sg_du);
    }

    TEST_CASE("generalized STA (kl > 0) also converges") {
        const double Ts = 1e-3;
        const double L = 2.0;
        const auto   art = design::synthesize_stsmc(L, Ts, 0.0, /*kl=*/1.5);
        REQUIRE(art.success);
        REQUIRE(art.k_lin == doctest::Approx(1.5));
        SuperTwistingController<double> c(art);

        auto d = [](double t) { return 0.5 * std::sin(3.0 * t); };
        const auto [worst_s, worst_du] = run(c, d, Ts, 8000, 1.0);
        CHECK(worst_s < 1e-2);
        CHECK(worst_du < 0.1);
    }

    TEST_CASE("boundary layer keeps the canonical loop bounded and continuous") {
        const double Ts = 1e-3;
        const auto   art = design::synthesize_stsmc(2.0, Ts, 0.0, 0.0, /*epsilon=*/0.05);
        REQUIRE(art.success);
        SuperTwistingController<double> c(art);

        auto d = [](double t) { return std::sin(2.0 * t); };
        const auto [worst_s, worst_du] = run(c, d, Ts, 8000, 1.0);
        CHECK(worst_s < 0.25);  // softened sign -> wider band than true sign (which hit <1e-2)
        CHECK(worst_du < 0.05); // still continuous
    }

    TEST_CASE("(r,y) overload builds s = lambda*e + e_dot then applies the STA") {
        // The (r, y) form is just a surface-builder over control(s): the first
        // call sees e = r - y, e_prev = 0, so e_dot = e/Ts. Verify it equals
        // calling control(s) with that same s on an identical controller.
        const double                    Ts = 1e-3;
        const double                    lambda = 5.0;
        SuperTwistingController<double> via_ry(design::synthesize_stsmc(2.0, Ts, lambda));
        SuperTwistingController<double> via_s(design::synthesize_stsmc(2.0, Ts, lambda));

        const double r = 1.0;
        const double y = 0.0;
        const double e = r - y;
        const double s = (lambda * e) + (e / Ts); // e_dot = (e - 0)/Ts
        CHECK(via_ry.control(r, y) == doctest::Approx(via_s.control(s)));
    }

    TEST_CASE("invalid design is inert; reset clears state") {
        SuperTwistingController<double> bad(design::synthesize_stsmc(-1.0, 1e-3));
        CHECK_FALSE(bad.valid());
        CHECK(bad.control(5.0) == doctest::Approx(0.0));

        SuperTwistingController<double> c(design::synthesize_stsmc(2.0, 1e-3));
        (void)c.control(1.0);
        (void)c.control(1.0);
        CHECK(c.integral_state() != 0.0);
        c.reset();
        CHECK(c.integral_state() == doctest::Approx(0.0));
    }

    TEST_CASE("float deployment via as<float>()") {
        const auto                     art = design::synthesize_stsmc(2.0, 1e-3);
        SuperTwistingController<float> c(art.as<float>());
        const float                    u = c.control(1.0f);
        CHECK(std::isfinite(u));
        CHECK(u < 0.0f); // s > 0 -> u pushes negative
    }

    TEST_CASE("super-twisting controller is constexpr-evaluable") {
        constexpr double final_s = []() consteval {
            auto                            art = design::synthesize_stsmc(2.0, 1e-3);
            SuperTwistingController<double> c(art);
            double                          s = 1.0;
            for (int k = 0; k < 4000; ++k) {
                const double u = c.control(s);
                s += 1e-3 * (u + 0.5 * wet::sin(2.0 * k * 1e-3));
            }
            return s;
        }();
        static_assert(final_s < 0.05 && final_s > -0.05, "STA must converge at compile time");
        CHECK(final_s == doctest::Approx(0.0).epsilon(0.05));
    }
}
