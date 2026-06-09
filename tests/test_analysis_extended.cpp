
#include <algorithm>
#include <cmath>

#include "wet/analysis/analysis.hpp"
#include "wet/analysis/stability.hpp"
#include "wet/matrix/matrix.hpp"
#include "wet/systems/state_space.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wet;

/**
 * @brief Tests for system analysis functions (analysis.hpp)
 *
 * Covers controllability, observability, Bode response, DC gain,
 * gain/phase margins, bandwidth, pole analysis, and damping.
 */

TEST_SUITE("Analysis - Controllability and Observability") {
    TEST_CASE("Controllable 2x2 system") {
        // A = [0 1; -2 -3], B = [0; 1]
        // Controllability matrix [B, AB] = [0 1; 1 -3] — rank 2
        Matrix<2, 2> A{{0.0, 1.0}, {-2.0, -3.0}};
        Matrix<2, 1> B{{0.0}, {1.0}};

        auto Co = stability::controllability_matrix(A, B);
        CHECK(stability::rank(Co) == 2);
        CHECK(stability::is_controllable(A, B));

        // Compile-time check
        static_assert(stability::is_controllable(
            Matrix<2, 2>{{0.0, 1.0}, {-2.0, -3.0}},
            Matrix<2, 1>{{0.0}, {1.0}}
        ));
    }

    TEST_CASE("Uncontrollable 2x2 system") {
        // A = [1 0; 0 2], B = [0; 0] — zero input, uncontrollable
        Matrix<2, 2> A{{1.0, 0.0}, {0.0, 2.0}};
        Matrix<2, 1> B{{0.0}, {0.0}};

        CHECK_FALSE(stability::is_controllable(A, B));
    }

    TEST_CASE("Observable 2x2 system") {
        Matrix<2, 2> A{{0.0, 1.0}, {-2.0, -3.0}};
        Matrix<1, 2> C{{1.0, 0.0}};

        auto Ob = stability::observability_matrix(A, C);
        CHECK(stability::rank(Ob) == 2);
        CHECK(stability::is_observable(A, C));
    }

    TEST_CASE("Unobservable 2x2 system") {
        // A diagonal, C only sees first state, second state unobservable
        Matrix<2, 2> A{{1.0, 0.0}, {0.0, 2.0}};
        Matrix<1, 2> C{{1.0, 0.0}};

        auto Ob = stability::observability_matrix(A, C);
        // Ob = [1 0; 1 0] — rank 1, not rank 2
        CHECK(stability::rank(Ob) == 1);
        CHECK_FALSE(stability::is_observable(A, C));
    }

    TEST_CASE("Controllable 3x3 system") {
        // Companion form: always controllable
        Matrix<3, 3> A{{0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {-6.0, -11.0, -6.0}};
        Matrix<3, 1> B{{0.0}, {0.0}, {1.0}};

        CHECK(stability::is_controllable(A, B));
    }

    TEST_CASE("MIMO controllability") {
        Matrix<2, 2> A{{0.0, 1.0}, {-2.0, -3.0}};
        Matrix<2, 2> B{{1.0, 0.0}, {0.0, 1.0}};

        auto Co = stability::controllability_matrix(A, B);
        // Controllability matrix is 2x4, should have rank 2
        CHECK(stability::rank(Co) == 2);
        CHECK(stability::is_controllable(A, B));
    }

    TEST_CASE("Matrix rank computation") {
        // Full rank 3x3
        Matrix<3, 3> M1{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
        CHECK(stability::rank(M1) == 3);

        // Rank 2 (third row = first + second)
        Matrix<3, 3> M2{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {1.0, 1.0, 0.0}};
        CHECK(stability::rank(M2) == 2);

        // Rank 1
        Matrix<3, 3> M3{{1.0, 2.0, 3.0}, {2.0, 4.0, 6.0}, {3.0, 6.0, 9.0}};
        CHECK(stability::rank(M3) == 1);
    }
}

TEST_SUITE("Analysis - Bode Response") {
    TEST_CASE("First-order lowpass Bode data") {
        // G(s) = 1/(s+1), DC gain = 0 dB, -3dB at ω=1 rad/s
        StateSpace<1, 1, 1, 0, 0> sys{
            .A = Matrix<1, 1>{{-1.0}},
            .B = Matrix<1, 1>{{1.0}},
            .C = Matrix<1, 1>{{1.0}},
            .D = Matrix<1, 1>{{0.0}}
        };

        auto omega = analysis::logspace(0.01, 100.0, 200);
        auto result = analysis::bode(sys, omega);

        // DC gain should be 0 dB
        CHECK(result.points[0].magnitude_db == doctest::Approx(0.0).epsilon(0.1));

        // At ω=1, magnitude should be -3 dB
        // Find the point closest to ω=1
        auto bw = result.bandwidth();
        REQUIRE(bw.has_value());
        CHECK(bw.value() == doctest::Approx(1.0).epsilon(0.05));

        // Phase at DC should be ~0°
        CHECK(result.points[0].phase_deg == doctest::Approx(0.0).epsilon(1.0));
    }

    TEST_CASE("Second-order system Bode response") {
        // G(s) = wn²/(s² + 2*zeta*wn*s + wn²)
        // wn = 10 rad/s, zeta = 0.5
        double                    wn = 10.0;
        double                    zeta = 0.5;
        StateSpace<2, 1, 1, 0, 0> sys{
            .A = Matrix<2, 2>{{0.0, 1.0}, {-wn * wn, -2 * zeta * wn}},
            .B = Matrix<2, 1>{{0.0}, {wn * wn}},
            .C = Matrix<1, 2>{{1.0, 0.0}},
            .D = Matrix<1, 1>{{0.0}}
        };

        auto omega = analysis::logspace(0.1, 1000.0, 500);
        auto result = analysis::bode(sys, omega);

        // DC gain should be 0 dB (unity)
        CHECK(result.points[0].magnitude_db == doctest::Approx(0.0).epsilon(0.1));

        // There should be a resonant peak around wn (zeta < 0.707)
        double peak_db = -300;
        for (const auto& pt : result.points) {
            peak_db = wet::max(pt.magnitude_db, peak_db);
        }
        CHECK(peak_db > 0.0); // Should have resonant peak above 0 dB
    }

    TEST_CASE("Bode from transfer function coefficients") {
        // G(s) = 10/(s+10), ascending powers: num = {10}, den = {10, 1}
        std::array<double, 1> num = {10.0};
        std::array<double, 2> den = {10.0, 1.0};

        auto omega = analysis::logspace(0.1, 1000.0, 200);
        auto result = analysis::bode(num, den, omega);

        // DC gain = 10/10 = 1 = 0 dB
        CHECK(result.points[0].magnitude_db == doctest::Approx(0.0).epsilon(0.1));
    }
}

TEST_SUITE("Analysis - Stability Margins") {
    TEST_CASE("Gain and phase margin of first-order system") {
        // G(s) = 10/(s+1) — gain margin infinite (phase never crosses -180°)
        // Phase margin: at crossover |G(jω)| = 1, find phase
        StateSpace<1, 1, 1, 0, 0> sys{
            .A = Matrix<1, 1>{{-1.0}},
            .B = Matrix<1, 1>{{10.0}},
            .C = Matrix<1, 1>{{1.0}},
            .D = Matrix<1, 1>{{0.0}}
        };

        auto omega = analysis::logspace(0.01, 1000.0, 1000);
        auto result = analysis::bode(sys, omega);

        // Phase margin should exist (there is a 0 dB crossing)
        auto pm = result.phase_margin();
        REQUIRE(pm.has_value());
        CHECK(pm->first > 0.0);  // Positive = stable
        CHECK(pm->second > 0.0); // Crossover frequency

        // Gain margin should not exist (first-order system phase never hits -180°)
        auto gm = result.gain_margin();
        CHECK_FALSE(gm.has_value());
    }

    TEST_CASE("Phase margin of integrator with gain") {
        // G(s) = K/s — always 90° phase lag, PM = 90° when |G|=1 at ω=K
        // Use state-space: A=0, B=K, C=1, D=0
        double                    K = 5.0;
        StateSpace<1, 1, 1, 0, 0> sys{
            .A = Matrix<1, 1>{{0.0}},
            .B = Matrix<1, 1>{{K}},
            .C = Matrix<1, 1>{{1.0}},
            .D = Matrix<1, 1>{{0.0}}
        };

        auto omega = analysis::logspace(0.01, 1000.0, 2000);
        auto result = analysis::bode(sys, omega);

        auto pm = result.phase_margin();
        REQUIRE(pm.has_value());
        // Pure integrator has -90° phase everywhere, so PM = 180 - 90 = 90°
        CHECK(pm->first == doctest::Approx(90.0).epsilon(2.0));
    }
}

TEST_SUITE("Analysis - Phase Wrapping") {
    TEST_CASE("canonical phase margin maps to (-180, 180]") {
        CHECK(analysis::canonical_phase_margin(540.0) == doctest::Approx(180.0));
        CHECK(analysis::canonical_phase_margin(181.0) == doctest::Approx(-179.0));
        CHECK(analysis::canonical_phase_margin(-180.0) == doctest::Approx(180.0));
        CHECK(analysis::canonical_phase_margin(-179.0) == doctest::Approx(-179.0));
    }

    TEST_CASE("canonical phase margin supports float") {
        constexpr float wrapped = analysis::canonical_phase_margin(181.0f);
        static_assert(wrapped < 0.0f);
        CHECK(wrapped == doctest::Approx(-179.0f));
    }
}

TEST_SUITE("Analysis - DC Gain") {
    TEST_CASE("DC gain of first-order system") {
        // G(s) = 5/(s+2) — DC gain = 5/2 = 2.5
        StateSpace<1, 1, 1, 0, 0> sys{
            .A = Matrix<1, 1>{{-2.0}},
            .B = Matrix<1, 1>{{5.0}},
            .C = Matrix<1, 1>{{1.0}},
            .D = Matrix<1, 1>{{0.0}}
        };

        auto dc = analysis::dcgain(sys);
        REQUIRE(dc.has_value());
        CHECK((*dc)(0, 0) == doctest::Approx(2.5).epsilon(1e-12));
    }

    TEST_CASE("DC gain of discrete system") {
        // Discrete first-order: A=0.9, B=0.1, C=1, D=0
        // DC gain = C*(I-A)^{-1}*B = 1*(1-0.9)^{-1}*0.1 = 10*0.1 = 1.0
        StateSpace<1, 1, 1, 0, 0> sys{
            .A = Matrix<1, 1>{{0.9}},
            .B = Matrix<1, 1>{{0.1}},
            .C = Matrix<1, 1>{{1.0}},
            .D = Matrix<1, 1>{{0.0}},
            .Ts = 0.01
        };

        auto dc = analysis::dcgain(sys);
        REQUIRE(dc.has_value());
        CHECK((*dc)(0, 0) == doctest::Approx(1.0).epsilon(1e-10));
    }
}

TEST_SUITE("Analysis - Pole Analysis") {
    TEST_CASE("Natural frequency and damping of 2nd order system") {
        // G(s) = wn²/(s² + 2*zeta*wn*s + wn²)
        // Poles at s = -zeta*wn ± wn*sqrt(1-zeta²)*j
        double       wn = 10.0;
        double       zeta = 0.7;
        Matrix<2, 2> A{{0.0, 1.0}, {-wn * wn, -2 * zeta * wn}};

        auto info = analysis::damp(A);

        // Both poles should have natural frequency ≈ wn
        for (const auto& p : info) {
            CHECK(p.natural_freq == doctest::Approx(wn).epsilon(1e-6));
            CHECK(p.damping_ratio == doctest::Approx(zeta).epsilon(1e-6));
        }
    }

    TEST_CASE("Continuous stability check") {
        // Stable: eigenvalues in LHP
        Matrix<2, 2> A_stable{{0.0, 1.0}, {-2.0, -3.0}};
        CHECK(analysis::is_stable_continuous(A_stable));

        // Unstable: positive real eigenvalue
        Matrix<2, 2> A_unstable{{0.0, 1.0}, {2.0, -1.0}};
        CHECK_FALSE(analysis::is_stable_continuous(A_unstable));
    }

    TEST_CASE("Pole computation") {
        Matrix<2, 2> A{{0.0, 1.0}, {-6.0, -5.0}};
        auto         p = analysis::poles(A);
        // Poles at s = -2, -3
        // Sort by real part
        auto r0 = p[0].real();
        auto r1 = p[1].real();
        if (r0 > r1) {
            wet::swap(r0, r1);
        }
        CHECK(r0 == doctest::Approx(-3.0).epsilon(1e-10));
        CHECK(r1 == doctest::Approx(-2.0).epsilon(1e-10));
    }
}

TEST_SUITE("Analysis - Unwrapped Margins and Discrete Bode") {
    TEST_CASE("Phase unwrap removes 180-degree discontinuity") {
        std::vector<double> wrapped{170.0, 179.0, -179.0, -170.0};
        auto                unwrapped = analysis::unwrap_phase_deg(wrapped);

        REQUIRE(unwrapped.size() == wrapped.size());
        CHECK(unwrapped[0] == doctest::Approx(170.0));
        CHECK(unwrapped[1] == doctest::Approx(179.0));
        CHECK(unwrapped[2] == doctest::Approx(181.0));
        CHECK(unwrapped[3] == doctest::Approx(190.0));
    }

    TEST_CASE("Unwrapped phase and gain margins handle wrapped trajectories") {
        analysis::BodeResult<double> result_pm{};
        result_pm.points.push_back({1.0, 1.122018454, 1.0, 170.0});
        result_pm.points.push_back({2.0, 0.891250938, -1.0, 170.0});

        auto pm = analysis::phase_margin_unwrapped(result_pm);
        REQUIRE(pm.has_value());
        CHECK(pm->first == doctest::Approx(-10.0).epsilon(1e-9));
        CHECK(pm->second == doctest::Approx(1.5).epsilon(1e-9));

        analysis::BodeResult<double> result_gm{};
        result_gm.points.push_back({10.0, 0.501187234, -6.0, -179.0});
        result_gm.points.push_back({20.0, 0.630957344, -4.0, 179.0});

        auto gm = analysis::gain_margin_unwrapped(result_gm);
        REQUIRE(gm.has_value());
        CHECK(gm->first == doctest::Approx(5.0).epsilon(1e-9));
        CHECK(gm->second == doctest::Approx(15.0).epsilon(1e-9));
    }

    TEST_CASE("Discrete bode evaluates frequency response on unit circle") {
        StateSpace<1, 1, 1, 0, 0> sys{
            .A = Matrix<1, 1>{{0.9}},
            .B = Matrix<1, 1>{{0.1}},
            .C = Matrix<1, 1>{{1.0}},
            .D = Matrix<1, 1>{{0.0}},
            .Ts = 0.01,
        };

        std::vector<double> omega{0.1, 10.0};
        auto                result = analysis::bode_discrete(sys, omega);

        REQUIRE(result.points.size() == omega.size());
        CHECK(result.points[0].magnitude_db == doctest::Approx(0.0).epsilon(0.1));

        // Cross-check one point against direct FRF evaluation at z = exp(j*w*Ts)
        const double               w = omega[1];
        const wet::complex<double> z{std::cos(w * sys.Ts), std::sin(w * sys.Ts)};
        const auto                 frf = eval_frf(sys, z);
        const double               mag_db_ref = 20.0 * std::log10(wet::abs(frf(0, 0)));
        CHECK(result.points[1].magnitude_db == doctest::Approx(mag_db_ref).epsilon(1e-10));
    }
}

TEST_SUITE("Analysis - Loop and Nyquist Utilities") {
    TEST_CASE("Continuous-time loop response returns L, S, T and Nyquist data") {
        // L(s) = 10/(s+1)
        StateSpace<1, 1, 1, 0, 0> loop{
            .A = Matrix<1, 1>{{-1.0}},
            .B = Matrix<1, 1>{{10.0}},
            .C = Matrix<1, 1>{{1.0}},
            .D = Matrix<1, 1>{{0.0}},
        };

        auto omega = analysis::logspace(0.01, 1000.0, 200);
        auto resp = analysis::loop_response(loop, omega);

        REQUIRE(resp.open_loop.points.size() == omega.size());
        REQUIRE(resp.sensitivity.points.size() == omega.size());
        REQUIRE(resp.complementary_sensitivity.points.size() == omega.size());
        REQUIRE(resp.nyquist.points.size() == omega.size());

        // At low frequency, L ~ 10 -> 20 dB
        CHECK(resp.open_loop.points.front().magnitude_db == doctest::Approx(20.0).epsilon(0.05));

        // For L(0)=10: S(0)=1/11, T(0)=10/11
        CHECK(resp.sensitivity.points.front().magnitude == doctest::Approx(1.0 / 11.0).epsilon(0.02));
        CHECK(resp.complementary_sensitivity.points.front().magnitude == doctest::Approx(10.0 / 11.0).epsilon(0.02));

        auto min_dist = resp.nyquist.min_distance_to_minus_one();
        REQUIRE(min_dist.has_value());
        CHECK(min_dist->first > 0.0);

        auto pm = resp.phase_margin_unwrapped();
        REQUIRE(pm.has_value());
        CHECK(pm->first > 0.0);
    }

    TEST_CASE("Discrete-time nyquist utility matches direct FRF") {
        StateSpace<1, 1, 1, 0, 0> sys{
            .A = Matrix<1, 1>{{0.9}},
            .B = Matrix<1, 1>{{0.1}},
            .C = Matrix<1, 1>{{1.0}},
            .D = Matrix<1, 1>{{0.0}},
            .Ts = 0.01,
        };

        std::vector<double> omega{10.0};
        auto                nyq = analysis::nyquist(sys, omega);

        REQUIRE(nyq.points.size() == 1);

        const double               w = omega[0];
        const wet::complex<double> z{std::cos(w * sys.Ts), std::sin(w * sys.Ts)};
        const auto                 frf = eval_frf(sys, z);
        CHECK(nyq.points[0].real == doctest::Approx(frf(0, 0).real()).epsilon(1e-12));
        CHECK(nyq.points[0].imag == doctest::Approx(frf(0, 0).imag()).epsilon(1e-12));
    }
}

TEST_SUITE("Analysis - Loop Summary Convenience") {
    TEST_CASE("loop_metrics condenses common robustness metrics in one call") {
        // L(s) = 10/(s+1)
        StateSpace<1, 1, 1, 0, 0> loop{
            .A = Matrix<1, 1>{{-1.0}},
            .B = Matrix<1, 1>{{10.0}},
            .C = Matrix<1, 1>{{1.0}},
            .D = Matrix<1, 1>{{0.0}},
        };

        const auto omega = analysis::logspace(0.01, 1000.0, 200);
        const auto summary = analysis::loop_metrics(loop, omega);

        REQUIRE(summary.phase_margin.has_value());
        REQUIRE(summary.bandwidth.has_value());
        REQUIRE(summary.min_nyquist_distance.has_value());

        CHECK(summary.phase_margin->first > 0.0);
        CHECK(summary.bandwidth.value() > 0.0);
        CHECK(summary.min_nyquist_distance->first > 0.0);

        // Some stable loops (for example first-order with no -180 crossing)
        // have undefined gain margin; if present, it must be positive.
        if (summary.gain_margin.has_value()) {
            CHECK(summary.gain_margin->first > 0.0);
        }

        // Sensitivity peak should be finite and not absurdly large for this stable first-order loop.
        CHECK(std::isfinite(summary.peak_sensitivity_db));
        CHECK(summary.peak_sensitivity_db < 20.0);
    }

    TEST_CASE("summarize_loop_response matches explicit loop_response post-processing") {
        StateSpace<1, 1, 1, 0, 0> loop{
            .A = Matrix<1, 1>{{-2.0}},
            .B = Matrix<1, 1>{{6.0}},
            .C = Matrix<1, 1>{{1.0}},
            .D = Matrix<1, 1>{{0.0}},
        };

        const auto omega = analysis::logspace(0.05, 800.0, 180);
        const auto resp = analysis::loop_response(loop, omega);
        const auto summary = analysis::summarize_loop_response(resp);

        REQUIRE(summary.phase_margin.has_value());
        REQUIRE(summary.bandwidth.has_value());
        REQUIRE(summary.min_nyquist_distance.has_value());

        CHECK(summary.phase_margin->first == doctest::Approx(resp.phase_margin_unwrapped()->first).epsilon(1e-12));
        CHECK(summary.bandwidth.value() == doctest::Approx(resp.closed_loop_bandwidth().value()).epsilon(1e-12));
        CHECK(summary.min_nyquist_distance->first == doctest::Approx(resp.nyquist.min_distance_to_minus_one()->first).epsilon(1e-12));

        const auto gm_ref = resp.gain_margin_unwrapped();
        CHECK(summary.gain_margin.has_value() == gm_ref.has_value());
        if (summary.gain_margin.has_value() && gm_ref.has_value()) {
            CHECK(summary.gain_margin->first == doctest::Approx(gm_ref->first).epsilon(1e-12));
            CHECK(summary.gain_margin->second == doctest::Approx(gm_ref->second).epsilon(1e-12));
        }
    }

    TEST_CASE("loop utilities accept transfer function directly") {
        // L(s) = 10/(s+1)
        TransferFunction<1, 2, double> loop_tf{
            .num = {10.0},
            .den = {1.0, 1.0},
        };

        const auto omega = analysis::logspace(0.01, 1000.0, 200);
        const auto resp = analysis::loop_response(loop_tf, omega);
        const auto summary = analysis::loop_metrics(loop_tf, omega);

        REQUIRE(resp.open_loop.points.size() == omega.size());
        REQUIRE(summary.phase_margin.has_value());
        REQUIRE(summary.bandwidth.has_value());
        CHECK(summary.phase_margin->first > 0.0);
        CHECK(summary.bandwidth.value() > 0.0);
    }
}
