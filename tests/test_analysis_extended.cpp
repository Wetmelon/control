#include <cmath>
#include <numbers>

#include "analysis.hpp"
#include "discretization.hpp"
#include "doctest.h"
#include "matlab.hpp"
#include "matrix.hpp"
#include "state_space.hpp"
#include "utility.hpp"

using namespace wetmelon::control;

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

        auto Co = analysis::ctrb(A, B);
        CHECK(analysis::rank(Co) == 2);
        CHECK(analysis::is_controllable(A, B));

        // Compile-time check
        static_assert(analysis::is_controllable(
            Matrix<2, 2>{{0.0, 1.0}, {-2.0, -3.0}},
            Matrix<2, 1>{{0.0}, {1.0}}
        ));
    }

    TEST_CASE("Uncontrollable 2x2 system") {
        // A = [1 0; 0 2], B = [0; 0] — zero input, uncontrollable
        Matrix<2, 2> A{{1.0, 0.0}, {0.0, 2.0}};
        Matrix<2, 1> B{{0.0}, {0.0}};

        CHECK_FALSE(analysis::is_controllable(A, B));
    }

    TEST_CASE("Observable 2x2 system") {
        Matrix<2, 2> A{{0.0, 1.0}, {-2.0, -3.0}};
        Matrix<1, 2> C{{1.0, 0.0}};

        auto Ob = analysis::obsv(A, C);
        CHECK(analysis::rank(Ob) == 2);
        CHECK(analysis::is_observable(A, C));
    }

    TEST_CASE("Unobservable 2x2 system") {
        // A diagonal, C only sees first state, second state unobservable
        Matrix<2, 2> A{{1.0, 0.0}, {0.0, 2.0}};
        Matrix<1, 2> C{{1.0, 0.0}};

        auto Ob = analysis::obsv(A, C);
        // Ob = [1 0; 1 0] — rank 1, not rank 2
        CHECK(analysis::rank(Ob) == 1);
        CHECK_FALSE(analysis::is_observable(A, C));
    }

    TEST_CASE("Controllable 3x3 system") {
        // Companion form: always controllable
        Matrix<3, 3> A{{0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {-6.0, -11.0, -6.0}};
        Matrix<3, 1> B{{0.0}, {0.0}, {1.0}};

        CHECK(analysis::is_controllable(A, B));
    }

    TEST_CASE("MIMO controllability") {
        Matrix<2, 2> A{{0.0, 1.0}, {-2.0, -3.0}};
        Matrix<2, 2> B{{1.0, 0.0}, {0.0, 1.0}};

        auto Co = analysis::ctrb(A, B);
        // Controllability matrix is 2x4, should have rank 2
        CHECK(analysis::rank(Co) == 2);
        CHECK(analysis::is_controllable(A, B));
    }

    TEST_CASE("Matrix rank computation") {
        // Full rank 3x3
        Matrix<3, 3> M1{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
        CHECK(analysis::rank(M1) == 3);

        // Rank 2 (third row = first + second)
        Matrix<3, 3> M2{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {1.0, 1.0, 0.0}};
        CHECK(analysis::rank(M2) == 2);

        // Rank 1
        Matrix<3, 3> M3{{1.0, 2.0, 3.0}, {2.0, 4.0, 6.0}, {3.0, 6.0, 9.0}};
        CHECK(analysis::rank(M3) == 1);
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

        auto omega = logspace(0.01, 100.0, 200);
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

        auto omega = logspace(0.1, 1000.0, 500);
        auto result = analysis::bode(sys, omega);

        // DC gain should be 0 dB (unity)
        CHECK(result.points[0].magnitude_db == doctest::Approx(0.0).epsilon(0.1));

        // There should be a resonant peak around wn (zeta < 0.707)
        double peak_db = -300;
        for (const auto& pt : result.points) {
            if (pt.magnitude_db > peak_db) {
                peak_db = pt.magnitude_db;
            }
        }
        CHECK(peak_db > 0.0); // Should have resonant peak above 0 dB
    }

    TEST_CASE("Bode from transfer function coefficients") {
        // G(s) = 10/(s+10), ascending powers: num = {10}, den = {10, 1}
        std::array<double, 1> num = {10.0};
        std::array<double, 2> den = {10.0, 1.0};

        auto omega = logspace(0.1, 1000.0, 200);
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

        auto omega = logspace(0.01, 1000.0, 1000);
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

        auto omega = logspace(0.01, 1000.0, 2000);
        auto result = analysis::bode(sys, omega);

        auto pm = result.phase_margin();
        REQUIRE(pm.has_value());
        // Pure integrator has -90° phase everywhere, so PM = 180 - 90 = 90°
        CHECK(pm->first == doctest::Approx(90.0).epsilon(2.0));
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
            std::swap(r0, r1);
        }
        CHECK(r0 == doctest::Approx(-3.0).epsilon(1e-10));
        CHECK(r1 == doctest::Approx(-2.0).epsilon(1e-10));
    }
}
