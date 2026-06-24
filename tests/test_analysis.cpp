#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <type_traits>
#include <vector>

#include "wet/analysis/analysis.hpp"
#include "wet/design/stability.hpp"
#include "wet/math/complex.hpp"
#include "wet/matrix/matrix.hpp"
#include "wet/systems/state_space.hpp"
#include "wet/systems/transfer_function.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wet;

TEST_CASE("poles() handles systems larger than 4 states") {
    // Diagonal A with known eigenvalues -1..-6 (6 states > old NX<=4 limit).
    Matrix<6, 6, double> A{};
    const double         expected[6] = {-1.0, -2.0, -3.0, -4.0, -5.0, -6.0};
    for (size_t i = 0; i < 6; ++i) {
        A(i, i) = expected[i];
    }
    auto p = analysis::poles(A);
    for (size_t i = 0; i < 6; ++i) {
        CHECK(p[i].real() == doctest::Approx(expected[i]).epsilon(1e-9));
        CHECK(p[i].imag() == doctest::Approx(0.0).epsilon(1e-9));
    }
    CHECK(analysis::is_stable_continuous(A));
}

TEST_CASE("step/impulse/initial response of 1/(s+1)") {
    // 1/(s+1): A=-1, B=1, C=1, D=0
    TransferFunction<1, 2, double> tf{{1.0}, {1.0, 1.0}};
    auto                           t = analysis::linspace(0.0, 5.0, 501);

    auto s = analysis::step(tf, t);
    auto i = analysis::impulse(tf, t);
    for (size_t k = 0; k < t.size(); ++k) {
        CHECK(s.y[k](0, 0) == doctest::Approx(1.0 - std::exp(-t[k])).epsilon(1e-3));
        CHECK(i.y[k](0, 0) == doctest::Approx(std::exp(-t[k])).epsilon(1e-3));
    }

    // initial response from x0=2: y(t) = 2 e^{-t}
    auto      ss = tf.to_state_space();
    ColVec<1> x0{2.0};
    auto      in = analysis::initial(ss, x0, t);
    for (size_t k = 0; k < t.size(); ++k) {
        CHECK(in.y[k][0] == doctest::Approx(2.0 * std::exp(-t[k])).epsilon(1e-3));
    }
}

TEST_CASE("step response of a 2x2 MIMO system separates input channels") {
    // Two decoupled first-order plants: A=diag(-1,-2), B=C=I, D=0.
    // Step on input j must drive only output j: y[k](i,j)=0 for i!=j.
    Matrix<2, 2, double> A{{-1.0, 0.0}, {0.0, -2.0}};
    Matrix<2, 2, double> B{{1.0, 0.0}, {0.0, 1.0}};
    Matrix<2, 2, double> C{{1.0, 0.0}, {0.0, 1.0}};
    const auto           sys = StateSpace{A, B, C};

    const auto s = analysis::step(sys, analysis::linspace(0.0, 5.0, 501));
    for (size_t k = 0; k < s.t.size(); ++k) {
        CHECK(s.y[k](0, 0) == doctest::Approx(1.0 - std::exp(-s.t[k])).epsilon(1e-3));
        CHECK(s.y[k](1, 1) == doctest::Approx(0.5 * (1.0 - std::exp(-2.0 * s.t[k]))).epsilon(1e-3));
        CHECK(s.y[k](0, 1) == doctest::Approx(0.0).epsilon(1e-9));
        CHECK(s.y[k](1, 0) == doctest::Approx(0.0).epsilon(1e-9));
    }
}

TEST_CASE("lsim drives a system with an arbitrary input (MIMO output)") {
    // 1/(s+1): step input via lsim should match the closed-form 1 - e^{-t}.
    // Two-output C = [1; 2] checks the MIMO result shape (y[k] is a ColVec).
    Matrix<1, 1, double> A{{-1.0}};
    Matrix<1, 1, double> B{{1.0}};
    Matrix<2, 1, double> C{{1.0}, {2.0}};
    const auto           sys = StateSpace{A, B, C};

    const auto                t = analysis::linspace(0.0, 5.0, 501);
    const std::vector<double> u(t.size(), 1.0); // unit step
    const auto                res = analysis::lsim(sys, u, t);

    REQUIRE(res.y.size() == t.size());
    for (size_t k = 0; k < t.size(); ++k) {
        const double expected = 1.0 - std::exp(-t[k]);
        CHECK(res.y[k][0] == doctest::Approx(expected).epsilon(1e-3));
        CHECK(res.y[k][1] == doctest::Approx(2.0 * expected).epsilon(1e-3));
    }
}

TEST_CASE("stepinfo on a first-order system: no overshoot, known rise/settling") {
    // 1/(s+1): y=1-e^{-t}. Rise(10-90%)=ln(9)≈2.197, settle(2%)=ln(50)≈3.912.
    TransferFunction<1, 2, double> tf{{1.0}, {1.0, 1.0}};
    const auto                     info = analysis::stepinfo(tf.to_state_space(), analysis::linspace(0.0, 10.0, 2001));

    CHECK(info.overshoot == doctest::Approx(0.0).epsilon(1e-6));
    CHECK(info.rise_time == doctest::Approx(2.197).epsilon(2e-2));
    CHECK(info.settling_time == doctest::Approx(3.912).epsilon(2e-2));
}

TEST_CASE("stepinfo on an underdamped 2nd-order system matches the overshoot formula") {
    // 1/(s^2+s+1): wn=1, zeta=0.5 → overshoot = exp(-pi*zeta/sqrt(1-zeta^2))*100 ≈ 16.3%.
    TransferFunction<1, 3, double> tf{{1.0}, {1.0, 1.0, 1.0}};
    const auto                     y = analysis::step(tf.to_state_space(), analysis::linspace(0.0, 20.0, 4001));

    std::vector<double> sig, t = y.t;
    for (const auto& yk : y.y) {
        sig.push_back(yk(0, 0));
    }
    const auto info = analysis::stepinfo(sig, t, sig.back());

    const double expected_os = std::exp(-std::numbers::pi_v<double> * 0.5 / std::sqrt(1.0 - 0.25)) * 100.0;
    CHECK(info.overshoot == doctest::Approx(expected_os).epsilon(2e-2));
    CHECK(info.peak > 1.0); // overshoots past the steady-state value of 1
}

TEST_CASE("lsiminfo reports extremes and settling of a signal") {
    // 1/(s+1) step: min at t=0 (=0), max →1, settles within 2% at ln(50)≈3.912.
    TransferFunction<1, 2, double> tf{{1.0}, {1.0, 1.0}};
    const auto                     resp = analysis::step(tf.to_state_space(), analysis::linspace(0.0, 10.0, 2001));

    std::vector<double> sig, t = resp.t;
    for (const auto& yk : resp.y) {
        sig.push_back(yk(0, 0));
    }
    const auto info = analysis::lsiminfo(sig, t, sig.back());

    CHECK(info.min == doctest::Approx(0.0).epsilon(1e-9));
    CHECK(info.min_time == doctest::Approx(0.0).epsilon(1e-9));
    CHECK(info.max == doctest::Approx(1.0).epsilon(1e-2));
    CHECK(info.settling_time == doctest::Approx(3.912).epsilon(2e-2));
}

TEST_CASE("pzmap of a transfer function finds poles and zeros") {
    // (s+2) / (s^2 + 3s + 2) = (s+2)/((s+1)(s+2)): zero at -2, poles at -1,-2.
    TransferFunction<2, 3, double> tf{{2.0, 1.0}, {2.0, 3.0, 1.0}};
    const auto                     pz = analysis::pzmap(tf);

    REQUIRE(pz.zeros.size() == 1);
    REQUIRE(pz.poles.size() == 2);
    CHECK(pz.zeros[0].real() == doctest::Approx(-2.0).epsilon(1e-9));

    // Poles are {-1, -2} in some order.
    double pmin = std::min(pz.poles[0].real(), pz.poles[1].real());
    double pmax = std::max(pz.poles[0].real(), pz.poles[1].real());
    CHECK(pmin == doctest::Approx(-2.0).epsilon(1e-9));
    CHECK(pmax == doctest::Approx(-1.0).epsilon(1e-9));
}

TEST_CASE("pzmap of a state-space system returns eigenvalue poles") {
    Matrix<2, 2, double> A{{0.0, 1.0}, {-2.0, -3.0}}; // char poly s^2+3s+2 → -1,-2
    const auto           pz = analysis::pzmap(A);

    REQUIRE(pz.poles.size() == 2);
    CHECK(pz.zeros.empty()); // transmission zeros not computed
    double pmin = std::min(pz.poles[0].real(), pz.poles[1].real());
    double pmax = std::max(pz.poles[0].real(), pz.poles[1].real());
    CHECK(pmin == doctest::Approx(-2.0).epsilon(1e-9));
    CHECK(pmax == doctest::Approx(-1.0).epsilon(1e-9));
}

TEST_CASE("linspace and logspace support float") {
    const auto lin = analysis::linspace(0.0f, 1.0f, 5);
    const auto log = analysis::logspace(1.0f, 100.0f, 3);

    CHECK(std::is_same_v<typename decltype(lin)::value_type, float>);
    CHECK(std::is_same_v<typename decltype(log)::value_type, float>);

    CHECK(lin.front() == doctest::Approx(0.0f));
    CHECK(lin.back() == doctest::Approx(1.0f));
    CHECK(log.front() == doctest::Approx(1.0f));
    CHECK(log.back() == doctest::Approx(100.0f));
}

TEST_SUITE("Stability Analysis") {
    TEST_CASE("Discrete stability check - stable system") {
        // Stable discrete system: eigenvalues inside unit circle
        Matrix<2, 2> A{{0.5, 0.0}, {0.0, 0.8}};
        CHECK(stability::is_stable_discrete(A) == true);
        CHECK(stability::stability_margin_discrete(A) > 0.0);
    }

    TEST_CASE("Discrete stability check - unstable system") {
        // Unstable: eigenvalue outside unit circle
        Matrix<2, 2> A{{1.5, 0.0}, {0.0, 0.5}};
        CHECK(stability::is_stable_discrete(A) == false);
        CHECK(stability::stability_margin_discrete(A) < 0.0);
    }
}
