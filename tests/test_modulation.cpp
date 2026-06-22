#include <cmath>
#include <numbers>

#include "wet/utility/modulation.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wet;

/**
 * @brief Tests for reference-frame transforms and power-electronics modulation
 */

TEST_SUITE("Transforms & Modulation") {
    TEST_CASE("Clarke transform") {
        // Test balanced three-phase
        const ColVec<3, float> abc = {1.0f, -0.5f, -0.5f};

        const auto [alpha, beta] = clarke_transform(abc);

        // For balanced system: α = (2a - b - c)/3, β = (b - c)/√3
        CHECK(alpha == doctest::Approx(1.0f));
        CHECK(beta == doctest::Approx(0.0f).epsilon(1e-6f));
    }

    TEST_CASE("Inverse Clarke transform") {
        // Test round-trip
        const ColVec<3, float> abc_orig = {1.0f, -0.5f, -0.5f};

        const auto ab = clarke_transform(abc_orig);
        const auto abc = inverse_clarke_transform(ab);

        CHECK(abc[0] == doctest::Approx(abc_orig[0]).epsilon(1e-6f));
        CHECK(abc[1] == doctest::Approx(abc_orig[1]).epsilon(1e-6f));
        CHECK(abc[2] == doctest::Approx(abc_orig[2]).epsilon(1e-6f));
    }

    TEST_CASE("Park transform") {
        // Test with θ = 0 (should be identity)
        const AlphaBeta<float> ab = {.alpha = 1.0f, .beta = 0.5f};
        const float            theta = 0.0f;

        const auto [d, q] = park_transform(ab, theta);

        CHECK(d == doctest::Approx(ab.alpha));
        CHECK(q == doctest::Approx(ab.beta));
    }

    TEST_CASE("Park transform with rotation") {
        const AlphaBeta<float> ab = {.alpha = 1.0f, .beta = 0.0f};
        const float            theta = std::numbers::pi_v<float> / 4.0f; // 45°

        const auto [d, q] = park_transform(ab, theta);

        const float expected_d = ab.alpha * std::cos(theta);
        const float expected_q = -ab.alpha * std::sin(theta);

        CHECK(d == doctest::Approx(expected_d).epsilon(1e-6f));
        CHECK(q == doctest::Approx(expected_q).epsilon(1e-6f));
    }

    TEST_CASE("Inverse Park transform") {
        // Test round-trip
        const DirectQuadrature<float> dq_orig = {.d = 0.8f, .q = 0.3f};
        const float                   theta = std::numbers::pi_v<float> / 6.0f; // 30°

        const auto ab = inverse_park_transform(dq_orig, theta);
        const auto [d, q] = park_transform(ab, theta);

        CHECK(d == doctest::Approx(dq_orig.d).epsilon(1e-6f));
        CHECK(q == doctest::Approx(dq_orig.q).epsilon(1e-6f));
    }

    TEST_CASE("Clarke-Park combined transform") {
        // Test three-phase to dq
        const ColVec<3, float> abc = {
            std::cos(0.0f),
            std::cos(2.0f * std::numbers::pi_v<float> / 3.0f),
            std::cos(4.0f * std::numbers::pi_v<float> / 3.0f),
        };
        const float theta = 0.0f;

        const auto [d, q] = clarke_park_transform(abc, theta);

        // At θ = 0, d should be the amplitude, q should be 0
        CHECK(d == doctest::Approx(1.0f).epsilon(1e-6f)); // Clarke transform normalizes to 1.0
        CHECK(q == doctest::Approx(0.0f).epsilon(1e-6f));
    }

    TEST_CASE("Inverse Park-Clarke combined transform") {
        // Test dq to three-phase round-trip
        const DirectQuadrature<float> dq = {.d = 1.0f, .q = 0.5f};
        const float                   theta = std::numbers::pi_v<float> / 4.0f;

        const auto abc = inverse_park_clarke_transform(dq, theta);
        const auto [d2, q2] = clarke_park_transform(abc, theta);

        CHECK(d2 == doctest::Approx(dq.d).epsilon(1e-6f));
        CHECK(q2 == doctest::Approx(dq.q).epsilon(1e-6f));
    }

    TEST_CASE("SVM duty cycles") {
        // Test zero voltage
        const auto svm = svm_duty_cycles<float>({.alpha = 0.0f, .beta = 0.0f}, 100.0f);

        CHECK(svm.duties[0] == doctest::Approx(0.5f));
        CHECK(svm.duties[1] == doctest::Approx(0.5f));
        CHECK(svm.duties[2] == doctest::Approx(0.5f));
        CHECK_FALSE(svm.is_clipped);

        // Test maximum linear voltage (peak phase = Vdc/√3 with SVPWM injection)
        const float v_max = 100.0f / std::numbers::sqrt3_v<float>;
        const auto  svm_max = svm_duty_cycles<float>({.alpha = v_max, .beta = 0.0f}, 100.0f);

        CHECK(svm_max.duties[0] >= 0.0f);
        CHECK(svm_max.duties[0] <= 1.0f);
        CHECK(svm_max.duties[1] >= 0.0f);
        CHECK(svm_max.duties[1] <= 1.0f);
        CHECK(svm_max.duties[2] >= 0.0f);
        CHECK(svm_max.duties[2] <= 1.0f);
        CHECK_FALSE(svm_max.is_clipped); // exactly on the inscribed circle, not clipped
    }

    TEST_CASE("Clarke αβ0: common-mode DC lands in zero, not αβ") {
        // Identical DC bias on all three phases is pure common-mode.
        const float d = 0.7f;
        const auto  abz = clarke_zero_transform<float>({d, d, d});

        CHECK(abz.alpha == doctest::Approx(0.0f).epsilon(1e-6f));
        CHECK(abz.beta == doctest::Approx(0.0f).epsilon(1e-6f));
        CHECK(abz.zero == doctest::Approx(d).epsilon(1e-6f));
    }

    TEST_CASE("Clarke αβ0: per-phase DC leaks into αβ") {
        // Offset on phase a only: 2d/3 into alpha, d/3 into zero.
        const float d = 0.9f;
        const auto  abz = clarke_zero_transform<float>({d, 0.0f, 0.0f});

        CHECK(abz.alpha == doctest::Approx(2.0f * d / 3.0f).epsilon(1e-6f));
        CHECK(abz.beta == doctest::Approx(0.0f).epsilon(1e-6f));
        CHECK(abz.zero == doctest::Approx(d / 3.0f).epsilon(1e-6f));
    }

    TEST_CASE("Clarke αβ0 round-trip abc → αβ0 → abc") {
        const ColVec<3, float> abc = {1.3f, -0.4f, 0.2f};
        const auto             abz = clarke_zero_transform<float>(abc);
        const auto             rt = inverse_clarke_zero_transform<float>(abz);

        for (size_t i = 0; i < 3; ++i) {
            CHECK(rt[i] == doctest::Approx(abc[i]).epsilon(1e-6f));
        }
        // αβ part agrees with the plain Clarke transform.
        const auto ab = clarke_transform<float>(abc);
        CHECK(abz.ab().alpha == doctest::Approx(ab.alpha).epsilon(1e-6f));
        CHECK(abz.ab().beta == doctest::Approx(ab.beta).epsilon(1e-6f));
    }

    TEST_CASE("Symmetrical components: balanced set is pure positive sequence") {
        using Cplx = wet::complex<float>;
        // Balanced positive-sequence phasors: a, a·e^{-j120}, a·e^{-j240}
        const Cplx a = {1.0f, 0.0f};
        const Cplx b = {-0.5f, -std::numbers::sqrt3_v<float> / 2.0f}; // 1∠-120°
        const Cplx c = {-0.5f, std::numbers::sqrt3_v<float> / 2.0f};  // 1∠-240°

        const auto seq = symmetrical_components<float>({a, b, c});

        CHECK(seq.zero.abs() == doctest::Approx(0.0f).epsilon(1e-6f));
        CHECK(seq.negative.abs() == doctest::Approx(0.0f).epsilon(1e-6f));
        CHECK(seq.positive.real() == doctest::Approx(1.0f).epsilon(1e-6f));
        CHECK(seq.positive.imag() == doctest::Approx(0.0f).epsilon(1e-6f));
    }

    TEST_CASE("Symmetrical components: round-trip 012 → abc → 012") {
        using Cplx = wet::complex<float>;
        const ColVec<3, Cplx> abc = {Cplx{1.0f, 0.2f}, Cplx{-0.4f, -0.9f}, Cplx{0.1f, 0.7f}};

        const auto seq = symmetrical_components<float>(abc);
        const auto rt = inverse_symmetrical_components<float>(seq);

        for (size_t i = 0; i < 3; ++i) {
            CHECK(rt[i].real() == doctest::Approx(abc[i].real()).epsilon(1e-6f));
            CHECK(rt[i].imag() == doctest::Approx(abc[i].imag()).epsilon(1e-6f));
        }
    }

    TEST_CASE("Power-invariant Clarke: magnitude scaling and round-trip") {
        using wet::Convention;
        const ColVec<3, float> abc = {1.0f, -0.4f, 0.3f};

        const auto amp = clarke_transform<float>(abc); // amplitude-invariant (default)
        const auto pwr = clarke_transform<float, Convention::PowerInvariant>(abc);

        // Power-invariant αβ is √(3/2)× the amplitude-invariant αβ.
        const float ratio = std::sqrt(1.5f);
        CHECK(pwr.alpha == doctest::Approx(amp.alpha * ratio).epsilon(1e-6f));
        CHECK(pwr.beta == doctest::Approx(amp.beta * ratio).epsilon(1e-6f));

        // Round-trip through the power-invariant inverse (rank-3, exact).
        const auto abz = clarke_zero_transform<float, Convention::PowerInvariant>(abc);
        const auto rt = inverse_clarke_zero_transform<float, Convention::PowerInvariant>(abz);
        for (size_t i = 0; i < 3; ++i) {
            CHECK(rt[i] == doctest::Approx(abc[i]).epsilon(1e-6f));
        }
    }

    TEST_CASE("dq0 round-trip abc → dq0 → abc and zero passthrough") {
        const ColVec<3, float> abc = {1.3f, -0.4f, 0.2f};
        const float            theta = 0.9f;

        const auto dqz = clarke_park_zero_transform<float>(abc, theta);
        const auto rt = inverse_park_clarke_zero_transform<float>(dqz, theta);
        for (size_t i = 0; i < 3; ++i) {
            CHECK(rt[i] == doctest::Approx(abc[i]).epsilon(1e-5f));
        }

        // Zero axis matches the Clarke zero and is unaffected by rotation.
        const auto abz = clarke_zero_transform<float>(abc);
        CHECK(dqz.zero == doctest::Approx(abz.zero).epsilon(1e-6f));
        // dq part agrees with the zero-free fused transform.
        const auto dq = clarke_park_transform<float>(abc, theta);
        CHECK(dqz.dq().d == doctest::Approx(dq.d).epsilon(1e-6f));
        CHECK(dqz.dq().q == doctest::Approx(dq.q).epsilon(1e-6f));
    }

    TEST_CASE("Instantaneous power: balanced load, αβ and dq agree") {
        // Balanced 3φ voltage, unity-PF current (in phase), amplitude 1 each.
        const float            tp = 2.0f * std::numbers::pi_v<float> / 3.0f;
        const float            th = 0.37f;
        const ColVec<3, float> v = {std::cos(th), std::cos(th - tp), std::cos(th + tp)};
        const ColVec<3, float> i = v; // unity power factor, equal amplitude

        const auto p_ab = instantaneous_power<float>(clarke_transform<float>(v), clarke_transform<float>(i));

        // P = 3/2 · V·I (peak) = 1.5 for unit peak, balanced, unity PF; Q = 0.
        CHECK(p_ab.p == doctest::Approx(1.5f).epsilon(1e-5f));
        CHECK(p_ab.q == doctest::Approx(0.0f).epsilon(1e-5f));

        // dq form agrees (frame-invariant scalars).
        const auto p_dq = instantaneous_power<float>(clarke_park_transform<float>(v, th), clarke_park_transform<float>(i, th));
        CHECK(p_dq.p == doctest::Approx(p_ab.p).epsilon(1e-5f));
        CHECK(p_dq.q == doctest::Approx(p_ab.q).epsilon(1e-5f));
    }

    TEST_CASE("Instantaneous power: convention scaling differs by 3/2") {
        using wet::Convention;
        const ColVec<3, float> v = {1.0f, -0.5f, -0.5f};
        const ColVec<3, float> i = {0.8f, -0.2f, -0.6f};

        const auto p_amp = instantaneous_power<float>(clarke_transform<float>(v), clarke_transform<float>(i));
        const auto vp = clarke_transform<float, Convention::PowerInvariant>(v);
        const auto ip = clarke_transform<float, Convention::PowerInvariant>(i);
        const auto p_pwr = instantaneous_power<float, Convention::PowerInvariant>(vp, ip);

        // Both report the same real watts.
        CHECK(p_pwr.p == doctest::Approx(p_amp.p).epsilon(1e-5f));
        CHECK(p_pwr.q == doctest::Approx(p_amp.q).epsilon(1e-5f));
    }

    TEST_CASE("InstantaneousPower: apparent power, angle, power factor") {
        // v on d-axis, current at 45° (equal active/reactive) → φ = -45°.
        const auto s = instantaneous_power<float>(DirectQuadrature<float>{1.0f, 0.0f}, DirectQuadrature<float>{1.0f, 1.0f});

        CHECK(s.p == doctest::Approx(1.5f));
        CHECK(s.q == doctest::Approx(-1.5f));
        CHECK(s.abs() == doctest::Approx(1.5f * std::numbers::sqrt2_v<float>));
        CHECK(s.arg() == doctest::Approx(-std::numbers::pi_v<float> / 4.0f));
        CHECK(s.power_factor() == doctest::Approx(1.0f / std::numbers::sqrt2_v<float>));

        // Degenerate: zero power → zero pf, no division by zero.
        CHECK(InstantaneousPower<float>{}.power_factor() == doctest::Approx(0.0f));
    }

    TEST_CASE("Complex power via αβ conj/product matches instantaneous_power") {
        const AlphaBeta<float> v = {0.9f, -0.3f};
        const AlphaBeta<float> i = {0.4f, 0.7f};

        // S = 3/2 · V · conj(I), expressed directly with the complex operators.
        const AlphaBeta<float> s = (v * i.conj()) * 1.5f;
        const auto             p = instantaneous_power<float>(v, i);

        CHECK(s.alpha == doctest::Approx(p.p));
        CHECK(s.beta == doctest::Approx(p.q));

        // conj() and complex operator* sanity: j · conj(j) = j · (−j) = 1.
        const AlphaBeta<float> j = {0.0f, 1.0f};
        const AlphaBeta<float> one = j * j.conj();
        CHECK(one.alpha == doctest::Approx(1.0f));
        CHECK(one.beta == doctest::Approx(0.0f));
    }

} // TEST_SUITE
