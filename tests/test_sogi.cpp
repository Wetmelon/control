
#include <cmath>
#include <numbers>

#include "sogi.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wetmelon::control;

TEST_CASE("SOGI design") {
    constexpr double omega_0 = 2 * std::numbers::pi * 50.0; // 50 Hz
    constexpr double k = 1.414;
    constexpr auto   sogi_sys = design::sogi_system<double>(omega_0, k);

    // Check matrix dimensions
    CHECK(sogi_sys.A.rows() == 2);
    CHECK(sogi_sys.A.cols() == 2);
    CHECK(sogi_sys.B.rows() == 2);
    CHECK(sogi_sys.B.cols() == 1);
    CHECK(sogi_sys.C.rows() == 2);
    CHECK(sogi_sys.C.cols() == 2);

    // Check A matrix structure
    CHECK(sogi_sys.A(0, 0) == doctest::Approx(-k * omega_0));
    CHECK(sogi_sys.A(0, 1) == doctest::Approx(-omega_0));
    CHECK(sogi_sys.A(1, 0) == doctest::Approx(omega_0));
    CHECK(sogi_sys.A(1, 1) == doctest::Approx(0.0));

    // Check B matrix
    CHECK(sogi_sys.B(0, 0) == doctest::Approx(k * omega_0));
    CHECK(sogi_sys.B(1, 0) == doctest::Approx(0.0));

    // Check C matrix (identity for direct state output)
    CHECK(sogi_sys.C(0, 0) == doctest::Approx(1.0));
    CHECK(sogi_sys.C(0, 1) == doctest::Approx(0.0));
    CHECK(sogi_sys.C(1, 0) == doctest::Approx(0.0));
    CHECK(sogi_sys.C(1, 1) == doctest::Approx(1.0));
}

TEST_CASE("MSOGI design") {
    constexpr double omega_0 = 2 * std::numbers::pi * 50.0;
    constexpr double k = 1.414;
    constexpr auto   msogi_sys = design::mstogi_system<double>(omega_0, k);

    // Check dimensions
    CHECK(msogi_sys.A.rows() == 3);
    CHECK(msogi_sys.A.cols() == 3);
    CHECK(msogi_sys.B.rows() == 3);
    CHECK(msogi_sys.B.cols() == 1);
    CHECK(msogi_sys.C.rows() == 2);
    CHECK(msogi_sys.C.cols() == 3);

    // Check A matrix structure
    CHECK(msogi_sys.A(0, 0) == doctest::Approx(-k * omega_0));
    CHECK(msogi_sys.A(0, 1) == doctest::Approx(-omega_0));
    CHECK(msogi_sys.A(1, 0) == doctest::Approx(omega_0));
    CHECK(msogi_sys.A(2, 1) == doctest::Approx(-omega_0 * omega_0));
}

TEST_CASE("SOGI runtime - resonator form") {
    constexpr float f0 = 50.0f;     // 50 Hz fundamental
    constexpr float Ts = 0.0001f;   // 10 kHz sample rate
    constexpr float alpha = 1.414f; // √2 damping gain
    SOGI<float>     sogi(f0, Ts, alpha);

    sogi.reset();

    // Test with sinusoidal input at resonant frequency
    const float omega_input = 2 * std::numbers::pi_v<float> * f0;

    float max_bp = 0.0f;
    float max_quad = 0.0f;

    // Run for a few cycles
    for (int i = 0; i < 1000; ++i) {
        const float t = i * Ts;
        const float input = std::sin(omega_input * t);

        const auto [bp, quad] = sogi(input);

        max_bp = std::max(max_bp, std::abs(bp));
        max_quad = std::max(max_quad, std::abs(quad));
    }

    // Should extract fundamental component
    CHECK(max_bp > 0.1f);
    CHECK(max_quad > 0.1f);
}

TEST_CASE("SOGI runtime - frequency retuning") {
    constexpr float Ts = 0.0001f; // 10 kHz sample rate
    constexpr float alpha = 1.414f;
    SOGI<float>     sogi(50.0f, Ts, alpha);

    // Retune to 60 Hz
    sogi.set_frequency(60.0f, Ts);

    const float omega_input = 2 * std::numbers::pi_v<float> * 60.0f;
    float       max_bp = 0.0f;

    for (int i = 0; i < 1000; ++i) {
        const float t = i * Ts;
        const float input = std::sin(omega_input * t);
        const auto [bp, quad] = sogi(input);
        max_bp = std::max(max_bp, std::abs(bp));
    }

    CHECK(max_bp > 0.1f);
}

TEST_CASE("SOGI design with notch output") {
    constexpr double omega_0 = 2 * std::numbers::pi * 50.0;
    constexpr double k = 1.414;
    constexpr auto   sogi_notch_sys = design::sogi_system_with_notch<double>(omega_0, k);

    CHECK(sogi_notch_sys.A.rows() == 2);
    CHECK(sogi_notch_sys.A.cols() == 2);
    CHECK(sogi_notch_sys.B.rows() == 2);
    CHECK(sogi_notch_sys.B.cols() == 1);
    CHECK(sogi_notch_sys.C.rows() == 3);
    CHECK(sogi_notch_sys.C.cols() == 2);
    CHECK(sogi_notch_sys.D.rows() == 3);
    CHECK(sogi_notch_sys.D.cols() == 1);

    // notch output y_notch = u - x1
    CHECK(sogi_notch_sys.C(2, 0) == doctest::Approx(-1.0));
    CHECK(sogi_notch_sys.C(2, 1) == doctest::Approx(0.0));
    CHECK(sogi_notch_sys.D(2, 0) == doctest::Approx(1.0));
}

TEST_CASE("SOGI runtime exposes notch output") {
    constexpr float f0 = 50.0f;
    constexpr float Ts = 0.0001f;
    constexpr float alpha = 1.414f;
    SOGI<float>     sogi(f0, Ts, alpha);

    for (int i = 0; i < 1000; ++i) {
        const float t = i * Ts;
        const float input = std::sin(2 * std::numbers::pi_v<float> * f0 * t);

        const auto out = sogi.process(input);

        CHECK(out.notch == doctest::Approx(input - out.bandpass).epsilon(1e-5));
        CHECK(sogi.notch() == doctest::Approx(out.notch).epsilon(1e-6));
        CHECK(sogi.bandpass() == doctest::Approx(out.bandpass).epsilon(1e-6));
        CHECK(sogi.quadrature() == doctest::Approx(out.quadrature).epsilon(1e-6));

        // Legacy pair-return API remains valid.
        const auto [bp_legacy, q_legacy] = sogi(input);
        CHECK(bp_legacy == doctest::Approx(sogi.bandpass()).epsilon(1e-6));
        CHECK(q_legacy == doctest::Approx(sogi.quadrature()).epsilon(1e-6));
    }
}