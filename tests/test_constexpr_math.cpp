#include <cmath>
#include <complex>

#include "constexpr_complex.hpp"
#include "constexpr_math.hpp"
#include "doctest.h"

TEST_SUITE("constexpr_math") {
    TEST_CASE("wet::sqrt matches std::sqrt") {
        // Test positive values
        CHECK(wet::sqrt(0.0) == doctest::Approx(std::sqrt(0.0)));
        CHECK(wet::sqrt(1.0) == doctest::Approx(std::sqrt(1.0)));
        CHECK(wet::sqrt(2.0) == doctest::Approx(std::sqrt(2.0)));
        CHECK(wet::sqrt(4.0) == doctest::Approx(std::sqrt(4.0)));
        CHECK(wet::sqrt(9.0) == doctest::Approx(std::sqrt(9.0)));
        CHECK(wet::sqrt(100.0) == doctest::Approx(std::sqrt(100.0)));
        CHECK(wet::sqrt(0.25) == doctest::Approx(std::sqrt(0.25)));
        CHECK(wet::sqrt(1e-10) == doctest::Approx(std::sqrt(1e-10)));
        CHECK(wet::sqrt(1e10) == doctest::Approx(std::sqrt(1e10)));

        // Test float
        CHECK(wet::sqrt(2.0f) == doctest::Approx(std::sqrt(2.0)));
        CHECK(wet::sqrt(0.5f) == doctest::Approx(std::sqrt(0.5)));
    }

    TEST_CASE("wet::abs matches std::abs") {
        CHECK(wet::abs(0.0) == std::abs(0.0));
        CHECK(wet::abs(1.0) == std::abs(1.0));
        CHECK(wet::abs(-1.0) == std::abs(-1.0));
        CHECK(wet::abs(3.14159) == std::abs(3.14159));
        CHECK(wet::abs(-3.14159) == std::abs(-3.14159));
        CHECK(wet::abs(1e-15) == std::abs(1e-15));
        CHECK(wet::abs(-1e-15) == std::abs(-1e-15));
        CHECK(wet::abs(1e15) == std::abs(1e15));
        CHECK(wet::abs(-1e15) == std::abs(-1e15));
    }

    TEST_CASE("wet::cbrt matches std::cbrt") {
        // Positive values
        CHECK(wet::cbrt(0.0) == doctest::Approx(std::cbrt(0.0)));
        CHECK(wet::cbrt(1.0) == doctest::Approx(std::cbrt(1.0)));
        CHECK(wet::cbrt(8.0) == doctest::Approx(std::cbrt(8.0)));
        CHECK(wet::cbrt(27.0) == doctest::Approx(std::cbrt(27.0)));
        CHECK(wet::cbrt(64.0) == doctest::Approx(std::cbrt(64.0)));
        CHECK(wet::cbrt(1000.0) == doctest::Approx(std::cbrt(1000.0)));

        // Negative values
        CHECK(wet::cbrt(-1.0) == doctest::Approx(std::cbrt(-1.0)));
        CHECK(wet::cbrt(-8.0) == doctest::Approx(std::cbrt(-8.0)));
        CHECK(wet::cbrt(-27.0) == doctest::Approx(std::cbrt(-27.0)));

        // Non-perfect cubes
        CHECK(wet::cbrt(2.0) == doctest::Approx(std::cbrt(2.0)));
        CHECK(wet::cbrt(10.0) == doctest::Approx(std::cbrt(10.0)));
        CHECK(wet::cbrt(-10.0) == doctest::Approx(std::cbrt(-10.0)));

        // Small and large values
        CHECK(wet::cbrt(1e-9) == doctest::Approx(std::cbrt(1e-9)));
        CHECK(wet::cbrt(1e9) == doctest::Approx(std::cbrt(1e9)));
    }

    TEST_CASE("wet::sqrt matches std::sqrt for complex") {
        using Complex = wet::complex<double>;
        using StdComplex = std::complex<double>;

        // Real positive
        CHECK(wet::sqrt(Complex(4.0, 0.0)).real() == doctest::Approx(std::sqrt(StdComplex(4.0, 0.0)).real()));
        CHECK(wet::sqrt(Complex(4.0, 0.0)).imag() == doctest::Approx(std::sqrt(StdComplex(4.0, 0.0)).imag()));

        // Real negative (pure imaginary result)
        CHECK(wet::sqrt(Complex(-4.0, 0.0)).real() == doctest::Approx(std::sqrt(StdComplex(-4.0, 0.0)).real()));
        CHECK(wet::sqrt(Complex(-4.0, 0.0)).imag() == doctest::Approx(std::sqrt(StdComplex(-4.0, 0.0)).imag()));

        // Pure imaginary
        CHECK(wet::sqrt(Complex(0.0, 4.0)).real() == doctest::Approx(std::sqrt(StdComplex(0.0, 4.0)).real()));
        CHECK(wet::sqrt(Complex(0.0, 4.0)).imag() == doctest::Approx(std::sqrt(StdComplex(0.0, 4.0)).imag()));

        // General complex
        CHECK(wet::sqrt(Complex(3.0, 4.0)).real() == doctest::Approx(std::sqrt(StdComplex(3.0, 4.0)).real()));
        CHECK(wet::sqrt(Complex(3.0, 4.0)).imag() == doctest::Approx(std::sqrt(StdComplex(3.0, 4.0)).imag()));

        // Negative imaginary
        CHECK(wet::sqrt(Complex(3.0, -4.0)).real() == doctest::Approx(std::sqrt(StdComplex(3.0, -4.0)).real()));
        CHECK(wet::sqrt(Complex(3.0, -4.0)).imag() == doctest::Approx(std::sqrt(StdComplex(3.0, -4.0)).imag()));

        // Zero
        CHECK(wet::sqrt(Complex(0.0, 0.0)).real() == doctest::Approx(0.0));
        CHECK(wet::sqrt(Complex(0.0, 0.0)).imag() == doctest::Approx(0.0));
    }

    TEST_CASE("wet::atan2 matches std::atan2") {
        // Standard quadrants
        CHECK(wet::atan2(0.0, 1.0) == doctest::Approx(std::atan2(0.0, 1.0)));     // 0
        CHECK(wet::atan2(1.0, 1.0) == doctest::Approx(std::atan2(1.0, 1.0)));     // π/4
        CHECK(wet::atan2(1.0, 0.0) == doctest::Approx(std::atan2(1.0, 0.0)));     // π/2
        CHECK(wet::atan2(1.0, -1.0) == doctest::Approx(std::atan2(1.0, -1.0)));   // 3π/4
        CHECK(wet::atan2(0.0, -1.0) == doctest::Approx(std::atan2(0.0, -1.0)));   // π
        CHECK(wet::atan2(-1.0, -1.0) == doctest::Approx(std::atan2(-1.0, -1.0))); // -3π/4
        CHECK(wet::atan2(-1.0, 0.0) == doctest::Approx(std::atan2(-1.0, 0.0)));   // -π/2
        CHECK(wet::atan2(-1.0, 1.0) == doctest::Approx(std::atan2(-1.0, 1.0)));   // -π/4

        // Various ratios
        CHECK(wet::atan2(3.0, 4.0) == doctest::Approx(std::atan2(3.0, 4.0)));
        CHECK(wet::atan2(4.0, 3.0) == doctest::Approx(std::atan2(4.0, 3.0)));
        CHECK(wet::atan2(0.5, 0.1) == doctest::Approx(std::atan2(0.5, 0.1)));
        CHECK(wet::atan2(0.1, 0.5) == doctest::Approx(std::atan2(0.1, 0.5)));
    }

    TEST_CASE("wet::cos matches std::cos") {
        constexpr double pi = 3.14159265358979323846;

        // Standard angles
        CHECK(wet::cos(0.0) == doctest::Approx(std::cos(0.0)));
        CHECK(wet::cos(pi / 6) == doctest::Approx(std::cos(pi / 6)));                        // 30°
        CHECK(wet::cos(pi / 4) == doctest::Approx(std::cos(pi / 4)));                        // 45°
        CHECK(wet::cos(pi / 3) == doctest::Approx(std::cos(pi / 3)));                        // 60°
        CHECK(wet::cos(pi / 2) == doctest::Approx(std::cos(pi / 2)).epsilon(1e-10));         // 90°
        CHECK(wet::cos(pi) == doctest::Approx(std::cos(pi)));                                // 180°
        CHECK(wet::cos(3 * pi / 2) == doctest::Approx(std::cos(3 * pi / 2)).epsilon(1e-10)); // 270°

        // Negative angles
        CHECK(wet::cos(-pi / 4) == doctest::Approx(std::cos(-pi / 4)));
        CHECK(wet::cos(-pi / 2) == doctest::Approx(std::cos(-pi / 2)).epsilon(1e-10));
        CHECK(wet::cos(-pi) == doctest::Approx(std::cos(-pi)));

        // Arbitrary angles
        CHECK(wet::cos(0.5) == doctest::Approx(std::cos(0.5)));
        CHECK(wet::cos(1.0) == doctest::Approx(std::cos(1.0)));
        CHECK(wet::cos(2.0) == doctest::Approx(std::cos(2.0)));
        CHECK(wet::cos(3.0) == doctest::Approx(std::cos(3.0)));
    }

    TEST_CASE("wet::sin matches std::sin") {
        constexpr double pi = 3.14159265358979323846;

        // Standard angles
        CHECK(wet::sin(0.0) == doctest::Approx(std::sin(0.0)));
        CHECK(wet::sin(pi / 6) == doctest::Approx(std::sin(pi / 6)));         // 30°
        CHECK(wet::sin(pi / 4) == doctest::Approx(std::sin(pi / 4)));         // 45°
        CHECK(wet::sin(pi / 3) == doctest::Approx(std::sin(pi / 3)));         // 60°
        CHECK(wet::sin(pi / 2) == doctest::Approx(std::sin(pi / 2)));         // 90°
        CHECK(wet::sin(pi) == doctest::Approx(std::sin(pi)).epsilon(1e-10));  // 180°
        CHECK(wet::sin(3 * pi / 2) == doctest::Approx(std::sin(3 * pi / 2))); // 270°

        // Negative angles
        CHECK(wet::sin(-pi / 4) == doctest::Approx(std::sin(-pi / 4)));
        CHECK(wet::sin(-pi / 2) == doctest::Approx(std::sin(-pi / 2)));
        CHECK(wet::sin(-pi) == doctest::Approx(std::sin(-pi)).epsilon(1e-10));

        // Arbitrary angles
        CHECK(wet::sin(0.5) == doctest::Approx(std::sin(0.5)));
        CHECK(wet::sin(1.0) == doctest::Approx(std::sin(1.0)));
        CHECK(wet::sin(2.0) == doctest::Approx(std::sin(2.0)));
        CHECK(wet::sin(3.0) == doctest::Approx(std::sin(3.0)));
    }

    TEST_CASE("constexpr verification") {
        // Verify all functions can be used in constexpr context
        constexpr double sqrt_val = wet::sqrt(4.0);
        constexpr double abs_val = wet::abs(-5.0);
        constexpr double cbrt_val = wet::cbrt(8.0);
        constexpr double atan2_val = wet::atan2(1.0, 1.0);
        constexpr double cos_val = wet::cos(0.0);
        constexpr double sin_val = wet::sin(0.0);
        constexpr auto   csqrt_val = wet::sqrt(wet::complex<double>(4.0, 0.0));

        CHECK(sqrt_val == doctest::Approx(2.0));
        CHECK(abs_val == doctest::Approx(5.0));
        CHECK(cbrt_val == doctest::Approx(2.0));
        CHECK(atan2_val == doctest::Approx(std::atan2(1.0, 1.0)));
        CHECK(cos_val == doctest::Approx(1.0));
        CHECK(sin_val == doctest::Approx(0.0));
        CHECK(csqrt_val.real() == doctest::Approx(2.0));
        CHECK(csqrt_val.imag() == doctest::Approx(0.0));
    }
}
