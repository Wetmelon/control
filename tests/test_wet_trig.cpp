#include <array>
#include <cmath>
#include <initializer_list>
#include <numbers>

#include "wet/math/wet_trig.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

// ~8 ULP accuracy on float32; epsilon(1e-5) gives comfortable margin
static constexpr float kEps = 1e-5f;

TEST_SUITE("wet_trig") {

    TEST_CASE("sin - known values") {
        using std::numbers::pi_v;
        CHECK(wet::sin(0.0f) == doctest::Approx(std::sin(0.0f)).epsilon(kEps));
        CHECK(wet::sin(pi_v<float> / 6.0f) == doctest::Approx(0.5f).epsilon(kEps));
        CHECK(wet::sin(pi_v<float> / 4.0f) == doctest::Approx(std::sin(pi_v<float> / 4.0f)).epsilon(kEps));
        CHECK(wet::sin(pi_v<float> / 3.0f) == doctest::Approx(std::sin(pi_v<float> / 3.0f)).epsilon(kEps));
        CHECK(wet::sin(pi_v<float> / 2.0f) == doctest::Approx(1.0f).epsilon(kEps));
        CHECK(wet::sin(pi_v<float>) == doctest::Approx(std::sin(pi_v<float>)).epsilon(kEps));
        CHECK(wet::sin(3.0f * pi_v<float> / 2.0f) == doctest::Approx(-1.0f).epsilon(kEps));
        CHECK(wet::sin(2.0f * pi_v<float>) == doctest::Approx(std::sin(2.0f * pi_v<float>)).epsilon(kEps));
        CHECK(wet::sin(-pi_v<float> / 2.0f) == doctest::Approx(-1.0f).epsilon(kEps));
    }

    TEST_CASE("sin - odd symmetry") {
        for (float x : {0.1f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f}) {
            CHECK(wet::sin(-x) == doctest::Approx(-wet::sin(x)).epsilon(kEps));
        }
    }

    TEST_CASE("sin - matches std::sin") {
        for (float x : {0.1f, 0.3f, 0.7f, 1.0f, 1.3f, 1.9f, 2.4f, 3.0f, -0.5f, -1.2f, -2.7f}) {
            CHECK(wet::sin(x) == doctest::Approx(std::sin(x)).epsilon(kEps));
        }
    }

    TEST_CASE("sin - large arguments") {
        CHECK(wet::sin(10.0f * std::numbers::pi_v<float>) == doctest::Approx(std::sin(10.0f * std::numbers::pi_v<float>)).epsilon(kEps));
        CHECK(wet::sin(100.0f) == doctest::Approx(std::sin(100.0f)).epsilon(kEps));
        CHECK(wet::sin(-100.0f) == doctest::Approx(std::sin(-100.0f)).epsilon(kEps));
    }

    // -------------------------------------------------------------------------

    TEST_CASE("cos - known values") {
        using std::numbers::pi_v;
        CHECK(wet::cos(0.0f) == doctest::Approx(1.0f).epsilon(kEps));
        CHECK(wet::cos(pi_v<float> / 3.0f) == doctest::Approx(0.5f).epsilon(kEps));
        CHECK(wet::cos(pi_v<float> / 4.0f) == doctest::Approx(std::cos(pi_v<float> / 4.0f)).epsilon(kEps));
        CHECK(wet::cos(pi_v<float> / 2.0f) == doctest::Approx(std::cos(pi_v<float> / 2.0f)).epsilon(kEps));
        CHECK(wet::cos(pi_v<float>) == doctest::Approx(-1.0f).epsilon(kEps));
        CHECK(wet::cos(2.0f * pi_v<float>) == doctest::Approx(1.0f).epsilon(kEps));
    }

    TEST_CASE("cos - even symmetry") {
        for (float x : {0.1f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f}) {
            CHECK(wet::cos(-x) == doctest::Approx(wet::cos(x)).epsilon(kEps));
        }
    }

    TEST_CASE("cos - matches std::cos") {
        for (float x : {0.1f, 0.3f, 0.7f, 1.0f, 1.3f, 1.9f, 2.4f, 3.0f, -0.5f, -1.2f, -2.7f}) {
            CHECK(wet::cos(x) == doctest::Approx(std::cos(x)).epsilon(kEps));
        }
    }

    // -------------------------------------------------------------------------

    TEST_CASE("sincos - matches individual sin and cos") {
        for (float x : {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, -0.7f, -2.3f}) {
            auto [s, c] = wet::sincos(x);
            CHECK(s == doctest::Approx(wet::sin(x)).epsilon(kEps));
            CHECK(c == doctest::Approx(wet::cos(x)).epsilon(kEps));
        }
    }

    TEST_CASE("sincos - Pythagorean identity") {
        for (float x : {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, -1.0f, -2.5f}) {
            auto [s, c] = wet::sincos(x);
            CHECK((s * s) + (c * c) == doctest::Approx(1.0f).epsilon(kEps));
        }
    }

    // -------------------------------------------------------------------------

    TEST_CASE("asin - known values") {
        using std::numbers::pi_v;
        CHECK(wet::asin(0.0f) == doctest::Approx(0.0f).epsilon(kEps));
        CHECK(wet::asin(0.5f) == doctest::Approx(pi_v<float> / 6.0f).epsilon(kEps));
        CHECK(wet::asin(1.0f) == doctest::Approx(pi_v<float> / 2.0f).epsilon(kEps));
        CHECK(wet::asin(-0.5f) == doctest::Approx(-pi_v<float> / 6.0f).epsilon(kEps));
        CHECK(wet::asin(-1.0f) == doctest::Approx(-pi_v<float> / 2.0f).epsilon(kEps));
    }

    TEST_CASE("asin - clamped at boundaries") {
        using std::numbers::pi_v;
        CHECK(wet::asin(1.5f) == doctest::Approx(pi_v<float> / 2.0f).epsilon(kEps));
        CHECK(wet::asin(-1.5f) == doctest::Approx(-pi_v<float> / 2.0f).epsilon(kEps));
    }

    TEST_CASE("asin - matches std::asin") {
        for (float x : {0.0f, 0.1f, 0.3f, 0.5f, 0.7f, 0.9f, -0.2f, -0.6f, -0.95f}) {
            CHECK(wet::asin(x) == doctest::Approx(std::asin(x)).epsilon(kEps));
        }
    }

    // -------------------------------------------------------------------------

    TEST_CASE("acos - known values") {
        using std::numbers::pi_v;
        CHECK(wet::acos(1.0f) == doctest::Approx(0.0f).epsilon(kEps));
        CHECK(wet::acos(0.5f) == doctest::Approx(pi_v<float> / 3.0f).epsilon(kEps));
        CHECK(wet::acos(0.0f) == doctest::Approx(pi_v<float> / 2.0f).epsilon(kEps));
        CHECK(wet::acos(-0.5f) == doctest::Approx(2.0f * pi_v<float> / 3.0f).epsilon(kEps));
        CHECK(wet::acos(-1.0f) == doctest::Approx(pi_v<float>).epsilon(kEps));
    }

    TEST_CASE("acos - clamped at boundaries") {
        using std::numbers::pi_v;
        CHECK(wet::acos(1.5f) == doctest::Approx(0.0f).epsilon(kEps));
        CHECK(wet::acos(-1.5f) == doctest::Approx(pi_v<float>).epsilon(kEps));
    }

    TEST_CASE("acos - matches std::acos") {
        for (float x : {0.0f, 0.1f, 0.3f, 0.5f, 0.7f, 0.9f, -0.2f, -0.6f, -0.95f}) {
            CHECK(wet::acos(x) == doctest::Approx(std::acos(x)).epsilon(kEps));
        }
    }

    TEST_CASE("acos - asin + acos = pi/2") {
        for (float x : {0.0f, 0.2f, 0.5f, 0.8f, 1.0f}) {
            CHECK(wet::asin(x) + wet::acos(x) == doctest::Approx(std::numbers::pi_v<float> / 2.0f).epsilon(kEps));
        }
    }

    // -------------------------------------------------------------------------

    TEST_CASE("atan - known values") {
        using std::numbers::pi_v;
        CHECK(wet::atan(0.0f) == doctest::Approx(0.0f).epsilon(kEps));
        CHECK(wet::atan(1.0f) == doctest::Approx(pi_v<float> / 4.0f).epsilon(kEps));
        CHECK(wet::atan(-1.0f) == doctest::Approx(-pi_v<float> / 4.0f).epsilon(kEps));
    }

    TEST_CASE("atan - |x| > 1 complement path") {
        for (float x : {2.0f, 5.0f, 10.0f, 100.0f, -3.0f, -7.0f}) {
            CHECK(wet::atan(x) == doctest::Approx(std::atan(x)).epsilon(kEps));
        }
    }

    TEST_CASE("atan - matches std::atan") {
        for (float x : {0.0f, 0.1f, 0.3f, 0.5f, 0.7f, 0.9f, 1.0f, 1.5f, 2.0f, -0.4f, -1.0f, -2.5f}) {
            CHECK(wet::atan(x) == doctest::Approx(std::atan(x)).epsilon(kEps));
        }
    }

    TEST_CASE("atan - odd symmetry") {
        for (float x : {0.3f, 1.0f, 2.5f, 10.0f}) {
            CHECK(wet::atan(-x) == doctest::Approx(-wet::atan(x)).epsilon(kEps));
        }
    }

    // -------------------------------------------------------------------------

    TEST_CASE("atan2 - quadrant coverage") {
        using std::numbers::pi_v;
        // Q1
        CHECK(wet::atan2(1.0f, 1.0f) == doctest::Approx(pi_v<float> / 4.0f).epsilon(kEps));
        // Positive axes
        CHECK(wet::atan2(1.0f, 0.0f) == doctest::Approx(pi_v<float> / 2.0f).epsilon(kEps));
        CHECK(wet::atan2(0.0f, 1.0f) == doctest::Approx(0.0f).epsilon(kEps));
        // Q2
        CHECK(wet::atan2(1.0f, -1.0f) == doctest::Approx(3.0f * pi_v<float> / 4.0f).epsilon(kEps));
        // Negative x axis
        CHECK(wet::atan2(0.0f, -1.0f) == doctest::Approx(pi_v<float>).epsilon(kEps));
        // Q3 / negative y
        CHECK(wet::atan2(-1.0f, 0.0f) == doctest::Approx(-pi_v<float> / 2.0f).epsilon(kEps));
        CHECK(wet::atan2(-1.0f, -1.0f) == doctest::Approx(-3.0f * pi_v<float> / 4.0f).epsilon(kEps));
        // Q4
        CHECK(wet::atan2(-1.0f, 1.0f) == doctest::Approx(-pi_v<float> / 4.0f).epsilon(kEps));
    }

    TEST_CASE("atan2 - matches std::atan2") {
        const std::array pairs = std::to_array<std::pair<float, float>>({
            {1.0f, 2.0f},
            {-1.0f, 2.0f},
            {1.0f, -2.0f},
            {-1.0f, -2.0f},
            {3.0f, 1.0f},
            {0.5f, 0.5f},
            {0.1f, 10.0f},
            {10.0f, 0.1f},
        });

        for (auto [y, x] : pairs) {
            CHECK(wet::atan2(y, x) == doctest::Approx(std::atan2(y, x)).epsilon(kEps));
        }
    }

    TEST_CASE("atan2 - consistent with atan for x > 0") {
        for (float t : {0.1f, 0.5f, 1.0f, 2.0f, 5.0f}) {
            CHECK(wet::atan2(t, 1.0f) == doctest::Approx(wet::atan(t)).epsilon(kEps));
        }
    }

} // TEST_SUITE("wet_trig")
