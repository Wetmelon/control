// Exercises the WET_MATH_BACKEND_WET profile in isolation. The macro is set
// here rather than in tests/wet_profile.hpp on purpose: the wet MathBackend<float>
// specialization is ODR-incompatible with the default std backend the rest of the
// suite links against, so it gets its own executable. This is the only place
// wet_backend.hpp (and the trig.hpp fast-float path it forwards to) is compiled
// and run — the guard that catches a future break like the truncated nearbyint.
#define WET_MATH_BACKEND_WET

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <cmath>

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"
#include "wet/math/math.hpp"

static constexpr float kEps = 1e-5f;

TEST_SUITE("wet_backend") {
    TEST_CASE("MathBackend<float> routes to the fast-float trig path") {
        CHECK(wet::MathBackend<float>::sin(1.0f) == doctest::Approx(std::sin(1.0f)).epsilon(kEps));
        CHECK(wet::MathBackend<float>::cos(1.0f) == doctest::Approx(std::cos(1.0f)).epsilon(kEps));
        CHECK(wet::MathBackend<float>::atan(0.7f) == doctest::Approx(std::atan(0.7f)).epsilon(kEps));
        CHECK(wet::MathBackend<float>::atan2(0.7f, -0.3f) == doctest::Approx(std::atan2(0.7f, -0.3f)).epsilon(kEps));
        CHECK(wet::MathBackend<float>::asin(0.4f) == doctest::Approx(std::asin(0.4f)).epsilon(kEps));
        CHECK(wet::MathBackend<float>::acos(0.4f) == doctest::Approx(std::acos(0.4f)).epsilon(kEps));
        CHECK(wet::MathBackend<float>::sqrt(2.0f) == doctest::Approx(std::sqrt(2.0f)).epsilon(kEps));

        auto [s, c] = wet::MathBackend<float>::sincos(0.9f);
        CHECK(s == doctest::Approx(std::sin(0.9f)).epsilon(kEps));
        CHECK(c == doctest::Approx(std::cos(0.9f)).epsilon(kEps));
    }

    TEST_CASE("MathBackend<float>::nearbyint rounds to even") {
        CHECK(wet::MathBackend<float>::nearbyint(2.5f) == 2.0f);
        CHECK(wet::MathBackend<float>::nearbyint(3.5f) == 4.0f);
        CHECK(wet::MathBackend<float>::nearbyint(-1.4f) == -1.0f);
        CHECK(wet::MathBackend<float>::nearbyint(100.0f) == 100.0f);
    }

    TEST_CASE("non-overridden float ops fall through to the std base") {
        CHECK(wet::MathBackend<float>::tan(0.5f) == doctest::Approx(std::tan(0.5f)).epsilon(kEps));
        CHECK(wet::MathBackend<float>::exp(1.0f) == doctest::Approx(std::exp(1.0f)).epsilon(kEps));
        CHECK(wet::MathBackend<float>::log(2.0f) == doctest::Approx(std::log(2.0f)).epsilon(kEps));
    }

    TEST_CASE("double path is the std fallback under the wet profile") {
        CHECK(wet::MathBackend<double>::sin(1.0) == doctest::Approx(std::sin(1.0)));
        // The public dispatcher routes double through that backend at runtime.
        CHECK(wet::sin(1.0) == doctest::Approx(std::sin(1.0)));
        CHECK(wet::wrap(7.0, -3.14159265358979, 3.14159265358979) == doctest::Approx(std::atan2(std::sin(7.0), std::cos(7.0))));
    }
}
