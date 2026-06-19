#pragma once

// Not self-contained. Bound by math_backend.hpp after MathBackend<T> is
// declared. Fast single-precision float math (trig.hpp) over the std:: fallback.
#include "math_backend.hpp"
#include "std_fallback.hpp"
#include "trig.hpp"

namespace wet {

template<>
struct MathBackend<float> : StdMathFallback<float> {
    static float sin(float x) { return wet::sin(x); }
    static float cos(float x) { return wet::cos(x); }
    static float asin(float x) { return wet::asin(x); }
    static float acos(float x) { return wet::acos(x); }
    static float atan(float x) { return wet::atan(x); }
    static float atan2(float y, float x) { return wet::atan2(y, x); }

    static wet::pair<float, float> sincos(float x) { return wet::sincos(x); }

    static float sqrt(float x) { return wet::sqrt(x); }
};

template<>
struct MathBackend<double> : StdMathFallback<double> {};

} // namespace wet
