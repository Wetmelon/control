#pragma once

// Not self-contained. Must be included after StdMathFallback<T> and MathBackend<T>
// are declared by math_backend.hpp. Include via wet_profile.hpp.
#include "math_backend.hpp"
#include "wet_trig.hpp"

namespace wet {

template<>
struct MathBackend<float> : StdMathFallback<float> {
    static float sin(float x) { return wet::sin(x); }
    static float cos(float x) { return wet::cos(x); }
    static float asin(float x) { return wet::asin(x); }
    static float acos(float x) { return wet::acos(x); }
    static float atan(float x) { return wet::atan(x); }
    static float atan2(float y, float x) { return wet::atan2(y, x); }

    static std::pair<float, float> sincos(float x) {
        const auto sc = wet::sincos(x);
        return {sc.sin, sc.cos};
    }
    static float sqrt(float x) { return wet::sqrt(x); }
};

template<>
struct MathBackend<double> : StdMathFallback<double> {};

} // namespace wet
