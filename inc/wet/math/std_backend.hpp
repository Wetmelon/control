#pragma once

// Not self-contained. Must be included after StdMathFallback<T> and MathBackend<T>
// are declared by math_backend.hpp. Include via wet_profile.hpp or the automatic
// fallback in math_backend.hpp — do not include directly.
#include "math_backend.hpp"

namespace wet {

template<>
struct MathBackend<float> : StdMathFallback<float> {};

template<>
struct MathBackend<double> : StdMathFallback<double> {};

} // namespace wet
