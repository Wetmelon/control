#pragma once

// Not self-contained. Bound by math_backend.hpp after MathBackend<T> is
// declared. Provides the std:: (<cmath>) backend — the hosted default.
#include "math_backend.hpp"
#include "std_fallback.hpp"

namespace wet {

template<>
struct MathBackend<float> : StdMathFallback<float> {};

template<>
struct MathBackend<double> : StdMathFallback<double> {};

} // namespace wet
