#pragma once

#include "math_backend.hpp"
#include "ti_arm_trig.hpp"

// Not self-contained. Must be included after StdMathFallback<T> and MathBackend<T>
// are declared by math_backend.hpp. Include via wet_profile.hpp only.

namespace wetmelon::control {

/**
 * @brief TI ARM math backend for float
 *
 * Routes float-precision trig and elementary functions to TI's optimized
 * ARM implementations. Functions not covered by the TI library (cbrt, exp,
 * log, log10, floor, ceil, pow) fall through to StdMathFallback<float>.
 *
 * Usage: include from wet_profile.hpp in your project's include path:
 * @code
 * // wet_profile.hpp
 * #include "ti_arm_backend.hpp"
 * @endcode
 *
 * @see StdMathFallback for inherited std:: fallbacks
 * @see ti_arm_trig.hpp for the underlying TI implementations
 */
template<>
struct MathBackend<float> : StdMathFallback<float> {
    static float sin(float x) { return ti_arm::sin(x); }
    static float cos(float x) { return ti_arm::cos(x); }
    static float tan(float x) { return ti_arm::sin(x) / ti_arm::cos(x); }
    static float atan2(float y, float x) { return ti_arm::atan2(y, x); }
    static float sqrt(float x) { return ti_arm::sqrt(x); }
    static float abs(float x) { return ti_arm::abs(x); }
    // cbrt, exp, log, log10, floor, ceil, pow inherited from StdMathFallback<float>
};

} // namespace wetmelon::control
