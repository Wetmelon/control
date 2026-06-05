#pragma once

#include <cmath>
#include <utility>

namespace wet {

/**
 * @brief Composable std:: base for math backends
 *
 * Backends inherit from this struct and override only the functions their
 * platform library provides. Any function not overridden falls through to
 * the corresponding <cmath> implementation.
 *
 * @tparam T Scalar type (float, double)
 */
template<typename T>
struct StdMathFallback {
    static T sin(T x) { return std::sin(x); }
    static T cos(T x) { return std::cos(x); }
    static T tan(T x) { return std::tan(x); }
    static T asin(T x) { return std::asin(x); }
    static T acos(T x) { return std::acos(x); }
    static T atan(T x) { return std::atan(x); }
    static T atan2(T y, T x) { return std::atan2(y, x); }

    /// Combined sin/cos. Returns {sin(x), cos(x)}. Platform backends should
    /// override this with a shared-range-reduction implementation.
    static std::pair<T, T> sincos(T x) { return {std::sin(x), std::cos(x)}; }
    static T               sqrt(T x) { return std::sqrt(x); }
    static T               abs(T x) { return std::abs(x); }
    static T               cbrt(T x) { return std::cbrt(x); }
    static T               exp(T x) { return std::exp(x); }
    static T               log(T x) { return std::log(x); }
    static T               log10(T x) { return std::log10(x); }
    static T               floor(T x) { return std::floor(x); }
    static T               ceil(T x) { return std::ceil(x); }

    static T    pow(T base, T exponent) { return std::pow(base, exponent); }
    static T    fmod(T x, T y) { return std::fmod(x, y); }
    static T    copysign(T mag, T sgn) { return std::copysign(mag, sgn); }
    static bool isfinite(T x) { return std::isfinite(x); }
};

/**
 * @brief Pluggable math backend for runtime scalar operations
 *
 * Primary template is intentionally undefined — a backend specialization must
 * be provided for each scalar type used at runtime. The recommended approach
 * is to create a wet_profile.hpp anywhere in your project's include path and
 * include your chosen backend there:
 *
 * @code
 * // wet_profile.hpp  (user-created, anywhere in include path)
 * #include "ti_arm_backend.hpp"            // or std_backend.hpp, arm_cmsis_dsp.hpp, etc.
 * @endcode
 *
 * If wet_profile.hpp is not found, std_backend.hpp is used automatically with
 * a compiler warning. To silence the warning on host/test builds, create a
 * wet_profile.hpp that explicitly includes std_backend.hpp.
 *
 * Backend authors: inherit from StdMathFallback<T> and override only the
 * functions your platform library provides. The rest fall through to <cmath>.
 *
 * @see StdMathFallback
 * @tparam T Scalar type (float, double)
 */
template<typename T>
struct MathBackend;

} // namespace wet

// Auto-discover user backend profile. If not found, fall back to std:: with a warning.
#if __has_include("wet_profile.hpp")
#include "wet_profile.hpp"
#else
#warning "wet_profile.hpp not found in include path. Using std:: math backend. Create wet_profile.hpp to select a platform backend."
#include "std_backend.hpp"
#endif
