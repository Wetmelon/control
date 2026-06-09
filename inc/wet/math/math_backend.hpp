#pragma once

#include <cmath>

#include "wet/backend.hpp"
#include "wet/config.hpp" // pulls the profile's backend-selection macros

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
    static wet::pair<T, T> sincos(T x) { return {std::sin(x), std::cos(x)}; }

    static T sqrt(T x) { return std::sqrt(x); }
    static T abs(T x) { return std::abs(x); }
    static T cbrt(T x) { return std::cbrt(x); }
    static T exp(T x) { return std::exp(x); }
    static T log(T x) { return std::log(x); }
    static T log10(T x) { return std::log10(x); }
    static T floor(T x) { return std::floor(x); }
    static T ceil(T x) { return std::ceil(x); }

    static T    pow(T base, T exponent) { return std::pow(base, exponent); }
    static T    fmod(T x, T y) { return std::fmod(x, y); }
    static T    copysign(T mag, T sgn) { return std::copysign(mag, sgn); }
    static bool isfinite(T x) { return std::isfinite(x); }
};

/**
 * @brief Pluggable math backend for runtime scalar operations
 *
 * Primary template is intentionally undefined — a backend specialization must
 * be provided for each scalar type used at runtime. The selection is driven by
 * the profile macros read through wet/config.hpp (set in your wet_profile.hpp):
 *
 *   - default                       → std_backend.hpp (the std:: backend)
 *   - WET_MATH_BACKEND_WET          → wet_backend.hpp (fast float math/trig.hpp)
 *   - WET_MATH_BACKEND_HEADER "h"   → include "h", which defines MathBackend<T>
 *
 * Backend authors: inherit from StdMathFallback<T> and override only the
 * functions your platform library provides. The rest fall through to <cmath>.
 *
 * @see StdMathFallback
 * @see config.hpp for the unified profile / macro surface
 * @tparam T Scalar type (float, double)
 */
template<typename T>
struct MathBackend;

} // namespace wet

// Bind the runtime backend selected by the profile macros (see config.hpp). The
// chosen header is included here, after the types above are declared, so that
// its MathBackend<T> specializations see StdMathFallback<T>. Default: std::.
#if defined(WET_MATH_BACKEND_HEADER)
#include WET_MATH_BACKEND_HEADER // IWYU pragma: keep
#elif defined(WET_MATH_BACKEND_WET)
#include "wet_backend.hpp" // IWYU pragma: keep
#else
#include "std_backend.hpp" // IWYU pragma: keep
#endif
