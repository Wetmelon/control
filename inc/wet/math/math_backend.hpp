#pragma once

/**
 * @file math_backend.hpp
 * @brief Pluggable runtime math backend selection (freestanding-safe).
 *
 * Declares the MathBackend<T> customization point and binds the chosen backend
 * implementation from the profile macros (see config.hpp). This header is
 * intentionally free of <cmath> so a freestanding build never pulls it — the
 * std:: implementation lives in std_fallback.hpp / std_backend.hpp.
 */

#include "wet/config.hpp" // pulls the profile's backend-selection macros

namespace wet {

/**
 * @brief Pluggable math backend for runtime scalar operations
 *
 * Primary template is intentionally undefined — a backend specialization must be
 * provided for each scalar type used at runtime. Selection is driven by the
 * profile macros read through wet/config.hpp (set in your wet_profile.hpp):
 *
 *   - default                       → std_backend.hpp (the std:: backend)
 *   - WET_MATH_BACKEND_WET          → wet_backend.hpp (fast float math/trig.hpp)
 *   - WET_MATH_BACKEND_FREESTANDING → series_backend.hpp (constexpr series, no <cmath>)
 *   - WET_MATH_BACKEND_HEADER "h"   → include "h", which defines MathBackend<T>
 *
 * Backend authors: inherit from StdMathFallback<T> (std_fallback.hpp) and
 * override only the functions your platform library provides; the rest fall
 * through to <cmath>. A freestanding backend instead routes to the constexpr
 * series (see series_backend.hpp) and pulls no hosted headers.
 *
 * @see std_fallback.hpp for StdMathFallback
 * @see config.hpp for the unified profile / macro surface
 * @tparam T Scalar type (float, double)
 */
template<typename T>
struct MathBackend;

} // namespace wet

// Bind the runtime backend selected by the profile macros (see config.hpp),
// after the MathBackend primary above is declared. Default: std::.
#if defined(WET_MATH_BACKEND_HEADER)
#include WET_MATH_BACKEND_HEADER // IWYU pragma: keep
#elif defined(WET_MATH_BACKEND_FREESTANDING)
#include "series_backend.hpp" // IWYU pragma: keep
#elif defined(WET_MATH_BACKEND_WET)
#include "wet_backend.hpp" // IWYU pragma: keep
#else
#include "std_backend.hpp" // IWYU pragma: keep
#endif
