#pragma once

/**
 * @file config.hpp
 * @brief Single configuration entry point for the wet library.
 *
 * The library reads all of its compile-time configuration from one place. A
 * user-authored `wet_profile.hpp`, discovered on the include path, sets
 * per-facility selection macros; each facility (containers in backend.hpp,
 * scalar math in math/math_backend.hpp) keys off those macros.
 *
 * `wet_profile.hpp` must define **macros only** — it must not include a backend
 * implementation. That keeps this header safe to read both early (backend.hpp,
 * before any wet types exist) and late (math_backend.hpp, after its types are
 * declared) without include-ordering hazards.
 *
 * Recognized macros (all optional — omit for the host defaults):
 *
 *   Containers (see backend.hpp):
 *     WET_BACKEND_ETL               wet::array/optional/... → ETL (else stdlib)
 *
 *   Scalar math (see math/math_backend.hpp):
 *     WET_MATH_BACKEND_WET          runtime float math → wet/math/trig.hpp
 *     WET_MATH_BACKEND_HEADER "h"   include "h", which defines MathBackend<T>
 *     (neither set → the std:: math backend)
 *
 * If no `wet_profile.hpp` is found, the library uses its host defaults — stdlib
 * containers and the std:: math backend — and emits a one-time warning so the
 * choice is never silent.
 *
 * @code
 * // wet_profile.hpp  (user-created, anywhere on the include path) — macros only
 * #define WET_BACKEND_ETL                              // ETL containers
 * #define WET_MATH_BACKEND_HEADER "my_cmsis_math.hpp"  // custom math backend
 * @endcode
 */

#if __has_include("wet_profile.hpp")
#include "wet_profile.hpp" // IWYU pragma: keep
#else
#warning "wet_profile.hpp not found in include path. Using host defaults (stdlib containers + std:: math). Create a wet_profile.hpp to select platform backends."
#endif
