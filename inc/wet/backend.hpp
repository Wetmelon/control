#pragma once

/**
 * @file backend.hpp
 * @brief Standard-library backend: the small set of `std`-replacement types the
 *        embeddable core needs, aliased into `wet`.
 *
 * The core does not hard-depend on the hosted C++ standard library. It uses a
 * handful of vocabulary types — `array`, `optional`, `tuple`/`pair`, a few
 * `<algorithm>`/`<utility>` helpers — through the `wet::` aliases defined here,
 * which resolve to one of two backends selected at compile time:
 *
 *   - **stdlib** (default): `wet::array` → `std::array`, etc. Hosted, and usable
 *     standalone (no third-party dependency).
 *   - **ETL** (`-DWET_BACKEND_ETL`): `wet::array` → `etl::array`, etc., for
 *     freestanding / embedded targets without a hosted standard library. Pairs
 *     with the [Embedded Template Library](https://www.etlcpp.com).
 *
 * We do not invent our own primitives — anything not from the stdlib comes from
 * ETL. `std::initializer_list` is intentionally *not* aliased: it is a core
 * language facility (`<initializer_list>`), available even freestanding.
 *
 * @note Selection is unified through `wet/config.hpp`: a single `wet_profile.hpp`
 *       sets `WET_BACKEND_ETL` (this header) alongside the math-backend macros
 *       (`wet/math/math_backend.hpp`). This header covers only the
 *       container/utility types.
 */

#include <initializer_list> // core language facility; freestanding-safe (see @note above)

#include "wet/config.hpp" // pulls the profile's backend-selection macros

namespace wet::numbers {

// Mathematical constants — our own definitions (no <numbers> dependency), so the
// core stays freestanding under either backend. Mirrors wet::numbers::*_v names
// so usage reads identically. (Resolves #21's "std::numbers replacement" item.)
template<typename T>
inline constexpr T pi_v = static_cast<T>(3.141592653589793238462643383279502884L);
template<typename T>
inline constexpr T e_v = static_cast<T>(2.718281828459045235360287471352662498L);
template<typename T>
inline constexpr T sqrt2_v = static_cast<T>(1.414213562373095048801688724209698079L);
template<typename T>
inline constexpr T sqrt3_v = static_cast<T>(1.732050807568877293527446341505872367L);
template<typename T>
inline constexpr T inv_pi_v = static_cast<T>(0.318309886183790671537767526745028724L);
template<typename T>
inline constexpr T inv_sqrt3_v = static_cast<T>(0.577350269189625764509148780501957456L);
template<typename T>
inline constexpr T ln2_v = static_cast<T>(0.693147180559945309417232121458176568L);
template<typename T>
inline constexpr T log2e_v = static_cast<T>(1.442695040888963407359924681001892137L);

} // namespace wet::numbers

#if defined(WET_BACKEND_ETL)

#include <etl/algorithm.h>
#include <etl/array.h>
#include <etl/optional.h>
#include <etl/tuple.h>
#include <etl/utility.h>

namespace wet {
using etl::array;
using etl::clamp;
using etl::forward;
using etl::get;
using etl::index_sequence;
using etl::make_index_sequence;
using etl::make_tuple;
using etl::max;
using etl::min;
using etl::move;
using etl::nullopt;
using etl::nullopt_t;
using etl::optional;
using etl::pair;
using etl::swap;
using etl::tuple;

// ETL's min/max are binary-only; supply the initializer_list overloads (by
// value) so `wet::min({...})` / `wet::max({...})` work on this backend too,
// matching the std backend's std::min/max(initializer_list).
template<typename T>
[[nodiscard]] constexpr T min(std::initializer_list<T> values) {
    T m = *values.begin();
    for (const T& v : values) {
        if (v < m) {
            m = v;
        }
    }
    return m;
}
template<typename T>
[[nodiscard]] constexpr T max(std::initializer_list<T> values) {
    T m = *values.begin();
    for (const T& v : values) {
        if (m < v) {
            m = v;
        }
    }
    return m;
}
} // namespace wet

#else // stdlib backend (default)

#include <algorithm>
#include <array>
#include <optional>
#include <tuple>
#include <utility>

namespace wet {
using std::array;
using std::clamp;
using std::forward;
using std::get;
using std::index_sequence;
using std::make_index_sequence;
using std::make_tuple;
using std::max;
using std::min;
using std::move;
using std::nullopt;
using std::nullopt_t;
using std::optional;
using std::pair;
using std::swap;
using std::tuple;
} // namespace wet

#endif

namespace wet {

/**
 * @brief Ordered {min, max} pair returned by value.
 *
 * Unlike std::minmax (which returns a pair of *references* to its arguments and
 * dangles on temporaries, and lives in the non-freestanding <algorithm>), this
 * returns by value and is backend-agnostic. Ties return {a, b}.
 */
template<typename T>
[[nodiscard]] constexpr pair<T, T> minmax(const T& a, const T& b) {
    return b < a ? pair<T, T>{b, a} : pair<T, T>{a, b};
}

/**
 * @brief Ordered {min, max} of an initializer list, returned by value.
 *
 * The by-value, backend-agnostic counterpart to `std::minmax(initializer_list)`
 * (which `etl::minmax` lacks entirely). Matches the standard's tie behaviour:
 * leftmost minimum, rightmost maximum. The list must be non-empty (UB otherwise,
 * as in the standard).
 */
template<typename T>
[[nodiscard]] constexpr pair<T, T> minmax(std::initializer_list<T> values) {
    T lo = *values.begin();
    T hi = *values.begin();
    for (const T& v : values) {
        if (v < lo) {
            lo = v;
        } // leftmost minimum
        if (!(v < hi)) {
            hi = v;
        } // rightmost maximum
    }
    return {lo, hi};
}

} // namespace wet
