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
 * @note The constexpr math backend (`wet::sin/sqrt/...`) is selected separately
 *       in `wet/math/`; this header covers only the container/utility types.
 */

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
