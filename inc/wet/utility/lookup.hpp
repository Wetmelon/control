#pragma once

/**
 * @file utility/lookup.hpp
 * @brief Breakpoint lookup tables with interpolation.
 *
 * Fixed-size, `constexpr`, allocation-free interpolation tables for the
 * non-affine "raw → engineering" mappings that a @ref AffineCal can't capture:
 * sensor linearization, gain-scheduling maps, fan / efficiency / torque-speed
 * curves. 1-D (linear or nearest) and 2-D (bilinear) over monotonic breakpoints.
 *
 * Breakpoints and grids are stored as the library's own @ref ColVec / @ref Matrix
 * types rather than raw arrays, so a table composes with the linear-algebra core.
 *
 * @see utility/scaling.hpp for the affine / polynomial conversions.
 */

#include <algorithm>
#include <cstddef>

#include "wet/matrix/matrix.hpp"   // Matrix, ColVec
#include "wet/utility/scaling.hpp" // lerp, inverse_lerp

namespace wet {

/// Out-of-range behaviour for a @ref Lut1D query beyond its breakpoints.
enum class Extrapolation {
    Clamp, //!< Hold the nearest endpoint value.
    Linear //!< Continue the slope of the nearest end segment.
};

/**
 * @brief Index of the interpolation segment containing @p x.
 *
 * For strictly increasing breakpoints @p xs, returns the largest `i` with
 * `xs[i] <= x`, clamped to `[0, N-2]` so `i` and `i+1` always bracket a valid
 * segment. O(log N) binary search.
 *
 * @tparam N Number of breakpoints (must be ≥ 2).
 */
template<size_t N, typename T>
[[nodiscard]] constexpr size_t lut_segment(const ColVec<N, T>& xs, T x) {
    static_assert(N >= 2, "lut_segment needs at least two breakpoints");
    if (x <= xs[1]) {
        return 0;
    }
    if (x >= xs[N - 2]) {
        return N - 2;
    }
    size_t lo = 1;
    size_t hi = N - 2;
    while (hi - lo > 1) {
        const size_t mid = lo + ((hi - lo) / 2);
        if (xs[mid] <= x) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

/**
 * @brief 1-D interpolating lookup table over monotonic breakpoints.
 *
 * Stores `N` strictly-increasing breakpoints @ref xs and their values @ref ys
 * as @ref ColVec. `operator()` does linear interpolation; @ref nearest does
 * nearest-neighbour. Queries outside `[xs[0], xs[N-1]]` follow the @ref oob
 * policy (clamp or linear extrapolation).
 *
 * @code
 * // NTC: resistance (kΩ) -> temperature (°C), falling curve.
 * constexpr Lut1D<3, float> ntc{{32.6f, 10.0f, 3.6f}, {0.0f, 25.0f, 50.0f}};
 * float t = ntc(measured_kohm);
 * @endcode
 *
 * @tparam N Number of breakpoints (must be ≥ 1).
 * @tparam T Scalar type (default: float)
 */
template<size_t N, typename T = float>
struct Lut1D {
    static_assert(N >= 1, "Lut1D needs at least one breakpoint");

    ColVec<N, T>  xs{};                      //!< Strictly increasing breakpoints.
    ColVec<N, T>  ys{};                      //!< Value at each breakpoint.
    Extrapolation oob{Extrapolation::Clamp}; //!< Out-of-range policy.

    /// Linearly interpolated value at @p x (extrapolation per @ref oob).
    [[nodiscard]] constexpr T operator()(T x) const {
        if constexpr (N == 1) {
            return ys[0];
        } else {
            if (x <= xs[0] && oob == Extrapolation::Clamp) {
                return ys[0];
            }
            if (x >= xs[N - 1] && oob == Extrapolation::Clamp) {
                return ys[N - 1];
            }
            const size_t i = lut_segment(xs, x);
            return lerp(ys[i], ys[i + 1], inverse_lerp(xs[i], xs[i + 1], x));
        }
    }

    /// Nearest-neighbour value at @p x (always clamps to the breakpoint range).
    [[nodiscard]] constexpr T nearest(T x) const {
        if constexpr (N == 1) {
            return ys[0];
        } else {
            if (x <= xs[0]) {
                return ys[0];
            }
            if (x >= xs[N - 1]) {
                return ys[N - 1];
            }
            const size_t i = lut_segment(xs, x);
            return (x - xs[i] <= xs[i + 1] - x) ? ys[i] : ys[i + 1];
        }
    }
};

/**
 * @brief 2-D bilinear interpolating lookup table over a regular grid.
 *
 * Row breakpoints @ref rows (length `R`) and column breakpoints @ref cols
 * (length `C`) index a grid of values @ref z (`z(r, c)`), stored as a
 * @ref Matrix. `operator()(r, c)` bilinearly interpolates; queries outside the
 * grid are clamped to the edge.
 *
 * @code
 * // Efficiency map vs (speed, torque):
 * Lut2D<2, 2, float> eff{{0.0f, 1.0f}, {0.0f, 1.0f},
 *                        {{0.80f, 0.85f}, {0.88f, 0.92f}}};
 * float e = eff(speed, torque);
 * @endcode
 *
 * @tparam R Number of row breakpoints (must be ≥ 1).
 * @tparam C Number of column breakpoints (must be ≥ 1).
 * @tparam T Scalar type (default: float)
 */
template<size_t R, size_t C, typename T = float>
struct Lut2D {
    static_assert(R >= 1 && C >= 1, "Lut2D needs at least one breakpoint per axis");

    ColVec<R, T>    rows{}; //!< Strictly increasing row breakpoints.
    ColVec<C, T>    cols{}; //!< Strictly increasing column breakpoints.
    Matrix<R, C, T> z{};    //!< Grid values, z(row, col).

    /// Bilinearly interpolated value at (@p r, @p c), clamped to the grid edges.
    [[nodiscard]] constexpr T operator()(T r, T c) const {
        const T rc = wet::clamp(r, rows[0], rows[R - 1]);
        const T cc = wet::clamp(c, cols[0], cols[C - 1]);

        const size_t i = (R == 1) ? 0 : lut_segment(rows, rc);
        const size_t j = (C == 1) ? 0 : lut_segment(cols, cc);

        const T tr = (R == 1) ? T{0} : inverse_lerp(rows[i], rows[i + 1], rc);
        const T tc = (C == 1) ? T{0} : inverse_lerp(cols[j], cols[j + 1], cc);

        const size_t i1 = (R == 1) ? 0 : i + 1;
        const size_t j1 = (C == 1) ? 0 : j + 1;

        const T top = lerp(z(i, j), z(i, j1), tc);
        const T bot = lerp(z(i1, j), z(i1, j1), tc);
        return lerp(top, bot, tr);
    }
};

} // namespace wet
