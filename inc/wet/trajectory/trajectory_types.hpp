#pragma once

/**
 * @file trajectory_types.hpp
 * @brief Shared value types and low-level helpers for the trajectory generators.
 *
 * Defines the data structures that appear across multiple profile families:
 * - @ref TrajectoryLimits — kinematic bounds (Vmax, Amax, Dmax, Jmax)
 * - @ref TrajectoryState — the commanded (position, velocity, acceleration, jerk) tuple
 * - @ref TrajectoryBoundary — endpoint derivative conditions for polynomial / spline BVPs
 * - @ref design::detail::factorial / @ref design::detail::falling_factorial — shared
 *   polynomial-coefficient helpers used by both @ref polynomial.hpp and @ref spline.hpp
 *
 * Include this header only when you need the shared types without pulling in a full
 * profile family. Normally just include the profile header you need (e.g.
 * `wet/trajectory/scurve.hpp`) — each one already pulls this in.
 */

#include <cstddef>
#include <type_traits>

#include "wet/math/math.hpp"

namespace wet {

/**
 * @brief Asymmetric kinematic limits for a trapezoidal or S-curve motion profile.
 *
 * @p max_acceleration bounds the approach-to-cruise ramp and @p max_deceleration
 * the approach-to-target ramp (they may differ). All three must be > 0.
 *
 * @tparam T Scalar type
 */
template<typename T = double>
struct TrajectoryLimits {

    T max_velocity{T{0}};     //!< |v| ≤ max_velocity (> 0)
    T max_acceleration{T{0}}; //!< accel-region bound Amax (> 0)
    T max_deceleration{T{0}}; //!< decel-region bound Dmax (> 0)
    T max_jerk{T{0}};         //!< |j| ≤ max_jerk (> 0; S-curve only, ignored by trapezoidal)

    /// Trapezoidal validity: positive v/a/d limits (jerk is not required).
    [[nodiscard]] constexpr bool valid() const {
        return (max_velocity > T{0}) && (max_acceleration > T{0}) && (max_deceleration > T{0});
    }

    /// S-curve validity: additionally requires a positive jerk limit.
    [[nodiscard]] constexpr bool valid_jerk_limited() const { return valid() && (max_jerk > T{0}); }
};

/**
 * @brief A point on a motion profile: commanded position, velocity, acceleration.
 * @tparam T Scalar type
 */
template<typename T = double>
struct TrajectoryState {

    T position{T{0}};
    T velocity{T{0}};
    T acceleration{T{0}};
    T jerk{T{0}}; //!< Only meaningful for jerk-continuous profiles (polynomial, S-curve)
};

/**
 * @brief Boundary conditions at one endpoint of a polynomial trajectory: a
 *        position and its time derivatives.
 *
 * How many derivatives are honored depends on the polynomial order: cubic uses
 * {position, velocity}; quintic adds acceleration; septic adds jerk.
 *
 * @tparam T Scalar type
 */
template<typename T = double>
struct TrajectoryBoundary {
    T position{T{0}};
    T velocity{T{0}};
    T acceleration{T{0}};
    T jerk{T{0}};
};

namespace design::detail {

/// k! (k ≤ 7 here, fits any scalar exactly).
template<typename T>
constexpr T factorial(size_t k) {
    T f{1};
    for (size_t i = 2; i <= k; ++i) {
        f *= static_cast<T>(i);
    }
    return f;
}

/// Falling factorial i·(i−1)···(i−k+1) = i! / (i−k)! — the k-th derivative
/// coefficient of tⁱ. Zero when i < k.
template<typename T>
constexpr T falling_factorial(size_t i, size_t k) {
    if (i < k) {
        return T{0};
    }
    T f{1};
    for (size_t m = 0; m < k; ++m) {
        f *= static_cast<T>(i - m);
    }
    return f;
}

} // namespace design::detail

} // namespace wet
