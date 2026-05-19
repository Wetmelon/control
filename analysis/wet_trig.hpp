/**
 * @file wet_trig.hpp
 * @brief Fast float sine and cosine with full-range wrapping
 *
 * Minimax polynomial approximations for sin/cos using nearbyint-based
 * range reduction.  Accepts any float input (not restricted to [0, 2pi]).
 *
 * Range reduction normalizes any angle x to g in [-0.5, 0.5] via:
 *
 *     n = nearbyint(x / pi)
 *     g = x / pi - n            (g is in half-periods)
 *
 * Then a single degree-7 minimax polynomial evaluates sin(pi*g):
 *
 *     sin(x)  = (-1)^n * sin(pi*g)
 *     cos(x)  = sin(x + pi/2)
 *     sincos  = one reduce, two poly evals via sin(pi*g) and sin(pi*(0.5-g))
 *
 * Accuracy: ~8 ULP max for float (comparable to TI ARM trig).
 * Performance: branchless, 17 instructions for sin on Cortex-M7.
 *
 * @note Compare with TI ARM's ti_arm::sin / ti_arm::cos which are
 *       restricted to [0, 2pi] and use manual quadrant branching.
 *
 * @see "Minimax Approximations" in Hart et al., "Computer Approximations" (1968)
 * @see analysis/minimax_trig.py for coefficient generation
 *
 * Example:
 * @code
 * #include "wet_trig.hpp"
 *
 * // Works for any angle — no manual wrapping needed
 * float s = wet::sin(3.7f);
 * float c = wet::cos(-100.0f);
 *
 * // sincos is cheaper than calling sin + cos separately
 * auto [sin_val, cos_val] = wet::sincos(angle);
 * @endcode
 */
#pragma once

namespace wet {

/**
 * @brief Paired sin/cos result
 *
 * Returned by sincos() to avoid redundant range reduction.
 */
struct SinCosResult {
    float sin; ///< sin(angle_rad)
    float cos; ///< cos(angle_rad)
};

/**
 * @brief Compute sine of an angle in radians
 *
 * Uses a degree-7 minimax polynomial on the half-period reduced argument.
 * Branchless — no quadrant if/else, sign correction via conditional negate.
 *
 * @param angle_rad  Angle in radians (any float value)
 * @return sin(angle_rad), accurate to ~8 ULP
 */
float sin(float angle_rad);

/**
 * @brief Compute cosine of an angle in radians
 *
 * Implemented as sin(angle_rad + pi/2). Same polynomial, one extra VADD.
 *
 * @param angle_rad  Angle in radians (any float value)
 * @return cos(angle_rad), accurate to ~8 ULP
 */
float cos(float angle_rad);

/**
 * @brief Compute sine and cosine simultaneously
 *
 * Performs range reduction once, then evaluates two polynomials:
 *
 *     sin(x) = (-1)^n * sin(pi*g)
 *     cos(x) = (-1)^n * sin(pi*(0.5 - g))
 *
 * This is cheaper than calling sin() + cos() separately (29 vs 35
 * instructions on Cortex-M7).
 *
 * @param angle_rad  Angle in radians (any float value)
 * @return SinCosResult with .sin and .cos fields
 */
SinCosResult sincos(float angle_rad);

/**
 * @brief Compute arcsine (inverse sine)
 *
 * Computes asin(x) for x in [-1, 1].  Uses a minimax polynomial approximation
 * with domain reduction via sqrt(1 - x²).
 *
 * @param x  Input value in [-1, 1]
 * @return asin(x) in [-π/2, π/2], accurate to ~8 ULP
 */
float asin(float x);

/**
 * @brief Compute arccosine (inverse cosine)
 *
 * Computes acos(x) for x in [-1, 1].  Uses a minimax polynomial approximation
 * with domain reduction via sqrt(1 - x²).
 *
 * @param x  Input value in [-1, 1]
 * @return acos(x) in [0, π], accurate to ~8 ULP
 */
float acos(float x);

/**
 * @brief Compute arctangent (inverse tangent)
 *
 * Computes atan(x) for any real x.  Uses a minimax polynomial approximation
 * with domain reduction for |x| > 1 via reciprocal.
 *
 * @param x  Input value (any float)
 * @return atan(x) in [-π/2, π/2], accurate to ~8 ULP
 */
float atan(float x);

/**
 * @brief Compute two-argument arctangent
 *
 * Computes atan2(y, x) using the signs and magnitudes of both arguments
 * to determine the correct quadrant.  Equivalent to atan(y/x) with proper
 * handling of edge cases.
 *
 * @param y  Y-coordinate (any float)
 * @param x  X-coordinate (any float)
 * @return atan2(y, x) in [-π, π], accurate to ~8 ULP
 */
float atan2(float y, float x);

} // namespace wet
