#pragma once

/**
 * @file math.hpp
 * @brief Math dispatch layer: constexpr series at compile time, MathBackend at runtime
 *
 * Each function uses std::is_constant_evaluated() to dispatch:
 * - At compile time: series expansions / Newton-Raphson (necessary for constexpr)
 * - At runtime: MathBackend<T> (platform-specific or std:: fallback)
 *
 * This ensures consteval design functions work while runtime controllers
 * get full hardware math performance.
 *
 * @see math_backend.hpp for backend selection via wet_profile.hpp
 */

#include <cmath>
#include <limits>
#include <numbers>
#include <type_traits>
#include <utility>

#include "math_backend.hpp"

namespace wet {

namespace detail {
/**
 * @brief Round to nearest integer (ties away from zero), returned as long long.
 *
 * Used for argument reduction in the constexpr trig/exp paths. Avoids std::round
 * (not constexpr-usable for our purposes) and the slow, precision-losing
 * subtract-in-a-loop reduction. The caller is responsible for keeping the
 * quotient within the range of long long; for the elementary functions here the
 * inputs that matter for compile-time design code are well within that range.
 */
template<typename T>
constexpr long long lround_away(T x) {
    return static_cast<long long>(x >= T{0} ? x + T{0.5} : x - T{0.5});
}
} // namespace detail

/**
 * @brief Compute square root using Newton's method (constexpr)
 *
 * Computes sqrt(x) using iterative Newton-Raphson: x_{n+1} = (x_n + S/x_n) / 2
 * Returns 0 for negative inputs (NaN not available in constexpr context).
 *
 * @tparam T Numeric type (must support arithmetic operations)
 * @param x  Value to compute square root of
 *
 * @return Square root of x, or 0 if x < 0
 */
template<typename T>
constexpr T sqrt(T x) {
    if (!std::is_constant_evaluated()) {
        return MathBackend<T>::sqrt(x);
    }
    if (x == T{0}) {
        return T{0};
    }
    if (x < T{0}) {
        return T{0}; // NaN not available in constexpr, return 0 for negative
    }

    // Newton's method: x_{n+1} = (x_n + S/x_n) / 2
    T guess = x > T{1} ? x / T{2} : T{1};
    for (int i = 0; i < 50; ++i) {
        T next = (guess + (x / guess)) / T{2};
        if (next == guess) {
            break;
        }
        guess = next;
    }
    return guess;
}

/**
 * @brief Compute absolute value (constexpr)
 *
 * @tparam T Numeric type (must support comparison and negation)
 * @param x  Value to take absolute value of
 *
 * @return |x|
 */
template<typename T>
constexpr T abs(T x) {
    if (!std::is_constant_evaluated()) {
        return MathBackend<T>::abs(x);
    }
    return x >= T{0} ? x : -x;
}

/**
 * @brief Compute cube root using Newton's method (constexpr)
 *
 * Computes cbrt(x) using iterative Newton-Raphson for cube roots.
 * Correctly handles negative inputs by preserving sign.
 *
 * @tparam T Numeric type (must support arithmetic operations)
 * @param x  Value to compute cube root of
 *
 * @return Cube root of x (preserves sign for negative x)
 */
template<typename T>
constexpr T cbrt(T x) {
    if (!std::is_constant_evaluated()) {
        return MathBackend<T>::cbrt(x);
    }
    if (x == T{0}) {
        return T{0};
    }

    bool neg = x < T{0};
    if (neg) {
        x = -x;
    }

    // Newton's method for cube root
    T guess = x > T{1} ? x / T{3} : T{1};
    for (int i = 0; i < 50; ++i) {
        T next = ((T{2} * guess) + (x / (guess * guess))) / T{3};
        if (next == guess) {
            break;
        }
        guess = next;
    }

    return neg ? -guess : guess;
}

/**
 * @brief Compute two-argument arctangent (constexpr)
 *
 * Computes atan2(y, x) ∈ [−π, π] using Taylor series with three-interval
 * range reduction:
 *
 *     |t| ≤ tan(π/8):           atan(t) directly
 *     tan(π/8) < |t| ≤ tan(3π/8): atan(t) = π/4 + atan((t−1)/(t+1))
 *     |t| > tan(3π/8):          atan(t) = π/2 − atan(1/t)
 *
 * @see Cody & Waite, "Software Manual for the Elementary Functions" (1980)
 *
 * @tparam T Numeric type (floating-point)
 * @param y  Y-coordinate
 * @param x  X-coordinate
 * @return Angle in radians ∈ [−π, π]
 */
template<typename T>
constexpr T atan2(T y, T x) {
    if (!std::is_constant_evaluated()) {
        return MathBackend<T>::atan2(y, x);
    }
    constexpr T pi = std::numbers::pi_v<T>;

    if (x == T{0}) {
        if (y > T{0}) {
            return pi / T{2};
        }
        if (y < T{0}) {
            return -pi / T{2};
        }
        return T{0};
    }

    // Use identity: atan(y/x) for different ranges
    T ratio = y / x;
    T atan_val;

    // For better accuracy, use range reduction:
    // atan(x) = pi/4 + atan((x-1)/(x+1)) for x > 0
    // This improves convergence near x = 1

    T abs_ratio = ratio >= T{0} ? ratio : -ratio;

    if (abs_ratio <= T{0.4142135623730951}) {
        // |x| <= tan(pi/8) ≈ 0.414, use Taylor series directly
        // atan(x) = x - x³/3 + x⁵/5 - x⁷/7 + ...
        T r2 = ratio * ratio;
        T term = ratio;
        atan_val = term;
        for (int n = 1; n <= 15; ++n) {
            term *= -r2;
            atan_val += term / T((2 * n) + 1);
        }
    } else if (abs_ratio <= T{2.4142135623730951}) {
        // tan(pi/8) < |x| <= tan(3*pi/8) ≈ 2.414
        // Use atan(x) = pi/4 + atan((x-1)/(x+1))
        T reduced = (abs_ratio - T{1}) / (abs_ratio + T{1});
        T r2 = reduced * reduced;
        T term = reduced;
        T atan_reduced = term;
        for (int n = 1; n <= 15; ++n) {
            term *= -r2;
            atan_reduced += term / T((2 * n) + 1);
        }
        atan_val = (pi / T{4}) + atan_reduced;
        if (ratio < T{0}) {
            atan_val = -atan_val;
        }
    } else {
        // |x| > tan(3*pi/8), use atan(x) = pi/2 - atan(1/x)
        T inv = T{1} / abs_ratio;
        T r2 = inv * inv;
        T term = inv;
        T atan_inv = term;
        for (int n = 1; n <= 15; ++n) {
            term *= -r2;
            atan_inv += term / T((2 * n) + 1);
        }
        atan_val = (pi / T{2}) - atan_inv;
        if (ratio < T{0}) {
            atan_val = -atan_val;
        }
    }

    // Adjust for quadrant
    if (x < T{0}) {
        atan_val += (y >= T{0} ? pi : -pi);
    }

    return atan_val;
}

/**
 * @brief Compute single-argument arctangent (constexpr) — atan(x) = atan2(x, 1).
 * @tparam T Numeric type (floating-point)
 * @return Angle in radians ∈ (−π/2, π/2)
 */
template<typename T>
constexpr T atan(T x) {
    if (!std::is_constant_evaluated()) {
        return MathBackend<T>::atan(x);
    }
    return atan2(x, T{1});
}

/**
 * @brief Compute arcsine (constexpr) — asin(x) = atan2(x, √(1−x²)).
 * Input is clamped to [−1, 1].
 * @tparam T Numeric type (floating-point)
 * @return Angle in radians ∈ [−π/2, π/2]
 */
template<typename T>
constexpr T asin(T x) {
    // Clamp the domain in both paths so behavior matches at compile and run time
    // (std::asin would return NaN for |x| > 1).
    constexpr T half_pi = std::numbers::pi_v<T> / T{2};
    if (x >= T{1}) {
        return half_pi;
    }
    if (x <= T{-1}) {
        return -half_pi;
    }
    if (!std::is_constant_evaluated()) {
        return MathBackend<T>::asin(x);
    }
    // sqrt((1−x)(1+x)) — algebraically equal to sqrt(1−x²) but avoids
    // catastrophic cancellation near |x| = 1, and the product cannot round
    // negative under IEEE rules.
    return atan2(x, sqrt((T{1} - x) * (T{1} + x)));
}

/**
 * @brief Compute arccosine (constexpr) — acos(x) = atan2(√(1−x²), x).
 * Input is clamped to [−1, 1].
 * @tparam T Numeric type (floating-point)
 * @return Angle in radians ∈ [0, π]
 */
template<typename T>
constexpr T acos(T x) {
    // Clamp the domain in both paths (std::acos would return NaN for |x| > 1).
    if (x >= T{1}) {
        return T{0};
    }
    if (x <= T{-1}) {
        return std::numbers::pi_v<T>;
    }
    if (!std::is_constant_evaluated()) {
        return MathBackend<T>::acos(x);
    }
    return atan2(sqrt((T{1} - x) * (T{1} + x)), x);
}

/**
 * @brief Compute cosine using Taylor series (constexpr)
 *
 * Computes cos(x) using Taylor series: cos(x) = 1 - x²/2! + x⁴/4! - x⁶/6! + ...
 * Input is reduced to [-π, π] for better accuracy.
 *
 * @tparam T Numeric type (floating-point)
 * @param x  Angle in radians
 *
 * @return cos(x)
 */
template<typename T>
constexpr T cos(T x) {
    if (!std::is_constant_evaluated()) {
        return MathBackend<T>::cos(x);
    }
    // Reduce to [-π, π] via k = round(x / 2π), x ← x − k·2π.
    // 2π is carried as a two-part (Cody–Waite) constant so that k·2π is formed
    // to ~66 bits, avoiding the precision loss of subtracting a single-double 2π
    // (and the unbounded iteration count of a subtract-in-a-loop reduction).
    constexpr T two_pi_hi = T{6.28318530693650245668};
    constexpr T two_pi_lo = T{2.43084020260247689728e-10};
    constexpr T inv_two_pi = T{0.15915494309189533577};
    const T     kreal = static_cast<T>(detail::lround_away(x * inv_two_pi));
    x = (x - (kreal * two_pi_hi)) - (kreal * two_pi_lo);

    // Taylor series: cos(x) = 1 - x²/2! + x⁴/4! - x⁶/6! + ...
    T x2 = x * x;
    T result = T{1};
    T term = T{1};
    for (int n = 1; n <= 12; ++n) {
        term *= -x2 / T(2 * n * ((2 * n) - 1));
        result += term;
    }
    return result;
}

/**
 * @brief Compute sine using Taylor series (constexpr)
 *
 * Computes sin(x) using Taylor series: sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + ...
 * Input is reduced to [-π, π] for better accuracy.
 *
 * @tparam T Numeric type (floating-point)
 * @param x  Angle in radians
 *
 * @return sin(x)
 */
template<typename T>
constexpr T sin(T x) {
    if (!std::is_constant_evaluated()) {
        return MathBackend<T>::sin(x);
    }
    // Reduce to [-π, π] via two-part 2π (see cos() for rationale).
    constexpr T two_pi_hi = T{6.28318530693650245668};
    constexpr T two_pi_lo = T{2.43084020260247689728e-10};
    constexpr T inv_two_pi = T{0.15915494309189533577};
    const T     kreal = static_cast<T>(detail::lround_away(x * inv_two_pi));
    x = (x - (kreal * two_pi_hi)) - (kreal * two_pi_lo);

    T x2 = x * x;
    T result = x;
    T term = x;
    for (int n = 1; n <= 12; ++n) {
        term *= -x2 / T((2 * n) * ((2 * n) + 1));
        result += term;
    }
    return result;
}

/**
 * @brief Combined sine and cosine
 *
 * Returns {sin(x), cos(x)}. At runtime a platform MathBackend can compute both
 * from a single range reduction (~half the cost of separate calls), which is
 * why FOC transforms (Park) should prefer this over calling sin and cos.
 *
 * @tparam T Numeric type (floating-point)
 * @param x  Angle in radians
 * @return std::pair<T, T> {sin(x), cos(x)}
 */
template<typename T>
constexpr std::pair<T, T> sincos(T x) {
    if (!std::is_constant_evaluated()) {
        return MathBackend<T>::sincos(x);
    }
    return {sin(x), cos(x)};
}

/**
 * @brief Compute tangent via continued fraction (constexpr)
 *
 * Reduces x to r ∈ [-π/2, π/2], then evaluates:
 *
 *     tan(r) = r / (1 − r² / (3 − r² / (5 − r² / (7 − ⋯))))
 *
 * using bottom-up (backward) recurrence. For |r| close to π/2 where the
 * continued fraction converges slowly, uses the identity
 * tan(r) = −1/tan(π/2 − r) with the complementary angle.
 *
 * @note Compare with MATLAB's tan(x).
 * @see Cuyt et al., "Handbook of Continued Fractions for Special Functions" (2008), §12.1
 *
 * @tparam T Numeric type (floating-point)
 * @param x  Angle in radians
 * @return tan(x)
 */
template<typename T>
constexpr T tan(T x) {
    if (!std::is_constant_evaluated()) {
        return MathBackend<T>::tan(x);
    }
    constexpr T pi = std::numbers::pi_v<T>;
    constexpr T half_pi = pi / T{2};

    // Reduce to r ∈ [-π/2, π/2] via k = round(x / π), with π carried as a
    // two-part (Cody–Waite) constant so that k·π is formed to ~66 bits.
    constexpr T pi_hi = T{3.14159265346825122834};
    constexpr T pi_lo = T{1.21542010130123844986e-10};
    constexpr T inv_pi = T{0.31830988618379067154};
    const T     kreal = static_cast<T>(detail::lround_away(x * inv_pi));
    T           r = (x - (kreal * pi_hi)) - (kreal * pi_lo);

    // Near ±π/2 the continued fraction converges slowly. Work with the
    // complementary angle via the cotangent identity, valid for r ∈ (0, π/2):
    //     tan(r) = cot(π/2 − r) = 1 / tan(π/2 − r)
    // The result's sign follows the sign of r (tan is odd), applied at the end.
    T abs_r = r >= T{0} ? r : -r;
    if (abs_r > T{1.2}) {
        T comp = half_pi - abs_r;
        // Evaluate tan(comp) via continued fraction (comp is small here)
        constexpr int N = 20;

        T x2 = comp * comp;
        T cf = T((2 * N) + 1);
        for (int i = N - 1; i >= 0; --i) {
            cf = T((2 * i) + 1) - (x2 / cf);
        }
        T tan_comp = comp / cf;
        T result = T{1} / tan_comp;
        return r >= T{0} ? result : -result;
    }

    // Continued fraction: tan(r) = r / (1 − r²/(3 − r²/(5 − ⋯)))
    // Evaluate bottom-up from depth N
    constexpr int N = 20;

    T x2 = r * r;
    T cf = T((2 * N) + 1);
    for (int i = N - 1; i >= 0; --i) {
        cf = T((2 * i) + 1) - (x2 / cf);
    }

    return r / cf;
}

/**
 * @brief Compute exponential function using Taylor series (constexpr)
 *
 * Computes exp(x) using Taylor series: exp(x) = 1 + x + x²/2! + x³/3! + ...
 *
 * @tparam T Numeric type (floating-point)
 * @param x  Exponent
 *
 * @return exp(x)
 */
template<typename T>
constexpr T exp(T x) {
    if (!std::is_constant_evaluated()) {
        return MathBackend<T>::exp(x);
    }
    // Overflow / underflow guards (double range; for float the 2^k scaling
    // below saturates to ±inf / 0 on its own well before these bounds).
    if (x > T{709.782712893384}) {
        return std::numeric_limits<T>::infinity();
    }
    if (x < T{-745.133219101941}) {
        return T{0};
    }

    // Argument reduction: x = k·ln2 + r with k = round(x/ln2) and |r| ≤ ln2/2,
    // so exp(x) = 2^k · exp(r). The Taylor series then only ever sees the small,
    // fast-converging remainder r ∈ [−0.347, 0.347] instead of the raw argument.
    // ln2 is carried as a two-part (Cody–Waite) constant for an accurate r.
    constexpr T     ln2_hi = T{6.93147180369123816490e-01};
    constexpr T     ln2_lo = T{1.90821492927058770002e-10};
    constexpr T     inv_ln2 = T{1.44269504088896340736};
    const long long k = detail::lround_away(x * inv_ln2);
    const T         kreal = static_cast<T>(k);
    const T         r = (x - (kreal * ln2_hi)) - (kreal * ln2_lo);

    T result = T{1};
    T term = T{1};
    for (int n = 1; n <= 14; ++n) {
        term *= r / T(n);
        result += term;
    }

    // Scale by 2^k. Done by repeated doubling/halving on the O(1)-magnitude
    // series result so intermediates stay finite — forming 2^k directly (e.g.
    // via pow) would overflow to ∞ on the final squaring for large k and that
    // is not a constant expression, even when the final exp(x) is finite.
    if (k >= 0) {
        for (long long i = 0; i < k; ++i) {
            result *= T{2};
        }
    } else {
        for (long long i = 0; i < -k; ++i) {
            result *= T{0.5};
        }
    }
    return result;
}

/**
 * @brief Compute natural logarithm using Newton-Raphson (constexpr)
 *
 * Solves eʸ = x via Newton-Raphson: y[n+1] = y[n] − (eʸⁿ − x)/eʸⁿ.
 * Argument reduction by powers of e keeps the iterate in [e⁻¹, e]
 * where convergence is quadratic.
 *
 * @tparam T Numeric type (floating-point)
 * @param x  Value (must be > 0)
 * @return ln(x), or 0 if x ≤ 0
 */
template<typename T>
constexpr T log(T x) {
    if (!std::is_constant_evaluated()) {
        return MathBackend<T>::log(x);
    }
    if (x <= T{0}) {
        return T{0};
    }

    // Argument reduction: find integer k such that y = x / e^k is in [e^{-1}, e]
    // Then ln(x) = ln(y) + k
    constexpr T e_val = T{2.718281828459045235360287};
    constexpr T e_inv = T{0.367879441171442321595524};

    T k = T{0};
    T y = x;
    while (y > e_val) {
        y *= e_inv;
        k += T{1};
    }
    while (y < e_inv) {
        y *= e_val;
        k -= T{1};
    }

    // Newton-Raphson on reduced y ∈ [e^{-1}, e], starting from y - 1
    T guess = y - T{1};
    for (int i = 0; i < 50; ++i) {
        T e_guess = exp(guess);
        T next = guess - ((e_guess - y) / e_guess);
        if (abs(next - guess) < T{1e-15}) {
            break;
        }
        guess = next;
    }
    return guess + k;
}

/**
 * @brief Compute power function (constexpr)
 *
 * Computes base^exp = exp(exp * ln(base)).
 * Assumes base > 0.
 *
 * @tparam T Numeric type (floating-point)
 * @param base  Base value (must be > 0)
 * @param exp   Exponent
 *
 * @return base^exp, or 0 if base <= 0
 */
template<typename T>
constexpr T pow(T base, T exponent) {
    if (!std::is_constant_evaluated()) {
        return MathBackend<T>::pow(base, exponent);
    }
    if (exponent == T{0}) {
        return T{1};
    }
    if (base <= T{0}) {
        return T{0};
    }
    return wet::exp(wet::log(base) * exponent);
}

/**
 * @brief  Compute integer power function (constexpr)
 *
 * @param base Base value
 * @param up Exponent (integer)
 *
 * @return base^up
 */
template<typename T>
constexpr T pow(T base, int up) {
    if (up == 0) {
        return T{1};
    }
    if (base == T{0}) {
        return T{0};
    }
    T   result = T{1};
    T   b = base;
    int exponent = up >= 0 ? up : -up;
    while (exponent > 0) {
        if (exponent % 2 == 1) {
            result *= b;
        }
        b *= b;
        exponent /= 2;
    }
    return up >= 0 ? result : T{1} / result;
}

/**
 * @brief Compute floor function (constexpr)
 *
 * Returns the largest integer less than or equal to x.
 *
 * @tparam T Numeric type (floating-point)
 * @param x  Value
 *
 * @return floor(x)
 */
template<typename T>
constexpr T floor(T x) {
    if (!std::is_constant_evaluated()) {
        return MathBackend<T>::floor(x);
    }
    T int_part = static_cast<long long>(x);
    if (x < T{0} && x != int_part) {
        int_part -= T{1};
    }
    return int_part;
}

/**
 * @brief Compute ceiling function (constexpr)
 *
 * Returns the smallest integer greater than or equal to x.
 *
 * @tparam T Numeric type (floating-point)
 * @param x  Value
 *
 * @return ceil(x)
 */
template<typename T>
constexpr T ceil(T x) {
    if (!std::is_constant_evaluated()) {
        return MathBackend<T>::ceil(x);
    }
    T int_part = static_cast<long long>(x);
    if (x > T{0} && x != int_part) {
        int_part += T{1};
    }
    return int_part;
}

/**
 * @brief Floating-point remainder (constexpr) — x − y·trunc(x/y), sign of x.
 *
 * Matches std::fmod's truncated-quotient convention.
 *
 * @tparam T Numeric type (floating-point)
 * @return Remainder of x/y, or 0 if y == 0
 */
template<typename T>
constexpr T fmod(T x, T y) {
    // Guard y == 0 in both paths (returns 0 rather than std::fmod's NaN).
    if (y == T{0}) {
        return T{0};
    }
    if (!std::is_constant_evaluated()) {
        return MathBackend<T>::fmod(x, y);
    }
    const T q = x / y;
    const T truncated = static_cast<T>(static_cast<long long>(q)); // toward zero
    return x - (truncated * y);
}

/**
 * @brief Compute base-10 logarithm (constexpr)
 *
 * Computes log10(x) = ln(x) / ln(10).
 * Assumes x > 0.
 *
 * @tparam T Numeric type (floating-point)
 * @param x  Value (must be > 0)
 *
 * @return log10(x), or 0 if x <= 0
 */
template<typename T>
constexpr T log10(T x) {
    if (!std::is_constant_evaluated()) {
        return MathBackend<T>::log10(x);
    }
    if (x <= T{0}) {
        return T{0};
    }
    return wet::log(x) / wet::log(T{10});
}

/**
 * @brief Sign function (constexpr)
 *
 * @param val Value
 *
 * @return -1 if val < 0, 1 if val > 0, 0 if val == 0
 */
template<typename T>
constexpr int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

/**
 * @brief Copy sign (constexpr) — magnitude of @p mag with the sign of @p sgn_src.
 * @tparam T Numeric type (floating-point)
 */
template<typename T>
constexpr T copysign(T mag, T sgn_src) {
    if (!std::is_constant_evaluated()) {
        return MathBackend<T>::copysign(mag, sgn_src);
    }
    const T m = mag >= T{0} ? mag : -mag;
    return sgn_src < T{0} ? -m : m;
}

/**
 * @brief Finiteness test (constexpr) — false for NaN and ±∞.
 *
 * |x| ≤ max() is false for both ±∞ (greater than max) and NaN (all comparisons
 * with NaN are false), and true for every finite value.
 *
 * @tparam T Numeric type (floating-point)
 */
template<typename T>
constexpr bool isfinite(T x) {
    if (!std::is_constant_evaluated()) {
        return MathBackend<T>::isfinite(x);
    }
    return abs(x) <= std::numeric_limits<T>::max();
}

} // namespace wet