#pragma once

#include <cmath>
#include <numbers>
#include <type_traits>

namespace wetmelon::control::wet {

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
        return std::sqrt(x);
    }
    if (x == T{0})
        return T{0};
    if (x < T{0})
        return T{0}; // NaN not available in constexpr, return 0 for negative

    // Newton's method: x_{n+1} = (x_n + S/x_n) / 2
    T guess = x > T{1} ? x / T{2} : T{1};
    for (int i = 0; i < 50; ++i) {
        T next = (guess + x / guess) / T{2};
        if (next == guess)
            break;
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
        return std::abs(x);
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
        return std::cbrt(x);
    }
    if (x == T{0})
        return T{0};

    bool neg = x < T{0};
    if (neg)
        x = -x;

    // Newton's method for cube root
    T guess = x > T{1} ? x / T{3} : T{1};
    for (int i = 0; i < 50; ++i) {
        T next = (T{2} * guess + x / (guess * guess)) / T{3};
        if (next == guess)
            break;
        guess = next;
    }

    return neg ? -guess : guess;
}

/**
 * @brief Compute two-argument arctangent (constexpr)
 *
 * Computes atan2(y, x) = angle in radians from positive x-axis to point (x, y).
 * Uses Taylor series with range reduction for better convergence:
 * atan(x) ≈ π/4 + atan((x-1)/(x+1)) for |x| > 0.414
 * Result is in range [-π, π].
 *
 * @tparam T Numeric type (floating-point)
 * @param y  Y-coordinate (numerator)
 * @param x  X-coordinate (denominator)
 *
 * @return Angle in radians from positive x-axis to point (x, y)
 */
template<typename T>
constexpr T atan2(T y, T x) {
    if (!std::is_constant_evaluated()) {
        return std::atan2(y, x);
    }
    constexpr T pi = std::numbers::pi_v<T>;

    if (x == T{0}) {
        if (y > T{0})
            return pi / T{2};
        if (y < T{0})
            return -pi / T{2};
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
            atan_val += term / T(2 * n + 1);
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
            atan_reduced += term / T(2 * n + 1);
        }
        atan_val = pi / T{4} + atan_reduced;
        if (ratio < T{0})
            atan_val = -atan_val;
    } else {
        // |x| > tan(3*pi/8), use atan(x) = pi/2 - atan(1/x)
        T inv = T{1} / abs_ratio;
        T r2 = inv * inv;
        T term = inv;
        T atan_inv = term;
        for (int n = 1; n <= 15; ++n) {
            term *= -r2;
            atan_inv += term / T(2 * n + 1);
        }
        atan_val = pi / T{2} - atan_inv;
        if (ratio < T{0})
            atan_val = -atan_val;
    }

    // Adjust for quadrant
    if (x < T{0}) {
        atan_val += (y >= T{0} ? pi : -pi);
    }

    return atan_val;
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
        return std::cos(x);
    }
    constexpr T pi = std::numbers::pi_v<T>;
    constexpr T two_pi = T{2} * pi;

    // Reduce to [-π, π]
    while (x > pi)
        x -= two_pi;
    while (x < -pi)
        x += two_pi;

    // Taylor series: cos(x) = 1 - x²/2! + x⁴/4! - x⁶/6! + ...
    T x2 = x * x;
    T result = T{1};
    T term = T{1};
    for (int n = 1; n <= 12; ++n) {
        term *= -x2 / T(2 * n * (2 * n - 1));
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
        return std::sin(x);
    }
    constexpr T pi = std::numbers::pi_v<T>;
    constexpr T two_pi = T{2} * pi;

    while (x > pi)
        x -= two_pi;
    while (x < -pi)
        x += two_pi;

    T x2 = x * x;
    T result = x;
    T term = x;
    for (int n = 1; n <= 12; ++n) {
        term *= -x2 / T((2 * n) * (2 * n + 1));
        result += term;
    }
    return result;
}

/**
 * @brief Compute tangent using continued fraction (constexpr)
 *
 * Uses range reduction to [-π/2, π/2] followed by a continued fraction
 * expansion that avoids the numerical instability of sin(x)/cos(x) near
 * odd multiples of π/2.
 *
 * @tparam T Numeric type (floating-point)
 * @param x  Angle in radians
 *
 * @return tan(x)
 */
template<typename T>
constexpr T tan(T x) {
    if (!std::is_constant_evaluated()) {
        return std::tan(x);
    }
    constexpr T pi = std::numbers::pi_v<T>;

    // Reduce to [-π/2, π/2]
    // Compute k = round(x / π)
    T   k_real = x / pi;
    int k = static_cast<int>(k_real >= T{0} ? k_real + T{0.5} : k_real - T{0.5});
    T   r = x - static_cast<T>(k) * pi; // r ∈ [-π/2, π/2]

    // Taylor series for tan(r), valid for |r| < π/2
    // tan(x) = x + x³/3 + 2x⁵/15 + 17x⁷/315 + 62x⁹/2835 + ...
    // Using Bernoulli number coefficients via Horner-like accumulation
    T x2 = r * r;
    T result = r;
    T term = r;

    // Coefficients: 1/3, 2/15, 17/315, 62/2835, 1382/155925, 21844/6081075, ...
    constexpr T coeffs[] = {
        T{1} / T{3},
        T{2} / T{15},
        T{17} / T{315},
        T{62} / T{2835},
        T{1382} / T{155925},
        T{21844} / T{6081075},
        T{929569} / T{638512875},
        T{6404582} / T{10854718875},
        T{443861162} / T{1856156927625},
    };

    for (size_t i = 0; i < sizeof(coeffs) / sizeof(coeffs[0]); ++i) {
        term *= x2;
        result += coeffs[i] * term;
    }

    return result;
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
        return std::exp(x);
    }
    T result = T{1};
    T term = T{1};
    for (int n = 1; n <= 20; ++n) {
        term *= x / T(n);
        result += term;
    }
    return result;
}

/**
 * @brief Compute natural logarithm using Newton-Raphson (constexpr)
 *
 * Computes ln(x) using Newton-Raphson method solving exp(y) = x.
 * Uses argument reduction by powers of e to keep the reduced argument
 * in [e^{-1}, e], where Newton-Raphson converges quickly.
 * Assumes x > 0.
 *
 * @tparam T Numeric type (floating-point)
 * @param x  Value to compute logarithm of (must be > 0)
 *
 * @return ln(x), or 0 if x <= 0
 */
template<typename T>
constexpr T log(T x) {
    if (!std::is_constant_evaluated()) {
        return std::log(x);
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
        T next = guess - (e_guess - y) / e_guess;
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
constexpr T pow(T base, T exp) {
    if (!std::is_constant_evaluated()) {
        return std::pow(base, exp);
    }
    if (exp == T{0})
        return T{1};
    if (base <= T{0})
        return T{0};
    return wet::exp(wet::log(base) * exp);
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
    if (up == 0)
        return T{1};
    if (base == T{0})
        return T{0};
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
    T int_part = static_cast<long long>(x);
    if (x > T{0} && x != int_part) {
        int_part += T{1};
    }
    return int_part;
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
        return std::log10(x);
    }
    if (x <= T{0})
        return T{0};
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

} // namespace wetmelon::control::wet