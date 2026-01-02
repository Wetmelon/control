#pragma once

#include <complex>
#include <numbers>
#include <type_traits>

namespace wet {

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
 * @brief Compute complex square root (constexpr)
 *
 * Computes the principal square root of a complex number z.
 * Uses the formula: sqrt(z) = sqrt((|z| + Re(z))/2) + i*sign(Im(z))*sqrt((|z| - Re(z))/2)
 *
 * @tparam T Underlying floating-point type
 * @param z  Complex number to compute square root of
 *
 * @return Principal square root of z
 */
template<typename T>
constexpr std::complex<T> csqrt(const std::complex<T>& z) {
    T re = z.real();
    T im = z.imag();

    if (re == T{0} && im == T{0}) {
        return std::complex<T>{T{0}, T{0}};
    }

    T mag = wet::sqrt(re * re + im * im);
    T r = wet::sqrt((mag + re) / T{2});
    T i = wet::sqrt((mag - re) / T{2});

    // Sign of imaginary part matches sign of input imaginary part
    if (im < T{0}) {
        i = -i;
    }

    return std::complex<T>{r, i};
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

} // namespace wet