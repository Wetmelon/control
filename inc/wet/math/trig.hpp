
#pragma once

/**
 * @file trig.hpp
 * @brief Fast float sine and cosine with full-range wrapping
 */

#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>

namespace wet {

inline float sin(float angle_rad);
inline float cos(float angle_rad);
inline float asin(float x);
inline float acos(float x);
inline float atan(float x);
inline float atan2(float y, float x);

inline float sqrt(float x);

struct SinCosResult {
    float sin;
    float cos;
};

inline SinCosResult sincos(float angle_rad);

} // namespace wet

namespace wet::detail {

constexpr float sin_coeffs[] = {
    +3.141582250595093e+0f,
    -5.167152404785156e+0f,
    +2.541962385177612e+0f,
    -5.547642707824707e-1f,
};

constexpr float asin_coeffs[] = {
    +1.570795416831970e+00f,
    -2.145179212093353e-01f,
    +8.791970461606979e-02f,
    -4.508748650550842e-02f,
    +1.950908824801445e-02f,
    -4.407065920531750e-03f,
};

constexpr float atan_coeffs[] = {
    +4.543576608284638e-07f,
    +9.999714493751526e-01f,
    +1.813957787817344e-04f,
    -3.311897814273834e-01f,
    -2.631079405546188e-02f,
    +3.089837431907654e-01f,
    -2.172826975584030e-01f,
    +5.104487016797066e-02f,
};

/**
 * @brief Decomposition of x = pi * (period_index + frac), frac in [-0.5, 0.5]
 *
 * Only the parity bit of period_index is needed at runtime to recover
 * sin/cos via sin(x) = (-1)^period_index * sin(pi*frac).
 */
struct Reduced {
    float frac;
    int   period_index;
};

static inline Reduced wrap(float x);
static inline float   sin_poly(float frac);

template<size_t N>
static inline float horner_eval(float x, const float (&coeffs)[N]);

template<size_t N>
static inline float estrin_eval(float x, const float (&coeffs)[N]);

} // namespace wet::detail

namespace wet {

/**
 * @brief Fast single-precision square root
 *
 * Emits the bare VSQRT instruction directly, skipping the errno/NaN libcall
 * fallback that std::sqrt carries unless -fno-math-errno is set.  Returns NaN
 * for x < 0, matching IEEE FPU behaviour (no errno, no trap).
 *
 * @param x  Input value (>= 0 for a real result)
 * @return sqrt(x)
 */
inline float sqrt(float x) {
#if defined(__ARM_FP)
    float result;
    asm("vsqrt.f32 %0, %1" : "=t"(result) : "t"(x));
    return result;
#else
    return std::sqrt(x);
#endif
}

/**
 * @brief Sine of an angle in radians
 *
 * Accepts any float; ~8 ULP accuracy.
 *
 * @param angle_rad  Angle in radians
 * @return sin(angle_rad)
 */
inline float sin(float angle_rad) {
    auto [frac, period_index] = detail::wrap(angle_rad);
    float s = detail::sin_poly(frac);

    return ((period_index & 1) != 0) ? -s : s;
}

/**
 * @brief Cosine of an angle in radians
 *
 * Uses the same unshifted reduction as sin, then cos(pi*frac) =
 * sin(pi*(0.5 - |frac|)).  Folding the pi/2 shift in *before* nearbyint
 * (as cos(x) = sin(x + pi/2) would) costs ~0.7e-6 of accuracy at large
 * |x|; the identity keeps the cos path on par with the sin path.
 *
 * @param angle_rad  Angle in radians
 * @return cos(angle_rad)
 */
inline float cos(float angle_rad) {
    auto [frac, period_index] = detail::wrap(angle_rad);
    float c = detail::sin_poly(0.5f - std::fabs(frac));

    return ((period_index & 1) != 0) ? -c : c;
}

/**
 * @brief Paired sin/cos from a single range reduction
 *
 * cos(pi*frac) = sin(pi*(0.5 - |frac|)); the fabs keeps the cos polynomial
 * argument in [0, 0.5] where the fit is valid.
 *
 * @param angle_rad  Angle in radians
 * @return SinCosResult{sin, cos}
 */
inline SinCosResult sincos(float angle_rad) {
    auto [frac, period_index] = detail::wrap(angle_rad);

    float s = detail::sin_poly(frac);
    float c = detail::sin_poly(0.5f - std::fabs(frac));

    if ((period_index & 1) != 0) {
        s = -s;
        c = -c;
    }

    return {.sin = s, .cos = c};
}

/**
 * @brief Arcsine, x in [-1, 1], result in [-pi/2, pi/2]
 *
 * @param x  Input in [-1, 1] (clamped if outside)
 * @return asin(x)
 */
inline float asin(float x) {
    bool negate = false;

    if (x < 0.0f) {
        x = -x;
        negate = true;
    }

    x = wet::min(x, 1.0f);

    float sqrt_1_minus_x = sqrt(1.0f - x);
    float p = detail::estrin_eval(x, detail::asin_coeffs);
    float result = (std::numbers::pi_v<float> / 2.0f) - (sqrt_1_minus_x * p);

    return negate ? -result : result;
}

/**
 * @brief Arccosine, x in [-1, 1], result in [0, pi]
 *
 * @param x  Input in [-1, 1] (clamped if outside)
 * @return acos(x)
 */
inline float acos(float x) {
    bool negate = false;

    if (x < 0.0f) {
        x = -x;
        negate = true;
    }

    x = wet::min(x, 1.0f);

    float sqrt_1_minus_x = sqrt(1.0f - x);
    float p = detail::estrin_eval(x, detail::asin_coeffs);
    float result = sqrt_1_minus_x * p;

    return negate ? (std::numbers::pi_v<float> - result) : result;
}

/**
 * @brief Arctangent, any float, result in [-pi/2, pi/2]
 *
 * |x| > 1 reduced via atan(x) = pi/2 - atan(1/x).
 *
 * @param x  Input value
 * @return atan(x)
 */
inline float atan(float x) {
    bool negate = false;
    bool complement = false;

    if (x < 0.0f) {
        x = -x;
        negate = true;
    }

    if (std::abs(x) > 1.0f) {
        x = 1.0f / x;
        complement = true;
    }

    float result = detail::estrin_eval(x, detail::atan_coeffs);

    if (complement) {
        result = (std::numbers::pi_v<float> / 2.0f) - result;
    }

    return negate ? -result : result;
}

/**
 * @brief Two-argument arctangent, result in [-pi, pi]
 *
 * Polynomial is fit on [0, 1]; min/max + sign tricks fold all four quadrants
 * onto that interval.
 *
 * @param y  Y-coordinate
 * @param x  X-coordinate
 * @return atan2(y, x)
 */
inline float atan2(float y, float x) {
    float ax = std::fabs(x);
    float ay = std::fabs(y);
    float lo = std::fmin(ax, ay);
    float hi = std::fmax(ax, ay) + std::numeric_limits<float>::min();
    float t = detail::estrin_eval(lo / hi, detail::atan_coeffs);

    float r = (ay > ax) ? ((std::numbers::pi_v<float> / 2.0f) - t) : t;
    r = (x >= 0.0f) ? r : (std::numbers::pi_v<float> - r);

    return std::copysign(r, y);
}

} // namespace wet

namespace wet::detail {

/**
 * @brief Value barrier used to prevent fast-math reassociation in Cody-Waite
 *        range reduction.
 *
 * Prefers compiler-provided reassociation barriers when available. On ARM FP
 * targets without that builtin, an empty inline-asm with a "+w" constraint keeps
 * the value in a VFP register while still acting as an optimization barrier.
 * Other targets fall back to identity.
 */
static inline float reassoc_barrier(float v) {
#if defined(__has_builtin)
#if __has_builtin(__builtin_assoc_barrier)
    return __builtin_assoc_barrier(v);
#endif
#endif

#if (defined(__GNUC__) || defined(__clang__)) && defined(__ARM_FP)
    asm volatile("" : "+w"(v));
#endif
    return v;
}

/**
 * @brief Cody-Waite range reduction of x (radians) to {frac, period_index},
 *        accurate to |x| ~25700 rad. The three pi-word subtractions must not be
 *        reassociated (collapses to single-step precision). Clang requires
 *        `#pragma clang fp reassociate(off)`; gcc needs memory barriers
 */
static inline Reduced wrap(float x) {
    constexpr float inv_pi = std::numbers::inv_pi_v<float>;
    constexpr float PI_HI = 3.140625f;          // pi, low 13 mantissa bits zeroed
    constexpr float PI_LO = 9.6765358467e-04f;  // pi - PI_HI
    constexpr float PI_LO2 = 5.1265658385e-12f; // pi - PI_HI - PI_LO

    float n = std::nearbyint(x * inv_pi);
    int   period_index = static_cast<int>(n);

    float r = reassoc_barrier(x - (n * PI_HI));
    r = reassoc_barrier(r - (n * PI_LO));
    r = reassoc_barrier(r - (n * PI_LO2));

    float frac = r * inv_pi;
    return {.frac = frac, .period_index = period_index};
}

/**
 * @brief Evaluate sin(pi*frac) using the degree-7 minimax polynomial
 *
 * sin(pi*frac) = frac * (c0 + c1*u + c2*u^2 + c3*u^3) with u = frac^2.
 * Precomputing u^2 lets GCC reorganize the FMAs as an even/odd split
 * for pipeline ILP; clang stays in Horner form.
 *
 * @param frac  Fractional half-period in [-0.5, 0.5]
 * @return sin(pi * frac)
 */
static inline float sin_poly(float frac) {
    float u = frac * frac;
    float u2 = u * u;

    float p = sin_coeffs[0];
    p += sin_coeffs[1] * u;
    p += sin_coeffs[2] * u2;
    p += sin_coeffs[3] * u2 * u;

    return frac * p;
}

/**
 * @brief Horner evaluation of a polynomial in float32
 *
 * Generic polynomial evaluator: p = coeffs[0] + x * (coeffs[1] + x * (...))
 * Compiles to a chain of FMA instructions on FPU architectures.
 *
 * @param x       Evaluation point
 * @param coeffs  Polynomial coefficients, lowest power first
 * @return p(x)
 */
template<size_t N>
static inline float horner_eval(float x, const float (&coeffs)[N]) {
    static_assert(N >= 1, "horner_eval requires at least one coefficient");
    float p = coeffs[N - 1];
    for (int i = static_cast<int>(N) - 2; i >= 0; --i) {
        p = coeffs[i] + (x * p);
    }
    return p;
}

/**
 * @brief Estrin's-scheme evaluation of a polynomial in float32
 *
 * Evaluates p(x) = coeffs[0] + coeffs[1]*x + ... + coeffs[N-1]*x^(N-1) by
 * pairwise combination: each level folds adjacent terms (b[i] = b[2i] +
 * power*b[2i+1]) and squares the power.  The per-level folds are mutually
 * independent, so the critical path is ~log2(N) dependent FMAs instead of
 * Horner's N. This form is also reassociable by -ffast-math, which Horner is not.
 *
 * @param x       Evaluation point
 * @param coeffs  Polynomial coefficients, lowest power first
 * @return p(x)
 */
template<size_t N>
static inline float estrin_eval(float x, const float (&coeffs)[N]) {
    static_assert(N >= 1, "estrin_eval requires at least one coefficient");
    float b[N];
    for (size_t i = 0; i < N; ++i) {
        b[i] = coeffs[i];
    }
    float power = x;
    for (size_t n = N; n > 1; n = (n + 1) / 2) {
        for (size_t i = 0; i < n / 2; ++i) {
            b[i] = b[2 * i] + (power * b[(2 * i) + 1]);
        }
        if (n % 2 != 0) {
            b[n / 2] = b[n - 1];
        }
        power = power * power;
    }
    return b[0];
}

} // namespace wet::detail
