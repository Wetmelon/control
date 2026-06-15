#pragma once

#include <cstdint>

#include "wet/backend.hpp"
#include "wet/math/complex.hpp"
#include "wet/math/math.hpp"
#include "wet/matrix/colvec.hpp"

namespace wet {

/**
 * @defgroup transforms Reference-Frame Transforms
 * @brief Coordinate transformations between the three-phase reference frames
 *
 * Clarke, Park, and their inverses map between the reference frames used in
 * field-oriented control (FOC) and grid-tie synchronization:
 *
 *   - **abc**  three-phase stationary frame (a column vector of phase quantities)
 *   - **αβ**   two-phase stationary frame (wet::AlphaBeta)
 *   - **αβ0**  two-phase stationary frame plus zero sequence (wet::AlphaBetaZero)
 *   - **dq**   rotor-synchronous rotating frame (wet::DirectQuadrature)
 *
 * Phase quantities live in a wet::ColVec<3, T> so they compose with the rest of
 * the linear-algebra library; the αβ and dq pairs use named structs so the two
 * orthogonal components never get silently swapped.
 *
 * Also provides the phasor-domain symmetrical-component (Fortescue) transform for
 * unbalanced / fault analysis.
 *
 * All transforms use the amplitude-invariant (2/3) convention, so the αβ and dq
 * magnitudes equal the peak phase amplitude.
 *
 * @see https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_transformation
 * @see https://en.wikipedia.org/wiki/Direct-quadrature-zero_transformation
 */

/**
 * @brief Scaling convention for the Clarke/Park family
 * @ingroup transforms
 *
 * Selects the magnitude normalisation of the αβ(0)/dq(0) transforms:
 *   - **AmplitudeInvariant** (2/3): αβ/dq magnitude equals the peak phase
 *     amplitude. The default; three-phase power is @f$ \tfrac32(v_\alpha i_\alpha
 *     + v_\beta i_\beta) @f$.
 *   - **PowerInvariant** (√(2/3), "Concordia"): the transform is orthonormal, so
 *     power is preserved directly — @f$ p = v_\alpha i_\alpha + v_\beta i_\beta @f$
 *     with no 3/2 factor.
 *
 * @see https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_transformation
 */
enum class Convention : std::uint8_t {
    AmplitudeInvariant,
    PowerInvariant,
};

/**
 * @brief Direct-quadrature (rotor-frame) component pair
 * @ingroup transforms
 *
 * Behaves as the complex number @f$ d + jq @f$: abs() / arg() give its polar form
 * and the arithmetic operators act component-wise (vector add/subtract and scalar
 * scaling), so dq references and feedforward terms compose naturally.
 */
template<typename T = float>
struct DirectQuadrature {
    T d, q;

    /// Phase angle @f$ \operatorname{atan2}(q, d) @f$ [rad].
    [[nodiscard]] constexpr T arg() const { return wet::atan2(q, d); }

    /// Magnitude @f$ \sqrt{d^2 + q^2} @f$.
    [[nodiscard]] constexpr T abs() const { return wet::sqrt((d * d) + (q * q)); }

    constexpr DirectQuadrature& operator+=(const DirectQuadrature& o) {
        d += o.d;
        q += o.q;
        return *this;
    }
    constexpr DirectQuadrature& operator-=(const DirectQuadrature& o) {
        d -= o.d;
        q -= o.q;
        return *this;
    }
    constexpr DirectQuadrature& operator*=(T s) {
        d *= s;
        q *= s;
        return *this;
    }
    constexpr DirectQuadrature& operator/=(T s) {
        d /= s;
        q /= s;
        return *this;
    }

    [[nodiscard]] constexpr DirectQuadrature operator-() const { return {-d, -q}; }

    [[nodiscard]] friend constexpr DirectQuadrature operator+(DirectQuadrature a, const DirectQuadrature& b) { return a += b; }
    [[nodiscard]] friend constexpr DirectQuadrature operator-(DirectQuadrature a, const DirectQuadrature& b) { return a -= b; }
    [[nodiscard]] friend constexpr DirectQuadrature operator*(DirectQuadrature v, T s) { return v *= s; }
    [[nodiscard]] friend constexpr DirectQuadrature operator*(T s, DirectQuadrature v) { return v *= s; }
    [[nodiscard]] friend constexpr DirectQuadrature operator/(DirectQuadrature v, T s) { return v /= s; }

    [[nodiscard]] friend constexpr bool operator==(const DirectQuadrature&, const DirectQuadrature&) = default;
};

/**
 * @brief Alpha-beta (stationary-frame) component pair
 * @ingroup transforms
 *
 * Behaves as the complex number @f$ \alpha + j\beta @f$: abs() / arg() give its
 * polar form and the arithmetic operators act component-wise (vector add/subtract
 * and scalar scaling).
 */
template<typename T = float>
struct AlphaBeta {
    T alpha, beta;

    /// Phase angle @f$ \operatorname{atan2}(\beta, \alpha) @f$ [rad].
    [[nodiscard]] constexpr T arg() const { return wet::atan2(beta, alpha); }

    /// Magnitude @f$ \sqrt{\alpha^2 + \beta^2} @f$.
    [[nodiscard]] constexpr T abs() const { return wet::sqrt((alpha * alpha) + (beta * beta)); }

    constexpr AlphaBeta& operator+=(const AlphaBeta& o) {
        alpha += o.alpha;
        beta += o.beta;
        return *this;
    }
    constexpr AlphaBeta& operator-=(const AlphaBeta& o) {
        alpha -= o.alpha;
        beta -= o.beta;
        return *this;
    }
    constexpr AlphaBeta& operator*=(T s) {
        alpha *= s;
        beta *= s;
        return *this;
    }
    constexpr AlphaBeta& operator/=(T s) {
        alpha /= s;
        beta /= s;
        return *this;
    }

    [[nodiscard]] constexpr AlphaBeta operator-() const { return {-alpha, -beta}; }

    [[nodiscard]] friend constexpr AlphaBeta operator+(AlphaBeta a, const AlphaBeta& b) { return a += b; }
    [[nodiscard]] friend constexpr AlphaBeta operator-(AlphaBeta a, const AlphaBeta& b) { return a -= b; }
    [[nodiscard]] friend constexpr AlphaBeta operator*(AlphaBeta v, T s) { return v *= s; }
    [[nodiscard]] friend constexpr AlphaBeta operator*(T s, AlphaBeta v) { return v *= s; }
    [[nodiscard]] friend constexpr AlphaBeta operator/(AlphaBeta v, T s) { return v /= s; }

    [[nodiscard]] friend constexpr bool operator==(const AlphaBeta&, const AlphaBeta&) = default;
};

/**
 * @brief Alpha-beta-zero (stationary-frame) component triple
 * @ingroup transforms
 *
 * The αβ pair plus the zero-sequence (common-mode) component, i.e. the full
 * rank-3 Clarke image. For balanced three-wire systems @f$ v_0 = 0 @f$ and this
 * carries no more information than AlphaBeta; retain it for unbalanced or
 * four-wire grids, where common-mode DC/offset and triplen harmonics live in the
 * zero channel rather than leaking into αβ.
 */
template<typename T = float>
struct AlphaBetaZero {
    T alpha, beta, zero;

    /// Drop the zero channel, returning just the αβ pair.
    [[nodiscard]] constexpr AlphaBeta<T> ab() const { return {alpha, beta}; }

    [[nodiscard]] friend constexpr bool operator==(const AlphaBetaZero&, const AlphaBetaZero&) = default;
};

/**
 * @brief Direct-quadrature-zero (rotor-frame) component triple
 * @ingroup transforms
 *
 * The dq pair plus the zero-sequence component, i.e. the full rank-3 Park image.
 * The Park rotation leaves the zero axis untouched (it passes straight through
 * from the Clarke zero), so this carries no extra information for balanced
 * three-wire drives; retain it for four-wire grid converters running dq0 current
 * control.
 */
template<typename T = float>
struct DirectQuadratureZero {
    T d, q, zero;

    /// Drop the zero channel, returning just the dq pair.
    [[nodiscard]] constexpr DirectQuadrature<T> dq() const { return {d, q}; }

    [[nodiscard]] friend constexpr bool operator==(const DirectQuadratureZero&, const DirectQuadratureZero&) = default;
};

/**
 * @brief Zero-retaining Clarke transform (abc → αβ0)
 * @ingroup transforms
 *
 * The full rank-3 amplitude-invariant Clarke transform, adding the zero-sequence
 * (common-mode) channel to αβ:
 * @f[
 *   \alpha = \tfrac{2a - b - c}{3}, \quad
 *   \beta  = \frac{b - c}{\sqrt{3}}, \quad
 *   v_0    = \tfrac{a + b + c}{3}
 * @f]
 * Common-mode content (equal on all three phases — e.g. shared sensor bias,
 * triplen harmonics) maps entirely to @f$ v_0 @f$ and cancels out of αβ;
 * independent per-phase offsets still appear in αβ and need upstream rejection.
 *
 * This is the rank-3 primitive; clarke_transform() is the αβ-only projection of
 * it (the zero channel is dead-code-eliminated when unused).
 *
 * MATLAB: `clarke()` with the zero row, `(2/3)·[1 -1/2 -1/2; 0 √3/2 -√3/2; 1/2 1/2 1/2]·[a;b;c]`
 *
 * @param abc Phase quantities {a, b, c}
 * @return αβ0 components
 */
template<typename T = float, Convention C = Convention::AmplitudeInvariant>
[[nodiscard]] constexpr AlphaBetaZero<T> clarke_zero_transform(const ColVec<3, T>& abc) {
    if constexpr (C == Convention::PowerInvariant) {
        // Orthonormal (Concordia) scaling: √(2/3) on αβ, 1/√3 on the zero row.
        constexpr T sqrt_2_3 = wet::numbers::sqrt2_v<T> * wet::numbers::inv_sqrt3_v<T>;
        constexpr T inv_sqrt2 = wet::numbers::sqrt2_v<T> / T{2};
        return {
            .alpha = sqrt_2_3 * (abc[0] - ((abc[1] + abc[2]) / T{2})),
            .beta = (abc[1] - abc[2]) * inv_sqrt2,
            .zero = (abc[0] + abc[1] + abc[2]) * wet::numbers::inv_sqrt3_v<T>,
        };
    } else {
        return {
            .alpha = ((T{2} * abc[0]) - abc[1] - abc[2]) / T{3},
            .beta = (abc[1] - abc[2]) * wet::numbers::inv_sqrt3_v<T>,
            .zero = (abc[0] + abc[1] + abc[2]) / T{3},
        };
    }
}

/**
 * @brief Inverse zero-retaining Clarke transform (αβ0 → abc)
 * @ingroup transforms
 *
 * Exact inverse of clarke_zero_transform(); the zero channel is added back to
 * every phase:
 * @f[
 *   a = \alpha + v_0, \quad
 *   b = -\tfrac{\alpha}{2} + \tfrac{\sqrt{3}}{2}\beta + v_0, \quad
 *   c = -\tfrac{\alpha}{2} - \tfrac{\sqrt{3}}{2}\beta + v_0
 * @f]
 *
 * MATLAB: `invclarke()` with the zero row
 *
 * @param abz αβ0 components
 * @return Phase quantities {a, b, c}
 */
template<typename T = float, Convention C = Convention::AmplitudeInvariant>
[[nodiscard]] constexpr ColVec<3, T> inverse_clarke_zero_transform(const AlphaBetaZero<T>& abz) {
    if constexpr (C == Convention::PowerInvariant) {
        // Inverse of the orthonormal forward transform is its transpose.
        constexpr T sqrt_2_3 = wet::numbers::sqrt2_v<T> * wet::numbers::inv_sqrt3_v<T>;
        constexpr T inv_sqrt2 = wet::numbers::sqrt2_v<T> / T{2};
        constexpr T inv_sqrt6 = inv_sqrt2 * wet::numbers::inv_sqrt3_v<T>;
        const T     z = abz.zero * wet::numbers::inv_sqrt3_v<T>;
        return {
            (sqrt_2_3 * abz.alpha) + z,
            (-inv_sqrt6 * abz.alpha) + (inv_sqrt2 * abz.beta) + z,
            (-inv_sqrt6 * abz.alpha) - (inv_sqrt2 * abz.beta) + z,
        };
    } else {
        const T half_sqrt3_beta = wet::numbers::sqrt3_v<T> * abz.beta / T{2};
        return {
            abz.alpha + abz.zero,
            (-abz.alpha / T{2}) + half_sqrt3_beta + abz.zero,
            (-abz.alpha / T{2}) - half_sqrt3_beta + abz.zero,
        };
    }
}

/**
 * @brief Clarke transform (abc → αβ)
 * @ingroup transforms
 *
 * The three-wire (balanced / zero-sequence-free) projection: the αβ slice of
 * clarke_zero_transform(). The zero channel is computed and discarded, which the
 * optimiser eliminates wherever it is unused.
 * @f[
 *   \alpha = \tfrac{2a - b - c}{3}, \qquad \beta = \frac{b - c}{\sqrt{3}}
 * @f]
 *
 * MATLAB: `clarke()` / `[alpha; beta] = (2/3) * [1 -1/2 -1/2; 0 √3/2 -√3/2] * [a;b;c]`
 *
 * @param abc Phase quantities {a, b, c}
 * @return αβ components
 */
template<typename T = float, Convention C = Convention::AmplitudeInvariant>
[[nodiscard]] constexpr AlphaBeta<T> clarke_transform(const ColVec<3, T>& abc) {
    return clarke_zero_transform<T, C>(abc).ab();
}

/**
 * @brief Inverse Clarke transform (αβ → abc)
 * @ingroup transforms
 *
 * The zero-sequence-free inverse: inverse_clarke_zero_transform() with
 * @f$ v_0 = 0 @f$.
 * @f[
 *   a = \alpha, \quad
 *   b = -\tfrac{\alpha}{2} + \tfrac{\sqrt{3}}{2}\beta, \quad
 *   c = -\tfrac{\alpha}{2} - \tfrac{\sqrt{3}}{2}\beta
 * @f]
 *
 * MATLAB: `invclarke()`
 *
 * @param ab αβ components
 * @return Phase quantities {a, b, c}
 */
template<typename T = float, Convention C = Convention::AmplitudeInvariant>
[[nodiscard]] constexpr ColVec<3, T> inverse_clarke_transform(const AlphaBeta<T>& ab) {
    return inverse_clarke_zero_transform<T, C>(AlphaBetaZero<T>{ab.alpha, ab.beta, T{0}});
}

/**
 * @brief Park transform (αβ → dq)
 * @ingroup transforms
 *
 * Rotates the stationary αβ frame into the rotor-synchronous dq frame:
 * @f[
 *   d =  \alpha\cos\theta + \beta\sin\theta, \qquad
 *   q = -\alpha\sin\theta + \beta\cos\theta
 * @f]
 *
 * MATLAB: `park()`
 *
 * @param ab    αβ components
 * @param theta Rotor electrical angle [rad]
 * @return dq components
 */
template<typename T = float>
[[nodiscard]] constexpr DirectQuadrature<T> park_transform(const AlphaBeta<T>& ab, T theta) {
    const auto [sin_theta, cos_theta] = wet::sincos(theta);

    const T d = (ab.alpha * cos_theta) + (ab.beta * sin_theta);
    const T q = (-ab.alpha * sin_theta) + (ab.beta * cos_theta);

    return {d, q};
}

/**
 * @brief Inverse Park transform (dq → αβ)
 * @ingroup transforms
 *
 * @f[
 *   \alpha = d\cos\theta - q\sin\theta, \qquad
 *   \beta  = d\sin\theta + q\cos\theta
 * @f]
 *
 * MATLAB: `invpark()`
 *
 * @param dq    dq components
 * @param theta Rotor electrical angle [rad]
 * @return αβ components
 */
template<typename T = float>
[[nodiscard]] constexpr AlphaBeta<T> inverse_park_transform(const DirectQuadrature<T>& dq, T theta) {
    const auto [sin_theta, cos_theta] = wet::sincos(theta);

    const T alpha = (dq.d * cos_theta) - (dq.q * sin_theta);
    const T beta = (dq.d * sin_theta) + (dq.q * cos_theta);

    return {alpha, beta};
}

/**
 * @brief Fused Clarke-Park transform (abc → dq)
 * @ingroup transforms
 *
 * Maps three-phase stationary quantities directly to the rotor frame; the usual
 * measurement-side step of an FOC loop. Algebraically identical to
 * `park_transform(clarke_transform(abc), theta)`, but evaluated in a single pass:
 * one sincos() call and no intermediate αβ result. The fusion is explicit rather
 * than left to the optimiser, so the operation count is deterministic without
 * `-ffast-math` (which the FP-reassociation needed to collapse the two-stage form
 * otherwise depends on).
 * @f[
 *   \alpha = \tfrac{2a - b - c}{3}, \quad \beta = \frac{b - c}{\sqrt{3}}, \qquad
 *   d =  \alpha\cos\theta + \beta\sin\theta, \quad
 *   q = -\alpha\sin\theta + \beta\cos\theta
 * @f]
 *
 * @tparam C     Scaling convention (default amplitude-invariant)
 * @param abc   Phase quantities {a, b, c}
 * @param theta Rotor electrical angle [rad]
 * @return dq components
 */
template<typename T = float, Convention C = Convention::AmplitudeInvariant>
[[nodiscard]] constexpr DirectQuadrature<T> clarke_park_transform(const ColVec<3, T>& abc, T theta) {
    const auto [sin_theta, cos_theta] = wet::sincos(theta);
    const auto ab = clarke_zero_transform<T, C>(abc).ab();

    return {
        .d = (ab.alpha * cos_theta) + (ab.beta * sin_theta),
        .q = (-ab.alpha * sin_theta) + (ab.beta * cos_theta),
    };
}

/**
 * @brief Fused inverse Park-Clarke transform (dq → abc)
 * @ingroup transforms
 *
 * Maps rotor-frame quantities directly to three phases; the usual command-side
 * step of an FOC loop. Algebraically identical to
 * `inverse_clarke_transform(inverse_park_transform(dq, theta))`, but evaluated in
 * a single pass: one sincos() call and no intermediate αβ result, so the
 * operation count is deterministic without `-ffast-math`.
 * @f[
 *   \alpha = d\cos\theta - q\sin\theta, \quad \beta = d\sin\theta + q\cos\theta, \qquad
 *   a = \alpha, \quad
 *   b = -\tfrac{\alpha}{2} + \tfrac{\sqrt{3}}{2}\beta, \quad
 *   c = -\tfrac{\alpha}{2} - \tfrac{\sqrt{3}}{2}\beta
 * @f]
 *
 * @tparam C     Scaling convention (default amplitude-invariant)
 * @param dq    dq components
 * @param theta Rotor electrical angle [rad]
 * @return Phase quantities {a, b, c}
 */
template<typename T = float, Convention C = Convention::AmplitudeInvariant>
[[nodiscard]] constexpr ColVec<3, T> inverse_park_clarke_transform(const DirectQuadrature<T>& dq, T theta) {
    const auto [sin_theta, cos_theta] = wet::sincos(theta);

    const T alpha = (dq.d * cos_theta) - (dq.q * sin_theta);
    const T beta = (dq.d * sin_theta) + (dq.q * cos_theta);

    return inverse_clarke_transform<T, C>(AlphaBeta<T>{alpha, beta});
}

/**
 * @brief Park transform with zero passthrough (αβ0 → dq0)
 * @ingroup transforms
 *
 * Rotates the αβ part into the rotor frame and carries the zero channel through
 * unchanged (the rotation does not couple to the zero axis). The rank-3 partner
 * to park_transform(), for four-wire dq0 control.
 *
 * @param abz   αβ0 components
 * @param theta Rotor electrical angle [rad]
 * @return dq0 components
 */
template<typename T = float>
[[nodiscard]] constexpr DirectQuadratureZero<T> park_zero_transform(const AlphaBetaZero<T>& abz, T theta) {
    const auto dq = park_transform(abz.ab(), theta);
    return {dq.d, dq.q, abz.zero};
}

/**
 * @brief Inverse Park transform with zero passthrough (dq0 → αβ0)
 * @ingroup transforms
 *
 * @param dqz   dq0 components
 * @param theta Rotor electrical angle [rad]
 * @return αβ0 components
 */
template<typename T = float>
[[nodiscard]] constexpr AlphaBetaZero<T> inverse_park_zero_transform(const DirectQuadratureZero<T>& dqz, T theta) {
    const auto ab = inverse_park_transform(dqz.dq(), theta);
    return {ab.alpha, ab.beta, dqz.zero};
}

/**
 * @brief Fused Clarke-Park transform with zero (abc → dq0)
 * @ingroup transforms
 *
 * Rank-3 measurement-side transform for four-wire converters: abc directly to
 * dq0, one sincos() call. The zero channel is the Clarke zero passed through the
 * rotation unchanged.
 *
 * @tparam C     Scaling convention (default amplitude-invariant)
 * @param abc   Phase quantities {a, b, c}
 * @param theta Rotor electrical angle [rad]
 * @return dq0 components
 */
template<typename T = float, Convention C = Convention::AmplitudeInvariant>
[[nodiscard]] constexpr DirectQuadratureZero<T> clarke_park_zero_transform(const ColVec<3, T>& abc, T theta) {
    return park_zero_transform(clarke_zero_transform<T, C>(abc), theta);
}

/**
 * @brief Fused inverse Park-Clarke transform with zero (dq0 → abc)
 * @ingroup transforms
 *
 * @tparam C     Scaling convention (default amplitude-invariant)
 * @param dqz   dq0 components
 * @param theta Rotor electrical angle [rad]
 * @return Phase quantities {a, b, c}
 */
template<typename T = float, Convention C = Convention::AmplitudeInvariant>
[[nodiscard]] constexpr ColVec<3, T> inverse_park_clarke_zero_transform(const DirectQuadratureZero<T>& dqz, T theta) {
    return inverse_clarke_zero_transform<T, C>(inverse_park_zero_transform(dqz, theta));
}

/**
 * @brief Instantaneous active and reactive power
 * @ingroup transforms
 *
 * Three-phase instantaneous power by the Akagi p–q theory. From the αβ voltage
 * and current vectors:
 * @f[
 *   p = k\,(v_\alpha i_\alpha + v_\beta i_\beta), \qquad
 *   q = k\,(v_\beta i_\alpha - v_\alpha i_\beta)
 * @f]
 * @f$ p @f$ is the real power transferred; @f$ q @f$ is the (Akagi) imaginary
 * power that circulates between phases without net transfer. The scale factor
 * @f$ k @f$ depends on the αβ convention: @f$ k = 3/2 @f$ for amplitude-invariant
 * (so this returns true three-phase watts/VAr), @f$ k = 1 @f$ for power-invariant.
 *
 * @tparam C Scaling convention of the αβ inputs (default amplitude-invariant)
 *
 * @see H. Akagi, Y. Kanazawa, A. Nabae, "Instantaneous reactive power
 *      compensators comprising switching devices without energy storage
 *      components," IEEE Trans. Ind. Appl., vol. IA-20, no. 3, pp. 625-630, 1984.
 */
template<typename T = float>
struct InstantaneousPower {
    T p = {}; ///< [W]   instantaneous active (real) power
    T q = {}; ///< [VAr] instantaneous reactive (Akagi imaginary) power
};

/// @copydoc InstantaneousPower
template<typename T = float, Convention C = Convention::AmplitudeInvariant>
[[nodiscard]] constexpr InstantaneousPower<T> instantaneous_power(const AlphaBeta<T>& v, const AlphaBeta<T>& i) {
    constexpr T k = (C == Convention::PowerInvariant) ? T{1} : T{1.5};
    return {
        .p = k * ((v.alpha * i.alpha) + (v.beta * i.beta)),
        .q = k * ((v.beta * i.alpha) - (v.alpha * i.beta)),
    };
}

/**
 * @brief Instantaneous active and reactive power from dq quantities
 * @ingroup transforms
 *
 * The rotor/synchronous-frame form of instantaneous_power():
 * @f[
 *   p = k\,(v_d i_d + v_q i_q), \qquad q = k\,(v_q i_d - v_d i_q)
 * @f]
 * Equivalent to the αβ form (both are frame-invariant scalars); use whichever
 * frame the signals are already in.
 *
 * @tparam C Scaling convention of the dq inputs (default amplitude-invariant)
 */
template<typename T = float, Convention C = Convention::AmplitudeInvariant>
[[nodiscard]] constexpr InstantaneousPower<T> instantaneous_power(const DirectQuadrature<T>& v, const DirectQuadrature<T>& i) {
    constexpr T k = (C == Convention::PowerInvariant) ? T{1} : T{1.5};
    return {
        .p = k * ((v.d * i.d) + (v.q * i.q)),
        .q = k * ((v.q * i.d) - (v.d * i.q)),
    };
}

/**
 * @brief Symmetrical (sequence) components of a three-phase phasor set
 * @ingroup transforms
 *
 * The Fortescue decomposition resolves an unbalanced set of three complex phasors
 * into three balanced sets: zero, positive, and negative sequence. Used for
 * unbalanced-grid / fault analysis and for sequence-domain current control.
 *
 * @tparam T Scalar type
 */
template<typename T = float>
struct SequenceComponents {
    complex<T> zero = {};     ///< Zero-sequence phasor V₀ (co-phasal)
    complex<T> positive = {}; ///< Positive-sequence phasor V₁ (abc rotation)
    complex<T> negative = {}; ///< Negative-sequence phasor V₂ (acb rotation)
};

/**
 * @brief Forward symmetrical-component (Fortescue) transform (abc → 012)
 * @ingroup transforms
 *
 * Decomposes three complex phase phasors into zero/positive/negative sequence
 * with the operator @f$ a = e^{j2\pi/3} @f$:
 * @f[
 *   \begin{bmatrix} V_0 \\ V_1 \\ V_2 \end{bmatrix}
 *   = \frac{1}{3}
 *   \begin{bmatrix} 1 & 1 & 1 \\ 1 & a & a^2 \\ 1 & a^2 & a \end{bmatrix}
 *   \begin{bmatrix} V_a \\ V_b \\ V_c \end{bmatrix}
 * @f]
 *
 * MATLAB: `sequence()` / `fortescue([Va; Vb; Vc])`
 *
 * @param abc Three-phase phasors {Va, Vb, Vc}
 * @return Sequence components {zero, positive, negative}
 *
 * @see https://en.wikipedia.org/wiki/Symmetrical_components
 * @see C. L. Fortescue, "Method of symmetrical co-ordinates applied to the
 *      solution of polyphase networks," Trans. AIEE, vol. 37, no. 2, pp.
 *      1027-1140, 1918.
 */
template<typename T = float>
[[nodiscard]] constexpr SequenceComponents<T> symmetrical_components(const ColVec<3, complex<T>>& abc) {
    // Rotation operator a = e^{j120°} and its square a² = e^{j240°} = e^{-j120°}.
    constexpr complex<T> a = {T{-0.5}, wet::numbers::sqrt3_v<T> / T{2}};
    constexpr complex<T> a2 = {T{-0.5}, -wet::numbers::sqrt3_v<T> / T{2}};

    return {
        .zero = (abc[0] + abc[1] + abc[2]) / T{3},
        .positive = (abc[0] + (a * abc[1]) + (a2 * abc[2])) / T{3},
        .negative = (abc[0] + (a2 * abc[1]) + (a * abc[2])) / T{3},
    };
}

/**
 * @brief Inverse symmetrical-component transform (012 → abc)
 * @ingroup transforms
 *
 * Recombines sequence components back into phase phasors:
 * @f[
 *   \begin{bmatrix} V_a \\ V_b \\ V_c \end{bmatrix}
 *   = \begin{bmatrix} 1 & 1 & 1 \\ 1 & a^2 & a \\ 1 & a & a^2 \end{bmatrix}
 *   \begin{bmatrix} V_0 \\ V_1 \\ V_2 \end{bmatrix}
 * @f]
 *
 * MATLAB: `isequence()` / `[Va; Vb; Vc] = A * [V0; V1; V2]`
 *
 * @param s Sequence components {zero, positive, negative}
 * @return Three-phase phasors {Va, Vb, Vc}
 *
 * @see https://en.wikipedia.org/wiki/Symmetrical_components
 */
template<typename T = float>
[[nodiscard]] constexpr ColVec<3, complex<T>> inverse_symmetrical_components(const SequenceComponents<T>& s) {
    constexpr complex<T> a = {T{-0.5}, wet::numbers::sqrt3_v<T> / T{2}};
    constexpr complex<T> a2 = {T{-0.5}, -wet::numbers::sqrt3_v<T> / T{2}};

    return {
        s.zero + s.positive + s.negative,
        s.zero + (a2 * s.positive) + (a * s.negative),
        s.zero + (a * s.positive) + (a2 * s.negative),
    };
}

} // namespace wet
