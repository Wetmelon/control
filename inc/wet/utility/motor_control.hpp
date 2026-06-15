#pragma once

#include "wet/backend.hpp"
#include "wet/math/math.hpp"
#include "wet/matrix/colvec.hpp"

namespace wet {

/**
 * @defgroup motor_control Motor Control Transforms
 * @brief Coordinate transformations and modulation for three-phase motor control
 *
 * Clarke, Park, and their inverses map between the three reference frames used
 * in field-oriented control (FOC):
 *
 *   - **abc**  three-phase stationary frame (a column vector of phase quantities)
 *   - **αβ**   two-phase stationary frame (wet::AlphaBeta)
 *   - **dq**   rotor-synchronous rotating frame (wet::DirectQuadrature)
 *
 * Phase quantities live in a wet::ColVec<3, T> so they compose with the rest of
 * the linear-algebra library; the αβ and dq pairs use named structs so the two
 * orthogonal components never get silently swapped.
 *
 * All transforms use the amplitude-invariant (2/3) convention, so the αβ and dq
 * magnitudes equal the peak phase amplitude.
 *
 * @see https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_transformation
 * @see https://en.wikipedia.org/wiki/Direct-quadrature-zero_transformation
 */

/**
 * @brief Direct-quadrature (rotor-frame) component pair
 * @ingroup motor_control
 */
template<typename T = float>
struct DirectQuadrature {
    T d, q;
};

/**
 * @brief Alpha-beta (stationary-frame) component pair
 * @ingroup motor_control
 */
template<typename T = float>
struct AlphaBeta {
    T alpha, beta;
};

/**
 * @brief Clarke transform (abc → αβ)
 * @ingroup motor_control
 *
 * Projects three-phase stationary quantities onto the two-phase stationary αβ
 * frame using the amplitude-invariant convention:
 * @f[
 *   \alpha = \tfrac{2a - b - c}{3}, \qquad \beta = \frac{b - c}{\sqrt{3}}
 * @f]
 *
 * MATLAB: `clarke()` / `[alpha; beta] = (2/3) * [1 -1/2 -1/2; 0 √3/2 -√3/2] * [a;b;c]`
 *
 * @param abc Phase quantities {a, b, c}
 * @return αβ components
 */
template<typename T = float>
[[nodiscard]] constexpr AlphaBeta<T> clarke_transform(const ColVec<3, T>& abc) {
    const T alpha = ((T{2} * abc[0]) - abc[1] - abc[2]) / T{3};
    const T beta = (abc[1] - abc[2]) * wet::numbers::inv_sqrt3_v<T>;
    return {alpha, beta};
}

/**
 * @brief Inverse Clarke transform (αβ → abc)
 * @ingroup motor_control
 *
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
template<typename T = float>
[[nodiscard]] constexpr ColVec<3, T> inverse_clarke_transform(const AlphaBeta<T>& ab) {
    const T half_sqrt3_beta = wet::numbers::sqrt3_v<T> * ab.beta / T{2};
    return {
        ab.alpha,
        (-ab.alpha / T{2}) + half_sqrt3_beta,
        (-ab.alpha / T{2}) - half_sqrt3_beta,
    };
}

/**
 * @brief Park transform (αβ → dq)
 * @ingroup motor_control
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
 * @ingroup motor_control
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
 * @ingroup motor_control
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
 * @param abc   Phase quantities {a, b, c}
 * @param theta Rotor electrical angle [rad]
 * @return dq components
 */
template<typename T = float>
[[nodiscard]] constexpr DirectQuadrature<T> clarke_park_transform(const ColVec<3, T>& abc, T theta) {
    const auto [sin_theta, cos_theta] = wet::sincos(theta);

    const T alpha = ((T{2} * abc[0]) - abc[1] - abc[2]) / T{3};
    const T beta = (abc[1] - abc[2]) * wet::numbers::inv_sqrt3_v<T>;

    return {
        .d = (alpha * cos_theta) + (beta * sin_theta),
        .q = (-alpha * sin_theta) + (beta * cos_theta),
    };
}

/**
 * @brief Fused inverse Park-Clarke transform (dq → abc)
 * @ingroup motor_control
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
 * @param dq    dq components
 * @param theta Rotor electrical angle [rad]
 * @return Phase quantities {a, b, c}
 */
template<typename T = float>
[[nodiscard]] constexpr ColVec<3, T> inverse_park_clarke_transform(const DirectQuadrature<T>& dq, T theta) {
    const auto [sin_theta, cos_theta] = wet::sincos(theta);

    const T alpha = (dq.d * cos_theta) - (dq.q * sin_theta);
    const T beta = (dq.d * sin_theta) + (dq.q * cos_theta);

    const T half_sqrt3_beta = wet::numbers::sqrt3_v<T> * beta / T{2};
    return {
        alpha,
        (-alpha / T{2}) + half_sqrt3_beta,
        (-alpha / T{2}) - half_sqrt3_beta,
    };
}

/**
 * @brief Min-max zero-sequence injection for space-vector PWM
 * @ingroup motor_control
 *
 * Returns the common-mode (zero-sequence) offset that, added equally to all
 * three phase references, centres them within the available bus and yields
 * continuous space-vector modulation (SVPWM). This is the offset that extends
 * the linear modulation range by 2/√3 (≈15.5%) over sinusoidal PWM without
 * affecting the line-to-line voltages.
 * @f[
 *   v_0 = -\frac{\max(v_a, v_b, v_c) + \min(v_a, v_b, v_c)}{2}
 * @f]
 *
 * @param v_abc Three-phase voltage references [V]
 * @return Zero-sequence offset to add to every phase [V]
 *
 * @see https://en.wikipedia.org/wiki/Space_vector_modulation
 */
template<typename T = float>
[[nodiscard]] constexpr T svpwm_zero_sequence(const ColVec<3, T>& v_abc) {
    const T v_max = wet::max({v_abc[0], v_abc[1], v_abc[2]});
    const T v_min = wet::min({v_abc[0], v_abc[1], v_abc[2]});
    return -(v_max + v_min) / T{2};
}

/**
 * @brief Space-vector PWM duty cycles from an αβ voltage command
 * @ingroup motor_control
 *
 * Resolves the αβ voltage command to phase voltages (inverse Clarke), applies
 * min-max zero-sequence injection (svpwm_zero_sequence()) to realise SVPWM, then
 * maps each phase to a half-bridge duty cycle referenced to the Vdc/2 bus
 * midpoint:
 * @f[
 *   d_x = \frac{1}{2} + \frac{v_x + v_0}{V_{dc}}, \qquad x \in \{a, b, c\}
 * @f]
 *
 * Voltages are in volts (the same units the current controller produces), so no
 * separate normalisation step is needed. Results are clamped to [0, 1]; clamping
 * only engages in over-modulation.
 *
 * @param v_ab αβ voltage command [V]
 * @param v_dc DC bus voltage [V]
 * @return Half-bridge duty cycles {a, b, c}, each in [0, 1]
 */
template<typename T = float>
[[nodiscard]] constexpr ColVec<3, T> svm_duty_cycles(const AlphaBeta<T>& v_ab, T v_dc) {
    const ColVec<3, T> v_phase = inverse_clarke_transform(v_ab);
    const T            v_0 = svpwm_zero_sequence(v_phase);

    ColVec<3, T> duty;
    for (size_t i = 0; i < 3; ++i) {
        duty[i] = wet::clamp(T{0.5} + ((v_phase[i] + v_0) / v_dc), T{0}, T{1});
    }
    return duty;
}

} // namespace wet
