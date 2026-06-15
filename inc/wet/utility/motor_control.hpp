#pragma once

#include "wet/matrix/colvec.hpp"
#include "wet/utility/transforms.hpp"

namespace wet {

/**
 * @defgroup motor_control Motor Control Modulation
 * @brief Space-vector modulation for three-phase inverters
 *
 * Maps an αβ voltage command to inverter half-bridge duty cycles. The reference-
 * frame transforms (Clarke, Park, symmetrical components) that feed these live in
 * @ref transforms.
 *
 * @see https://en.wikipedia.org/wiki/Space_vector_modulation
 */

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
 * @see A. M. Hava, R. J. Kerkman, T. A. Lipo, "Simple analytical and graphical
 *      methods for carrier-based PWM-VSI drives," IEEE Trans. Power Electron.,
 *      vol. 14, no. 1, pp. 49-61, 1999. doi:10.1109/63.737592
 *      (establishes the carrier-based min-max injection ⇔ SVPWM equivalence).
 * @see D. G. Holmes, T. A. Lipo, "Pulse Width Modulation for Power Converters:
 *      Principles and Practice," IEEE Press, 2003, ch. 3.
 */
template<typename T = float>
[[nodiscard]] constexpr T svpwm_zero_sequence(const ColVec<3, T>& v_abc) {
    const T v_max = wet::max({v_abc[0], v_abc[1], v_abc[2]});
    const T v_min = wet::min({v_abc[0], v_abc[1], v_abc[2]});
    return -(v_max + v_min) / T{2};
}

/**
 * @brief Result of svm_duty_cycles(): the half-bridge duties plus an
 *        over-modulation flag.
 * @ingroup motor_control
 *
 * The duties are always a usable command (clamped to [0, 1]); is_clipped reports
 * whether that clamping engaged, i.e. the requested voltage exceeded the
 * realizable hexagon. It is advisory, not a failure — hence a flag rather than a
 * wet::optional / wet::expected wrapper.
 *
 * @tparam T Scalar type
 */
template<typename T = float>
struct SvmDuties {
    ColVec<3, T> duties = {};        ///< [pu] half-bridge duties {a, b, c}, each in [0, 1]
    bool         is_clipped = false; ///< true if any duty was clamped (over-modulation)
};

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
 * @return SvmDuties: the {a, b, c} duties (each in [0, 1]) and an is_clipped flag.
 *         The duties are always valid (clamped); is_clipped just reports whether
 *         the command fell outside the realizable hexagon (over-modulation).
 *
 * @see https://en.wikipedia.org/wiki/Space_vector_modulation
 * @see A. M. Hava, R. J. Kerkman, T. A. Lipo, "Simple analytical and graphical
 *      methods for carrier-based PWM-VSI drives," IEEE Trans. Power Electron.,
 *      vol. 14, no. 1, pp. 49-61, 1999. doi:10.1109/63.737592
 */
template<typename T = float>
[[nodiscard]] constexpr SvmDuties<T> svm_duty_cycles(const AlphaBeta<T>& v_ab, T v_dc) {
    const ColVec<3, T> v_phase = inverse_clarke_transform(v_ab);
    const T            v_0 = svpwm_zero_sequence(v_phase);

    SvmDuties<T> result;
    for (size_t i = 0; i < 3; ++i) {
        const T raw = T{0.5} + ((v_phase[i] + v_0) / v_dc);
        const T sat = wet::clamp(raw, T{0}, T{1});
        result.is_clipped = result.is_clipped || (sat != raw);
        result.duties[i] = sat;
    }
    return result;
}

} // namespace wet
