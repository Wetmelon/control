#pragma once

#include "wet/filters/sogi.hpp"
#include "wet/math/math.hpp"
#include "wet/utility/transforms.hpp"

namespace wet {

/**
 * @brief Instantaneous positive-sequence αβ from a quadrature signal pair
 * @ingroup transforms
 *
 * The αβ-domain analogue of the (phasor) Fortescue transform. Given the αβ vector
 * @p v and its 90°-lagged quadrature @p qv (the `q = e^{-j\pi/2}` operator,
 * supplied by a SOGI/MSTOGI quadrature-signal generator), the instantaneous
 * positive-sequence component is
 * @f[
 *   \begin{bmatrix} v_\alpha^+ \\ v_\beta^+ \end{bmatrix}
 *   = \frac{1}{2}\begin{bmatrix} 1 & -q \\ q & 1 \end{bmatrix}
 *     \begin{bmatrix} v_\alpha \\ v_\beta \end{bmatrix}
 *   = \frac{1}{2}\begin{bmatrix} v_\alpha - q v_\beta \\ q v_\alpha + v_\beta \end{bmatrix}
 * @f]
 *
 * @param v  αβ vector
 * @param qv 90°-lagged quadrature of @p v (per-axis SOGI quadrature output)
 * @return Instantaneous positive-sequence αβ
 *
 * @see https://en.wikipedia.org/wiki/Symmetrical_components
 * @see P. Rodríguez et al., "Decoupled double synchronous reference frame PLL for
 *      power converters control," IEEE Trans. Power Electron., vol. 22, no. 2,
 *      pp. 584-592, 2007. doi:10.1109/TPEL.2006.890000
 */
template<typename T = float>
[[nodiscard]] constexpr AlphaBeta<T> positive_sequence_ab(const AlphaBeta<T>& v, const AlphaBeta<T>& qv) {
    return {T{0.5} * (v.alpha - qv.beta), T{0.5} * (qv.alpha + v.beta)};
}

/**
 * @brief Instantaneous negative-sequence αβ from a quadrature signal pair
 * @ingroup transforms
 *
 * Companion to positive_sequence_ab():
 * @f[
 *   \begin{bmatrix} v_\alpha^- \\ v_\beta^- \end{bmatrix}
 *   = \frac{1}{2}\begin{bmatrix} 1 & q \\ -q & 1 \end{bmatrix}
 *     \begin{bmatrix} v_\alpha \\ v_\beta \end{bmatrix}
 *   = \frac{1}{2}\begin{bmatrix} v_\alpha + q v_\beta \\ -q v_\alpha + v_\beta \end{bmatrix}
 * @f]
 *
 * @param v  αβ vector
 * @param qv 90°-lagged quadrature of @p v
 * @return Instantaneous negative-sequence αβ
 */
template<typename T = float>
[[nodiscard]] constexpr AlphaBeta<T> negative_sequence_ab(const AlphaBeta<T>& v, const AlphaBeta<T>& qv) {
    return {T{0.5} * (v.alpha + qv.beta), T{0.5} * (-qv.alpha + v.beta)};
}

/**
 * @brief Dual-SOGI three-phase positive-sequence PLL (DSOGI-PLL)
 *
 * Grid synchronization for three-phase systems, robust to unbalance and
 * distortion. Two quadrature-signal generators (one per αβ axis) feed an
 * instantaneous positive-sequence calculator (positive_sequence_ab()); a
 * synchronous-reference-frame (SRF) PLL then locks to the extracted positive
 * sequence, driving its q-axis projection to zero.
 *
 * Pipeline per step (input is the αβ grid vector, e.g. clarke_transform(v_abc)):
 *   1. SOGI-QSG on @f$ v_\alpha @f$ and @f$ v_\beta @f$ → in-phase + quadrature.
 *   2. Positive-sequence αβ via the `q`-operator matrix.
 *   3. Park to dq on the current phase estimate; the q-axis is the phase error.
 *   4. PI loop filter → frequency, integrated to phase.
 *
 * The quadrature generator is a template parameter: the default SOGI is adequate
 * for clean inputs; substitute MSTOGI when the αβ signals carry DC offset (its
 * extra TOGI state rejects DC in the quadrature channel).
 *
 * @note Like SinglePhasePLL this is a tracker, not a SISOController: its entry
 *       point is `step(const AlphaBeta&, T Ts)` returning void, with state read
 *       via accessors. The negative-sequence component is also exposed for
 *       unbalance monitoring / sequence-domain control.
 *
 * @tparam T         Scalar type
 * @tparam Resonator Quadrature-signal generator (SOGI or MSTOGI)
 *
 * @see P. Rodríguez et al., "Decoupled double synchronous reference frame PLL for
 *      power converters control," IEEE Trans. Power Electron., 22(2), 2007.
 */
template<typename T = float, template<typename> class Resonator = SOGI>
class DsogiPll {
public:
    struct Parameters {
        T Kp{};             ///< SRF-PLL proportional gain
        T Ki{};             ///< SRF-PLL integral gain
        T sogi_gain{};      ///< SOGI damping gain (√2 ≈ unity-Q)
        T integrator_max{}; ///< [Hz] anti-windup limit (max)
        T integrator_min{}; ///< [Hz] anti-windup limit (min)
        T output_max{};     ///< [Hz] frequency output clamp (max)
        T output_min{};     ///< [Hz] frequency output clamp (min)
    } params{};

    /**
     * @param f_nom Nominal grid frequency [Hz]
     * @param Ts    Sample time [s]
     */
    constexpr DsogiPll(T f_nom, T Ts)
        : nominal_frequency(f_nom), frequency_estimate(f_nom) {
        const T max_deviation = f_nom * T{0.5};
        params.integrator_max = max_deviation;
        params.integrator_min = -max_deviation;
        params.output_max = f_nom + max_deviation;
        params.output_min = f_nom - max_deviation;
        params.Kp = T{10} * Ts;
        params.Ki = T{100} * Ts;
        params.sogi_gain = wet::numbers::sqrt2_v<T>;
    }

    /// One synchronization step from the stationary-frame αβ grid vector.
    constexpr void step(const AlphaBeta<T>& v_ab, const T Ts) {
        // Per-axis quadrature signal generation at the current frequency estimate.
        const auto [va, qva] = sogi_alpha(v_ab.alpha, frequency_estimate, params.sogi_gain, Ts);
        const auto [vb, qvb] = sogi_beta(v_ab.beta, frequency_estimate, params.sogi_gain, Ts);

        const AlphaBeta<T> v = {va, vb};
        const AlphaBeta<T> qv = {qva, qvb};
        positive_ab_ = positive_sequence_ab(v, qv);
        negative_ab_ = negative_sequence_ab(v, qv);

        // SRF-PLL: q-axis projection of the positive sequence is the phase error
        // (v_q = |v⁺|·sin(φ − θ̂) → 0 at lock, with v_d → +|v⁺|).
        positive_dq_ = park_transform(positive_ab_, phase_estimate);
        const T phase_error = positive_dq_.q;

        // PI loop filter with anti-windup → frequency estimate [Hz].
        integrator_state += params.Ki * phase_error * Ts;
        integrator_state = wet::clamp(integrator_state, params.integrator_min, params.integrator_max);
        frequency_estimate = nominal_frequency + (params.Kp * phase_error) + integrator_state;
        frequency_estimate = wet::clamp(frequency_estimate, params.output_min, params.output_max);

        // Integrate frequency to phase, wrapped to [0, 2π).
        const T two_pi = T{2} * wet::numbers::pi_v<T>;
        phase_estimate += two_pi * frequency_estimate * Ts;
        phase_estimate = wet::fmod(phase_estimate, two_pi);
        if (phase_estimate < T{0}) {
            phase_estimate += two_pi;
        }
    }

    [[nodiscard]] constexpr T                   frequency() const { return frequency_estimate; }
    [[nodiscard]] constexpr T                   phase() const { return phase_estimate; }
    [[nodiscard]] constexpr AlphaBeta<T>        positive_sequence() const { return positive_ab_; }
    [[nodiscard]] constexpr AlphaBeta<T>        negative_sequence() const { return negative_ab_; }
    [[nodiscard]] constexpr DirectQuadrature<T> positive_dq() const { return positive_dq_; }

    constexpr void reset() {
        sogi_alpha = {};
        sogi_beta = {};
        integrator_state = T{0};
        phase_estimate = T{0};
        frequency_estimate = nominal_frequency;
        positive_ab_ = {};
        negative_ab_ = {};
        positive_dq_ = {};
    }

private:
    Resonator<T> sogi_alpha{};
    Resonator<T> sogi_beta{};

    T nominal_frequency{};
    T integrator_state{};
    T phase_estimate{};
    T frequency_estimate{};

    AlphaBeta<T>        positive_ab_{};
    AlphaBeta<T>        negative_ab_{};
    DirectQuadrature<T> positive_dq_{};
};

} // namespace wet
