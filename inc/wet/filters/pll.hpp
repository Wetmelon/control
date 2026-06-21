#pragma once

#include "wet/controllers/pid.hpp"
#include "wet/filters/sogi.hpp"
#include "wet/math/math.hpp"
#include "wet/matrix/colvec.hpp"
#include "wet/utility/transforms.hpp"

namespace wet {

/**
 * @brief Single-Phase PLL
 *
 * Uses a Second-Order Generalized Integrator (SOGI) for quadrature signal generation.
 *
 * The loop filter is a canonical @ref PIController acting on the phase error and
 * producing a frequency *offset* about nominal; the inverter-style output clamp
 * plus back-calculation give proper anti-windup (the integrator is bled back out
 * of saturation rather than merely clipped). The phase is a wrapping integrator —
 * @ref wet::wrap with @f$\pm\pi@f$ bounds keeps it in @f$[-\pi,\pi)@f$ each step (a
 * single round-to-nearest, no runtime divide for constant bounds), which also keeps
 * the argument inside the accurate range of the fast @ref wet math backend's range
 * reduction over long runs.
 *
 * @note A phase-locked loop is a tracker, not a reference-following controller,
 *       and does not satisfy @ref SISOController -- its entry point is
 *       `step(T input, T Ts)` returning `void`, with state read out via member
 *       accessors. Use it as a frequency / phase estimator that feeds *into* a
 *       SISOController (e.g. a current-loop PR controller riding the PLL's
 *       phase estimate), not as a block inside `Cascade<Outer, Inner>`.
 */
template<typename T>
struct SinglePhasePLL {
    /// PI loop filter. Output is the frequency offset [Hz] about nominal; `u_min`
    /// / `u_max` bound that offset (the frequency clamp) and `Kbc` back-calculates
    /// the integrator out of saturation. Tune via its public gains.
    PIController<T> loop_filter{};

    /// [1/s] Integrator bleed rate (0 = pure integrator). The leak pulls the
    /// accumulated frequency offset back toward zero with time constant 1/leak, so
    /// a transient or biased phase error doesn't park a permanent offset.
    T integrator_leak{};

    constexpr explicit SinglePhasePLL(T Fnom)
        : nominal_frequency(Fnom), frequency_estimate(Fnom) {
        // Frequency may deviate up to ±50% of nominal; that bounds the loop-filter
        // output (the offset about nominal).
        const T max_freq_deviation = Fnom * T{0.5};

        // Sample-rate-independent loop-filter gains: step() applies Ts, so the
        // gains themselves carry no Ts. Kbc = Kp is the standard PI tracking
        // constant (T_t = T_i) for back-calculation anti-windup.
        loop_filter.Kp = T{10};
        loop_filter.Ki = T{100};
        loop_filter.Kbc = loop_filter.Kp;
        loop_filter.u_max = max_freq_deviation;
        loop_filter.u_min = -max_freq_deviation;
    }

    constexpr void step(T input, const T Ts) {
        // Generate bandpass and quadrature signals using SOGI.
        const auto [bp, quadrature] = sogi(input, frequency_estimate, wet::numbers::sqrt2_v<T>, Ts);
        (void)bp;

        // Phase error from the SOGI quadrature mixer. qv' lags the input by 90° at
        // the SOGI centre, so for an input above the current estimate the DC of
        // input·qv' is negative — the loop must move the estimate by −(input·qv')
        // to chase it (same sign convention as the SOGI-FLL's ω̇ = −Γ·ε·qv', see
        // SogiFll). The earlier +input·qv' was positive feedback and ran the
        // estimate to a frequency rail.
        const T phase_error = -(input * quadrature);

        // Optional leaky integrator: bleed any parked offset toward zero (skip on a
        // non-positive Ts, which would bleed with the wrong sign).
        if (integrator_leak != T{0} && Ts > T{0}) {
            loop_filter.integral -= integrator_leak * loop_filter.integral * Ts;
        }

        // PI loop filter (with anti-windup) on the phase error → frequency offset.
        // r = phase_error, y = 0 ⇒ output = Kp·phase_error + Ki·∫phase_error.
        frequency_estimate = nominal_frequency + loop_filter.control(phase_error, T{0}, Ts);

        // Wrapping integrator: integrate frequency [Hz] to phase [rad], kept in
        // [-π, π) by the cheap round-to-nearest wrap (no runtime divide here).
        const T pi = wet::numbers::pi_v<T>;
        phase_estimate = wet::wrap(phase_estimate + (T{2} * pi * frequency_estimate * Ts), -pi, pi);
    }

    /// Estimated frequency [Hz].
    [[nodiscard]] constexpr T frequency() const {
        return frequency_estimate;
    }

    /// Estimated phase [rad], wrapped to [-π, π).
    [[nodiscard]] constexpr T phase() const {
        return phase_estimate;
    }

    constexpr void reset() {
        sogi.reset();
        loop_filter.reset();
        phase_estimate = T{0};
        frequency_estimate = nominal_frequency;
    }

private:
    MSTOGI<T> sogi{}; // SOGI for quadrature signal generation

    T nominal_frequency{};  // Center frequency of the PLL bandpass filter
    T phase_estimate{};     // Estimated phase of the input signal
    T frequency_estimate{}; // Estimated frequency of the input signal
};

/**
 * @brief Synchronous-reference-frame (SRF) PLL for balanced three-phase input
 *
 * The textbook three-phase PLL: Clarke-Park the input onto the current phase
 * estimate, then drive the q-axis projection to zero with a PI loop filter.
 * At lock @f$ v_q = |v|\sin(\varphi - \hat\theta) \to 0 @f$ and @f$ v_d \to +|v| @f$.
 *
 * The loop filter is a canonical @ref PIController (same proper-anti-windup and
 * wrapping-integrator treatment as @ref SinglePhasePLL).
 *
 * Assumes a balanced set (no negative- or zero-sequence). For unbalanced or
 * distorted grids use @ref DsogiPll, which adds SOGI-based sequence separation
 * ahead of the same loop filter.
 *
 * @note Like @ref SinglePhasePLL this is a tracker, not a @ref SISOController:
 *       its entry point is `step(abc, Ts)` returning `void`, with state read out
 *       via accessors.
 */
template<typename T>
struct ThreePhasePLL {
    /// PI loop filter; output is the frequency offset [Hz] about nominal. See
    /// @ref SinglePhasePLL::loop_filter.
    PIController<T> loop_filter{};

    /**
     * @param f_nom Nominal frequency [Hz]
     */
    constexpr explicit ThreePhasePLL(T f_nom) : nominal_frequency(f_nom), frequency_estimate(f_nom) {
        const T max_freq_deviation = f_nom * T{0.5};
        loop_filter.Kp = T{10};
        loop_filter.Ki = T{100};
        loop_filter.Kbc = loop_filter.Kp;
        loop_filter.u_max = max_freq_deviation;
        loop_filter.u_min = -max_freq_deviation;
    }

    /**
     * @brief One synchronization step.
     *
     * @param abc [V] Phase voltages (balanced)
     * @param Ts  [s] Sample time
     */
    constexpr void step(const wet::ColVec<3, T>& abc, const T Ts) {
        // q-axis projection on the current phase estimate is the phase error.
        const T phase_error = clarke_park_transform(abc, phase_estimate).q;

        // PI loop filter (with anti-windup) → frequency estimate [Hz].
        frequency_estimate = nominal_frequency + loop_filter.control(phase_error, T{0}, Ts);

        // Wrapping integrator: frequency [Hz] → phase [rad], kept in [-π, π).
        const T pi = wet::numbers::pi_v<T>;
        phase_estimate = wet::wrap(phase_estimate + (T{2} * pi * frequency_estimate * Ts), -pi, pi);
    }

    /// Estimated frequency [Hz].
    [[nodiscard]] constexpr T frequency() const {
        return frequency_estimate;
    }

    /// Estimated phase [rad], wrapped to [-π, π).
    [[nodiscard]] constexpr T phase() const {
        return phase_estimate;
    }

    constexpr void reset() {
        loop_filter.reset();
        phase_estimate = T{0};
        frequency_estimate = nominal_frequency;
    }

private:
    T nominal_frequency{};  // [Hz] Center frequency of the PLL
    T phase_estimate{};     // [rad] Estimated phase of the input signal
    T frequency_estimate{}; // [Hz] Estimated frequency of the input signal
};

/**
 * @brief Sensorless rotor flux/position estimator for a PMSM, with optional
 *        sensor fusion.
 *
 * A nonlinear flux observer in the stationary @f$\alpha\beta@f$ frame estimates
 * the permanent-magnet flux vector; a second-order tracking PLL then locks onto
 * its angle, yielding a smooth electrical angle and speed with no shaft sensor.
 *
 * **Flux observer** (paper eqns 4, 6, 8). With stator voltage @f$v@f$, current
 * @f$i@f$, resistance @f$R@f$, inductance @f$L@f$ and PM flux @f$\lambda@f$, the
 * total flux state @f$x@f$ and the PM-flux estimate @f$\eta@f$ evolve as
 * @f[
 *   \dot x = v - R\,i + \frac{g}{2}\,\frac{\lambda^2 - |\eta|^2}{\lambda^2}\,\eta,
 *   \qquad \eta = x - L\,i ,
 * @f]
 * where @f$g@f$ is the observer gain. The correction term pulls @f$|\eta|@f$ onto
 * the circle of radius @f$\lambda@f$, rejecting the integrator drift a pure
 * back-EMF integrator would accumulate.
 *
 * **Tracking PLL.** Gains follow from the desired bandwidth @f$\omega_b@f$ as
 * @f$k_p = 2\omega_b@f$, @f$k_i = \frac{1}{4} k_p^2@f$ (critically damped). The angle
 * is predicted with the velocity estimate and corrected from the wrapped angle
 * innovation; the velocity is the integral path.
 *
 * **Fusion.** The sensorless angle degrades at low speed (the back-EMF
 * @f$\propto\omega\lambda@f$ vanishes). Passing a measured electrical angle (e.g.
 * interpolated Hall) to the fusing `update` overload blends the two angle
 * innovations by a speed-dependent weight: the sensor dominates below
 * @ref Parameters::fusion_blend_speed and the sensorless estimate above it, a
 * standard complementary handoff. With no sensor (the plain overload) it runs as
 * a pure sensorless estimator.
 *
 * @code
 * SensorlessEstimator<float> est{{.phase_resistance = 0.05f, .phase_inductance = 20e-6f,
 *                                 .pm_flux_linkage = 1.6e-3f, .observer_gain = 1000.0f,
 *                                 .pll_bandwidth = 1000.0f, .pole_pairs = 7.0f}};
 * // sensorless:
 * est.update(clarke_transform(i_abc), v_ab_applied, Ts);
 * float theta = est.phase();                  // [elec rad]
 * float w_mech = est.mechanical_velocity();   // [mech rad/s]
 * @endcode
 *
 * @note Like the other estimators here this is a tracker, not a @ref
 *       SISOController. `update()` returns false (and resets) if the discrete PLL
 *       gain is unstable for the supplied @p Ts (@f$k_p T_s \ge 1@f$).
 *
 * @see C. Lee, J. Hong, K. Nam, R. Ortega, L. Praly, A. Astolfi, "Sensorless
 *      Control of Surface-Mount Permanent-Magnet Synchronous Motors Based on a
 *      Nonlinear Observer," IEEE Trans. Power Electron., 25(2), 2010.
 *      doi:10.1109/TPEL.2009.2027894
 *
 * @tparam T Scalar type (float for embedded deployment)
 */
template<typename T = float>
struct SensorlessEstimator {
    struct Parameters {
        T phase_resistance{};   ///< [Ω]  Stator phase resistance R
        T phase_inductance{};   ///< [H]  Stator inductance L
        T pm_flux_linkage{};    ///< [Wb] PM flux linkage λ
        T observer_gain{};      ///< [rad/s] Nonlinear observer gain g
        T pll_bandwidth{};      ///< [rad/s] Tracking-PLL bandwidth
        T pole_pairs{T{1}};     ///< Pole pairs (electrical-to-mechanical ratio)
        T fusion_blend_speed{}; ///< [elec rad/s] speed below which a fused sensor
                                ///< angle is trusted; 0 ⇒ sensorless only
    } params{};

    constexpr SensorlessEstimator() = default;
    constexpr explicit SensorlessEstimator(const Parameters& p) : params(p) {}

    /// One pure-sensorless step.
    /// @param i_ab        [A] measured stator current (αβ, e.g. clarke_transform(i_abc))
    /// @param v_ab_applied [V] stator voltage applied in the previous step (αβ)
    /// @param Ts          [s] sample time
    /// @return false (and resets) if the discrete PLL gain is unstable for @p Ts
    constexpr bool update(const AlphaBeta<T>& i_ab, const AlphaBeta<T>& v_ab_applied, T Ts) {
        return run(i_ab, v_ab_applied, Ts, /*has_sensor=*/false, T{0});
    }

    /// One step fusing a measured electrical angle (e.g. interpolated Hall).
    /// @param sensor_phase [elec rad] measured electrical angle
    /// @copydetails update(const AlphaBeta<T>&, const AlphaBeta<T>&, T)
    constexpr bool update(const AlphaBeta<T>& i_ab, const AlphaBeta<T>& v_ab_applied, T Ts, T sensor_phase) {
        return run(i_ab, v_ab_applied, Ts, /*has_sensor=*/true, sensor_phase);
    }

    /// Estimated electrical angle [rad], wrapped to [-π, π).
    [[nodiscard]] constexpr T phase() const { return phase_; }
    /// Estimated electrical velocity [elec rad/s].
    [[nodiscard]] constexpr T electrical_velocity() const { return omega_; }
    /// Estimated mechanical velocity [mech rad/s].
    [[nodiscard]] constexpr T mechanical_velocity() const {
        return omega_ / wet::max(params.pole_pairs, T{1});
    }
    /// PM-flux estimate η [Wb] (αβ); its magnitude tracks λ at lock.
    [[nodiscard]] constexpr AlphaBeta<T> pm_flux() const { return eta_; }

    constexpr void reset() {
        flux_ = {};
        eta_ = {};
        phase_ = T{0};
        omega_ = T{0};
    }

private:
    constexpr bool run(const AlphaBeta<T>& i_ab, const AlphaBeta<T>& v_ab_applied, T Ts, bool has_sensor, T sensor_phase) {
        // Tracking-PLL gains from bandwidth; critically damped (k_i = k_p²/4).
        const T pll_kp = T{2} * params.pll_bandwidth;
        const T pll_ki = T{0.25} * pll_kp * pll_kp;

        // Discrete-time stability guard on the proportional correction.
        if (!(Ts * pll_kp < T{1})) {
            reset();
            return false;
        }

        // --- Nonlinear flux observer (paper eqns 4, 6, 8) -------------------
        // Flux-driving voltage y = v - R·i, integrated to predict the flux state.
        const AlphaBeta<T> y = v_ab_applied - (params.phase_resistance * i_ab);
        flux_ += y * Ts;
        eta_ = flux_ - (params.phase_inductance * i_ab);

        // Observer correction pulls |η| onto the λ-circle.
        const T lambda_sq = params.pm_flux_linkage * params.pm_flux_linkage;
        const T eta_sq = (eta_.alpha * eta_.alpha) + (eta_.beta * eta_.beta);
        const T correction_gain = T{0.5} * (params.observer_gain / lambda_sq) * (lambda_sq - eta_sq);
        flux_ += (correction_gain * eta_) * Ts;
        eta_ = flux_ - (params.phase_inductance * i_ab);

        const T theta_flux = eta_.arg(); // atan2(η_β, η_α)

        // --- Second-order tracking PLL -------------------------------------
        constexpr T pi = wet::numbers::pi_v<T>;

        // Predict the angle forward with the velocity estimate.
        phase_ = wet::wrap(phase_ + (Ts * omega_), -pi, pi);

        // Angle innovation, optionally fused with the sensor by a speed-dependent
        // complementary weight (sensor below blend speed, sensorless above it).
        T delta = wet::wrap(theta_flux - phase_, -pi, pi);
        if (has_sensor && params.fusion_blend_speed > T{0}) {
            const T w_sensorless = wet::clamp(wet::abs(omega_) / params.fusion_blend_speed, T{0}, T{1});
            const T delta_sensor = wet::wrap(sensor_phase - phase_, -pi, pi);
            delta = (w_sensorless * delta) + ((T{1} - w_sensorless) * delta_sensor);
        }

        // Correct angle (proportional) and velocity (integral).
        phase_ = wet::wrap(phase_ + (Ts * pll_kp * delta), -pi, pi);
        omega_ += Ts * pll_ki * delta;
        return true;
    }

    AlphaBeta<T> flux_{};      // [Vs] integrated stator flux state x
    AlphaBeta<T> eta_{};       // [Wb] PM-flux estimate η = x − L·i
    T            phase_{T{0}}; // [elec rad] tracked angle
    T            omega_{T{0}}; // [elec rad/s] tracked velocity
};

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
 * The loop filter is a canonical @ref PIController and the phase a wrapping
 * integrator — the same treatment as @ref SinglePhasePLL.
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
    /// PI loop filter; output is the frequency offset [Hz] about nominal. See
    /// @ref SinglePhasePLL::loop_filter.
    PIController<T> loop_filter{};

    /// SOGI damping gain (√2 ≈ unity-Q).
    T sogi_gain{wet::numbers::sqrt2_v<T>};

    /**
     * @param f_nom Nominal grid frequency [Hz]
     */
    constexpr explicit DsogiPll(T f_nom)
        : nominal_frequency(f_nom), frequency_estimate(f_nom) {
        const T max_freq_deviation = f_nom * T{0.5};
        loop_filter.Kp = T{10};
        loop_filter.Ki = T{100};
        loop_filter.Kbc = loop_filter.Kp;
        loop_filter.u_max = max_freq_deviation;
        loop_filter.u_min = -max_freq_deviation;
    }

    /// One synchronization step from the stationary-frame αβ grid vector.
    constexpr void step(const AlphaBeta<T>& v_ab, const T Ts) {
        // Per-axis quadrature signal generation at the current frequency estimate.
        const auto [va, qva] = sogi_alpha(v_ab.alpha, frequency_estimate, sogi_gain, Ts);
        const auto [vb, qvb] = sogi_beta(v_ab.beta, frequency_estimate, sogi_gain, Ts);

        const AlphaBeta<T> v = {va, vb};
        const AlphaBeta<T> qv = {qva, qvb};
        positive_ab_ = positive_sequence_ab(v, qv);
        negative_ab_ = negative_sequence_ab(v, qv);

        // SRF-PLL: q-axis projection of the positive sequence is the phase error
        // (v_q = |v⁺|·sin(φ − θ̂) → 0 at lock, with v_d → +|v⁺|).
        positive_dq_ = park_transform(positive_ab_, phase_estimate);
        const T phase_error = positive_dq_.q;

        // PI loop filter (with anti-windup) → frequency estimate [Hz].
        frequency_estimate = nominal_frequency + loop_filter.control(phase_error, T{0}, Ts);

        // Wrapping integrator: frequency [Hz] → phase [rad], kept in [-π, π).
        const T pi = wet::numbers::pi_v<T>;
        phase_estimate = wet::wrap(phase_estimate + (T{2} * pi * frequency_estimate * Ts), -pi, pi);
    }

    /// Estimated frequency [Hz].
    [[nodiscard]] constexpr T frequency() const { return frequency_estimate; }
    /// Estimated phase [rad], wrapped to [-π, π).
    [[nodiscard]] constexpr T                   phase() const { return phase_estimate; }
    [[nodiscard]] constexpr AlphaBeta<T>        positive_sequence() const { return positive_ab_; }
    [[nodiscard]] constexpr AlphaBeta<T>        negative_sequence() const { return negative_ab_; }
    [[nodiscard]] constexpr DirectQuadrature<T> positive_dq() const { return positive_dq_; }

    constexpr void reset() {
        sogi_alpha = {};
        sogi_beta = {};
        loop_filter.reset();
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
    T phase_estimate{};
    T frequency_estimate{};

    AlphaBeta<T>        positive_ab_{};
    AlphaBeta<T>        negative_ab_{};
    DirectQuadrature<T> positive_dq_{};
};

} // namespace wet
