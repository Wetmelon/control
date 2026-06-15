#pragma once

#include <limits>

#include "wet/backend.hpp"
#include "wet/controllers/pid.hpp"
#include "wet/math/math.hpp"
#include "wet/matrix/colvec.hpp"
#include "wet/utility/motor_control.hpp"

namespace wet {

/**
 * @brief Result of one FOController::step(), carrying the actuator command plus
 *        the saturation/measurement signals an outer (velocity/position) loop
 *        needs to propagate anti-windup back up a cascade.
 *
 * @tparam T Scalar type
 */
template<typename T = float>
struct FocResult {
    ColVec<3, T>        duties = {};         ///< [pu] half-bridge duties {a,b,c}, each in [0,1]
    DirectQuadrature<T> Idq = {};            ///< [A] measured current = realized value (outer-loop u_sat)
    bool                v_saturated = false; ///< |Vdq| hit the SVPWM voltage circle (use to gate outer back-calc)
    bool                svm_clipped = false; ///< duties hit [0,1] (over-modulation safety net)
    T                   v_excess = T{0};     ///< |Vdq|/Vmax before limiting (>1 ⇒ saturated); 0 when unlimited
};

/**
 * @brief Result of FOController::current_controller(): the dq voltage command
 *        plus its saturation signals.
 *
 * Vdq is always a usable command (clamped to the voltage circle); is_saturated /
 * v_excess are advisory flags for the caller's anti-windup, hence a flag-carrying
 * struct rather than a wet::optional / wet::expected wrapper.
 *
 * @tparam T Scalar type
 */
template<typename T = float>
struct DqCommand {
    DirectQuadrature<T> Vdq = {};             ///< [V] dq voltage target (clamped to |Vdq| ≤ Vmax)
    bool                is_saturated = false; ///< |Vdq| hit the voltage circle
    T                   v_excess = T{0};      ///< |Vdq|/Vmax before limiting (>1 ⇒ saturated); 0 when unlimited
};

template<typename T = float>
struct FOController {
    using DQ = DirectQuadrature<T>;

    DQ Ldq = {};
    T  R = {};
    T  lambda = {};
    T  omega = {};

    /// Fraction of the SVPWM voltage circle (Vdc/√3) made available to the
    /// regulator. 1.0 = full linear range; reserve headroom with a smaller value.
    T max_modulation = T{1};

    /// Add plant-model (inverse) feedforward: resistive drop R·I and the deadbeat
    /// inductor voltage L·dI/dt. OFF by default: these cancel the plant's R and
    /// reshape the loop, which is inconsistent with tune()'s pole placement (it
    /// assumes the PI closes around the bare 1/(Ls+R)). Cross-axis decoupling and
    /// back-EMF feedforward are always applied regardless of this flag.
    bool plant_inversion_ff = false;

    // Canonical PI controllers operating on orthogonal vectors
    PIController<T> dctrl = {};
    PIController<T> qctrl = {};

    constexpr FOController() = default;
    constexpr FOController(DQ Ldq, T R, T lambda, T omega, T omega_bw = T{1000})
        : Ldq(Ldq), R(R), lambda(lambda), omega(omega) {
        tune(omega_bw); // Seed the Kp, Ki gains
    }

    /**
     * @brief PI current-loop gains by closed-loop pole placement
     * @ingroup motor_control
     *
     * Tunes one dq current axis modeled as a series R-L plant @f$ G(s) = 1/(Ls+R) @f$
     * regulated by a PI controller @f$ K_p + K_i/s @f$. The closed-loop characteristic
     * polynomial
     * @f[
     *   s^2 + \frac{R + K_p}{L}\,s + \frac{K_i}{L}
     * @f]
     * is matched to the canonical second-order form @f$ s^2 + 2\zeta\omega_n s + \omega_n^2 @f$,
     * giving
     * @f[
     *   K_i = L\,\omega_n^2, \qquad K_p = 2\zeta\omega_n L - R .
     * @f]
     * The default @f$ \zeta = 1 @f$ places a critically damped double pole at
     * @f$ -\omega_n @f$ (no oscillation). @p omega_bw is that pole frequency
     * @f$ \omega_n @f$ in rad/s — the closed-loop current-loop bandwidth.
     *
     * The back-calculation gain is seeded to @f$ K_{bc} = K_p @f$, i.e. a tracking
     * time constant @f$ T_t = T_i = K_p/K_i @f$, the standard anti-windup choice for
     * a PI.
     *
     * @note With the model-inversion feedforward in FOController::current_controller
     *       active, these poles govern the feedback (disturbance / model-error
     *       rejection) dynamics; reference tracking is dominated by the feedforward.
     *
     * @see Harnefors & Nee, "Model-based current control of AC machines using the
     *      internal model control method", IEEE T-IA 34(1), 1998.
     *
     * @param L        [H]      Axis inductance (L_d or L_q)
     * @param R        [ohm]    Phase resistance
     * @param omega_bw [rad/s]  Desired closed-loop pole frequency (bandwidth)
     * @param Ts       [s]      Controller sample time
     * @param zeta     [-]      Closed-loop damping ratio (default 1, critically damped)
     * @return PI gains as a design::PIDResult
     */
    constexpr void tune(T omega_bw, T zeta = T{1}) {
        dctrl.Kp = T{2} * zeta * omega_bw * Ldq.d - R;
        dctrl.Ki = Ldq.d * omega_bw * omega_bw;
        dctrl.Kbc = dctrl.Kp;

        qctrl.Kp = T{2} * zeta * omega_bw * Ldq.q - R;
        qctrl.Ki = Ldq.q * omega_bw * omega_bw;
        qctrl.Kbc = qctrl.Kp;
    }

    /**
     * @brief Bumpless disable: integrators track an externally applied dq voltage
     *
     * Puts both PI loops into tracking mode so that, while the regulator is
     * sidelined (inverter disabled, fault handling, or an alternate command
     * source), their integrators preload toward @p Vdq_track. Re-enabling with
     * enable() then resumes Auto without a command jump.
     */
    constexpr void disable(const DQ& Vdq_track = {}) {
        dctrl.disable(Vdq_track.d);
        qctrl.disable(Vdq_track.q);
    }

    /// Resume Auto control on both dq loops (bumpless after disable()).
    constexpr void enable() {
        dctrl.enable();
        qctrl.enable();
    }

    /// Clear both dq integrators (controller stays in its current mode).
    constexpr void reset() {
        dctrl.reset();
        qctrl.reset();
    }

    /**
     * @brief One step of a full FOC control cycle
     *
     * @see https://www.ti.com/lit/an/sprabz0a/sprabz0a.pdf
     *
     * @param[in] Idq_ref   [A] Target current in DQ frame
     * @param[in] Iabc      [A] Individual Measured phase currents
     * @param[in] theta     [rad] Electrical angle
     * @param[in] Vdc       [V] DC Bus Voltage
     * @param[in] Ts        [s] Control step execution period
     *
     * @return FocResult: duties plus measured current and saturation signals for
     *         cascade anti-windup propagation.
     */
    [[nodiscard]] FocResult<T> step(const DQ& Idq_ref, const ColVec<3, T>& Iabc, const T theta, const T Vdc, const T Ts) {
        // Largest dq voltage magnitude the inverter can synthesize in the SVPWM
        // linear range (peak phase voltage = Vdc/√3), scaled by max_modulation.
        const T Vmax = max_modulation * Vdc * wet::numbers::inv_sqrt3_v<T>;

        FocResult<T> result;
        result.Idq = clarke_park_transform(Iabc, theta);

        const auto cmd = current_controller(Idq_ref, result.Idq, Ts, Vmax);
        result.v_saturated = cmd.is_saturated;
        result.v_excess = cmd.v_excess;

        const auto Vab = inverse_park_transform(cmd.Vdq, theta);
        const auto svm = svm_duty_cycles(Vab, Vdc);
        result.duties = svm.duties;
        result.svm_clipped = svm.is_clipped;
        return result;
    }

    /**
     * @brief One step of the core FOC algorithm, assuming balanced 3ph
     *
     * @see https://discourse.odriverobotics.com/t/recommended-books-resources/382/17
     *
     * Structure: always-on disturbance feedforward (cross-axis decoupling and
     * back-EMF) + PI feedback on the dq errors, optionally plus plant-inversion
     * feedforward (plant_inversion_ff). Plant parameters (Ldq, R, lambda) and
     * electrical speed (omega) are taken from the controller's members; tune the
     * PI loops with tune().
     *
     * @param[in] Idq_ref  [A] Target current in DQ frame
     * @param[in] Idq      [A] Measured current in DQ frame
     * @param[in] Ts       [s] Controller sample time
     * @param[in] Vmax     [V] Available dq voltage magnitude (SVPWM circle radius);
     *                         the command is held to |Vdq| ≤ Vmax with integrator
     *                         anti-windup. Defaults to ∞ (no limiting).
     *
     * @return DqCommand: the dq voltage target plus is_saturated / v_excess flags
     *         for the caller's anti-windup. Vdq is always valid (clamped).
     */
    [[nodiscard]] DqCommand<T> current_controller(const DQ& Idq_ref, const DQ& Idq, const T Ts, const T Vmax = std::numeric_limits<T>::infinity()) {

        // "Always-On" feedforward terms cancel controller dependence on omega
        DQ Vdq = {
            .d = -(omega * Ldq.q * Idq_ref.q),
            .q = (omega * Ldq.d * Idq_ref.d) + (omega * lambda),
        };

        // Default-off plant inversion
        // Enabling this cancels the plant R/L poles and invalidates the PI controller tuning
        if (plant_inversion_ff) {
            const DQ Idq_dot = {
                .d = (Idq_ref.d - Idq.d) / Ts,
                .q = (Idq_ref.q - Idq.q) / Ts,
            };
            Vdq.d += (Ldq.d * Idq_dot.d) + (R * Idq.d);
            Vdq.q += (Ldq.q * Idq_dot.q) + (R * Idq.q);
        }

        // PI feedback on the dq current errors.
        Vdq.d += dctrl.control(Idq_ref.d, Idq.d, Ts);
        Vdq.q += qctrl.control(Idq_ref.q, Idq.q, Ts);

        // Circular voltage limit + integrator anti-windup. The inverter can only
        // synthesize |Vdq| ≤ Vmax (the SVPWM voltage circle); a per-axis box clamp
        // would distort the voltage-vector angle, so the magnitude is scaled and
        // the clipped amount is back-calculated out of each PI integrator.
        const T Vmag = wet::sqrt((Vdq.d * Vdq.d) + (Vdq.q * Vdq.q));

        DqCommand<T> cmd;
        cmd.is_saturated = Vmag > Vmax;
        cmd.v_excess = Vmag / Vmax; // Vmax == ∞ ⇒ 0
        if (cmd.is_saturated) {
            const T  scale = Vmax / Vmag;
            const DQ Vsat = {
                .d = Vdq.d * scale,
                .q = Vdq.q * scale,
            };

            dctrl.back_calculate(Vdq.d, Vsat.d, Ts);
            qctrl.back_calculate(Vdq.q, Vsat.q, Ts);

            Vdq = Vsat;
        }

        cmd.Vdq = Vdq;
        return cmd;
    }
};
}; // namespace wet
