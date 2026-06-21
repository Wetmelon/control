#pragma once

#include <limits>

#include "wet/backend.hpp"
#include "wet/controllers/pid.hpp"
#include "wet/matrix/colvec.hpp"
#include "wet/utility/modulation.hpp"
#include "wet/utility/transforms.hpp"

namespace wet {

namespace design {

/**
 * @defgroup foc_design Field-Oriented Control Design
 * @brief Closed-form design maps for a PMSM field-oriented drive
 *
 * Pure functions of the machine nameplate (R, L_dq, λ, pole pairs) and the bus.
 * They are @c constexpr so they fold to literals whenever the motor parameters
 * are compile-time constants — the common single-motor firmware case — yet stay
 * usable at runtime for parameter identification or gain scheduling.
 */

/**
 * @brief Current-loop PI gains by closed-loop pole placement on the R–L plant
 * @ingroup foc_design
 *
 * One dq current axis is the series R–L plant @f$ G(s) = 1/(Ls+R) @f$ regulated
 * by a PI @f$ K_p + K_i/s @f$. Matching the closed-loop characteristic polynomial
 * @f$ s^2 + \frac{R+K_p}{L}s + \frac{K_i}{L} @f$ to the canonical form
 * @f$ s^2 + 2\zeta\omega_n s + \omega_n^2 @f$ gives
 * @f[
 *   K_p = 2\zeta\omega_n L - R, \qquad K_i = L\,\omega_n^2 .
 * @f]
 * @f$ \zeta = 1 @f$ (default) places a critically damped double pole at
 * @f$ -\omega_n @f$, so @p omega_bw is the closed-loop current-loop bandwidth.
 *
 * The back-calculation gain is seeded to @f$ K_p @f$ (tracking time constant
 * @f$ T_t = K_p/K_i @f$), the standard anti-windup choice for a PI.
 *
 * The proportional setpoint weight @p b selects the reference structure *without*
 * moving the closed-loop poles — it only repositions the reference→output zero at
 * @f$ -K_i/K_p @f$. @p b = 1 is a standard PI (P acts on the error); @p b = 0 is
 * an **I-P** loop (P acts on the measurement only). I-P removes the PI's
 * proportional step-kick and the reference overshoot from that zero, while leaving
 * disturbance rejection unchanged — useful on a current loop whose @f$ i_q @f$
 * reference steps hard, since the kick is the usual trigger of the SVPWM
 * voltage-circle limit. With FOController's feedforward supplying the transient,
 * I-P feedback is the natural pairing.
 *
 * @note @p omega_bw is bounded by the sampling rate — keep it roughly an order of
 *       magnitude below the PWM/control frequency (@f$ \omega_n \lesssim 2\pi
 *       f_{sw}/10 @f$) for the continuous-time pole placement to hold.
 *
 * @see Harnefors & Nee, "Model-based current control of AC machines using the
 *      internal model control method", IEEE T-IA 34(1), 1998.
 * @see Åström & Hägglund, "Advanced PID Control", 2006, §4.4 — setpoint weighting.
 *
 * @param L        [H]      Axis inductance (L_d or L_q)
 * @param R        [ohm]    Phase resistance
 * @param omega_bw [rad/s]  Desired closed-loop pole frequency (bandwidth)
 * @param zeta     [-]      Closed-loop damping ratio (default 1, critically damped)
 * @param b        [-]      Proportional setpoint weight: 1 = PI, 0 = I-P
 * @return PIDResult with Kp, Ki, Kbc (= Kp) and the setpoint weight b set
 */
template<typename T = double>
[[nodiscard]] constexpr PIDResult<T>
current_loop_pi(T L, T R, T omega_bw, T zeta = T{1}, T b = T{1}) {
    PIDResult<T> result{};
    result.Kp = (T{2} * zeta * omega_bw * L) - R;
    result.Ki = L * omega_bw * omega_bw;
    result.Kbc = result.Kp; // T_t = T_i: standard PI anti-windup tracking constant
    result.b = b;           // 1 = PI (P on error), 0 = I-P (P on measurement)
    return result;
}

/**
 * @brief Torque constant @f$ K_t @f$ of a PMSM (amplitude-invariant convention)
 * @ingroup foc_design
 *
 * With the amplitude-invariant Clarke/Park scaling the electromagnetic torque of
 * a non-salient machine is @f$ T_e = \frac{3}{2} p\,\lambda\,i_q @f$, so
 * @f[
 *   K_t = \frac{3}{2}\,p\,\lambda \qquad [\mathrm{Nm/A}] .
 * @f]
 *
 * @param pole_pairs @f$ p @f$ Number of pole pairs
 * @param lambda     [Wb] Permanent-magnet flux linkage @f$ \lambda @f$
 * @return @f$ K_t @f$ [Nm/A]
 */
template<typename T = double>
[[nodiscard]] constexpr T torque_constant_from_flux(T pole_pairs, T lambda) {
    return T{1.5} * pole_pairs * lambda;
}

/**
 * @brief PM flux linkage from a motor's torque constant (amplitude-invariant)
 * @ingroup foc_design
 *
 * Inverse of torque_constant_from_flux(): a datasheet usually quotes @f$ K_t @f$
 * rather than the flux linkage the plant model needs, so
 * @f[
 *   \lambda = \frac{K_t}{\frac{3}{2}\,p} \qquad [\mathrm{Wb}] .
 * @f]
 *
 * @note @p Kt must be the amplitude (peak per-phase) torque constant that matches
 *       the amplitude-invariant Clarke/Park convention, i.e. @f$ T_e = K_t i_q @f$
 *       with @f$ i_q @f$ the peak current. A @f$ K_t @f$ quoted per RMS current is
 *       @f$ \sqrt 3 @f$ larger and must be divided by @f$ \sqrt 3 @f$ first.
 *
 * @param pole_pairs @f$ p @f$ Number of pole pairs
 * @param Kt         [Nm/A] Torque constant (amplitude convention)
 * @return @f$ \lambda @f$ [Wb]
 */
template<typename T = double>
[[nodiscard]] constexpr T flux_from_torque_constant(T pole_pairs, T Kt) {
    return Kt / (T{1.5} * pole_pairs);
}

/**
 * @brief Torque constant from the datasheet velocity constant @f$ K_v @f$
 * @ingroup foc_design
 *
 * Hobby/BLDC datasheets quote @f$ K_v @f$ in RPM per applied volt, which for an
 * SVPWM drive is effectively referenced to the **peak line-to-line** back-EMF
 * (≈ the no-load bus voltage). The motor's listed @f$ K_t @f$ and @f$ K_v @f$ then
 * satisfy @f$ K_t K_v \approx 8.27 @f$:
 * @f[
 *   K_t = \frac{60\sqrt 3}{4\pi}\,\frac{1}{K_v} = \frac{8.27}{K_v} \qquad [\mathrm{Nm/A}] ,
 * @f]
 * with @f$ 60/2\pi @f$ the RPM→rad/s conversion and @f$ \sqrt3/2 = (3/2)/\sqrt3 @f$
 * lumping the @f$ 3/2 @f$ amplitude-torque factor with the @f$ \sqrt3 @f$
 * phase→line-line factor. The returned @f$ K_t @f$ is amplitude convention
 * (matches torque_constant_from_flux()).
 *
 * @warning Referencing sets the constant @f$ K_t K_v = 45/(\pi f) @f$, where the
 *          referenced voltage is @f$ f\,\hat e_{phase} @f$: peak line-line
 *          (@f$ f=\sqrt3 @f$) → 8.27, RMS line-line (@f$ f=\sqrt{3/2} @f$) → 11.70
 *          (=√2·8.27), peak phase (@f$ f=1 @f$) → 14.32, RMS phase → 20.25. This
 *          function assumes the hobby/RC peak-line-line (per-bus-volt) sense — the
 *          same one ODrive uses. If a @f$ K_v @f$/@f$ K_e @f$ is specified per RMS
 *          line-to-line (common on industrial servos), use 11.70/@f$ K_v @f$, or
 *          invert @f$ K_e @f$ directly.
 *
 * @see P. Pillay & R. Krishnan, "Modeling, simulation and analysis of PM motor
 *      drives, Part I: PMSM drive," IEEE T-IA 25(2), 1989 — @f$ T_e=\frac{3}{2} p\lambda i_q @f$.
 * @see R. Krishnan, "Permanent Magnet Synchronous and Brushless DC Motor Drives,"
 *      CRC Press, 2010, Ch. 9 — @f$ K_t @f$/@f$ K_e @f$ equivalence and referencing.
 *
 * @param Kv [RPM/V] Velocity constant (RMS line-to-line convention)
 * @return @f$ K_t @f$ [Nm/A], amplitude convention
 */
template<typename T = double>
[[nodiscard]] constexpr T torque_constant_from_Kv(T Kv) {
    // 60*sqrt(3)/(4*pi) = (30/pi) * (sqrt(3)/2)
    constexpr T c = (T{30} / wet::numbers::pi_v<T>)*(wet::numbers::sqrt3_v<T> / T{2});
    return c / Kv;
}

/**
 * @brief PM flux linkage from the datasheet velocity constant @f$ K_v @f$
 * @ingroup foc_design
 *
 * Composes torque_constant_from_Kv() with flux_from_torque_constant():
 * @f$ \lambda = K_t(K_v) / (\frac{3}{2} p) @f$. See torque_constant_from_Kv() for the
 * @f$ K_v @f$ convention and its caveats.
 *
 * @param pole_pairs @f$ p @f$ Number of pole pairs
 * @param Kv         [RPM/V] Velocity constant (RMS line-to-line convention)
 * @return @f$ \lambda @f$ [Wb]
 */
template<typename T = double>
[[nodiscard]] constexpr T flux_from_Kv(T pole_pairs, T Kv) {
    return flux_from_torque_constant(pole_pairs, torque_constant_from_Kv(Kv));
}

/**
 * @brief Motor constant @f$ K_m @f$ (torque per √copper-loss) — a figure of merit
 * @ingroup foc_design
 *
 * @f[
 *   K_m = \frac{K_t}{\sqrt{R}} \qquad [\mathrm{Nm}/\sqrt{\mathrm W}] ,
 * @f]
 * quantifies how much torque a winding produces per square-root watt of resistive
 * loss — winding-count invariant, so it ranks motors independently of turns. Use
 * @p R consistent with @p Kt's current reference (per-phase resistance with the
 * amplitude/peak @f$ K_t @f$). Unlike the @f$ K_t @f$/@f$ \lambda @f$/@f$ K_v @f$
 * trio this is **not** invertible to the plant parameters — it is an output metric.
 *
 * @param Kt [Nm/A] Torque constant (amplitude convention)
 * @param R  [ohm]  Phase resistance
 * @return @f$ K_m @f$ [Nm/√W]
 */
template<typename T = double>
[[nodiscard]] constexpr T motor_constant(T Kt, T R) {
    return Kt / wet::sqrt(R);
}

/**
 * @brief q-axis current command for a requested torque (non-salient PMSM, Id=0)
 * @ingroup foc_design
 *
 * Inverts @f$ T_e = K_t i_q @f$: @f$ i_q = T_e / K_t @f$. Saliency torque
 * @f$ (L_d-L_q)i_d i_q @f$ is neglected, exact for an SPMSM under @f$ i_d = 0 @f$.
 *
 * @param Te         [Nm] Requested electromagnetic torque
 * @param pole_pairs @f$ p @f$ Number of pole pairs
 * @param lambda     [Wb] Permanent-magnet flux linkage
 * @return @f$ i_q @f$ [A]
 */
template<typename T = double>
[[nodiscard]] constexpr T iq_from_torque(T Te, T pole_pairs, T lambda) {
    return Te / torque_constant_from_flux(pole_pairs, lambda);
}

/**
 * @brief Radius of the SVPWM voltage circle (max synthesizable @f$ |V_{dq}| @f$)
 * @ingroup foc_design
 *
 * The largest dq voltage magnitude the inverter can produce in the SVPWM linear
 * range is the peak phase voltage @f$ V_{dc}/\sqrt 3 @f$, scaled by the available
 * modulation fraction:
 * @f[
 *   V_{max} = m\,\frac{V_{dc}}{\sqrt 3} .
 * @f]
 *
 * @param Vdc            [V] DC-bus voltage
 * @param max_modulation [-] Fraction of the voltage circle to use (default 1)
 * @return @f$ V_{max} @f$ [V]
 */
template<typename T = double>
[[nodiscard]] constexpr T voltage_circle_radius(T Vdc, T max_modulation = T{1}) {
    return max_modulation * Vdc * wet::numbers::inv_sqrt3_v<T>;
}

/**
 * @brief Base (corner) electrical speed where the voltage circle is first hit
 * @ingroup foc_design
 *
 * Neglecting the resistive drop, the steady-state stator voltage magnitude at
 * operating current @f$ (i_d, i_q) @f$ is @f$ \omega\sqrt{(L_q i_q)^2 +
 * (L_d i_d + \lambda)^2} @f$. The base speed is where this reaches @f$ V_{max} @f$:
 * @f[
 *   \omega_{base} = \frac{V_{max}}{\sqrt{(L_q i_q)^2 + (L_d i_d + \lambda)^2}} .
 * @f]
 * Above @f$ \omega_{base} @f$ the drive must field-weaken (drive @f$ i_d<0 @f$) to
 * stay inside the circle. For an unloaded SPMSM (@f$ i_d=i_q=0 @f$) this reduces
 * to @f$ \omega_{base} = V_{max}/\lambda @f$.
 *
 * @param Vmax   [V]   Voltage circle radius (see voltage_circle_radius())
 * @param Ldq    [H]   dq inductance pair @f$ (L_d, L_q) @f$
 * @param lambda [Wb]  Permanent-magnet flux linkage
 * @param Idq    [A]   Operating-point dq current (default 0 — no-load corner)
 * @return @f$ \omega_{base} @f$ [electrical rad/s]
 */
template<typename T = double>
[[nodiscard]] constexpr T
base_speed(T Vmax, const DirectQuadrature<T>& Ldq, T lambda, const DirectQuadrature<T>& Idq = {}) {
    const T flux_q = Ldq.q * Idq.q;
    const T flux_d = (Ldq.d * Idq.d) + lambda;
    return Vmax / wet::sqrt((flux_q * flux_q) + (flux_d * flux_d));
}

} // namespace design

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
     * @brief Seed both dq PI loops by closed-loop pole placement on the R–L plant
     *
     * Applies design::current_loop_pi() to each axis (using this controller's Ldq
     * and R) and loads the resulting gains, seeding each back-calculation gain to
     * @f$ K_{bc} = K_p @f$. See current_loop_pi() for the pole-placement derivation.
     *
     * @note With the model-inversion feedforward in FOController::current_controller
     *       active, these poles govern the feedback (disturbance / model-error
     *       rejection) dynamics; reference tracking is dominated by the feedforward.
     *       This makes @p b = 0 (I-P) the natural choice: feedforward supplies the
     *       step, and the proportional term stays off the reference, killing the
     *       voltage-kick/overshoot without affecting disturbance rejection.
     *
     * @param omega_bw [rad/s]  Desired closed-loop pole frequency (bandwidth)
     * @param zeta     [-]      Closed-loop damping ratio (default 1, critically damped)
     * @param b        [-]      Proportional setpoint weight: 1 = PI (P on error,
     *                          default), 0 = I-P (P on measurement, no step-kick)
     */
    constexpr void tune(T omega_bw, T zeta = T{1}, T b = T{1}) {
        const auto d = design::current_loop_pi(Ldq.d, R, omega_bw, zeta, b);
        const auto q = design::current_loop_pi(Ldq.q, R, omega_bw, zeta, b);

        dctrl.Kp = d.Kp;
        dctrl.Ki = d.Ki;
        dctrl.Kbc = d.Kbc; // T_t = T_i: standard PI anti-windup tracking constant
        dctrl.b = d.b;

        qctrl.Kp = q.Kp;
        qctrl.Ki = q.Ki;
        qctrl.Kbc = q.Kbc;
        qctrl.b = q.b;
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
        const T Vmag = Vdq.abs();

        DqCommand<T> cmd;
        cmd.is_saturated = Vmag > Vmax;
        cmd.v_excess = Vmag / Vmax; // Vmax == ∞ ⇒ 0
        if (cmd.is_saturated) {
            const T  scale = Vmax / Vmag;
            const DQ Vsat = Vdq * scale;

            dctrl.back_calculate(Vdq.d, Vsat.d, Ts);
            qctrl.back_calculate(Vdq.q, Vsat.q, Ts);

            Vdq = Vsat;
        }

        cmd.Vdq = Vdq;
        return cmd;
    }
};
}; // namespace wet
