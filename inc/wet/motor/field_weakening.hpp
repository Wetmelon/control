#pragma once

#include <concepts>
#include <cstdint>
#include <initializer_list>
#include <limits>

#include "wet/backend.hpp"    // wet::min, wet::max
#include "wet/math/math.hpp"  // wet::abs, wet::sqrt
#include "wet/motor/foc.hpp"  // design::voltage_circle_radius
#include "wet/transforms.hpp" // DirectQuadrature

namespace wet {

namespace design {

/**
 * @brief Feedforward field-weakening d-axis current from the voltage ellipse
 * @ingroup foc_design
 *
 * Above base speed the back-EMF inflates the stator voltage until it reaches the
 * inverter limit. Neglecting the resistive drop, the steady-state voltage magnitude
 * is @f$ |V| = \omega\sqrt{(L_d i_d + \lambda)^2 + (L_q i_q)^2} @f$; holding that at
 * @f$ V_{max} @f$ for the operating @f$ i_q @f$ pins the d-axis flux and gives the
 * required current
 * @f[
 *   i_d = \frac{-\lambda + \sqrt{(V_{max}/\omega)^2 - (L_q i_q)^2}}{L_d} ,
 * @f]
 * the larger (least-negative) root — the minimum weakening that respects the limit.
 * Below base speed the bracket exceeds @f$ \lambda @f$, so @f$ i_d>0 @f$ would be
 * returned; it is clamped to 0 (no weakening — the base reference governs). When the
 * radicand is negative even @f$ i_d @f$ cannot hold the voltage (the q-axis current
 * alone overruns the ellipse), so the d-flux is driven to zero (@f$ i_d=-\lambda/L_d @f$,
 * the characteristic current) and the q-axis must give way — left to the current-limit
 * stage / MTPV.
 *
 * This is the open-loop term of feedforward field weakening; pair it with a small
 * voltage-feedback trim (see @ref motor::FieldWeakening) to absorb model error.
 *
 * @see Sul, "Control of Electric Machine Drive Systems", IEEE/Wiley 2011, §6 — flux weakening.
 *
 * @param iq          [A]   q-axis current at the operating point
 * @param omega_elec  [rad/s] electrical speed
 * @param Vmax        [V]   available dq voltage magnitude (e.g. voltage_circle_radius)
 * @param Ldq         [H]   dq inductance pair @f$ (L_d, L_q) @f$
 * @param lambda      [Wb]  permanent-magnet flux linkage
 * @param omega_min   [rad/s] speed below which weakening is disabled (default 1e-3)
 * @return @f$ i_d \le 0 @f$ [A] (0 below base speed)
 */
template<typename T = double>
[[nodiscard]] constexpr T field_weakening_id(
    T iq, T omega_elec, T Vmax, const DirectQuadrature<T>& Ldq, T lambda, T omega_min = T{1e-3}
) {
    const T w = wet::abs(omega_elec);
    if (w <= omega_min) {
        return T{0}; // standstill / very low speed: no field weakening
    }
    const T vw = Vmax / w; // available flux magnitude Vmax/ω
    const T rad = (vw * vw) - ((Ldq.q * iq) * (Ldq.q * iq));
    if (rad <= T{0}) {
        return -lambda / Ldq.d; // even zero d-flux can't hold V — weaken fully
    }
    const T id = (-lambda + wet::sqrt(rad)) / Ldq.d;
    return id < T{0} ? id : T{0}; // only ever weaken (below base speed ⇒ 0)
}

} // namespace design

namespace motor {

/// Field-weakening law selection.
enum class FwMethod : std::uint8_t {
    VoltageFeedback, //!< reactive PI on the voltage margin (model-independent, robust)
    Feedforward,     //!< analytic id from the voltage ellipse + a small voltage-margin trim
};

/**
 * @brief Configuration for @ref FieldWeakening.
 * @tparam T Scalar type
 */
template<typename T = float>
struct FieldWeakeningConfig {
    DirectQuadrature<T> Ldq{};                                //!< [H] dq inductances (Feedforward only)
    T                   lambda{};                             //!< [Wb] PM flux linkage (Feedforward only)
    T                   i_max{std::numeric_limits<T>::max()}; //!< [A] stator current limit (magnitude)
    T                   v_margin{T{0.95}};                    //!< [-] fraction of the voltage circle to regulate to
    T                   ki{};                                 //!< [A/(V·s)] integrator gain (≥ 0)
    FwMethod            method{FwMethod::VoltageFeedback};    //!< which weakening law
};

/**
 * @brief Field-weakening current-reference regulator (voltage-feedback or feedforward)
 *
 * Sits between a base dq reference (from @ref MtpaReference or a LUT) and the inner
 * current loop. It pushes @f$ i_d @f$ negative when the inverter runs out of voltage,
 * then enforces the stator current circle by trading q-axis (torque) current for the
 * d-axis (flux) current it just spent — so flux is protected and torque gives way, the
 * correct priority above base speed. Source-agnostic: it adjusts whatever base
 * reference it is handed, so the same block serves an MTPA or a LUT front end.
 *
 * Two selectable laws (@ref FwMethod), both feeding the same current-limit stage:
 * - **VoltageFeedback** — integrate the voltage margin @f$ V_{lim}-|V_{dq}| @f$ into a
 *   weakening increment @f$ i_{d,fw}\le 0 @f$ added to the base reference. Needs no
 *   machine parameters and is insensitive to model error: the production default.
 * - **Feedforward** — take the analytic @ref design::field_weakening_id for this speed
 *   and operating point, combine it with the base reference (the more-weakened of the
 *   two), and add the same integral term as a small trim that mops up model error.
 *
 * Anti-windup is by back-calculation: after the current circle clamps @f$ i_d @f$, the
 * integrator is reset to the realized weakening, so it cannot wind past the limit and
 * recovers promptly when voltage headroom returns. Pass the previous tick's commanded
 * @f$ V_{dq} @f$ (the current loop's demand) as the voltage feedback.
 *
 * @see design::field_weakening_id — the feedforward law.
 * @see Sul, "Control of Electric Machine Drive Systems", IEEE/Wiley 2011, §6.
 *
 * @tparam T Scalar type
 */
template<typename T = float>
class FieldWeakening {
public:
    constexpr FieldWeakening() = default;
    constexpr explicit FieldWeakening(const FieldWeakeningConfig<T>& cfg) : cfg_(cfg) {}

    /**
     * @brief Adjust a base dq reference for the voltage and current limits.
     *
     * @param ref_base   [A] base (id,iq) reference (e.g. from @ref MtpaReference)
     * @param Vdq        [V] previous commanded dq voltage (current-loop demand)
     * @param omega_elec [rad/s] electrical speed (Feedforward only; ignored otherwise)
     * @param Vdc        [V] DC-bus voltage
     * @param dt         [s] tick period
     * @return limited (id,iq) reference inside the current circle
     */
    [[nodiscard]] constexpr DirectQuadrature<T>
    update(const DirectQuadrature<T>& ref_base, const DirectQuadrature<T>& Vdq, T omega_elec, T Vdc, T dt) {
        const T Vlim = cfg_.v_margin * design::voltage_circle_radius(Vdc);
        const T e = Vlim - Vdq.abs(); // voltage margin; < 0 while saturating

        // Integral weakening / trim — only ever weakens (id_int ≤ 0).
        id_int_ = clamp(id_int_ + (cfg_.ki * e * dt), -cfg_.i_max, T{0});

        T id_base = ref_base.d;
        if (cfg_.method == FwMethod::Feedforward) {
            const T id_ff = design::field_weakening_id(ref_base.q, omega_elec, Vlim, cfg_.Ldq, cfg_.lambda);
            id_base = wet::min({ref_base.d, id_ff}); // the more-weakened of base / feedforward
        }

        // Current circle: protect flux (id) first, give the q-axis the remainder.
        const T id = clamp(id_base + id_int_, -cfg_.i_max, T{0});
        const T iq_budget = wet::sqrt(wet::max({T{0}, (cfg_.i_max * cfg_.i_max) - (id * id)}));
        const T iq = clamp(ref_base.q, -iq_budget, iq_budget);

        id_int_ = clamp(id - id_base, -cfg_.i_max, T{0}); // back-calculate to the realized weakening
        return {.d = id, .q = iq};
    }

    /// Clear the weakening integrator (e.g. on drive disable).
    constexpr void reset() { id_int_ = T{0}; }

    [[nodiscard]] constexpr T weakening_current() const { return id_int_; } //!< [A] current id_fw integral

private:
    static constexpr T clamp(T x, T lo, T hi) { return wet::max({lo, wet::min({hi, x})}); }

    FieldWeakeningConfig<T> cfg_{};
    T                       id_int_{}; //!< weakening integrator state [A] (≤ 0)
};

/**
 * @brief Concept for a pluggable field-weakening / current-reference policy.
 *
 * A policy maps a base dq current reference to the (id,iq) an inner current loop
 * should track, given the previous voltage demand, electrical speed, bus voltage,
 * and tick period. @ref FieldWeakening (either method) and @ref NoFieldWeakening
 * model it; @ref PmacServo is generic over it so a drive can select — or supply —
 * its weakening strategy without the servo knowing which.
 *
 * @tparam P Policy type
 * @tparam T Scalar type
 */
template<typename P, typename T = float>
concept FieldWeakeningPolicy = requires(P p, const DirectQuadrature<T>& dq, T s) {
    { p.update(dq, dq, s, s, s) } -> std::same_as<DirectQuadrature<T>>;
};

/**
 * @brief Null field-weakening policy — passes the base reference through unchanged.
 *
 * The default for a drive that never field-weakens (surface-PM below base speed):
 * the base reference (e.g. @ref MtpaReference, or plain @f$ i_d=0 @f$) is the command,
 * and current limiting is left to the outer loop. Selecting this keeps @ref PmacServo
 * exactly as it behaves without weakening, at zero runtime cost.
 *
 * @tparam T Scalar type
 */
template<typename T = float>
struct NoFieldWeakening {
    /// Identity: the base reference is the tracked reference.
    [[nodiscard]] constexpr DirectQuadrature<T>
    update(const DirectQuadrature<T>& ref_base, const DirectQuadrature<T>& /*Vdq*/, T /*omega_elec*/, T /*Vdc*/, T /*dt*/) const {
        return ref_base;
    }
};

static_assert(FieldWeakeningPolicy<FieldWeakening<float>, float>);
static_assert(FieldWeakeningPolicy<NoFieldWeakening<float>, float>);

} // namespace motor
} // namespace wet
