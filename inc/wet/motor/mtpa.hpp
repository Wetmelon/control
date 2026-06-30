#pragma once

#include "wet/math/math.hpp"  // wet::abs, wet::hypot
#include "wet/motor/foc.hpp"  // design::torque_constant_from_flux, iq_from_torque
#include "wet/transforms.hpp" // DirectQuadrature

namespace wet {

namespace design {

/**
 * @brief MTPA d-axis current on the trajectory for a given q-axis current
 * @ingroup foc_design
 *
 * The maximum-torque-per-ampere (MTPA) locus minimizes the stator current
 * magnitude @f$ |i| = \sqrt{i_d^2 + i_q^2} @f$ for a commanded torque. Applying a
 * Lagrange multiplier to the amplitude-invariant torque
 * @f$ T_e = \frac{3}{2}p\,[\lambda i_q + (L_d - L_q)i_d i_q] @f$ subject to fixed
 * @f$ T_e @f$ gives the locus condition @f$ (L_d-L_q)i_d^2 + \lambda i_d -
 * (L_d-L_q)i_q^2 = 0 @f$, whose physical root (the one with @f$ i_d \le 0 @f$ for a
 * reluctance-assisting machine) is
 * @f[
 *   i_d = \frac{\lambda - \sqrt{\lambda^2 + 4(L_q-L_d)^2 i_q^2}}{2\,(L_q - L_d)} .
 * @f]
 * A non-salient machine (@f$ L_d = L_q @f$) has no reluctance torque, so the locus
 * collapses to @f$ i_d = 0 @f$ — returned directly to avoid the @f$ 0/0 @f$ above.
 * Saliency is judged relative to the inductance scale (the test is
 * @f$ |L_q-L_d| > \epsilon\,|L_d+L_q| @f$), not as an absolute henry threshold.
 *
 * @see Sul, "Control of Electric Machine Drive Systems", IEEE/Wiley 2011, §5 — MTPA.
 * @see Morimoto et al., "Expansion of operating limits for PM motor by current
 *      vector control considering inverter capacity", IEEE T-IA 26(5), 1990.
 *
 * @param iq      [A]  q-axis current
 * @param lambda  [Wb] permanent-magnet flux linkage
 * @param Ldq     [H]  dq inductance pair @f$ (L_d, L_q) @f$
 * @param eps     [-]  relative saliency threshold (default 1e-6)
 * @return @f$ i_d @f$ [A] on the MTPA locus (@f$ \le 0 @f$ for @f$ L_q > L_d @f$)
 */
template<typename T = double>
[[nodiscard]] constexpr T mtpa_id_from_iq(T iq, T lambda, const DirectQuadrature<T>& Ldq, T eps = T{1e-6}) {
    const T Lsal = Ldq.q - Ldq.d; // saliency (> 0 for IPMSM / SynRM)
    if (wet::abs(Lsal) <= eps * wet::abs(Ldq.d + Ldq.q)) {
        return T{0}; // non-salient: MTPA is id = 0
    }
    const T disc = wet::hypot(lambda, T{2} * Lsal * iq); // √(λ² + 4 Lₛₐₗ² iq²)
    return (lambda - disc) / (T{2} * Lsal);
}

/**
 * @brief MTPA dq current reference for a commanded torque
 * @ingroup foc_design
 *
 * Returns the @f$ (i_d, i_q) @f$ operating point that produces @p Te with the
 * smallest current magnitude. For a non-salient machine this is exact in closed
 * form (@f$ i_d = 0,\ i_q = T_e / K_t @f$, reusing @ref iq_from_torque). For a
 * salient machine the q-axis current must satisfy the implicit torque balance
 * @f$ T_e = \frac{3}{2}p\,[\lambda i_q - (L_q-L_d)\,i_d(i_q)\,i_q] @f$ along the
 * MTPA locus @ref mtpa_id_from_iq, solved by Newton iteration from the non-salient
 * seed @f$ T_e/K_t @f$. The torque is monotonic along the locus, so the iteration
 * converges in a handful of steps; @p iters is fixed (no convergence test) to keep
 * the call branch-free and @c constexpr-foldable.
 *
 * Field weakening above base speed is **not** applied here — this is the
 * unconstrained MTPA point. Bound it to the inverter with
 * @ref voltage_circle_radius / @ref base_speed in the caller (a later layer).
 *
 * @param Te          [Nm] commanded electromagnetic torque (signed)
 * @param lambda      [Wb] permanent-magnet flux linkage
 * @param Ldq         [H]  dq inductance pair @f$ (L_d, L_q) @f$
 * @param pole_pairs  @f$ p @f$ number of pole pairs
 * @param iters       [-]  Newton iterations for the salient solve (default 8)
 * @return @f$ (i_d, i_q) @f$ [A] on the MTPA locus
 */
template<typename T = double>
[[nodiscard]] constexpr DirectQuadrature<T>
mtpa_reference(T Te, T lambda, const DirectQuadrature<T>& Ldq, T pole_pairs, int iters = 8) {
    T       iq = iq_from_torque(Te, pole_pairs, lambda); // non-salient seed = Te / Kt
    const T Lsal = Ldq.q - Ldq.d;
    if (wet::abs(Lsal) <= T{1e-6} * wet::abs(Ldq.d + Ldq.q)) {
        return {.d = T{0}, .q = iq}; // SPMSM: seed is already exact
    }

    const T k = T{1.5} * pole_pairs; // torque per [λ iq + (Ld-Lq) id iq]
    for (int n = 0; n < iters; ++n) {
        const T disc = wet::hypot(lambda, T{2} * Lsal * iq);
        const T id = (lambda - disc) / (T{2} * Lsal);
        const T torque = k * ((lambda * iq) - (Lsal * id * iq));
        // d(torque)/d(iq) along the locus, with d(id)/d(iq) = -2 Lₛₐₗ iq / disc.
        const T dtorque = k * (lambda - (Lsal * id) + (T{2} * Lsal * Lsal * iq * iq / disc));
        iq -= (torque - Te) / dtorque;
    }
    return {.d = mtpa_id_from_iq(iq, lambda, Ldq), .q = iq};
}

} // namespace design

namespace motor {

/**
 * @brief Maximum-torque-per-ampere current-reference generator (PMSM / IPMSM / SynRM)
 *
 * The block form of @ref design::mtpa_reference: holds the machine nameplate and
 * maps a torque command to the minimum-current @f$ (i_d, i_q) @f$ reference for an
 * inner current loop (e.g. @ref FOController). Stateless — a cached functor — so it
 * is cheap to call every control tick and re-tunable by reassigning the nameplate.
 *
 * For a surface-PM machine the reference is the familiar @f$ i_d = 0 @f$; for a
 * salient machine it drives @f$ i_d < 0 @f$ to harvest reluctance torque, cutting
 * the current (and copper loss) for the same torque. This is the below-base-speed
 * reference; voltage-circle field weakening above base speed is a future layer
 * (the hooks are @ref design::voltage_circle_radius / @ref design::base_speed).
 *
 * @see design::mtpa_reference — the closed-form / Newton law this wraps.
 *
 * @tparam T Scalar type
 */
template<typename T = float>
class MtpaReference {
public:
    constexpr MtpaReference() = default;

    /**
     * @param Ldq        [H]  dq inductance pair @f$ (L_d, L_q) @f$
     * @param lambda     [Wb] permanent-magnet flux linkage
     * @param pole_pairs @f$ p @f$ number of pole pairs
     */
    constexpr MtpaReference(DirectQuadrature<T> Ldq, T lambda, T pole_pairs)
        : Ldq_(Ldq), lambda_(lambda), pole_pairs_(pole_pairs) {}

    /// dq current reference [A] for a commanded torque [Nm] (signed).
    [[nodiscard]] constexpr DirectQuadrature<T> operator()(T torque_cmd) const {
        return design::mtpa_reference(torque_cmd, lambda_, Ldq_, pole_pairs_);
    }

    [[nodiscard]] constexpr const DirectQuadrature<T>& inductance() const { return Ldq_; }

    [[nodiscard]] constexpr T flux_linkage() const { return lambda_; }
    [[nodiscard]] constexpr T pole_pairs() const { return pole_pairs_; }

private:
    DirectQuadrature<T> Ldq_{};
    T                   lambda_{};
    T                   pole_pairs_{T{1}};
};

} // namespace motor
} // namespace wet
