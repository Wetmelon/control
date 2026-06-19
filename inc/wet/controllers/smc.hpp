#pragma once

#include <type_traits>

#include "wet/math/math.hpp"

namespace wet {

namespace design {

/**
 * @struct SMCResult
 * @brief Tuning parameters for a first-order sliding-mode controller.
 *
 * Just the three numbers @ref SMCController needs. SMC has no gain *formula* the
 * way LQR or pole-placement do — you pick the sliding-surface slope @p lambda and
 * the switching gain @p k directly, so this struct is a plain parameter bundle
 * rather than the output of a synthesis step. Use `.as<U>()` to down-cast for an
 * embedded build (e.g. `double` design → `float` on an MCU).
 *
 * @tparam T Scalar type (default: double)
 */
template<typename T = double>
struct SMCResult {
    T lambda{}; //!< Sliding-surface slope λ in s = λ·e + ė [1/s]. Larger = faster but noisier.
    T k{};      //!< Switching gain (must exceed the disturbance/uncertainty bound) [output units].
    T b0{};     //!< Control effectiveness ṡ = b0·u + … ; the nominal plant input gain.

    /// Re-cast the parameters to a different scalar type (e.g. double → float).
    template<typename U>
    [[nodiscard]] constexpr auto as() const {
        return SMCResult<U>{lambda, k, b0};
    }
};

/**
 * @brief Bundle hand-picked SMC parameters into an @ref SMCResult.
 *
 * Convenience factory so call sites read `design::smc(λ, k, b0)` symmetrically
 * with the other controllers' design functions. There is no computation: SMC is
 * tuned by choosing @p lambda and @p k directly (see @ref SMCController for how
 * to pick them).
 *
 * @param lambda Sliding-surface slope λ [1/s] (> 0).
 * @param k      Switching gain (> the worst-case lumped disturbance).
 * @param b0     Control effectiveness / input gain (> 0).
 * @return SMCResult carrying the three parameters.
 */
template<typename T = double>
[[nodiscard]] constexpr SMCResult<T> smc(T lambda, T k, T b0) {
    return SMCResult<T>{lambda, k, b0};
}

} // namespace design

/**
 * @ingroup discrete_controllers
 * @brief First-order sliding-mode controller (SMC) for a SISO plant.
 *
 * Sliding-mode control is a *robust nonlinear* law: it forces the tracking error
 * onto a chosen line in the error/error-rate plane (the **sliding surface**) and
 * then keeps it there, regardless of bounded disturbances or model error. That
 * disturbance-rejection robustness is what you buy versus a PID.
 *
 * **How it works, step by step.** With error `e = r − y` and its rate `ė`, define
 * the sliding surface
 *
 * @f[ s = \lambda\,e + \dot e @f]
 *
 * `s = 0` is a line through the origin with slope `−λ`; on it the error decays as
 * `e(t) = e(0)\,\mathrm{e}^{-\lambda t}`, so **λ sets the closed-loop speed once
 * sliding**. The control drives `s` to zero with a relay (bang-bang) term:
 *
 * @f[ u = -\frac{k}{b_0}\,\operatorname{sign}(s) @f]
 *
 * As long as the switching gain `k` exceeds the worst-case lumped disturbance,
 * `s` reaches 0 in finite time and stays there. The price of the hard `sign()`
 * is **chattering** — buzzing as `u` slams between ±k/b₀ every sample. The
 * @p phi boundary layer fixes that: inside a band `|s| < phi` the relay is
 * replaced by a smooth ramp `s/(phi + |s|)`, trading a little steady-state
 * accuracy for a continuous command. For *continuous* control with no such
 * trade-off, see the super-twisting controller (@ref SuperTwistingController),
 * which hides the discontinuity under an integrator.
 *
 * **Picking the knobs (Arduino-friendly):**
 *  - `λ` — start near your desired bandwidth in rad/s; bigger is faster but
 *    amplifies measurement noise through `ė`.
 *  - `k` — increase until disturbances are rejected; if it buzzes, that's
 *    chattering — add a `phi` boundary layer rather than dropping `k`.
 *  - `b0` — your plant's input gain (how much `ṡ` moves per unit `u`); if unsure,
 *    set 1 and fold the scaling into `k`.
 *
 * @note `ė` is estimated by backward difference `(e − e_prev)/Ts`, so a noisy `y`
 *       feeds noise straight into the surface — filter the measurement (or its
 *       derivative) on a real sensor. `Ts` is passed per call (not stored), so
 *       the same controller works at any/varying loop rate.
 *
 * Example — regulate a position with a boundary layer to avoid actuator buzz:
 * @code
 * using namespace wet;
 * constexpr auto gains = design::smc(20.0f, 5.0f, 1.0f); // λ, k, b0
 * SMCController<float> ctrl(gains);
 * // In the loop: Ts = 1 ms, phi = 0.05 boundary layer to avoid chatter.
 * float u = ctrl.control(r, y, 1e-3f, 0.05f);
 * @endcode
 *
 * @tparam T Scalar type (default: float)
 *
 * @see SuperTwistingController for the chatter-free (second-order) sibling.
 * @see Utkin, "Variable Structure Systems with Sliding Modes," IEEE TAC 22(2),
 *      1977, https://doi.org/10.1109/TAC.1977.1101446
 * @see Slotine & Li, *Applied Nonlinear Control*, Prentice Hall, 1991 (ch. 7) —
 *      the standard textbook treatment of the surface/boundary-layer design.
 */
template<typename T = float>
class SMCController {
    T lambda{}; //!< Sliding-surface slope λ [1/s].
    T k{};      //!< Switching gain (> disturbance bound).
    T b0{};     //!< Control effectiveness / input gain.

    T error_prev{}; //!< Previous error, for the backward-difference rate ė.

public:
    constexpr SMCController() = default;

    constexpr SMCController(const design::SMCResult<T>& result)
        : lambda(result.lambda), k(result.k), b0(result.b0) {}

    template<typename U>
    constexpr SMCController(const SMCController<U>& other)
        : lambda(other.lambda), k(other.k), b0(other.b0), error_prev(other.error_prev) {}

    /**
     * @brief Run one control step: u = −(k/b0)·sign(λ·e + ė).
     *
     * @param r   Reference (setpoint).
     * @param y   Measurement (plant output).
     * @param Ts  Sample time [s] since the last call; used for the backward-
     *            difference rate `ė = (e − e_prev)/Ts`. Pass your actual loop
     *            period — it need not be constant.
     * @param phi Boundary-layer thickness (default 0 = hard `sign()`). Set
     *            `phi > 0` to soften the relay and suppress chattering; bigger
     *            `phi` = smoother command but larger steady-state error.
     *
     * @return Control command `u`.
     */
    [[nodiscard]] constexpr T control(T r, T y, T Ts, T phi = T{0.0}) {
        T error = r - y;

        // Compute derivative of error using backward difference
        T dot_e = (error - error_prev) / Ts;
        error_prev = error;

        // Define the "sliding surface"
        // Lambda weights error relative to its derivative.
        T s = (lambda * error) + dot_e;

        // Compute control using saturation function for boundary layer
        if (phi <= T{0.0}) {
            return -(k / b0) * wet::sgn(s);
        } else {
            return -(k / b0) * s / (phi + wet::abs(s));
        }
    }

    constexpr void reset() {
        error_prev = T{};
    }
};

} // namespace wet