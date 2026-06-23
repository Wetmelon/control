#pragma once

#include <cstdint>
#include <limits>

#include "wet/backend.hpp"
#include "wet/systems/transfer_function.hpp"

namespace wet {
namespace design {

/**
 * @struct PIDResult
 * @brief 2-DOF PID controller design result
 *
 * Contains gains and setpoint weights for a two-degree-of-freedom PID controller.
 * The setpoint weights `b` and `c` decouple reference tracking from disturbance
 * rejection, allowing each to be tuned independently.
 *
 * Special cases:
 * - b=1, c=1: Standard PID (P and D on error)
 * - b=1, c=0: PI-D (D on measurement - no derivative kick)
 * - b=0, c=0: I-PD (P and D on measurement - no setpoint kick)
 *
 * @see Astrom & Hagglund, "Advanced PID Control" (2006), Sec. 4.4
 */
template<typename T = double>
struct PIDResult {
    T Kp{};
    T Ki{};
    T Kd{};
    T u_min = -std::numeric_limits<T>::max();
    T u_max = std::numeric_limits<T>::max();
    T i_min = -std::numeric_limits<T>::max();
    T i_max = std::numeric_limits<T>::max();
    T Kbc = T{0}; ///< Back-calculation anti-windup gain
    T b = T{1};   ///< Proportional setpoint weight (0=I-PD, 1=standard PID)
    T c = T{1};   ///< Derivative setpoint weight   (0=PI-D, 1=standard PID)
    T Tf = T{0};  ///< Derivative filter time constant (0 = unfiltered)

    template<typename U>
    [[nodiscard]] constexpr auto as() const {
        return PIDResult<U>{
            static_cast<U>(Kp), static_cast<U>(Ki), static_cast<U>(Kd),
            static_cast<U>(u_min), static_cast<U>(u_max), static_cast<U>(i_min), static_cast<U>(i_max),
            static_cast<U>(Kbc), static_cast<U>(b), static_cast<U>(c), static_cast<U>(Tf)
        };
    }

    /**
     * @brief Continuous-time controller transfer function C(s).
     *
     * Returns @f$C(s) = K_p + K_i/s + K_d s/(1 + T_f s)@f$ as a 2nd-order TF in
     * ascending powers of s, so PID drops into the analysis tooling (Bode,
     * `series`/`feedback`, `discretize`) like the lead-lag / PR design results.
     * `Tf = 0` gives the ideal form @f$(K_d s^2 + K_p s + K_i)/s@f$.
     */
    [[nodiscard]] constexpr TransferFunction<3, 3, T> to_tf() const {
        return TransferFunction<3, 3, T>{
            .num = {Ki, Kp + (Ki * Tf), (Kp * Tf) + Kd},
            .den = {T{0}, T{1}, Tf},
        };
    }
};

/**
 * @brief 2-DOF PID controller design
 *
 * Constructs a PIDResult with the given gains and optional setpoint weights.
 *
 * The control law is:
 *
 *     u = Kp(b*r - y) + Ki*integral(r-y)dt + Kd*d/dt(c*r - y)
 *
 * @param Kp    Proportional gain
 * @param Ki    Integral gain
 * @param Kd    Derivative gain
 * @param u_min Minimum control output
 * @param u_max Maximum control output
 * @param i_min Minimum integrator value
 * @param i_max Maximum integrator value
 * @param Kbc   Back-calculation anti-windup gain (0 = clamping only)
 * @param b     Proportional setpoint weight (default: 1 - standard PID)
 * @param c     Derivative setpoint weight   (default: 1 - standard PID)
 * @param Tf    Derivative filter time constant (default: 0 - unfiltered)
 *
 * @return PIDResult with the specified parameters
 */
template<typename T = double>
[[nodiscard]] constexpr PIDResult<T> pid(
    T Kp, T Ki, T Kd,
    T u_min = -std::numeric_limits<T>::max(),
    T u_max = std::numeric_limits<T>::max(),
    T i_min = -std::numeric_limits<T>::max(),
    T i_max = std::numeric_limits<T>::max(),
    T Kbc = T{0},
    T b = T{1},
    T c = T{1},
    T Tf = T{0}
) {
    return PIDResult<T>{Kp, Ki, Kd, u_min, u_max, i_min, i_max, Kbc, b, c, Tf};
}

} // namespace design

/**
 * @brief Compile-time selection of the PID control-law structure.
 *
 * Picks which specialization of @ref PIDController is instantiated. Distinct
 * from @ref PIDRuntimeMode, which selects runtime behavior (Auto vs Tracking)
 * at each tick.
 */
enum class PIDMode : std::uint8_t {
    P,
    PI,
    PID,
};

/**
 * @brief Runtime operating mode for @ref PIDController.
 *
 * - `Auto`:     standard control law with back-calculation anti-windup. The
 *               controller follows the reference.
 * - `Tracking`: the controller's output is `clamp(u_track, u_min, u_max)`
 *               and on every tick the integrator (and derivative state) is
 *               pre-loaded so that the next `enable()` call resumes in `Auto`
 *               *without a bump* in command. This is the standard ISA-PID
 *               pattern for operator manual mode, master/follower handoff,
 *               and multi-controller bumpless transfer.
 *
 * Semantically `u_track` is the same quantity as the `u_sat` argument of
 * @ref SISOControllerWithBackCalculation::back_calculate — both represent the
 * value actually being applied to the plant. Use `back_calculate` when the
 * controller stays in `Auto` but its output was clamped downstream; use
 * `disable(u_track)` when the controller is being temporarily replaced by
 * another command source.
 */
enum class PIDRuntimeMode : std::uint8_t {
    Auto,
    Tracking,
};

template<typename T = float, PIDMode Mode = PIDMode::PID>
struct PIDController;

/**
 * @ingroup discrete_controllers
 * @brief Discrete 2-DOF PID controller specialization.
 *
 * Control law (2-DOF, optional first-order derivative filter):
 * @f[
 *   u = K_p(b\,r - y) + K_i\!\int (r-y)\,dt + \frac{K_d\, s}{1 + T_f s}(c\,r - y)
 * @f]
 *
 * @code
 * auto                k = design::pid(2.0, 5.0, 0.1); // Kp, Ki, Kd
 * PIDController<double> c(k);
 * double              u = c.control(r, y, Ts);
 * @endcode
 *
 * MATLAB equivalent: `C = pid(Kp, Ki, Kd, Tf)` (2-DOF weights via `pid2`).
 * @see Åström & Hägglund, "Advanced PID Control" (2006), Sec. 3.3–4.4
 *
 * @note The sample time @p Ts is supplied per call to control() / back_calculate(),
 *       not stored on the controller. The gains are continuous-time: the integrator
 *       uses backward (implicit) Euler — the current error is folded in
 *       (`integral += e*Ts`) *before* it forms the output, so this tick's error acts
 *       this tick (no forward-Euler one-sample lag) — and the derivative is `Δ/Ts`.
 *       Each tick uses the *actual* elapsed period, so the loop tolerates jitter and
 *       multi-rate execution with no reconfiguration. A non-positive @p Ts holds the
 *       state and returns the current command (no divide-by-zero, no backward step).
 *
 * @note Consequence for tuning: gains from a continuous-time designer
 *       (design::ziegler_nichols, cohen_coon, simc, lambda_tuning, pid_from_bandwidth)
 *       are Ts-agnostic, but gains from a *discretizing* designer —
 *       design::pid_pole_placement, which bakes Ts into `a = exp(-Ts/τ)` — are valid
 *       only at their design Ts; pass that same Ts at runtime or the pole placement is
 *       invalidated.
 */
template<typename T>
struct PIDController<T, PIDMode::PID> {
    T Kp{};
    T Ki{};
    T Kd{};
    T u_min = -std::numeric_limits<T>::max();
    T u_max = std::numeric_limits<T>::max();
    T i_min = -std::numeric_limits<T>::max();
    T i_max = std::numeric_limits<T>::max();
    T Kbc = T{0}; ///< Back-calculation anti-windup gain
    T b = T{1};   ///< Proportional setpoint weight
    T c = T{1};   ///< Derivative setpoint weight
    T Tf = T{0};  ///< Derivative filter time constant (0 = unfiltered)

    T    integral = T{0};        ///< Integrator state
    T    prev_cr_minus_y = T{0}; ///< Previous value of (c*r - y) for derivative
    T    deriv = T{0};           ///< Filtered derivative term (carries Kd)
    bool first_ = true;          ///< Seed derivative history on the first control() tick

    PIDRuntimeMode runtime_mode{PIDRuntimeMode::Auto}; ///< Auto (control) or Tracking (follow u_track)
    T              u_track{T{0}};                      ///< External tracking signal (used only in Tracking mode)

    constexpr PIDController() = default;

    constexpr explicit PIDController(const design::PIDResult<T>& result)
        : Kp(result.Kp),
          Ki(result.Ki),
          Kd(result.Kd),
          u_min(result.u_min),
          u_max(result.u_max),
          i_min(result.i_min),
          i_max(result.i_max),
          Kbc(result.Kbc),
          b(result.b),
          c(result.c),
          Tf(result.Tf) {}

    template<typename U>
    constexpr explicit PIDController(const PIDController<U, PIDMode::PID>& other)
        : Kp(other.Kp),
          Ki(other.Ki),
          Kd(other.Kd),
          u_min(other.u_min),
          u_max(other.u_max),
          i_min(other.i_min),
          i_max(other.i_max),
          Kbc(other.Kbc),
          b(other.b),
          c(other.c),
          Tf(other.Tf),
          integral(other.integral),
          prev_cr_minus_y(other.prev_cr_minus_y),
          deriv(other.deriv),
          first_(other.first_),
          runtime_mode(other.runtime_mode),
          u_track(other.u_track) {}

    /**
     * @brief Compute 2-DOF PID control output.
     *
     * In Auto mode runs the standard 2-DOF control law with back-calculation
     * anti-windup. In Tracking mode the output is `clamp(u_track, u_min, u_max)`
     * and the integrator (plus derivative-state) is pre-loaded so that
     * `enable()` resumes Auto without a bump in command. The derivative-state
     * update happens in both modes so that a Tracking-to-Auto transition
     * doesn't kick the derivative term with a stale prev value.
     */
    [[nodiscard]] constexpr T control(T r, T y, T Ts) {
        // Guard a non-positive sample time (uninitialized / stalled dt): the
        // derivative divides by Ts and the integrator steps by Ts, so a bad dt would
        // produce inf/NaN or integrate backwards. Hold all state and emit the current
        // command, preserving the active mode.
        if (Ts <= T{0}) {
            if (runtime_mode == PIDRuntimeMode::Tracking) {
                return wet::clamp(u_track, u_min, u_max);
            }
            return wet::clamp((Kp * ((b * r) - y)) + (Ki * integral), u_min, u_max);
        }

        const T cr_minus_y = (c * r) - y;
        if (first_) {
            prev_cr_minus_y = cr_minus_y; // seed on the first tick: no derivative kick
            first_ = false;
        }
        const T dX = cr_minus_y - prev_cr_minus_y;
        prev_cr_minus_y = cr_minus_y; // updated in both modes for bumpless re-engagement

        // Filtered derivative term D = Kd*s/(1 + Tf*s) on (c*r - y), backward Euler.
        // Tf == 0 reduces exactly to the raw Kd*(ΔX)/Ts. Updated in both modes so a
        // Tracking->Auto transition doesn't kick the derivative with a stale value.
        deriv = ((Tf * deriv) + (Kd * dX)) / (Ts + Tf);

        const T e = r - y;

        if (runtime_mode == PIDRuntimeMode::Tracking) {
            // Preload so a re-enabled Auto tick reproduces u_track. Auto integrates
            // with backward Euler (integral += e*Ts before the output is formed), so
            // store the pre-integration value: target_post - e*Ts.
            if (Ki != T{0}) {
                const T target = ((u_track - (Kp * ((b * r) - y)) - deriv) / Ki) - (e * Ts);
                integral = wet::clamp(target, i_min, i_max);
            }
            return wet::clamp(u_track, u_min, u_max);
        }

        // Backward (implicit) Euler: fold the current error into the integrator
        // before it drives the output, so this tick's error acts this tick rather
        // than one sample later (the forward-Euler 1-tick delay). Clamp immediately
        // so the output reflects the bounded integrator on the same tick.
        integral += e * Ts;
        integral = wet::clamp(integral, i_min, i_max);
        const T u_unsat = (Kp * ((b * r) - y)) + (Ki * integral) + deriv;
        const T u = wet::clamp(u_unsat, u_min, u_max);

        // Back-calculation anti-windup: bleed the saturation excess back out of the
        // integrator (effective next tick). Kbc == 0 -> plain integrator clamping.
        if (Kbc != T{0}) {
            integral += (Ts / Kbc) * (u - u_unsat);
            integral = wet::clamp(integral, i_min, i_max);
        }

        return u;
    }

    constexpr void reset() {
        integral = T{0};
        prev_cr_minus_y = T{0};
        deriv = T{0};
        first_ = true;
        // Mode and u_track preserved -- reset is for clearing accumulated state,
        // not for operator-mode changes.
    }

    /// Switch to Auto mode. Integrator state preserved (which is the point of
    /// the bumpless preload that happened while Tracking).
    constexpr void enable() { runtime_mode = PIDRuntimeMode::Auto; }

    /// Switch to Tracking mode. `track` is the value being applied to the
    /// plant while this controller is sidelined -- typically the master /
    /// alternate controller's actual output. Same semantic as `u_sat` in
    /// `back_calculate`.
    constexpr void disable(T track) {
        runtime_mode = PIDRuntimeMode::Tracking;
        u_track = track;
    }

    [[nodiscard]] constexpr bool is_enabled() const { return runtime_mode == PIDRuntimeMode::Auto; }

    /**
     * @brief Anti-windup hook driven by a downstream stage.
     *
     * Call this when the value returned from `control()` was further clamped
     * or rejected outside this controller -- e.g. a cascade clamped the
     * inner reference, or a master rate limiter capped the command. Winds
     * down the integrator by `(u_sat - u_unsat) * Ts / Kbc` so the next tick
     * does not push further into saturation. No-op when `Kbc == 0`
     * (back-calculation not configured), matching the in-loop path which then
     * relies on the integrator clamp alone.
     */
    constexpr void back_calculate(T u_unsat, T u_sat, T Ts) {
        if (Kbc == T{0} || u_unsat == u_sat) {
            return;
        }
        integral += Ts * ((u_sat - u_unsat) / Kbc);
        integral = wet::clamp(integral, i_min, i_max);
    }
};

/**
 * @ingroup discrete_controllers
 * @brief Discrete 2-DOF PI controller specialization.
 *
 * @note @p Ts is supplied per call to control() / back_calculate(), not stored; a
 *       non-positive @p Ts holds the state and returns the current command. See
 *       PIDController<T, PIDMode::PID> for the rate-handling rationale and the
 *       discretization caveat (gains from design::pid_pole_placement are Ts-locked).
 */
template<typename T>
struct PIDController<T, PIDMode::PI> {
    T Kp{};
    T Ki{};
    T u_min = -std::numeric_limits<T>::max();
    T u_max = std::numeric_limits<T>::max();
    T i_min = -std::numeric_limits<T>::max();
    T i_max = std::numeric_limits<T>::max();
    T Kbc = T{0}; ///< Back-calculation anti-windup gain
    T b = T{1};   ///< Proportional setpoint weight

    T integral = T{0}; ///< Integrator state

    PIDRuntimeMode runtime_mode{PIDRuntimeMode::Auto}; ///< Auto (control) or Tracking (follow u_track)
    T              u_track{T{0}};                      ///< External tracking signal (used only in Tracking mode)

    constexpr PIDController() = default;

    constexpr explicit PIDController(const design::PIDResult<T>& result)
        : Kp(result.Kp),
          Ki(result.Ki),
          u_min(result.u_min),
          u_max(result.u_max),
          i_min(result.i_min),
          i_max(result.i_max),
          Kbc(result.Kbc),
          b(result.b) {}

    template<typename U>
    constexpr explicit PIDController(const PIDController<U, PIDMode::PI>& other)
        : Kp(other.Kp),
          Ki(other.Ki),
          u_min(other.u_min),
          u_max(other.u_max),
          i_min(other.i_min),
          i_max(other.i_max),
          Kbc(other.Kbc),
          b(other.b),
          integral(other.integral),
          runtime_mode(other.runtime_mode),
          u_track(other.u_track) {}

    /**
     * @brief Compute PI control output.
     *
     * In Auto mode runs the standard control law with back-calculation
     * anti-windup. In Tracking mode the output is `clamp(u_track, u_min, u_max)`
     * and the integrator is pre-loaded for bumpless re-engagement.
     */
    [[nodiscard]] constexpr T control(T r, T y, T Ts) {
        // Guard a non-positive sample time (uninitialized / stalled dt): the
        // integrator steps by Ts, so a bad dt would integrate backwards. Hold state
        // and emit the current command, preserving the active mode.
        if (Ts <= T{0}) {
            if (runtime_mode == PIDRuntimeMode::Tracking) {
                return wet::clamp(u_track, u_min, u_max);
            }
            return wet::clamp((Kp * ((b * r) - y)) + (Ki * integral), u_min, u_max);
        }

        const T e = r - y;

        if (runtime_mode == PIDRuntimeMode::Tracking) {
            // Preload so a re-enabled Auto tick reproduces u_track. Auto integrates
            // with backward Euler (integral += e*Ts before the output), so store the
            // pre-integration value: target_post - e*Ts.
            if (Ki != T{0}) {
                const T target = ((u_track - (Kp * ((b * r) - y))) / Ki) - (e * Ts);
                integral = wet::clamp(target, i_min, i_max);
            }
            return wet::clamp(u_track, u_min, u_max);
        }

        // Backward (implicit) Euler: integrate the current error before it drives
        // the output, so it acts this tick rather than one sample later. Clamp
        // immediately so the output reflects the bounded integrator this tick.
        integral += e * Ts;
        integral = wet::clamp(integral, i_min, i_max);
        const T u_unsat = (Kp * ((b * r) - y)) + (Ki * integral);
        const T u = wet::clamp(u_unsat, u_min, u_max);

        // Back-calculation anti-windup (effective next tick); Kbc == 0 -> clamping.
        if (Kbc != T{0}) {
            integral += (Ts / Kbc) * (u - u_unsat);
            integral = wet::clamp(integral, i_min, i_max);
        }
        return u;
    }

    constexpr void reset() {
        integral = T{0};
    }

    constexpr void enable() { runtime_mode = PIDRuntimeMode::Auto; }
    constexpr void disable(T track) {
        runtime_mode = PIDRuntimeMode::Tracking;
        u_track = track;
    }
    [[nodiscard]] constexpr bool is_enabled() const { return runtime_mode == PIDRuntimeMode::Auto; }

    /**
     * @brief Anti-windup hook driven by a downstream stage.
     *
     * @see PIDController<T, PIDMode::PID>::back_calculate for semantics.
     */
    constexpr void back_calculate(T u_unsat, T u_sat, T Ts) {
        if (Kbc == T{0} || u_unsat == u_sat) {
            return;
        }
        integral += Ts * ((u_sat - u_unsat) / Kbc);
        integral = wet::clamp(integral, i_min, i_max);
    }
};

/**
 * @ingroup discrete_controllers
 * @brief Discrete proportional controller specialization.
 */
template<typename T>
struct PIDController<T, PIDMode::P> {
    T Kp{};
    T u_min = -std::numeric_limits<T>::max();
    T u_max = std::numeric_limits<T>::max();
    T b = T{1}; ///< Proportional setpoint weight

    PIDRuntimeMode runtime_mode{PIDRuntimeMode::Auto}; ///< Auto (control) or Tracking (follow u_track)
    T              u_track{T{0}};                      ///< External tracking signal (used only in Tracking mode)

    constexpr PIDController() = default;

    constexpr explicit PIDController(T Kp_)
        : Kp(Kp_) {}

    constexpr explicit PIDController(const design::PIDResult<T>& result)
        : Kp(result.Kp), u_min(result.u_min), u_max(result.u_max), b(result.b) {}

    template<typename U>
    constexpr explicit PIDController(const PIDController<U, PIDMode::P>& other)
        : Kp(other.Kp), u_min(other.u_min), u_max(other.u_max), b(other.b), runtime_mode(other.runtime_mode), u_track(other.u_track) {}

    /**
     * @brief Compute proportional control output.
     *
     * P-only has no integral / derivative state, so tracking mode is
     * trivially bumpless: in Tracking mode `control()` just returns
     * `clamp(u_track, u_min, u_max)`. The mode API is exposed for symmetry
     * with the PI / PID specializations so generic code can `disable()` /
     * `enable()` any PIDController uniformly.
     */
    [[nodiscard]] constexpr T control(T r, T y) {
        if (runtime_mode == PIDRuntimeMode::Tracking) {
            return wet::clamp(u_track, u_min, u_max);
        }
        return wet::clamp(Kp * ((b * r) - y), u_min, u_max);
    }

    /// (r, y, Ts) overload for uniform use across the PIDController family; Ts is
    /// ignored (P has no rate-dependent term).
    [[nodiscard]] constexpr T control(T r, T y, T /*Ts*/) { return control(r, y); }

    constexpr void reset() {}

    constexpr void enable() { runtime_mode = PIDRuntimeMode::Auto; }
    constexpr void disable(T track) {
        runtime_mode = PIDRuntimeMode::Tracking;
        u_track = track;
    }
    [[nodiscard]] constexpr bool is_enabled() const { return runtime_mode == PIDRuntimeMode::Auto; }
};

template<typename T>
PIDController(const design::PIDResult<T>&) -> PIDController<T, PIDMode::PID>;

template<typename T = float>
using PController = PIDController<T, PIDMode::P>;

template<typename T = float>
using PIController = PIDController<T, PIDMode::PI>;

} // namespace wet
