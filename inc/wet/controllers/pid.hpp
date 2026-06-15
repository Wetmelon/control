#pragma once

#include <cstdint>
#include <limits>

#include "wet/backend.hpp"

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
    T u_min = -std::numeric_limits<T>::infinity();
    T u_max = std::numeric_limits<T>::infinity();
    T i_min = -std::numeric_limits<T>::infinity();
    T i_max = std::numeric_limits<T>::infinity();
    T Kbc = T{0}; ///< Back-calculation anti-windup gain
    T b = T{1};   ///< Proportional setpoint weight (0=I-PD, 1=standard PID)
    T c = T{1};   ///< Derivative setpoint weight   (0=PI-D, 1=standard PID)

    template<typename U>
    [[nodiscard]] constexpr auto as() const {
        return PIDResult<U>{(U)Kp, (U)Ki, (U)Kd, (U)u_min, (U)u_max, (U)i_min, (U)i_max, (U)Kbc, (U)b, (U)c};
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
 *
 * @return PIDResult with the specified parameters
 */
template<typename T = double>
[[nodiscard]] constexpr PIDResult<T> pid(
    T Kp, T Ki, T Kd,
    T u_min = -std::numeric_limits<T>::infinity(),
    T u_max = std::numeric_limits<T>::infinity(),
    T i_min = -std::numeric_limits<T>::infinity(),
    T i_max = std::numeric_limits<T>::infinity(),
    T Kbc = T{0},
    T b = T{1},
    T c = T{1}
) {
    return PIDResult<T>{Kp, Ki, Kd, u_min, u_max, i_min, i_max, Kbc, b, c};
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
 */
template<typename T>
struct PIDController<T, PIDMode::PID> {
    T Kp{};
    T Ki{};
    T Kd{};
    T u_min = -std::numeric_limits<T>::infinity();
    T u_max = std::numeric_limits<T>::infinity();
    T i_min = -std::numeric_limits<T>::infinity();
    T i_max = std::numeric_limits<T>::infinity();
    T Kbc = T{0}; ///< Back-calculation anti-windup gain
    T b = T{1};   ///< Proportional setpoint weight
    T c = T{1};   ///< Derivative setpoint weight

    T integral = T{0};        ///< Integrator state
    T prev_cr_minus_y = T{0}; ///< Previous value of (c*r - y) for derivative

    PIDRuntimeMode runtime_mode{PIDRuntimeMode::Auto}; ///< Auto (control) or Tracking (follow u_track)
    T              u_track{T{0}};                      ///< External tracking signal (used only in Tracking mode)

    constexpr PIDController() = default;

    constexpr explicit PIDController(const design::PIDResult<T>& result)
        : Kp(result.Kp), Ki(result.Ki), Kd(result.Kd), u_min(result.u_min), u_max(result.u_max), i_min(result.i_min), i_max(result.i_max), Kbc(result.Kbc), b(result.b), c(result.c) {}

    template<typename U>
    constexpr explicit PIDController(const PIDController<U, PIDMode::PID>& other)
        : Kp(other.Kp), Ki(other.Ki), Kd(other.Kd), u_min(other.u_min), u_max(other.u_max), i_min(other.i_min), i_max(other.i_max), Kbc(other.Kbc), b(other.b), c(other.c), integral(other.integral), prev_cr_minus_y(other.prev_cr_minus_y), runtime_mode(other.runtime_mode), u_track(other.u_track) {}

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
        const T cr_minus_y = (c * r) - y;
        const T derivative = (cr_minus_y - prev_cr_minus_y) / Ts;
        prev_cr_minus_y = cr_minus_y; // updated in both modes for bumpless re-engagement

        if (runtime_mode == PIDRuntimeMode::Tracking) {
            // Preload integrator so a future Auto tick would produce u_track:
            //   u_would_be = Kp*(b*r - y) + Ki*integral + Kd*derivative ≡ u_track
            if (Ki != T{0}) {
                const T target = (u_track - (Kp * ((b * r) - y)) - (Kd * derivative)) / Ki;
                integral = wet::clamp(target, i_min, i_max);
            }
            return wet::clamp(u_track, u_min, u_max);
        }

        const T e = r - y;
        const T u_unsat = (Kp * ((b * r) - y)) + (Ki * integral) + (Kd * derivative);
        const T u = wet::clamp(u_unsat, u_min, u_max);

        if (Kbc != T{0}) {
            integral += Ts * (e + ((u - u_unsat) / Kbc));
        } else {
            integral += e * Ts;
        }
        integral = wet::clamp(integral, i_min, i_max);

        return u;
    }

    constexpr void reset() {
        integral = T{0};
        prev_cr_minus_y = T{0};
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
     * does not push further into saturation. When `Kbc == 0` falls back to a
     * straight conditional rollback equal to one sample's worth of unwind.
     */
    constexpr void back_calculate(T u_unsat, T u_sat, T Ts) {
        if (u_unsat == u_sat) {
            return;
        }
        if (Kbc != T{0}) {
            integral += Ts * ((u_sat - u_unsat) / Kbc);
        } else {
            integral += Ts * (u_sat - u_unsat);
        }
        integral = wet::clamp(integral, i_min, i_max);
    }
};

/**
 * @ingroup discrete_controllers
 * @brief Discrete 2-DOF PI controller specialization.
 */
template<typename T>
struct PIDController<T, PIDMode::PI> {
    T Kp{};
    T Ki{};
    T u_min = -std::numeric_limits<T>::infinity();
    T u_max = std::numeric_limits<T>::infinity();
    T i_min = -std::numeric_limits<T>::infinity();
    T i_max = std::numeric_limits<T>::infinity();
    T Kbc = T{0}; ///< Back-calculation anti-windup gain
    T b = T{1};   ///< Proportional setpoint weight

    T integral = T{0}; ///< Integrator state

    PIDRuntimeMode runtime_mode{PIDRuntimeMode::Auto}; ///< Auto (control) or Tracking (follow u_track)
    T              u_track{T{0}};                      ///< External tracking signal (used only in Tracking mode)

    constexpr PIDController() = default;

    constexpr explicit PIDController(const design::PIDResult<T>& result)
        : Kp(result.Kp), Ki(result.Ki), u_min(result.u_min), u_max(result.u_max), i_min(result.i_min), i_max(result.i_max), Kbc(result.Kbc), b(result.b) {}

    template<typename U>
    constexpr explicit PIDController(const PIDController<U, PIDMode::PI>& other)
        : Kp(other.Kp), Ki(other.Ki), u_min(other.u_min), u_max(other.u_max), i_min(other.i_min), i_max(other.i_max), Kbc(other.Kbc), b(other.b), integral(other.integral), runtime_mode(other.runtime_mode), u_track(other.u_track) {}

    /**
     * @brief Compute PI control output.
     *
     * In Auto mode runs the standard control law with back-calculation
     * anti-windup. In Tracking mode the output is `clamp(u_track, u_min, u_max)`
     * and the integrator is pre-loaded for bumpless re-engagement.
     */
    [[nodiscard]] constexpr T control(T r, T y, T Ts) {
        if (runtime_mode == PIDRuntimeMode::Tracking) {
            if (Ki != T{0}) {
                const T target = (u_track - (Kp * ((b * r) - y))) / Ki;
                integral = wet::clamp(target, i_min, i_max);
            }
            return wet::clamp(u_track, u_min, u_max);
        }

        const T e = r - y;
        const T u_unsat = (Kp * ((b * r) - y)) + (Ki * integral);
        const T u = wet::clamp(u_unsat, u_min, u_max);

        if (Kbc != T{0}) {
            integral += Ts * (e + ((u - u_unsat) / Kbc));
        } else {
            integral += e * Ts;
        }
        integral = wet::clamp(integral, i_min, i_max);
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
        if (u_unsat == u_sat) {
            return;
        }
        if (Kbc != T{0}) {
            integral += Ts * ((u_sat - u_unsat) / Kbc);
        } else {
            integral += Ts * (u_sat - u_unsat);
        }
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
    T u_min = -std::numeric_limits<T>::infinity();
    T u_max = std::numeric_limits<T>::infinity();
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
