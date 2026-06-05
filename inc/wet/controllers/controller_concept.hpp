#pragma once

/**
 * @file controller_concept.hpp
 * @brief Concepts describing the SISO controller protocol used across the
 *        library, including the optional anti-windup hook that lets
 *        @ref Cascade propagate saturation feedback through user-defined
 *        controllers.
 *
 * The library composes controllers by *protocol*, not inheritance. Any class
 * that advertises the right member signatures plays in `Cascade`, in tuning
 * harnesses, in mock plants for tests, etc. — including user-written
 * controllers (e.g. an MPPT conductance tracker, a stateful gain-scheduled
 * integrator, a non-PID compensator).
 *
 * Two concept levels:
 *
 * - @ref SISOController: the minimum surface — `T control(R r, Y y)` and
 *   `void reset()`. Any block that produces a command from a reference and a
 *   measurement satisfies this. Stateless controllers (`PController`) satisfy
 *   it without doing anything else.
 *
 * - @ref SISOControllerWithBackCalculation: refines the above with
 *   `void back_calculate(T u_unsat, T u_sat)`. When a downstream stage cannot
 *   deliver the commanded value (cascade clamps `r_inner`, plant input
 *   saturates, master rate-limit fires), the upstream controller is told
 *   *what was actually realized* so it can unwind its integrator / freeze
 *   internal state. Implementing this turns a passive block into a
 *   well-behaved member of a cascaded loop.
 *
 * Stateful user controllers should implement `back_calculate` to whatever
 * makes sense for their internal dynamics:
 * - PI / PID: `integral += (u_sat - u_unsat) * Ts / Kbc` (textbook
 *   back-calculation; reduces to a clamp-and-rollback when `Kbc → ∞`).
 * - MPPT (incremental conductance): freeze the conductance estimate update on
 *   the next tick when `u_sat != u_unsat`, since the y signal won't reflect
 *   the commanded perturbation.
 * - Pure proportional / open-loop blocks: no internal state to wind up; the
 *   hook is unnecessary (don't satisfy the refining concept).
 *
 * @see cascade.hpp for how `Cascade<Outer, Inner>` uses these concepts to
 *      propagate saturation feedback across the loop boundary.
 * @see "Advanced PID Control" (Åström & Hägglund, 2006), §6 on integrator
 *      windup and back-calculation for the math behind PI/PID's hook.
 */

#include <concepts>

namespace wet {

/**
 * @brief A SISO controller usable as a block in cascades and tuning harnesses.
 *
 * Requires:
 *  - `T control(R r, Y y)` — compute the command for reference `r` and
 *    measurement `y`. Return type may be any scalar type the consumer can use.
 *  - `void reset()` — zero out internal state.
 *
 * The reference and measurement may be different types (e.g. position
 * reference, velocity measurement) — the concept does not constrain them.
 *
 * @tparam C Controller type.
 * @tparam R Reference type (scalar; default `float`).
 * @tparam Y Measurement type (scalar; default `R`).
 */
template<typename C, typename R = float, typename Y = R>
concept SISOController = requires(C controller, R r, Y y) {
    { controller.control(r, y) };
    { controller.reset() } -> std::same_as<void>;
};

/**
 * @brief A @ref SISOController that exposes an anti-windup hook.
 *
 * Implementing this concept lets the controller participate in cascade-level
 * saturation propagation. The cascade calls `back_calculate(u_unsat, u_sat)`
 * after a downstream stage clamps or rejects the commanded value, so the
 * upstream controller can react — typically by unwinding integral state or
 * freezing parameter updates that would otherwise drift on a stale
 * measurement.
 *
 * The two scalars represent the upstream's pre-clamp command (`u_unsat`) and
 * the value actually realized downstream (`u_sat`). When they agree the hook
 * may be skipped by the caller; when they differ, the magnitude `u_sat -
 * u_unsat` is the saturation amount.
 *
 * @tparam C Controller type.
 * @tparam R Reference type.
 * @tparam Y Measurement type.
 * @tparam U Command type — typically the same as the controller's `control()`
 *           return type. Defaults to `R`.
 */
template<typename C, typename R = float, typename Y = R, typename U = R>
concept SISOControllerWithBackCalculation = SISOController<C, R, Y>
                                         && requires(C controller, U u_unsat, U u_sat) {
                                                { controller.back_calculate(u_unsat, u_sat) } -> std::same_as<void>;
                                            };

/**
 * @brief A controller that exposes runtime enable / disable mode control.
 *
 * Implementing this concept lets a controller be sidelined (driven from an
 * external command source) and later re-engaged without bumping the
 * commanded value. The classic industrial-PID pattern:
 *
 * - `disable(u_track)` switches the controller to tracking mode. While
 *   tracking, `control()` returns `u_track` (clamped to the controller's
 *   own limits) and internal state (integrator, derivative-state) is
 *   pre-loaded so a future Auto tick produces `u_track`.
 * - `enable()` returns to Auto mode. Because the internal state was
 *   pre-loaded each tick during tracking, the first Auto command matches
 *   the tracking value -- bumpless transfer.
 * - `is_enabled()` lets callers query the current mode.
 *
 * Semantically `u_track` is the same quantity as the `u_sat` argument of
 * @ref SISOControllerWithBackCalculation -- both represent the value
 * actually being applied to the plant. The difference is timing:
 * `back_calculate` is called after `control()` in the same tick when the
 * value was clamped downstream; `disable(u_track)` is called between
 * ticks when an external command source has taken over.
 *
 * Orthogonal to @ref SISOController -- mode control can apply to any
 * controller, not only SISO ones. Compose the two at the use site
 * (`SISOController<C, R, Y> && SISOControllerWithModeControl<C, U>`) when
 * both surfaces are needed.
 *
 * @tparam C Controller type.
 * @tparam U Tracking-signal type. Defaults to `float`.
 */
template<typename C, typename U = float>
concept SISOControllerWithModeControl = requires(C controller, U track) {
    { controller.enable() };
    { controller.disable(track) };
    { controller.is_enabled() } -> std::convertible_to<bool>;
};

} // namespace wet
