#pragma once

/**
 * @file cascade.hpp
 * @brief Generic two-loop cascade composition for SISO controllers.
 *
 * Cascade control composes an outer and inner controller using either a
 * shared measurement signal or distinct measurements per loop:
 *
 * Shared measurement:
 *
 *     r_inner = outer.control(r, y)
 *     u       = inner.control(r_inner, y)
 *
 * Distinct measurements (with optional per-loop feedforward):
 *
 *     r_inner = outer.control(r, y_outer) + ff_inner
 *     u       = inner.control(r_inner, y_inner, ...)
 *
 * where `r` is the top-level reference, `y` is the measured output, and `u`
 * is the plant command.
 *
 * This header provides:
 * - `Cascade<Outer, Inner>`: generic runtime composition
 * - `CascadePPI<T>`: convenience alias outer-P / inner-PID
 * - `Cascade3<Outer, Middle, Inner>`: nested 3-layer composition helper
 *
 * @see Åström & Hägglund, "Advanced PID Control" (2006), cascade structures
 */

#include <concepts>
#include <limits>
#include <type_traits>

#include "wet/backend.hpp"
#include "wet/controllers/controller_concept.hpp"
#include "wet/controllers/pid.hpp"

namespace wet {

namespace detail {

template<typename Controller, typename R, typename Y>
using cascade_inner_ref_t = std::remove_cvref_t<decltype(std::declval<Controller&>().control(std::declval<R>(), std::declval<Y>()))>;

} // namespace detail

/**
 * @ingroup discrete_controllers
 * @brief Generic two-loop cascade composition with anti-windup propagation.
 *
 * The outer loop computes an inner reference from the top-level command and
 * measured output; the inner loop then computes the plant command.
 *
 *     r_inner_unsat = outer.control(r, y)
 *     r_inner       = clamp(r_inner_unsat, r_inner_min, r_inner_max)
 *     outer.back_calculate(r_inner_unsat, r_inner)    // when supported and clamped
 *     u             = inner.control(r_inner, y)
 *
 * For non-shared measurements, use the variadic overload (also clamps r_inner
 * and propagates back-calculation):
 *
 *     r_inner_unsat = outer.control(r, y_outer) + ff_inner
 *     r_inner       = clamp(r_inner_unsat, r_inner_min, r_inner_max)
 *     outer.back_calculate(r_inner_unsat, r_inner)    // when supported and clamped
 *     u             = inner.control(r_inner, y_inner, ...)
 *
 * Anti-windup propagation is opt-in: the outer controller only sees the
 * `back_calculate(u_unsat, u_sat)` callback when it satisfies
 * @ref SISOControllerWithBackCalculation. Stateless outers (`PController`) do
 * not need it; stateful PI/PID and user controllers (e.g. a stateful MPPT
 * tracker) implement the hook to freeze or unwind internal state when the
 * cascade clamps `r_inner`.
 *
 * This enables recursive multi-layer compositions where `Inner` is itself a
 * `Cascade<...>` instance (e.g. outer ESC → middle voltage → inner current).
 * Saturation propagates one layer at a time: the outermost cascade clamps
 * its inner reference; that inner (a sub-cascade) further clamps its own
 * inner reference; and so on.
 *
 * @tparam Outer Outer-loop controller type.
 * @tparam Inner Inner-loop controller type.
 * @tparam T     Scalar type used for the r_inner saturation limits.
 *               Defaults to `float`; pick `double` for design-time work.
 */
template<typename Outer, typename Inner, typename T = float>
class Cascade {
public:
    /// Lower clamp on the inner-loop reference; default is `-inf` (no clamp).
    T r_inner_min{-std::numeric_limits<T>::infinity()};
    /// Upper clamp on the inner-loop reference; default is `+inf` (no clamp).
    T r_inner_max{std::numeric_limits<T>::infinity()};

    constexpr Cascade() = default;

    /**
     * @brief Construct from explicit outer and inner controllers.
     */
    constexpr Cascade(const Outer& outer, const Inner& inner)
        : outer_(outer), inner_(inner) {}

    /**
     * @brief Construct with explicit r_inner clamp limits.
     */
    constexpr Cascade(const Outer& outer, const Inner& inner, T r_min, T r_max)
        : r_inner_min(r_min), r_inner_max(r_max), outer_(outer), inner_(inner) {}

    /**
     * @brief Cascade control tick with shared measurement.
     * @param r Top-level reference.
     * @param y Measurement shared by outer and inner loops.
     * @return Inner-loop output command.
     */
    template<typename R, typename Y>
        requires(SISOController<Outer, R, Y> && requires(Inner& inner, detail::cascade_inner_ref_t<Outer, R, Y> r_inner, Y y) {
            { inner.control(r_inner, y) };
            { inner.reset() } -> std::same_as<void>;
        })
    [[nodiscard]] constexpr auto control(R r, Y y) {
        using RInner = detail::cascade_inner_ref_t<Outer, R, Y>;
        const RInner r_inner_unsat = outer_.control(r, y);
        const RInner r_inner = clamp_inner_reference<RInner>(r_inner_unsat);
        propagate_back_calculation<R, Y, RInner>(r_inner_unsat, r_inner);
        return inner_.control(r_inner, y);
    }

    /**
     * @brief Cascade control tick with distinct measurements and feedforward.
     *
     * Computes `r_inner_unsat = outer.control(r, y_outer) + ff_inner`,
     * clamps to `[r_inner_min, r_inner_max]`, calls `outer.back_calculate`
     * if the outer supports it and the value was clamped, then forwards the
     * clamped reference plus remaining measurement arguments to the inner
     * layer.
     *
     * Set `ff_inner` to zero when feedforward is not used.
     */
    template<typename R, typename YOuter, typename FFInner, typename... InnerArgs>
        requires(sizeof...(InnerArgs) > 0) && SISOController<Outer, R, YOuter>
             && requires(detail::cascade_inner_ref_t<Outer, R, YOuter> r_inner, FFInner ff_inner) {
                    { r_inner + ff_inner };
                } && requires(Inner& inner, decltype(std::declval<detail::cascade_inner_ref_t<Outer, R, YOuter>>() + std::declval<FFInner>()) r_inner_ff, InnerArgs... args) {
                    { inner.control(r_inner_ff, args...) };
                    { inner.reset() } -> std::same_as<void>;
                }
    [[nodiscard]] constexpr auto control(R r, YOuter y_outer, FFInner ff_inner, InnerArgs... inner_args) {
        using RInnerFF = std::remove_cvref_t<decltype(std::declval<detail::cascade_inner_ref_t<Outer, R, YOuter>>() + std::declval<FFInner>())>;
        const RInnerFF r_inner_unsat = outer_.control(r, y_outer) + ff_inner;
        const RInnerFF r_inner = clamp_inner_reference<RInnerFF>(r_inner_unsat);
        propagate_back_calculation<R, YOuter, RInnerFF>(r_inner_unsat, r_inner);
        return inner_.control(r_inner, inner_args...);
    }

    /**
     * @brief Reset both outer and inner controller states.
     */
    constexpr void reset() {
        outer_.reset();
        inner_.reset();
    }

    [[nodiscard]] constexpr Outer&       outer() { return outer_; }
    [[nodiscard]] constexpr const Outer& outer() const { return outer_; }
    [[nodiscard]] constexpr Inner&       inner() { return inner_; }
    [[nodiscard]] constexpr const Inner& inner() const { return inner_; }

private:
    template<typename RInner>
    constexpr RInner clamp_inner_reference(const RInner& r_inner_unsat) const {
        const auto lo = static_cast<RInner>(r_inner_min);
        const auto hi = static_cast<RInner>(r_inner_max);
        if (r_inner_unsat < lo) {
            return lo;
        }
        if (r_inner_unsat > hi) {
            return hi;
        }
        return r_inner_unsat;
    }

    template<typename R, typename Y, typename RInner>
    constexpr void propagate_back_calculation(const RInner& r_inner_unsat, const RInner& r_inner_sat) {
        if constexpr (SISOControllerWithBackCalculation<Outer, R, Y, RInner>) {
            if (r_inner_unsat != r_inner_sat) {
                outer_.back_calculate(r_inner_unsat, r_inner_sat);
            }
        } else {
            (void)r_inner_unsat;
            (void)r_inner_sat;
        }
    }

    Outer outer_{};
    Inner inner_{};
};

/**
 * @brief Convenience alias for outer-P / inner-PI cascade.
 * @tparam T Scalar type.
 */
template<typename T = float>
using CascadePPI = Cascade<PIDController<T, PIDMode::P>, PIDController<T, PIDMode::PI>, T>;

/**
 * @brief Convenience alias for three nested cascade layers.
 *
 * Equivalent to `Cascade<Outer, Cascade<Middle, Inner>>`.
 */
template<typename Outer, typename Middle, typename Inner>
using Cascade3 = Cascade<Outer, Cascade<Middle, Inner>>;

} // namespace wet
