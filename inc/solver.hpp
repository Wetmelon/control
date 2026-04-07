#pragma once

/**
 * @defgroup solver ODE Solver Wrappers
 * @brief Fixed-step and adaptive-step ODE solvers wrapping integrator methods
 *
 * Provides high-level solver classes that manage time-stepping, result storage,
 * event detection, and zero-crossing location on top of the low-level integrator
 * methods defined in integrator.hpp.
 *
 * Usage:
 * @code
 *   RK4<2> rk4;
 *   FixedStepSolver solver(rk4, 0.01);
 *   auto result = solver.solve(f, x0, {0.0, 10.0});
 *   for (const auto& [t, x] : result) { ... }
 * @endcode
 */

#include <algorithm>
#include <functional>
#include <tuple>
#include <utility>
#include <vector>

#include "integrator.hpp"

namespace wetmelon::control {

/**
 * @brief Result of an ODE solve operation
 *
 * Stores the full time and state history from a simulation.
 * Supports range-based for loops over (time, state) pairs.
 *
 * @tparam NX Number of states
 * @tparam T  Scalar type
 */
template<size_t NX, typename T = double>
struct SolveResult {
    std::vector<T>             t; ///< Time points
    std::vector<ColVec<NX, T>> x; ///< State vectors at each time point

    bool   success = true;
    size_t nfev = 0; ///< Number of function evaluations

    /// Iterator for range-based for: for (const auto& [t, x] : result)
    struct Iterator {
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = std::pair<T, const ColVec<NX, T>&>;
        using reference = value_type;

        Iterator(const std::vector<T>* t_vec, const std::vector<ColVec<NX, T>>* x_vec, size_t idx)
            : t_vec_(t_vec), x_vec_(x_vec), idx_(idx) {}

        reference operator*() const { return {(*t_vec_)[idx_], (*x_vec_)[idx_]}; }

        Iterator& operator++() {
            ++idx_;
            return *this;
        }

        Iterator operator++(int) {
            Iterator tmp = *this;
            ++idx_;
            return tmp;
        }

        bool operator==(const Iterator& other) const { return idx_ == other.idx_; }
        bool operator!=(const Iterator& other) const { return idx_ != other.idx_; }

    private:
        const std::vector<T>*             t_vec_;
        const std::vector<ColVec<NX, T>>* x_vec_;
        size_t                            idx_;
    };

    Iterator begin() const { return Iterator(&t, &x, 0); }
    Iterator end() const { return Iterator(&t, &x, t.size()); }

    /// Number of recorded time steps
    size_t size() const { return t.size(); }
};

/**
 * @brief Fixed-step ODE solver
 *
 * Wraps any integrator with a fixed time step. Supports optional callbacks
 * for step notification, early stopping, and event detection.
 *
 * @tparam NX             Number of states
 * @tparam T              Scalar type
 * @tparam IntegratorType Integrator struct (e.g., RK4<NX,T>)
 */
template<size_t NX, typename T, typename IntegratorType>
class FixedStepSolver {
public:
    explicit FixedStepSolver(IntegratorType integrator, T step_size = T(0.01))
        : integrator_(integrator), h_(step_size) {}

    /**
     * @brief Solve dx/dt = f(t, x) over [t0, tf]
     *
     * @param f      Right-hand side callable: f(t, x) -> ColVec<NX,T>
     * @param x0     Initial state
     * @param t_span Pair {t0, tf}
     * @return SolveResult with full time/state history
     */
    template<typename F>
    SolveResult<NX, T> solve(F&& f, const ColVec<NX, T>& x0, const std::pair<T, T>& t_span) const {
        // Reset multi-step integrators if they have a reset() method
        if constexpr (requires { integrator_.reset(); }) {
            integrator_.reset();
        }

        SolveResult<NX, T> result;
        T                  t0 = t_span.first;
        T                  tf = t_span.second;
        T                  t = t0;
        ColVec<NX, T>      x = x0;

        result.success = true;
        size_t n_steps = static_cast<size_t>((tf - t0) / h_) + 2;
        result.t.reserve(n_steps);
        result.x.reserve(n_steps);

        result.t.push_back(t);
        result.x.push_back(x);

        while (t < tf - T(1e-15)) {
            T step = h_;
            if (t + step > tf) {
                step = tf - t;
            }

            IntegrationResult<NX, T> step_result = integrator_.evolve(f, x, t, step);
            ++result.nfev;

            t += step;
            x = step_result.x;
            result.t.push_back(t);
            result.x.push_back(x);

            // Step callback
            if (on_step_) {
                on_step_(t, x);
            }

            // Event detection
            if (event_detector_) {
                auto [triggered, new_state, stop] = event_detector_(t, x);
                if (triggered) {
                    x = new_state;
                    result.x.back() = x;
                }
                if (stop) {
                    break;
                }
            }

            // Stop condition
            if (stop_condition_ && stop_condition_(t, x)) {
                break;
            }
        }

        return result;
    }

    /// Set callback invoked after each accepted step
    template<typename Callback>
    void set_on_step(Callback&& cb) {
        on_step_ = std::forward<Callback>(cb);
    }

    /// Set condition to stop simulation early: returns true to stop
    template<typename Condition>
    void set_stop_condition(Condition&& cond) {
        stop_condition_ = std::forward<Condition>(cond);
    }

    /**
     * @brief Set event detector
     *
     * Callable returning a struct/tuple of (bool triggered, ColVec<NX,T> new_state, bool stop).
     * If triggered, state is replaced. If stop, simulation ends.
     */
    template<typename Detector>
    void set_event_detector(Detector&& det) {
        event_detector_ = std::forward<Detector>(det);
    }

    void set_step_size(T h) { h_ = h; }
    T    get_step_size() const { return h_; }

private:
    mutable IntegratorType integrator_;
    T                      h_;

    std::function<void(T, const ColVec<NX, T>&)>                                  on_step_;
    std::function<bool(T, const ColVec<NX, T>&)>                                  stop_condition_;
    std::function<std::tuple<bool, ColVec<NX, T>, bool>(T, const ColVec<NX, T>&)> event_detector_;
};
/// CTAD deduction guides for FixedStepSolver
template<template<size_t, typename> class Int, size_t NX, typename T>
FixedStepSolver(Int<NX, T>, T) -> FixedStepSolver<NX, T, Int<NX, T>>;

template<template<size_t, typename> class Int, size_t NX, typename T>
FixedStepSolver(Int<NX, T>) -> FixedStepSolver<NX, T, Int<NX, T>>;
/**
 * @brief Adaptive-step ODE solver
 *
 * Wraps an integrator that provides error estimates (e.g., RK45, RK23)
 * and adjusts step size to maintain a specified tolerance.
 * Supports zero-crossing detection via bisection.
 *
 * @tparam NX             Number of states
 * @tparam T              Scalar type
 * @tparam IntegratorType Integrator struct with error output (e.g., RK45<NX,T>)
 */
template<size_t NX, typename T, typename IntegratorType>
class AdaptiveStepSolver {
public:
    explicit AdaptiveStepSolver(IntegratorType integrator, T initial_step = T(0.01), T tol = T(1e-6), T min_step = T(1e-8), T max_step = T(1.0), size_t max_nfev = 1000000)
        : integrator_(integrator), h0_(initial_step), tol_(tol), h_min_(min_step), h_max_(max_step), max_nfev_(max_nfev) {}

    /**
     * @brief Solve dx/dt = f(t, x) over [t0, tf] with adaptive stepping
     *
     * @param f      Right-hand side callable: f(t, x) -> ColVec<NX,T>
     * @param x0     Initial state
     * @param t_span Pair {t0, tf}
     * @return SolveResult with full time/state history
     */
    template<typename F>
    SolveResult<NX, T> solve(F&& f, const ColVec<NX, T>& x0, const std::pair<T, T>& t_span) const {
        if constexpr (requires { integrator_.reset(); }) {
            integrator_.reset();
        }

        SolveResult<NX, T> result;
        T                  t0 = t_span.first;
        T                  tf = t_span.second;
        T                  t = t0;
        ColVec<NX, T>      x = x0;
        T                  h = h0_;

        result.success = true;
        result.t.reserve(1000);
        result.x.reserve(1000);
        result.t.push_back(t);
        result.x.push_back(x);

        constexpr T      safety_factor = T(0.9);
        constexpr T      max_scale = T(5.0);
        constexpr T      min_scale = T(0.2);
        constexpr size_t max_consecutive_rejections = 100;
        size_t           consecutive_rejections = 0;

        while (t < tf - T(1e-15)) {
            if (result.nfev >= max_nfev_) {
                result.success = false;
                return result;
            }

            T target_step = h;
            if (t + target_step > tf) {
                target_step = tf - t;
            }

            IntegrationResult<NX, T> step_result = integrator_.evolve(f, x, t, target_step);
            ++result.nfev;

            T error_norm = step_result.error;
            T scale = T(1.0);

            if (error_norm > T(0)) {
                scale = safety_factor * wet::pow(tol_ / error_norm, T(0.2));
                scale = std::clamp(scale, min_scale, max_scale);
            }

            // Accept step if error is within tolerance or at minimum step
            if (error_norm <= tol_ || target_step <= h_min_) {
                // Check zero crossings before accepting
                bool step_accepted = false;
                if (!zero_crossings_.empty()) {
                    auto crossing = detect_zero_crossing(f, x, t, step_result.x, t + target_step);
                    if (crossing.found) {
                        // Integrate up to the crossing time
                        T                        dt_cross = crossing.time - t;
                        IntegrationResult<NX, T> cross_result = integrator_.evolve(f, x, t, dt_cross);
                        ++result.nfev;

                        t = crossing.time;
                        x = cross_result.x;
                        step_accepted = true;

                        result.t.push_back(t);
                        result.x.push_back(x);

                        if (on_step_) {
                            on_step_(t, x);
                        }

                        if (event_detector_) {
                            auto [triggered, new_state, stop] = event_detector_(t, x);
                            if (triggered) {
                                x = new_state;
                                result.x.back() = x;
                            }
                            if (stop) {
                                return result;
                            }
                        }

                        if (stop_condition_ && stop_condition_(t, x)) {
                            return result;
                        }
                    }
                }

                if (!step_accepted) {
                    t += target_step;
                    x = step_result.x;
                    result.t.push_back(t);
                    result.x.push_back(x);

                    if (on_step_) {
                        on_step_(t, x);
                    }
                }

                consecutive_rejections = 0;
                h = std::clamp(target_step * scale, h_min_, h_max_);

                if (event_detector_ && !step_accepted) {
                    auto [triggered, new_state, stop] = event_detector_(t, x);
                    if (triggered) {
                        x = new_state;
                        result.x.back() = x;
                    }
                    if (stop) {
                        return result;
                    }
                }

                if (stop_condition_ && !step_accepted && stop_condition_(t, x)) {
                    return result;
                }
            } else {
                // Reject step, retry with smaller step
                h = std::clamp(target_step * scale, h_min_, h_max_);
                ++consecutive_rejections;

                if (consecutive_rejections >= max_consecutive_rejections) {
                    result.success = false;
                    return result;
                }
            }
        }

        return result;
    }

    /// Add a zero-crossing function to monitor
    template<typename ZCF>
    void add_zero_crossing(ZCF&& zcf) {
        zero_crossings_.push_back(std::forward<ZCF>(zcf));
    }

    template<typename Callback>
    void set_on_step(Callback&& cb) {
        on_step_ = std::forward<Callback>(cb);
    }

    template<typename Condition>
    void set_stop_condition(Condition&& cond) {
        stop_condition_ = std::forward<Condition>(cond);
    }

    template<typename Detector>
    void set_event_detector(Detector&& det) {
        event_detector_ = std::forward<Detector>(det);
    }

    void set_tolerance(T tol) { tol_ = tol; }
    T    get_tolerance() const { return tol_; }

private:
    struct CrossingResult {
        bool found = false;
        T    time = T(0);
    };

    template<typename F>
    CrossingResult detect_zero_crossing(F&& /*f*/, const ColVec<NX, T>& x_start, T t_start, const ColVec<NX, T>& x_end, T t_end) const {
        for (const auto& zcf : zero_crossings_) {
            T z_start = zcf(t_start, x_start);
            T z_end = zcf(t_end, x_end);

            if ((z_start > T(0) && z_end < T(0)) || (z_start < T(0) && z_end > T(0))) {
                T crossing_time = locate_zero_crossing(zcf, t_start, x_start, t_end, x_end);
                return {true, crossing_time};
            }
        }
        return {};
    }

    T locate_zero_crossing(const std::function<T(T, const ColVec<NX, T>&)>& zc, T t_a, const ColVec<NX, T>& x_a, T t_b, const ColVec<NX, T>& x_b) const {
        T t_left = t_a, t_right = t_b;
        T z_left = zc(t_a, x_a);

        for (int iter = 0; iter < 50; ++iter) {
            T    t_mid = (t_left + t_right) / T(2);
            T    alpha = (t_mid - t_a) / (t_b - t_a);
            auto x_mid = (T(1) - alpha) * x_a + alpha * x_b;
            T    z_mid = zc(t_mid, x_mid);

            if ((z_left > T(0) && z_mid > T(0)) || (z_left < T(0) && z_mid < T(0))) {
                t_left = t_mid;
                z_left = z_mid;
            } else {
                t_right = t_mid;
            }

            if (wet::abs(z_mid) < T(1e-12)) {
                break;
            }
        }

        return (t_left + t_right) / T(2);
    }

    mutable IntegratorType integrator_;
    T                      h0_;
    T                      tol_;
    T                      h_min_;
    T                      h_max_;
    size_t                 max_nfev_;

    std::function<void(T, const ColVec<NX, T>&)>                                  on_step_;
    std::function<bool(T, const ColVec<NX, T>&)>                                  stop_condition_;
    std::function<std::tuple<bool, ColVec<NX, T>, bool>(T, const ColVec<NX, T>&)> event_detector_;
    std::vector<std::function<T(T, const ColVec<NX, T>&)>>                        zero_crossings_;
};

/// CTAD deduction guides for AdaptiveStepSolver
template<template<size_t, typename> class Int, size_t NX, typename T>
AdaptiveStepSolver(Int<NX, T>, T, T, T, T, size_t) -> AdaptiveStepSolver<NX, T, Int<NX, T>>;

template<template<size_t, typename> class Int, size_t NX, typename T>
AdaptiveStepSolver(Int<NX, T>, T, T) -> AdaptiveStepSolver<NX, T, Int<NX, T>>;

template<template<size_t, typename> class Int, size_t NX, typename T>
AdaptiveStepSolver(Int<NX, T>, T) -> AdaptiveStepSolver<NX, T, Int<NX, T>>;

template<template<size_t, typename> class Int, size_t NX, typename T>
AdaptiveStepSolver(Int<NX, T>) -> AdaptiveStepSolver<NX, T, Int<NX, T>>;

} // namespace wetmelon::control
