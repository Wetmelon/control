#pragma once

#include <algorithm>
#include <functional>
#include <variant>
#include <vector>

#include "integrator.hpp"
#include "types.hpp"

namespace control {

// Event handling for discontinuities and zero crossings
struct EventResult {
    bool        triggered = false;
    ColVec      new_state;               // Modified state after event (if triggered)
    bool        reset_dynamics = false;  // Flag indicating dynamics should change
    bool        stop           = false;  // Flag to stop the simulation
    std::string message;                 // Optional event description
};

using EventDetector = std::function<EventResult(double t, const ColVec& x)>;
using StepCallback  = std::function<void(double t, const ColVec& x)>;
using StopCondition = std::function<bool(double t, const ColVec& x)>;

// Zero-crossing function: returns value that crosses zero at event
using ZeroCrossingFcn = std::function<double(double t, const ColVec& x)>;

struct SolveResult {
    std::vector<double> t;  // time points
    std::vector<ColVec> x;  // states (x) at each time point

    bool        success = true;
    std::string message;
    size_t      nfev = 0;  // number of function evaluations

    // Iterator support for range-based for loops: for(const auto& [t, x] : result)
    struct Iterator {
        using iterator_category = std::forward_iterator_tag;
        using difference_type   = std::ptrdiff_t;
        using value_type        = std::pair<double, const ColVec&>;
        using pointer           = const value_type*;
        using reference         = value_type;

        Iterator(const std::vector<double>* t_vec, const std::vector<ColVec>* x_vec, size_t idx)
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
        const std::vector<double>* t_vec_;
        const std::vector<ColVec>* x_vec_;
        size_t                     idx_;
    };

    Iterator begin() const { return Iterator(&t, &x, 0); }
    Iterator end() const { return Iterator(&t, &x, t.size()); }
};

template <typename IntegratorType>
    requires FixedStepIntegrator<IntegratorType>
class FixedStepSolver {
   public:
    explicit FixedStepSolver(double step_size = 0.01) : h_(step_size) {}

    // Set callback for each integration step
    void set_on_step_callback(StepCallback cb) {
        on_step_callback_ = std::move(cb);
    }

    // Set condition to stop simulation early
    void set_stop_condition(StopCondition cond) {
        stop_condition_ = std::move(cond);
    }

    // Set event detector for discontinuities and zero crossings
    void set_event_detector(EventDetector detector) {
        event_detector_ = std::move(detector);
    }

    // Solve generic ODE: dx/dt = f(t, x)
    // Templated on function type for inlining and zero-overhead abstraction
    template <typename F>
    SolveResult solve(F&&                              f,
                      const ColVec&                    x0,
                      const std::pair<double, double>& t_span,
                      const std::vector<double>&       t_eval = {}) const
        requires ODEIntegrator<IntegratorType> && std::invocable<F, double, const ColVec&>
    {
        if constexpr (requires { integrator_.reset(); }) {
            integrator_.reset();
        }

        SolveResult result;
        double      t0 = t_span.first;
        double      tf = t_span.second;
        double      t  = t0;
        ColVec      x  = x0;

        result.success = true;
        result.t.reserve(static_cast<size_t>((tf - t0) / h_) + 2);
        result.x.reserve(static_cast<size_t>((tf - t0) / h_) + 2);

        result.t.push_back(t);
        result.x.push_back(x);

        size_t next_eval = 0;
        if (!t_eval.empty()) {
            while (next_eval < t_eval.size() && t_eval[next_eval] <= t0 + 1e-15) {
                ++next_eval;
            }
        }

        // Fixed-step integration
        while (t < tf - 1e-15) {
            double step = h_;
            if (t + step > tf) step = tf - t;
            if (!t_eval.empty() && next_eval < t_eval.size() && t + step > t_eval[next_eval]) {
                step = t_eval[next_eval] - t;
            }

            IntegrationResult stepRes = integrator_.evolve(f, x, t, step);
            ++result.nfev;

            t += step;
            x = stepRes.x;
            result.t.push_back(t);
            result.x.push_back(x);

            // Call step callback if set
            if (on_step_callback_) {
                on_step_callback_(t, x);
            }

            // Check for events (discontinuities, zero crossings)
            if (event_detector_) {
                EventResult event = event_detector_(t, x);
                if (event.triggered) {
                    x               = event.new_state;
                    result.x.back() = x;  // Update the stored state
                    if (!event.message.empty()) {
                        result.message = event.message;
                    }
                    // Note: reset_dynamics flag could be used to change ODE function
                    // but that would require more complex solver redesign
                }
            }

            // Check stop condition
            if (stop_condition_ && stop_condition_(t, x)) {
                result.message = "Stopped by user-defined condition";
                break;
            }

            if (!t_eval.empty()) {
                const double current_t = t;
                while (next_eval < t_eval.size() && std::abs(current_t - t_eval[next_eval]) < 1e-9) {
                    ++next_eval;
                }
            }
        }

        return result;
    }

    void   set_step_size(double h) { h_ = h; }
    double get_step_size() const { return h_; }

   private:
    IntegratorType integrator_;
    double         h_;

    StepCallback  on_step_callback_;
    StopCondition stop_condition_;
    EventDetector event_detector_;
};

template <typename IntegratorType>
    requires AdaptiveStepIntegrator<IntegratorType>
class AdaptiveStepSolver {
   public:
    explicit AdaptiveStepSolver(double initial_step = 0.01,
                                double tol          = 1e-6,
                                double min_step     = 1e-8,
                                double max_step     = 1.0,
                                size_t max_nfev     = 1000000)
        : h0_(initial_step), tol_(tol), h_min_(min_step), h_max_(max_step), max_nfev_(max_nfev) {}

    // Set callback for each integration step
    void set_on_step_callback(StepCallback cb) {
        on_step_callback_ = std::move(cb);
    }

    // Set condition to stop simulation early
    void set_stop_condition(StopCondition cond) {
        stop_condition_ = std::move(cond);
    }

    // Set event detector for discontinuities and zero crossings
    void set_event_detector(EventDetector detector) {
        event_detector_ = std::move(detector);
    }

    // Set zero-crossing functions to monitor during integration
    void set_zero_crossings(std::vector<ZeroCrossingFcn> crossings) {
        zero_crossings_ = std::move(crossings);
    }

    // Solve generic ODE: dx/dt = f(t, x) with adaptive stepping
    // Templated on function type for inlining and zero-overhead abstraction
    template <typename F>
    SolveResult solve(F&&                              f,
                      const ColVec&                    x0,
                      const std::pair<double, double>& t_span,
                      const std::vector<double>&       t_eval = {}) const
        requires ODEIntegrator<IntegratorType> && std::invocable<F, double, const ColVec&>
    {
        if constexpr (requires { integrator_.reset(); }) {
            integrator_.reset();
        }

        SolveResult result;
        double      t0 = t_span.first;
        double      tf = t_span.second;
        double      t  = t0;
        ColVec      x  = x0;
        double      h  = h0_;

        result.success = true;

        // Determine if we're using t_eval for output
        const bool use_t_eval = !t_eval.empty();

        if (use_t_eval) {
            result.t.reserve(t_eval.size());
            result.x.reserve(t_eval.size());
        } else {
            result.t.reserve(1000);
            result.x.reserve(1000);
            // Store initial state only if not using t_eval
            result.t.push_back(t);
            result.x.push_back(x);
        }

        size_t next_eval = 0;
        if (use_t_eval) {
            // Check if initial time matches first eval point and store it
            while (next_eval < t_eval.size() && t_eval[next_eval] <= t0 + 1e-15) {
                if (std::abs(t_eval[next_eval] - t0) < 1e-9) {
                    result.t.push_back(t0);
                    result.x.push_back(x0);
                }
                ++next_eval;
            }
        }

        constexpr double safety_factor              = 0.9;
        constexpr double max_scale                  = 5.0;
        constexpr double min_scale                  = 0.2;
        constexpr size_t max_consecutive_rejections = 100;
        size_t           consecutive_rejections     = 0;

        while (t < tf - 1e-15) {
            // Safety check: prevent infinite loops
            if (result.nfev >= max_nfev_) {
                result.success = false;
                result.message = "Maximum number of function evaluations exceeded. System may be too stiff.";
                return result;
            }

            // Determine target step size, considering final time and evaluation points
            double target_step = h;

            // Limit step to not overshoot final time
            if (t + target_step > tf) {
                target_step = tf - t;
            }

            // Check if we need to hit a specific evaluation point
            if (use_t_eval && next_eval < t_eval.size()) {
                double dist_to_eval = t_eval[next_eval] - t;
                // If eval point is within our step, adjust to hit it exactly
                if (dist_to_eval < target_step) {
                    target_step = dist_to_eval;
                }
            }

            IntegrationResult stepRes = integrator_.evolve(f, x, t, target_step);
            ++result.nfev;

            // Estimate error and adjust step size
            double error_norm = stepRes.error;
            double scale      = 1.0;

            if (error_norm > 0) {
                scale = safety_factor * std::pow(tol_ / error_norm, 0.2);
                scale = std::clamp(scale, min_scale, max_scale);
            }

            // Accept step if error is within tolerance OR if we're at minimum step size
            if (error_norm <= tol_ || target_step <= h_min_) {
                // Check for zero crossings in this step before accepting it
                bool step_accepted = false;
                if (!zero_crossings_.empty()) {
                    auto crossing_result = detect_zero_crossing(f, x, t, stepRes.x, t + target_step);
                    if (crossing_result.crossing_found) {
                        // Found a crossing - integrate up to the exact time
                        double            crossing_time = crossing_result.crossing_time;
                        IntegrationResult crossingRes   = integrator_.evolve(f, x, t, crossing_time - t);
                        ++result.nfev;

                        t             = crossing_time;
                        x             = crossingRes.x;
                        step_accepted = true;

                        // Store the crossing point
                        if (use_t_eval && next_eval < t_eval.size()) {
                            if (std::abs(t - t_eval[next_eval]) < 1e-9) {
                                result.t.push_back(t);
                                result.x.push_back(x);
                                ++next_eval;
                            }
                        } else if (!use_t_eval) {
                            result.t.push_back(t);
                            result.x.push_back(x);
                        }

                        // Call step callback
                        if (on_step_callback_) {
                            on_step_callback_(t, x);
                        }

                        // Trigger event if detector is set
                        if (event_detector_) {
                            EventResult event = event_detector_(t, x);
                            if (event.triggered) {
                                x = event.new_state;
                                // Update the most recently stored state
                                if (!result.x.empty()) {
                                    result.x.back() = x;
                                }
                                if (!event.message.empty()) {
                                    result.message = event.message;
                                }
                                // Check if event requests to stop simulation
                                if (event.stop) {
                                    return result;
                                }
                            }
                        }

                        // Check stop condition
                        if (stop_condition_ && stop_condition_(t, x)) {
                            result.message = "Stopped by user-defined condition";
                            return result;
                        }
                    }
                }

                if (!step_accepted) {
                    // No crossing - accept the full step
                    t += target_step;
                    x = stepRes.x;

                    // Store result only if not using t_eval OR we just hit an eval point
                    if (use_t_eval && next_eval < t_eval.size()) {
                        if (std::abs(t - t_eval[next_eval]) < 1e-9) {
                            result.t.push_back(t);
                            result.x.push_back(x);
                            ++next_eval;
                        }
                    } else if (!use_t_eval) {
                        // Store all steps when not using t_eval
                        result.t.push_back(t);
                        result.x.push_back(x);
                    }

                    // Call step callback if set
                    if (on_step_callback_) {
                        on_step_callback_(t, x);
                    }
                }

                consecutive_rejections = 0;

                // Update step size for next iteration based on error estimate
                h = std::clamp(target_step * scale, h_min_, h_max_);
            } else {
                // Reject step and retry with smaller step size
                h = std::clamp(target_step * scale, h_min_, h_max_);
                ++consecutive_rejections;

                // Safety check: if we're rejecting too many steps in a row, something is wrong
                if (consecutive_rejections >= max_consecutive_rejections) {
                    result.success = false;
                    result.message = "Too many consecutive step rejections. System may be too stiff or ill-conditioned.";
                    return result;
                }
            }
        }

        return result;
    }

    void   set_tolerance(double tol) { tol_ = tol; }
    double get_tolerance() const { return tol_; }

   private:
    struct CrossingResult {
        bool   crossing_found = false;
        double crossing_time  = 0.0;
        size_t crossing_index = 0;
    };

    template <typename F>
    CrossingResult detect_zero_crossing(F&& f, const ColVec& x_start, double t_start,
                                        const ColVec& x_end, double t_end) const {
        // Check each zero-crossing function for sign changes
        for (size_t i = 0; i < zero_crossings_.size(); ++i) {
            double z_start = zero_crossings_[i](t_start, x_start);
            double z_end   = zero_crossings_[i](t_end, x_end);

            // Sign change indicates crossing
            if ((z_start > 0 && z_end < 0) || (z_start < 0 && z_end > 0)) {
                // Use bisection to find exact crossing time
                double crossing_time = locate_zero_crossing(f, zero_crossings_[i],
                                                            t_start, x_start, t_end, x_end);
                return {true, crossing_time, i};
            }
        }
        return {false, 0.0, 0};
    }

    template <typename F>
    double locate_zero_crossing(F&& /*f*/, const ZeroCrossingFcn& zc,
                                double t_a, const ColVec& x_a, double t_b, const ColVec& x_b) const {
        // Bisection method to find zero crossing
        double                  t_left = t_a, t_right = t_b;
        ColVec                  x_left = x_a, x_right = x_b;
        double                  z_left  = zc(t_a, x_a);
        [[maybe_unused]] double z_right = zc(t_b, x_b);

        for (int iter = 0; iter < 50; ++iter) {  // Max 50 bisection iterations
            double t_mid = (t_left + t_right) / 2.0;

            // Interpolate state at t_mid (linear for simplicity)
            double alpha = (t_mid - t_a) / (t_b - t_a);
            ColVec x_mid = (1 - alpha) * x_a + alpha * x_b;

            double z_mid = zc(t_mid, x_mid);

            if ((z_left > 0 && z_mid > 0) || (z_left < 0 && z_mid < 0)) {
                t_left = t_mid;
                z_left = z_mid;
                x_left = x_mid;
            } else {
                t_right = t_mid;
                z_right = z_mid;
                x_right = x_mid;
            }

            if (std::abs(z_mid) < 1e-12) break;  // Close enough to zero
        }

        return (t_left + t_right) / 2.0;
    }

    IntegratorType integrator_;
    double         h0_;
    double         tol_;
    double         h_min_;
    double         h_max_;
    size_t         max_nfev_;

    StepCallback                 on_step_callback_;
    StopCondition                stop_condition_;
    EventDetector                event_detector_;
    std::vector<ZeroCrossingFcn> zero_crossings_;
};

// ExactSolver: Uses matrix exponential for analytical solution of LTI systems
// Only supports constant input (not time-varying)
class ExactSolver {
   public:
    // Solve LTI system: dx/dt = Ax + Bu with constant input

    SolveResult solve(const Matrix&                    A,
                      const Matrix&                    B,
                      const ColVec&                    x0,
                      const ColVec&                    u,
                      const std::pair<double, double>& t_span,
                      const std::vector<double>&       t_eval) const {
        SolveResult result;
        double      t0 = t_span.first;
        double      tf = t_span.second;

        // Precompute constant terms
        const auto I = Matrix::Identity(A.rows(), A.cols());
        Matrix     Ainv;
        bool       haveAinv = true;
        try {
            Ainv = A.inverse();
        } catch (...) {
            haveAinv = false;
        }

        if (!haveAinv) {
            result.success = false;
            result.message = "Matrix A is singular, cannot compute exact solution";
            return result;
        }

        // Exact solution: x(t) = e^(A(t-t0))x0 + A^(-1)(e^(A(t-t0)) - I)Bu
        auto compute = [&](double t) -> Matrix {
            double     dt       = t - t0;
            const auto E        = (A * dt).exp();
            const auto integral = Ainv * (E - I) * B * u;
            return E * x0 + integral;
        };

        result.success = true;
        result.nfev    = 0;  // No function evaluations for exact solution

        if (t_eval.empty()) {
            // Just return final time
            result.t.push_back(tf);
            result.x.push_back(compute(tf));
        } else {
            // Evaluate at requested time points
            for (double tt : t_eval) {
                if (tt < t0) continue;
                if (tt > tf) continue;
                result.t.push_back(tt);
                result.x.push_back(compute(tt));
            }
        }

        return result;
    }
};

// AdaptiveExactSolver: adaptive sampling wrapper around ExactSolver for shape preservation
// If `t_eval` is non-empty, delegates directly to ExactSolver to evaluate at requested points.
struct AdaptiveExactSolver {
    AdaptiveExactSolver(double tol = 1e-4, size_t max_depth = 20) : tol_(tol), max_depth_(max_depth) {}

    SolveResult solve(const Matrix&                    A,
                      const Matrix&                    B,
                      const ColVec&                    x0,
                      const ColVec&                    u,
                      const std::pair<double, double>& t_span,
                      const std::vector<double>&       t_eval = {}) const {
        ExactSolver exact;

        // If the user requested specific sample times, just delegate to ExactSolver
        if (!t_eval.empty()) {
            return exact.solve(A, B, x0, u, t_span, t_eval);
        }

        const double t0 = t_span.first;
        const double tf = t_span.second;

        // helper to compute state at a single time using ExactSolver
        auto compute_state = [&](double t) -> ColVec {
            auto r = exact.solve(A, B, x0, u, t_span, std::vector<double>{t});
            if (r.x.empty()) return ColVec::Zero(x0.size());
            return r.x.front();
        };

        SolveResult result;
        // compute endpoints
        ColVec xl = compute_state(t0);
        ColVec xr = compute_state(tf);

        std::vector<double> times;
        std::vector<ColVec> vals;
        times.reserve(256);
        vals.reserve(256);

        // push left endpoint
        times.push_back(t0);
        vals.push_back(xl);

        // recursive midpoint refinement
        std::function<void(double, const ColVec&, double, const ColVec&, int)> refine;
        refine = [&](double tl, const ColVec& xl_in, double tr, const ColVec& xr_in, int depth) {
            double tm = 0.5 * (tl + tr);
            // avoid degenerate intervals
            if (tm <= tl || tm >= tr) return;

            ColVec xm = compute_state(tm);

            // linear interpolation of endpoints at tm
            double alpha  = (tm - tl) / (tr - tl);
            ColVec interp = (1.0 - alpha) * xl_in + alpha * xr_in;

            // error estimate: max absolute component error
            double err = 0.0;
            for (int i = 0; i < xm.size(); ++i) err = std::max(err, std::abs(xm(i) - interp(i)));

            if (err > tol_ && depth < static_cast<int>(max_depth_)) {
                refine(tl, xl_in, tm, xm, depth + 1);
                times.push_back(tm);
                vals.push_back(xm);
                refine(tm, xm, tr, xr_in, depth + 1);
            } else {
                // accept midpoint (no further refinement)
                times.push_back(tm);
                vals.push_back(xm);
            }
        };

        // refine whole interval
        refine(t0, xl, tf, xr, 0);

        // push right endpoint
        times.push_back(tf);
        vals.push_back(xr);

        // assemble result (times are in increasing order by construction)
        result.success = true;
        result.nfev    = 0;  // ExactSolver internally counts nothing here
        result.t.reserve(times.size());
        result.x.reserve(vals.size());
        for (size_t i = 0; i < times.size(); ++i) {
            result.t.push_back(times[i]);
            result.x.push_back(vals[i]);
        }

        return result;
    }

   private:
    double tol_;
    size_t max_depth_;
};
}  // namespace control
