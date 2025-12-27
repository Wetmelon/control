#pragma once

#include <algorithm>
#include <functional>
#include <variant>
#include <vector>

#include "integrator.hpp"
#include "types.hpp"

namespace control {

// Solve continuous Lyapunov equation A*X + X*A^T + Q = 0 for X
// Primary solver: Schur-based Bartels–Stewart (complex Schur variant)
// Fallback: numerical integral approximation for large/stiff problems
Matrix solve_continuous_lyap(const Matrix& A, const Matrix& Q);

// InputFcn can be either a constant column vector or a function of time
struct InputFcn : public std::variant<ColVec, std::function<ColVec(double)>> {
    using std::variant<ColVec, std::function<ColVec(double)>>::variant;
};

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

    // Solve generic ODE: dx/dt = f(t, x) with adaptive stepping
    // Templated on function type for inlining and zero-overhead abstraction
    template <typename F>
    SolveResult solve(F&&                              f,
                      const Matrix&                    x0,
                      const std::pair<double, double>& t_span,
                      const std::vector<double>&       t_eval = {}) const
        requires ODEIntegrator<IntegratorType> && std::invocable<F, double, const Matrix&>
    {
        if constexpr (requires { integrator_.reset(); }) {
            integrator_.reset();
        }

        SolveResult result;
        double      t0 = t_span.first;
        double      tf = t_span.second;
        double      t  = t0;
        Matrix      x  = x0;
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
                t += target_step;
                x                      = stepRes.x;
                consecutive_rejections = 0;

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
    IntegratorType integrator_;
    double         h0_;
    double         tol_;
    double         h_min_;
    double         h_max_;
    size_t         max_nfev_;
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
}  // namespace control
