#include "solver.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>

#include "unsupported/Eigen/MatrixFunctions"  // IWYU pragma: keep

namespace control {

IntegrationResult Solver::evolveDiscrete(const Matrix& A, const Matrix& B, const Matrix& x, const Matrix& u) const {
    return {A * x + B * u, 0.0};
}

IntegrationResult Solver::evolveForwardEuler(const Matrix& A, const Matrix& B, const Matrix& x, const Matrix& u, double h) const {
    return {x + h * (A * x + B * u), 0.0};
}

IntegrationResult Solver::evolveBackwardEuler(const Matrix& A, const Matrix& B, const Matrix& x, const Matrix& u, double h) const {
    const auto I   = Matrix::Identity(A.rows(), A.cols());
    const auto lhs = I - h * A;
    const auto rhs = x + h * B * u;
    return {lhs.colPivHouseholderQr().solve(rhs), 0.0};
}

IntegrationResult Solver::evolveTrapezoidal(const Matrix& A, const Matrix& B, const Matrix& x, const Matrix& u, double h) const {
    const auto I   = Matrix::Identity(A.rows(), A.cols());
    const auto lhs = I - (h / 2.0) * A;
    const auto rhs = (I + (h / 2.0) * A) * x + (h / 2.0) * B * u;
    return {lhs.colPivHouseholderQr().solve(rhs), 0.0};
}

IntegrationResult Solver::evolveRK4(const Matrix& A, const Matrix& B, const Matrix& x, const Matrix& u, double h) const {
    Matrix k1 = A * x + B * u;
    Matrix k2 = A * (x + h / 2.0 * k1) + B * u;
    Matrix k3 = A * (x + h / 2.0 * k2) + B * u;
    Matrix k4 = A * (x + h * k3) + B * u;
    return {x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 0.0};
}

IntegrationResult Solver::evolveRK45(const Matrix& A, const Matrix& B, const Matrix& x, const Matrix& u, double h) const {
    auto f = [&](const Matrix& x) { return A * x + B * u; };

    Matrix k1 = f(x);
    Matrix k2 = f(x + h * (1.0 / 4.0) * k1);
    Matrix k3 = f(x + h * (3.0 / 32.0 * k1 + 9.0 / 32.0 * k2));
    Matrix k4 = f(x + h * (1932.0 / 2197.0 * k1 - 7200.0 / 2197.0 * k2 + 7296.0 / 2197.0 * k3));
    Matrix k5 = f(x + h * (439.0 / 216.0 * k1 - 8.0 * k2 + 3680.0 / 513.0 * k3 - 845.0 / 4104.0 * k4));
    Matrix k6 = f(x + h * (-8.0 / 27.0 * k1 + 2.0 * k2 - 3544.0 / 2565.0 * k3 + 1859.0 / 4104.0 * k4 - 11.0 / 40.0 * k5));

    Matrix x4 = x + h * (25.0 / 216.0 * k1 + 1408.0 / 2565.0 * k3 + 2197.0 / 4104.0 * k4 - 1.0 / 5.0 * k5);
    Matrix x5 = x + h * (16.0 / 135.0 * k1 + 6656.0 / 12825.0 * k3 + 28561.0 / 56430.0 * k4 - 9.0 / 50.0 * k5 + 2.0 / 55.0 * k6);

    double error = (x5 - x4).norm();
    return {x5, error};
}

IntegrationResult Solver::evolveRK45(const std::function<Matrix(double, const Matrix&)>& f, const Matrix& x, double t, double h, size_t* nfev) const {
    Matrix k1 = f(t, x);
    Matrix k2 = f(t + h * (1.0 / 4.0), x + h * (1.0 / 4.0) * k1);
    Matrix k3 = f(t + h * (3.0 / 8.0), x + h * (3.0 / 32.0 * k1 + 9.0 / 32.0 * k2));
    Matrix k4 = f(t + h * (12.0 / 13.0), x + h * (1932.0 / 2197.0 * k1 - 7200.0 / 2197.0 * k2 + 7296.0 / 2197.0 * k3));
    Matrix k5 = f(t + h, x + h * (439.0 / 216.0 * k1 - 8.0 * k2 + 3680.0 / 513.0 * k3 - 845.0 / 4104.0 * k4));
    Matrix k6 = f(t + h * (1.0 / 2.0), x + h * (-8.0 / 27.0 * k1 + 2.0 * k2 - 3544.0 / 2565.0 * k3 + 1859.0 / 4104.0 * k4 - 11.0 / 40.0 * k5));

    Matrix x4 = x + h * (25.0 / 216.0 * k1 + 1408.0 / 2565.0 * k3 + 2197.0 / 4104.0 * k4 - 1.0 / 5.0 * k5);
    Matrix x5 = x + h * (16.0 / 135.0 * k1 + 6656.0 / 12825.0 * k3 + 28561.0 / 56430.0 * k4 - 9.0 / 50.0 * k5 + 2.0 / 55.0 * k6);

    double error = (x5 - x4).norm();
    if (nfev) {
        *nfev += 6;  // six stage evaluations per RK45 step
    }
    return {x5, error};
}

IntegrationResult Solver::evolveExact(const Matrix& A, const Matrix& B, const Matrix& x, const Matrix& u, double final_t) const {
    // Exact solution of LTI system from t=0 to t=final_t
    const auto I        = Matrix::Identity(A.rows(), A.cols());
    const auto E        = (A * final_t).exp();
    const auto Ainv     = A.inverse();
    const auto integral = Ainv * (E - I) * B * u;
    return {E * x + integral, 0.0};
}

IntegrationResult Solver::evolveFixedTimestep(const Matrix& A, const Matrix& B, const Matrix& x, const Matrix& u, double h) const {
    switch (method_) {
        case IntegrationMethod::ForwardEuler:
            return evolveForwardEuler(A, B, x, u, h);
        case IntegrationMethod::BackwardEuler:
            return evolveBackwardEuler(A, B, x, u, h);
        case IntegrationMethod::Trapezoidal:
            return evolveTrapezoidal(A, B, x, u, h);
        case IntegrationMethod::RK4:
            return evolveRK4(A, B, x, u, h);
        case IntegrationMethod::RK45:
            return evolveRK45(A, B, x, u, h);
        case IntegrationMethod::Exact:
            return evolveExact(A, B, x, u, h);
        default:
            return {x, 0.0};  // No change
    }
}

AdaptiveStepResult Solver::evolveAdaptiveTimestep(const Matrix& A, const Matrix& B, const Matrix& x, const Matrix& u, double time, double tol) {
    while (true) {
        auto res = evolveRK45(A, B, x, u, h_);

        if (res.error < tol) {
            // Accept step
            double step_used = h_;
            h_ *= 1.1;
            return {.x = res.x, .step_size = step_used};
        } else {
            // Reduce step size
            h_ *= 0.5;
            if (h_ < 1e-10) {
                // Prevent excessively small step sizes
                h_ = 1e-10;
            }
        }
    }
}

// Minimal SciPy-like RK45 integrator (generic ODE fun(t,x))
SolveResult Solver::solve(const std::function<Matrix(double, const Matrix&)>& fun,
                          const Matrix&                                       x0,
                          const std::pair<double, double>&                    t_span,
                          const std::vector<double>&                          t_eval,
                          IntegrationMethod                                   method,
                          double                                              atol,
                          double                                              rtol,
                          double                                              max_step,
                          double                                              first_step,
                          bool /*dense_output*/) const {
    (void)method;
    SolveResult result;
    double      t0 = t_span.first;
    double      tf = t_span.second;
    double      t  = t0;
    Matrix      x  = x0;

    result.nfev = 0;

    // push initial point
    result.t.push_back(t);
    result.x.push_back(x);

    size_t next_eval = 0;
    if (!t_eval.empty()) {
        // ensure t_eval is sorted and begins >= t0
        while (next_eval < t_eval.size() && t_eval[next_eval] <= t0 + 1e-15) {
            // already at or before initial time
            ++next_eval;
        }
    }

    double h = first_step > 0.0 ? first_step : std::min(0.01, tf - t);
    if (max_step > 0.0) h = std::min(h, max_step);

    while (t < tf - 1e-15) {
        if (max_step > 0.0 && h > max_step) h = max_step;
        if (t + h > tf) h = tf - t;
        if (!t_eval.empty() && next_eval < t_eval.size() && t + h > t_eval[next_eval]) {
            h = t_eval[next_eval] - t;
        }

        // Use the functor-based evolveRK45 primitive (increments nfev)
        auto   stepRes = evolveRK45(fun, x, t, h, &result.nfev);
        Matrix x5      = stepRes.x;
        double error   = stepRes.error;
        double sc      = atol + rtol * std::max(x5.norm(), x.norm());

        if (error <= sc) {
            // accept
            t += h;
            x = x5;
            result.t.push_back(t);
            result.x.push_back(x);

            // advance t_eval pointer if used
            if (!t_eval.empty()) {
                while (next_eval < t_eval.size() && std::abs(result.t.back() - t_eval[next_eval]) < 1e-9) {
                    ++next_eval;
                }
            }

            // adjust step
            double factor = (error == 0.0) ? 2.0 : std::min(4.0, std::max(0.1, 0.84 * std::pow(sc / error, 0.25)));
            h             = h * factor;
            if (max_step > 0.0) h = std::min(h, max_step);
        } else {
            // reject and reduce step
            double factor = std::max(0.1, 0.84 * std::pow(sc / error, 0.25));
            h             = h * factor;
            if (h < 1e-12) {
                result.success = false;
                result.message = "Step size underflow";
                return result;
            }
        }
    }

    return result;
}

// LTI convenience solver
SolveResult Solver::solveLTI(const Matrix&                    A,
                             const Matrix&                    B,
                             const Matrix&                    x0,
                             const Matrix&                    u_const,
                             const std::pair<double, double>& t_span,
                             const std::vector<double>&       t_eval,
                             IntegrationMethod                method,
                             double                           atol,
                             double                           rtol,
                             double                           max_step,
                             double                           first_step) const {
    // If Exact requested, compute directly for each requested time
    SolveResult result;
    double      t0 = t_span.first;
    if (method == IntegrationMethod::Exact) {
        // precompute
        const auto I = Matrix::Identity(A.rows(), A.cols());
        Matrix     Ainv;
        bool       haveAinv = true;
        try {
            Ainv = A.inverse();
        } catch (...) {
            haveAinv = false;
        }

        if (!haveAinv) {
            // Fallback to numeric integration if A is not invertible
            auto fun = [&](double /*t*/, const Matrix& x) { return A * x + B * u_const; };
            return solve(fun, x0, t_span, t_eval, IntegrationMethod::RK45, atol, rtol, max_step, first_step, false);
        }

        auto compute = [&](double t) -> Matrix {
            double     dt       = t - t0;
            const auto E        = (A * dt).exp();
            const auto integral = Ainv * (E - I) * B * u_const;
            return E * x0 + integral;
        };

        if (t_eval.empty()) {
            double tf = t_span.second;
            result.t.push_back(tf);
            result.x.push_back(compute(tf));
        } else {
            for (double tt : t_eval) {
                if (tt < t0) continue;
                result.t.push_back(tt);
                result.x.push_back(compute(tt));
            }
        }
        return result;
    }

    // Otherwise, use generic solver with fun(t,x) = A*x + B*u_const
    auto fun = [&](double /*t*/, const Matrix& x) { return A * x + B * u_const; };
    return solve(fun, x0, t_span, t_eval, method, atol, rtol, max_step, first_step, false);
}

SolveResult Solver::solveLTI(const Matrix&                        A,
                             const Matrix&                        B,
                             const Matrix&                        x0,
                             const std::function<Matrix(double)>& u_func,
                             const std::pair<double, double>&     t_span,
                             const std::vector<double>&           t_eval,
                             const IntegrationMethod              method,
                             double                               atol,
                             double                               rtol,
                             double                               max_step,
                             double                               first_step) const {
    // Generic case: build fun(t,x) that queries u_func
    auto fun = [&](double t, const Matrix& x) { return A * x + B * u_func(t); };
    return solve(fun, x0, t_span, t_eval, method, atol, rtol, max_step, first_step, false);
}

}  // namespace control
