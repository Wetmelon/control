#pragma once

#include <functional>
#include <optional>
#include <vector>

#include "types.hpp"

namespace control {

class Solver {
   public:
    Solver(IntegrationMethod method = IntegrationMethod::RK45, std::optional<double> timestep = std::nullopt)
        : method_(method), timestep_(timestep) {}

    void setIntegrationMethod(IntegrationMethod method) { method_ = method; }
    void setTimestep(std::optional<double> timestep) { timestep_ = timestep; }

    IntegrationMethod     getIntegrationMethod() const { return method_; }
    std::optional<double> getTimestep() const { return timestep_; }

    IntegrationResult  evolveFixedTimestep(const Matrix& A, const Matrix& B, const Matrix& x, const Matrix& u, double h) const;
    IntegrationResult  evolveDiscrete(const Matrix& A, const Matrix& B, const Matrix& x, const Matrix& u) const;
    AdaptiveStepResult evolveAdaptiveTimestep(const Matrix& A, const Matrix& B, const Matrix& x, const Matrix& u, double hInitial, double tol);

    // Generic ODE solver (SciPy-like minimal): integrate fun(t, x) over t_span.
    SolveResult solve(const std::function<Matrix(double, const Matrix&)>& fun,
                      const Matrix&                                       x0,
                      const std::pair<double, double>&                    t_span,
                      const std::vector<double>&                          t_eval       = {},
                      IntegrationMethod                                   method       = IntegrationMethod::RK45,
                      double                                              atol         = 1e-6,
                      double                                              rtol         = 1e-3,
                      double                                              max_step     = 0.0,
                      double                                              first_step   = 0.0,
                      bool                                                dense_output = false) const;

    SolveResult solveLTI(const Matrix&                    A,
                         const Matrix&                    B,
                         const Matrix&                    x0,
                         const Matrix&                    u_const,
                         const std::pair<double, double>& t_span,
                         const std::vector<double>&       t_eval     = {},
                         IntegrationMethod                method     = IntegrationMethod::Exact,
                         double                           atol       = 1e-6,
                         double                           rtol       = 1e-3,
                         double                           max_step   = 0.0,
                         double                           first_step = 0.0) const;

    SolveResult solveLTI(const Matrix&                        A,
                         const Matrix&                        B,
                         const Matrix&                        x0,
                         const std::function<Matrix(double)>& u_func,
                         const std::pair<double, double>&     t_span,
                         const std::vector<double>&           t_eval     = {},
                         IntegrationMethod                    method     = IntegrationMethod::RK45,
                         double                               atol       = 1e-6,
                         double                               rtol       = 1e-3,
                         double                               max_step   = 0.0,
                         double                               first_step = 0.0) const;

    // Templated wrapper: accept any callable for u_func (function pointer, lambda, functor)
    template <typename UFunc>
    SolveResult solveLTI(const Matrix&                    A,
                         const Matrix&                    B,
                         const Matrix&                    x0,
                         const UFunc&                     u_func,
                         const std::pair<double, double>& t_span,
                         const std::vector<double>&       t_eval     = {},
                         const IntegrationMethod          method     = IntegrationMethod::RK45,
                         double                           atol       = 1e-6,
                         double                           rtol       = 1e-3,
                         double                           max_step   = 0.0,
                         double                           first_step = 0.0) const {
        return solveLTI(A, B, x0, std::function<Matrix(double)>(u_func), t_span, t_eval, method, atol, rtol, max_step, first_step);
    }

   private:
    IntegrationResult evolveForwardEuler(const Matrix& A, const Matrix& B, const Matrix& x, const Matrix& u, double h) const;
    IntegrationResult evolveBackwardEuler(const Matrix& A, const Matrix& B, const Matrix& x, const Matrix& u, double h) const;
    IntegrationResult evolveTrapezoidal(const Matrix& A, const Matrix& B, const Matrix& x, const Matrix& u, double h) const;
    IntegrationResult evolveRK4(const Matrix& A, const Matrix& B, const Matrix& x, const Matrix& u, double h) const;
    IntegrationResult evolveRK45(const Matrix& A, const Matrix& B, const Matrix& x, const Matrix& u, double h) const;
    IntegrationResult evolveRK45(const std::function<Matrix(double, const Matrix&)>& f, const Matrix& x, double t, double h, size_t* nfev = nullptr) const;

    // Templated wrapper: accept any callable for the ODE right-hand side
    template <typename F>
    IntegrationResult evolveRK45(const F&& f, const Matrix& x, double t, double h, size_t* nfev = nullptr) const {
        return evolveRK45(std::function<Matrix(double, const Matrix&)>(f), x, t, h, nfev);
    }
    IntegrationResult evolveExact(const Matrix& A, const Matrix& B, const Matrix& x, const Matrix& u, double final_t) const;

    IntegrationMethod     method_;
    std::optional<double> timestep_;

    double h_ = 0.01;
};

}  // namespace control
