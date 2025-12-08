#pragma once

#include <cmath>
#include <format>
#include <functional>
#include <optional>
#include <vector>

#include "Eigen/Dense"

namespace control {
using Matrix = Eigen::MatrixXd;

enum class DiscretizationMethod {
    ZOH,
    FOH,
    Bilinear,
    Tustin,
};

enum class IntegrationMethod {
    ForwardEuler,
    BackwardEuler,
    Trapezoidal,
    RK4,
    RK45,
    Exact,
};

enum class SystemType {
    Continuous,
    Discrete,
};

struct TransferFunction {
    Eigen::MatrixXd num, den;
};

struct FrequencyResponse {
    std::vector<double> freq;       // Frequency in Hz
    std::vector<double> magnitude;  // Magnitude in dB
    std::vector<double> phase;      // Phase in degrees
};

struct StepResponse {
    std::vector<double> time;
    std::vector<double> output;
};

struct IntegrationResult {
    Matrix x;
    double error;
};

struct AdaptiveStepResult {
    Matrix x;
    double step_size;
};

struct SolveResult {
    std::vector<double> t;  // time points
    std::vector<Matrix> x;  // states (x) at each time point
    bool                success = true;
    std::string         message;
    size_t              nfev = 0;  // number of function evaluations
};

class DiscreteStateSpace;    // Forward declaration
class ContinuousStateSpace;  // Forward declaration
using StateSpace = ContinuousStateSpace;

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

    // Templated wrapper: accept any callable (function pointer, lambda, functor)
    template <typename Fun>
    SolveResult solve(const Fun&                       fun,
                      const Matrix&                    x0,
                      const std::pair<double, double>& t_span,
                      const std::vector<double>&       t_eval       = {},
                      IntegrationMethod                method       = IntegrationMethod::RK45,
                      double                           atol         = 1e-6,
                      double                           rtol         = 1e-3,
                      double                           max_step     = 0.0,
                      double                           first_step   = 0.0,
                      bool                             dense_output = false) const {
        return solve(std::function<Matrix(double, const Matrix&)>(fun), x0, t_span, t_eval, method, atol, rtol, max_step, first_step, dense_output);
    }

    /**
     * @brief Solve a linear time-invariant (LTI) state-space system with a constant input.
     *
     * Solves the continuous-time LTI system x' = Ax + Bu where `u` is constant
     * (provided by `u_const`). If `method == IntegrationMethod::Exact` and `A` is
     * invertible the function computes the matrix-exponential solution
     * x(t) = exp(A*(t - t0)) x0 + A^{-1} (exp(A*(t - t0)) - I) B u_const for
     * each requested time. If `A` is not invertible, the implementation falls back
     * to numeric integration using the generic `solve` function (RK45 by default).
     *
     * @param A State matrix (n x n).
     * @param B Input matrix (n x m).
     * @param x0 Initial state vector (n x 1) at time t_span.first.
     * @param u_const Constant input vector (m x 1). Used for the full integration.
     * @param t_span Pair of (t0, tf) specifying the integration interval.
     * @param t_eval Optional vector of time points at which to return the solution.
     *               If empty, the solver returns the state only at `tf` when using
     *               the exact method; the generic solver will produce a time series
     *               according to its adaptive stepping behavior.
     * @param method Integration method to use. Use `IntegrationMethod::Exact` to
     *               compute the analytical matrix-exponential solution when possible.
     * @param atol Absolute tolerance for adaptive solvers (used when numeric integration is chosen).
     * @param rtol Relative tolerance for adaptive solvers.
     * @param max_step Maximum allowed step size for numeric integrators (<=0 disables).
     * @param first_step Initial step size suggestion for adaptive integrators (<=0 lets solver pick).
     * @return SolveResult containing `t` (time points) and `x` (state vectors).
     */
    /**
     * @param C Optional output matrix C. If provided, `SolveResult::y` will be
     *          populated with y = Cx + Du (where available).
     * @param D Optional feedthrough matrix D. If provided, `SolveResult::y` will be
     *          populated with y = Cx + Du (where available).
     */
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

    /**
     * @brief Solve a linear time-invariant (LTI) state-space system with a time-varying input.
     *
     * Solves x' = Ax + Bu(t) where `u_func` is a callable that returns the
     * input vector at a given time. This overload wraps `u_func` into the generic
     * ODE solver and uses the provided `method` and tolerances to integrate.
     *
     * @param A State matrix (n x n).
     * @param B Input matrix (n x m).
     * @param x0 Initial state vector (n x 1) at time t_span.first.
     * @param u_func Callable `double -> Matrix` that returns the input vector at time t.
     * @param t_span Pair of (t0, tf) specifying the integration interval.
     * @param t_eval Optional vector of time points at which to return the solution.
     *               If empty, the solver will return the points generated by the integrator.
     * @param method Integration method to use for numeric integration (Exact is not used here).
     * @param atol Absolute tolerance for adaptive solvers.
     * @param rtol Relative tolerance for adaptive solvers.
     * @param max_step Maximum allowed step size for integrator (<=0 disables).
     * @param first_step Initial step size suggestion for adaptive integrators (<=0 lets solver pick).
     * @return SolveResult containing `t` (time points) and `x` (state vectors).
     */
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

template <typename Derived>
class StateSpaceBase {
   protected:
    StateSpaceBase(const Matrix& A, const Matrix& B, const Matrix& C, const Matrix& D)
        : A(A), B(B), C(C), D(D) {}

   public:
    Matrix output(const Matrix& x, const Matrix& u) const { return C * x + D * u; }

    StepResponse step(double tStart = 0.0, double tEnd = 10.0, Matrix uStep = Matrix()) const {
        if (uStep.size() == 0) {
            uStep = Matrix::Ones(B.cols(), 1);
        }

        return static_cast<const Derived*>(this)->stepImpl(tStart, tEnd, uStep);
    };

    FrequencyResponse bode(double fStart = 0.1, double fEnd = 1.0e4, size_t numFreq = 1000) const {
        return static_cast<const Derived*>(this)->bodeImpl(fStart, fEnd, numFreq);
    }

    const Eigen::MatrixXd A = {}, B = {}, C = {}, D = {};
};

class ContinuousStateSpace : public StateSpaceBase<ContinuousStateSpace> {
   public:
    ContinuousStateSpace(const Matrix& A, const Matrix& B, const Matrix& C, const Matrix& D)
        : StateSpaceBase(A, B, C, D) {}

    // Convert to discrete-time state space using specified method
    DiscreteStateSpace discretize(double Ts, DiscretizationMethod method = DiscretizationMethod::ZOH, std::optional<double> prewarp = std::nullopt) const;

   private:
    friend class StateSpaceBase<ContinuousStateSpace>;

    StepResponse      stepImpl(double tStart, double tEnd, Matrix uStep) const;
    FrequencyResponse bodeImpl(double fStart, double fEnd, size_t numFreq) const;
};

class DiscreteStateSpace : public StateSpaceBase<DiscreteStateSpace> {
   public:
    DiscreteStateSpace(const Matrix& A, const Matrix& B, const Matrix& C, const Matrix& D, double Ts)
        : StateSpaceBase(A, B, C, D), Ts(Ts) {}

   private:
    friend class StateSpaceBase<DiscreteStateSpace>;

    StepResponse      stepImpl(double tStart, double tEnd, Matrix uStep) const;
    FrequencyResponse bodeImpl(double fStart, double fEnd, size_t numFreq) const;

    const double Ts;
};

template <class Derived>
Derived ss(const Matrix& A, const Matrix& B, const Matrix& C, const Matrix& D, std::optional<double> Ts = std::nullopt) {
    if (Ts.has_value()) {
        return Derived{A, B, C, D, Ts.value()};
    } else {
        return Derived{A, B, C, D};
    }
}

template <class Derived>
Derived ss(const TransferFunction& tf) {
    // Convert transfer function to state space using controllable canonical form
    const int n = tf.den.cols() - 1;  // Order of the system
    const int m = tf.num.cols() - 1;  // Order of the numerator

    Matrix A = Matrix::Zero(n, n);
    Matrix B = Matrix::Zero(n, 1);
    Matrix C = Matrix::Zero(1, n);
    Matrix D = Matrix::Zero(1, 1);

    // Fill A matrix
    for (int i = 0; i < n - 1; ++i) {
        A(i, i + 1) = 1.0;
    }
    for (int i = 0; i < n; ++i) {
        A(n - 1, i) = -tf.den(0, i) / tf.den(0, n);
    }

    // Fill B matrix
    B(n - 1, 0) = 1.0;

    // Fill C and D matrices
    for (int i = 0; i <= m; ++i) {
        C(0, i) = tf.num(0, i) / tf.den(0, n);
    }
    if (m < n) {
        D(0, 0) = 0.0;
    } else {
        D(0, 0) = tf.num(0, m) / tf.den(0, n);
    }

    return ss<Derived>(A, B, C, D);
}

template <typename SystemType>
auto c2d(const SystemType& sys, double Ts, DiscretizationMethod method = DiscretizationMethod::ZOH, std::optional<double> prewarp = std::nullopt) {
    return sys.discretize(Ts, method, prewarp);
}

template <typename SystemType>
std::string formatStateSpaceMatrices(const SystemType& sys) {
    std::string result = "A = \n";
    for (int i = 0; i < sys.A.rows(); ++i) {
        for (int j = 0; j < sys.A.cols(); ++j) {
            result += std::format("{:>10.4f}", sys.A(i, j));
        }
        result += "\n";
    }
    result += "\nB = \n";
    for (int i = 0; i < sys.B.rows(); ++i) {
        for (int j = 0; j < sys.B.cols(); ++j) {
            result += std::format("{:>10.4f}", sys.B(i, j));
        }
        result += "\n";
    }
    result += "\nC = \n";
    for (int i = 0; i < sys.C.rows(); ++i) {
        for (int j = 0; j < sys.C.cols(); ++j) {
            result += std::format("{:>10.4f}", sys.C(i, j));
        }
        result += "\n";
    }
    result += "\nD = \n";
    for (int i = 0; i < sys.D.rows(); ++i) {
        for (int j = 0; j < sys.D.cols(); ++j) {
            result += std::format("{:>10.4f}", sys.D(i, j));
        }
        result += "\n";
    }
    return result;
}
};  // namespace control

template <>
struct std::formatter<control::ContinuousStateSpace> {
    constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }

    auto format(const control::ContinuousStateSpace& sys, std::format_context& ctx) const {
        return std::format_to(ctx.out(), "{}", control::formatStateSpaceMatrices(sys));
    }
};

template <>
struct std::formatter<control::DiscreteStateSpace> {
    constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }

    auto format(const control::DiscreteStateSpace& sys, std::format_context& ctx) const {
        return std::format_to(ctx.out(), "{}", control::formatStateSpaceMatrices(sys));
    }
};
