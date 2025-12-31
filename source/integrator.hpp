#pragma once

#include <concepts>
#include <variant>

#include "types.hpp"
#include "unsupported/Eigen/MatrixFunctions"  // IWYU pragma: keep

namespace control {

// Type alias for ODE right-hand side function: dx/dt = f(t, x)
using ODEFunction = std::function<ColVec(double, const ColVec&)>;

struct IntegrationResult {
    ColVec x;
    double error;
};

// Concepts for integrator classification
template <typename T>
concept FixedStepIntegrator = requires(T integrator) {
    // Must have a member indicating it's a fixed step integrator
    { T::is_fixed_step } -> std::convertible_to<bool>;
} && T::is_fixed_step;

template <typename T>
concept AdaptiveStepIntegrator = requires(T integrator) {
    // Must have a member indicating it's an adaptive step integrator
    { T::is_adaptive_step } -> std::convertible_to<bool>;
} && T::is_adaptive_step;

// Concept for integrators that support general ODEs
template <typename T>
concept ODEIntegrator = requires(T integrator, ODEFunction f, const ColVec& x, double t, double h) {
    { integrator.evolve(f, x, t, h) } -> std::same_as<IntegrationResult>;
};

// Concept for integrators that support LTI systems
template <typename T>
concept LTIIntegrator = requires(T integrator, const Matrix& A, const Matrix& B, const ColVec& x, const ColVec& u, double h) {
    { integrator.evolve(A, B, x, u, h) } -> std::same_as<IntegrationResult>;
};

template <typename Derived>
struct IntegratorBase {
   public:
    template <typename F>
    IntegrationResult evolve(F&& f, const ColVec& x, double t, double h) const {
        return static_cast<const Derived*>(this)->evolve(std::forward<F>(f), x, t, h);
    }

    IntegrationResult evolve(const Matrix& A, const Matrix& B, const ColVec& x, const ColVec& u, double h) const {
        return static_cast<const Derived*>(this)->evolve(A, B, x, u, h);
    }
};

struct Discrete : public IntegratorBase<Discrete> {
    static constexpr bool is_fixed_step    = true;
    static constexpr bool is_adaptive_step = false;

    IntegrationResult evolve(const Matrix& A, const Matrix& B, const ColVec& x, const ColVec& u) const {
        return {A * x + B * u, 0.0};
    }

    template <typename F>
    IntegrationResult evolve(F&& /* f */, const ColVec& /* x */, double /* t */, double /* h */) const {
        // For general ODEs, no discrete solution available
        throw std::runtime_error("Discrete integrator not implemented for general ODEs");
    }
};

struct ForwardEuler : public IntegratorBase<ForwardEuler> {
    static constexpr bool is_fixed_step    = true;
    static constexpr bool is_adaptive_step = false;

    template <typename F>
    IntegrationResult evolve(F&& f, const ColVec& x, double t, double h) const {
        k1 = f(t, x);
        return {x + h * k1, 0.0};
    }

    IntegrationResult evolve(const Matrix& A, const Matrix& B, const ColVec& x, const ColVec& u, double h) const {
        x_next = x + h * (A * x + B * u);
        return {x_next, 0.0};
    }

   private:
    mutable ColVec k1, x_next;
};

struct BackwardEuler : public IntegratorBase<BackwardEuler> {
    static constexpr bool is_fixed_step    = true;
    static constexpr bool is_adaptive_step = false;

    IntegrationResult evolve(const Matrix& A, const Matrix& B, const ColVec& x, const ColVec& u, double h) const {
        Matrix I = Matrix::Identity(A.rows(), A.cols());
        x_next   = (I - h * A).inverse() * (x + h * B * u);
        return {x_next, 0.0};
    }

    template <typename F>
    IntegrationResult evolve(F&& f, const ColVec& x, double t, double h) const {
        const size_t max_iter = 10;
        const double tol      = 1e-10;

        // Initial guess: explicit Euler
        ColVec y = x + h * f(t, x);
        for (size_t i = 0; i < max_iter; ++i) {
            y_next = x + h * f(t + h, y);
            if ((y_next - y).norm() <= tol) {
                y = y_next;
                break;
            }
            y = y_next;
        }
        return {y, 0.0};
    }

   private:
    mutable ColVec y_next, x_next;
};

struct BDF2 : public IntegratorBase<BDF2> {
    static constexpr bool is_fixed_step    = true;
    static constexpr bool is_adaptive_step = false;

    BDF2() : first_step(true) {}

    // BDF2 coefficients: x_{n+1} = (4/3)*x_n - (1/3)*x_{n-1} + (2/3)*h*f(t_{n+1}, x_{n+1})
    static constexpr double c0 = 4.0 / 3.0;   // Coefficient for x_n
    static constexpr double c1 = -1.0 / 3.0;  // Coefficient for x_{n-1}
    static constexpr double c2 = 2.0 / 3.0;   // Coefficient for h*f

    IntegrationResult evolve(const Matrix& A, const Matrix& B, const ColVec& x, const ColVec& u, double h) const {
        Matrix I = Matrix::Identity(A.rows(), A.cols());

        if (first_step) {
            // Use Backward Euler for first step
            Matrix x_next = (I - h * A).inverse() * (x + h * B * u);
            x_prev        = x;
            first_step    = false;
            return {x_next, 0.0};
        }

        // BDF2: (I - (2/3)*h*A)*x_{n+1} = (4/3)*x_n - (1/3)*x_{n-1} + (2/3)*h*B*u
        Matrix lhs    = I - c2 * h * A;
        ColVec rhs    = c0 * x + c1 * x_prev + c2 * h * B * u;
        ColVec x_next = lhs.inverse() * rhs;

        x_prev = x;
        return {x_next, 0.0};
    }

    template <typename F>
    IntegrationResult evolve(F&& f, const ColVec& x, double t, double h) const {
        constexpr size_t max_iter = 20;
        constexpr double tol      = 1e-10;

        if (first_step) {
            // Use Backward Euler for first step
            ColVec y = x + h * f(t, x);  // Initial guess
            for (size_t i = 0; i < max_iter; ++i) {
                ColVec y_next = x + h * f(t + h, y);
                if ((y_next - y).norm() <= tol) {
                    y = y_next;
                    break;
                }
                y = y_next;
            }
            x_prev     = x;
            first_step = false;
            return {y, 0.0};
        }

        // BDF2: x_{n+1} = (4/3)*x_n - (1/3)*x_{n-1} + (2/3)*h*f(t_{n+1}, x_{n+1})
        // Solve using fixed-point iteration
        ColVec y = c0 * x + c1 * x_prev;  // Initial guess without implicit term

        for (size_t i = 0; i < max_iter; ++i) {
            ColVec y_next = c0 * x + c1 * x_prev + c2 * h * f(t + h, y);
            if ((y_next - y).norm() <= tol) {
                y = y_next;
                break;
            }
            y = y_next;
        }

        x_prev = x;
        return {y, 0.0};
    }

    // Reset method for starting new integrations
    void reset() const {
        first_step = true;
    }

   private:
    mutable ColVec x_prev;      // Previous state (for multistep)
    mutable bool   first_step;  // Track if this is the first step
};

struct Trapezoidal : public IntegratorBase<Trapezoidal> {
    static constexpr bool is_fixed_step    = true;
    static constexpr bool is_adaptive_step = false;

    IntegrationResult evolve(const Matrix& A, const Matrix& B, const ColVec& x, const ColVec& u, double h) const {
        Matrix I = Matrix::Identity(A.rows(), A.cols());
        Matrix M = (I - 0.5 * h * A).inverse();
        x_next   = M * ((I + 0.5 * h * A) * x + h * B * u);
        return {x_next, 0.0};
    }

    template <typename F>
    IntegrationResult evolve(F&& f, const ColVec& x, double t, double h) const {
        k1 = f(t, x);
        xp = x + h * k1;
        k2 = f(t + h, xp);
        return {x + 0.5 * h * (k1 + k2), 0.0};
    }

   private:
    mutable ColVec k1, k2, xp, x_next;
};

struct RK4 : public IntegratorBase<RK4> {
    static constexpr bool is_fixed_step    = true;
    static constexpr bool is_adaptive_step = false;

    IntegrationResult evolve(const Matrix& A, const Matrix& B, const ColVec& x, const ColVec& u, double h) const {
        // RK4 for LTI system
        auto f = [&](double, const ColVec& x) { return A * x + B * u; };
        return evolve(f, x, 0.0, h);
    }

    template <typename F>
    IntegrationResult evolve(F&& f, const ColVec& x, double t, double h) const {
        k1 = f(t, x);
        k2 = f(t + 0.5 * h, x + 0.5 * h * k1);
        k3 = f(t + 0.5 * h, x + 0.5 * h * k2);
        k4 = f(t + h, x + h * k3);

        x_next = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
        return {x_next, 0.0};
    }

   private:
    mutable ColVec k1, k2, k3, k4, x_next;
};

struct RK45 : public IntegratorBase<RK45> {
    static constexpr bool is_fixed_step    = true;  // Can be used with fixed steps (ignoring error estimate)
    static constexpr bool is_adaptive_step = true;  // Can be used with adaptive steps (using error estimate)

    IntegrationResult evolve(const Matrix& A, const Matrix& B, const ColVec& x, const ColVec& u, double h) const {
        // Evolution of LTI system is a special case of generic ODE solver
        auto f = [&](double, const ColVec& x) { return A * x + B * u; };
        return evolve(f, x, 0.0, h);
    }

    template <typename F>
    IntegrationResult evolve(F&& f, const ColVec& x, double t, double h) const {
        // RK45 coefficients (precomputed to avoid divisions in hot loop)
        constexpr double c2 = 1.0 / 4.0, c3 = 3.0 / 8.0, c4 = 12.0 / 13.0, c6 = 1.0 / 2.0;
        constexpr double a21 = 1.0 / 4.0;
        constexpr double a31 = 3.0 / 32.0, a32 = 9.0 / 32.0;
        constexpr double a41 = 1932.0 / 2197.0, a42 = -7200.0 / 2197.0, a43 = 7296.0 / 2197.0;
        constexpr double a51 = 439.0 / 216.0, a52 = -8.0, a53 = 3680.0 / 513.0, a54 = -845.0 / 4104.0;
        constexpr double a61 = -8.0 / 27.0, a62 = 2.0, a63 = -3544.0 / 2565.0, a64 = 1859.0 / 4104.0, a65 = -11.0 / 40.0;
        constexpr double b41 = 25.0 / 216.0, b43 = 1408.0 / 2565.0, b44 = 2197.0 / 4104.0, b45 = -1.0 / 5.0;
        constexpr double b51 = 16.0 / 135.0, b53 = 6656.0 / 12825.0, b54 = 28561.0 / 56430.0, b55 = -9.0 / 50.0, b56 = 2.0 / 55.0;

        k1 = f(t, x);
        k2 = f(t + h * c2, x + h * a21 * k1);
        k3 = f(t + h * c3, x + h * (a31 * k1 + a32 * k2));
        k4 = f(t + h * c4, x + h * (a41 * k1 + a42 * k2 + a43 * k3));
        k5 = f(t + h, x + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4));
        k6 = f(t + h * c6, x + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5));

        x4 = x + h * (b41 * k1 + b43 * k3 + b44 * k4 + b45 * k5);
        x5 = x + h * (b51 * k1 + b53 * k3 + b54 * k4 + b55 * k5 + b56 * k6);

        double error = (x5 - x4).norm();
        return {x5, error};
    }

   private:
    mutable ColVec k1, k2, k3, k4, k5, k6, x4, x5;
};

struct Heun : public IntegratorBase<Heun> {
    static constexpr bool is_fixed_step    = true;
    static constexpr bool is_adaptive_step = false;

    IntegrationResult evolve(const Matrix& A, const Matrix& B, const ColVec& x, const ColVec& u, double h) const {
        auto f = [&](double, const ColVec& x) { return A * x + B * u; };
        return evolve(f, x, 0.0, h);
    }

    template <typename F>
    IntegrationResult evolve(F&& f, const ColVec& x, double t, double h) const {
        // Heun's method (Improved Euler, RK2)
        // Predictor-corrector: first estimate then correct
        constexpr double c2 = 1.0;
        constexpr double b1 = 0.5, b2 = 0.5;

        k1 = f(t, x);
        k2 = f(t + c2 * h, x + h * k1);
        return {x + h * (b1 * k1 + b2 * k2), 0.0};
    }

   private:
    mutable ColVec k1, k2;
};

struct RK23 : public IntegratorBase<RK23> {
    static constexpr bool is_fixed_step    = true;
    static constexpr bool is_adaptive_step = true;  // Has embedded error estimate

    IntegrationResult evolve(const Matrix& A, const Matrix& B, const ColVec& x, const ColVec& u, double h) const {
        auto f = [&](double, const ColVec& x) { return A * x + B * u; };
        return evolve(f, x, 0.0, h);
    }

    template <typename F>
    IntegrationResult evolve(F&& f, const ColVec& x, double t, double h) const {
        // Bogacki-Shampine (2,3) pair
        constexpr double c2 = 0.5, c3 = 0.75;
        constexpr double a21 = 0.5;
        constexpr double a32 = 0.75;
        constexpr double b1 = 2.0 / 9.0, b2 = 1.0 / 3.0, b3 = 4.0 / 9.0;
        constexpr double e1 = 7.0 / 24.0, e2 = 0.25, e3 = 1.0 / 3.0, e4 = 0.125;

        k1 = f(t, x);
        k2 = f(t + c2 * h, x + h * a21 * k1);
        k3 = f(t + c3 * h, x + h * a32 * k2);

        x3 = x + h * (b1 * k1 + b2 * k2 + b3 * k3);
        k4 = f(t + h, x3);

        // 2nd order solution (for error estimate)
        x2 = x + h * (e1 * k1 + e2 * k2 + e3 * k3 + e4 * k4);

        double error = (x3 - x2).norm();
        return {x3, error};  // Return 3rd order solution
    }

   private:
    mutable ColVec k1, k2, k3, k4, x2, x3;
};

struct RK3 : public IntegratorBase<RK3> {
    static constexpr bool is_fixed_step    = true;
    static constexpr bool is_adaptive_step = false;

    IntegrationResult evolve(const Matrix& A, const Matrix& B, const ColVec& x, const ColVec& u, double h) const {
        auto f = [&](double, const ColVec& x) { return A * x + B * u; };
        return evolve(f, x, 0.0, h);
    }

    template <typename F>
    IntegrationResult evolve(F&& f, const ColVec& x, double t, double h) const {
        // Classical 3rd order Runge-Kutta
        constexpr double c2  = 0.5;
        constexpr double a21 = 0.5;
        constexpr double a31 = -1.0, a32 = 2.0;
        constexpr double b1 = 1.0 / 6.0, b2 = 4.0 / 6.0, b3 = 1.0 / 6.0;

        k1 = f(t, x);
        k2 = f(t + c2 * h, x + h * a21 * k1);
        k3 = f(t + h, x + h * (a31 * k1 + a32 * k2));
        return {x + h * (b1 * k1 + b2 * k2 + b3 * k3), 0.0};
    }

   private:
    mutable ColVec k1, k2, k3;
};

struct DP5 : public IntegratorBase<DP5> {
    static constexpr bool is_fixed_step    = true;
    static constexpr bool is_adaptive_step = false;

    IntegrationResult evolve(const Matrix& A, const Matrix& B, const ColVec& x, const ColVec& u, double h) const {
        auto f = [&](double, const ColVec& x) { return A * x + B * u; };
        return evolve(f, x, 0.0, h);
    }

    template <typename F>
    IntegrationResult evolve(F&& f, const ColVec& x, double t, double h) const {
        // Dormand-Prince 5th order (fixed-step)
        // Using the same coefficients as RK45 but only returning the 5th order solution
        constexpr double c2 = 1.0 / 5.0, c3 = 3.0 / 10.0, c4 = 4.0 / 5.0, c5 = 8.0 / 9.0;
        constexpr double a21 = 1.0 / 5.0;
        constexpr double a31 = 3.0 / 40.0, a32 = 9.0 / 40.0;
        constexpr double a41 = 44.0 / 45.0, a42 = -56.0 / 15.0, a43 = 32.0 / 9.0;
        constexpr double a51 = 19372.0 / 6561.0, a52 = -25360.0 / 2187.0, a53 = 64448.0 / 6561.0, a54 = -212.0 / 729.0;
        constexpr double a61 = 9017.0 / 3168.0, a62 = -355.0 / 33.0, a63 = 46732.0 / 5247.0, a64 = 49.0 / 176.0, a65 = -5103.0 / 18656.0;
        constexpr double b1 = 35.0 / 384.0, b3 = 500.0 / 1113.0, b4 = 125.0 / 192.0, b5 = -2187.0 / 6784.0, b6 = 11.0 / 84.0;

        k1 = f(t, x);
        k2 = f(t + c2 * h, x + h * a21 * k1);
        k3 = f(t + c3 * h, x + h * (a31 * k1 + a32 * k2));
        k4 = f(t + c4 * h, x + h * (a41 * k1 + a42 * k2 + a43 * k3));
        k5 = f(t + c5 * h, x + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4));
        k6 = f(t + h, x + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5));

        x_next = x + h * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6);
        return {x_next, 0.0};
    }

   private:
    mutable ColVec k1, k2, k3, k4, k5, k6, x_next;
};

struct Exact : public IntegratorBase<Exact> {
    static constexpr bool is_fixed_step    = true;
    static constexpr bool is_adaptive_step = false;

    IntegrationResult evolve(const Matrix& A, const Matrix& /*B*/, const ColVec& x, const ColVec& /*u*/, double h) const {
        Matrix expAh  = (A * h).exp();
        ColVec x_next = expAh * x;
        return {x_next, 0.0};
    }

    template <typename F>
    IntegrationResult evolve(F&& /* f */, const ColVec& /* x */, double /* t */, double /* h */) const {
        throw std::runtime_error("Exact integrator not implemented for general ODEs");
    }
};

struct Integrator : public std::variant<Discrete, ForwardEuler, BackwardEuler, Trapezoidal, BDF2, Heun, RK3, RK23, RK4, RK45, DP5, Exact> {
    using variant::variant;
};

};  // namespace control