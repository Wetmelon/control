#pragma once

#include <variant>

#include "colvec.hpp"
#include "matrix.hpp"
#include "matrix_functions.hpp"

namespace wetmelon::control {

/**
 * @struct IntegrationResult
 * @brief Result of an integration step
 *
 * @tparam NX Number of states
 * @tparam T  Scalar type
 */
template<size_t NX, typename T = double>
struct IntegrationResult {
    ColVec<NX, T> x;
    T             error;
};

/**
 * @brief Discrete-time integrator (no integration, just one step)
 */
template<size_t NX, typename T = double>
struct Discrete {
    template<size_t NU>
    constexpr IntegrationResult<NX, T> evolve(const Matrix<NX, NX, T>& A, const Matrix<NX, NU, T>& B, const ColVec<NX, T>& x, const ColVec<NU, T>& u) const {
        return {A * x + B * u, 0.0};
    }
};

/**
 * @brief Exact integrator for LTI systems
 *
 *
 *
 * @tparam NX Number of states
 * @tparam T  Scalar type
 */
template<size_t NX, typename T = double>
struct Exact {

    /**
     * @brief Evolve LTI system state exactly over step h
     *
     * @param A     System matrix
     * @param B     Input matrix
     * @param x     Current state
     * @param u     Input (held constant over the duration of the step)
     * @param h     [s] Step size
     *
     * @return
     */
    template<size_t NU>
    constexpr IntegrationResult<NX, T> evolve(const Matrix<NX, NX, T>& A, const Matrix<NX, NU, T>& B, const ColVec<NX, T>& x, const ColVec<NU, T>& u, T h) const {
        // Augmented matrix
        Matrix<NX + NU, NX + NU, T> M = Matrix<NX + NU, NX + NU, T>::zeros();

        M.template block<NX, NX>(0, 0) = A;
        M.template block<NX, NU>(0, NX) = B;

        // Matrix exponential
        auto expM = mat::exp(M * h);

        // Extract blocks
        auto expAh = expM.template block<NX, NX>(0, 0);
        auto Gamma = expM.template block<NX, NU>(0, NX);

        // Exact update
        ColVec<NX, T> x_next = expAh * x + Gamma * u;

        return {x_next, T(0)};
    }
};

/**
 * @brief Forward Euler integrator
 *
 * Good for simple problems, but not very accurate or stable. Use with small step sizes.
 *
 * @tparam NX Number of states
 * @tparam T  Scalar type
 */
template<size_t NX, typename T = double>
struct ForwardEuler {

    /**
     * @brief Evolve the system state using Forward Euler method
     *
     * @tparam NX   Number of states
     * @tparam T    Scalar type
     * @tparam NU   Number of inputs
     *
     * @param f     Right-hand side function: dx/dt = f(t, x)
     * @param x     Current state
     * @param t     [s] Current time (for passing to f)
     * @param h     [s] Step size
     *
     * @return Next state and estimated error (always 0 for Forward Euler)
     */
    constexpr IntegrationResult<NX, T> evolve(auto&& f, const ColVec<NX, T>& x, T t, T h) const {
        k1 = f(t, x);
        return {x + h * k1, 0.0};
    }

    template<size_t NU>
    constexpr IntegrationResult<NX, T> evolve(const Matrix<NX, NX, T>& A, const Matrix<NX, NU, T>& B, const ColVec<NX, T>& x, const ColVec<NU, T>& u, T h) const {
        x_next = x + h * (A * x + B * u);
        return {x_next, 0.0};
    }

private:
    mutable ColVec<NX, T> k1, x_next;
};

/**
 * @brief Backward Euler integrator
 *
 * Implicit method, more stable than Forward Euler for stiff problems.
 *
 */
template<size_t NX, typename T = double>
struct BackwardEuler {

    /**
     * @brief Evolve LTI system state using Backward Euler method
     *
     * @tparam NX   Number of states
     * @tparam T    Scalar type
     * @tparam NU   Number of inputs
     *
     * @param A     System matrix
     * @param B     Input matrix
     * @param x     Current state
     * @param u     Current input
     * @param h     [s] Step size
     *
     * @return      Next state and estimated error (always 0 for Backward Euler)
     */
    template<size_t NU>
    constexpr IntegrationResult<NX, T> evolve(const Matrix<NX, NX, T>& A, const Matrix<NX, NU, T>& B, const ColVec<NX, T>& x, const ColVec<NU, T>& u, T h) const {
        Matrix I = Matrix<NX, NX, T>::identity();
        x_next = (I - h * A).inverse() * (x + h * B * u);
        return {x_next, 0.0};
    }

    /**
     * @brief       Evolve nonlinear system state using Backward Euler method
     *
     * @param f     Right-hand side function: dx/dt = f(t, x)
     * @param x     Current state
     * @param t     [s] Current time (for passing to f)
     * @param h     [s] Step size
     *
     * @return      Next state and estimated error (always 0 for Backward Euler)
     */
    constexpr IntegrationResult<NX, T> evolve(auto&& f, const ColVec<NX, T>& x, T t, T h) const {
        const size_t max_iter = 10;
        const double tol = 1e-10;

        // Initial guess: explicit Euler
        ColVec<NX, T> y = x + h * f(t, x);
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
    mutable ColVec<NX, T> y_next, x_next;
};

/**
 * @brief Backward Differentiation Formula 2 (BDF2) integrator
 *
 * Multistep implicit method, good stability and accuracy for stiff problems.
 *
 */
template<size_t NX, typename T = double>
struct BDF2 {
    BDF2() : first_step(true) {}

    // BDF2 coefficients: x_{n+1} = (4/3)*x_n - (1/3)*x_{n-1} + (2/3)*h*f(t_{n+1}, x_{n+1})
    static constexpr double c0 = 4.0 / 3.0;  // Coefficient for x_n
    static constexpr double c1 = -1.0 / 3.0; // Coefficient for x_{n-1}
    static constexpr double c2 = 2.0 / 3.0;  // Coefficient for h*f

    /**
     * @brief
     *
     * @param A
     * @param B
     * @param x
     * @param u
     * @param h
     *
     * @return
     */
    template<size_t NU>
    constexpr IntegrationResult<NX, T> evolve(const Matrix<NX, NX, T>& A, const Matrix<NX, NU, T>& B, const ColVec<NX, T>& x, const ColVec<NU, T>& u, T h) const {
        Matrix I = Matrix<NX, NX, T>::identity();

        if (first_step) {
            // Use Backward Euler for first step
            Matrix x_next = (I - h * A).inverse() * (x + h * B * u);
            x_prev = x;
            first_step = false;
            return {x_next, 0.0};
        }

        // BDF2: (I - (2/3)*h*A)*x_{n+1} = (4/3)*x_n - (1/3)*x_{n-1} + (2/3)*h*B*u
        Matrix lhs = I - c2 * h * A;
        ColVec rhs = c0 * x + c1 * x_prev + c2 * h * B * u;
        ColVec x_next = lhs.inverse() * rhs;

        x_prev = x;
        return {x_next, 0.0};
    }

    template<typename F>
    constexpr IntegrationResult<NX, T> evolve(F&& f, const ColVec<NX, T>& x, T t, T h) const {
        constexpr size_t max_iter = 20;
        constexpr double tol = 1e-10;

        if (first_step) {
            // Use Backward Euler for first step
            ColVec y = x + h * f(t, x); // Initial guess
            for (size_t i = 0; i < max_iter; ++i) {
                ColVec y_next = x + h * f(t + h, y);
                if ((y_next - y).norm() <= tol) {
                    y = y_next;
                    break;
                }
                y = y_next;
            }
            x_prev = x;
            first_step = false;
            return {y, 0.0};
        }

        // BDF2: x_{n+1} = (4/3)*x_n - (1/3)*x_{n-1} + (2/3)*h*f(t_{n+1}, x_{n+1})
        // Solve using fixed-point iteration
        ColVec y = c0 * x + c1 * x_prev; // Initial guess without implicit term

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
    mutable ColVec<NX, T> x_prev;     // Previous state (for multistep)
    mutable bool          first_step; // Track if this is the first step
};

/**
 * @brief  Trapezoidal (Tustin) integrator
 *
 * Good accuracy and stability for a wide range of problems.
 *
 */
template<size_t NX, typename T = double>
struct Trapezoidal {
    template<size_t NU>
    constexpr IntegrationResult<NX, T> evolve(const Matrix<NX, NX, T>& A, const Matrix<NX, NU, T>& B, const ColVec<NX, T>& x, const ColVec<NU, T>& u, T h) const {
        Matrix I = Matrix<NX, NX, T>::identity();
        Matrix M = (I - 0.5 * h * A).inverse();
        x_next = M * ((I + 0.5 * h * A) * x + h * B * u);
        return {x_next, 0.0};
    }

    template<typename F>
    constexpr IntegrationResult<NX, T> evolve(F&& f, const ColVec<NX, T>& x, T t, T h) const {
        k1 = f(t, x);
        xp = x + h * k1;
        k2 = f(t + h, xp);
        return {x + 0.5 * h * (k1 + k2), 0.0};
    }

private:
    mutable ColVec<NX, T> k1, k2, xp, x_next;
};

template<size_t NX, typename T = double>
struct RK4 {
    template<size_t NU>
    constexpr IntegrationResult<NX, T> evolve(const Matrix<NX, NX, T>& A, const Matrix<NX, NU, T>& B, const ColVec<NX, T>& x, const ColVec<NU, T>& u, T h) const {
        // RK4 for LTI system
        auto f = [&](double, const ColVec<NX, T>& x) { return A * x + B * u; };
        return evolve(f, x, 0.0, h);
    }

    template<typename F>
    constexpr IntegrationResult<NX, T> evolve(F&& f, const ColVec<NX, T>& x, T t, T h) const {
        k1 = f(t, x);
        k2 = f(t + 0.5 * h, x + 0.5 * h * k1);
        k3 = f(t + 0.5 * h, x + 0.5 * h * k2);
        k4 = f(t + h, x + h * k3);

        x_next = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
        return {x_next, 0.0};
    }

private:
    mutable ColVec<NX, T> k1, k2, k3, k4, x_next;
};

template<size_t NX, typename T = double>
struct RK45 {
    template<size_t NU>
    constexpr IntegrationResult<NX, T> evolve(const Matrix<NX, NX, T>& A, const Matrix<NX, NU, T>& B, const ColVec<NX, T>& x, const ColVec<NU, T>& u, T h) const {
        // Evolution of LTI system is a special case of generic ODE solver
        auto f = [&](T, const ColVec<NX, T>& x) { return A * x + B * u; };
        return evolve(f, x, 0.0, h);
    }

    template<typename F>
    constexpr IntegrationResult<NX, T> evolve(F&& f, const ColVec<NX, T>& x, T t, T h) const {
        // RK45 coefficients (precomputed to avoid divisions in hot loop)
        constexpr T c2 = 1.0 / 4.0, c3 = 3.0 / 8.0, c4 = 12.0 / 13.0, c6 = 1.0 / 2.0;
        constexpr T a21 = 1.0 / 4.0;
        constexpr T a31 = 3.0 / 32.0, a32 = 9.0 / 32.0;
        constexpr T a41 = 1932.0 / 2197.0, a42 = -7200.0 / 2197.0, a43 = 7296.0 / 2197.0;
        constexpr T a51 = 439.0 / 216.0, a52 = -8.0, a53 = 3680.0 / 513.0, a54 = -845.0 / 4104.0;
        constexpr T a61 = -8.0 / 27.0, a62 = 2.0, a63 = -3544.0 / 2565.0, a64 = 1859.0 / 4104.0, a65 = -11.0 / 40.0;
        constexpr T b41 = 25.0 / 216.0, b43 = 1408.0 / 2565.0, b44 = 2197.0 / 4104.0, b45 = -1.0 / 5.0;
        constexpr T b51 = 16.0 / 135.0, b53 = 6656.0 / 12825.0, b54 = 28561.0 / 56430.0, b55 = -9.0 / 50.0, b56 = 2.0 / 55.0;

        k1 = f(t, x);
        k2 = f(t + h * c2, x + h * a21 * k1);
        k3 = f(t + h * c3, x + h * (a31 * k1 + a32 * k2));
        k4 = f(t + h * c4, x + h * (a41 * k1 + a42 * k2 + a43 * k3));
        k5 = f(t + h, x + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4));
        k6 = f(t + h * c6, x + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5));

        x4 = x + h * (b41 * k1 + b43 * k3 + b44 * k4 + b45 * k5);
        x5 = x + h * (b51 * k1 + b53 * k3 + b54 * k4 + b55 * k5 + b56 * k6);

        T error = (x5 - x4).norm();
        return {x5, error};
    }

private:
    mutable ColVec<NX, T> k1, k2, k3, k4, k5, k6, x4, x5;
};

template<size_t NX, typename T = double>
struct Heun {
    template<size_t NU>
    constexpr IntegrationResult<NX, T> evolve(const Matrix<NX, NX, T>& A, const Matrix<NX, NU, T>& B, const ColVec<NX, T>& x, const ColVec<NU, T>& u, T h) const {
        auto f = [&](T, const ColVec<NX, T>& x) { return A * x + B * u; };
        return evolve(f, x, 0.0, h);
    }

    template<typename F>
    constexpr IntegrationResult<NX, T> evolve(F&& f, const ColVec<NX, T>& x, T t, T h) const {
        // Heun's method (Improved Euler, RK2)
        // Predictor-corrector: first estimate then correct
        constexpr double c2 = 1.0;
        constexpr double b1 = 0.5, b2 = 0.5;

        k1 = f(t, x);
        k2 = f(t + c2 * h, x + h * k1);
        return {x + h * (b1 * k1 + b2 * k2), 0.0};
    }

private:
    mutable ColVec<NX, T> k1, k2;
};

template<size_t NX, typename T = double>
struct RK23 {
    template<size_t NU>
    constexpr IntegrationResult<NX, T> evolve(const Matrix<NX, NX, T>& A, const Matrix<NX, NU, T>& B, const ColVec<NX, T>& x, const ColVec<NU, T>& u, T h) const {
        auto f = [&](double, const ColVec<NX, T>& x) { return A * x + B * u; };
        return evolve(f, x, 0.0, h);
    }

    template<typename F>
    constexpr IntegrationResult<NX, T> evolve(F&& f, const ColVec<NX, T>& x, T t, T h) const {
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
        return {x3, error}; // Return 3rd order solution
    }

private:
    mutable ColVec<NX, T> k1, k2, k3, k4, x2, x3;
};

template<size_t NX, typename T = double>
struct RK3 {
    template<size_t NU>
    constexpr IntegrationResult<NX, T> evolve(const Matrix<NX, NX, T>& A, const Matrix<NX, NU, T>& B, const ColVec<NX, T>& x, const ColVec<NU, T>& u, T h) const {
        auto f = [&](double, const ColVec<NX, T>& x) { return A * x + B * u; };
        return evolve(f, x, 0.0, h);
    }

    template<typename F>
    constexpr IntegrationResult<NX, T> evolve(F&& f, const ColVec<NX, T>& x, T t, T h) const {
        // Classical 3rd order Runge-Kutta
        constexpr double c2 = 0.5;
        constexpr double a21 = 0.5;
        constexpr double a31 = -1.0, a32 = 2.0;
        constexpr double b1 = 1.0 / 6.0, b2 = 4.0 / 6.0, b3 = 1.0 / 6.0;

        k1 = f(t, x);
        k2 = f(t + c2 * h, x + h * a21 * k1);
        k3 = f(t + h, x + h * (a31 * k1 + a32 * k2));
        return {x + h * (b1 * k1 + b2 * k2 + b3 * k3), 0.0};
    }

private:
    mutable ColVec<NX, T> k1, k2, k3;
};

template<size_t NX, typename T = double>
struct DP5 {
    template<size_t NU>
    constexpr IntegrationResult<NX, T> evolve(const Matrix<NX, NX, T>& A, const Matrix<NX, NU, T>& B, const ColVec<NX, T>& x, const ColVec<NU, T>& u, T h) const {
        auto f = [&](double, const ColVec<NX, T>& x) { return A * x + B * u; };
        return evolve(f, x, 0.0, h);
    }

    template<typename F>
    constexpr IntegrationResult<NX, T> evolve(F&& f, const ColVec<NX, T>& x, T t, T h) const {
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
    mutable ColVec<NX, T> k1, k2, k3, k4, k5, k6, x_next;
};

template<size_t NX, typename T = double>
struct Integrator : public std::variant<
                        Discrete<NX, T>,
                        ForwardEuler<NX, T>,
                        BackwardEuler<NX, T>,
                        Trapezoidal<NX, T>,
                        BDF2<NX, T>,
                        Heun<NX, T>,
                        RK3<NX, T>,
                        RK23<NX, T>,
                        RK4<NX, T>,
                        RK45<NX, T>,
                        DP5<NX, T>,
                        Exact<NX, T> > {};

}; // namespace wetmelon::control