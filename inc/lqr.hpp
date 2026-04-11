#pragma once

#include <cstddef>

#include "discretization.hpp"
#include "matrix.hpp"
#include "matrix/cholesky.hpp"
#include "ricatti.hpp"
#include "stability.hpp"
#include "state_space.hpp"

namespace wetmelon::control {
namespace online {

/**
 * @struct LQRResult
 * @brief Linear-Quadratic Regulator design result
 *
 * Mirrors MATLAB®'s [K,S,P] = lqr(...) output structure, containing optimal gain,
 * Riccati solution, and closed-loop pole information.
 */
template<size_t NX, size_t NU, typename T = double>
struct LQRResult {
    Matrix<NU, NX, T>           K{}; //!< Optimal gain: u = -K*x
    Matrix<NX, NX, T>           S{}; //!< Solution to Riccati equation
    ColVec<NX, wet::complex<T>> e{}; //!< Full complex closed-loop poles (eigenvalues)
    bool                        success{false};

    template<typename U>
    [[nodiscard]] consteval auto as() const {
        return LQRResult<NX, NU, U>{
            K.template as<U>(),
            S.template as<U>(),
            e.template as<wet::complex<U>>(),
            success
        };
    }

    /**
     * @brief Check if the closed-loop system is stable
     *
     * Stability is determined by checking if all closed-loop lie within the unit circle.
     *
     * @return true if stable, false otherwise
     */
    [[nodiscard]] constexpr bool is_stable() const {
        for (size_t i = 0; i < NX; ++i) {
            if (e[i].abs() >= T{1.0}) {
                return false;
            }
        }
        return true;
    }
};

/**
 * @brief Discrete-time Linear-Quadratic Regulator design (runtime version)
 *
 * @param A  State transition matrix
 * @param B  Control input matrix
 * @param Q  State cost matrix
 * @param R  Input cost matrix
 * @param N  (optional) Cross-term cost matrix
 *
 * @return LQRResult containing gain matrix and solution to DARE
 */
template<size_t NX, size_t NU, typename T = double>
[[nodiscard]] constexpr LQRResult<NX, NU, T> dlqr(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NX, NX, T>& Q,
    const Matrix<NU, NU, T>& R,
    const Matrix<NX, NU, T>& N = Matrix<NX, NU, T>{}
) {
    LQRResult<NX, NU, T> result{};

    const auto dare_opt = dare(A, B, Q, R, N);
    if (!dare_opt) {
        return result;
    }
    const Matrix<NX, NX, T> S = dare_opt.value();

    //! Solve (R + BᵀSB) K = BᵀSA + Nᵀ via Cholesky (R + BᵀSB is positive definite)
    const Matrix<NU, NU, T> denom = R + B.transpose() * S * B;
    const Matrix<NU, NX, T> rhs = B.transpose() * S * A + N.transpose();
    const auto              K_opt = mat::cholesky_solve(denom, rhs);

    if (!K_opt) {
        return result;
    }

    Matrix<NU, NX, T> K = K_opt.value();

    result = LQRResult<NX, NU, T>{K, S, stability::closed_loop_poles(A, B, K), true};
    return result;
}

/**
 * @brief Design discrete LQR from continuous-time system via discretization (runtime version)
 *
 * @param A   State transition matrix (continuous-time)
 * @param B   Control input matrix (continuous-time)
 * @param Q   State cost matrix
 * @param R   Input cost matrix
 * @param Ts  Sampling time
 * @param N   (optional) Cross-term cost matrix
 *
 * @return LQRResult containing gain matrix and solution to DARE
 */
template<size_t NX, size_t NU, typename T = double>
[[nodiscard]] constexpr LQRResult<NX, NU, T> lqrd(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NX, NX, T>& Q,
    const Matrix<NU, NU, T>& R,
    T                        Ts,
    const Matrix<NX, NU, T>& N = Matrix<NX, NU, T>{}
) {
    StateSpace<NX, NU, NX, NX, NX, T> sys_c{A, B, Matrix<NX, NX, T>::identity()};
    const auto                        sys_d = discretize(sys_c, Ts, DiscretizationMethod::ZOH);
    return dlqr(sys_d.A, sys_d.B, Q, R, N);
}

/**
 * @brief LQR from continuous state-space system via discretization (runtime version)
 *
 * @param sys State-space system (continuous-time)
 * @param Q   State cost matrix
 * @param R   Input cost matrix
 * @param Ts  Sampling time
 * @param N   (optional) Cross-term cost matrix
 *
 * @return LQRResult containing gain matrix and solution to DARE
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = 0, size_t NV = 0, typename T = double>
[[nodiscard]] constexpr LQRResult<NX, NU, T> lqrd(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NX, NX, T>&                 Q,
    const Matrix<NU, NU, T>&                 R,
    T                                        Ts,
    const Matrix<NX, NU, T>&                 N = Matrix<NX, NU, T>{}
) {
    return lqrd(sys.A, sys.B, Q, R, Ts, N);
}

} // namespace online

namespace design {

/**
 * @struct LQRResult
 * @brief Linear-Quadratic Regulator design result
 *
 * Mirrors MATLAB®'s [K,S,P] = lqr(...) output structure, containing optimal gain,
 * Riccati solution, and closed-loop pole information.
 */
template<size_t NX, size_t NU, typename T = double>
struct LQRResult {
    Matrix<NU, NX, T>           K{}; //!< Optimal gain: u = -K*x
    Matrix<NX, NX, T>           S{}; //!< Solution to Riccati equation
    ColVec<NX, wet::complex<T>> e{}; //!< Full complex closed-loop poles (eigenvalues)
    bool                        success{false};

    template<typename U>
    [[nodiscard]] consteval auto as() const {
        return LQRResult<NX, NU, U>{
            K.template as<U>(),
            S.template as<U>(),
            e.template as<wet::complex<U>>(),
            success
        };
    }

    /**
     * @brief Check if the closed-loop system is stable
     *
     * Stability is determined by checking if all closed-loop lie within the unit circle.
     *
     * @return true if stable, false otherwise
     */
    [[nodiscard]] constexpr bool is_stable() const {
        for (size_t i = 0; i < NX; ++i) {
            if (e[i].abs() >= T{1.0}) {
                return false;
            }
        }
        return true;
    }
};

/**
 * @brief Discrete Linear-Quadratic Regulator design
 *
 * Designs optimal gain for discrete system: x[k+1] = A*x[k] + B*u[k]
 *
 * @param A  State transition matrix
 * @param B  Control input matrix
 * @param Q  State cost matrix
 * @param R  Input cost matrix
 * @param N  (optional) Cross-term cost matrix
 *
 * @return LQRResult containing gain matrix and solution to DARE
 */
template<size_t NX, size_t NU, typename T = double>
[[nodiscard]] consteval LQRResult<NX, NU, T> dlqr(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NX, NX, T>& Q,
    const Matrix<NU, NU, T>& R,
    const Matrix<NX, NU, T>& N = Matrix<NX, NU, T>{}
) {
    LQRResult<NX, NU, T> result{};

    const auto dare_opt = dare(A, B, Q, R, N);
    if (!dare_opt) {
        return result;
    }

    const Matrix<NX, NX, T> S = dare_opt.value();
    const Matrix<NU, NU, T> denom = R + B.transpose() * S * B;
    const Matrix<NU, NX, T> rhs = B.transpose() * S * A + N.transpose();
    const auto              K_opt = mat::cholesky_solve(denom, rhs);

    if (!K_opt) {
        return result;
    }

    Matrix<NU, NX, T> K = K_opt.value();
    return LQRResult<NX, NU, T>{K, S, stability::closed_loop_poles(A, B, K), true};
}

/**
 * @brief Design discrete LQR from continuous-time system via discretization
 *
 * @param A   State transition matrix (continuous-time)
 * @param B   Control input matrix (continuous-time)
 * @param Q   State cost matrix
 * @param R   Input cost matrix
 * @param Ts  Sampling time
 * @param N   (optional) Cross-term cost matrix
 *
 * @return LQRResult containing gain matrix and solution to DARE
 */
template<size_t NX, size_t NU, typename T = double>
[[nodiscard]] consteval LQRResult<NX, NU, T> lqrd(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NX, NX, T>& Q,
    const Matrix<NU, NU, T>& R,
    T                        Ts,
    const Matrix<NX, NU, T>& N = Matrix<NX, NU, T>{}
) {
    // Create continuous state-space and discretize
    StateSpace sys_c{A, B, Matrix<NX, NX, T>::identity()};
    const auto sys_d = discretize(sys_c, Ts, DiscretizationMethod::ZOH);

    return dlqr(sys_d.A, sys_d.B, Q, R, N);
}

/**
 * @brief Design discrete LQR from continuous-time state-space system via discretization
 *
 * @param sys  State-space system (continuous-time)
 * @param Q    State cost matrix
 * @param R    Input cost matrix
 * @param Ts   Sampling time
 * @param N    (optional) Cross-term cost matrix
 *
 * @return LQRResult containing gain matrix and solution to DARE
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = 0, size_t NV = 0, typename T = double>
[[nodiscard]] consteval LQRResult<NX, NU, T> lqrd(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NX, NX, T>&                 Q,
    const Matrix<NU, NU, T>&                 R,
    T                                        Ts,
    const Matrix<NX, NU, T>&                 N = Matrix<NX, NU, T>{}
) {
    return lqrd(sys.A, sys.B, Q, R, Ts, N);
}

/**
 * @brief Alias for discrete-time LQR design (consteval version)
 *
 * @param A  State transition matrix
 * @param B  Control input matrix
 * @param Q  State cost matrix
 * @param R  Input cost matrix
 * @param N  (optional) Cross-term cost matrix
 *
 * @return LQRResult containing gain matrix and solution to DARE
 */
template<size_t NX, size_t NU, typename T = double>
[[nodiscard]] consteval LQRResult<NX, NU, T> discrete_lqr(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NX, NX, T>& Q,
    const Matrix<NU, NU, T>& R,
    const Matrix<NX, NU, T>& N = Matrix<NX, NU, T>{}
) {
    return dlqr(A, B, Q, R, N);
}

/**
 * @brief Alias for discrete LQR from continuous system via discretization (consteval version)
 *
 * @param A   State transition matrix (continuous-time)
 * @param B   Control input matrix (continuous-time)
 * @param Q   State cost matrix
 * @param R   Input cost matrix
 * @param Ts  Sampling time
 * @param N   (optional) Cross-term cost matrix
 *
 * @return LQRResult containing gain matrix and solution to DARE
 */
template<size_t NX, size_t NU, typename T = double>
[[nodiscard]] consteval LQRResult<NX, NU, T> discrete_lqr_from_continuous(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NX, NX, T>& Q,
    const Matrix<NU, NU, T>& R,
    T                        Ts,
    const Matrix<NX, NU, T>& N = Matrix<NX, NU, T>{}
) {
    return lqrd(A, B, Q, R, Ts, N);
}

/**
 * @brief Alias for discrete LQR from continuous state-space system via discretization (consteval version)
 *
 * @param sys  State-space system (continuous-time)
 * @param Q    State cost matrix
 * @param R    Input cost matrix
 * @param Ts   Sampling time
 * @param N    (optional) Cross-term cost matrix
 *
 * @return LQRResult containing gain matrix and solution to DARE
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = 0, size_t NV = 0, typename T = double>
[[nodiscard]] consteval LQRResult<NX, NU, T> discrete_lqr_from_continuous(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NX, NX, T>&                 Q,
    const Matrix<NU, NU, T>&                 R,
    T                                        Ts,
    const Matrix<NX, NU, T>&                 N = Matrix<NX, NU, T>{}
) {
    return lqrd(sys, Q, R, Ts, N);
}

} // namespace design

/**
 * @defgroup discrete_controllers Discrete-Time Controllers
 * @brief Ready-to-use discrete-time controllers for embedded systems (ISR/RTOS)
 *
 * Use design:: namespace functions to create these from continuous or discrete models.
 * Use online:: namespace for fast runtime linearization and system identification.
 */

/**
 * @ingroup discrete_controllers
 * @brief Discrete Linear-Quadratic Regulator (LQR)
 *
 * State-feedback controller u = -K*x that minimizes a quadratic cost function.
 * Designed for regulation (driving state to zero) or tracking (following state reference).
 *
 * @tparam NX Number of states
 * @tparam NU Number of control inputs
 * @tparam T  Scalar type (default: double)
 */
template<size_t NX, size_t NU, typename T = double>
struct LQR {
    Matrix<NU, NX, T> K{};

    constexpr LQR() = default;
    constexpr explicit LQR(const Matrix<NU, NX, T>& K_) : K(K_) {}

    // Compile-time only constructor for design:: results
    consteval LQR(const design::LQRResult<NX, NU, T>& result) : K(result.K) {}

    // Runtime constructor for online:: results
    constexpr LQR(const online::LQRResult<NX, NU, T>& result) : K(result.K) {}

    template<typename U>
    constexpr LQR(const LQR<NX, NU, U>& other) : K(other.getK()) {}

    /**
     * @brief Compute regulator control law
     *
     * Drives state to zero: u = -K*x
     *
     * @param x Current state vector
     * @return Control input vector u
     */
    [[nodiscard]] constexpr ColVec<NU, T> control(const ColVec<NX, T>& x) const {
        return ColVec<NU, T>(-K * x);
    }

    /**
     * @brief Compute servo control law
     *
     * Tracks state reference: u = -K*(x - x_ref)
     *
     * @param x     Current state vector
     * @param x_ref Reference state vector
     * @return Control input vector u
     */
    [[nodiscard]] constexpr ColVec<NU, T> control(const ColVec<NX, T>& x, const ColVec<NX, T>& x_ref) const {
        return ColVec<NU, T>(-K * (x - x_ref));
    }

    [[nodiscard]] constexpr const Matrix<NU, NX, T>& getK() const { return K; }
};

} // namespace wetmelon::control
