#pragma once

#include <cstddef>

#include "wet/analysis/riccati.hpp"
#include "wet/analysis/stability.hpp"
#include "wet/matrix/matrix.hpp"
#include "wet/systems/discretization.hpp"
#include "wet/systems/state_space.hpp"

namespace wet {

namespace design {

/**
 * @struct LQRResult
 * @brief Linear-Quadratic Regulator design result
 *
 * Contains the optimal gain K, Riccati solution S, and closed-loop poles.
 * Use .as<float>() to convert for embedded deployment.
 *
 * @note Mirrors MATLAB's [K,S,P] = dlqr(...) output structure.
 * @see "Optimal Control" (Anderson & Moore, 1990), §4.3
 */
template<size_t NX, size_t NU, typename T = double>
struct LQRResult {
    Matrix<NU, NX, T>           K{};            ///< Optimal gain: u = −Kx
    Matrix<NX, NX, T>           S{};            ///< DARE solution (positive semidefinite)
    ColVec<NX, wet::complex<T>> e{};            ///< Closed-loop poles (eigenvalues of A − BK)
    bool                        success{false}; ///< true if DARE converged

    template<typename U>
    [[nodiscard]] constexpr auto as() const {
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
 * @brief Discrete-time Linear-Quadratic Regulator design
 *
 * Computes the optimal state-feedback gain K that minimizes the cost function:
 *
 *     J = Σ [ xᵀQx + uᵀRu + 2xᵀNu ]
 *
 * subject to the discrete-time dynamics x[k+1] = Ax[k] + Bu[k].
 *
 * The gain is applied as u = -Kx. The solution is found via the Discrete
 * Algebraic Riccati Equation (DARE).
 *
 * @note Compare with MATLAB's dlqr(A, B, Q, R, N).
 *
 * @see dare() for the underlying Riccati solver
 * @see discrete_lqr_from_continuous() to design from a continuous-time system
 * @see "Optimal Control" (Anderson & Moore, 1990), Chapter 4
 *
 * @param A  State transition matrix (NX × NX)
 * @param B  Control input matrix (NX × NU)
 * @param Q  State cost matrix (NX × NX, positive semidefinite)
 * @param R  Input cost matrix (NU × NU, positive definite)
 * @param N  Cross-term cost matrix (NX × NU, default: zero)
 * @return LQRResult with gain K, Riccati solution S, and closed-loop poles
 */
template<size_t NX, size_t NU, typename T = double>
[[nodiscard]] constexpr LQRResult<NX, NU, T> discrete_lqr(
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
    const Matrix<NU, NU, T> denom = R + B.t() * S * B;
    const Matrix<NU, NX, T> rhs = B.t() * S * A + N.t();
    const auto              K_opt = mat::cholesky_solve(denom, rhs);

    if (!K_opt) {
        return result;
    }

    Matrix<NU, NX, T> K = K_opt.value();

    result = LQRResult<NX, NU, T>{K, S, stability::closed_loop_poles(A, B, K), true};
    return result;
}

/**
 * @brief Design discrete LQR from continuous-time system via discretization
 *
 * Discretizes the continuous-time system (A, B) using ZOH, then solves the
 * discrete LQR problem on the resulting system.
 *
 * @note Compare with MATLAB's lqrd(A, B, Q, R, Ts).
 *
 * @see discrete_lqr() for the discrete-time design
 * @see discretize() for the ZOH discretization step
 *
 * @param A   State transition matrix (continuous-time, NX × NX)
 * @param B   Control input matrix (continuous-time, NX × NU)
 * @param Q   State cost matrix (NX × NX, positive semidefinite)
 * @param R   Input cost matrix (NU × NU, positive definite)
 * @param Ts  Sampling time [s]
 * @param N   Cross-term cost matrix (NX × NU, default: zero)
 * @return LQRResult with gain K, Riccati solution S, and closed-loop poles
 */
template<size_t NX, size_t NU, typename T = double>
[[nodiscard]] constexpr LQRResult<NX, NU, T> discrete_lqr_from_continuous(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NX, NX, T>& Q,
    const Matrix<NU, NU, T>& R,
    T                        Ts,
    const Matrix<NX, NU, T>& N = Matrix<NX, NU, T>{}
) {
    StateSpace<NX, NU, NX, NX, NX, T> sys_c{A, B, Matrix<NX, NX, T>::identity()};
    const auto                        sys_d = discretize(sys_c, Ts, DiscretizationMethod::ZOH);
    return discrete_lqr(sys_d.A, sys_d.B, Q, R, N);
}

/**
 * @brief LQR from continuous state-space system via discretization
 *
 * @param sys State-space system (continuous-time)
 * @param Q   State cost matrix (NX × NX, positive semidefinite)
 * @param R   Input cost matrix (NU × NU, positive definite)
 * @param Ts  Sampling time [s]
 * @param N   Cross-term cost matrix (NX × NU, default: zero)
 * @return LQRResult with gain K, Riccati solution S, and closed-loop poles
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = 0, size_t NV = 0, typename T = double>
[[nodiscard]] constexpr LQRResult<NX, NU, T> discrete_lqr_from_continuous(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NX, NX, T>&                 Q,
    const Matrix<NU, NU, T>&                 R,
    T                                        Ts,
    const Matrix<NX, NU, T>&                 N = Matrix<NX, NU, T>{}
) {
    return discrete_lqr_from_continuous(sys.A, sys.B, Q, R, Ts, N);
}

} // namespace design
/**
 * @ingroup discrete_controllers
 * @brief Runtime Linear-Quadratic Regulator
 *
 * Lightweight state-feedback controller: u = −Kx (regulation) or u = −K(x − x_ref) (tracking).
 * Stores only the gain matrix K. One matrix-vector multiply per call — suitable for ISR.
 *
 * @see design::discrete_lqr() to compute K
 * @see "Optimal Control" (Anderson & Moore, 1990), §4.1
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

    constexpr LQR(const design::LQRResult<NX, NU, T>& result) : K(result.K) {} // NOLINT

    template<typename U>
    constexpr LQR(const LQR<NX, NU, U>& other) : K(other.getK()) {} // NOLINT

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

} // namespace wet
