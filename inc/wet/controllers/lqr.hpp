#pragma once

#include <cstddef>

#include "wet/design/riccati.hpp"
#include "wet/design/stability.hpp"
#include "wet/math/complex.hpp"
#include "wet/matrix/block.hpp"
#include "wet/matrix/functions.hpp"
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
     * Stability is determined by checking if all closed-loop poles lie within the unit circle.
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

    const auto K_opt = lqr_gain(A, B, S, R, N);
    if (!K_opt) {
        return result;
    }

    Matrix<NU, NX, T> K = K_opt.value();

    result = LQRResult<NX, NU, T>{K, S, stability::closed_loop_poles(A, B, K), true};
    return result;
}

/**
 * @struct LQRCost
 * @brief Discretized LQR cost weights (Q, R, N) for a sampled-data problem
 */
template<size_t NX, size_t NU, typename T = double>
struct LQRCost {
    Matrix<NX, NX, T> Q{}; ///< Discrete state cost
    Matrix<NU, NU, T> R{}; ///< Discrete input cost
    Matrix<NX, NU, T> N{}; ///< Discrete cross-term cost
};

/**
 * @brief Discretize a continuous LQR cost integral over one sample (Van Loan)
 *
 * Maps the continuous running cost
 * @f[
 *   J = \int_0^\infty \big( x^\top Q x + 2 x^\top N u + u^\top R u \big)\, dt
 * @f]
 * to its exact discrete equivalent @f$ \sum (x^\top Q_d x + 2 x^\top N_d u + u^\top R_d u) @f$
 * for a zero-order-hold input. Naively reusing the continuous @f$ (Q,R,N) @f$ on
 * the discretized dynamics is only first-order accurate in @f$ T_s @f$; this is exact.
 *
 * Augmenting the held input as constant states @f$ \bar A = [A\;B;\,0\;0] @f$ with
 * weight @f$ \bar Q = [Q\;N;\,N^\top\;R] @f$, the discrete weights follow from a
 * single matrix exponential (Van Loan, 1978):
 * @f[
 *   \exp\!\left( \begin{bmatrix} -\bar A^\top & \bar Q \\ 0 & \bar A \end{bmatrix} T_s \right)
 *     = \begin{bmatrix} M_{11} & M_{12} \\ 0 & M_{22} \end{bmatrix}, \quad
 *   \begin{bmatrix} Q_d & N_d \\ N_d^\top & R_d \end{bmatrix} = M_{22}^\top M_{12}.
 * @f]
 *
 * @note This is the cost-discretization step of MATLAB's lqrd(A, B, Q, R, Ts).
 * @see "Computing Integrals Involving the Matrix Exponential" (Van Loan, 1978),
 *      IEEE TAC 23(3), https://doi.org/10.1109/TAC.1978.1101743
 *
 * @param A   State transition matrix (continuous-time, NX × NX)
 * @param B   Control input matrix (continuous-time, NX × NU)
 * @param Q   Continuous state cost (NX × NX, positive semidefinite)
 * @param R   Continuous input cost (NU × NU, positive definite)
 * @param Ts  Sampling time [s]
 * @param N   Continuous cross-term cost (NX × NU, default: zero)
 * @return Discrete cost weights {Q_d, R_d, N_d}
 */
template<size_t NX, size_t NU, typename T = double>
[[nodiscard]] constexpr LQRCost<NX, NU, T> discretize_lqr_cost(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NX, NX, T>& Q,
    const Matrix<NU, NU, T>& R,
    T                        Ts,
    const Matrix<NX, NU, T>& N = Matrix<NX, NU, T>{}
) {
    constexpr size_t NA = NX + NU; // augmented (state + held input) dimension

    //! Augmented dynamics Ā = [A B; 0 0] and weight Q̄ = [Q N; Nᵀ R]
    const Matrix<NU, NX, T> Nt = N.t();
    Matrix<NA, NA, T>       Abar{};
    Abar.template block<NX, NX>(0, 0) = A;
    Abar.template block<NX, NU>(0, NX) = B;

    Matrix<NA, NA, T> Qbar{};
    Qbar.template block<NX, NX>(0, 0) = Q;
    Qbar.template block<NX, NU>(0, NX) = N;
    Qbar.template block<NU, NX>(NX, 0) = Nt;
    Qbar.template block<NU, NU>(NX, NX) = R;

    //! Van Loan block Z = [-Āᵀ Q̄; 0 Ā]·Ts, then exp(Z) = [M11 M12; 0 M22]
    const Matrix<NA, NA, T>   AbarT = Abar.t();
    const Matrix<NA, NA, T>   neg_AbarT = AbarT * (-Ts);
    const Matrix<NA, NA, T>   Qbar_Ts = Qbar * Ts;
    const Matrix<NA, NA, T>   Abar_Ts = Abar * Ts;
    Matrix<2 * NA, 2 * NA, T> Z{};
    Z.template block<NA, NA>(0, 0) = neg_AbarT;
    Z.template block<NA, NA>(0, NA) = Qbar_Ts;
    Z.template block<NA, NA>(NA, NA) = Abar_Ts;

    const Matrix<2 * NA, 2 * NA, T> G = mat::expm(Z);
    const Matrix<NA, NA, T>         M12 = G.template block<NA, NA>(0, NA);
    const Matrix<NA, NA, T>         M22 = G.template block<NA, NA>(NA, NA);
    const Matrix<NA, NA, T>         M22t = M22.t();

    //! Q̄_d = M22ᵀ·M12; symmetrize to scrub round-off asymmetry before slicing
    Matrix<NA, NA, T>       Qbar_d = M22t * M12;
    const Matrix<NA, NA, T> Qbar_dt = Qbar_d.t();
    Qbar_d = (Qbar_d + Qbar_dt) * T{0.5};

    return LQRCost<NX, NU, T>{
        Qbar_d.template block<NX, NX>(0, 0),
        Qbar_d.template block<NU, NU>(NX, NX),
        Qbar_d.template block<NX, NU>(0, NX)
    };
}

/**
 * @brief Design discrete LQR from continuous-time system via discretization
 *
 * Discretizes both the dynamics (ZOH) and the cost integral (Van Loan), then
 * solves the discrete LQR problem. Equivalent to MATLAB's lqrd — discretizing
 * the cost is what distinguishes this from feeding continuous Q, R, N into the
 * sampled dynamics, which is only first-order accurate in Ts.
 *
 * @note Compare with MATLAB's lqrd(A, B, Q, R, Ts).
 *
 * @see discrete_lqr() for the discrete-time design
 * @see discretize() for the ZOH dynamics discretization step
 * @see discretize_lqr_cost() for the Van Loan cost discretization step
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
    const auto                        cost = discretize_lqr_cost(A, B, Q, R, Ts, N);
    return discrete_lqr(sys_d.A, sys_d.B, cost.Q, cost.R, cost.N);
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

/**
 * @brief Continuous-time Linear-Quadratic Regulator design
 *
 * Computes the optimal state-feedback gain K that minimizes
 *
 *     J = ∫₀^∞ ( xᵀQx + uᵀRu + 2xᵀNu ) dt
 *
 * subject to the continuous-time dynamics ẋ = Ax + Bu, applied as u = −Kx.
 * Solves the Continuous Algebraic Riccati Equation (CARE) for the stabilizing
 * S, then forms the gain K = R⁻¹(BᵀS + Nᵀ).
 *
 * @note Compare with MATLAB's [K,S,e] = lqr(A, B, Q, R, N).
 * @note Unlike @ref discrete_lqr, the returned `e` are *continuous-time*
 *       (s-plane) closed-loop poles of (A − BK); stability is Re(e) < 0, so
 *       @ref LQRResult::is_stable (a unit-circle test) does not apply here.
 *
 * @see care() for the underlying Riccati solver
 * @see discrete_lqr_from_continuous() for the sampled-data (digital) design
 * @see "Optimal Control" (Anderson & Moore, 1990), §3.3
 *
 * @param A  State matrix (NX × NX)
 * @param B  Control input matrix (NX × NU)
 * @param Q  State cost matrix (NX × NX, positive semidefinite)
 * @param R  Input cost matrix (NU × NU, positive definite)
 * @param N  Cross-term cost matrix (NX × NU, default: zero)
 * @return LQRResult with gain K, CARE solution S, and continuous-time poles
 */
template<size_t NX, size_t NU, typename T = double>
[[nodiscard]] constexpr LQRResult<NX, NU, T> continuous_lqr(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NX, NX, T>& Q,
    const Matrix<NU, NU, T>& R,
    const Matrix<NX, NU, T>& N = Matrix<NX, NU, T>{}
) {
    LQRResult<NX, NU, T> result{};

    const auto care_opt = care(A, B, Q, R, N);
    if (!care_opt) {
        return result;
    }
    const Matrix<NX, NX, T> S = care_opt.value();

    // Continuous-time optimal gain K = R⁻¹(BᵀS + Nᵀ), via a decomposition solve
    // of R·K = BᵀS + Nᵀ rather than forming R⁻¹ explicitly (better conditioned
    // for NU > 1; matches care_schur's lu_solve usage).
    const auto K_opt = mat::lu_solve(R, (B.transpose() * S) + N.transpose());
    if (!K_opt) {
        return result;
    }
    const Matrix<NU, NX, T> K = K_opt.value();

    result = LQRResult<NX, NU, T>{K, S, stability::closed_loop_poles(A, B, K), true};
    return result;
}

/// @brief Continuous LQR from a continuous-time @ref StateSpace (uses A, B).
template<size_t NX, size_t NU, size_t NY, size_t NW = 0, size_t NV = 0, typename T = double>
[[nodiscard]] constexpr LQRResult<NX, NU, T> continuous_lqr(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NX, NX, T>&                 Q,
    const Matrix<NU, NU, T>&                 R,
    const Matrix<NX, NU, T>&                 N = Matrix<NX, NU, T>{}
) {
    return continuous_lqr(sys.A, sys.B, Q, R, N);
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
