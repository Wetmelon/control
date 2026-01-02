#pragma once

#include <cmath>
#include <complex>

#include "discretization.hpp"
#include "eigen.hpp"
#include "ricatti.hpp"
#include "state_space.hpp"

/**
 * @defgroup servo_design Servo Design Utilities
 * @brief Enumeration for servo controller degrees of freedom
 */

/**
 * @brief Servo design choices for tracking controllers
 *
 * Specifies whether to use 1-DOF (error feedback only) or 2-DOF (error feedback + feedforward)
 * control architecture.
 */
enum class ServoDOF {
    OneDOF, //!< 1-DOF: integrator on error only
    TwoDOF  //!< 2-DOF: integrator + feedforward
};

/**
 * @defgroup stability_analysis Stability Analysis
 * @brief Functions to analyze closed-loop stability of control systems
 *
 * For continuous systems: stable if all eigenvalues have Re(λ) < 0 (left half plane)
 * For discrete systems: stable if all eigenvalues have |λ| < 1 (inside unit circle)
 */

namespace stability {

/**
 * @brief Check if a continuous-time system matrix A is stable
 *
 * A continuous system is stable if all eigenvalues have negative real parts (LHP).
 *
 * @tparam N   Number of states (must be ≤ 4)
 * @tparam T   Numeric type (default: double)
 * @param A    State matrix to check
 *
 * @return true if all eigenvalues have Re(λ) < 0, false otherwise
 */
template<size_t N, typename T = double>
[[nodiscard]] constexpr bool is_stable_continuous(const Matrix<N, N, T>& A) {
    static_assert(N <= 4, "Stability analysis only supported for systems up to 4 states");
    auto eigen = compute_eigenvalues(A);
    if (!eigen.converged)
        return false;

    for (size_t i = 0; i < N; ++i) {
        if (eigen.values[i].real() >= T{0}) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Check if a discrete-time system matrix A is stable
 *
 * A discrete system is stable if all eigenvalues have magnitude less than 1 (inside unit circle).
 *
 * @tparam N   Number of states (must be ≤ 4)
 * @tparam T   Numeric type (default: double)
 * @param A    State matrix to check
 *
 * @return true if all eigenvalues satisfy |λ| < 1, false otherwise
 */
template<size_t N, typename T = double>
[[nodiscard]] constexpr bool is_stable_discrete(const Matrix<N, N, T>& A) {
    static_assert(N <= 4, "Stability analysis only supported for systems up to 4 states");
    auto eigen = compute_eigenvalues(A);
    if (!eigen.converged)
        return false;

    for (size_t i = 0; i < N; ++i) {
        T magnitude = wet::sqrt(
            eigen.values[i].real() * eigen.values[i].real() + eigen.values[i].imag() * eigen.values[i].imag()
        );
        if (magnitude >= T{1}) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Check closed-loop stability for continuous system with state feedback
 *
 * Checks stability of the closed-loop system A_cl = A - B*K with feedback u = -K*x.
 *
 * @tparam NX  Number of states
 * @tparam NU  Number of inputs
 * @tparam T   Numeric type (default: double)
 * @param A    State matrix
 * @param B    Control input matrix
 * @param K    State feedback gain matrix (u = -K*x)
 *
 * @return true if closed-loop system is stable, false otherwise
 */
template<size_t NX, size_t NU, typename T = double>
[[nodiscard]] constexpr bool is_closed_loop_stable_continuous(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NU, NX, T>& K
) {
    Matrix<NX, NX, T> A_cl = A - B * K;
    return is_stable_continuous(A_cl);
}

/**
 * @brief Check closed-loop stability for discrete system with state feedback
 *
 * Checks stability of the closed-loop system A_cl = A - B*K with feedback u = -K*x.
 *
 * @tparam NX  Number of states
 * @tparam NU  Number of inputs
 * @tparam T   Numeric type (default: double)
 * @param A    State matrix
 * @param B    Control input matrix
 * @param K    State feedback gain matrix (u = -K*x)
 *
 * @return true if closed-loop system is stable, false otherwise
 */
template<size_t NX, size_t NU, typename T = double>
[[nodiscard]] constexpr bool is_closed_loop_stable_discrete(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NU, NX, T>& K
) {
    Matrix<NX, NX, T> A_cl = A - B * K;
    return is_stable_discrete(A_cl);
}

/**
 * @brief Compute stability margin for continuous system
 *
 * Returns the distance to the stability boundary (imaginary axis).
 * Computed as the negative of the most positive real eigenvalue part.
 * Positive values indicate stability; larger values indicate more stability margin.
 *
 * @tparam N   Number of states (must be ≤ 4)
 * @tparam T   Numeric type (default: double)
 * @param A    State matrix
 *
 * @return Stability margin (positive = stable, negative = unstable)
 */
template<size_t N, typename T = double>
[[nodiscard]] constexpr T stability_margin_continuous(const Matrix<N, N, T>& A) {
    static_assert(N <= 4, "Stability margin only supported for systems up to 4 states");
    auto eigen = compute_eigenvalues(A);
    if (!eigen.converged)
        return T{1}; // Return unstable indicator

    T max_real = eigen.values[0].real();
    for (size_t i = 1; i < N; ++i) {
        if (eigen.values[i].real() > max_real) {
            max_real = eigen.values[i].real();
        }
    }
    return -max_real; // Positive means stable, larger is more stable
}

/**
 * @brief Compute stability margin for discrete system
 *
 * Returns the distance to the stability boundary (unit circle).
 * Computed as 1 - (maximum magnitude eigenvalue).
 * Positive values indicate stability; larger values indicate more stability margin.
 *
 * @tparam N   Number of states (must be ≤ 4)
 * @tparam T   Numeric type (default: double)
 * @param A    State matrix
 *
 * @return Stability margin (positive = stable, negative = unstable)
 */
template<size_t N, typename T = double>
[[nodiscard]] constexpr T stability_margin_discrete(const Matrix<N, N, T>& A) {
    static_assert(N <= 4, "Stability margin only supported for systems up to 4 states");
    auto eigen = compute_eigenvalues(A);
    if (!eigen.converged)
        return T{-1}; // Return unstable indicator

    T max_mag = T{0};
    for (size_t i = 0; i < N; ++i) {
        T magnitude = wet::sqrt(
            eigen.values[i].real() * eigen.values[i].real() + eigen.values[i].imag() * eigen.values[i].imag()
        );
        if (magnitude > max_mag) {
            max_mag = magnitude;
        }
    }
    return T{1} - max_mag; // Positive means stable, larger is more stable
}

/**
 * @brief Compute closed-loop poles (eigenvalues) with state feedback
 *
 * Computes the eigenvalues of the closed-loop state matrix A_cl = A - B*K.
 * These poles determine the closed-loop system dynamics.
 *
 * @tparam NX  Number of states (must be ≤ 4)
 * @tparam NU  Number of inputs
 * @tparam T   Numeric type (default: double)
 * @param A    State matrix
 * @param B    Control input matrix
 * @param K    State feedback gain matrix (u = -K*x)
 *
 * @return Vector of closed-loop pole locations (eigenvalues as complex numbers)
 */
template<size_t NX, size_t NU, typename T = double>
[[nodiscard]] constexpr ColVec<NX, std::complex<T>> closed_loop_poles(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NU, NX, T>& K
) {
    static_assert(NX <= 4, "Pole computation only supported for systems up to 4 states");
    Matrix<NX, NX, T> A_cl = A - B * K;
    auto              eigen = compute_eigenvalues(A_cl);
    return eigen.values;
}

} // namespace stability

namespace design {

/**
 * @defgroup control_design Control Design Algorithms
 * @brief MATLAB-style API functions for LQR, LQI, LQG, and Kalman filter design
 *
 * These functions mirror MATLAB's Control System Toolbox API for familiarity.
 * Both design:: (consteval) and online:: (constexpr) variants are provided.
 */
/**
 * @struct LQRResult
 * @brief Linear-Quadratic Regulator design result
 *
 * Mirrors MATLAB's [K,S,P] = lqr(...) output structure, containing optimal gain,
 * Riccati solution, and closed-loop pole information.
 */
template<size_t NX, size_t NU, typename T = double>
struct LQRResult {
    Matrix<NU, NX, T>           K{};                    //!< Optimal gain: u = -K*x
    Matrix<NX, NX, T>           S{};                    //!< Solution to Riccati equation
    ColVec<NX, T>               poles{};                //!< Closed-loop poles (real parts)
    ColVec<NX, std::complex<T>> poles_complex{};        //!< Full complex closed-loop poles
    bool                        is_stable{false};       //!< Closed-loop stability flag
    T                           stability_margin{T{0}}; //!< Distance to instability

    constexpr LQRResult() = default;

    constexpr LQRResult(
        const Matrix<NU, NX, T>& K_,
        const Matrix<NX, NX, T>& S_,
        const ColVec<NX, T>&     poles_
    ) : K(K_), S(S_), poles(poles_) {}

    // Full constructor with stability info
    constexpr LQRResult(
        const Matrix<NU, NX, T>&           K_,
        const Matrix<NX, NX, T>&           S_,
        const ColVec<NX, T>&               poles_,
        const ColVec<NX, std::complex<T>>& poles_complex_,
        bool                               is_stable_,
        T                                  stability_margin_
    ) : K(K_), S(S_), poles(poles_), poles_complex(poles_complex_), is_stable(is_stable_), stability_margin(stability_margin_) {}

    template<typename U>
    constexpr LQRResult(const LQRResult<NX, NU, U>& other)
        : K(other.K), S(other.S), poles(other.poles), poles_complex(other.poles_complex), is_stable(other.is_stable), stability_margin(static_cast<T>(other.stability_margin)) {}
};

/**
 * @struct LQIResult
 * @brief Linear-Quadratic Integral controller design result
 *
 * Contains gains and Riccati solution for servo control with integral action.
 */
template<size_t NX, size_t NU, size_t NY, typename T = double>
struct LQIResult {
    Matrix<NX, NX, T>           A{};                   //!< System A matrix
    Matrix<NX, NU, T>           B{};                   //!< System B matrix
    Matrix<NY, NX, T>           C{};                   //!< System C matrix
    Matrix<NU, NX, T>           Kx{};                  //!< State gain
    Matrix<NU, NY, T>           Ki{};                  //!< Integral gain
    Matrix<NX + NY, NX + NY, T> S{};                   //!< Riccati solution for augmented system
    ServoDOF                    dof{ServoDOF::OneDOF}; //!< Servo configuration (1DOF or 2DOF)

    constexpr LQIResult() = default;
    constexpr LQIResult(
        const Matrix<NX, NX, T>&           A_,
        const Matrix<NX, NU, T>&           B_,
        const Matrix<NY, NX, T>&           C_,
        const Matrix<NU, NX, T>&           Kx_,
        const Matrix<NU, NY, T>&           Ki_,
        const Matrix<NX + NY, NX + NY, T>& S_,
        ServoDOF                           dof_ = ServoDOF::OneDOF
    ) : A(A_), B(B_), C(C_), Kx(Kx_), Ki(Ki_), S(S_), dof(dof_) {}

    template<typename U>
    constexpr LQIResult(const LQIResult<NX, NU, NY, U>& other)
        : A(other.A), B(other.B), C(other.C), Kx(other.Kx), Ki(other.Ki), S(other.S), dof(other.dof) {}
};

/**
 * @struct KalmanResult
 * @brief Kalman filter design result
 *
 * Contains filter gains and covariance matrices for optimal state estimation.
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
struct KalmanResult {
    StateSpace<NX, NU, NY, NW, NV, T> sys{}; //!< System model
    Matrix<NW, NW, T>                 Q{};   //!< Process noise covariance
    Matrix<NV, NV, T>                 R{};   //!< Measurement noise covariance
    Matrix<NX, NY, T>                 L{};   //!< Kalman gain (steady-state)
    Matrix<NX, NX, T>                 P{};   //!< Error covariance (steady-state)

    constexpr KalmanResult() = default;
    constexpr KalmanResult(
        const StateSpace<NX, NU, NY, NW, NV, T>& sys_,
        const Matrix<NW, NW, T>&                 Q_,
        const Matrix<NV, NV, T>&                 R_,
        const Matrix<NX, NY, T>&                 L_ = Matrix<NX, NY, T>{},
        const Matrix<NX, NX, T>&                 P_ = Matrix<NX, NX, T>{}
    ) : sys(sys_), Q(Q_), R(R_), L(L_), P(P_) {}

    template<typename U>
    constexpr KalmanResult(const KalmanResult<NX, NU, NY, NW, NV, U>& other)
        : sys(other.sys), Q(other.Q), R(other.R), L(other.L), P(other.P) {}
};

/**
 * @struct LQGResult
 * @brief Linear-Quadratic-Gaussian controller design result
 *
 * Combines LQR and Kalman filter designs for separation principle-based control.
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
struct LQGResult {
    LQRResult<NX, NU, T>                lqr{};    //!< LQR design result
    KalmanResult<NX, NU, NY, NW, NV, T> kalman{}; //!< Kalman filter result

    constexpr LQGResult() = default;
    constexpr LQGResult(
        const LQRResult<NX, NU, T>&                lqr_,
        const KalmanResult<NX, NU, NY, NW, NV, T>& kalman_
    ) : lqr(lqr_), kalman(kalman_) {}

    template<typename U>
    constexpr LQGResult(const LQGResult<NX, NU, NY, NW, NV, U>& other)
        : lqr(other.lqr), kalman(other.kalman) {}
};

/**
 * @struct LQGIResult
 * @brief Linear-Quadratic-Gaussian Integral controller design result
 *
 * Combines LQI and Kalman filter designs for servo control with state estimation.
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
struct LQGIResult {
    LQIResult<NX, NU, NY, T>            lqi{};    //!< LQI design result
    KalmanResult<NX, NU, NY, NW, NV, T> kalman{}; //!< Kalman filter result

    constexpr LQGIResult() = default;
    constexpr LQGIResult(
        const LQIResult<NX, NU, NY, T>&            lqi_,
        const KalmanResult<NX, NU, NY, NW, NV, T>& kalman_
    ) : lqi(lqi_), kalman(kalman_) {}

    template<typename U>
    constexpr LQGIResult(const LQGIResult<NX, NU, NY, NW, NV, U>& other)
        : lqi(other.lqi), kalman(other.kalman) {}
};

/**
 * @defgroup lqr_helpers LQR Result Helper Functions
 * @brief Internal functions to construct LQRResult with stability analysis
 */
namespace detail {

/**
 * @brief Create LQRResult for continuous-time system with stability analysis
 *
 * @tparam NX  Number of states
 * @tparam NU  Number of inputs
 * @tparam T   Numeric type (default: double)
 * @param A    State matrix
 * @param B    Control input matrix
 * @param K    Optimal gain matrix
 * @param S    Riccati equation solution
 *
 * @return LQRResult with computed poles, stability flag, and stability margin
 */
template<size_t NX, size_t NU, typename T = double>
[[nodiscard]] constexpr LQRResult<NX, NU, T> make_lqr_result_continuous(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NU, NX, T>& K,
    const Matrix<NX, NX, T>& S
) {
    // Compute closed-loop matrix
    Matrix<NX, NX, T> A_cl = A - B * K;

    // Get full eigenvalues
    auto poles_complex = stability::closed_loop_poles(A, B, K);

    // Extract real parts for backward compatibility
    ColVec<NX, T> poles{};
    for (size_t i = 0; i < NX; ++i) {
        poles[i] = poles_complex[i].real();
    }

    // Check stability (continuous: Re(λ) < 0)
    bool is_stable = stability::is_stable_continuous(A_cl);

    // Compute stability margin
    T margin = stability::stability_margin_continuous(A_cl);

    return LQRResult<NX, NU, T>{K, S, poles, poles_complex, is_stable, margin};
}

/**
 * @brief Create LQRResult for discrete-time system with stability analysis
 *
 * @tparam NX  Number of states
 * @tparam NU  Number of inputs
 * @tparam T   Numeric type (default: double)
 * @param A    State matrix
 * @param B    Control input matrix
 * @param K    Optimal gain matrix
 * @param S    Riccati equation solution
 *
 * @return LQRResult with computed poles, stability flag, and stability margin
 */
template<size_t NX, size_t NU, typename T = double>
[[nodiscard]] constexpr LQRResult<NX, NU, T> make_lqr_result_discrete(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NU, NX, T>& K,
    const Matrix<NX, NX, T>& S
) {
    // Compute closed-loop matrix
    Matrix<NX, NX, T> A_cl = A - B * K;

    // Get full eigenvalues
    auto poles_complex = stability::closed_loop_poles(A, B, K);

    // Extract real parts for backward compatibility
    ColVec<NX, T> poles{};
    for (size_t i = 0; i < NX; ++i) {
        poles[i] = poles_complex[i].real();
    }

    // Check stability (discrete: |λ| < 1)
    bool is_stable = stability::is_stable_discrete(A_cl);

    // Compute stability margin
    T margin = stability::stability_margin_discrete(A_cl);

    return LQRResult<NX, NU, T>{K, S, poles, poles_complex, is_stable, margin};
}

} // namespace detail

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
    const Matrix<NX, NX, T> S = dare(A, B, Q, R);

    // K = (R + B'*S*B)^{-1} * (B'*S*A + N')
    const Matrix<NU, NU, T> denom = R + B.transpose() * S * B;
    const auto              denom_inv = denom.inverse();

    Matrix<NU, NX, T> K{};
    if (denom_inv) {
        K = denom_inv.value() * (B.transpose() * S * A + N.transpose());
    }

    return detail::make_lqr_result_discrete(A, B, K, S);
}

/**
 * @brief Linear-Quadratic Regulator design for continuous-time systems
 *
 * Designs optimal gain for continuous system: dx/dt = A*x + B*u
 *
 * @param A  State transition matrix
 * @param B  Control input matrix
 * @param Q  State cost matrix
 * @param R  Input cost matrix
 * @param N  (optional) Cross-term cost matrix
 *
 * @return LQRResult containing gain matrix and solution to CARE
 */
template<size_t NX, size_t NU, typename T = double>
[[nodiscard]] consteval LQRResult<NX, NU, T> lqr(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NX, NX, T>& Q,
    const Matrix<NU, NU, T>& R,
    const Matrix<NX, NU, T>& N = Matrix<NX, NU, T>{}
) {
    // For continuous LQR with cross-term N, transform to standard form:
    // Q_bar = Q - N*R^{-1}*N', then solve CARE with Q_bar
    const auto R_inv = R.inverse();
    if (!R_inv) {
        return LQRResult<NX, NU, T>{};
    }

    Matrix<NX, NX, T> Q_bar = Q;
    if constexpr (NX > 0 && NU > 0) {
        // Only apply cross-term correction if N is non-zero
        Q_bar = Q - N * R_inv.value() * N.transpose();
    }

    // Use CARE to solve for S
    const Matrix<NX, NX, T> S = care(A, B, Q_bar, R);

    // K = R^{-1} * (B'*S + N')
    const Matrix<NU, NX, T> K = R_inv.value() * (B.transpose() * S + N.transpose());

    return detail::make_lqr_result_continuous(A, B, K, S);
}

/**
 * @brief Linear-Quadratic Regulator design for continuous-time state-space systems
 *
 * @param sys  State-space system
 * @param Q    State cost matrix
 * @param R    Input cost matrix
 * @param N    (optional) Cross-term cost matrix
 *
 * @return LQRResult containing gain matrix and solution to CARE
 */
template<size_t NX, size_t NU, size_t NY, typename T = double>
[[nodiscard]] consteval LQRResult<NX, NU, T> lqr(
    const StateSpace<NX, NU, NY, NX, NY, T>& sys,
    const Matrix<NX, NX, T>&                 Q,
    const Matrix<NU, NU, T>&                 R,
    const Matrix<NX, NU, T>&                 N = Matrix<NX, NU, T>{}
) {
    return lqr(sys.A, sys.B, Q, R, N);
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
template<size_t NX, size_t NU, size_t NY, typename T = double>
[[nodiscard]] consteval LQRResult<NX, NU, T> lqrd(
    const StateSpace<NX, NU, NY, NX, NY, T>& sys,
    const Matrix<NX, NX, T>&                 Q,
    const Matrix<NU, NU, T>&                 R,
    T                                        Ts,
    const Matrix<NX, NU, T>&                 N = Matrix<NX, NU, T>{}
) {
    return lqrd(sys.A, sys.B, Q, R, Ts, N);
}

/**
 * @brief Linear-Quadratic Integral design for tracking with servo action
 *
 * @param sys  State-space system
 * @param Q    Augmented state cost matrix (state + integral error)
 * @param R    Input cost matrix
 * @param dof  Servo degrees of freedom (1DOF or 2DOF)
 *
 * @return LQIResult containing state and integral gains
 */
template<size_t NX, size_t NU, size_t NY, typename T = double>
[[nodiscard]] consteval LQIResult<NX, NU, NY, T> lqi(
    const StateSpace<NX, NU, NY, NX, NY, T>& sys,
    const Matrix<NX + NY, NX + NY, T>&       Q,
    const Matrix<NU, NU, T>&                 R,
    ServoDOF                                 dof = ServoDOF::OneDOF
) {
    // Build augmented system: [x; xi] where xi integrates (y - r)
    // A_aug = [A 0; C I], B_aug = [B; 0]
    Matrix<NX + NY, NX + NY, T> A_aug{};
    Matrix<NX + NY, NU, T>      B_aug{};

    // Fill A_aug = [A 0; C I]
    for (size_t i = 0; i < NX; ++i) {
        for (size_t j = 0; j < NX; ++j) {
            A_aug(i, j) = sys.A(i, j);
        }
    }
    for (size_t i = 0; i < NY; ++i) {
        for (size_t j = 0; j < NX; ++j) {
            A_aug(NX + i, j) = sys.C(i, j);
        }
        A_aug(NX + i, NX + i) = T{1};
    }

    // Fill B_aug = [B; 0]
    for (size_t i = 0; i < NX; ++i) {
        for (size_t j = 0; j < NU; ++j) {
            B_aug(i, j) = sys.B(i, j);
        }
    }

    // Solve DARE for augmented system
    Matrix<NX + NY, NX + NY, T> P_aug = dare(A_aug, B_aug, Q, R);

    // Compute augmented gain: K_aug = (R + B'PB)^{-1} * B'PA
    const Matrix<NU, NU, T> S = R + B_aug.transpose() * P_aug * B_aug;
    const auto              S_inv = S.inverse();

    Matrix<NU, NX + NY, T> K_aug{};
    if (S_inv) {
        K_aug = S_inv.value() * B_aug.transpose() * P_aug * A_aug;
    }

    // Extract Kx and Ki from K_aug
    Matrix<NU, NX, T> Kx{};
    Matrix<NU, NY, T> Ki{};
    for (size_t i = 0; i < NU; ++i) {
        for (size_t j = 0; j < NX; ++j) {
            Kx(i, j) = K_aug(i, j);
        }
        for (size_t j = 0; j < NY; ++j) {
            Ki(i, j) = K_aug(i, NX + j);
        }
    }

    return LQIResult<NX, NU, NY, T>(sys.A, sys.B, sys.C, Kx, Ki, P_aug, dof);
}

/**
 * @brief Linear-Quadratic-Gaussian regulator design combining LQR and Kalman filter
 *
 * @param sys     State-space system
 * @param Q_lqr   State cost for LQR
 * @param R_lqr   Input cost for LQR
 * @param Q_kf    Process noise covariance for Kalman filter
 * @param R_kf    Measurement noise covariance for Kalman filter
 * @param N       (optional) Cross-term cost matrix
 *
 * @return LQGResult combining LQR and Kalman filter designs
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
[[nodiscard]] consteval LQGResult<NX, NU, NY, NW, NV, T> lqg(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NX, NX, T>&                 Q_lqr, // State cost
    const Matrix<NU, NU, T>&                 R_lqr, // Input cost
    const Matrix<NW, NW, T>&                 Q_kf,  // Process noise covariance
    const Matrix<NV, NV, T>&                 R_kf,  // Measurement noise covariance
    const Matrix<NX, NU, T>&                 N = Matrix<NX, NU, T>{}
) {
    // Solve DARE for LQR gain
    const Matrix<NX, NX, T> P = dare(sys.A, sys.B, Q_lqr, R_lqr);

    // Compute LQR gain: K = (R + B'PB)^{-1} * (B'PA + N')
    const Matrix<NU, NU, T> S = R_lqr + sys.B.transpose() * P * sys.B;
    const auto              S_inv = S.inverse();

    Matrix<NU, NX, T> K{};
    if (S_inv) {
        K = S_inv.value() * (sys.B.transpose() * P * sys.A + N.transpose());
    }

    // Create LQR result
    LQRResult<NX, NU, T> lqr_result{K, P, ColVec<NX, T>{}};

    // Create Kalman result
    KalmanResult<NX, NU, NY, NW, NV, T> kalman_result{sys, Q_kf, R_kf};

    return LQGResult<NX, NU, NY, NW, NV, T>(lqr_result, kalman_result);
}

/**
 * @brief Linear-Quadratic-Gaussian with integral action for tracking
 *
 * @param sys      State-space system
 * @param Q_aug    Augmented state cost (state + integral error)
 * @param R        Input cost matrix
 * @param Q_kf     Process noise covariance for Kalman filter
 * @param R_kf     Measurement noise covariance for Kalman filter
 * @param dof      Servo degrees of freedom (1DOF or 2DOF)
 *
 * @return LQGIResult with integral action for tracking
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
[[nodiscard]] consteval LQGIResult<NX, NU, NY, NW, NV, T> lqgtrack(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NX + NY, NX + NY, T>&       Q_aug, // Augmented state cost (state + integral)
    const Matrix<NU, NU, T>&                 R,     // Input cost
    const Matrix<NW, NW, T>&                 Q_kf,  // Process noise covariance
    const Matrix<NV, NV, T>&                 R_kf,  // Measurement noise covariance
    ServoDOF                                 dof = ServoDOF::TwoDOF
) {
    // Build augmented system for LQI
    Matrix<NX + NY, NX + NY, T> A_aug{};
    Matrix<NX + NY, NU, T>      B_aug{};

    for (size_t i = 0; i < NX; ++i) {
        for (size_t j = 0; j < NX; ++j) {
            A_aug(i, j) = sys.A(i, j);
        }
    }
    for (size_t i = 0; i < NY; ++i) {
        for (size_t j = 0; j < NX; ++j) {
            A_aug(NX + i, j) = sys.C(i, j);
        }
        A_aug(NX + i, NX + i) = T{1};
    }
    for (size_t i = 0; i < NX; ++i) {
        for (size_t j = 0; j < NU; ++j) {
            B_aug(i, j) = sys.B(i, j);
        }
    }

    // Solve DARE for augmented system
    Matrix<NX + NY, NX + NY, T> P_aug = dare(A_aug, B_aug, Q_aug, R);

    // Compute augmented gain
    const Matrix<NU, NU, T> S = R + B_aug.transpose() * P_aug * B_aug;
    const auto              S_inv = S.inverse();

    Matrix<NU, NX + NY, T> K_aug{};
    if (S_inv) {
        K_aug = S_inv.value() * B_aug.transpose() * P_aug * A_aug;
    }

    // Extract Kx and Ki
    Matrix<NU, NX, T> Kx{};
    Matrix<NU, NY, T> Ki{};
    for (size_t i = 0; i < NU; ++i) {
        for (size_t j = 0; j < NX; ++j) {
            Kx(i, j) = K_aug(i, j);
        }
        for (size_t j = 0; j < NY; ++j) {
            Ki(i, j) = K_aug(i, NX + j);
        }
    }

    // Create LQI result
    LQIResult<NX, NU, NY, T> lqi_result(sys.A, sys.B, sys.C, Kx, Ki, P_aug, dof);

    // Create Kalman result
    KalmanResult<NX, NU, NY, NW, NV, T> kalman_result{sys, Q_kf, R_kf};

    return LQGIResult<NX, NU, NY, NW, NV, T>(lqi_result, kalman_result);
}

/**
 * @brief Combine separate Kalman filter and LQR designs into an LQG controller
 *
 * @param kest        Kalman filter design result
 * @param lqr_result  LQR controller design result
 *
 * @return LQGResult combining the provided Kalman and LQR designs
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
[[nodiscard]] consteval LQGResult<NX, NU, NY, NW, NV, T> lqgreg(
    const KalmanResult<NX, NU, NY, NW, NV, T>& kest,
    const LQRResult<NX, NU, T>&                lqr_result
) {
    return LQGResult<NX, NU, NY, NW, NV, T>(lqr_result, kest);
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
 * @brief Alias for continuous-time LQR design (consteval version)
 *
 * @param A  State transition matrix
 * @param B  Control input matrix
 * @param Q  State cost matrix
 * @param R  Input cost matrix
 * @param N  (optional) Cross-term cost matrix
 *
 * @return LQRResult containing gain matrix and solution to CARE
 */
template<size_t NX, size_t NU, typename T = double>
[[nodiscard]] consteval LQRResult<NX, NU, T> continuous_lqr(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NX, NX, T>& Q,
    const Matrix<NU, NU, T>& R,
    const Matrix<NX, NU, T>& N = Matrix<NX, NU, T>{}
) {
    return lqr(A, B, Q, R, N);
}

/**
 * @brief Alias for continuous-time LQR design for state-space systems (consteval version)
 *
 * @param sys  State-space system
 * @param Q    State cost matrix
 * @param R    Input cost matrix
 * @param N    (optional) Cross-term cost matrix
 *
 * @return LQRResult containing gain matrix and solution to CARE
 */
template<size_t NX, size_t NU, size_t NY, typename T = double>
[[nodiscard]] consteval LQRResult<NX, NU, T> continuous_lqr(
    const StateSpace<NX, NU, NY, NX, NY, T>& sys,
    const Matrix<NX, NX, T>&                 Q,
    const Matrix<NU, NU, T>&                 R,
    const Matrix<NX, NU, T>&                 N = Matrix<NX, NU, T>{}
) {
    return lqr(sys, Q, R, N);
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
template<size_t NX, size_t NU, size_t NY, typename T = double>
[[nodiscard]] consteval LQRResult<NX, NU, T> discrete_lqr_from_continuous(
    const StateSpace<NX, NU, NY, NX, NY, T>& sys,
    const Matrix<NX, NX, T>&                 Q,
    const Matrix<NU, NU, T>&                 R,
    T                                        Ts,
    const Matrix<NX, NU, T>&                 N = Matrix<NX, NU, T>{}
) {
    return lqrd(sys, Q, R, Ts, N);
}

/**
 * @brief Alias for LQR with integral action for tracking (consteval version)
 *
 * @param sys  State-space system
 * @param Q    Augmented state cost matrix (state + integral error)
 * @param R    Input cost matrix
 * @param dof  Servo degrees of freedom (1DOF or 2DOF)
 *
 * @return LQIResult containing state and integral gains
 */
template<size_t NX, size_t NU, size_t NY, typename T = double>
[[nodiscard]] consteval LQIResult<NX, NU, NY, T> lqr_with_integral(
    const StateSpace<NX, NU, NY, NX, NY, T>& sys,
    const Matrix<NX + NY, NX + NY, T>&       Q,
    const Matrix<NU, NU, T>&                 R,
    ServoDOF                                 dof = ServoDOF::OneDOF
) {
    return lqi(sys, Q, R, dof);
}

/**
 * @brief Alias for LQG regulator design combining LQR and Kalman filter (consteval version)
 *
 * @param sys     State-space system
 * @param Q_lqr   State cost for LQR
 * @param R_lqr   Input cost for LQR
 * @param Q_kf    Process noise covariance for Kalman filter
 * @param R_kf    Measurement noise covariance for Kalman filter
 * @param N       (optional) Cross-term cost matrix
 *
 * @return LQGResult combining LQR and Kalman filter designs
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
[[nodiscard]] consteval LQGResult<NX, NU, NY, NW, NV, T> lqg_regulator(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NX, NX, T>&                 Q_lqr,
    const Matrix<NU, NU, T>&                 R_lqr,
    const Matrix<NW, NW, T>&                 Q_kf,
    const Matrix<NV, NV, T>&                 R_kf,
    const Matrix<NX, NU, T>&                 N = Matrix<NX, NU, T>{}
) {
    return lqg(sys, Q_lqr, R_lqr, Q_kf, R_kf, N);
}

/**
 * @brief Alias for LQG with integral action for servo tracking (consteval version)
 *
 * @param sys      State-space system
 * @param Q_aug    Augmented state cost (state + integral error)
 * @param R        Input cost matrix
 * @param Q_kf     Process noise covariance for Kalman filter
 * @param R_kf     Measurement noise covariance for Kalman filter
 * @param dof      Servo degrees of freedom (1DOF or 2DOF)
 *
 * @return LQGIResult with integral action for tracking
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
[[nodiscard]] consteval LQGIResult<NX, NU, NY, NW, NV, T> lqg_servo(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NX + NY, NX + NY, T>&       Q_aug,
    const Matrix<NU, NU, T>&                 R,
    const Matrix<NW, NW, T>&                 Q_kf,
    const Matrix<NV, NV, T>&                 R_kf,
    ServoDOF                                 dof = ServoDOF::TwoDOF
) {
    return lqgtrack(sys, Q_aug, R, Q_kf, R_kf, dof);
}

/**
 * @brief Combine separate Kalman and LQR results into an LQG controller
 *
 * @param kest        Kalman filter result
 * @param lqr_result  LQR controller result
 *
 * @return LQG controller result structure
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
[[nodiscard]] consteval LQGResult<NX, NU, NY, NW, NV, T> lqg_from_parts(
    const KalmanResult<NX, NU, NY, NW, NV, T>& kest,
    const LQRResult<NX, NU, T>&                lqr_result
) {
    return lqgreg(kest, lqr_result);
}

} // namespace design

// ============================================================================
// online:: Namespace - Runtime versions of MATLAB-style functions
// ============================================================================

namespace online {

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
[[nodiscard]] constexpr design::LQRResult<NX, NU, T> dlqr(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NX, NX, T>& Q,
    const Matrix<NU, NU, T>& R,
    const Matrix<NX, NU, T>& N = Matrix<NX, NU, T>{}
) {
    const Matrix<NX, NX, T> S = dare(A, B, Q, R);

    const Matrix<NU, NU, T> denom = R + B.transpose() * S * B;
    const auto              denom_inv = denom.inverse();

    Matrix<NU, NX, T> K{};
    if (denom_inv) {
        K = denom_inv.value() * (B.transpose() * S * A + N.transpose());
    }

    return design::detail::make_lqr_result_discrete(A, B, K, S);
}

/**
 * @brief Continuous-time Linear-Quadratic Regulator design (runtime version)
 *
 * @param A  State transition matrix
 * @param B  Control input matrix
 * @param Q  State cost matrix
 * @param R  Input cost matrix
 * @param N  (optional) Cross-term cost matrix
 *
 * @return LQRResult containing gain matrix and solution to CARE
 */
template<size_t NX, size_t NU, typename T = double>
[[nodiscard]] constexpr design::LQRResult<NX, NU, T> lqr(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NX, NX, T>& Q,
    const Matrix<NU, NU, T>& R,
    const Matrix<NX, NU, T>& N = Matrix<NX, NU, T>{}
) {
    const auto R_inv = R.inverse();
    if (!R_inv) {
        return design::LQRResult<NX, NU, T>{};
    }

    Matrix<NX, NX, T> Q_bar = Q;
    if constexpr (NX > 0 && NU > 0) {
        Q_bar = Q - N * R_inv.value() * N.transpose();
    }

    const Matrix<NX, NX, T> S = care(A, B, Q_bar, R);

    const Matrix<NU, NX, T> K = R_inv.value() * (B.transpose() * S + N.transpose());

    return design::detail::make_lqr_result_continuous(A, B, K, S);
}

// lqr overload for StateSpace
template<size_t NX, size_t NU, size_t NY, typename T = double>
[[nodiscard]] constexpr design::LQRResult<NX, NU, T> lqr(
    const StateSpace<NX, NU, NY, NX, NY, T>& sys,
    const Matrix<NX, NX, T>&                 Q,
    const Matrix<NU, NU, T>&                 R,
    const Matrix<NX, NU, T>&                 N = Matrix<NX, NU, T>{}
) {
    return online::lqr(sys.A, sys.B, Q, R, N);
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
[[nodiscard]] constexpr design::LQRResult<NX, NU, T> lqrd(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NX, NX, T>& Q,
    const Matrix<NU, NU, T>& R,
    T                        Ts,
    const Matrix<NX, NU, T>& N = Matrix<NX, NU, T>{}
) {
    StateSpace<NX, NU, NX, NX, NX, T> sys_c{A, B, Matrix<NX, NX, T>::identity()};
    const auto                        sys_d = discretize(sys_c, Ts, DiscretizationMethod::ZOH);
    return online::dlqr(sys_d.A, sys_d.B, Q, R, N);
}

// lqrd overload for StateSpace
template<size_t NX, size_t NU, size_t NY, typename T = double>
[[nodiscard]] constexpr design::LQRResult<NX, NU, T> lqrd(
    const StateSpace<NX, NU, NY, NX, NY, T>& sys,
    const Matrix<NX, NX, T>&                 Q,
    const Matrix<NU, NU, T>&                 R,
    T                                        Ts,
    const Matrix<NX, NU, T>&                 N = Matrix<NX, NU, T>{}
) {
    return online::lqrd(sys.A, sys.B, Q, R, Ts, N);
}

/**
 * @brief Linear-Quadratic Integral design for tracking with servo action (runtime version)
 *
 * @param sys  State-space system
 * @param Q    Augmented state cost matrix (state + integral error)
 * @param R    Input cost matrix
 * @param dof  Servo degrees of freedom (1DOF or 2DOF)
 *
 * @return LQIResult containing state and integral gains
 */
template<size_t NX, size_t NU, size_t NY, typename T = double>
[[nodiscard]] constexpr design::LQIResult<NX, NU, NY, T> lqi(
    const StateSpace<NX, NU, NY, NX, NY, T>& sys,
    const Matrix<NX + NY, NX + NY, T>&       Q,
    const Matrix<NU, NU, T>&                 R,
    ServoDOF                                 dof = ServoDOF::OneDOF
) {
    // Build augmented system
    Matrix<NX + NY, NX + NY, T> A_aug{};
    Matrix<NX + NY, NU, T>      B_aug{};

    for (size_t i = 0; i < NX; ++i) {
        for (size_t j = 0; j < NX; ++j) {
            A_aug(i, j) = sys.A(i, j);
        }
    }
    for (size_t i = 0; i < NY; ++i) {
        for (size_t j = 0; j < NX; ++j) {
            A_aug(NX + i, j) = sys.C(i, j);
        }
        A_aug(NX + i, NX + i) = T{1};
    }
    for (size_t i = 0; i < NX; ++i) {
        for (size_t j = 0; j < NU; ++j) {
            B_aug(i, j) = sys.B(i, j);
        }
    }

    // Solve DARE for augmented system
    Matrix<NX + NY, NX + NY, T> P_aug = dare(A_aug, B_aug, Q, R);

    // Compute augmented gain
    const Matrix<NU, NU, T> S = R + B_aug.transpose() * P_aug * B_aug;
    const auto              S_inv = S.inverse();

    Matrix<NU, NX + NY, T> K_aug{};
    if (S_inv) {
        K_aug = S_inv.value() * B_aug.transpose() * P_aug * A_aug;
    }

    // Extract Kx and Ki
    Matrix<NU, NX, T> Kx{};
    Matrix<NU, NY, T> Ki{};
    for (size_t i = 0; i < NU; ++i) {
        for (size_t j = 0; j < NX; ++j) {
            Kx(i, j) = K_aug(i, j);
        }
        for (size_t j = 0; j < NY; ++j) {
            Ki(i, j) = K_aug(i, NX + j);
        }
    }

    return design::LQIResult<NX, NU, NY, T>(sys.A, sys.B, sys.C, Kx, Ki, P_aug, dof);
}

/**
 * @brief Linear-Quadratic-Gaussian regulator design combining LQR and Kalman filter (runtime version)
 *
 * @param sys     State-space system
 * @param Q_lqr   State cost for LQR
 * @param R_lqr   Input cost for LQR
 * @param Q_kf    Process noise covariance for Kalman filter
 * @param R_kf    Measurement noise covariance for Kalman filter
 * @param N       (optional) Cross-term cost matrix
 *
 * @return LQGResult combining LQR and Kalman filter designs
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
[[nodiscard]] constexpr design::LQGResult<NX, NU, NY, NW, NV, T> lqg(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NX, NX, T>&                 Q_lqr,
    const Matrix<NU, NU, T>&                 R_lqr,
    const Matrix<NW, NW, T>&                 Q_kf,
    const Matrix<NV, NV, T>&                 R_kf,
    const Matrix<NX, NU, T>&                 N = Matrix<NX, NU, T>{}
) {
    // Solve DARE for LQR gain
    const Matrix<NX, NX, T> P = dare(sys.A, sys.B, Q_lqr, R_lqr);

    // Compute LQR gain
    const Matrix<NU, NU, T> S = R_lqr + sys.B.transpose() * P * sys.B;
    const auto              S_inv = S.inverse();

    Matrix<NU, NX, T> K{};
    if (S_inv) {
        K = S_inv.value() * (sys.B.transpose() * P * sys.A + N.transpose());
    }

    // Create LQR result and Kalman result
    design::LQRResult<NX, NU, T>                lqr_result{K, P, ColVec<NX, T>{}};
    design::KalmanResult<NX, NU, NY, NW, NV, T> kalman_result{sys, Q_kf, R_kf};

    return design::LQGResult<NX, NU, NY, NW, NV, T>(lqr_result, kalman_result);
}

/**
 * @brief Linear-Quadratic-Gaussian with integral action for tracking (runtime version)
 *
 * @param sys      State-space system
 * @param Q_aug    Augmented state cost (state + integral error)
 * @param R        Input cost matrix
 * @param Q_kf     Process noise covariance for Kalman filter
 * @param R_kf     Measurement noise covariance for Kalman filter
 * @param dof      Servo degrees of freedom (1DOF or 2DOF)
 *
 * @return LQGIResult with integral action for tracking
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
[[nodiscard]] constexpr design::LQGIResult<NX, NU, NY, NW, NV, T> lqgtrack(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NX + NY, NX + NY, T>&       Q_aug,
    const Matrix<NU, NU, T>&                 R,
    const Matrix<NW, NW, T>&                 Q_kf,
    const Matrix<NV, NV, T>&                 R_kf,
    ServoDOF                                 dof = ServoDOF::TwoDOF
) {
    // Build augmented system for LQI
    Matrix<NX + NY, NX + NY, T> A_aug{};
    Matrix<NX + NY, NU, T>      B_aug{};

    for (size_t i = 0; i < NX; ++i) {
        for (size_t j = 0; j < NX; ++j) {
            A_aug(i, j) = sys.A(i, j);
        }
    }
    for (size_t i = 0; i < NY; ++i) {
        for (size_t j = 0; j < NX; ++j) {
            A_aug(NX + i, j) = sys.C(i, j);
        }
        A_aug(NX + i, NX + i) = T{1};
    }
    for (size_t i = 0; i < NX; ++i) {
        for (size_t j = 0; j < NU; ++j) {
            B_aug(i, j) = sys.B(i, j);
        }
    }

    // Solve DARE for augmented system
    Matrix<NX + NY, NX + NY, T> P_aug = dare(A_aug, B_aug, Q_aug, R);

    // Compute augmented gain
    const Matrix<NU, NU, T> S = R + B_aug.transpose() * P_aug * B_aug;
    const auto              S_inv = S.inverse();

    Matrix<NU, NX + NY, T> K_aug{};
    if (S_inv) {
        K_aug = S_inv.value() * B_aug.transpose() * P_aug * A_aug;
    }

    // Extract Kx and Ki
    Matrix<NU, NX, T> Kx{};
    Matrix<NU, NY, T> Ki{};
    for (size_t i = 0; i < NU; ++i) {
        for (size_t j = 0; j < NX; ++j) {
            Kx(i, j) = K_aug(i, j);
        }
        for (size_t j = 0; j < NY; ++j) {
            Ki(i, j) = K_aug(i, NX + j);
        }
    }

    // Create LQI result and Kalman result
    design::LQIResult<NX, NU, NY, T>            lqi_result(sys.A, sys.B, sys.C, Kx, Ki, P_aug, dof);
    design::KalmanResult<NX, NU, NY, NW, NV, T> kalman_result{sys, Q_kf, R_kf};

    return design::LQGIResult<NX, NU, NY, NW, NV, T>(lqi_result, kalman_result);
}

/**
 * @brief Combine separate Kalman filter and LQR designs into an LQG controller (runtime version)
 *
 * @param kest        Kalman filter design result
 * @param lqr_result  LQR controller design result
 *
 * @return LQGResult combining the provided Kalman and LQR designs
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
[[nodiscard]] constexpr design::LQGResult<NX, NU, NY, NW, NV, T> lqgreg(
    const design::KalmanResult<NX, NU, NY, NW, NV, T>& kest,
    const design::LQRResult<NX, NU, T>&                lqr_result
) {
    return design::LQGResult<NX, NU, NY, NW, NV, T>(lqr_result, kest);
}

// ============================================================================
// Long-Named Wrapper Functions (aliases for clarity)
// ============================================================================

/**
 * @brief Design discrete LQR for already-discrete system
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
[[nodiscard]] constexpr design::LQRResult<NX, NU, T> discrete_lqr(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NX, NX, T>& Q,
    const Matrix<NU, NU, T>& R,
    const Matrix<NX, NU, T>& N = Matrix<NX, NU, T>{}
) {
    return dlqr(A, B, Q, R, N);
}

/**
 * @brief Design continuous LQR
 *
 * @param A State transition matrix
 * @param B Control input matrix
 * @param Q State cost matrix
 * @param R Input cost matrix
 * @param N (optional) Cross-term cost matrix
 *
 * @return LQRResult containing gain matrix and solution to CARE
 */
template<size_t NX, size_t NU, typename T = double>
[[nodiscard]] constexpr design::LQRResult<NX, NU, T> continuous_lqr(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NX, NX, T>& Q,
    const Matrix<NU, NU, T>& R,
    const Matrix<NX, NU, T>& N = Matrix<NX, NU, T>{}
) {
    return lqr(A, B, Q, R, N);
}

/**
 * @brief Design continuous LQR for state-space system
 *
 * @param sys State-space system
 * @param Q   State cost matrix
 * @param R   Input cost matrix
 * @param N   (optional) Cross-term cost matrix
 *
 * @return LQRResult containing gain matrix and solution to CARE
 */
template<size_t NX, size_t NU, size_t NY, typename T = double>
[[nodiscard]] constexpr design::LQRResult<NX, NU, T> continuous_lqr(
    const StateSpace<NX, NU, NY, NX, NY, T>& sys,
    const Matrix<NX, NX, T>&                 Q,
    const Matrix<NU, NU, T>&                 R,
    const Matrix<NX, NU, T>&                 N = Matrix<NX, NU, T>{}
) {
    return lqr(sys, Q, R, N);
}

/**
 * @brief Design discrete LQR from continuous-time system
 *
 * @param A  State transition matrix (continuous-time)
 * @param B  Control input matrix (continuous-time)
 * @param Q  State cost matrix
 * @param R  Input cost matrix
 * @param Ts Sampling time
 * @param N  (optional) Cross-term cost matrix
 *
 * @return LQRResult containing gain matrix and solution to DARE
 */
template<size_t NX, size_t NU, typename T = double>
[[nodiscard]] constexpr design::LQRResult<NX, NU, T> discrete_lqr_from_continuous(
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
 * @brief Design discrete LQR from continuous-time state-space system
 *
 * @param sys State-space system (continuous-time)
 * @param Q   State cost matrix
 * @param R   Input cost matrix
 * @param Ts  Sampling time
 * @param N   (optional) Cross-term cost matrix
 *
 * @return LQRResult containing gain matrix and solution to DARE
 */
template<size_t NX, size_t NU, size_t NY, typename T = double>
[[nodiscard]] constexpr design::LQRResult<NX, NU, T> discrete_lqr_from_continuous(
    const StateSpace<NX, NU, NY, NX, NY, T>& sys,
    const Matrix<NX, NX, T>&                 Q,
    const Matrix<NU, NU, T>&                 R,
    T                                        Ts,
    const Matrix<NX, NU, T>&                 N = Matrix<NX, NU, T>{}
) {
    return lqrd(sys, Q, R, Ts, N);
}

/**
 * @brief Design continuous LQR with integral action for state-space system
 *
 * @param sys State-space system
 * @param Q   Augmented state cost matrix (state + integral)
 * @param R   Input cost matrix
 * @param dof Servo degrees of freedom (1DOF or 2DOF)
 *
 * @return LQIResult containing gains and solution to DARE
 */
template<size_t NX, size_t NU, size_t NY, typename T = double>
[[nodiscard]] constexpr design::LQIResult<NX, NU, NY, T> lqr_with_integral(
    const StateSpace<NX, NU, NY, NX, NY, T>& sys,
    const Matrix<NX + NY, NX + NY, T>&       Q,
    const Matrix<NU, NU, T>&                 R,
    ServoDOF                                 dof = ServoDOF::OneDOF
) {
    return lqi(sys, Q, R, dof);
}

/**
 * @brief  LQG regulator design combining LQR and Kalman filter
 *
 * @param sys    State-space system
 * @param Q_lqr  State cost for LQR
 * @param R_lqr  Input cost for LQR
 * @param Q_kf   Process noise covariance for Kalman filter
 * @param R_kf   Measurement noise covariance for Kalman filter
 * @param N      (optional) Cross-term cost matrix
 *
 * @return LQGResult combining LQR and Kalman filter designs
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
[[nodiscard]] constexpr design::LQGResult<NX, NU, NY, NW, NV, T> lqg_regulator(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NX, NX, T>&                 Q_lqr,
    const Matrix<NU, NU, T>&                 R_lqr,
    const Matrix<NW, NW, T>&                 Q_kf,
    const Matrix<NV, NV, T>&                 R_kf,
    const Matrix<NX, NU, T>&                 N = Matrix<NX, NU, T>{}
) {
    return lqg(sys, Q_lqr, R_lqr, Q_kf, R_kf, N);
}

/**
 * @brief LQG with integral action for tracking
 *
 * @param sys    State-space system
 * @param Q_aug  Augmented state cost (state + integral)
 * @param R      Input cost
 * @param Q_kf   Process noise covariance
 * @param R_kf   Measurement noise covariance
 * @param dof    Servo degrees of freedom (1DOF or 2DOF)
 *
 * @return  LQGIResult with integral action for tracking
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
[[nodiscard]] constexpr design::LQGIResult<NX, NU, NY, NW, NV, T> lqg_servo(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NX + NY, NX + NY, T>&       Q_aug,
    const Matrix<NU, NU, T>&                 R,
    const Matrix<NW, NW, T>&                 Q_kf,
    const Matrix<NV, NV, T>&                 R_kf,
    ServoDOF                                 dof = ServoDOF::TwoDOF
) {
    return lqgtrack(sys, Q_aug, R, Q_kf, R_kf, dof);
}

/**
 * @brief Combine separate Kalman and LQR results into an LQG controller
 *
 * @param kest          Kalman filter design result
 * @param lqr_result    LQR controller design result
 *
 * @return LQG Regulator combining the provided Kalman and LQR results
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
[[nodiscard]] constexpr design::LQGResult<NX, NU, NY, NW, NV, T> lqg_from_parts(
    const design::KalmanResult<NX, NU, NY, NW, NV, T>& kest,
    const design::LQRResult<NX, NU, T>&                lqr_result
) {
    return lqgreg(kest, lqr_result);
}

} // namespace online
