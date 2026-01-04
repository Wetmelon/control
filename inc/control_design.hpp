#pragma once

#include <cmath>
#include <limits>
#include <utility>

#include "constexpr_complex.hpp"
#include "discretization.hpp"
#include "eigen.hpp"
#include "matrix.hpp"
#include "ricatti.hpp"
#include "state_space.hpp"

namespace wetmelon::control {

/**
 * @defgroup stability_analysis Stability Analysis
 * @brief Functions to analyze closed-loop stability of control systems
 *
 * For continuous systems: stable if all eigenvalues have Re(λ) < 0 (left half plane)
 * For discrete systems: stable if all eigenvalues have |λ| < 1 (inside unit circle)
 */
namespace stability {

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
        T magnitude = wet::abs(eigen.values[i]);
        if (magnitude >= T{1}) {
            return false;
        }
    }
    return true;
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
[[nodiscard]] constexpr ColVec<NX, wet::complex<T>> closed_loop_poles(
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
 * @brief MATLAB®-style API functions for LQR, LQI, LQG, and Kalman filter design
 *
 * These functions mirror MATLAB®'s Control System Toolbox API for familiarity.
 * Both design:: (consteval) and online:: (constexpr) variants are provided.
 */

/**
 * @struct KalmanResult
 * @brief Kalman filter design result
 *
 * Contains filter gains and covariance matrices for optimal state estimation.
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
struct KalmanResult {
    StateSpace<NX, NU, NY, NW, NV, T> sys{};          //!< System model
    Matrix<NW, NW, T>                 Q{};            //!< Process noise covariance
    Matrix<NV, NV, T>                 R{};            //!< Measurement noise covariance
    Matrix<NX, NY, T>                 L{};            //!< Kalman gain (steady-state)
    Matrix<NX, NX, T>                 P{};            //!< Error covariance (steady-state)
    bool                              success{false}; //!< Indicates filter design success

    template<typename U>
    [[nodiscard]] consteval auto as() const {
        return KalmanResult<NX, NU, NY, NW, NV, U>{
            sys.template as<U>(),
            Q.template as<U>(),
            R.template as<U>(),
            L.template as<U>(),
            P.template as<U>(),
            success
        };
    }
};

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
 * @struct LQIResult
 * @brief Linear-Quadratic Integral controller design result
 *
 * Contains gains and Riccati solution for servo control with integral action.
 */
template<size_t NX, size_t NU, size_t NY, typename T = double>
struct LQIResult {
    Matrix<NU, NX, T>                Kx{};           //!< State gain
    Matrix<NU, NY, T>                Ki{};           //!< Integral gain
    Matrix<NX + NY, NX + NY, T>      S{};            //!< Riccati solution for augmented system
    ColVec<NX + NY, wet::complex<T>> e{};            //!< Full complex closed-loop poles (eigenvalues)
    bool                             success{false}; //!< Indicates Riccati solve success

    template<typename U>
    [[nodiscard]] consteval auto as() const {
        return LQIResult<NX, NU, NY, U>{
            Kx.template as<U>(),
            Ki.template as<U>(),
            S.template as<U>(),
            e.template as<wet::complex<U>>(),
            success
        };
    }
};

/**
 * @struct LQGResult
 * @brief Linear-Quadratic-Gaussian controller design result
 *
 * Combines LQR and Kalman filter designs for separation principle-based control.
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
struct LQGResult {
    LQRResult<NX, NU, T>                lqr{};          //!< LQR design result
    KalmanResult<NX, NU, NY, NW, NV, T> kalman{};       //!< Kalman filter result
    bool                                success{false}; //!< Indicates combined design success

    template<typename U>
    [[nodiscard]] consteval auto as() const {
        return LQGResult<NX, NU, NY, NW, NV, U>{
            lqr.template as<U>(),
            kalman.template as<U>(),
            success
        };
    }
};

/**
 * @struct LQGIResult
 * @brief Linear-Quadratic-Gaussian Integral controller design result
 *
 * Combines LQI and Kalman filter designs for servo control with state estimation.
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
struct LQGIResult {
    LQIResult<NX, NU, NY, T>            lqi{};          //!< LQI design result
    KalmanResult<NX, NU, NY, NW, NV, T> kalman{};       //!< Kalman filter result
    bool                                success{false}; //!< Indicates combined design success

    template<typename U>
    [[nodiscard]] consteval auto as() const {
        return LQGIResult<NX, NU, NY, NW, NV, U>{
            lqi.template as<U>(),
            kalman.template as<U>(),
            success
        };
    }
};

/**
 * @defgroup lqr_helpers LQR Result Helper Functions
 * @brief Internal functions to construct LQRResult with stability analysis
 */
namespace detail {

/**
 * @brief Symmetrize a square matrix
 */
template<size_t N, typename T>
[[nodiscard]] constexpr Matrix<N, N, T> symmetrize(const Matrix<N, N, T>& M) {
    Matrix<N, N, T> sym = M;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i + 1; j < N; ++j) {
            T avg = (sym(i, j) + sym(j, i)) * T{0.5};
            sym(i, j) = avg;
            sym(j, i) = avg;
        }
    }
    return sym;
}

/**
 * @brief Condition number of symmetric positive-definite matrix using eigenvalues
 */
template<size_t N, typename T>
[[nodiscard]] constexpr std::optional<T> condition_number_spd(const Matrix<N, N, T>& M) {
    auto eig = compute_eigenvalues_qr(M);
    if (!eig.converged)
        return std::nullopt;

    T min_eig = std::numeric_limits<T>::max();
    T max_eig = T{0};
    for (size_t i = 0; i < N; ++i) {
        T val = eig.eigenvalues_real(i, i);
        if (val <= T{0})
            return std::nullopt;
        if (val < min_eig)
            min_eig = val;
        if (val > max_eig)
            max_eig = val;
    }

    if (min_eig == T{0})
        return std::nullopt;

    return max_eig / min_eig;
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
    design::LQRResult<NX, NU, T> result{};

    const auto dare_opt = dare(A, B, Q, R, N);
    if (!dare_opt) {
        return result;
    }

    const Matrix<NX, NX, T> S = dare_opt.value();
    const Matrix<NU, NU, T> denom = R + B.transpose() * S * B;
    const auto              denom_inv = denom.inverse();

    if (!denom_inv) {
        return result;
    }

    Matrix<NU, NX, T> K = denom_inv.value() * (B.transpose() * S * A + N.transpose());
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
 * @return LQIResult containing state and integral gains
 */
template<size_t NX, size_t NU, size_t NY, typename T = double>
[[nodiscard]] consteval LQIResult<NX, NU, NY, T> lqi(
    const StateSpace<NX, NU, NY, NX, NY, T>& sys,
    const Matrix<NX + NY, NX + NY, T>&       Q,
    const Matrix<NU, NU, T>&                 R
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
    const auto dare_opt = dare(A_aug, B_aug, Q, R);
    if (!dare_opt) {
        return LQIResult<NX, NU, NY, T>{};
    }
    Matrix<NX + NY, NX + NY, T> P_aug = dare_opt.value();

    // Compute augmented gain: K_aug = (R + B'PB)^{-1} * B'PA
    const Matrix<NU, NU, T> S = R + B_aug.transpose() * P_aug * B_aug;
    const auto              S_inv = S.inverse();

    if (!S_inv) {
        return LQIResult<NX, NU, NY, T>{};
    }

    Matrix<NU, NX + NY, T> K_aug{};
    K_aug = S_inv.value() * B_aug.transpose() * P_aug * A_aug;

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

    ColVec<NX + NY, wet::complex<T>> poles = stability::closed_loop_poles(A_aug, B_aug, K_aug);
    return LQIResult<NX, NU, NY, T>(Kx, Ki, P_aug, poles, true);
}

/**
 * @brief Steady-state Kalman filter design
 *
 * Designs optimal steady-state Kalman gain for discrete system: x[k+1] = A*x[k] + w[k], y[k] = C*x[k] + v[k]
 *
 * @param sys  State-space system (discrete-time)
 * @param Q    Process noise covariance
 * @param R    Measurement noise covariance
 *
 * @return KalmanResult containing steady-state gain and covariance
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
[[nodiscard]] consteval KalmanResult<NX, NU, NY, NW, NV, T> kalman(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NW, NW, T>&                 Q,
    const Matrix<NV, NV, T>&                 R
) {
    KalmanResult<NX, NU, NY, NW, NV, T> result{sys, Q, R};

    // Solve filter DARE: P = A*P*A' + Q - A*P*C'*(C*P*C' + R)^{-1}*C*P*A'
    // This is equivalent to dare(A', C', Q, R)
    const auto dare_opt = dare(sys.A.transpose(), sys.C.transpose(), Q, R);
    if (!dare_opt) {
        return result;
    }
    result.P = dare_opt.value();

    // Compute Kalman gain: L = P*C'*(C*P*C' + R)^{-1}
    const Matrix<NY, NY, T> S = sys.C * result.P * sys.C.transpose() + R;
    const auto              S_inv = S.inverse();
    if (!S_inv) {
        return result;
    }
    result.L = result.P * sys.C.transpose() * S_inv.value();

    result.success = true;
    return result;
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
    const auto lqr_result = dlqr(sys.A, sys.B, Q_lqr, R_lqr, N);
    const auto kalman_result = kalman(sys, Q_kf, R_kf);
    return LQGResult<NX, NU, NY, NW, NV, T>{lqr_result, kalman_result, lqr_result.success && kalman_result.success};
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
    const Matrix<NV, NV, T>&                 R_kf   // Measurement noise covariance
) {
    const auto lqi_result = lqi(sys, Q_aug, R);
    const auto kalman_result = kalman(sys, Q_kf, R_kf);
    return LQGIResult<NX, NU, NY, NW, NV, T>{lqi_result, kalman_result, lqi_result.success && kalman_result.success};
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
    return LQGResult<NX, NU, NY, NW, NV, T>{lqr_result, kest, lqr_result.success && kest.success};
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
    const Matrix<NU, NU, T>&                 R
) {
    return lqi(sys, Q, R);
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
    const Matrix<NV, NV, T>&                 R_kf
) {
    return lqgtrack(sys, Q_aug, R, Q_kf, R_kf);
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
// online:: Namespace - Runtime versions of MATLAB®-style functions
// ============================================================================

namespace online {

// Forward declarations of online result types
template<size_t NX, size_t NU, typename T = double>
struct LQRResult;

template<size_t NX, size_t NU, size_t NY, typename T = double>
struct LQIResult;

template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
struct KalmanResult;

template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
struct LQGResult;

template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
struct LQGIResult;

/**
 * @brief Numerical linearization around operating point (central differences)
 */
template<size_t NX, size_t NU, typename F, typename T = double>
[[nodiscard]] constexpr std::pair<Matrix<NX, NX, T>, Matrix<NX, NU, T>> linearize(
    const F&             f,
    const ColVec<NX, T>& x,
    const ColVec<NU, T>& u,
    T                    eps = T{1e-5}
) {
    Matrix<NX, NX, T> A = Matrix<NX, NX, T>::zeros();
    Matrix<NX, NU, T> B = Matrix<NX, NU, T>::zeros();

    // State Jacobian
    for (size_t j = 0; j < NX; ++j) {
        ColVec<NX, T> x_plus = x;
        ColVec<NX, T> x_minus = x;
        x_plus[j] += eps;
        x_minus[j] -= eps;
        ColVec<NX, T> f_plus = f(x_plus, u);
        ColVec<NX, T> f_minus = f(x_minus, u);
        ColVec<NX, T> diff = (f_plus - f_minus) * (T{0.5} / eps);
        for (size_t i = 0; i < NX; ++i) {
            A(i, j) = diff[i];
        }
    }

    // Input Jacobian
    for (size_t j = 0; j < NU; ++j) {
        ColVec<NU, T> u_plus = u;
        ColVec<NU, T> u_minus = u;
        u_plus[j] += eps;
        u_minus[j] -= eps;
        ColVec<NX, T> f_plus = f(x, u_plus);
        ColVec<NX, T> f_minus = f(x, u_minus);
        ColVec<NX, T> diff = (f_plus - f_minus) * (T{0.5} / eps);
        for (size_t i = 0; i < NX; ++i) {
            B(i, j) = diff[i];
        }
    }

    return {A, B};
}

/**
 * @struct LQRResult
 * @brief Runtime LQR design result (online namespace)
 */
template<size_t NX, size_t NU, typename T>
struct LQRResult {
    Matrix<NU, NX, T>           K{};
    Matrix<NX, NX, T>           S{};
    ColVec<NX, wet::complex<T>> e{};
    bool                        success{false};

    template<typename U>
    [[nodiscard]] constexpr auto as() const {
        return LQRResult<NX, NU, U>{
            K.template as<U>(),
            S.template as<U>(),
            e.template as<wet::complex<U>>(),
            success
        };
    }
};

/**
 * @struct LQIResult
 * @brief Runtime LQI design result (online namespace)
 */
template<size_t NX, size_t NU, size_t NY, typename T>
struct LQIResult {
    Matrix<NU, NX, T>                Kx{};
    Matrix<NU, NY, T>                Ki{};
    Matrix<NX + NY, NX + NY, T>      S{};
    ColVec<NX + NY, wet::complex<T>> e{};
    bool                             success{false};

    template<typename U>
    [[nodiscard]] constexpr auto as() const {
        return LQIResult<NX, NU, NY, U>{
            Kx.template as<U>(),
            Ki.template as<U>(),
            S.template as<U>(),
            e.template as<wet::complex<U>>(),
            success
        };
    }
};

/**
 * @struct KalmanResult
 * @brief Runtime Kalman filter design result (online namespace)
 */
template<size_t NX, size_t NU, size_t NY, size_t NW, size_t NV, typename T>
struct KalmanResult {
    StateSpace<NX, NU, NY, NW, NV, T> sys{};
    Matrix<NW, NW, T>                 Q{};
    Matrix<NV, NV, T>                 R{};
    Matrix<NX, NY, T>                 L{};
    Matrix<NX, NX, T>                 P{};
    bool                              success{false};

    template<typename U>
    [[nodiscard]] constexpr auto as() const {
        return KalmanResult<NX, NU, NY, NW, NV, U>{
            sys.template as<U>(),
            Q.template as<U>(),
            R.template as<U>(),
            L.template as<U>(),
            P.template as<U>(),
            success
        };
    }
};

/**
 * @struct LQGResult
 * @brief Runtime LQG design result (online namespace)
 */
template<size_t NX, size_t NU, size_t NY, size_t NW, size_t NV, typename T>
struct LQGResult {
    LQRResult<NX, NU, T>                lqr{};
    KalmanResult<NX, NU, NY, NW, NV, T> kalman{};
    bool                                success{false};

    template<typename U>
    [[nodiscard]] constexpr auto as() const {
        return LQGResult<NX, NU, NY, NW, NV, U>{
            lqr.template as<U>(),
            kalman.template as<U>(),
            success
        };
    }
};

/**
 * @struct LQGIResult
 * @brief Runtime LQGI design result (online namespace)
 */
template<size_t NX, size_t NU, size_t NY, size_t NW, size_t NV, typename T>
struct LQGIResult {
    LQIResult<NX, NU, NY, T>            lqi{};
    KalmanResult<NX, NU, NY, NW, NV, T> kalman{};
    bool                                success{false};

    template<typename U>
    [[nodiscard]] constexpr auto as() const {
        return LQGIResult<NX, NU, NY, NW, NV, U>{
            lqi.template as<U>(),
            kalman.template as<U>(),
            success
        };
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
[[nodiscard]] constexpr online::LQRResult<NX, NU, T> dlqr(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NX, NX, T>& Q,
    const Matrix<NU, NU, T>& R,
    const Matrix<NX, NU, T>& N = Matrix<NX, NU, T>{}
) {
    online::LQRResult<NX, NU, T> result{};

    const auto dare_opt = dare(A, B, Q, R, N);
    if (!dare_opt) {
        return result;
    }
    const Matrix<NX, NX, T> S = dare_opt.value();

    //! Compute (R + B'SB) and invert - skip expensive condition number check at runtime
    const Matrix<NU, NU, T> denom = R + B.transpose() * S * B;
    const auto              denom_inv = denom.inverse();

    if (!denom_inv) {
        return result;
    }

    Matrix<NU, NX, T> K = denom_inv.value() * (B.transpose() * S * A + N.transpose());

    result = online::LQRResult<NX, NU, T>{K, S, stability::closed_loop_poles(A, B, K), true};
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
[[nodiscard]] constexpr online::LQRResult<NX, NU, T> lqrd(
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
[[nodiscard]] constexpr online::LQRResult<NX, NU, T> lqrd(
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
[[nodiscard]] constexpr online::LQIResult<NX, NU, NY, T> lqi(
    const StateSpace<NX, NU, NY, NX, NY, T>& sys,
    const Matrix<NX + NY, NX + NY, T>&       Q,
    const Matrix<NU, NU, T>&                 R
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
    const auto dare_opt = dare(A_aug, B_aug, Q, R);
    if (!dare_opt) {
        return online::LQIResult<NX, NU, NY, T>{};
    }
    Matrix<NX + NY, NX + NY, T> P_aug = dare_opt.value();

    // Compute augmented gain - skip expensive condition number check at runtime
    const Matrix<NU, NU, T> denom = R + B_aug.transpose() * P_aug * B_aug;
    const auto              denom_inv = denom.inverse();

    if (!denom_inv) {
        return online::LQIResult<NX, NU, NY, T>{};
    }
    Matrix<NU, NX + NY, T> K_aug{};
    K_aug = denom_inv.value() * B_aug.transpose() * P_aug * A_aug;
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

    ColVec<NX + NY, wet::complex<T>> poles = stability::closed_loop_poles(A_aug, B_aug, K_aug);
    return online::LQIResult<NX, NU, NY, T>(Kx, Ki, P_aug, poles, true);
}

/**
 * @brief Steady-state Kalman filter design (runtime version)
 *
 * Designs optimal steady-state Kalman gain for discrete system: x[k+1] = A*x[k] + w[k], y[k] = C*x[k] + v[k]
 *
 * @param sys  State-space system (discrete-time)
 * @param Q    Process noise covariance
 * @param R    Measurement noise covariance
 *
 * @return KalmanResult containing steady-state gain and covariance
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
[[nodiscard]] constexpr online::KalmanResult<NX, NU, NY, NW, NV, T> kalman(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NW, NW, T>&                 Q,
    const Matrix<NV, NV, T>&                 R
) {
    online::KalmanResult<NX, NU, NY, NW, NV, T> result{sys, Q, R};

    // Solve filter DARE: P = A*P*A' + Q - A*P*C'*(C*P*C' + R)^{-1}*C*P*A'
    // This is equivalent to dare(A', C', Q, R)
    const auto dare_opt = dare(sys.A.transpose(), sys.C.transpose(), Q, R);
    if (!dare_opt) {
        return result;
    }
    result.P = dare_opt.value();

    // Compute Kalman gain: L = P*C'*(C*P*C' + R)^{-1}
    const Matrix<NY, NY, T> S = sys.C * result.P * sys.C.transpose() + R;
    const auto              S_inv = S.inverse();
    if (!S_inv) {
        return result;
    }
    result.L = result.P * sys.C.transpose() * S_inv.value();

    result.success = true;
    return result;
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
[[nodiscard]] constexpr online::LQGResult<NX, NU, NY, NW, NV, T> lqg(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NX, NX, T>&                 Q_lqr,
    const Matrix<NU, NU, T>&                 R_lqr,
    const Matrix<NW, NW, T>&                 Q_kf,
    const Matrix<NV, NV, T>&                 R_kf,
    const Matrix<NX, NU, T>&                 N = Matrix<NX, NU, T>{}
) {
    const auto lqr_result = dlqr(sys.A, sys.B, Q_lqr, R_lqr, N);
    const auto kalman_result = kalman(sys, Q_kf, R_kf);
    return online::LQGResult<NX, NU, NY, NW, NV, T>{lqr_result, kalman_result, lqr_result.success && kalman_result.success};
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
[[nodiscard]] constexpr online::LQGIResult<NX, NU, NY, NW, NV, T> lqgtrack(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NX + NY, NX + NY, T>&       Q_aug,
    const Matrix<NU, NU, T>&                 R,
    const Matrix<NW, NW, T>&                 Q_kf,
    const Matrix<NV, NV, T>&                 R_kf
) {
    const auto lqi_result = lqi(sys, Q_aug, R);
    const auto kalman_result = kalman(sys, Q_kf, R_kf);
    return online::LQGIResult<NX, NU, NY, NW, NV, T>{lqi_result, kalman_result, lqi_result.success && kalman_result.success};
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
[[nodiscard]] constexpr online::LQGResult<NX, NU, NY, NW, NV, T> lqgreg(
    const online::KalmanResult<NX, NU, NY, NW, NV, T>& kest,
    const online::LQRResult<NX, NU, T>&                lqr_result
) {
    return online::LQGResult<NX, NU, NY, NW, NV, T>{lqr_result, kest, lqr_result.success && kest.success};
}

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
[[nodiscard]] constexpr online::LQRResult<NX, NU, T> discrete_lqr_from_continuous(
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
[[nodiscard]] constexpr online::LQRResult<NX, NU, T> discrete_lqr_from_continuous(
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
    const Matrix<NU, NU, T>&                 R
) {
    return lqi(sys, Q, R);
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
    const Matrix<NV, NV, T>&                 R_kf
) {
    return lqgtrack(sys, Q_aug, R, Q_kf, R_kf);
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
} // namespace wetmelon::control