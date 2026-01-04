#pragma once

#include "discretization.hpp"
#include "matrix.hpp"
#include "ricatti.hpp"
#include "stability.hpp"
#include "state_space.hpp"

namespace wetmelon::control {
namespace online {

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

    //! Compute (R + B'SB) and invert - skip expensive condition number check at runtime
    const Matrix<NU, NU, T> denom = R + B.transpose() * S * B;
    const auto              denom_inv = denom.inverse();

    if (!denom_inv) {
        return result;
    }

    Matrix<NU, NX, T> K = denom_inv.value() * (B.transpose() * S * A + N.transpose());

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

// lqrd overload for StateSpace
template<size_t NX, size_t NU, size_t NY, typename T = double>
[[nodiscard]] constexpr LQRResult<NX, NU, T> lqrd(
    const StateSpace<NX, NU, NY, NX, NY, T>& sys,
    const Matrix<NX, NX, T>&                 Q,
    const Matrix<NU, NU, T>&                 R,
    T                                        Ts,
    const Matrix<NX, NU, T>&                 N = Matrix<NX, NU, T>{}
) {
    return lqrd(sys.A, sys.B, Q, R, Ts, N);
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
[[nodiscard]] constexpr LQIResult<NX, NU, NY, T> lqi(
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
        return LQIResult<NX, NU, NY, T>{};
    }
    Matrix<NX + NY, NX + NY, T> P_aug = dare_opt.value();

    // Compute augmented gain - skip expensive condition number check at runtime
    const Matrix<NU, NU, T> denom = R + B_aug.transpose() * P_aug * B_aug;
    const auto              denom_inv = denom.inverse();

    if (!denom_inv) {
        return LQIResult<NX, NU, NY, T>{};
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
    return LQIResult<NX, NU, NY, T>(Kx, Ki, P_aug, poles, true);
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
[[nodiscard]] constexpr KalmanResult<NX, NU, NY, NW, NV, T> kalman(
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
[[nodiscard]] constexpr LQGResult<NX, NU, NY, NW, NV, T> lqg(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NX, NX, T>&                 Q_lqr,
    const Matrix<NU, NU, T>&                 R_lqr,
    const Matrix<NW, NW, T>&                 Q_kf,
    const Matrix<NV, NV, T>&                 R_kf,
    const Matrix<NX, NU, T>&                 N = Matrix<NX, NU, T>{}
) {
    const auto lqr_result = dlqr(sys.A, sys.B, Q_lqr, R_lqr, N);
    const auto kalman_result = kalman(sys, Q_kf, R_kf);
    return LQGResult<NX, NU, NY, NW, NV, T>{lqr_result, kalman_result, lqr_result.success && kalman_result.success};
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
[[nodiscard]] constexpr LQGIResult<NX, NU, NY, NW, NV, T> lqgtrack(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NX + NY, NX + NY, T>&       Q_aug,
    const Matrix<NU, NU, T>&                 R,
    const Matrix<NW, NW, T>&                 Q_kf,
    const Matrix<NV, NV, T>&                 R_kf
) {
    const auto lqi_result = lqi(sys, Q_aug, R);
    const auto kalman_result = kalman(sys, Q_kf, R_kf);
    return LQGIResult<NX, NU, NY, NW, NV, T>{lqi_result, kalman_result, lqi_result.success && kalman_result.success};
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
[[nodiscard]] constexpr LQGResult<NX, NU, NY, NW, NV, T> lqgreg(
    const KalmanResult<NX, NU, NY, NW, NV, T>& kest,
    const LQRResult<NX, NU, T>&                lqr_result
) {
    return LQGResult<NX, NU, NY, NW, NV, T>{lqr_result, kest, lqr_result.success && kest.success};
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
[[nodiscard]] constexpr LQRResult<NX, NU, T> discrete_lqr(
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
[[nodiscard]] constexpr LQRResult<NX, NU, T> discrete_lqr_from_continuous(
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
[[nodiscard]] constexpr LQRResult<NX, NU, T> discrete_lqr_from_continuous(
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
[[nodiscard]] constexpr LQIResult<NX, NU, NY, T> lqr_with_integral(
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
[[nodiscard]] constexpr LQGResult<NX, NU, NY, NW, NV, T> lqg_regulator(
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
[[nodiscard]] constexpr LQGIResult<NX, NU, NY, NW, NV, T> lqg_servo(
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
[[nodiscard]] constexpr LQGResult<NX, NU, NY, NW, NV, T> lqg_from_parts(
    const KalmanResult<NX, NU, NY, NW, NV, T>& kest,
    const LQRResult<NX, NU, T>&                lqr_result
) {
    return lqgreg(kest, lqr_result);
}

} // namespace online
} // namespace wetmelon::control