#pragma once

#include <cmath>

#include "matrix.hpp"
#include "matrix/cholesky.hpp"
#include "state_space.hpp"

namespace wetmelon::control {

namespace design {

/**
 * @struct KalmanResult
 * @brief Kalman filter design result
 *
 * Contains filter gains and covariance matrices for optimal state estimation.
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = 0, size_t NV = 0, typename T = double>
struct KalmanResult {
    StateSpace<NX, NU, NY, NW, NV, T> sys{};          //!< System model
    Matrix<NW, NW, T>                 Q{};            //!< Process noise covariance
    Matrix<NV, NV, T>                 R{};            //!< Measurement noise covariance
    Matrix<NX, NY, T>                 L{};            //!< Kalman gain (steady-state)
    Matrix<NX, NX, T>                 P{};            //!< Error covariance (steady-state)
    bool                              success{false}; //!< Indicates filter design success

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
 * @brief Steady-state Kalman filter design
 *
 * Designs optimal steady-state Kalman gain for discrete system: x[k+1] = A*x[k] + B*u[k] + G*w[k], y[k] = C*x[k] + D*u[k] + H*v[k]
 *
 * @param sys  State-space system (discrete-time)
 * @param Q    Process noise covariance (covariance of w[k])
 * @param R    Measurement noise covariance (covariance of v[k])
 *
 * @return KalmanResult containing steady-state gain and covariance
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = 0, size_t NV = 0, typename T = double>
[[nodiscard]] constexpr KalmanResult<NX, NU, NY, NW, NV, T> kalman(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NW, NW, T>&                 Q,
    const Matrix<NV, NV, T>&                 R
) {
    KalmanResult<NX, NU, NY, NW, NV, T> result{sys, Q, R};

    // Compute effective noise covariances accounting for G and H
    const Matrix<NW, NW, T> Q_eff = sys.G * Q * sys.G.t();
    const Matrix<NV, NV, T> R_eff = sys.H * R * sys.H.t();

    // Fast path: R_eff ≈ 0 with square, invertible C → analytical solution
    const T r_eps = std::is_same_v<T, float> ? T{1e-6} : T{1e-10};
    if (R_eff.norm() < r_eps) {
        if constexpr (NY == NX) {
            const auto C_inv = sys.C.inverse();
            if (C_inv.has_value()) {
                result.P = Q_eff;
                result.L = C_inv.value();
                result.success = true;
                return result;
            }
        }
    }

    // Solve filter DARE: P = A*P*A' + Q_eff - A*P*C'*(C*P*C' + R_eff)^{-1}*C*P*A'
    // dare() handles R ≥ 0 (falls back to RDE iteration when R is singular)
    const auto dare_opt = dare(sys.A.transpose(), sys.C.transpose(), Q_eff, R_eff);
    if (!dare_opt) {
        return result;
    }
    result.P = dare_opt.value();

    // Compute Kalman gain: L = PCᵀS⁻¹ → solve S Lᵀ = CP
    const Matrix<NY, NY, T> S = sys.C * result.P * sys.C.t() + R_eff;
    const Matrix<NY, NX, T> CP = sys.C * result.P;
    auto                    L_opt = mat::cholesky_solve(S, CP);
    if (!L_opt) {
        L_opt = mat::lu_solve(S, CP);
    }
    if (!L_opt) {
        return result;
    }
    result.L = L_opt.value().transpose();

    result.success = true;
    return result;
}
} // namespace design

// For embedded systems running in ISR or RTOS scheduler.
// Assumes sys contains discrete-time matrices (use discretization functions to discretize).
template<size_t NX, size_t NU, size_t NY, size_t NW = 0, size_t NV = 0, typename T = double>
struct KalmanFilter {
    constexpr KalmanFilter() = default;

    constexpr KalmanFilter(
        const StateSpace<NX, NU, NY, NW, NV, T>& sys_,
        const Matrix<NW, NW, T>&                 Q_,
        const Matrix<NV, NV, T>&                 R_,
        const ColVec<NX, T>&                     x0 = ColVec<NX, T>{},
        const Matrix<NX, NX, T>&                 P0 = Matrix<NX, NX, T>::identity()
    ) : sys(sys_), x(x0), P(P0), Q(Q_), R(R_) {}

    // Type conversion constructor
    template<typename U>
    constexpr KalmanFilter(const KalmanFilter<NX, NU, NY, NW, NV, U>& other)
        : sys(other.model()),
          x(other.state()),
          P(other.covariance()),
          Q(other.process_noise_covariance()),
          R(other.measurement_noise_covariance()),
          innov(other.innovation()) {}

    // Predict: x[k+1|k] = A*x[k|k] + B*u[k], P[k+1|k] = A*P*A' + G*Q*G'
    constexpr void predict(const ColVec<NU, T>& u = ColVec<NU, T>{}) {
        x = sys.A * x + sys.B * u;
        P = sys.A * P * sys.A.t() + sys.G * Q * sys.G.t();
    }

    // Measurement update: returns false if innovation covariance is singular
    constexpr bool update(const ColVec<NY, T>& y, const ColVec<NU, T>& u = ColVec<NU, T>{}) {
        const auto y_pred = sys.C * x + sys.D * u;
        innov = y - y_pred;

        const auto              Ct = sys.C.transpose();
        const auto              Ht = sys.H.transpose();
        const Matrix<NY, NY, T> S = sys.C * P * Ct + sys.H * R * Ht;

        // K = PCᵀS⁻¹ → solve S Kᵀ = C P via Cholesky (S is symmetric positive definite)
        const auto K_opt = mat::cholesky_solve(S, sys.C * P);
        if (!K_opt) {
            return false;
        }

        const Matrix<NX, NY, T> K = K_opt.value().transpose();
        x = x + K * innov;

        // Joseph form: P = (I - K*C) * P * (I - K*C)' + K*H*R*H'*K'
        const auto I_KC = Matrix<NX, NX, T>::identity() - K * sys.C;
        const auto KH = K * sys.H;
        P = I_KC * P * I_KC.t() + KH * R * KH.t();
        return true;
    }

    // Accessors
    [[nodiscard]] constexpr const auto& model() const { return sys; }
    [[nodiscard]] constexpr const auto& innovation() const { return innov; }
    [[nodiscard]] constexpr const auto& state() const { return x; }
    [[nodiscard]] constexpr const auto& covariance() const { return P; }
    [[nodiscard]] constexpr const auto& process_noise_covariance() const { return Q; }
    [[nodiscard]] constexpr const auto& measurement_noise_covariance() const { return R; }

private:
    StateSpace<NX, NU, NY, NW, NV, T> sys{};
    ColVec<NX, T>                     x{};
    Matrix<NX, NX, T>                 P{};
    Matrix<NW, NW, T>                 Q{};
    Matrix<NV, NV, T>                 R{};
    ColVec<NY, T>                     innov{};
};

} // namespace wetmelon::control
