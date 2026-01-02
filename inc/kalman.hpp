#pragma once

#include <cmath>

#include "matrix.hpp"
#include "state_space.hpp"

// Jacobian carrier types to avoid tuples
template<typename T, size_t NX>
struct StateJacobian {
    ColVec<NX, T>     x_pred{};
    Matrix<NX, NX, T> F{};
    Matrix<NX, NX, T> G{Matrix<NX, NX, T>::identity()};
};

template<typename T, size_t NY, size_t NX>
struct MeasJacobian {
    ColVec<NY, T>     z_pred{};
    Matrix<NY, NX, T> H{};
    Matrix<NY, NY, T> M{Matrix<NY, NY, T>::identity()};
};

// ---------------------------------------------------------------------------
// Linear Kalman Filter - Discrete-time only
// ---------------------------------------------------------------------------
// For embedded systems running in ISR or RTOS scheduler.
// Assumes sys contains discrete-time matrices (use design:: functions to discretize).
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
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
          y(other.innovation()) {}

    // Predict: x[k+1|k] = A*x[k|k] + B*u[k], P[k+1|k] = A*P*A' + G*Q*G'
    constexpr void predict(const ColVec<NU, T>& u = ColVec<NU, T>{}) {
        x = ColVec<NX, T>(sys.A * x + sys.B * u);
        P = sys.A * P * sys.A.transpose() + sys.G * Q * sys.G.transpose();
    }

    // Measurement update: returns false if innovation covariance is singular
    constexpr bool update(const ColVec<NY, T>& z, const ColVec<NU, T>& u = ColVec<NU, T>{}) {
        const auto z_pred = sys.C * x + sys.D * u;
        y = z - z_pred;

        const auto              Ct = sys.C.transpose();
        const auto              Ht = sys.H.transpose();
        const Matrix<NY, NY, T> S = sys.C * P * Ct + sys.H * R * Ht;

        const auto S_inv = S.inverse();
        if (!S_inv) {
            return false;
        }

        const Matrix<NX, NY, T> K = P * Ct * S_inv.value();
        x = ColVec<NX, T>(x + K * y);

        // Joseph form: P = (I - K*C) * P * (I - K*C)' + K*H*R*H'*K'
        const auto I_KC = Matrix<NX, NX, T>::identity() - K * sys.C;
        const auto KH = K * sys.H;
        P = I_KC * P * I_KC.transpose() + KH * R * KH.transpose();
        return true;
    }

    // Accessors
    [[nodiscard]] constexpr const auto& model() const { return sys; }
    [[nodiscard]] constexpr const auto& innovation() const { return y; }
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
    ColVec<NY, T>                     y{};
};

// ---------------------------------------------------------------------------
// Extended Kalman Filter - Discrete-time only
// ---------------------------------------------------------------------------
// For nonlinear systems on embedded hardware. User provides discrete-time Jacobians:
//   state_fn: (x, u) -> StateJacobian{x_next, F, G}  (F = df/dx, G = df/dw)
//   meas_fn:  (x, u) -> MeasJacobian{z_pred, H, M}   (H = dh/dx, M = dh/dv)
template<size_t NX, size_t NU, size_t NY, typename T = double>
struct ExtendedKalmanFilter {
    constexpr ExtendedKalmanFilter() = default;

    constexpr ExtendedKalmanFilter(
        const ColVec<NX, T>&     x0,
        const Matrix<NX, NX, T>& P0,
        const Matrix<NX, NX, T>& Q_,
        const Matrix<NY, NY, T>& R_
    ) : x(x0), P(P0), Q(Q_), R(R_) {}

    // Type conversion constructor
    template<typename U>
    constexpr ExtendedKalmanFilter(const ExtendedKalmanFilter<NX, NU, NY, U>& other)
        : x(other.state()), P(other.covariance()), Q(other.process_noise_covariance()), R(other.measurement_noise_covariance()), y(other.innovation()) {}

    // Predict: x[k+1|k] = f(x[k|k], u[k]), P[k+1|k] = F*P*F' + G*Q*G'
    template<typename StateFn>
    constexpr void predict(StateFn&& f, const ColVec<NU, T>& u = ColVec<NU, T>{}) {
        StateJacobian<T, NX> sj = f(x, u);
        x = sj.x_pred;
        P = sj.F * P * sj.F.transpose() + sj.G * Q * sj.G.transpose();
    }

    // Measurement update: returns false if innovation covariance is singular
    template<typename MeasFn>
    constexpr bool update(MeasFn&& h, const ColVec<NY, T>& z, const ColVec<NU, T>& u = ColVec<NU, T>{}) {
        MeasJacobian<T, NY, NX> mj = h(x, u);
        const auto              Ht = mj.H.transpose();
        y = z - mj.z_pred;
        const Matrix<NY, NY, T> S = mj.H * P * Ht + mj.M * R * mj.M.transpose();

        const auto S_inv = S.inverse();
        if (!S_inv)
            return false;

        const Matrix<NX, NY, T> K = P * Ht * S_inv.value();
        x = ColVec<NX, T>(x + K * y);

        // Joseph form
        const auto I_KH = Matrix<NX, NX, T>::identity() - K * mj.H;
        const auto KM = K * mj.M;
        P = I_KH * P * I_KH.transpose() + KM * R * KM.transpose();
        return true;
    }

    // Accessors
    [[nodiscard]] constexpr const auto& innovation() const { return y; }
    [[nodiscard]] constexpr const auto& state() const { return x; }
    [[nodiscard]] constexpr const auto& covariance() const { return P; }
    [[nodiscard]] constexpr const auto& process_noise_covariance() const { return Q; }
    [[nodiscard]] constexpr const auto& measurement_noise_covariance() const { return R; }

private:
    ColVec<NX, T>     x{};
    Matrix<NX, NX, T> P{};
    Matrix<NX, NX, T> Q{};
    Matrix<NY, NY, T> R{};
    ColVec<NY, T>     y{};
};
