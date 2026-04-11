#pragma once

#include <concepts>
#include <cstddef>

#include "matrix.hpp"
#include "matrix/cholesky.hpp"

namespace wetmelon::control {
namespace online {

}

namespace design {

} // namespace design

// EKF state prediction result: x_next = f(x, u), with Jacobians F = ∂f/∂x, G = ∂f/∂w
template<typename T, size_t NX>
struct StateJacobian {
    ColVec<NX, T>     x_pred{};                         // Predicted state
    Matrix<NX, NX, T> F{};                              // State transition Jacobian ∂f/∂x
    Matrix<NX, NX, T> G{Matrix<NX, NX, T>::identity()}; // Process noise Jacobian ∂f/∂w
};

// EKF state function: (x, u) -> StateJacobian
template<typename Fn, typename T, size_t NX, size_t NU>
concept EKFStateFn = requires(Fn&& fn, const ColVec<NX, T>& x, const ColVec<NU, T>& u) {
    { fn(x, u) } -> std::convertible_to<StateJacobian<T, NX>>;
};

// EKF/ESKF measurement result: y_pred = h(x), with Jacobians H = ∂h/∂x, M = ∂h/∂v
template<typename T, size_t NY, size_t NX>
struct MeasJacobian {
    ColVec<NY, T>     y_pred{};                         // Predicted measurement
    Matrix<NY, NX, T> H{};                              // Measurement Jacobian ∂h/∂x
    Matrix<NY, NY, T> M{Matrix<NY, NY, T>::identity()}; // Measurement noise Jacobian ∂h/∂v
};

// EKF measurement function: (x, u) -> MeasJacobian
template<typename Fn, typename T, size_t NX, size_t NU, size_t NY>
concept EKFMeasFn = requires(Fn&& fn, const ColVec<NX, T>& x, const ColVec<NU, T>& u) {
    { fn(x, u) } -> std::convertible_to<MeasJacobian<T, NY, NX>>;
};

// For nonlinear systems on embedded hardware. User provides discrete-time Jacobians:
//   state_fn: (x, u) -> StateJacobian{x_next, F, G}  (F = df/dx, G = df/dw)
//   meas_fn:  (x, u) -> MeasJacobian{y_pred, H, M}   (H = dh/dx, M = dh/dv)
template<size_t NX, size_t NU, size_t NY, typename T = double>
struct ExtendedKalmanFilter {
    constexpr ExtendedKalmanFilter() = default;

    constexpr ExtendedKalmanFilter(
        const ColVec<NX, T>&     x0,
        const Matrix<NX, NX, T>& P0,
        const Matrix<NX, NX, T>& Q_
    ) : x(x0), P(P0), Q(Q_) {}

    // Type conversion constructor
    template<typename U>
    constexpr ExtendedKalmanFilter(const ExtendedKalmanFilter<NX, NU, NY, U>& other)
        : x(other.state()), P(other.covariance()), Q(other.process_noise_covariance()), innov(other.innovation()) {}

    // Predict: x[k+1|k] = f(x[k|k], u[k]), P[k+1|k] = F*P*F' + G*Q*G'
    //   state_fn: callable (x, u) -> StateJacobian{x_pred, F, G}
    template<typename StateFn>
        requires EKFStateFn<StateFn, T, NX, NU>
    constexpr void predict(StateFn&& state_fn, const ColVec<NU, T>& u = ColVec<NU, T>{}) {
        StateJacobian<T, NX> sj = state_fn(x, u);
        x = sj.x_pred;
        P = sj.F * P * sj.F.transpose() + sj.G * Q * sj.G.transpose();
    }

    // Measurement update: returns false if innovation covariance is singular
    //   meas_fn: callable (x, u) -> MeasJacobian{y_pred, H, M}
    //   R:       measurement noise covariance (must match the sensor model in meas_fn)
    template<typename MeasFn>
        requires EKFMeasFn<MeasFn, T, NX, NU, NY>
    constexpr bool update(MeasFn&& meas_fn, const ColVec<NY, T>& y, const Matrix<NY, NY, T>& R, const ColVec<NU, T>& u = ColVec<NU, T>{}) {
        MeasJacobian<T, NY, NX> mj = meas_fn(x, u);
        const auto              Ht = mj.H.transpose();
        innov = y - mj.y_pred;
        const Matrix<NY, NY, T> S = mj.H * P * Ht + mj.M * R * mj.M.transpose();

        // K = PHᵀS⁻¹ → solve S Kᵀ = H P via Cholesky (S is symmetric positive definite)
        const auto K_opt = mat::cholesky_solve(S, mj.H * P);
        if (!K_opt) {
            return false;
        }

        const Matrix<NX, NY, T> K = K_opt.value().transpose();
        x = x + K * innov;

        // Joseph form
        const auto I_KH = Matrix<NX, NX, T>::identity() - K * mj.H;
        const auto KM = K * mj.M;
        P = I_KH * P * I_KH.transpose() + KM * R * KM.transpose();
        return true;
    }

    // Accessors
    [[nodiscard]] constexpr const auto& innovation() const { return innov; }
    [[nodiscard]] constexpr const auto& state() const { return x; }
    [[nodiscard]] constexpr const auto& covariance() const { return P; }
    [[nodiscard]] constexpr const auto& process_noise_covariance() const { return Q; }

private:
    ColVec<NX, T>     x{};
    Matrix<NX, NX, T> P{};
    Matrix<NX, NX, T> Q{};
    ColVec<NY, T>     innov{};
};

} // namespace wetmelon::control