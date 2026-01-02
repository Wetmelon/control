#pragma once

#include <cmath>
#include <concepts>

#include "matrix.hpp"
#include "state_space.hpp"

// ---------------------------------------------------------------------------
// Jacobian carrier types - explicit structs avoid tuples and document expected returns
// ---------------------------------------------------------------------------

// EKF state prediction result: x_next = f(x, u), with Jacobians F = ∂f/∂x, G = ∂f/∂w
template<typename T, size_t NX>
struct StateJacobian {
    ColVec<NX, T>     x_pred{};                         // Predicted state
    Matrix<NX, NX, T> F{};                              // State transition Jacobian ∂f/∂x
    Matrix<NX, NX, T> G{Matrix<NX, NX, T>::identity()}; // Process noise Jacobian ∂f/∂w
};

// EKF/ESKF measurement result: z_pred = h(x), with Jacobians H = ∂h/∂x, M = ∂h/∂v
template<typename T, size_t NY, size_t NX>
struct MeasJacobian {
    ColVec<NY, T>     z_pred{};                         // Predicted measurement
    Matrix<NY, NX, T> H{};                              // Measurement Jacobian ∂h/∂x
    Matrix<NY, NY, T> M{Matrix<NY, NY, T>::identity()}; // Measurement noise Jacobian ∂h/∂v
};

// ESKF error-state prediction result: Jacobians only (nominal state updated externally)
template<typename T, size_t NDX>
struct ErrorStateJacobian {
    Matrix<NDX, NDX, T> F{}; // Error state transition Jacobian ∂(δx_next)/∂(δx)
    Matrix<NDX, NDX, T> G{}; // Process noise Jacobian (maps Q to error state)
};

// ---------------------------------------------------------------------------
// Concepts for callable signatures - zero overhead, compile-time checked
// ---------------------------------------------------------------------------

// EKF state function: (x, u) -> StateJacobian
template<typename Fn, typename T, size_t NX, size_t NU>
concept EKFStateFn = requires(Fn&& fn, const ColVec<NX, T>& x, const ColVec<NU, T>& u) {
    { fn(x, u) } -> std::convertible_to<StateJacobian<T, NX>>;
};

// EKF measurement function: (x, u) -> MeasJacobian
template<typename Fn, typename T, size_t NX, size_t NU, size_t NY>
concept EKFMeasFn = requires(Fn&& fn, const ColVec<NX, T>& x, const ColVec<NU, T>& u) {
    { fn(x, u) } -> std::convertible_to<MeasJacobian<T, NY, NX>>;
};

// ESKF predict function: (dt) -> ErrorStateJacobian
template<typename Fn, typename T, size_t NDX>
concept ESKFPredictFn = requires(Fn&& fn, T dt) {
    { fn(dt) } -> std::convertible_to<ErrorStateJacobian<T, NDX>>;
};

// ESKF measurement function: () -> MeasJacobian (no state arg - user captures nominal state)
template<typename Fn, typename T, size_t NDX, size_t NY>
concept ESKFMeasFn = requires(Fn&& fn) {
    { fn() } -> std::convertible_to<MeasJacobian<T, NY, NDX>>;
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
        requires EKFStateFn<StateFn, T, NX, NU>
    constexpr void predict(StateFn&& f, const ColVec<NU, T>& u = ColVec<NU, T>{}) {
        StateJacobian<T, NX> sj = f(x, u);
        x = sj.x_pred;
        P = sj.F * P * sj.F.transpose() + sj.G * Q * sj.G.transpose();
    }

    // Measurement update: returns false if innovation covariance is singular
    template<typename MeasFn>
        requires EKFMeasFn<MeasFn, T, NX, NU, NY>
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

// ---------------------------------------------------------------------------
// Error-State Kalman Filter (ESKF) - Discrete-time
// ---------------------------------------------------------------------------
// Indirect KF for attitude estimation. Estimates small errors δx rather than full state.
// Perfect for IMU fusion: propagates quaternion via gyro integration, corrects with accel/mag.
//
// Error state: δx = [δθ (3 dof), δb_g (3 dof), δb_a (3 dof), ...]
// Nominal state: q_nom (quaternion), b_g_nom, b_a_nom
//
// User provides:
//   f_nominal: (q_nom, b_g, b_a, gyro_meas, accel_meas, dt) -> (q_new, error_jacobian F, noise_jacobian G)
//   h_meas:    (q_nom, accel_meas) -> (accel_predicted, H_matrix, M_matrix)
template<size_t NDX, size_t NY, typename T = double>
struct ErrorStateKalmanFilter {
    // Error state dimension typically: 3 (attitude) + 3 (gyro bias) + 3 (accel bias) = 9
    // Nominal state (quaternion + biases) is tracked separately and not part of the KF

    constexpr ErrorStateKalmanFilter() = default;

    constexpr ErrorStateKalmanFilter(
        const Matrix<NDX, NDX, T>& P0,
        const Matrix<NDX, NDX, T>& Q_,
        const Matrix<NY, NY, T>&   R_
    ) : P(P0), Q(Q_), R(R_), delta_x(ColVec<NDX, T>{}) {}

    // Type conversion constructor
    template<typename U>
    constexpr ErrorStateKalmanFilter(const ErrorStateKalmanFilter<NDX, NY, U>& other)
        : P(other.covariance()),
          Q(other.process_noise_covariance()),
          R(other.measurement_noise_covariance()),
          delta_x(other.error_state()),
          y(other.innovation()) {}

    // Predict: Propagate error covariance (nominal state updated externally by user)
    // User provides: (dt) -> ErrorStateJacobian{F, G}
    //   F = error state transition Jacobian (∂δx_next/∂δx)
    //   G = process noise Jacobian (maps Q to error covariance)
    // P[k+1|k] = F * P[k|k] * F' + G * Q * G'
    template<typename PredictFn>
        requires ESKFPredictFn<PredictFn, T, NDX>
    constexpr void predict(PredictFn&& propagate_nominal, const T dt) {
        ErrorStateJacobian<T, NDX> ej = propagate_nominal(dt);
        P = ej.F * P * ej.F.transpose() + ej.G * Q * ej.G.transpose();

        // Error state decays naturally (doesn't accumulate without measurement correction)
        // δx[k+1|k] ≈ 0 (optional: can decay via F matrix if modeling error dynamics)
    }

    // Measurement update: Correct error state from innovation
    // User provides: () -> MeasJacobian{z_pred, H, M}
    //   z_pred = h(nominal_state) - predicted measurement
    //   H = measurement Jacobian (∂h/∂δx)
    //   M = measurement noise Jacobian (∂h/∂v, usually identity)
    // P update: P = (I - K*H) * P * (I - K*H)' + K*M*R*M'*K'  (Joseph form)
    template<typename MeasFn>
        requires ESKFMeasFn<MeasFn, T, NDX, NY>
    constexpr bool update(MeasFn&& h, const ColVec<NY, T>& z_meas) {
        MeasJacobian<T, NY, NDX> mj = h();

        y = z_meas - mj.z_pred; // Innovation

        const auto              Ht = mj.H.transpose();
        const Matrix<NY, NY, T> S = mj.H * P * Ht + mj.M * R * mj.M.transpose();

        const auto S_inv = S.inverse();
        if (!S_inv)
            return false;

        // Kalman gain
        const Matrix K = P * Ht * S_inv.value();

        // Update error state: δx = K * δz
        delta_x = ColVec(K * y);

        // Covariance update (Joseph form for numerical stability)
        const auto I_KH = Matrix<NDX, NDX, T>::identity() - K * mj.H;
        const auto KM = K * mj.M;
        P = I_KH * P * I_KH.transpose() + KM * R * KM.transpose();

        return true;
    }

    // Reset error state after applying corrections to nominal state
    // Call this after you've updated the nominal quaternion/biases with delta_x
    constexpr void reset_error_state() {
        delta_x = ColVec<NDX, T>{};
    }

    // Accessors
    [[nodiscard]] constexpr const auto& error_state() const { return delta_x; }
    [[nodiscard]] constexpr const auto& covariance() const { return P; }
    [[nodiscard]] constexpr const auto& process_noise_covariance() const { return Q; }
    [[nodiscard]] constexpr const auto& measurement_noise_covariance() const { return R; }
    [[nodiscard]] constexpr const auto& innovation() const { return y; }

    // Setters for runtime tuning
    constexpr void set_process_noise_covariance(const Matrix<NDX, NDX, T>& Q_new) { Q = Q_new; }
    constexpr void set_measurement_noise_covariance(const Matrix<NY, NY, T>& R_new) { R = R_new; }
    constexpr void set_covariance(const Matrix<NDX, NDX, T>& P_new) { P = P_new; }

private:
    Matrix<NDX, NDX, T> P{};       // Error covariance
    Matrix<NDX, NDX, T> Q{};       // Process noise covariance
    Matrix<NY, NY, T>   R{};       // Measurement noise covariance
    ColVec<NDX, T>      delta_x{}; // Error state δx
    ColVec<NY, T>       y{};       // Innovation
};