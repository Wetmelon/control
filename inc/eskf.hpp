#pragma once

#include <cstddef>

#include "ekf.hpp"
#include "matrix.hpp"

namespace wetmelon::control {
namespace online {
/**
 * @struct ESKFResult
 * @brief Runtime Error-State Kalman Filter design result (online namespace)
 */
template<size_t NDX, size_t NY, typename T>
struct ESKFResult {
    Matrix<NDX, NDX, T> Q{};
    Matrix<NY, NY, T>   R{};
    Matrix<NDX, NDX, T> P0{};
    bool                success{false};

    template<typename U>
    [[nodiscard]] constexpr auto as() const {
        return ESKFResult<NDX, NY, U>{
            Q.template as<U>(),
            R.template as<U>(),
            P0.template as<U>(),
            success
        };
    }
};

/**
 * @brief Error-State Kalman Filter design from IMU sensor specifications (runtime version)
 *
 * Computes process and measurement noise covariances for ESKF from sensor noise densities.
 * Assumes 9D error state: [attitude errors (3), gyro biases (3), accel biases (3)]
 * and 9D measurements: [accel (3), gyro (3), mag (3)] - though gyro not used in update.
 *
 * @param gyro_noise_density   Gyroscope noise density [rad/s/sqrt(Hz)]
 * @param accel_noise_density  Accelerometer noise density [m/s^2/sqrt(Hz)]
 * @param mag_noise_density    Magnetometer noise density [unit/sqrt(Hz)]
 * @param gyro_bias_rw         Gyroscope bias random walk [rad/s^1.5]
 * @param accel_bias_rw        Accelerometer bias random walk [m/s^2.5]
 * @param mag_bias_rw          Magnetometer bias random walk [unit/s^1.5]
 * @param dt                   Sampling time [s]
 * @param initial_attitude_uncertainty  Initial attitude uncertainty [rad]
 * @param initial_bias_uncertainty      Initial bias uncertainty [rad/s or m/s^2]
 *
 * @return ESKFResult with computed covariances
 */
template<typename T = double>
[[nodiscard]] constexpr ESKFResult<9, 9, T> eskf_design(
    T gyro_noise_density,
    T accel_noise_density,
    T mag_noise_density,
    T gyro_bias_rw,
    T accel_bias_rw,
    T mag_bias_rw,
    T dt,
    T initial_attitude_uncertainty = T{0.1},
    T initial_bias_uncertainty = T{0.01}
) {
    ESKFResult<9, 9, T> result{};

    // Process noise covariance Q (discrete-time)
    T gyro_var = gyro_noise_density * gyro_noise_density * dt;
    T gyro_bias_var = gyro_bias_rw * gyro_bias_rw * dt;
    T accel_bias_var = accel_bias_rw * accel_bias_rw * dt;
    T mag_bias_var = mag_bias_rw * mag_bias_rw * dt;

    result.Q = Matrix<9, 9, T>::zeros();
    result.Q(0, 0) = gyro_var;
    result.Q(1, 1) = gyro_var;
    result.Q(2, 2) = gyro_var;
    result.Q(3, 3) = gyro_bias_var;
    result.Q(4, 4) = gyro_bias_var;
    result.Q(5, 5) = gyro_bias_var;
    result.Q(6, 6) = accel_bias_var;
    result.Q(7, 7) = accel_bias_var;
    result.Q(8, 8) = accel_bias_var;

    // Measurement noise covariance R
    T accel_var = accel_noise_density * accel_noise_density;
    T mag_var = mag_noise_density * mag_noise_density;

    result.R = Matrix<9, 9, T>::zeros();
    result.R(0, 0) = accel_var;
    result.R(1, 1) = accel_var;
    result.R(2, 2) = accel_var;
    result.R(3, 3) = gyro_var;
    result.R(4, 4) = gyro_var;
    result.R(5, 5) = gyro_var;
    result.R(6, 6) = mag_var;
    result.R(7, 7) = mag_var;
    result.R(8, 8) = mag_var;

    // Initial covariance P0
    result.P0 = Matrix<9, 9, T>::zeros();
    result.P0(0, 0) = initial_attitude_uncertainty * initial_attitude_uncertainty;
    result.P0(1, 1) = initial_attitude_uncertainty * initial_attitude_uncertainty;
    result.P0(2, 2) = initial_attitude_uncertainty * initial_attitude_uncertainty;
    for (size_t i = 3; i < 9; ++i) {
        result.P0(i, i) = initial_bias_uncertainty * initial_bias_uncertainty;
    }

    result.success = true;
    return result;
}

} // namespace online

namespace design {
/**
 * @struct ESKFResult
 * @brief Error-State Kalman Filter design result
 *
 * Contains covariance matrices for ESKF initialization and tuning.
 */
template<size_t NDX, size_t NY, typename T = double>
struct ESKFResult {
    Matrix<NDX, NDX, T> Q{};            //!< Process noise covariance
    Matrix<NY, NY, T>   R{};            //!< Measurement noise covariance
    Matrix<NDX, NDX, T> P0{};           //!< Initial error covariance
    bool                success{false}; //!< Indicates design success

    template<typename U>
    [[nodiscard]] consteval auto as() const {
        return ESKFResult<NDX, NY, U>{
            Q.template as<U>(),
            R.template as<U>(),
            P0.template as<U>(),
            success
        };
    }
};

/**
 * @brief Error-State Kalman Filter design from IMU sensor specifications
 *
 * Computes process and measurement noise covariances for ESKF from sensor noise densities.
 * Assumes 6D error state: [attitude errors (3), gyro biases (3)]
 * and 6D measurements: [accel (3), mag (3)].
 *
 * @param gyro_noise_density   Gyroscope noise density [rad/s/sqrt(Hz)]
 * @param accel_noise_density  Accelerometer noise density [m/s^2/sqrt(Hz)]
 * @param mag_noise_density    Magnetometer noise density [unit/sqrt(Hz)]
 * @param gyro_bias_rw         Gyroscope bias random walk [rad/s^1.5]
 * @param dt                   Sampling time [s]
 * @param initial_attitude_uncertainty  Initial attitude uncertainty [rad]
 * @param initial_bias_uncertainty      Initial bias uncertainty [rad/s]
 *
 * @return ESKFResult with computed covariances
 */
template<size_t NDX = 6, size_t NY = 6, typename T = double>
[[nodiscard]] consteval ESKFResult<NDX, NY, T> eskf_design(
    T gyro_noise_density,
    T accel_noise_density,
    T mag_noise_density,
    T gyro_bias_rw,
    T dt,
    T initial_attitude_uncertainty = T{0.1},
    T initial_bias_uncertainty = T{0.01}
) {
    static_assert(NDX == 6 && NY == 6, "ESKF design currently supports NDX=6, NY=6 only");
    ESKFResult<NDX, NY, T> result{};

    // Process noise covariance Q (discrete-time)
    // Attitude errors: gyro noise integrated over dt
    T gyro_var = gyro_noise_density * gyro_noise_density * dt;
    // Bias errors: random walk integrated
    T gyro_bias_var = gyro_bias_rw * gyro_bias_rw * dt;

    result.Q = Matrix<NDX, NDX, T>::zeros();
    result.Q(0, 0) = gyro_var;
    result.Q(1, 1) = gyro_var;
    result.Q(2, 2) = gyro_var;
    result.Q(3, 3) = gyro_bias_var;
    result.Q(4, 4) = gyro_bias_var;
    result.Q(5, 5) = gyro_bias_var;

    // Measurement noise covariance R
    T accel_var = accel_noise_density * accel_noise_density;
    T mag_var = mag_noise_density * mag_noise_density;

    result.R = Matrix<NY, NY, T>::zeros();
    result.R(0, 0) = accel_var;
    result.R(1, 1) = accel_var;
    result.R(2, 2) = accel_var;
    result.R(3, 3) = mag_var;
    result.R(4, 4) = mag_var;
    result.R(5, 5) = mag_var;

    // Initial covariance P0
    result.P0 = Matrix<NDX, NDX, T>::zeros();
    result.P0(0, 0) = initial_attitude_uncertainty * initial_attitude_uncertainty;
    result.P0(1, 1) = initial_attitude_uncertainty * initial_attitude_uncertainty;
    result.P0(2, 2) = initial_attitude_uncertainty * initial_attitude_uncertainty;
    result.P0(3, 3) = initial_bias_uncertainty * initial_bias_uncertainty;
    result.P0(4, 4) = initial_bias_uncertainty * initial_bias_uncertainty;
    result.P0(5, 5) = initial_bias_uncertainty * initial_bias_uncertainty;

    result.success = true;
    return result;
}

} // namespace design

// ESKF error-state prediction result: Jacobians only (nominal state updated externally)
template<typename T, size_t NDX>
struct ErrorStateJacobian {
    Matrix<NDX, NDX, T> F{}; // Error state transition Jacobian ∂(δx_next)/∂(δx)
    Matrix<NDX, NDX, T> G{}; // Process noise Jacobian (maps Q to error state)
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

    consteval ErrorStateKalmanFilter(
        const design::ESKFResult<NDX, NY, T>& result
    ) : P(result.P0), Q(result.Q), R(result.R), delta_x(ColVec<NDX, T>{}) {}

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
    // Optionally provide a G matrix to adjust covariance (e.g., for attitude reset)
    constexpr void reset_error_state(const Matrix<NDX, NDX, T>& G = Matrix<NDX, NDX, T>::identity()) {
        delta_x = ColVec<NDX, T>{};
        P = G * P * G.transpose();
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

// CTAD deduction guide for ErrorStateKalmanFilter
template<typename T, size_t NDX, size_t NY>
ErrorStateKalmanFilter(Matrix<NDX, NDX, T>, Matrix<NDX, NDX, T>, Matrix<NY, NY, T>) -> ErrorStateKalmanFilter<NDX, NY, T>;

} // namespace wetmelon::control