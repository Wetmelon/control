#pragma once

/**
 * @file sensor_fusion.hpp
 * @brief Orientation estimation filters for IMU sensor fusion
 *
 * Provides several filters of increasing sophistication:
 *   - ComplementaryFilter: simple α-blend of gyro and accelerometer
 *   - MadgwickFilter: gradient-descent AHRS (accel + mag)
 *   - MahonyFilter: nonlinear complementary filter with PI correction
 *   - ESKFOrientationFilter: full error-state Kalman filter
 *
 * @see Madgwick, "An efficient orientation filter" (2010)
 * @see Mahony et al., "Nonlinear Complementary Filters on SO(3)" (2008)
 * @see Solà et al., "Quaternion kinematics for the error-state Kalman filter" (2017)
 */

#include "eskf.hpp"
#include "wet/geometry.hpp"

namespace wet {

/**
 * @brief Simple complementary filter for orientation estimation
 *
 * Blends gyroscope integration (high-frequency) with accelerometer tilt
 * (low-frequency) using a tunable alpha parameter.
 *
 * @tparam T Scalar type (default: float)
 */
template<typename T = float>
class ComplementaryFilter {
    T             alpha;
    Quaternion<T> q;

public:
    constexpr ComplementaryFilter(T alpha = T{0.98}) : alpha(alpha), q(Quaternion<T>::identity()) {}

    constexpr void update(const Vec3<T>& accel, const Vec3<T>& gyro, T dt) {
        Vec3<T> a_norm = accel.normalized();

        T roll = wet::atan2(-a_norm[1], -a_norm[2]);
        T pitch = wet::atan2(a_norm[0], a_norm.norm());

        Euler<T, EulerOrder::ZYX> euler(T{0}, pitch, roll);

        Quaternion<T> q_accel = Quaternion<T>::from_euler(euler);

        q = q.integrate_body_rates(gyro, dt);
        q = Quaternion<T>::slerp(q, q_accel, T{1} - alpha);
        q = q.normalized();
    }

    [[nodiscard]] constexpr const Quaternion<T>& orientation() const { return q; }
};

/**
 * @brief Madgwick gradient-descent AHRS filter
 *
 * Fuses gyroscope, accelerometer, and magnetometer using a gradient-descent
 * algorithm to minimize orientation error.
 *
 * @see Madgwick, "An efficient orientation filter for inertial and
 *      inertial/magnetic sensor arrays" (2010)
 *
 * @tparam T Scalar type (default: float)
 */
template<typename T = float>
class MadgwickFilter {
    T             beta;
    Quaternion<T> q;

public:
    constexpr MadgwickFilter(T beta = T{0.1}) : beta(beta), q(Quaternion<T>::identity()) {}

    constexpr void update(const Vec3<T>& accel, const Vec3<T>& gyro, const Vec3<T>& mag, T dt) {
        Vec3<T> a_norm = accel.normalized();
        Vec3<T> m_norm = mag.normalized();

        Vec3<T> f = q.rotate(a_norm.cross(Vec3<T>{T{0}, T{0}, T{1}}));
        Vec3<T> h = q.rotate(m_norm);
        Vec3<T> J = f + h;

        T J_norm = J.norm();
        if (J_norm > T{1e-9}) {
            J = J / J_norm;
        }

        // q̇_gyro = ½ q ⊗ ω
        Quaternion<T> omega_quat{T{0}, gyro[0], gyro[1], gyro[2]};
        Quaternion<T> q_dot_gyro = T{0.5} * q * omega_quat;

        // q̇_error = β · ∇f / ‖∇f‖
        Quaternion<T> J_quat{T{0}, J[0], J[1], J[2]};
        Quaternion<T> q_dot_error = beta * J_quat;

        Quaternion<T> q_dot = Quaternion<T>(q_dot_gyro - q_dot_error);

        q = q + q_dot * dt;
        q = q.normalized();
    }

    [[nodiscard]] constexpr const Quaternion<T>& orientation() const { return q; }
};

/**
 * @brief Mahony nonlinear complementary filter with PI correction
 *
 * Uses a proportional-integral controller on the orientation error
 * to correct gyroscope drift.
 *
 * @see Mahony et al., "Nonlinear Complementary Filters on the Special
 *      Orthogonal Group" (2008)
 *
 * @tparam T Scalar type (default: float)
 */
template<typename T = float>
class MahonyFilter {
    T             Kp, Ki;
    Vec3<T>       integral_error;
    Quaternion<T> q;

public:
    constexpr MahonyFilter(T Kp = T{0.5}, T Ki = T{0}) : Kp(Kp), Ki(Ki), q(Quaternion<T>::identity()) {}

    constexpr void update(const Vec3<T>& accel, const Vec3<T>& gyro, const Vec3<T>& mag, T dt) {
        Vec3<T> a_norm = accel.normalized();
        Vec3<T> m_norm = mag.normalized();

        Vec3<T> v = q.rotate(Vec3<T>{T{0}, T{0}, T{1}});
        Vec3<T> error = a_norm.cross(v);

        integral_error += error * dt;
        Vec3<T> feedback = Kp * error + Ki * integral_error;

        Vec3<T> omega = gyro + feedback;
        q = q.integrate_body_rates(omega, dt);
        q = q.normalized();
    }

    [[nodiscard]] constexpr const Quaternion<T>& orientation() const { return q; }
};

/**
 * @brief Full ESKF predict+update+inject cycle for 6-axis IMU fusion
 *
 * Performs one complete ESKF iteration:
 *   1. Nominal state propagation (gyro integration)
 *   2. Error covariance prediction
 *   3. Measurement update (accelerometer + magnetometer)
 *   4. Error injection into nominal state and covariance reset
 *
 * @see Solà et al., "Quaternion kinematics for the error-state Kalman filter" (2017), §5–6
 *
 * @param eskf       Error-state Kalman filter instance
 * @param q_nom      Nominal quaternion (modified in place)
 * @param b_g_nom    Nominal gyro bias estimate (modified in place)
 * @param accel_meas Accelerometer measurement [m/s²]
 * @param gyro_meas  Gyroscope measurement [rad/s]
 * @param mag_meas   Magnetometer measurement [normalized or raw]
 * @param dt         Timestep [s]
 * @param g_vec      Gravity vector in world frame (default: [0, 0, −9.81])
 * @param m_vec      Magnetic field reference in world frame (default: [0, 1, 0])
 */
template<size_t NDX, size_t NY, typename T>
constexpr void eskf_update_imu(
    ErrorStateKalmanFilter<NDX, NY, T>& eskf,
    Quaternion<T>&                      q_nom,
    Vec3<T>&                            b_g_nom,
    const Vec3<T>&                      accel_meas,
    const Vec3<T>&                      gyro_meas,
    const Vec3<T>&                      mag_meas,
    T                                   dt,
    const Vec3<T>&                      g_vec = Vec3<T>{T{0}, T{0}, T{-9.81}},
    const Vec3<T>&                      m_vec = Vec3<T>{T{0}, T{1}, T{0}}
) {
    // 1. Predict step: Propagate nominal state with gyro integration
    Vec3<T> omega_corrected = gyro_meas - b_g_nom;
    q_nom = q_nom.integrate_body_rates(omega_corrected, dt);

    // Predict Jacobians: F and G for error dynamics (Sola's continuous-time error model)
    Matrix<NDX, NDX, T> F = Matrix<NDX, NDX, T>::identity();
    // Attitude error dynamics: δθ_dot = -δb_g (gyro bias coupling)
    F(0, 3) = -dt;
    F(1, 4) = -dt;
    F(2, 5) = -dt;
    // Bias dynamics: constant (F is identity for biases)

    Matrix<NDX, NDX, T> G = Matrix<NDX, NDX, T>::identity(); // Noise input matrix (identity for additive noise)

    // Error state propagation
    auto predict_fn = [F, G](T) -> ErrorStateJacobian<T, NDX> {
        return {F, G};
    };

    eskf.predict(predict_fn, dt);

    // 2. Update step: Correct with measurements (standard Kalman update)
    ColVec<NY, T>      y_pred{};
    ColVec<NY, T>      y_meas{};
    Matrix<NY, NDX, T> H = Matrix<NY, NDX, T>::zeros();

    size_t idx = 0;

    // Accelerometer measurement
    Vec3<T> accel_pred = q_nom.rotate(g_vec);
    y_pred.template block<3, 1>(idx, 0) = accel_pred;
    y_meas.template block<3, 1>(idx, 0) = accel_meas;

    const Vec3<T>   g_body = q_nom.rotate(g_vec);
    Matrix<3, 2, T> H_accel{};
    H_accel(0, 0) = g_body[2];
    H_accel(0, 1) = -g_body[1];
    H_accel(1, 0) = -g_body[2];
    H_accel(1, 1) = g_body[0];
    H_accel(2, 0) = g_body[1];
    H_accel(2, 1) = -g_body[0];
    H.template block<3, 2>(idx, 0) = H_accel;
    idx += 3;

    // Magnetometer measurement
    const Vec3<T> mag_pred = q_nom.rotate(m_vec);
    y_pred.template block<3, 1>(idx, 0) = mag_pred;
    y_meas.template block<3, 1>(idx, 0) = mag_meas;

    const Vec3<T>   m_body = q_nom.rotate(m_vec);
    Matrix<3, 2, T> H_mag{};
    H_mag(0, 0) = m_body[2];
    H_mag(0, 1) = -m_body[1];
    H_mag(1, 0) = -m_body[2];
    H_mag(1, 1) = m_body[0];
    H_mag(2, 0) = m_body[1];
    H_mag(2, 1) = -m_body[0];
    H.template block<3, 2>(idx, 0) = H_mag;

    Matrix<NY, NY, T> M = Matrix<NY, NY, T>::identity();

    auto meas_fn = [y_pred, H, M]() -> MeasJacobian<T, NY, NDX> {
        return {y_pred, H, M};
    };

    eskf.update(meas_fn, y_meas);

    // 3. Inject errors into nominal state (Sola's reset step)
    const auto    delta_x = eskf.error_state();
    const Vec3<T> delta_theta{delta_x[0], delta_x[1], delta_x[2]};

    // Update quaternion with small-angle approximation (Sola: q_nom^+ = exp(δθ) ⊗ q_nom^-)
    const T delta_norm = delta_theta.norm();
    if (delta_norm > T{1e-9}) {
        const T             half = delta_norm * T{0.5};
        const T             s = wet::sin(half) / delta_norm;
        const Quaternion<T> delta_q{wet::cos(half), delta_theta[0] * s, delta_theta[1] * s, delta_theta[2] * s};
        q_nom = delta_q * q_nom;
        q_nom = q_nom.normalized();
    }

    // Update gyro biases (linear injection)
    const Vec3<T> delta_b_g{delta_x[3], delta_x[4], delta_x[5]};
    b_g_nom = b_g_nom + delta_b_g;

    // 4. Reset error state with proper Jacobian G (Sola: P^+ = G P^- G^T)
    // G = ∂(nominal^+)/∂(delta_x^-), block-diagonal for attitude and biases
    Matrix<NDX, NDX, T> G_reset = Matrix<NDX, NDX, T>::identity();
    if (delta_norm > T{1e-9}) {
        // Attitude block: G_att ≈ I - [δθ]× (skew-symmetric, small-angle approximation)
        Matrix<3, 3, T> skew = Matrix<3, 3, T>::zeros();
        skew(0, 1) = -delta_theta[2];
        skew(0, 2) = delta_theta[1];
        skew(1, 0) = delta_theta[2];
        skew(1, 2) = -delta_theta[0];
        skew(2, 0) = -delta_theta[1];
        skew(2, 1) = delta_theta[0];
        G_reset.template block<3, 3>(0, 0) = Matrix<3, 3, T>::identity() - skew;
    }
    // Bias block: G_bias = I (linear)

    eskf.reset_error_state(G_reset);
}

/**
 * @brief ESKF-based orientation estimator (convenience wrapper)
 *
 * Wraps ErrorStateKalmanFilter with IMU-specific predict/update logic.
 * Estimates attitude quaternion and gyroscope bias from accelerometer,
 * gyroscope, and magnetometer measurements.
 *
 * @see Solà et al., "Quaternion kinematics for the error-state Kalman filter" (2017)
 *
 * @tparam T Scalar type (default: float)
 */
template<typename T = float>
class ESKFOrientationFilter {
    ErrorStateKalmanFilter<6, 6, T> eskf;
    Quaternion<T>                   q_nom;
    Vec3<T>                         b_g_nom;

public:
    constexpr ESKFOrientationFilter(T gyro_noise = T{0.003}, T accel_noise = T{0.03}, T mag_noise = T{0.3}, T gyro_bias_rw = T{0.0001}, T dt = T{0.01}) {
        const auto eskf_result = design::eskf_design(gyro_noise, accel_noise, mag_noise, gyro_bias_rw, dt);
        eskf = ErrorStateKalmanFilter<6, 6, T>{eskf_result};
        q_nom = Quaternion<T>::identity();
        b_g_nom = {T{0}, T{0}, T{0}};
    }

    constexpr void update(const Vec3<T>& accel, const Vec3<T>& gyro, const Vec3<T>& mag, T dt, const Vec3<T>& g_vec = {T{0}, T{0}, T{-9.81}}, const Vec3<T>& m_vec = {T{0}, T{1}, T{0}}) {
        eskf_update_imu(eskf, q_nom, b_g_nom, accel, gyro, mag, dt, g_vec, m_vec);
    }

    [[nodiscard]] constexpr const Quaternion<T>& orientation() const { return q_nom; }
    [[nodiscard]] constexpr const Vec3<T>&       gyro_bias() const { return b_g_nom; }
};

} // namespace wet
