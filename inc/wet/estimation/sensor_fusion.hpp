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
#include "wet/math/geometry.hpp"

namespace wet {

namespace detail {

/// Negative skew-symmetric (cross-product) matrix: skew_neg(v)·u = -(v × u) = u × v.
/// Used for the small-angle attitude Jacobian -[v]× and the error-reset I - [δθ]×.
template<typename T>
[[nodiscard]] constexpr Matrix<3, 3, T> skew_neg(const Vec3<T>& v) {
    Matrix<3, 3, T> m = Matrix<3, 3, T>::zeros();
    m(0, 1) = v[2];
    m(0, 2) = -v[1];
    m(1, 0) = -v[2];
    m(1, 2) = v[0];
    m(2, 0) = v[1];
    m(2, 1) = -v[0];
    return m;
}

} // namespace detail

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

        // Body gravity for attitude q (world→body) is R(q)·[0,0,-g] = g·[…],
        // giving a_norm = [-sinθ, sinφ·cosθ, -cosφ·cosθ]. Invert for roll/pitch.
        // The pitch denominator is the horizontal (y,z) magnitude, not the full
        // vector norm (which is 1 after normalization and would collapse the tilt).
        T roll = wet::atan2(a_norm[1], -a_norm[2]);
        T pitch = wet::atan2(-a_norm[0], wet::hypot(a_norm[1], a_norm[2]));

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

        // Field-alignment error in the body frame (gravity + magnetometer), the
        // gradient the filter descends. Gravity: measured field crossed with the
        // estimated world-up axis in body, R(q)·[0,0,1] (antiparallel at the fixed
        // point). Magnetometer: refer the field to the world frame, flatten to the
        // horizontal reference (north = +Y), rotate the estimate back to body, and
        // cross (estimate × meas) — opposite order to gravity since the mag
        // estimate is parallel to the measurement at the fixed point.
        const Vec3<T> v = q.rotate(Vec3<T>{T{0}, T{0}, T{1}});
        Vec3<T>       e = a_norm.cross(v);

        const Vec3<T> h = q.conjugate().rotate(m_norm);
        const T       b_horizontal = wet::hypot(h[0], h[1]);
        const Vec3<T> w = q.rotate(Vec3<T>{T{0}, b_horizontal, h[2]});
        e += w.cross(m_norm);

        // q̇_gyro = ½ q ⊗ ω
        const Quaternion<T> q_dot_gyro = T{0.5} * q * Quaternion<T>{T{0}, gyro[0], gyro[1], gyro[2]};

        // Gradient-descent correction: ½ q ⊗ [0, e], normalized so the step size
        // is fixed at β regardless of error magnitude (the Madgwick characteristic).
        // +e drives the error toward zero, so the term is added to the gyro rate.
        Quaternion<T> grad = T{0.5} * q * Quaternion<T>{T{0}, e[0], e[1], e[2]};
        const T       grad_norm = grad.norm();
        if (grad_norm > T{1e-9}) {
            grad = grad / grad_norm;
        }

        const Quaternion<T> q_dot = Quaternion<T>(q_dot_gyro + beta * grad);
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

        // Gravity (accelerometer) correction: measured field crossed with the
        // estimated world-up axis in body, R(q)·[0,0,1]. (At the fixed point
        // a_norm is antiparallel to this axis, which is the stable equilibrium.)
        Vec3<T> v = q.rotate(Vec3<T>{T{0}, T{0}, T{1}});
        Vec3<T> error = a_norm.cross(v);

        // Magnetometer correction (yaw): refer the measured field to the world
        // frame, flatten it to the horizontal reference direction (north = +Y,
        // matching this file's world frame), rotate that estimate back to body,
        // and cross it with the measurement. The estimate is parallel to the
        // measurement at the fixed point, so the cross order is (estimate × meas)
        // — opposite to gravity — to keep the same (stable) sign. Without this
        // term yaw drift is uncorrected.
        const Vec3<T> h = q.conjugate().rotate(m_norm);
        const T       b_horizontal = wet::hypot(h[0], h[1]);
        const Vec3<T> w = q.rotate(Vec3<T>{T{0}, b_horizontal, h[2]});
        error += w.cross(m_norm);

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

    // Measurement Jacobian wrt the attitude error: y = R(δq)·v_body, and with the
    // left-multiplicative world-frame error used in the injection step below,
    // ∂y/∂δθ = -[v_body]×  (full 3×3, all three attitude columns). The bias
    // columns (3..5) are zero. A truncated 3×2 block leaves yaw unobservable.
    const Vec3<T>   g_body = q_nom.rotate(g_vec);
    Matrix<3, 3, T> H_accel = detail::skew_neg(g_body);
    H.template block<3, 3>(idx, 0) = H_accel;
    idx += 3;

    // Magnetometer measurement
    const Vec3<T> mag_pred = q_nom.rotate(m_vec);
    y_pred.template block<3, 1>(idx, 0) = mag_pred;
    y_meas.template block<3, 1>(idx, 0) = mag_meas;

    const Vec3<T>   m_body = q_nom.rotate(m_vec);
    Matrix<3, 3, T> H_mag = detail::skew_neg(m_body);
    H.template block<3, 3>(idx, 0) = H_mag;

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
        // Attitude block: G_att ≈ I - [δθ]× = I + skew_neg(δθ) (small-angle).
        G_reset.template block<3, 3>(0, 0) = Matrix<3, 3, T>::identity() + detail::skew_neg(delta_theta);
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
    constexpr ESKFOrientationFilter(T gyro_noise = T{0.003}, T accel_noise = T{0.03}, T mag_noise = T{0.3}, T gyro_bias_rw = T{0.0001}, T dt = T{0.01})
        : eskf(design::eskf_design(gyro_noise, accel_noise, mag_noise, gyro_bias_rw, dt)),
          q_nom(Quaternion<T>::identity()),
          b_g_nom{T{0}, T{0}, T{0}} {}

    constexpr void update(const Vec3<T>& accel, const Vec3<T>& gyro, const Vec3<T>& mag, T dt, const Vec3<T>& g_vec = {T{0}, T{0}, T{-9.81}}, const Vec3<T>& m_vec = {T{0}, T{1}, T{0}}) {
        eskf_update_imu(eskf, q_nom, b_g_nom, accel, gyro, mag, dt, g_vec, m_vec);
    }

    [[nodiscard]] constexpr const Quaternion<T>& orientation() const { return q_nom; }
    [[nodiscard]] constexpr const Vec3<T>&       gyro_bias() const { return b_g_nom; }
};

} // namespace wet
