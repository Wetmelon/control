#pragma once

#include "eskf.hpp"
#include "rotation.hpp"

namespace wetmelon::control {

// Simple complementary filter for orientation estimation
template<typename T = float>
class ComplementaryFilter {
    T             alpha; // blending factor (0-1, higher = more trust in gyro)
    Quaternion<T> q;

public:
    ComplementaryFilter(T alpha = 0.98f) : alpha(alpha), q(Quaternion<T>::identity()) {}

    void update(const Vec3<T>& accel, const Vec3<T>& gyro, T dt) {
        // Normalize accel for tilt
        Vec3<T> a_norm = accel.normalized();

        // Compute tilt from accel
        T roll = wet::atan2(-a_norm[1], -a_norm[2]);
        T pitch = wet::atan2(a_norm[0], a_norm.norm());

        Euler<T, EulerOrder::ZYX> euler(0.0f, pitch, roll);

        Quaternion<T> q_accel = Quaternion<T>::from_euler(euler);

        // Integrate gyro
        q = q.integrate_body_rates(gyro, dt);

        // Blend
        q = Quaternion<T>::slerp(q, q_accel, 1.0f - alpha);
        q = q.normalized();
    }

    const Quaternion<T>& getOrientation() const { return q; }
};

// Madgwick gradient descent filter
template<typename T = float>
class MadgwickFilter {
    T             beta; // gradient descent step size
    Quaternion<T> q;

public:
    MadgwickFilter(T beta = 0.1f) : beta(beta), q(Quaternion<T>::identity()) {}

    void update(const Vec3<T>& accel, const Vec3<T>& gyro, const Vec3<T>& mag, T dt) {
        // Madgwick gradient descent algorithm
        Vec3<T> a_norm = accel.normalized();
        Vec3<T> m_norm = mag.normalized();

        // Gradient descent to minimize error
        Vec3<T> f = q.rotate(a_norm.cross(Vec3<T>{0, 0, 1})); // gravity error
        Vec3<T> h = q.rotate(m_norm);                         // mag error
        Vec3<T> J = f + h;                                    // combined gradient

        // Normalize gradient for stability
        T J_norm = J.norm();
        if (J_norm > T{1e-9}) {
            J = J / J_norm;
        }

        // Compute quaternion derivatives
        // q̇_gyro = ½ q ⊗ ω
        Quaternion<T> omega_quat{T{0}, gyro[0], gyro[1], gyro[2]};
        Quaternion<T> q_dot_gyro = T{0.5} * q * omega_quat;

        // q̇_error = β * ∇f / ||∇f||
        Quaternion<T> J_quat{T{0}, J[0], J[1], J[2]};
        Quaternion<T> q_dot_error = beta * J_quat;

        // Combined derivative: q̇ = q̇_gyro - q̇_error
        Quaternion<T> q_dot = Quaternion<T>(q_dot_gyro - q_dot_error);

        // Integrate: q[k+1] = q[k] + q̇[k] * dt
        q = q + q_dot * dt;
        q = q.normalized();
    }

    const Quaternion<T>& getOrientation() const { return q; }
};

// Mahony PI controller filter
template<typename T = float>
class MahonyFilter {
    T             Kp, Ki;
    Vec3<T>       integral_error;
    Quaternion<T> q;

public:
    MahonyFilter(T Kp = 0.5f, T Ki = 0.0f) : Kp(Kp), Ki(Ki), q(Quaternion<T>::identity()) {}

    void update(const Vec3<T>& accel, const Vec3<T>& gyro, const Vec3<T>& mag, T dt) {
        // Simplified Mahony (full version includes PI feedback)
        Vec3<T> a_norm = accel.normalized();
        Vec3<T> m_norm = mag.normalized();

        // Compute errors
        Vec3<T> v = q.rotate(Vec3<T>{0, 0, 1}); // gravity in body
        Vec3<T> error = a_norm.cross(v);

        // PI feedback
        integral_error += error * dt;
        Vec3<T> feedback = Kp * error + Ki * integral_error;

        // Update gyro
        Vec3<T> omega = gyro + feedback;
        q = q.integrate_body_rates(omega, dt);
        q = q.normalized();
    }

    const Quaternion<T>& getOrientation() const { return q; }
};

// Implementation of eskf_update_imu function
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
    ColVec<NY, T>      z_pred{};
    ColVec<NY, T>      z_meas{};
    Matrix<NY, NDX, T> H = Matrix<NY, NDX, T>::zeros();

    size_t idx = 0;

    // Accelerometer measurement
    Vec3<T> accel_pred = q_nom.rotate(g_vec);
    z_pred.template block<3, 1>(idx, 0) = accel_pred;
    z_meas.template block<3, 1>(idx, 0) = accel_meas;

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
    z_pred.template block<3, 1>(idx, 0) = mag_pred;
    z_meas.template block<3, 1>(idx, 0) = mag_meas;

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

    auto meas_fn = [z_pred, H, M]() -> MeasJacobian<T, NY, NDX> {
        return {z_pred, H, M};
    };

    eskf.update(meas_fn, z_meas);

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

// Error-State Kalman Filter for orientation
template<typename T = float>
class ESKFOrientationFilter {
    ErrorStateKalmanFilter<6, 6, T> eskf;
    Quaternion<T>                   q_nom;
    Vec3<T>                         b_g_nom;

public:
    consteval ESKFOrientationFilter(T gyro_noise = 0.003, T accel_noise = 0.03, T mag_noise = 0.3, T gyro_bias_rw = 0.0001, T dt = 0.01) {
        // Use default values for consteval function
        eskf = design::eskf_design(gyro_noise, accel_noise, mag_noise, gyro_bias_rw, dt);
        q_nom = Quaternion<T>::identity();
        b_g_nom = {0, 0, 0};
    }

    void update(const Vec3<T>& accel, const Vec3<T>& gyro, const Vec3<T>& mag, T dt, const Vec3<T>& g_vec = {0, 0, -9.81f}, const Vec3<T>& m_vec = {0, 1, 0}) {
        eskf_update_imu(eskf, q_nom, b_g_nom, accel, gyro, mag, dt, g_vec, m_vec);
    }

    const Quaternion<T>& getOrientation() const { return q_nom; }
    const Vec3<T>&       getGyroBias() const { return b_g_nom; }
};

} // namespace wetmelon::control
