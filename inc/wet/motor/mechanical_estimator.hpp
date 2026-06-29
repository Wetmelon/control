#pragma once

#include "wet/estimation/kalman.hpp" // KalmanFilter (full-covariance, multirate)
#include "wet/math/math.hpp"         // wrap, numbers::pi_v
#include "wet/matrix/colvec.hpp"
#include "wet/matrix/matrix.hpp"
#include "wet/systems/discretization.hpp" // discretize (ZOH)
#include "wet/systems/state_space.hpp"    // StateSpace

namespace wet {

namespace design {

/**
 * @brief Continuous state-space model of a 1-DOF rotational drivetrain with an
 *        augmented load-torque state.
 * @ingroup foc_design
 *
 * State @f$ x = [\theta_m,\ \omega_m,\ \tau_{load}]^\top @f$ (mechanical angle,
 * speed, and an unknown load torque modelled as a random walk), input @f$ u =
 * \tau_{em} @f$ [Nm] (electromagnetic torque), output @f$ y = \theta_m @f$ (a measured
 * angle):
 * @f[
 *   \dot\theta = \omega, \quad
 *   \dot\omega = \frac{\tau_{em} - \tau_{load} - b\,\omega}{J}, \quad
 *   \dot\tau_{load} = 0,
 * @f]
 * giving
 * @f[
 *   A = \begin{bmatrix} 0 & 1 & 0\\ 0 & -b/J & -1/J\\ 0 & 0 & 0\end{bmatrix},\
 *   B = \begin{bmatrix} 0\\ 1/J\\ 0\end{bmatrix},\ C = [1\ 0\ 0].
 * @f]
 * Taking torque (not @f$ i_q @f$) as the input keeps the model linear and
 * machine-agnostic: the caller maps current to torque with the appropriate magnetic
 * model (@ref electromagnetic_torque, including reluctance for a salient machine), so a
 * linear Kalman filter (not an EKF) estimates @f$ [\theta,\omega,\tau_{load}] @f$ — see
 * @ref MechanicalEstimator. Noise inputs are @f$ G = I_3 @f$ (per-state process noise)
 * and @f$ H = 1 @f$ (angle-measurement noise). Continuous (Ts = 0); discretize with
 * @ref discretize.
 *
 * @param J  [kg·m²] reflected inertia (> 0).
 * @param b  [Nm·s]  viscous friction (≥ 0).
 * @return Continuous @ref StateSpace<3,1,1,3,1>.
 */
template<typename T = double>
[[nodiscard]] constexpr StateSpace<3, 1, 1, 3, 1, T> rotational_load_ss(T J, T b) {
    StateSpace<3, 1, 1, 3, 1, T> sys{
        .A = {
            {0, T{1}, 0},
            {0, -b / J, -T{1} / J},
            {0, 0, 0},
        },

        .B = {
            {0},
            {T{1} / J},
            {0},
        },

        .C = {{T{1}, 0, 0}},
        .D = {},

        .G = Matrix<3, 3, T>::identity(), // per-state process noise
        .H = Matrix<1, 1, T>{{T{1}}},     // angle-measurement noise
    };
    return sys;
}

} // namespace design

namespace motor {

/**
 * @brief Configuration for @ref MechanicalEstimator.
 *
 * @p Q (per-step process-noise covariance) and the per-source measurement variances
 * are the tuning knobs and are motor/scale dependent. @f$ \tau_{load} @f$'s process
 * noise sets how fast the load estimate may drift: small (slowly-varying load); only
 * raise it if the load genuinely changes quickly. The defaults are a sane starting
 * point, not a substitute for tuning to the machine.
 *
 * @tparam T Scalar type.
 */
template<typename T = double>
struct MechanicalEstimatorConfig {
    T J{T{1}};             //!< [kg·m²] reflected inertia (> 0)
    T b{T{0}};             //!< [Nm·s] viscous friction (≥ 0)
    T Ts{T{1} / T{10000}}; //!< [s] predict step period (the fast rate)

    T r_encoder{T{1e-10}};   //!< [rad²] encoder angle measurement variance (high-trust: gain≈1, angle tracks the encoder tightly)
    T r_sensorless{T{1e-3}}; //!< [rad²] sensorless angle measurement variance
    T r_accel{T{1e-1}};      //!< [(rad/s²)²] load angular-acceleration measurement variance

    ColVec<3, T>    x0{};                                                        //!< initial state estimate
    Matrix<3, 3, T> Q = Matrix<3, 3, T>::diagonal({T{1e-12}, T{1e-6}, T{1e-8}}); //!< per-step process-noise covariance [θ, ω, τ_load]
    Matrix<3, 3, T> P0 = Matrix<3, 3, T>::identity();                            //!< initial covariance
};

/**
 * @brief Cheap-predict mechanical estimator for position, speed, and load torque.
 *
 * A linear Kalman filter over @f$ [\theta_m,\ \omega_m,\ \tau_{load}] @f$. The predict
 * step (from the electromagnetic torque @f$ \tau_{em} @f$) is cheap enough for the current-loop rate (e.g.
 * 24 kHz); measurement updates run slower and multirate — a skipped update just lets the
 * covariance grow. Three optional channels fuse into the same estimate, each with its own
 * noise: encoder angle (low noise), sensorless angle (higher noise), and load angular
 * acceleration (which observes @f$ \tau_{load} @f$ through the dynamics).
 *
 * The angle state stays wrapped to @f$ [-\pi,\pi) @f$ forever, so control is numerically
 * bounded however long the shaft spins. Feed a wrapped angle; the innovation is the
 * shortest arc @f$ \mathrm{wrap}(y-\hat\theta) @f$, so measurement and estimate fuse
 * correctly across the @f$ \pm\pi @f$ seam. Multi-turn position is a turn counter (@ref
 * turns) bumped on each seam crossing; @ref theta_unwrapped is the continuous angle.
 *
 * @see KalmanFilter — the full-covariance runtime filter.
 * @see Simon, "Optimal State Estimation" (2006), §5 — the linear Kalman filter.
 *
 * @tparam T Scalar type.
 */
template<typename T = double>
class MechanicalEstimator {
public:
    constexpr MechanicalEstimator() = default;

    constexpr explicit MechanicalEstimator(const MechanicalEstimatorConfig<T>& config)
        // Load-acceleration measurement model: α = (τ_em − b·ω − τ_load)/J, i.e.
        // C_accel·x + D_accel·τ_em, linear in the same state.
        : C_accel_(Matrix<1, 3, T>{{T{0}, -config.b / config.J, -T{1} / config.J}}),
          D_accel_(Matrix<1, 1, T>{{T{1} / config.J}}),
          r_encoder_(config.r_encoder),
          r_sensorless_(config.r_sensorless),
          r_accel_(config.r_accel),
          theta_prev_(T{0}) {
        auto sys = discretize(design::rotational_load_ss(config.J, config.b), config.Ts, DiscretizationMethod::ZOH);
        kf_ = KalmanFilter<3, 1, 1, 3, 1, T>{sys, config.Q, Matrix<1, 1, T>{{config.r_encoder}}, config.x0, config.P0};
        theta_prev_ = wrap(kf_.state()[0], -pi(), pi());
        kf_.set_state(0, theta_prev_);
    }

    /// Predict one step from the electromagnetic torque [Nm] (cheap; run every tick).
    constexpr void predict(T tau_em) {
        kf_.predict(ColVec<1, T>{tau_em}); // dead-reckons θ forward by ω·Ts between measurements
        wrap_and_count();                  // roll θ into [-π,π) and tally turn seam crossings
    }

    /// Correct from a wrapped encoder angle in [-π,π) [rad]. Returns false if singular.
    constexpr bool update_encoder(T theta_mech) { return update_angle(theta_mech, r_encoder_); }

    /// Correct from a wrapped sensorless angle in [-π,π) [rad] (higher noise).
    constexpr bool update_sensorless(T theta_mech) { return update_angle(theta_mech, r_sensorless_); }

    /**
     * @brief Correct from a load angular-acceleration measurement [rad/s²].
     *
     * Uses the dynamics as the measurement model — @f$ \alpha = (\tau_{em} - b\omega -
     * \tau_{load})/J @f$ — so an accelerometer on the (rigidly-coupled) load directly
     * observes the load torque, sharpening @f$ \tau_{load} @f$ and @f$ \omega @f$ even
     * with only a motor encoder. Pass the same @p tau_em used in @ref predict. For a
     * linear accelerometer, convert to angular acceleration and remove gravity first.
     */
    constexpr bool update_load_accel(T alpha, T tau_em) {
        const bool ok = kf_.update(ColVec<1, T>{alpha}, C_accel_, D_accel_, Matrix<1, 1, T>{{r_accel_}}, ColVec<1, T>{tau_em});
        wrap_and_count();
        return ok;
    }

    [[nodiscard]] constexpr T theta() const { return kf_.state()[0]; }                                 //!< [rad] mechanical angle, [-π,π)
    [[nodiscard]] constexpr T turns() const { return turns_; }                                         //!< [turns] whole mechanical turns
    [[nodiscard]] constexpr T theta_unwrapped() const { return (turns_ * two_pi()) + kf_.state()[0]; } //!< [rad] continuous angle
    [[nodiscard]] constexpr T omega() const { return kf_.state()[1]; }                                 //!< [rad/s] mechanical speed
    [[nodiscard]] constexpr T load_torque() const { return kf_.state()[2]; }                           //!< [Nm] estimated load torque

    [[nodiscard]] constexpr const Matrix<3, 3, T>& covariance() const { return kf_.covariance(); }

    constexpr void reset(const ColVec<3, T>& x0 = ColVec<3, T>{}, const Matrix<3, 3, T>& P0 = Matrix<3, 3, T>::identity()) {
        kf_.reset(x0, P0);
        theta_prev_ = wrap(kf_.state()[0], -pi(), pi());
        kf_.set_state(0, theta_prev_);
        turns_ = T{0};
    }

private:
    static constexpr T pi() { return wet::numbers::pi_v<T>; }
    static constexpr T two_pi() { return T{2} * wet::numbers::pi_v<T>; }

    // Angle channels share the model's nominal C = [1 0 0], D = 0; only R differs.
    // The measurement is placed next to the current estimate so the KF's innovation
    // (z' − θ̂) equals the shortest arc wrap(z − θ̂) across the ±π seam. A small encoder
    // R makes the gain ≈ 1, so the angle tracks the encoder tightly (good commutation)
    // while the innovation still sharpens ω and τ_load through the cross-covariance.
    constexpr bool update_angle(T theta_meas, T r) {
        const T    z_adj = kf_.state()[0] + wrap(theta_meas - kf_.state()[0], -pi(), pi());
        const bool ok = kf_.update(ColVec<1, T>{z_adj}, kf_.model().C, kf_.model().D, Matrix<1, 1, T>{{r}});
        wrap_and_count();
        return ok;
    }

    // Re-wrap θ into [-π,π) and count a turn when it crosses the ±π seam (the wrapped
    // angle's motion is realized partly in predict and partly in the measurement update,
    // so both call this).
    constexpr void wrap_and_count() {
        const T theta_w = wrap(kf_.state()[0], -pi(), pi());
        const T delta = theta_w - theta_prev_;
        if (delta < -pi()) {
            turns_ += T{1}; // crossed +π → −π (forward)
        } else if (delta > pi()) {
            turns_ -= T{1}; // crossed −π → +π (reverse)
        }
        kf_.set_state(0, theta_w);
        theta_prev_ = theta_w;
    }

    KalmanFilter<3, 1, 1, 3, 1, T> kf_{};
    Matrix<1, 3, T>                C_accel_{}; // load-acceleration measurement model
    Matrix<1, 1, T>                D_accel_{}; // its iq feedthrough

    T r_encoder_{T{1e-10}};
    T r_sensorless_{T{1e-3}};
    T r_accel_{T{1e-1}};

    T turns_{T{0}};      //!< whole mechanical turns accumulated from seam crossings
    T theta_prev_{T{0}}; //!< previous wrapped θ, for turn-crossing detection
};

} // namespace motor
} // namespace wet
