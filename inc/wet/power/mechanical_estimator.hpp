#pragma once

#include "wet/estimation/kalman.hpp" // KalmanFilter (full-covariance, multirate)
#include "wet/matrix/colvec.hpp"
#include "wet/matrix/matrix.hpp"
#include "wet/systems/discretization.hpp" // discretize (ZOH)
#include "wet/systems/state_space.hpp"    // StateSpace

namespace wet {

namespace design {

/**
 * @brief Continuous state-space model of a 1-DOF mechanical drivetrain.
 * @ingroup foc_design
 *
 * State @f$ x = [\theta_m,\ \omega_m,\ \tau_{load}]^\top @f$ (mechanical angle,
 * speed, and an unknown load torque modelled as a random walk), input @f$ u = i_q @f$
 * [A], output @f$ y = \theta_m @f$ (a measured angle):
 * @f[
 *   \dot\theta = \omega, \quad
 *   \dot\omega = \frac{K_t i_q - \tau_{load} - b\,\omega}{J}, \quad
 *   \dot\tau_{load} = 0,
 * @f]
 * giving
 * @f[
 *   A = \begin{bmatrix} 0 & 1 & 0\\ 0 & -b/J & -1/J\\ 0 & 0 & 0\end{bmatrix},\
 *   B = \begin{bmatrix} 0\\ K_t/J\\ 0\end{bmatrix},\ C = [1\ 0\ 0].
 * @f]
 * The dynamics are linear in the state, so a linear Kalman filter (not an EKF)
 * estimates @f$ [\theta,\omega,\tau_{load}] @f$ from angle measurements and the
 * known @f$ i_q @f$ — see @ref MechanicalEstimator. Noise inputs are @f$ G = I_3 @f$
 * (per-state process noise) and @f$ H = 1 @f$ (angle-measurement noise). Continuous
 * (Ts = 0); discretize with @ref discretize.
 *
 * @param J  [kg·m²] reflected inertia (> 0).
 * @param b  [Nm·s]  viscous friction (≥ 0).
 * @param Kt [Nm/A]  torque constant.
 * @return Continuous @ref StateSpace<3,1,1,3,1>.
 */
template<typename T = double>
[[nodiscard]] constexpr StateSpace<3, 1, 1, 3, 1, T> mechanical_ss(T J, T b, T Kt) {
    StateSpace<3, 1, 1, 3, 1, T> sys{};
    sys.A(0, 1) = T{1};
    sys.A(1, 1) = -b / J;
    sys.A(1, 2) = -T{1} / J;
    sys.B(1, 0) = Kt / J;
    sys.C(0, 0) = T{1};
    sys.G = Matrix<3, 3, T>::identity(); // per-state process noise
    sys.H = Matrix<1, 1, T>{{T{1}}};     // angle-measurement noise
    return sys;
}

} // namespace design

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
    T               J{T{1}};                                                     //!< [kg·m²] reflected inertia (> 0)
    T               b{T{0}};                                                     //!< [Nm·s] viscous friction (≥ 0)
    T               Kt{T{1}};                                                    //!< [Nm/A] torque constant
    T               Ts{T{1} / T{10000}};                                         //!< [s] predict step period (the fast rate)
    Matrix<3, 3, T> Q = Matrix<3, 3, T>::diagonal({T{1e-12}, T{1e-6}, T{1e-8}}); //!< per-step process-noise covariance [θ, ω, τ_load]
    T               r_encoder{T{1e-8}};                                          //!< [rad²] encoder angle measurement variance
    T               r_sensorless{T{1e-3}};                                       //!< [rad²] sensorless angle measurement variance
    T               r_accel{T{1e-1}};                                            //!< [(rad/s²)²] load angular-acceleration measurement variance
    ColVec<3, T>    x0{};                                                        //!< initial state estimate
    Matrix<3, 3, T> P0 = Matrix<3, 3, T>::identity();                            //!< initial covariance
};

/**
 * @brief Cheap-predict mechanical estimator for position, speed, and load torque.
 *
 * A linear Kalman filter over @f$ [\theta_m,\ \omega_m,\ \tau_{load}] @f$ built on
 * @ref design::mechanical_ss. The predict step is a 3×3 covariance propagation, cheap
 * enough to run at the current-loop rate (e.g. 24 kHz) from the commanded @f$ i_q @f$;
 * measurement updates from a sensor run at a slower rate, multirate — skipping updates
 * simply lets the covariance grow until the next one. Optional measurement channels feed
 * the same filter, each with its own noise: a motor-encoder angle (low noise), a
 * sensorless angle (higher noise), and a load angular-acceleration measurement (which
 * observes the load torque directly through the dynamics). Apply whichever the hardware
 * provides; the sensorless and accelerometer channels are mixed into the same estimate.
 *
 * The angle state is continuous (multi-turn): feed continuous, unwrapped mechanical
 * angle to the updates (accumulate encoder counts; unwrap a sensorless electrical
 * phase to mechanical before passing it). The Kalman innovation is @f$ y - C\hat x @f$,
 * so measurements must be pre-unwrapped to agree with the continuous state.
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
        // Load-acceleration measurement model: α = (Kt·iq − b·ω − τ_load)/J, i.e.
        // C_accel·x + D_accel·iq, linear in the same state.
        : C_accel_(Matrix<1, 3, T>{{T{0}, -config.b / config.J, -T{1} / config.J}}),
          D_accel_(Matrix<1, 1, T>{{config.Kt / config.J}}),
          r_encoder_(config.r_encoder),
          r_sensorless_(config.r_sensorless),
          r_accel_(config.r_accel) {
        auto sys = discretize(design::mechanical_ss(config.J, config.b, config.Kt), config.Ts, DiscretizationMethod::ZOH);
        // Run the covariance with G = I so Q is the discrete process-noise covariance.
        sys.G = Matrix<3, 3, T>::identity();
        sys.H = Matrix<1, 1, T>{{T{1}}};
        kf_ = KalmanFilter<3, 1, 1, 3, 1, T>{sys, config.Q, Matrix<1, 1, T>{{config.r_encoder}}, config.x0, config.P0};
    }

    /// Predict one step from the commanded q-axis current [A] (cheap; run every tick).
    constexpr void predict(T iq) { kf_.predict(ColVec<1, T>{iq}); }

    /// Correct from an encoder angle [rad] (continuous/unwrapped). Returns false if singular.
    constexpr bool update_encoder(T theta_mech) { return update_angle(theta_mech, r_encoder_); }

    /// Correct from a sensorless angle [rad] (continuous/unwrapped, higher noise).
    constexpr bool update_sensorless(T theta_mech) { return update_angle(theta_mech, r_sensorless_); }

    /**
     * @brief Correct from a load angular-acceleration measurement [rad/s²].
     *
     * Uses the dynamics as the measurement model — @f$ \alpha = (K_t i_q - b\omega -
     * \tau_{load})/J @f$ — so an accelerometer on the (rigidly-coupled) load directly
     * observes the load torque, sharpening @f$ \tau_{load} @f$ and @f$ \omega @f$ even
     * with only a motor encoder. Pass the same @p iq used in @ref predict. For a linear
     * accelerometer, convert to angular acceleration and remove gravity first.
     */
    constexpr bool update_load_accel(T alpha, T iq) {
        return kf_.update(ColVec<1, T>{alpha}, C_accel_, D_accel_, Matrix<1, 1, T>{{r_accel_}}, ColVec<1, T>{iq});
    }

    [[nodiscard]] constexpr T theta() const { return kf_.state()[0]; }       //!< [rad] mechanical angle
    [[nodiscard]] constexpr T omega() const { return kf_.state()[1]; }       //!< [rad/s] mechanical speed
    [[nodiscard]] constexpr T load_torque() const { return kf_.state()[2]; } //!< [Nm] estimated load torque

    [[nodiscard]] constexpr const Matrix<3, 3, T>& covariance() const { return kf_.covariance(); }

    constexpr void reset(const ColVec<3, T>& x0 = ColVec<3, T>{}, const Matrix<3, 3, T>& P0 = Matrix<3, 3, T>::identity()) {
        kf_.reset(x0, P0);
    }

private:
    // Angle channels share the model's nominal C = [1 0 0], D = 0; only R differs.
    constexpr bool update_angle(T theta_mech, T r) {
        return kf_.update(ColVec<1, T>{theta_mech}, kf_.model().C, kf_.model().D, Matrix<1, 1, T>{{r}});
    }

    KalmanFilter<3, 1, 1, 3, 1, T> kf_{};
    Matrix<1, 3, T>                C_accel_{}; // load-acceleration measurement model
    Matrix<1, 1, T>                D_accel_{}; // its iq feedthrough

    T r_encoder_{T{1e-8}};
    T r_sensorless_{T{1e-3}};
    T r_accel_{T{1e-1}};
};

} // namespace wet
