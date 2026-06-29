#pragma once

#include "wet/math/math.hpp"
#include "wet/matrix/colvec.hpp"

namespace wet {
namespace motor {

/**
 * @brief Configuration for @ref RotorObserver.
 * @tparam T Scalar type.
 */
template<typename T = double>
struct RotorObserverConfig {
    T            bandwidth{T{2000}}; //!< [rad/s] tracking bandwidth (critically damped); larger = faster, noisier
    ColVec<2, T> x0{};               //!< initial state [θ (rad), ω (rad/s)]
};

/**
 * @brief Kinematic rotor angle/speed tracker (PLL) for motor commutation.
 *
 * A second-order phase-locked loop over @f$ [\theta_m,\ \omega_m] @f$ — the classic
 * encoder tracking loop. @ref predict extrapolates the angle as @f$ \theta \mathrel{+}=
 * \omega\,dt @f$ at the current-loop rate (a fresh commutation angle between the slower
 * encoder samples); an encoder correction at its own rate steers both states toward the
 * measurement:
 * @f[
 *   \theta \mathrel{+}= k_p\,\Delta\,dt, \qquad \omega \mathrel{+}= k_i\,\Delta\,dt,
 *   \qquad \Delta = \mathrm{wrap}(\theta_{meas} - \theta).
 * @f]
 * Tuning is a single knob — @ref RotorObserverConfig::bandwidth — with the gains set
 * critically damped (double pole at @f$ -\omega_{bw} @f$): @f$ k_p = 2\,\omega_{bw} @f$,
 * @f$ k_i = \omega_{bw}^2 @f$. No inertia, friction, or torque: nothing to identify, no
 * drivetrain model to be wrong, and a constant load (→ constant speed) leaves zero
 * steady-state error because the dynamics never enter the loop. Load/friction/inertia are
 * estimated elsewhere and applied to the servo as a torque feedforward.
 *
 * The angle stays wrapped to @f$ [-\pi,\pi) @f$ forever, so commutation is numerically
 * bounded however long the shaft spins; feed a wrapped angle (the @f$ \Delta @f$ is taken
 * on the shortest arc, fusing across the @f$ \pm\pi @f$ seam). Multi-turn position is a
 * turn counter (@ref turns) bumped on each seam crossing; @ref theta_unwrapped is the
 * continuous angle.
 *
 * @see MechanicalEstimator — the torque-driven 3-state variant that also estimates load.
 * @see ODrive's encoder PLL (`pll_kp = 2·bw`, `pll_ki = 0.25·pll_kp²`).
 *
 * @tparam T Scalar type.
 */
template<typename T = double>
class RotorObserver {
public:
    constexpr explicit RotorObserver(T bandwidth = T{1000}, ColVec<2, T> x0 = {})
        : kp_(T{2} * bandwidth), ki_(bandwidth * bandwidth), theta_(wrap(x0[0], -pi, pi)) {}

    constexpr explicit RotorObserver(const RotorObserverConfig<T>& config)
        : RotorObserver<T>(config.bandwidth, config.x0) {}

    /// Set the tracking bandwidth [rad/s] (critically damped). Users' "faster/slower" knob.
    constexpr void set_bandwidth(T bandwidth) {
        kp_ = T{2} * bandwidth;
        ki_ = bandwidth * bandwidth;
    }

    /// Predict over @p dt [s]: extrapolate θ by ω·dt (cheap; run every tick).
    constexpr ColVec<2, T> predict(T dt) {
        theta_ += omega_ * dt;
        wrap_and_count();

        return {theta_, omega_};
    }

    /// Correct from a wrapped encoder angle in [-π,π) [rad] sampled @p dt [s] ago.
    constexpr ColVec<2, T> update(T theta_mech, T dt) {
        const T delta = wrap(theta_mech - theta_, -pi, pi);
        theta_ += kp_ * delta * dt;
        omega_ += ki_ * delta * dt;
        wrap_and_count();

        return {theta_, omega_};
    }

    [[nodiscard]] constexpr T theta() const { return theta_; } //!< [rad] mechanical angle, [-π,π)
    [[nodiscard]] constexpr T turns() const { return turns_; } //!< [turns] whole mechanical turns
    [[nodiscard]] constexpr T omega() const { return omega_; } //!< [rad/s] mechanical speed

    [[nodiscard]] constexpr T theta_unwrapped() const { return (turns_ * two_pi) + theta_; } //!< [rad] continuous angle

    constexpr void reset(const ColVec<2, T>& x0 = ColVec<2, T>{}) {
        theta_ = wrap(x0[0], -pi, pi);
        omega_ = x0[1];
        theta_prev_ = theta_;
        turns_ = T{0};
    }

private:
    static constexpr T pi = wet::numbers::pi_v<T>;
    static constexpr T two_pi = T{2} * wet::numbers::pi_v<T>;

    // Re-wrap θ into [-π,π) and count a turn on each ±π seam crossing.
    constexpr void wrap_and_count() {
        const T theta_w = wrap(theta_, -pi, pi);
        const T delta = theta_w - theta_prev_;
        if (delta < -pi) {
            turns_ += T{1}; // crossed +π → −π (forward)
        } else if (delta > pi) {
            turns_ -= T{1}; // crossed −π → +π (reverse)
        }
        theta_ = theta_w;
        theta_prev_ = theta_w;
    }

    T kp_{T{2} * T{2000}};    //!< proportional gain = 2·bandwidth
    T ki_{T{2000} * T{2000}}; //!< integral gain = bandwidth²
    T theta_{T{0}};           //!< [rad] wrapped mechanical angle [-π,π)
    T theta_prev_{T{0}};      //!< [rad] previous wrapped θ, for turn-crossing detection
    T omega_{T{0}};           //!< [rad/s] mechanical speed
    T turns_{T{0}};           //!< whole mechanical turns from seam crossings
};

} // namespace motor
} // namespace wet
