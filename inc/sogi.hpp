#pragma once

#include <numbers>
#include <utility>

#include "discretization.hpp"
#include "state_space.hpp"
#include "wetmelon_math.hpp"

namespace wetmelon::control {

namespace design {

/**
 * @brief Second-Order Generalized Integrator (SOGI) design
 *
 * SOGI provides bandpass filtering for grid synchronization.
 * Produces quadrature signals (90° phase shift).
 *
 * Transfer functions:
 * H_bp(s) = (k*ω₀*s) / (s² + k*ω₀*s + ω₀²)    [bandpass]
 * H_q(s) = (k*ω₀²) / (s² + k*ω₀*s + ω₀²)     [quadrature]
 *
 * @param omega_0 Fundamental frequency [rad/s]
 * @param k Damping gain (typically 1.0-2.0)
 * @param T Scalar type
 * @return StateSpace<2, 1, 2, 0, 0, T> SOGI system
 */
template<typename T = double>
[[nodiscard]] constexpr StateSpace<2, 1, 2, 0, 0, T> sogi_system(T omega_0, T k = std::numbers::sqrt2_v<T>) {
    StateSpace<2, 1, 2, 0, 0, T> sys{
        .A = Matrix<2, 2, T>{
            {-k * omega_0, -omega_0},
            {omega_0, T{0}},
        },

        .B = Matrix<2, 1, T>{
            {k * omega_0},
            {T{0}},
        },

        .C = Matrix<2, 2, T>{
            {T{1}, T{0}},
            {T{0}, T{1}},
        },

        .D = Matrix<2, 1, T>::zeros(),
    };
    return sys;
}

/**
 * @brief Second-Order Generalized Integrator (SOGI) design (discrete-time)
 *
 * Discrete-time SOGI with bandpass and quadrature outputs.
 *
 * @param omega_0 Fundamental frequency [rad/s]
 * @param k Damping gain (typically 1.0-2.0)
 * @param Ts Sample time [s]
 * @param T Scalar type
 * @return StateSpace<2, 1, 2, 0, 0, T> Discrete-time SOGI system
 */
template<typename T = float>
[[nodiscard]] constexpr StateSpace<2, 1, 2, 0, 0, T> sogi_system(T omega_0, T k, T Ts) {
    auto sys_c = sogi_system<T>(omega_0, k);
    return discretize(sys_c, Ts, DiscretizationMethod::Tustin);
}

/**
 * @brief Mixed Second-Third Order Generalized Integrator (MSTOGI)
 *
 * Enhanced SOGI with high-pass filtering on quadrature output.
 * Better harmonic rejection than standard SOGI.
 *
 * @param omega_0 Fundamental frequency [rad/s]
 * @param k Damping gain
 * @param T Scalar type
 * @return StateSpace<3, 1, 2, 0, 0, T> MSTOGI system
 */
template<typename T = double>
[[nodiscard]] constexpr StateSpace<3, 1, 2, 0, 0, T> mstogi_system(T omega_0, T k = std::numbers::sqrt2_v<T>) {
    const T k_omega = k * omega_0;
    const T omega_sq = omega_0 * omega_0;

    StateSpace<3, 1, 2, 0, 0, T> sys{
        .A = Matrix<3, 3, T>{
            {-k_omega, -omega_0, T{0}},
            {omega_0, T{0}, T{0}},
            {T{0}, -omega_sq, T{0}},
        },

        .B = Matrix<3, 1, T>{
            {k_omega},
            {T{0}},
            {T{0}},
        },

        .C = Matrix<2, 3, T>{
            {T{1}, T{0}, T{0}},
            {T{0}, T{0}, T{1}},
        },

        .D = Matrix<2, 1, T>::zeros(),
    };

    return sys;
}

/**
 * @brief Mixed Second-Third Order Generalized Integrator (MSTOGI) design (discrete-time)
 *
 * Discrete-time MSTOGI for enhanced grid synchronization with improved harmonic rejection.
 *
 * @param omega_0 Fundamental frequency [rad/s]
 * @param k Damping gain
 * @param Ts Sample time [s]
 * @param T Scalar type
 * @return StateSpace<3, 1, 2, 0, 0, T> Discrete-time MSTOGI system
 */
template<typename T = float>
[[nodiscard]] constexpr StateSpace<3, 1, 2, 0, 0, T> mstogi_system(T omega_0, T k, T Ts) {
    auto sys_c = mstogi_system<T>(omega_0, k);
    return discretize(sys_c, Ts, DiscretizationMethod::ZOH);
}

} // namespace design

/**
 * @brief SOGI filter realized as a discrete resonator with error feedback
 *
 * Uses the peaking resonator state-space form with poles at p = r·e^(±jω₀):
 *
 *     A = [ r·cos(ω₀)   r·sin(ω₀) ]     B = [1]     C = [0  1]     D = [0]
 *         [-r·sin(ω₀)   r·cos(ω₀) ]         [0]
 *
 * The SOGI wraps this with error feedback on x₁ (bandpass output):
 *
 *     x₁[k+1] = (r·cos(ω₀) − α)·x₁ + r·sin(ω₀)·x₂ + α·u
 *     x₂[k+1] = −r·sin(ω₀)·x₁ + r·cos(ω₀)·x₂
 *
 * where ω₀ = 2π·f₀/fₛ is the discrete resonant frequency, r is the pole
 * radius (1 for undamped), and α controls bandwidth via error feedback gain.
 *
 * No discretization step is needed — the resonator is natively discrete.
 * Frequency retuning requires only recomputing cos/sin.
 *
 * @see "Understanding Digital Signal Processing" (Lyons, 2011), §13.36
 *
 * @tparam T Scalar type (float for embedded deployment)
 */
template<typename T = float>
class SOGI {
public:
    struct Params {
        T omega_0{}; ///< Discrete resonant frequency [rad/sample] = 2π·f₀/fₛ
        T alpha{};   ///< Error feedback gain (controls bandwidth)
        T r{1};      ///< Pole radius (1 = undamped resonator)
    } params{};

private:
    T cos_w0_{1};     ///< Cached cos(ω₀)
    T sin_w0_{0};     ///< Cached sin(ω₀)
    T x1_{0}, x2_{0}; ///< Resonator states: x₁ = bandpass, x₂ = quadrature

public:
    constexpr SOGI() = default;

    /**
     * @brief Construct from physical parameters
     * @param f0 Fundamental frequency [Hz]
     * @param Ts Sample time [s]
     * @param alpha Error feedback gain (bandwidth control, typically √2 ≈ 1.414)
     * @param r Pole radius (default 1, undamped resonator)
     */
    constexpr SOGI(T f0, T Ts, T alpha, T r = T{1}) {
        params.omega_0 = T{2} * std::numbers::pi_v<T> * f0 * Ts;
        params.alpha = alpha;
        params.r = r;
        update_trig();
    }

    /**
     * @brief Update resonant frequency (e.g., from PLL feedback)
     * @param f0 New fundamental frequency [Hz]
     * @param Ts Sample time [s]
     */
    constexpr void set_frequency(T f0, T Ts) {
        params.omega_0 = T{2} * std::numbers::pi_v<T> * f0 * Ts;
        update_trig();
    }

    /**
     * @brief Update discrete resonant frequency directly
     * @param omega_0 Discrete resonant frequency [rad/sample]
     */
    constexpr void set_omega(T omega_0) {
        params.omega_0 = omega_0;
        update_trig();
    }

    /**
     * @brief Process input sample through resonator with error feedback
     * @param u Input sample
     * @return std::pair<T, T> {bandpass, quadrature}
     */
    constexpr std::pair<T, T> operator()(T u) {
        const T r_cos = params.r * cos_w0_;
        const T r_sin = params.r * sin_w0_;

        const T x1_new = (r_cos - params.alpha) * x1_ + r_sin * x2_ + params.alpha * u;
        const T x2_new = -r_sin * x1_ + r_cos * x2_;

        x1_ = x1_new;
        x2_ = x2_new;

        return {x1_, x2_};
    }

    /**
     * @brief Reset resonator state
     */
    constexpr void reset() {
        x1_ = x2_ = T{0};
    }

private:
    constexpr void update_trig() {
        cos_w0_ = wet::cos(params.omega_0);
        sin_w0_ = wet::sin(params.omega_0);
    }
};

} // namespace wetmelon::control
