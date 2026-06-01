#pragma once

#include <numbers>
#include <utility>

#include "wet/math/wetmelon_math.hpp"
#include "wet/systems/discretization.hpp"
#include "wet/systems/state_space.hpp"

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
 * H_notch(s) = (s² + ω₀²) / (s² + k*ω₀*s + ω₀²) [notch]
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
 * @brief SOGI design with explicit notch output channel
 *
 * Outputs are:
 * - y0 = bandpass
 * - y1 = quadrature
 * - y2 = notch = u - bandpass
 *
 * @param omega_0 Fundamental frequency [rad/s]
 * @param k Damping gain (typically 1.0-2.0)
 * @param T Scalar type
 * @return StateSpace<2, 1, 3, 0, 0, T> SOGI system with notch output
 */
template<typename T = double>
[[nodiscard]] constexpr StateSpace<2, 1, 3, 0, 0, T> sogi_system_with_notch(T omega_0, T k = std::numbers::sqrt2_v<T>) {
    StateSpace<2, 1, 3, 0, 0, T> sys{
        .A = Matrix<2, 2, T>{
            {-k * omega_0, -omega_0},
            {omega_0, T{0}},
        },

        .B = Matrix<2, 1, T>{
            {k * omega_0},
            {T{0}},
        },

        .C = Matrix<3, 2, T>{
            {T{1}, T{0}},  // bandpass
            {T{0}, T{1}},  // quadrature
            {T{-1}, T{0}}, // notch = u - x1
        },

        .D = Matrix<3, 1, T>{
            {T{0}},
            {T{0}},
            {T{1}},
        },
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
 * @brief SOGI design with explicit notch output channel (discrete-time)
 *
 * @param omega_0 Fundamental frequency [rad/s]
 * @param k Damping gain (typically 1.0-2.0)
 * @param Ts Sample time [s]
 * @param T Scalar type
 * @return StateSpace<2, 1, 3, 0, 0, T> Discrete-time SOGI system with notch output
 */
template<typename T = float>
[[nodiscard]] constexpr StateSpace<2, 1, 3, 0, 0, T> sogi_system_with_notch(T omega_0, T k, T Ts) {
    auto sys_c = sogi_system_with_notch<T>(omega_0, k);
    return discretize(sys_c, Ts, DiscretizationMethod::Tustin);
}

/**
 * @brief Mixed Second/Third-Order Generalized Integrator (MSTOGI)
 *
 * A standard SOGI-QSG augmented with a Third-Order Generalized Integrator
 * (TOGI) that estimates and removes the DC / offset component the plain
 * SOGI-QSG quadrature output would otherwise pass. The plain `qv′` channel of a
 * SOGI-QSG is a low-pass that has unity gain at DC; a biased or offset grid
 * signal therefore corrupts the quadrature estimate. The MSTOGI subtracts a
 * co-tuned TOGI estimate so the quadrature channel rejects DC.
 *
 * States `[v′, v″, v‴]` = [in-phase, quadrature, TOGI-tracker]. With the
 * post-gain bus `w = k·ω₀·(v − v′)`:
 *
 *     v̇′  = k·ω₀·(v − v′) − ω₀·v″     (SOGI in-phase integrator)
 *     v̇″  = ω₀·v′                       (SOGI quadrature integrator)
 *     v̇‴  = k·ω₀·(v − v′) − ω₀·v‴       (TOGI: first-order, self-damped)
 *
 * Outputs: `v_o = v′` (band-pass, unity at ω₀) and `q·v_o = v″ − v‴`
 * (DC-rejecting quadrature). The transfer functions are
 *
 *     v_o / v   = k·ω₀·s / (s² + k·ω₀·s + ω₀²)
 *     q·v_o / v = k·ω₀·s·(ω₀ − s) / [ (s + ω₀)·(s² + k·ω₀·s + ω₀²) ]
 *
 * The q·v_o numerator has a zero at s = 0 (DC rejection) and the pair evaluates
 * to (1∠0°, 1∠−90°) at s = jω₀ — a clean unity-magnitude quadrature pair.
 *
 * @note Discretize with the exact ZOH (`DiscretizationMethod::ZOH`, the
 *       `mstogi_system(omega_0, k, Ts)` overload) so the resonant poles land
 *       exactly on the unit circle at z = e^{±jω₀Tₛ}; Tustin warps them.
 *
 * @see Rodríguez et al., "Discrete-time implementation of second order
 *      generalized integrators for grid converters," IECON 2008,
 *      https://doi.org/10.1109/IECON.2008.4757983
 *
 * @param omega_0 Fundamental frequency [rad/s]
 * @param k Damping gain (typically √2 for ~unity-Q SOGI tuning)
 * @param T Scalar type
 * @return StateSpace<3, 1, 2, 0, 0, T> MSTOGI system (outputs: v_o, q·v_o)
 */
template<typename T = double>
[[nodiscard]] constexpr StateSpace<3, 1, 2, 0, 0, T> mstogi_system(T omega_0, T k = std::numbers::sqrt2_v<T>) {
    const T k_omega = k * omega_0;

    StateSpace<3, 1, 2, 0, 0, T> sys{
        .A = Matrix<3, 3, T>{
            {-k_omega, -omega_0, T{0}},
            {omega_0, T{0}, T{0}},
            {-k_omega, T{0}, -omega_0},
        },

        .B = Matrix<3, 1, T>{
            {k_omega},
            {T{0}},
            {k_omega},
        },

        .C = Matrix<2, 3, T>{
            {T{1}, T{0}, T{0}},  // v_o   = v′      (band-pass)
            {T{0}, T{1}, T{-1}}, // q·v_o = v″ − v‴ (DC-rejecting quadrature)
        },

        .D = Matrix<2, 1, T>::zeros(),
    };

    return sys;
}

/**
 * @brief Mixed Second/Third-Order Generalized Integrator (MSTOGI) design (discrete-time)
 *
 * Exact-ZOH discretization of the continuous MSTOGI so the resonant poles sit
 * precisely at z = e^{±jω₀Tₛ} (no Tustin frequency warping).
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
    struct Output {
        T bandpass{};
        T quadrature{};
        T notch{};
    };

    struct Params {
        T omega_0{}; ///< Discrete resonant frequency [rad/sample] = 2π·f₀/fₛ
        T alpha{};   ///< Error feedback gain (controls bandwidth)
        T r{1};      ///< Pole radius (1 = undamped resonator)
    } params{};

private:
    T cos_w0_{1};     ///< Cached cos(ω₀)
    T sin_w0_{0};     ///< Cached sin(ω₀)
    T x1_{0}, x2_{0}; ///< Resonator states: x₁ = bandpass, x₂ = quadrature
    T notch_{0};      ///< Latest notch output (u - bandpass)

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
        const auto y = process(u);
        return {y.bandpass, y.quadrature};
    }

    /**
     * @brief Process input sample and return bandpass, quadrature, and notch outputs
     * @param u Input sample
     * @return Output {bandpass, quadrature, notch}
     */
    [[nodiscard]] constexpr Output process(T u) {
        const T r_cos = params.r * cos_w0_;
        const T r_sin = params.r * sin_w0_;

        const T x1_new = (r_cos - params.alpha) * x1_ + r_sin * x2_ + params.alpha * u;
        const T x2_new = -r_sin * x1_ + r_cos * x2_;

        x1_ = x1_new;
        x2_ = x2_new;
        notch_ = u - x1_;

        return Output{x1_, x2_, notch_};
    }

    /**
     * @brief Latest bandpass output
     */
    [[nodiscard]] constexpr T bandpass() const {
        return x1_;
    }

    /**
     * @brief Latest quadrature output
     */
    [[nodiscard]] constexpr T quadrature() const {
        return x2_;
    }

    /**
     * @brief Latest notch output
     */
    [[nodiscard]] constexpr T notch() const {
        return notch_;
    }

    /**
     * @brief Reset resonator state
     */
    constexpr void reset() {
        x1_ = x2_ = T{0};
        notch_ = T{0};
    }

private:
    constexpr void update_trig() {
        cos_w0_ = wet::cos(params.omega_0);
        sin_w0_ = wet::sin(params.omega_0);
    }
};

/**
 * @brief MSTOGI runtime: SOGI-QSG with a TOGI stage for DC-rejecting quadrature
 *
 * Allocation-free runtime for the MSTOGI structure (see @ref design::mstogi_system).
 * The continuous 3-state model is discretized once at construction with an exact
 * ZOH (resonant poles exactly at z = e^{±jω₀Tₛ}), then each tick is a single
 * 3×3 matrix–vector product plus the output combination:
 *
 *     x[k+1] = A_d·x[k] + B_d·v[k]
 *     v_o    = x₁                       (band-pass, unity gain at ω₀)
 *     q·v_o  = x₂ − x₃                  (quadrature with DC rejected by the TOGI)
 *
 * Unlike the standard @ref SOGI resonator (whose quadrature `qv′` passes DC),
 * this rejects a DC/offset component in the quadrature channel — the reason to
 * reach for MSTOGI in a single-phase PLL or QSG on an offset-prone signal.
 *
 * Frequency retuning re-runs the ZOH discretization (`set_frequency`); that is a
 * design-weight operation (matrix exponential), so retune at a slow rate, not
 * every ISR tick.
 *
 * @see design::mstogi_system() for the model and transfer functions
 * @see Rodríguez et al., IECON 2008, https://doi.org/10.1109/IECON.2008.4757983
 *
 * @tparam T Scalar type (float for embedded deployment)
 */
template<typename T = float>
class MSTOGI {
public:
    struct Output {
        T band_pass{};  ///< v_o = v′ (in-phase, band-pass)
        T quadrature{}; ///< q·v_o = v″ − v‴ (DC-rejecting quadrature)
    };

    constexpr MSTOGI() = default;

    /**
     * @brief Construct from physical parameters.
     * @param f0 Fundamental frequency [Hz]
     * @param Ts Sample time [s]
     * @param k  Damping gain (default √2)
     */
    constexpr MSTOGI(T f0, T Ts, T k = std::numbers::sqrt2_v<T>)
        : f0_(f0), Ts_(Ts), k_(k) {
        discretize_model();
    }

    /**
     * @brief Retune the resonant frequency (re-runs ZOH discretization).
     * @param f0 New fundamental frequency [Hz]
     */
    constexpr void set_frequency(T f0) {
        f0_ = f0;
        discretize_model();
    }

    /**
     * @brief Process one input sample.
     * @param v Input sample
     * @return Output{band_pass, quadrature}
     */
    [[nodiscard]] constexpr Output process(T v) {
        x_ = Ad_ * x_ + Bd_ * v; // x[k+1] = A_d·x[k] + B_d·v[k]
        return Output{x_[0], x_[1] - x_[2]};
    }

    /// @brief Process one sample, returning {band_pass, quadrature} as a pair.
    constexpr std::pair<T, T> operator()(T v) {
        const auto y = process(v);
        return {y.band_pass, y.quadrature};
    }

    [[nodiscard]] constexpr T band_pass() const { return x_[0]; }
    [[nodiscard]] constexpr T quadrature() const { return x_[1] - x_[2]; }

    constexpr void reset() { x_ = ColVec<3, T>{}; }

private:
    constexpr void discretize_model() {
        const T    omega_0 = T{2} * std::numbers::pi_v<T> * f0_;
        const auto sys_d = design::mstogi_system<T>(omega_0, k_, Ts_);
        Ad_ = sys_d.A;
        Bd_ = sys_d.B;
    }

    T               f0_{};
    T               Ts_{};
    T               k_{std::numbers::sqrt2_v<T>};
    Matrix<3, 3, T> Ad_{Matrix<3, 3, T>::identity()};
    Matrix<3, 1, T> Bd_{};
    ColVec<3, T>    x_{};
};

} // namespace wetmelon::control
