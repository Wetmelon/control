#pragma once

#include "wet/backend.hpp"
#include "wet/math/math.hpp"
#include "wet/matrix/colvec.hpp"
#include "wet/systems/state_space.hpp"

namespace wet {

namespace design {

/**
 * @brief Second-Order Generalized Integrator (SOGI) design
 *
 * SOGI provides bandpass filtering for grid synchronization.
 * Produces quadrature signals (90° phase shift).
 *
 * Transfer functions:
 * H_bp(s) = (alpha*ω₀*s) / (s² + alpha*ω₀*s + ω₀²)    [bandpass]
 * H_q(s) = (alpha*ω₀²) / (s² + alpha*ω₀*s + ω₀²)     [quadrature]
 * H_notch(s) = (s² + ω₀²) / (s² + alpha*ω₀*s + ω₀²) [notch]
 *
 * @param w0 Fundamental frequency [rad/s]
 * @param alpha Damping gain (typically 1.0-2.0)
 * @param T Scalar type
 * @return StateSpace<2, 1, 2, 0, 0, T> SOGI system
 */
template<typename T = double>
[[nodiscard]] constexpr StateSpace<2, 1, 2, 0, 0, T> sogi_system(T w0, T alpha = wet::numbers::sqrt2_v<T>) {
    return {
        .A = Matrix<2, 2, T>{
            {-alpha * w0, -w0},
            {w0, T{0}},
        },

        .B = Matrix<2, 1, T>{
            {alpha * w0},
            {T{0}},
        },

        .C = Matrix<2, 2, T>{
            {T{1}, T{0}},
            {T{0}, T{1}},
        },

        .D = Matrix<2, 1, T>::zeros(),
    };
}

/**
 * @brief Second-Order Generalized Integrator (SOGI) design (discrete-time)
 *
 * Discrete-time SOGI with bandpass and quadrature outputs.
 *
 * @param w0 Fundamental frequency [rad/s]
 * @param alpha Damping gain (typically 1.0-2.0)
 * @param Ts Sample time [s]
 * @param T Scalar type
 * @return StateSpace<2, 1, 2, 0, 0, T> Discrete-time SOGI system
 */
template<typename T = float>
[[nodiscard]] constexpr StateSpace<2, 1, 2, 0, 0, T> sogi_system(T w0, T alpha, T Ts) {
    const auto [sin_wt, cos_wt] = wet::sincos(w0 * Ts); // sin(wT), cos(wT)

    // Discrete closed-loop resonator with u_r = alpha * (u - bandpass)
    // and state x = [quadrature, bandpass]^T.
    const T alpha_one_minus_cos = alpha * (T{1} - cos_wt);
    const T alpha_sin = alpha * sin_wt;

    return {
        .A = Matrix<2, 2, T>{
            {cos_wt, sin_wt - alpha_one_minus_cos},
            {-sin_wt, cos_wt - alpha_sin},
        },

        .B = Matrix<2, 1, T>{
            {alpha_one_minus_cos},
            {alpha_sin},
        },

        .C = Matrix<2, 2, T>{
            {T{0}, T{1}}, // bandpass
            {T{1}, T{0}}, // quadrature
        },

        .D = Matrix<2, 1, T>::zeros(),
    };
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
 * post-gain bus `w = alpha·ω₀·(v − v′)`:
 *
 *     v̇′  = alpha·ω₀·(v − v′) − ω₀·v″   (SOGI in-phase integrator)
 *     v̇″  = ω₀·v′                   (SOGI quadrature integrator)
 *     v̇‴  = alpha·ω₀·(v − v′) − ω₀·v‴   (TOGI first-order, self-damped)
 *
 * Outputs: `v_o = v′` (band-pass, unity at ω₀) and `q·v_o = v″ − v‴`
 * (DC-rejecting quadrature). The transfer functions are
 *
 *     v_o / v   = alpha·ω₀·s / (s² + alpha·ω₀·s + ω₀²)
 *     q·v_o / v = alpha·ω₀·s·(ω₀ − s) / [ (s + ω₀)·(s² + alpha·ω₀·s + ω₀²) ]
 *
 * The q·v_o numerator has a zero at s = 0 (DC rejection) and the pair evaluates
 * to (1∠0°, 1∠−90°) at s = jω₀ — a clean unity-magnitude quadrature pair.
 *
 * @note Discretize with the exact ZOH (`DiscretizationMethod::ZOH`, the
 *       `mstogi_system(omega_0, alpha, Ts)` overload) so the resonant poles land
 *       exactly on the unit circle at z = e^{±jω₀Tₛ}; Tustin warps them.
 *
 * @see Rodríguez et al., "Discrete-time implementation of second order
 *      generalized integrators for grid converters," IECON 2008,
 *      https://doi.org/10.1109/IECON.2008.4757983
 *
 * @param omega_0 Fundamental frequency [rad/s]
 * @param alpha Damping gain (typically √2 for ~unity-Q SOGI tuning)
 * @param T Scalar type
 * @return StateSpace<3, 1, 2, 0, 0, T> MSTOGI system (outputs: v_o, q·v_o)
 */
template<typename T = double>
[[nodiscard]] constexpr StateSpace<3, 1, 2, 0, 0, T> mstogi_system(T w0, T alpha = wet::numbers::sqrt2_v<T>) {
    const T alpha_omega = alpha * w0;

    return {
        .A = Matrix<3, 3, T>{
            {-alpha_omega, -w0, T{0}},
            {w0, T{0}, T{0}},
            {-alpha_omega, T{0}, -w0},
        },

        .B = Matrix<3, 1, T>{
            {alpha_omega},
            {T{0}},
            {alpha_omega},
        },

        .C = Matrix<2, 3, T>{
            {T{1}, T{0}, T{0}},  // v_o   = v′      (band-pass)
            {T{0}, T{1}, T{-1}}, // q·v_o = v″ − v‴ (DC-rejecting quadrature)
        },

        .D = Matrix<2, 1, T>::zeros(),
    };
}

/**
 * @brief Mixed Second/Third-Order Generalized Integrator (MSTOGI) design (discrete-time)
 *
 * Exact-ZOH discretization of the continuous MSTOGI so the resonant poles sit
 * precisely at z = e^{±jω₀Tₛ} (no Tustin frequency warping).
 *
 * @param omega_0 Fundamental frequency [rad/s]
 * @param alpha Damping gain
 * @param Ts Sample time [s]
 * @param T Scalar type
 * @return StateSpace<3, 1, 2, 0, 0, T> Discrete-time MSTOGI system
 */
template<typename T = float>
[[nodiscard]] constexpr StateSpace<3, 1, 2, 0, 0, T> mstogi_system(T w0, T alpha, T Ts) {
    const T omega_tm = w0 * Ts;
    const auto [sin_wt, cos_wt] = wet::sincos(omega_tm); // sin(wT), cos(wT)

    // Closed-loop discrete realization matching runtime MSTOGI update:
    // u_r = alpha * (v - band_pass)
    // x_r1' = cos*x_r1 + (sin - alpha*(1-cos))*x_r2 + alpha*(1-cos)*v
    // x_r2' = -sin*x_r1 + (cos - alpha*sin)*x_r2 + alpha*sin*v
    // x_t'  = a*x_t - alpha*(1-a)*x_r2 + alpha*(1-a)*v, where a = exp(-wT)
    const T a_togi = wet::exp(-omega_tm);
    const T alpha_one_minus_cos = alpha * (T{1} - cos_wt);
    const T alpha_one_minus_a = alpha * (T{1} - a_togi);
    const T alpha_sin = alpha * sin_wt;

    return {
        .A = Matrix<3, 3, T>{
            {cos_wt, sin_wt - alpha_one_minus_cos, T{0}},
            {-sin_wt, cos_wt - alpha_sin, T{0}},
            {T{0}, -alpha_one_minus_a, a_togi},
        },

        .B = Matrix<3, 1, T>{
            {alpha_one_minus_cos},
            {alpha_sin},
            {alpha_one_minus_a},
        },

        .C = Matrix<2, 3, T>{
            {T{0}, T{1}, T{0}},  // band_pass
            {T{1}, T{0}, T{-1}}, // quadrature
        },

        .D = Matrix<2, 1, T>::zeros(),
    };
}

} // namespace design

/**
 * @brief Runtime SOGI wrapper around design::sogi_system(w0, alpha, Ts)
 *
 * This minimal runtime object stores state and performs one-step updates using
 * the discrete design realization each call:
 *
 *   y = Cx
 *   x = Ax + Bu
 *
 * where A/B/C come from `design::sogi_system(w0, alpha, Ts)` and
 * `w0 = 2*pi*freq`.
 *
 * @see "Understanding Digital Signal Processing" (Lyons, 2011), §13.36
 *
 * @tparam T Scalar type (float for embedded deployment)
 */
template<typename T = float>
class SOGI {
public:
    constexpr SOGI() = default;

    [[nodiscard]] constexpr wet::pair<T, T> operator()(T in, T freq, T alpha, T Ts) {
        const auto wT = freq * T{2} * wet::numbers::pi_v<T> * Ts;
        const auto [sin_wt, cos_wt] = wet::sincos(wT); // sin(wT), cos(wT)

        const StateSpace sys = {
            .A = Matrix<2, 2, T>{
                {cos_wt, sin_wt},  // Quadrature
                {-sin_wt, cos_wt}, // Band-pass
            },

            .B = ColVec<2, T>{
                T{1} - cos_wt,
                sin_wt,
            },

            .C = Matrix<2, 2, T>{
                {T{0}, T{1}}, // band-pass
                {T{1}, T{0}}, // quadrature
            },
        };

        const auto u = (in - x(1)) * alpha;

        y = sys.C * x;
        x = (sys.A * x) + (sys.B * u);

        return {y(0), y(1)};
    }

private:
    ColVec<2, T> x = {};
    ColVec<2, T> y = {};
};

/**
 * @brief Runtime MSTOGI with exact resonator and forward-Euler washout
 *
 * This minimal runtime object stores state and performs one-step updates each
 * call:
 *
 *   y = Cx
 *   x = Ax + Bu
 *
 * where `w0 = 2*pi*freq`.
 *
 * Runtime keeps the resonator on the exact sin/cos discretization and uses a
 * forward-Euler step for the TOGI washout branch:
 *
 *   x_t[k+1] = (1 - w0*Ts) * x_t[k] + alpha*w0*Ts * (u[k] - bp[k])
 *
 * This removes the per-sample exp() from the hot path while preserving DC
 * rejection behavior for practical sample rates (w0*Ts << 1).
 *
 * @see design::mstogi_system() for the exact-ZOH design model and transfer functions
 * @see Rodríguez et al., IECON 2008, https://doi.org/10.1109/IECON.2008.4757983
 *
 * @tparam T Scalar type (float for embedded deployment)
 */
template<typename T = float>
class MSTOGI {
public:
    constexpr MSTOGI() = default;

    [[nodiscard]] constexpr wet::pair<T, T> operator()(T in, T freq, T alpha, T Ts) {
        const T wT = freq * T{2} * wet::numbers::pi_v<T> * Ts;
        const auto [sin_wt, cos_wt] = wet::sincos(wT); // sin(wT), cos(wT)

        const StateSpace sys = {
            .A = Matrix<2, 2, T>{
                {cos_wt, sin_wt},
                {-sin_wt, cos_wt},
            },

            .B = ColVec<2, T>{
                T{1} - cos_wt,
                sin_wt,
            },

            .C = Matrix<2, 2, T>{
                {T{0}, T{1}}, // band-pass
                {T{1}, T{0}}, // quadrature
            },
        };

        // Update outputs
        const T u = (in - x(1)) * alpha;

        // Washout of DC offset in quadrature output
        togi_state += (u - togi_state) * wT;

        // Update internal state
        y = sys.C * x;
        x = (sys.A * x) + (sys.B * u);

        return {y(0), y(1) - togi_state};
    }

private:
    ColVec<2, T> x = {};
    ColVec<2, T> y = {};

    T togi_state = {};
};

} // namespace wet
