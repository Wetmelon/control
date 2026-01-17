#pragma once

#include <cmath>
#include <numbers>

#include "discretization.hpp"
#include "matrix.hpp"
#include "state_space.hpp"
#include "transfer_function.hpp"

namespace wetmelon::control {

namespace design {

/**
 * @brief DSP coefficients for first-order IIR filter (design-time)
 */
template<typename T = float>
struct FirstOrderCoeffs {
    T b0{0}, b1{0}, a1{0};

    /**
     * @brief Convert coefficients to different scalar type
     * @tparam U Target scalar type
     * @return FirstOrderCoeffs<U> with converted coefficients
     */
    template<typename U>
    [[nodiscard]] consteval FirstOrderCoeffs<U> as() const {
        return {static_cast<U>(b0), static_cast<U>(b1), static_cast<U>(a1)};
    }
};

/**
 * @brief DSP coefficients for second-order IIR filter (design-time)
 */
template<typename T = float>
struct SecondOrderCoeffs {
    T b0{0}, b1{0}, b2{0}, a1{0}, a2{0};

    /**
     * @brief Convert coefficients to different scalar type
     * @tparam U Target scalar type
     * @return SecondOrderCoeffs<U> with converted coefficients
     */
    template<typename U>
    [[nodiscard]] consteval SecondOrderCoeffs<U> as() const {
        return {static_cast<U>(b0), static_cast<U>(b1), static_cast<U>(b2), static_cast<U>(a1), static_cast<U>(a2)};
    }
};

} // namespace design

namespace online {

/**
 * @brief DSP coefficients for first-order IIR filter (runtime)
 */
template<typename T = float>
struct FirstOrderCoeffs {
    T b0{0}, b1{0}, a1{0};

    /**
     * @brief Convert coefficients to different scalar type
     * @tparam U Target scalar type
     * @return FirstOrderCoeffs<U> with converted coefficients
     */
    template<typename U>
    [[nodiscard]] constexpr FirstOrderCoeffs<U> as() const {
        return {static_cast<U>(b0), static_cast<U>(b1), static_cast<U>(a1)};
    }
};

/**
 * @brief DSP coefficients for second-order IIR filter (runtime)
 */
template<typename T = float>
struct SecondOrderCoeffs {
    T b0{0}, b1{0}, b2{0}, a1{0}, a2{0};

    /**
     * @brief Convert coefficients to different scalar type
     * @tparam U Target scalar type
     * @return SecondOrderCoeffs<U> with converted coefficients
     */
    template<typename U>
    [[nodiscard]] constexpr SecondOrderCoeffs<U> as() const {
        return {static_cast<U>(b0), static_cast<U>(b1), static_cast<U>(b2), static_cast<U>(a1), static_cast<U>(a2)};
    }
};

} // namespace online

namespace design {

/**
 * @defgroup filters Filter Design
 * @brief Continuous-time filter design functions
 *
 * Functions for designing common filters used in control systems.
 * All filters are designed in continuous-time and can be discretized.
 */

/**
 * @brief First-order low-pass filter design
 *
 * Designs a discrete-time first-order low-pass filter with given cutoff frequency.
 * Returns DSP coefficients ready for runtime use.
 *
 * @param fc Cutoff frequency [Hz]
 * @param Ts Sample time [s]
 * @param T Scalar type
 * @return FirstOrderCoeffs<T> DSP coefficients
 */
template<typename T = float>
[[nodiscard]] consteval FirstOrderCoeffs<T> lowpass_1st(T fc, T Ts) {
    // Design discrete-time first-order low-pass filter using bilinear transform
    const T omega_c = T{2} * std::numbers::pi_v<T> * fc;
    const T k = T{2} / Ts; // Pre-warping factor

    const T             denom = omega_c + k;
    FirstOrderCoeffs<T> coeffs = {
        .b0 = omega_c / denom,
        .b1 = omega_c / denom,
        .a1 = (omega_c - k) / denom,
    };

    return coeffs;
}

/**
 * @brief First-order low-pass filter design (continuous-time)
 *
 * Creates a continuous-time first-order low-pass filter with given cutoff frequency.
 * Returns transfer function for further processing or discretization.
 *
 * @param fc Cutoff frequency [Hz]
 * @param T Scalar type
 * @return TransferFunction<2, 2, T> continuous-time transfer function
 */
template<typename T = double>
[[nodiscard]] consteval TransferFunction<2, 2, T> lowpass_1st(T fc) {
    const T tau = T{1} / (T{2} * std::numbers::pi_v<T> * fc);
    return TransferFunction<2, 2, T>{{T{1}, T{0}}, {tau, T{1}}};
}

/**
 * @brief Second-order low-pass filter design
 *
 * Designs a discrete-time second-order low-pass filter with given cutoff frequency and damping.
 * Returns DSP coefficients ready for runtime use.
 *
 * @param fc Cutoff frequency [Hz]
 * @param Ts Sample time [s]
 * @param zeta Damping ratio (0.707 for Butterworth)
 * @param T Scalar type
 * @return SecondOrderCoeffs<T> DSP coefficients
 */
template<typename T = float>
[[nodiscard]] constexpr SecondOrderCoeffs<T> lowpass_2nd(T fc, T Ts, T zeta = T{0.707}) {
    // Design discrete-time second-order low-pass filter using bilinear transform
    const T omega_0 = T{2} * std::numbers::pi_v<T> * fc;
    const T k = T{2} / Ts; // Pre-warping factor for bilinear transform
    const T k_sq = k * k;
    const T omega_0_sq = omega_0 * omega_0;
    const T two_zeta_omega = T{2} * zeta * omega_0;

    // Denominator of continuous TF: s² + 2ζω₀s + ω₀²
    const T denom = omega_0_sq * k_sq - two_zeta_omega * k + T{1};

    // Bilinear transform coefficients
    SecondOrderCoeffs<T> coeffs;
    coeffs.b0 = omega_0_sq * k_sq / denom;
    coeffs.b1 = T{2} * omega_0_sq * k_sq / denom;
    coeffs.b2 = omega_0_sq * k_sq / denom;
    coeffs.a1 = (T{2} * omega_0_sq * k_sq - T{2}) / denom;
    coeffs.a2 = (omega_0_sq * k_sq + two_zeta_omega * k + T{1}) / denom;

    return coeffs;
}

/**
 * @brief Second-order low-pass filter design (continuous-time)
 *
 * Creates a continuous-time second-order low-pass filter with given cutoff frequency and damping.
 * Returns transfer function for further processing or discretization.
 *
 * @param fc Cutoff frequency [Hz]
 * @param zeta Damping ratio (0.707 for Butterworth)
 * @param T Scalar type
 * @return TransferFunction<3, 3, T> continuous-time transfer function
 */
template<typename T = double>
[[nodiscard]] consteval TransferFunction<3, 3, T> lowpass_2nd(T fc, T zeta = T{0.707}) {
    const T omega_0 = T{2} * std::numbers::pi_v<T> * fc;
    const T omega_0_sq = omega_0 * omega_0;
    const T two_zeta_omega = T{2} * zeta * omega_0;

    return TransferFunction<3, 3, T>{
        {omega_0_sq, T{0}, T{0}},          // num: ω₀²
        {T{1}, two_zeta_omega, omega_0_sq} // den: s² + 2ζω₀s + ω₀²
    };
}

/**
 * @brief Butterworth low-pass filter design
 *
 * Creates a continuous-time Butterworth low-pass filter of specified order.
 * Butterworth filters have maximally flat magnitude response in passband.
 *
 * @param fc Cutoff frequency [Hz]
 * @param order Filter order (1-4 supported)
 * @param T Scalar type
 * @return StateSpace system representing the filter
 */
template<size_t Order, typename T = double>
    requires(Order >= 1 && Order <= 4)
[[nodiscard]] consteval auto butterworth_lowpass(T fc) {
    const T omega_c = T{2} * std::numbers::pi_v<T> * fc;

    if constexpr (Order == 1) {
        return lowpass_1st<T>(fc).to_state_space();
    } else if constexpr (Order == 2) {
        // Butterworth order 2: ζ = 0.707
        return lowpass_2nd<T>(fc, T{0.707}).to_state_space();
    } else if constexpr (Order == 3) {
        // Butterworth order 3: cascade of order 1 and order 2
        auto tf1 = lowpass_1st<T>(fc);
        auto tf2 = lowpass_2nd<T>(fc, T{0.707});
        return (tf1 * tf2).to_state_space();
    } else { // Order == 4
        // Butterworth order 4: two cascaded order 2 sections
        auto tf2 = lowpass_2nd<T>(fc, T{0.707});
        return (tf2 * tf2).to_state_space();
    }
}

/**
 * @brief Butterworth low-pass filter design (discrete-time)
 *
 * Creates a discrete-time Butterworth low-pass filter of specified order.
 * Butterworth filters have maximally flat magnitude response in passband.
 *
 * @param fc Cutoff frequency [Hz]
 * @param Ts Sample time [s]
 * @param order Filter order (1-4 supported)
 * @param T Scalar type
 * @return StateSpace system representing the discrete-time filter
 */
template<size_t Order, typename T = float>
    requires(Order >= 1 && Order <= 4)
[[nodiscard]] consteval auto butterworth_lowpass(T fc, T Ts) {
    // Get continuous-time system
    auto sys_c = butterworth_lowpass<Order, T>(fc);

    // Discretize using bilinear transform
    return discretize(sys_c, Ts, DiscretizationMethod::Tustin);
}

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
[[nodiscard]] consteval StateSpace<2, 1, 2, 0, 0, T> sogi(T omega_0, T k = T{1.414}) {
    // State-space representation of SOGI
    // x1' = -k*ω₀*x1 - ω₀*x2 + k*ω₀*u
    // x2' = ω₀*x1

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
[[nodiscard]] consteval StateSpace<2, 1, 2, 0, 0, T> sogi(T omega_0, T k, T Ts) {
    // Start with continuous-time SOGI
    auto sys_c = sogi<T>(omega_0, k);

    // Discretize using Tustin (bilinear transform)
    return discretize(sys_c, Ts, DiscretizationMethod::Tustin);
}

/**
 * @brief Mixed Second-Third Order Generalized Integrator (MSOGI)
 *
 * Enhanced SOGI with high-pass filtering on quadrature output.
 * Better harmonic rejection than standard SOGI.
 *
 * @param omega_0 Fundamental frequency [rad/s]
 * @param k Damping gain
 * @param T Scalar type
 * @return StateSpace<3, 1, 2, 0, 0, T> MSOGI system
 */
template<typename T = double>
[[nodiscard]] consteval StateSpace<3, 1, 2, 0, 0, T> msogi(T omega_0, T k = T{1.414}) {
    StateSpace<3, 1, 2, 0, 0, T> sys{};

    // MSOGI adds a high-pass filter to the quadrature output
    // Extended state-space with HPF on x2

    const T k_omega = k * omega_0;
    const T omega_sq = omega_0 * omega_0;

    // A matrix
    sys.A(0, 0) = -k_omega;
    sys.A(0, 1) = -omega_0;
    sys.A(0, 2) = T{0};
    sys.A(1, 0) = omega_0;
    sys.A(1, 1) = T{0};
    sys.A(1, 2) = T{0};
    sys.A(2, 0) = T{0};
    sys.A(2, 1) = -omega_sq; // HPF on quadrature
    sys.A(2, 2) = T{0};

    // B matrix
    sys.B(0, 0) = k_omega;
    sys.B(1, 0) = T{0};
    sys.B(2, 0) = T{0};

    // C matrix (outputs: [bandpass, filtered quadrature])
    sys.C(0, 0) = T{1}; // x1 = bandpass
    sys.C(0, 1) = T{0};
    sys.C(0, 2) = T{0};
    sys.C(1, 0) = T{0};
    sys.C(1, 1) = T{0};
    sys.C(1, 2) = T{1}; // x3 = filtered quadrature

    // D matrix
    sys.D(0, 0) = T{0};
    sys.D(1, 0) = T{0};

    return sys;
}

/**
 * @brief Mixed Second-Third Order Generalized Integrator (MSOGI) design (discrete-time)
 *
 * Discrete-time MSOGI for enhanced grid synchronization with improved harmonic rejection.
 *
 * @param omega_0 Fundamental frequency [rad/s]
 * @param k Damping gain
 * @param Ts Sample time [s]
 * @param T Scalar type
 * @return StateSpace<3, 1, 2, 0, 0, T> Discrete-time MSOGI system
 */
template<typename T = float>
[[nodiscard]] consteval StateSpace<3, 1, 2, 0, 0, T> msogi(T omega_0, T k, T Ts) {
    // Start with continuous-time MSOGI
    auto sys_c = msogi<T>(omega_0, k);

    // Discretize using Zero-Order Hold (ZOH)
    return discretize(sys_c, Ts, DiscretizationMethod::ZOH);
}

/**
 * @brief First-order Pade approximation of time delay
 *
 * Approximates e^(-sT) with a first-order rational transfer function:
 * H(s) = (1 - sT/2) / (1 + sT/2)
 *
 * @param T_delay Time delay [s]
 * @param T Scalar type
 * @return TransferFunction<2, 2, T> representing the delay approximation
 */
template<typename T = double>
[[nodiscard]] consteval TransferFunction<2, 2, T> pade_delay_1st(T T_delay) {
    const T half_T = T_delay / T{2};
    return TransferFunction<2, 2, T>{
        {T{1}, -half_T}, // num: 1 - sT/2
        {T{1}, half_T}   // den: 1 + sT/2
    };
}

/**
 * @brief First-order Pade approximation of time delay (discrete-time)
 *
 * Discrete-time version of the first-order Pade delay approximation.
 *
 * @param T_delay Time delay [s]
 * @param Ts Sample time [s]
 * @param T Scalar type
 * @return TransferFunction<2, 2, T> discrete-time delay approximation
 */
template<typename T = float>
[[nodiscard]] consteval TransferFunction<2, 2, T> pade_delay_1st(T T_delay, T Ts) {
    // Get continuous-time approximation and discretize
    auto tf_c = pade_delay_1st<T>(T_delay);
    auto sys_c = tf_c.to_state_space();
    auto sys_d = discretize(sys_c, Ts, DiscretizationMethod::Tustin);
    return TransferFunction<2, 2, T>::from_state_space(sys_d);
}

/**
 * @brief Second-order Pade approximation of time delay
 *
 * Approximates e^(-sT) with a second-order rational transfer function:
 * H(s) = (1 - sT/2 + (sT)²/12) / (1 + sT/2 + (sT)²/12)
 *
 * @param T_delay Time delay [s]
 * @param T Scalar type
 * @return TransferFunction<3, 3, T> representing the delay approximation
 */
template<typename T = double>
[[nodiscard]] consteval TransferFunction<3, 3, T> pade_delay_2nd(T T_delay) {
    const T half_T = T_delay / T{2};
    const T T_sq_12 = T_delay * T_delay / T{12};
    return TransferFunction<3, 3, T>{
        {T{1}, -half_T, T_sq_12}, // num: 1 - sT/2 + (sT)²/12
        {T{1}, half_T, T_sq_12}   // den: 1 + sT/2 + (sT)²/12
    };
}

/**
 * @brief Second-order Pade approximation of time delay (discrete-time)
 *
 * Discrete-time version of the second-order Pade delay approximation.
 *
 * @param T_delay Time delay [s]
 * @param Ts Sample time [s]
 * @param T Scalar type
 * @return TransferFunction<3, 3, T> discrete-time delay approximation
 */
template<typename T = float>
[[nodiscard]] consteval TransferFunction<3, 3, T> pade_delay_2nd(T T_delay, T Ts) {
    // Get continuous-time approximation and discretize
    auto tf_c = pade_delay_2nd<T>(T_delay);
    auto sys_c = tf_c.to_state_space();
    auto sys_d = discretize(sys_c, Ts, DiscretizationMethod::Tustin);
    return TransferFunction<3, 3, T>::from_state_space(sys_d);
}

/**
 * @brief Convert StateSpace system to first-order DSP coefficients
 *
 * Assumes the system is first-order (1 state). If continuous-time, discretizes first.
 *
 * @param sys State-space system
 * @param Ts Sample time (only used if sys is continuous)
 * @param T Scalar type
 * @return FirstOrderCoeffs<T> DSP coefficients
 */
template<typename T = float>
[[nodiscard]] consteval FirstOrderCoeffs<T> to_coeffs(const StateSpace<1, 1, 1, 0, 0, T>& sys, T Ts = T{0}) {
    StateSpace<1, 1, 1, 0, 0, T> sys_d = sys;
    if (sys.Ts == T{0}) {
        // Discretize if continuous
        sys_d = discretize(sys, Ts, DiscretizationMethod::Tustin);
    }

    FirstOrderCoeffs<T> coeffs;
    coeffs.b0 = sys_d.D(0, 0);
    coeffs.b1 = sys_d.C(0, 0) * sys_d.B(0, 0) - sys_d.D(0, 0) * sys_d.A(0, 0);
    coeffs.a1 = -sys_d.A(0, 0);
    return coeffs;
}

/**
 * @brief Convert TransferFunction to first-order DSP coefficients
 *
 * Assumes the transfer function is first-order. Discretizes using bilinear transform.
 *
 * @param tf Transfer function
 * @param Ts Sample time
 * @param T Scalar type
 * @return FirstOrderCoeffs<T> DSP coefficients
 */
template<typename T = float>
[[nodiscard]] consteval FirstOrderCoeffs<T> to_coeffs(const TransferFunction<1, 2, T>& tf, T Ts) {
    const auto sys_d = discretize(tf.to_state_space(), Ts, DiscretizationMethod::Tustin);
    return to_coeffs(sys_d);
}

template<typename T = float>
[[nodiscard]] consteval FirstOrderCoeffs<T> to_coeffs(const TransferFunction<2, 2, T>& tf, T Ts) {
    const auto sys_d = discretize(tf.to_state_space(), Ts, DiscretizationMethod::Tustin);
    return to_coeffs(sys_d);
}

/**
 * @brief Convert StateSpace system to second-order DSP coefficients
 *
 * Assumes the system is second-order (2 states). If continuous-time, discretizes first.
 *
 * @param sys State-space system
 * @param Ts Sample time (only used if sys is continuous)
 * @param T Scalar type
 * @return SecondOrderCoeffs<T> DSP coefficients
 */
template<typename T = float>
[[nodiscard]] consteval SecondOrderCoeffs<T> to_coeffs(const StateSpace<2, 1, 1, 0, 0, T>& sys, T Ts = T{0}) {
    StateSpace<2, 1, 1, 0, 0, T> sys_d = sys;
    if (sys.Ts == T{0}) {
        // Discretize if continuous
        sys_d = discretize(sys, Ts, DiscretizationMethod::Tustin);
    }

    SecondOrderCoeffs<T> coeffs;
    coeffs.b0 = sys_d.D(0, 0);
    coeffs.b1 = sys_d.C(0, 0) * sys_d.B(0, 0) + sys_d.C(0, 1) * sys_d.B(1, 0);
    coeffs.b2 = sys_d.C(0, 1) * sys_d.B(1, 0);
    coeffs.a1 = -(sys_d.A(0, 0) + sys_d.A(1, 1));
    coeffs.a2 = sys_d.A(0, 0) * sys_d.A(1, 1) - sys_d.A(0, 1) * sys_d.A(1, 0);
    return coeffs;
}

/**
 * @brief Convert TransferFunction to second-order DSP coefficients
 *
 * Assumes the transfer function is second-order. Discretizes using bilinear transform.
 *
 * @param tf Transfer function
 * @param Ts Sample time
 * @param T Scalar type
 * @return SecondOrderCoeffs<T> DSP coefficients
 */
template<typename T = float>
[[nodiscard]] consteval SecondOrderCoeffs<T> to_coeffs(const TransferFunction<3, 3, T>& tf, T Ts) {
    const auto sys_d = discretize(tf.to_state_space(), Ts, DiscretizationMethod::Tustin);
    return to_coeffs(sys_d);
}

} // namespace design

namespace online {

/**
 * @brief Runtime filter implementations
 *
 * These are discretized versions of the design-time filters for runtime use.
 */

/**
 * @brief First-order low-pass filter design (runtime)
 *
 * Computes discrete-time first-order low-pass filter coefficients at runtime.
 * Returns DSP coefficients ready for filter construction.
 *
 * @param fc Cutoff frequency [Hz]
 * @param Ts Sample time [s]
 * @param T Scalar type
 * @return FirstOrderCoeffs<T> DSP coefficients
 */
template<typename T = float>
[[nodiscard]] constexpr FirstOrderCoeffs<T> lowpass_1st(T fc, T Ts) {
    // Design discrete-time first-order low-pass filter using bilinear transform
    const T omega_c = T{2} * std::numbers::pi_v<T> * fc;
    const T k = T{2} / Ts; // Pre-warping factor

    const T             denom = omega_c + k;
    FirstOrderCoeffs<T> coeffs = {
        .b0 = omega_c / denom,
        .b1 = omega_c / denom,
        .a1 = (omega_c - k) / denom,
    };

    return coeffs;
}

/**
 * @brief Second-order low-pass filter design (runtime)
 *
 * Computes discrete-time second-order low-pass filter coefficients at runtime.
 * Returns DSP coefficients ready for filter construction.
 *
 * @param fc Cutoff frequency [Hz]
 * @param Ts Sample time [s]
 * @param zeta Damping ratio (0.707 for Butterworth)
 * @param T Scalar type
 * @return SecondOrderCoeffs<T> DSP coefficients
 */
template<typename T = float>
[[nodiscard]] constexpr SecondOrderCoeffs<T> lowpass_2nd(T fc, T Ts, T zeta = T{0.707}) {
    // Design discrete-time second-order low-pass filter using bilinear transform
    const T omega_0 = T{2} * std::numbers::pi_v<T> * fc;
    const T k = T{2} / Ts; // Pre-warping factor for bilinear transform
    const T k_sq = k * k;
    const T omega_0_sq = omega_0 * omega_0;
    const T two_zeta_omega = T{2} * zeta * omega_0;

    // Denominator of continuous TF: s² + 2ζω₀s + ω₀²
    const T denom = omega_0_sq * k_sq - two_zeta_omega * k + T{1};

    // Bilinear transform coefficients
    SecondOrderCoeffs<T> coeffs;
    coeffs.b0 = omega_0_sq * k_sq / denom;
    coeffs.b1 = T{2} * omega_0_sq * k_sq / denom;
    coeffs.b2 = omega_0_sq * k_sq / denom;
    coeffs.a1 = (T{2} * omega_0_sq * k_sq - T{2}) / denom;
    coeffs.a2 = (omega_0_sq * k_sq + two_zeta_omega * k + T{1}) / denom;

    return coeffs;
}

} // namespace online

/**
 * @brief Nth-order low-pass filter
 * @tparam N filter order
 */
template<size_t N, typename T = float>
class LowPass {
private:
    std::array<T, N + 1> b{};      //!< Numerator coefficients
    std::array<T, N>     a{};      //!< Denominator coefficients
    std::array<T, N>     x_prev{}; //!< Previous inputs
    std::array<T, N>     y_prev{}; //!< Previous outputs

public:
    /**
     * @brief Simple constructor with cutoff frequency and sample time
     *
     * Fast online design of 1st order filter using ZOH method.
     *
     * @param fc Cutoff frequency [Hz]
     * @param Ts_sample Sample time [s]
     *
     */
    constexpr LowPass(T fc, T Ts_sample)
        requires(N == 1)
    {
        *this = LowPass<1, T>(online::lowpass_1st<T>(fc, Ts_sample));
    }

    constexpr LowPass(const TransferFunction<N + 1, N + 1, T> tf, T Ts_sample) {
        auto sys_c = tf.to_state_space();
        auto sys_d = discretize(sys_c, Ts_sample, DiscretizationMethod::Tustin);

        // Compute impulse response
        constexpr size_t MaxK = 2 * N;

        std::array<T, MaxK + 1> h{};
        h[0] = sys_d.D(0, 0);
        Matrix<N, N, T> A_pow{};
        for (size_t i = 0; i < N; ++i)
            A_pow(i, i) = T{1};
        for (size_t k = 1; k <= MaxK; ++k) {
            Matrix<N, 1, T> temp = A_pow * sys_d.B;
            h[k] = (sys_d.C * temp)(0, 0);
            A_pow = A_pow * sys_d.A;
        }

        // Set b coefficients
        for (size_t i = 0; i <= N; ++i)
            b[i] = h[i];

        // Compute a coefficients
        if constexpr (N > 0) {
            Matrix<N, N, T> M{};
            Matrix<N, 1, T> v{};
            for (size_t i = 0; i < N; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    M(i, j) = h[N + 1 + i - (j + 1)];
                }
                v(i, 0) = -h[N + 1 + i];
            }
            auto M_inv = M.inverse();
            // Assume invertible for valid systems
            auto a_vec = *M_inv * v;
            for (size_t j = 0; j < N; ++j) {
                a[j] = a_vec(j, 0);
            }
        }
    }

    constexpr LowPass(const std::array<T, N + 1>& b_, const std::array<T, N>& a_)
        : b(b_), a(a_), x_prev{}, y_prev{} {}

    constexpr LowPass(const online::FirstOrderCoeffs<T>& coeffs)
        requires(N == 1)
        : b{coeffs.b0, coeffs.b1}, a{coeffs.a1}, x_prev{}, y_prev{} {}

    constexpr auto operator()(T x) {
        T y = b[0] * x;
        for (size_t i = 0; i < N; ++i) {
            y += b[i + 1] * x_prev[i];
            y -= a[i] * y_prev[i];
        }

        // Update previous states
        for (size_t i = N - 1; i > 0; --i) {
            x_prev[i] = x_prev[i - 1];
            y_prev[i] = y_prev[i - 1];
        }
        if (N > 0) {
            x_prev[0] = x;
            y_prev[0] = y;
        }

        return y;
    }

    constexpr void reset() {
        x_prev = {};
        y_prev = {};
    }

    // Default constructor
    constexpr LowPass() = default;

    // Copy constructors
    constexpr LowPass(const LowPass& other) = default;
    constexpr LowPass& operator=(const LowPass& other) = default;

    // Move constructors
    constexpr LowPass(LowPass&& other) noexcept = default;
    constexpr LowPass& operator=(LowPass&& other) noexcept = default;
};

/**
 * @brief SOGI filter (runtime)
 */
template<typename T = float>
class SOGI {
public:
    T omega_0{0};   //!< Fundamental frequency [rad/s]
    T k{1.414};     //!< Damping gain
    T Ts{0};        //!< Sample time [s]
    T x1{0}, x2{0}; //!< States

    // Discrete system matrices (for ZOH/Tustin)
    Matrix<2, 2, T> A_d;
    Matrix<2, 1, T> B_d;
    Matrix<2, 2, T> C_d;
    Matrix<2, 1, T> D_d;

    DiscretizationMethod method{DiscretizationMethod::ForwardEuler}; //!< Discretization method

    /**
     * @brief Default constructor (uninitialized)
     */
    constexpr SOGI() = default;

    /**
     * @brief Construct with fundamental frequency, damping, sample time
     * @param omega_0 Fundamental frequency [rad/s]
     * @param k Damping gain
     * @param Ts Sample time [s]
     * @param method Discretization method (default: ForwardEuler)
     */
    constexpr SOGI(T omega_0_, T k_, T Ts_, DiscretizationMethod method_ = DiscretizationMethod::ForwardEuler)
        : omega_0(omega_0_), k(k_), Ts(Ts_), method(method_) {
        if (method != DiscretizationMethod::ForwardEuler) {
            // Pre-compute discrete matrices for ZOH/Tustin
            // Inline continuous-time SOGI system creation
            StateSpace<2, 1, 2, 0, 0, T> sys_c{
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

            auto sys_d = discretize(sys_c, Ts, method);
            A_d = sys_d.A;
            B_d = sys_d.B;
            C_d = sys_d.C;
            D_d = sys_d.D;
        }
    }

    /**
     * @brief Process input sample
     * @param u Input sample
     * @return std::pair<T, T> {bandpass_output, quadrature_output}
     */
    constexpr std::pair<T, T> operator()(T u) {
        if (method == DiscretizationMethod::ForwardEuler) {
            // Forward Euler discretization (original implementation)
            const T x1_dot = -k * omega_0 * x1 - omega_0 * x2 + k * omega_0 * u;
            const T x2_dot = omega_0 * x1;

            x1 += x1_dot * Ts;
            x2 += x2_dot * Ts;

            return {x1, x2}; // {bandpass, quadrature}
        } else {
            // ZOH or Tustin: use pre-computed discrete matrices
            Matrix<2, 1, T> x_vec{{x1}, {x2}};
            Matrix<1, 1, T> u_vec{{u}};

            // State update: x[k+1] = A_d * x[k] + B_d * u[k]
            Matrix<2, 1, T> x_new = A_d * x_vec + B_d * u_vec;

            x1 = x_new(0, 0);
            x2 = x_new(1, 0);

            // Output: y[k] = C_d * x[k] + D_d * u[k]
            Matrix<2, 1, T> y = C_d * x_vec + D_d * u_vec;

            return {y(0, 0), y(1, 0)}; // {bandpass, quadrature}
        }
    }

    /**
     * @brief Reset SOGI state
     */
    constexpr void reset() {
        x1 = x2 = T{0};
    }
};

/**
 * @brief Discrete-time delay buffer
 *
 * Implements a fixed-size delay buffer for discrete-time systems.
 * Useful for modeling transport delays, computational delays, etc.
 */
template<size_t MaxDelay, typename T = float>
class Delay {
private:
    std::array<T, MaxDelay> buffer_{};         //!< Circular buffer for delayed samples
    size_t                  write_idx_{0};     //!< Current write position
    size_t                  delay_samples_{1}; //!< Number of samples to delay

public:
    /**
     * @brief Initialize delay buffer
     * @param delay_samples Number of samples to delay (must be <= MaxDelay)
     */
    constexpr void init(size_t delay_samples) {
        delay_samples_ = delay_samples;
        reset();
    }

    /**
     * @brief Process input sample and return delayed output
     * @param input Current input sample
     * @return Delayed output sample
     */
    constexpr T operator()(T input) {
        // Store current input
        buffer_[write_idx_] = input;

        // Calculate read position
        size_t read_idx = (write_idx_ + MaxDelay - delay_samples_) % MaxDelay;

        // Get delayed output
        T output = buffer_[read_idx];

        // Update write index
        write_idx_ = (write_idx_ + 1) % MaxDelay;

        return output;
    }

    /**
     * @brief Reset delay buffer to zero
     */
    constexpr void reset() {
        buffer_ = {};
        write_idx_ = 0;
    }

    /**
     * @brief Get current delay in samples
     * @return Number of delay samples
     */
    constexpr size_t get_delay_samples() const { return delay_samples_; }
};

} // namespace wetmelon::control
