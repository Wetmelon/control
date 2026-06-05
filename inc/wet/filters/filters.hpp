#pragma once

#include <cmath>
#include <numbers>

#include "wet/math/wetmelon_math.hpp"
#include "wet/matrix/matrix.hpp"
#include "wet/systems/discretization.hpp"
#include "wet/systems/state_space.hpp"
#include "wet/systems/transfer_function.hpp"

namespace wet {

namespace design {

/**
 * @brief DSP coefficients for first-order IIR filter
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
 * @brief DSP coefficients for second-order IIR filter
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

/**
 * @defgroup filters Filter Design
 * @brief Filter design and filter coefficient functions
 *
 * Functions for designing common filters used in control systems.
 * Filters can be designed in continuous-time or discrete-time, with
 * compile-time and runtime support through constexpr evaluation.
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
[[nodiscard]] constexpr TransferFunction<2, 2, T> lowpass_1st(T fc) {
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
    // Discretize H(s) = ω₀² / (s² + 2ζω₀s + ω₀²) with the bilinear transform
    // s ← k·(1 − z⁻¹)/(1 + z⁻¹), k = 2/Ts. Clearing (1 + z⁻¹)² gives a numerator
    // proportional to (1 + z⁻¹)² (tap shape 1 : 2 : 1) over a denominator with
    //   a0 = k² + 2ζω₀k + ω₀².
    const T omega_0 = T{2} * std::numbers::pi_v<T> * fc;
    const T k = T{2} / Ts; // bilinear gain (cutoff assumed well below Nyquist; no pre-warp)
    const T k_sq = k * k;
    const T omega_0_sq = omega_0 * omega_0;
    const T two_zeta_omega = T{2} * zeta * omega_0;

    const T a0 = k_sq + two_zeta_omega * k + omega_0_sq;

    SecondOrderCoeffs<T> coeffs;
    coeffs.a1 = (T{2} * omega_0_sq - T{2} * k_sq) / a0;
    coeffs.a2 = (k_sq - two_zeta_omega * k + omega_0_sq) / a0;

    // The numerator is ω₀²·(1 + z⁻¹)², so the taps are in fixed ratio 1 : 2 : 1 and
    // their only free parameter is the overall scale. Pin that scale from the
    // unit-DC-gain identity H(1) = 1 ⇔ (b0 + b1 + b2) = (1 + a1 + a2): deriving the
    // taps from that sum makes unity DC gain hold *by construction*, so it survives
    // -ffast-math reassociation in downstream builds instead of depending on the raw
    // bilinear numerator/denominator terms cancelling exactly.
    const T dc_sum = T{1} + coeffs.a1 + coeffs.a2;
    coeffs.b0 = dc_sum / T{4};
    coeffs.b1 = dc_sum / T{2};
    coeffs.b2 = dc_sum / T{4};

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
[[nodiscard]] constexpr TransferFunction<3, 3, T> lowpass_2nd(T fc, T zeta = T{0.707}) {
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
[[nodiscard]] constexpr auto butterworth_lowpass(T fc) {
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
[[nodiscard]] constexpr auto butterworth_lowpass(T fc, T Ts) {
    // Get continuous-time system
    auto sys_c = butterworth_lowpass<Order, T>(fc);

    // Discretize using bilinear transform
    return discretize(sys_c, Ts, DiscretizationMethod::Tustin);
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
[[nodiscard]] constexpr TransferFunction<2, 2, T> pade_delay_1st(T T_delay) {
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
[[nodiscard]] constexpr TransferFunction<2, 2, T> pade_delay_1st(T T_delay, T Ts) {
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
[[nodiscard]] constexpr TransferFunction<3, 3, T> pade_delay_2nd(T T_delay) {
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
[[nodiscard]] constexpr TransferFunction<3, 3, T> pade_delay_2nd(T T_delay, T Ts) {
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
[[nodiscard]] constexpr FirstOrderCoeffs<T> to_coeffs(const StateSpace<1, 1, 1, 0, 0, T>& sys, T Ts = T{0}) {
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
[[nodiscard]] constexpr FirstOrderCoeffs<T> to_coeffs(const TransferFunction<1, 2, T>& tf, T Ts) {
    const auto sys_d = discretize(tf.to_state_space(), Ts, DiscretizationMethod::Tustin);
    return to_coeffs(sys_d);
}

template<typename T = float>
[[nodiscard]] constexpr FirstOrderCoeffs<T> to_coeffs(const TransferFunction<2, 2, T>& tf, T Ts) {
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
[[nodiscard]] constexpr SecondOrderCoeffs<T> to_coeffs(const StateSpace<2, 1, 1, 0, 0, T>& sys, T Ts = T{0}) {
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
[[nodiscard]] constexpr SecondOrderCoeffs<T> to_coeffs(const TransferFunction<3, 3, T>& tf, T Ts) {
    const auto sys_d = discretize(tf.to_state_space(), Ts, DiscretizationMethod::Tustin);
    return to_coeffs(sys_d);
}

// ============================================================================
// Biquad (second-order IIR) designs — RBJ audio-EQ cookbook formulas
// ============================================================================
//
// Coefficients use the normalized difference equation
//   y[n] = b0·x[n] + b1·x[n-1] + b2·x[n-2] − a1·y[n-1] − a2·y[n-2]
// matching SecondOrderCoeffs and the Biquad runtime. Frequencies are designed
// directly in the digital domain: ω₀ = 2π·f·Ts. Quality factor Q controls
// bandwidth (BW in octaves ≈ asinh(1/2Q)·2/ln2 near ω₀); larger Q is narrower.

namespace detail {
template<typename T>
[[nodiscard]] constexpr SecondOrderCoeffs<T> normalize_biquad(T b0, T b1, T b2, T a0, T a1, T a2) {
    const T inv = T{1} / a0;
    return SecondOrderCoeffs<T>{b0 * inv, b1 * inv, b2 * inv, a1 * inv, a2 * inv};
}
} // namespace detail

/**
 * @brief Second-order band-reject (notch) filter.
 *
 * Rejects a narrow band around f0 (gain → 0 at f0) while passing the rest
 * (gain → 1). Transfer function:
 *
 *     H(z) = (1 − 2cosω₀ z⁻¹ + z⁻²) / (1 + α) / (… z⁻¹ …),  α = sinω₀ / (2Q)
 *
 * @note Compare with MATLAB's iirnotch(w0, bw).
 * @param f0 Notch (center) frequency [Hz]
 * @param Q  Quality factor (higher = narrower notch)
 * @param Ts Sample time [s]
 * @return SecondOrderCoeffs<T> normalized biquad coefficients
 * @see "Cookbook formulae for audio EQ biquad filter coefficients" (Bristow-Johnson)
 */
template<typename T = float>
[[nodiscard]] constexpr SecondOrderCoeffs<T> notch(T f0, T Q, T Ts) {
    const T w0 = T{2} * std::numbers::pi_v<T> * f0 * Ts;
    const T cw = wet::cos(w0);
    const T alpha = wet::sin(w0) / (T{2} * Q);
    return detail::normalize_biquad<T>(T{1}, T{-2} * cw, T{1}, T{1} + alpha, T{-2} * cw, T{1} - alpha);
}

/**
 * @brief Second-order band-pass filter (constant 0 dB peak gain).
 *
 * Passes a band around f0 (gain → 1 at f0), rejecting DC and high frequencies.
 *
 * @param f0 Center frequency [Hz]
 * @param Q  Quality factor (higher = narrower band)
 * @param Ts Sample time [s]
 * @return SecondOrderCoeffs<T> normalized biquad coefficients
 * @see "Cookbook formulae for audio EQ biquad filter coefficients" (Bristow-Johnson)
 */
template<typename T = float>
[[nodiscard]] constexpr SecondOrderCoeffs<T> bandpass(T f0, T Q, T Ts) {
    const T w0 = T{2} * std::numbers::pi_v<T> * f0 * Ts;
    const T cw = wet::cos(w0);
    const T alpha = wet::sin(w0) / (T{2} * Q);
    return detail::normalize_biquad<T>(alpha, T{0}, -alpha, T{1} + alpha, T{-2} * cw, T{1} - alpha);
}

/**
 * @brief Second-order high-pass filter (RBJ).
 *
 * Counterpart to lowpass_2nd. Note this family is parameterized by Q
 * (= 1 / 2ζ); the default Q = 1/√2 is the maximally-flat (Butterworth) response.
 *
 * @param fc Cutoff frequency [Hz]
 * @param Ts Sample time [s]
 * @param Q  Quality factor (default 1/√2 = Butterworth)
 * @return SecondOrderCoeffs<T> normalized biquad coefficients
 * @see lowpass_2nd()
 */
template<typename T = float>
[[nodiscard]] constexpr SecondOrderCoeffs<T> highpass_2nd(T fc, T Ts, T Q = T{1} / std::numbers::sqrt2_v<T>) {
    const T w0 = T{2} * std::numbers::pi_v<T> * fc * Ts;
    const T cw = wet::cos(w0);
    const T alpha = wet::sin(w0) / (T{2} * Q);
    const T b0 = (T{1} + cw) / T{2};
    return detail::normalize_biquad<T>(b0, -(T{1} + cw), b0, T{1} + alpha, T{-2} * cw, T{1} - alpha);
}

/**
 * @brief Peaking (bell) EQ filter: boost or cut a band around f0.
 *
 * @param f0      Center frequency [Hz]
 * @param Q       Quality factor (higher = narrower bell)
 * @param gain_db Peak gain at f0 [dB] (positive = boost, negative = cut)
 * @param Ts      Sample time [s]
 * @return SecondOrderCoeffs<T> normalized biquad coefficients
 * @see "Cookbook formulae for audio EQ biquad filter coefficients" (Bristow-Johnson)
 */
template<typename T = float>
[[nodiscard]] constexpr SecondOrderCoeffs<T> peaking(T f0, T Q, T gain_db, T Ts) {
    const T A = wet::pow(T{10}, gain_db / T{40});
    const T w0 = T{2} * std::numbers::pi_v<T> * f0 * Ts;
    const T cw = wet::cos(w0);
    const T alpha = wet::sin(w0) / (T{2} * Q);
    return detail::normalize_biquad<T>(
        T{1} + (alpha * A), T{-2} * cw, T{1} - (alpha * A),
        T{1} + (alpha / A), T{-2} * cw, T{1} - (alpha / A)
    );
}

/**
 * @brief Low-shelf EQ filter: boost or cut everything below fc.
 *
 * @param fc      Shelf corner frequency [Hz]
 * @param gain_db Shelf gain [dB]
 * @param Ts      Sample time [s]
 * @param Q       Shelf shape (default 1/√2)
 * @return SecondOrderCoeffs<T> normalized biquad coefficients
 * @see "Cookbook formulae for audio EQ biquad filter coefficients" (Bristow-Johnson)
 */
template<typename T = float>
[[nodiscard]] constexpr SecondOrderCoeffs<T> lowshelf(T fc, T gain_db, T Ts, T Q = T{1} / std::numbers::sqrt2_v<T>) {
    const T A = wet::pow(T{10}, gain_db / T{40});
    const T w0 = T{2} * std::numbers::pi_v<T> * fc * Ts;
    const T cw = wet::cos(w0);
    const T alpha = wet::sin(w0) / (T{2} * Q);
    const T tsa = T{2} * wet::sqrt(A) * alpha;
    const T Ap = A + T{1};
    const T Am = A - T{1};
    return detail::normalize_biquad<T>(
        A * (Ap - (Am * cw) + tsa),
        T{2} * A * (Am - (Ap * cw)),
        A * (Ap - (Am * cw) - tsa),
        Ap + (Am * cw) + tsa,
        T{-2} * (Am + (Ap * cw)),
        Ap + (Am * cw) - tsa
    );
}

/**
 * @brief High-shelf EQ filter: boost or cut everything above fc.
 *
 * @param fc      Shelf corner frequency [Hz]
 * @param gain_db Shelf gain [dB]
 * @param Ts      Sample time [s]
 * @param Q       Shelf shape (default 1/√2)
 * @return SecondOrderCoeffs<T> normalized biquad coefficients
 * @see "Cookbook formulae for audio EQ biquad filter coefficients" (Bristow-Johnson)
 */
template<typename T = float>
[[nodiscard]] constexpr SecondOrderCoeffs<T> highshelf(T fc, T gain_db, T Ts, T Q = T{1} / std::numbers::sqrt2_v<T>) {
    const T A = wet::pow(T{10}, gain_db / T{40});
    const T w0 = T{2} * std::numbers::pi_v<T> * fc * Ts;
    const T cw = wet::cos(w0);
    const T alpha = wet::sin(w0) / (T{2} * Q);
    const T tsa = T{2} * wet::sqrt(A) * alpha;
    const T Ap = A + T{1};
    const T Am = A - T{1};
    return detail::normalize_biquad<T>(
        A * (Ap + (Am * cw) + tsa),
        T{-2} * A * (Am + (Ap * cw)),
        A * (Ap + (Am * cw) - tsa),
        Ap - (Am * cw) + tsa,
        T{2} * (Am - (Ap * cw)),
        Ap - (Am * cw) - tsa
    );
}
} // namespace design
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
        *this = LowPass<1, T>(design::lowpass_1st<T>(fc, Ts_sample));
    }

    constexpr LowPass(const TransferFunction<N + 1, N + 1, T> tf, T Ts_sample) {
        auto sys_c = tf.to_state_space();
        auto sys_d = discretize(sys_c, Ts_sample, DiscretizationMethod::Tustin);

        // Compute impulse response
        constexpr size_t MaxK = 2 * N;

        std::array<T, MaxK + 1> h{};
        h[0] = sys_d.D(0, 0);
        Matrix<N, N, T> A_pow{};
        for (size_t i = 0; i < N; ++i) {
            A_pow(i, i) = T{1};
        }
        for (size_t k = 1; k <= MaxK; ++k) {
            Matrix<N, 1, T> temp = A_pow * sys_d.B;
            h[k] = (sys_d.C * temp)(0, 0);
            A_pow = A_pow * sys_d.A;
        }

        // Set b coefficients
        for (size_t i = 0; i <= N; ++i) {
            b[i] = h[i];
        }

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

    constexpr LowPass(const design::FirstOrderCoeffs<T>& coeffs)
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

    constexpr ~LowPass() = default;
};

/**
 * @brief Second-order IIR (biquad) section runtime.
 *
 * Runs a SecondOrderCoeffs section in Direct Form I:
 *   y[n] = b0·x[n] + b1·x[n-1] + b2·x[n-2] − a1·y[n-1] − a2·y[n-2]
 *
 * Pair with any design:: biquad designer (notch, bandpass, highpass_2nd,
 * peaking, lowshelf, highshelf, lowpass_2nd).
 *
 * @code
 * Biquad<float> notch{design::notch(50.0f, 5.0f, 1.0f / 1000.0f)};
 * float clean = notch(sample);
 * @endcode
 */
template<typename T = float>
class Biquad {
public:
    constexpr Biquad() = default;

    constexpr explicit Biquad(const design::SecondOrderCoeffs<T>& c)
        : b0_(c.b0), b1_(c.b1), b2_(c.b2), a1_(c.a1), a2_(c.a2) {}

    /// Process one sample.
    constexpr T operator()(T x) {
        const T y = (b0_ * x) + (b1_ * x1_) + (b2_ * x2_) - (a1_ * y1_) - (a2_ * y2_);
        x2_ = x1_;
        x1_ = x;
        y2_ = y1_;
        y1_ = y;
        return y;
    }

    /// Reset the internal delay line.
    constexpr void reset() {
        x1_ = x2_ = y1_ = y2_ = T{0};
    }

private:
    T b0_{1}, b1_{0}, b2_{0}, a1_{0}, a2_{0};
    T x1_{0}, x2_{0}, y1_{0}, y2_{0};
};

/**
 * @brief Cascade of second-order sections (SOS) for higher-order IIR filters.
 *
 * Chains NSections biquads in series. Cascading is the numerically preferred
 * realization for higher-order IIR filters (vs. a single high-order section).
 *
 * @tparam NSections Number of biquad sections
 * @tparam T         Scalar type
 */
template<size_t NSections, typename T = float>
class BiquadCascade {
public:
    constexpr BiquadCascade() = default;

    constexpr explicit BiquadCascade(const std::array<design::SecondOrderCoeffs<T>, NSections>& sections) {
        for (size_t i = 0; i < NSections; ++i) {
            sections_[i] = Biquad<T>(sections[i]);
        }
    }

    /// Process one sample through all sections in series.
    constexpr T operator()(T x) {
        for (size_t i = 0; i < NSections; ++i) {
            x = sections_[i](x);
        }
        return x;
    }

    /// Reset every section.
    constexpr void reset() {
        for (size_t i = 0; i < NSections; ++i) {
            sections_[i].reset();
        }
    }

private:
    std::array<Biquad<T>, NSections> sections_{};
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

} // namespace wet
