#pragma once

/**
 * @defgroup pr_controller Proportional-Resonant Controller
 * @brief PR controller for AC reference tracking
 *
 * PR controllers provide infinite gain at a specific resonant frequency,
 * achieving zero steady-state error for sinusoidal references. They are
 * the AC equivalent of PI controllers for DC references.
 *
 * Common applications:
 * - Grid-tied inverter current control
 * - Active power filter harmonic compensation
 * - Rotating reference frame alternatives (no Park transform needed)
 *
 * Transfer function: C(s) = Kp + 2*Ki*wc*s / (s² + 2*wc*s + w0²)
 * where:
 *   Kp = proportional gain
 *   Ki = resonant (integral) gain
 *   w0 = resonant frequency (e.g. 2*pi*50 for 50Hz grid)
 *   wc = cutoff frequency / bandwidth of the resonant term (for non-ideal PR)
 *
 * An ideal PR has wc = 0: C(s) = Kp + Ki*s / (s² + w0²)
 * but this has zero bandwidth around w0 and is sensitive to frequency variations.
 */

#include <cmath>
#include <numbers>

#include "constexpr_math.hpp"
#include "state_space.hpp"

namespace wetmelon::control {

namespace design {

/**
 * @struct PRResult
 * @brief Proportional-Resonant controller design result
 */
template<typename T = double>
struct PRResult {
    T Kp{}; //!< Proportional gain
    T Ki{}; //!< Resonant (integral) gain
    T w0{}; //!< Resonant frequency (rad/s)
    T wc{}; //!< Cutoff bandwidth of resonant term (rad/s), 0 = ideal
    T Ts{}; //!< Sampling time (seconds)

    template<typename U>
    [[nodiscard]] consteval auto as() const {
        return PRResult<U>{
            static_cast<U>(Kp), static_cast<U>(Ki),
            static_cast<U>(w0), static_cast<U>(wc), static_cast<U>(Ts)
        };
    }
};

/**
 * @brief Design a Proportional-Resonant controller
 *
 * @param Kp  Proportional gain
 * @param Ki  Resonant gain
 * @param w0  Resonant frequency (rad/s, e.g. 2*pi*50 for 50Hz)
 * @param wc  Cutoff bandwidth of resonant term (rad/s). Use 0 for ideal PR.
 *            Typical values: 5–15 rad/s for non-ideal PR.
 * @param Ts  Sampling time (seconds)
 * @return PRResult with design parameters
 */
template<typename T = double>
[[nodiscard]] consteval PRResult<T> pr(T Kp, T Ki, T w0, T wc, T Ts) {
    return PRResult<T>{Kp, Ki, w0, wc, Ts};
}

/**
 * @brief Design multiple-harmonic PR controller gains
 *
 * For harmonic compensation, returns an array of PRResult for harmonics
 * 1, 3, 5, 7, ... (or user-specified harmonic numbers).
 *
 * @param Kp          Proportional gain (shared across all harmonics)
 * @param Ki_fund     Resonant gain for fundamental
 * @param w_fund      Fundamental frequency (rad/s)
 * @param wc          Cutoff bandwidth (rad/s)
 * @param Ts          Sampling time
 * @param harmonics   Array of harmonic numbers (e.g. {1, 3, 5, 7})
 * @return Array of PRResult, one per harmonic
 */
template<size_t N, typename T = double>
[[nodiscard]] consteval std::array<PRResult<T>, N>
pr_harmonics(T Kp, T Ki_fund, T w_fund, T wc, T Ts, const std::array<size_t, N>& harmonics) {
    std::array<PRResult<T>, N> results{};
    for (size_t i = 0; i < N; ++i) {
        T w_h = w_fund * static_cast<T>(harmonics[i]);
        // Typically reduce Ki for higher harmonics
        T Ki_h = Ki_fund / static_cast<T>(harmonics[i]);
        results[i] = PRResult<T>{(i == 0) ? Kp : T{0}, Ki_h, w_h, wc, Ts};
    }
    return results;
}

} // namespace design

namespace online {

/**
 * @struct PRResult
 * @brief Proportional-Resonant controller design result (runtime)
 */
template<typename T = double>
struct PRResult {
    T Kp{};
    T Ki{};
    T w0{};
    T wc{};
    T Ts{};

    template<typename U>
    [[nodiscard]] constexpr auto as() const {
        return PRResult<U>{
            static_cast<U>(Kp), static_cast<U>(Ki),
            static_cast<U>(w0), static_cast<U>(wc), static_cast<U>(Ts)
        };
    }
};

/**
 * @brief Design a Proportional-Resonant controller (runtime)
 */
template<typename T = double>
[[nodiscard]] constexpr PRResult<T> pr(T Kp, T Ki, T w0, T wc, T Ts) {
    return PRResult<T>{Kp, Ki, w0, wc, Ts};
}

} // namespace online

/**
 * @ingroup discrete_controllers
 * @brief Discrete Proportional-Resonant Controller
 *
 * Implements a non-ideal PR controller in discrete time using Tustin
 * (bilinear) discretization of the resonant term.
 *
 * Continuous-time resonant term: R(s) = 2*Ki*wc*s / (s² + 2*wc*s + w0²)
 * Discretized via Tustin: s = (2/Ts)*(z-1)/(z+1)
 *
 * @tparam T Scalar type (default: float)
 */
template<typename T = float>
struct PRController {
    T Kp{};
    T Ki{};
    T w0{};
    T wc{};
    T Ts{};

    // Internal state for the discretized resonant term (2nd order IIR)
    T x1{0}; //!< State variable 1 (output history)
    T x2{0}; //!< State variable 2 (output history)
    T u1{0}; //!< Input history 1
    T u2{0}; //!< Input history 2

    // Pre-computed IIR coefficients
    T b0_r{0}, b1_r{0}, b2_r{0};
    T a1_r{0}, a2_r{0};

    constexpr PRController() = default;

    consteval PRController(const design::PRResult<T>& result)
        : Kp(result.Kp), Ki(result.Ki), w0(result.w0), wc(result.wc), Ts(result.Ts) {
        compute_coefficients();
    }

    constexpr PRController(const online::PRResult<T>& result)
        : Kp(result.Kp), Ki(result.Ki), w0(result.w0), wc(result.wc), Ts(result.Ts) {
        compute_coefficients();
    }

    template<typename U>
    constexpr PRController(const PRController<U>& other)
        : Kp(other.Kp), Ki(other.Ki), w0(other.w0), wc(other.wc), Ts(other.Ts), x1(other.x1), x2(other.x2), u1(other.u1), u2(other.u2), b0_r(other.b0_r), b1_r(other.b1_r), b2_r(other.b2_r), a1_r(other.a1_r), a2_r(other.a2_r) {}

    /**
     * @brief Compute control output
     *
     * @param error Current error (reference - measurement)
     * @return Control output u
     */
    [[nodiscard]] constexpr T control(T error) {
        // Resonant term: 2nd order IIR filter
        T resonant = b0_r * error + b1_r * u1 + b2_r * u2 - a1_r * x1 - a2_r * x2;

        // Update history
        x2 = x1;
        x1 = resonant;
        u2 = u1;
        u1 = error;

        return Kp * error + resonant;
    }

    constexpr void reset() {
        x1 = T{0};
        x2 = T{0};
        u1 = T{0};
        u2 = T{0};
    }

    /**
     * @brief Update resonant frequency (for grid frequency adaptation)
     *
     * @param new_w0 New resonant frequency (rad/s)
     */
    constexpr void set_frequency(T new_w0) {
        w0 = new_w0;
        compute_coefficients();
    }

private:
    constexpr void compute_coefficients() {
        // Tustin discretization of R(s) = 2*Ki*wc*s / (s² + 2*wc*s + w0²)
        // s = (2/Ts)*(z-1)/(z+1)
        T k = T{2} / Ts; // Tustin substitution factor

        // Denominator: s² + 2*wc*s + w0²
        // Substituting s = k*(z-1)/(z+1):
        // k²*(z-1)²/(z+1)² + 2*wc*k*(z-1)/(z+1) + w0²
        // Multiplying through by (z+1)²:
        // k²*(z-1)² + 2*wc*k*(z-1)*(z+1) + w0²*(z+1)²
        T a0_s = k * k + T{2} * wc * k + w0 * w0;
        T a1_s = T{2} * (w0 * w0 - k * k);
        T a2_s = k * k - T{2} * wc * k + w0 * w0;

        // Numerator: 2*Ki*wc*s = 2*Ki*wc*k*(z-1)/(z+1)
        // Multiplying through by (z+1)²:
        // 2*Ki*wc*k*(z-1)*(z+1) = 2*Ki*wc*k*(z²-1)
        T b0_s = T{2} * Ki * wc * k;
        T b1_s = T{0};
        T b2_s = -T{2} * Ki * wc * k;

        // Normalize by a0
        b0_r = b0_s / a0_s;
        b1_r = b1_s / a0_s;
        b2_r = b2_s / a0_s;
        a1_r = a1_s / a0_s;
        a2_r = a2_s / a0_s;
    }
};

/**
 * @ingroup discrete_controllers
 * @brief Multi-harmonic PR Controller
 *
 * Combines proportional gain with multiple resonant terms for harmonic
 * compensation. Each resonant term tracks one harmonic frequency.
 *
 * @tparam N Number of harmonic terms
 * @tparam T Scalar type (default: float)
 */
template<size_t N, typename T = float>
struct MultiPRController {
    T               Kp{};
    PRController<T> resonants[N]{};

    constexpr MultiPRController() = default;

    template<size_t M>
    consteval MultiPRController(const std::array<design::PRResult<T>, M>& results)
        requires(M == N)
    {
        Kp = results[0].Kp;
        for (size_t i = 0; i < N; ++i) {
            auto r = results[i];
            r.Kp = T{0}; // Kp is applied once, not per-harmonic
            resonants[i] = PRController<T>(r);
        }
    }

    /**
     * @brief Compute control output (sum of all resonant terms + Kp)
     */
    [[nodiscard]] constexpr T control(T error) {
        T u = Kp * error;
        for (size_t i = 0; i < N; ++i) {
            u += resonants[i].control(error);
        }
        return u;
    }

    constexpr void reset() {
        for (size_t i = 0; i < N; ++i) {
            resonants[i].reset();
        }
    }
};

} // namespace wetmelon::control
