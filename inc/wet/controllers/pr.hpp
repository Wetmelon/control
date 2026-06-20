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
 * Transfer function (non-ideal PR):
 * @f[
 *   C(s) = K_p + \frac{2 K_i \omega_c\, s}{s^2 + 2 \omega_c s + \omega_0^2}
 * @f]
 * where @f$K_p@f$ is the proportional gain, @f$K_i@f$ the resonant (integral)
 * gain, @f$\omega_0@f$ the resonant frequency (e.g. @f$2\pi\cdot 50@f$ for a
 * 50 Hz grid) and @f$\omega_c@f$ the resonant-term bandwidth.
 *
 * An ideal PR has @f$\omega_c = 0@f$, @f$C(s) = K_p + K_i s/(s^2 + \omega_0^2)@f$,
 * but its zero bandwidth makes it fragile to grid-frequency drift.
 *
 * MATLAB equivalent:
 * @code{.m}
 *   s = tf('s');
 *   C = Kp + 2*Ki*wc*s / (s^2 + 2*wc*s + w0^2);
 * @endcode
 *
 * @see R. Teodorescu et al., "Proportional-resonant controllers and filters for
 *      grid-connected voltage-source converters," IEE Proc. Electr. Power Appl.,
 *      2006. DOI: 10.1049/ip-epa:20060008
 * @see D. N. Zmood and D. G. Holmes, "Stationary frame current regulation of PWM
 *      inverters with zero steady-state error," IEEE Trans. Power Electron.,
 *      2003. DOI: 10.1109/TPEL.2003.810852
 */

#include "wet/filters/filters.hpp"
#include "wet/math/math.hpp"
#include "wet/systems/discretization.hpp"
#include "wet/systems/state_space.hpp"
#include "wet/systems/transfer_function.hpp"

namespace wet {

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
    [[nodiscard]] constexpr auto as() const {
        return PRResult<U>{
            static_cast<U>(Kp), static_cast<U>(Ki),
            static_cast<U>(w0), static_cast<U>(wc), static_cast<U>(Ts)
        };
    }

    /**
     * @brief Convert to continuous-time transfer function
     *
     * Returns the full PR law C(s) = Kp + 2*Ki*wc*s / (s² + 2*wc*s + w0²)
     * as a single 2nd-order TF. Numerator and denominator in ascending
     * powers of s.
     *
     *   num = {Kp*w0², 2*wc*(Kp+Ki), Kp}
     *   den = {w0², 2*wc, 1}
     */
    [[nodiscard]] constexpr TransferFunction<3, 3, T> to_tf() const {
        return TransferFunction<3, 3, T>{
            .num = {Kp * w0 * w0, T{2} * wc * (Kp + Ki), Kp},
            .den = {w0 * w0, T{2} * wc, T{1}},
        };
    }

    /**
     * @brief Convert to continuous-time state-space (2nd order SISO)
     *
     * Controllable canonical form of C(s); D = Kp carries the proportional
     * feedthrough, C = [0, 2*Ki*wc] the resonant output.
     */
    [[nodiscard]] constexpr StateSpace<2, 1, 1, 0, 0, T> to_ss() const {
        return StateSpace<2, 1, 1, 0, 0, T>{
            .A = Matrix<2, 2, T>{{T{0}, T{1}}, {-w0 * w0, -T{2} * wc}},
            .B = Matrix<2, 1, T>{{T{0}}, {T{1}}},
            .C = Matrix<1, 2, T>{{T{0}, T{2} * Ki * wc}},
            .D = Matrix<1, 1, T>{{Kp}},
        };
    }

    /**
     * @brief Convert to discrete-time state-space
     *
     * @param method Discretization method (default: Tustin)
     */
    [[nodiscard]] constexpr StateSpace<2, 1, 1, 0, 0, T>
    to_discrete_ss(DiscretizationMethod method = DiscretizationMethod::Tustin) const {
        return discretize(to_ss(), Ts, method);
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
[[nodiscard]] constexpr PRResult<T> pr(T Kp, T Ki, T w0, T wc, T Ts) {
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
[[nodiscard]] constexpr wet::array<PRResult<T>, N>
pr_harmonics(T Kp, T Ki_fund, T w_fund, T wc, T Ts, const wet::array<size_t, N>& harmonics) {
    wet::array<PRResult<T>, N> results{};
    for (size_t i = 0; i < N; ++i) {
        T w_h = w_fund * static_cast<T>(harmonics[i]);
        // Typically reduce Ki for higher harmonics
        T Ki_h = Ki_fund / static_cast<T>(harmonics[i]);
        results[i] = PRResult<T>{(i == 0) ? Kp : T{0}, Ki_h, w_h, wc, Ts};
    }
    return results;
}

} // namespace design

/**
 * @ingroup discrete_controllers
 * @brief Discrete Proportional-Resonant Controller
 *
 * Non-ideal PR controller realized as a proportional path plus a Tustin-
 * discretized resonant @ref Biquad. The resonant frequency is pre-warped so the
 * bilinear map lands the digital peak exactly on w0, preserving zero
 * steady-state error at the target frequency (the unwarped map places the peak
 * at the bilinear-warped frequency, worst for high harmonics / low fs).
 *
 * Continuous-time resonant term @f$R(s) = 2 K_i \omega_c s/(s^2 + 2\omega_c s + \omega_0^2)@f$,
 * discretized with @f$s = (2/T_s)(z-1)/(z+1)@f$.
 *
 * @code
 * // 50 Hz grid current regulator at 10 kHz
 * auto                d = design::pr(1.0, 200.0, 2 * std::numbers::pi * 50, 10.0, 1e-4);
 * PRController<double> pr(d);
 * double              u = pr.control(i_ref, i_meas); // (r, y) form
 * @endcode
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
    T Kbc{T{0}}; //!< Back-calculation tracking constant; larger = slower unwind (0 = unit-rate fallback, matches PID)

    //! Discretized resonant term (Direct Form I biquad, inert until designed).
    Biquad<T> resonant{design::SecondOrderCoeffs<T>{}};

    constexpr PRController() = default;

    constexpr PRController(const design::PRResult<T>& result)
        : Kp(result.Kp), Ki(result.Ki), w0(result.w0), wc(result.wc), Ts(result.Ts) {
        compute_coefficients();
    }

    template<typename U>
    constexpr explicit PRController(const PRController<U>& other)
        : Kp(static_cast<T>(other.Kp)), Ki(static_cast<T>(other.Ki)), w0(static_cast<T>(other.w0)), wc(static_cast<T>(other.wc)), Ts(static_cast<T>(other.Ts)), Kbc(static_cast<T>(other.Kbc)), resonant(other.resonant) {}

    /**
     * @brief Compute control output
     *
     * @param error Current error (reference - measurement)
     * @return Control output u
     */
    [[nodiscard]] constexpr T control(T error) {
        return Kp * error + resonant(error);
    }

    /**
     * @brief Reference-tracking overload satisfying SISOController.
     *
     * Computes the error internally and forwards to the error-form `control()`.
     * Lets this controller be used directly in `Cascade<Outer, PRController>`
     * and in tuning harnesses that speak the (r, y) protocol.
     */
    [[nodiscard]] constexpr T control(T r, T y) {
        return control(r - y);
    }

    /// Clear the resonant filter's delay line; gains and config are preserved.
    constexpr void reset() {
        resonant.reset();
    }

    /**
     * @brief Anti-windup hook for cascade-level saturation propagation.
     *
     * Damped unwind: adds `(u_sat - u_unsat) * Ts / Kbc` to the most recent
     * resonant output so the next tick's resonant contribution is pulled back
     * toward the realizable command. Same shape and sign convention as
     * `PIDController::back_calculate` (`Ts` is per-call, not stored), so a
     * generic anti-windup wrapper can drive either uniformly. No-op when
     * `Kbc == 0` (back-calculation not configured).
     *
     * @param u_unsat Command this controller requested.
     * @param u_sat   Command actually applied after downstream clamping.
     * @param Ts      Sample time (s).
     */
    constexpr void back_calculate(T u_unsat, T u_sat, T Ts) {
        if (Kbc == T{0} || u_unsat == u_sat) {
            return;
        }
        resonant.set_last_output(resonant.last_output() + Ts * ((u_sat - u_unsat) / Kbc));
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
        // Ts <= 0 means a continuous-time design (e.g. a design factory's default
        // Ts = 0) was handed to a discrete controller — no valid Tustin mapping.
        // Zero the resonant term so control() falls back to pure proportional.
        if (!(Ts > T{0})) {
            resonant.set_coefficients(design::SecondOrderCoeffs<T>{});
            return;
        }

        const T k = T{2} / Ts; // Tustin substitution factor 2/Ts

        // Pre-warp the resonant frequency so the bilinear map places the digital
        // peak exactly on w0 — w_pw = (2/Ts)*tan(w0*Ts/2). Without this the peak
        // lands at the warped frequency and zero-steady-state error at w0 erodes
        // (negligible at 50 Hz/10 kHz, real for high harmonics / low fs). Skip
        // when the resonance is at/above Nyquist (tan undefined past pi/2).
        T       w0_d = w0;
        const T half = w0 * Ts / T{2};
        if (half < wet::numbers::pi_v<T> / T{2}) {
            w0_d = k * wet::tan(half);
        }

        // Tustin of R(s) = 2*Ki*wc*s / (s² + 2*wc*s + w0_d²), s = k*(z-1)/(z+1),
        // numerator and denominator multiplied through by (z+1)²:
        //   den: k²(z-1)² + 2*wc*k(z-1)(z+1) + w0_d²(z+1)²
        //   num: 2*Ki*wc*k(z-1)(z+1) = 2*Ki*wc*k(z²-1)
        const T a0 = k * k + T{2} * wc * k + w0_d * w0_d;
        const T a1 = T{2} * (w0_d * w0_d - k * k);
        const T a2 = k * k - T{2} * wc * k + w0_d * w0_d;
        const T b0 = T{2} * Ki * wc * k;
        const T b2 = -T{2} * Ki * wc * k;

        resonant.set_coefficients(design::detail::normalize_biquad<T>(b0, T{0}, b2, a0, a1, a2));
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
    T                              Kp{};
    wet::array<PRController<T>, N> resonants{};

    constexpr MultiPRController() = default;

    template<size_t M>
    constexpr MultiPRController(const wet::array<design::PRResult<T>, M>& results)
        requires(M == N)
        : Kp(results[0].Kp) {
        for (size_t i = 0; i < N; ++i) {
            auto r = results[i];
            r.Kp = T{0}; // Kp is applied once, not per-harmonic
            resonants[i] = PRController<T>(r);
        }
    }

    template<typename U>
    constexpr explicit MultiPRController(const MultiPRController<N, U>& other) : Kp(static_cast<T>(other.Kp)) {
        for (size_t i = 0; i < N; ++i) {
            resonants[i] = PRController<T>(other.resonants[i]);
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

    /**
     * @brief Reference-tracking overload satisfying SISOController.
     */
    [[nodiscard]] constexpr T control(T r, T y) {
        return control(r - y);
    }

    constexpr void reset() {
        for (size_t i = 0; i < N; ++i) {
            resonants[i].reset();
        }
    }
};

} // namespace wet
