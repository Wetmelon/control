#pragma once

#include <algorithm>
#include <numbers>

#include "wet/filters/sogi.hpp"
#include "wet/math/wetmelon_math.hpp"

namespace wetmelon::control {
/**
 * @brief Single-Phase PLL
 *
 * Uses a Second-Order Generalized Integrator (SOGI) for quadrature signal generation.
 *
 * @note A phase-locked loop is a tracker, not a reference-following controller,
 *       and does not satisfy @ref SISOController -- its entry point is
 *       `step(T input, T Ts)` returning `void`, with state read out via member
 *       accessors. Use it as a frequency / phase estimator that feeds *into* a
 *       SISOController (e.g. a current-loop PR controller riding the PLL's
 *       phase estimate), not as a block inside `Cascade<Outer, Inner>`.
 */
template<typename T>
struct SinglePhasePLL {
    struct Parameters {
        T Kp{}; // Proportional gain
        T Ki{}; // Integral gain

        T integrator_leak{}; // [1/s] Integrator bleed rate (0 = pure integrator)

        T integrator_max{}; // [Hz] Maximum integrator state (anti-windup limit)
        T integrator_min{}; // [Hz] Minimum integrator state (anti-windup limit)

        T output_max{}; // [Hz] Maximum output frequency
        T output_min{}; // [Hz] Minimum output frequency
    } params{};

    constexpr SinglePhasePLL(T Fnom, T Ts)
        : nominal_frequency(Fnom), frequency_estimate(Fnom) {
        // Set integrator limits based on expected frequency range
        T max_freq_deviation = Fnom * T{0.5};
        params.integrator_max = max_freq_deviation;
        params.integrator_min = -max_freq_deviation;

        // Set output frequency limits around nominal frequency
        params.output_max = Fnom + max_freq_deviation;
        params.output_min = Fnom - max_freq_deviation;

        // Set proportional and integral gains
        params.Kp = T{10} * Ts;
        params.Ki = T{100} * Ts;
    }

    constexpr void step(T input, const T Ts) {
        // Generate bandpass and quadrature signals using SOGI
        const auto [bp, quadrature] = sogi(input, frequency_estimate, std::numbers::sqrt2_v<T>, Ts);
        (void)bp;

        // Phase error is the product of input and quadrature signal
        T phase_error = input * quadrature;

        // Proportional term
        T proportional = params.Kp * phase_error;

        // Integral term with leak (bleed) and anti-windup. The leak term
        // −leak·integrator_state·Ts pulls the accumulated frequency offset back
        // toward zero (i.e. the estimate back toward nominal) with time constant
        // 1/leak, so a transient or biased phase error doesn't park a permanent
        // offset in the integrator. leak == 0 recovers a pure integrator.
        integrator_state += ((params.Ki * phase_error) - (params.integrator_leak * integrator_state)) * Ts;
        integrator_state = std::clamp(integrator_state, params.integrator_min, params.integrator_max);

        // Compute output frequency estimate [Hz] around nominal frequency
        frequency_estimate = nominal_frequency + proportional + integrator_state;
        frequency_estimate = std::clamp(frequency_estimate, params.output_min, params.output_max);

        // Update phase estimate [rad] from frequency estimate [Hz]
        const T two_pi = T{2} * std::numbers::pi_v<T>;
        phase_estimate += two_pi * frequency_estimate * Ts;
        phase_estimate = wet::fmod(phase_estimate, two_pi);
        if (phase_estimate < T{0}) {
            phase_estimate += two_pi;
        }
    }

    [[nodiscard]] constexpr T frequency() const {
        return frequency_estimate;
    }

    [[nodiscard]] constexpr T phase() const {
        return phase_estimate;
    }

    constexpr void reset() {
        // sogi.reset();
        integrator_state = T{0};
        phase_estimate = T{0};
        frequency_estimate = nominal_frequency;
    }

private:
    SOGI<T> sogi{}; // SOGI for quadrature signal generation

    T nominal_frequency{};  // Center frequency of the PLL bandpass filter
    T integrator_state{};   // State of the integral term
    T phase_estimate{};     // Estimated phase of the input signal
    T frequency_estimate{}; // Estimated frequency of the input signal
};
} // namespace wetmelon::control
