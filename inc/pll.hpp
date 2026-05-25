#pragma once

#include <algorithm>
#include <cmath>
#include <numbers>

#include "sogi.hpp"

namespace wetmelon::control {
/**
 * @brief Single-Phase PLL
 *
 * Uses a Second-Order Generalized Integrator (SOGI) for quadrature signal generation.
 */
template<typename T>
struct SinglePhasePLL {
    struct Parameters {
        T Kp{}; // Proportional gain
        T Ki{}; // Integral gain

        T integrator_max{}; // [Hz] Maximum integrator state (anti-windup limit)
        T integrator_min{}; // [Hz] Minimum integrator state (anti-windup limit)

        T output_max{}; // [Hz] Maximum output frequency
        T output_min{}; // [Hz] Minimum output frequency
    } params{};

    constexpr SinglePhasePLL(T Fnom, T alpha, T Ts)
        : sogi(Fnom, Ts, alpha), nominal_frequency(Fnom), frequency_estimate(Fnom) {
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
        const auto [bp, quadrature] = sogi(input);
        (void)bp;

        // Phase error is the product of input and quadrature signal
        T phase_error = input * quadrature;

        // Proportional term
        T proportional = params.Kp * phase_error;

        // Integral term with anti-windup
        integrator_state += params.Ki * phase_error * Ts;
        integrator_state = std::clamp(integrator_state, params.integrator_min, params.integrator_max);

        // Compute output frequency estimate [Hz] around nominal frequency
        frequency_estimate = nominal_frequency + proportional + integrator_state;
        frequency_estimate = std::clamp(frequency_estimate, params.output_min, params.output_max);

        // Update phase estimate [rad] from frequency estimate [Hz]
        const T two_pi = T{2} * std::numbers::pi_v<T>;
        phase_estimate += two_pi * frequency_estimate * Ts;
        phase_estimate = std::fmod(phase_estimate, two_pi);
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
        sogi.reset();
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
