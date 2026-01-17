#pragma once

#include <algorithm>
#include <numbers>
#include <tuple>

#include "constexpr_math.hpp"

namespace wetmelon::control {

/**
 * @defgroup motor_control Motor Control Transforms
 * @brief Coordinate transformations for motor control
 *
 * Clarke, Park, and inverse transforms commonly used in three-phase motor control.
 */

/**
 * @brief Clarke transform (ABC to αβ)
 *
 * Transforms three-phase stationary coordinates to two-phase stationary coordinates.
 * Used in field-oriented control (FOC) for induction and synchronous motors.
 *
 * @param a Phase A
 * @param b Phase B
 * @param c Phase C
 * @param T Scalar type
 * @return std::pair<T, T> {α, β} components
 */
template<typename T = float>
[[nodiscard]] constexpr std::pair<T, T> clarke_transform(T a, T b, T c) {
    const T alpha = (T{2} * a - b - c) / T{3};
    const T beta = (b - c) / std::numbers::sqrt3_v<T>;
    return {alpha, beta};
}

/**
 * @brief Inverse Clarke transform (αβ to ABC)
 *
 * Transforms two-phase stationary coordinates back to three-phase coordinates.
 *
 * @param alpha α component
 * @param beta β component
 * @param T Scalar type
 * @return std::tuple<T, T, T> {A, B, C} phases
 */
template<typename T = float>
[[nodiscard]] constexpr std::tuple<T, T, T> inverse_clarke_transform(T alpha, T beta) {
    const T a = alpha;
    const T b = -alpha / T{2} + std::numbers::sqrt3_v<T> * beta / T{2};
    const T c = -alpha / T{2} - std::numbers::sqrt3_v<T> * beta / T{2};
    return {a, b, c};
}

/**
 * @brief Park transform (αβ to dq)
 *
 * Transforms two-phase stationary coordinates to rotating dq coordinates.
 * Requires rotor angle θ for synchronous reference frame.
 *
 * @param alpha α component
 * @param beta β component
 * @param theta Rotor angle [rad]
 * @param T Scalar type
 * @return std::pair<T, T> {d, q} components
 */
template<typename T = float>
[[nodiscard]] constexpr std::pair<T, T> park_transform(T alpha, T beta, T theta) {
    const T cos_theta = wet::cos(theta);
    const T sin_theta = wet::sin(theta);

    const T d = alpha * cos_theta + beta * sin_theta;
    const T q = -alpha * sin_theta + beta * cos_theta;

    return {d, q};
}

/**
 * @brief Inverse Park transform (dq to αβ)
 *
 * Transforms rotating dq coordinates back to stationary αβ coordinates.
 *
 * @param d d component
 * @param q q component
 * @param theta Rotor angle [rad]
 * @param T Scalar type
 * @return std::pair<T, T> {α, β} components
 */
template<typename T = float>
[[nodiscard]] constexpr std::pair<T, T> inverse_park_transform(T d, T q, T theta) {
    const T cos_theta = wet::cos(theta);
    const T sin_theta = wet::sin(theta);

    const T alpha = d * cos_theta - q * sin_theta;
    const T beta = d * sin_theta + q * cos_theta;

    return {alpha, beta};
}

/**
 * @brief Combined Clarke-Park transform (ABC to dq)
 *
 * Direct transformation from three-phase to rotating dq coordinates.
 *
 * @param a Phase A
 * @param b Phase B
 * @param c Phase C
 * @param theta Rotor angle [rad]
 * @param T Scalar type
 * @return std::pair<T, T> {d, q} components
 */
template<typename T = float>
[[nodiscard]] constexpr std::pair<T, T> clarke_park_transform(T a, T b, T c, T theta) {
    const auto [alpha, beta] = clarke_transform(a, b, c);
    return park_transform(alpha, beta, theta);
}

/**
 * @brief Combined inverse Park-Clarke transform (dq to ABC)
 *
 * Direct transformation from rotating dq coordinates to three-phase.
 *
 * @param d d component
 * @param q q component
 * @param theta Rotor angle [rad]
 * @param T Scalar type
 * @return std::tuple<T, T, T> {A, B, C} phases
 */
template<typename T = float>
[[nodiscard]] constexpr std::tuple<T, T, T> inverse_park_clarke_transform(T d, T q, T theta) {
    const auto [alpha, beta] = inverse_park_transform(d, q, theta);
    return inverse_clarke_transform(alpha, beta);
}

/**
 * @brief Space Vector Modulation (SVM) duty cycles
 *
 * Calculates PWM duty cycles for three-phase inverter using SVM.
 * Returns duty cycles for phases A, B, C in range [0, 1].
 *
 * @param v_alpha α voltage component
 * @param v_beta β voltage component
 * @param v_dc DC bus voltage
 * @param T Scalar type
 * @return std::tuple<T, T, T> {duty_A, duty_B, duty_C}
 */
template<typename T = float>
[[nodiscard]] constexpr std::tuple<T, T, T> svm_duty_cycles(T v_alpha, T v_beta, T v_dc) {
    // Normalize voltages
    const T v_a_norm = v_alpha / (v_dc / std::numbers::sqrt3_v<T>);
    const T v_b_norm = v_beta / (v_dc / std::numbers::sqrt3_v<T>);

    // Third harmonic injection for better DC bus utilization
    const T v_0 = T{0}; // Zero sequence component

    // Calculate duty cycles
    const T duty_a = T{0.5} + v_a_norm / T{2} + v_0 / T{2};
    const T duty_b = T{0.5} - v_a_norm / T{4} + v_b_norm * std::numbers::sqrt3_v<T> / T{4} + v_0 / T{2};
    const T duty_c = T{0.5} - v_a_norm / T{4} - v_b_norm * std::numbers::sqrt3_v<T> / T{4} + v_0 / T{2};

    return {std::clamp(duty_a, T{0}, T{1}), std::clamp(duty_b, T{0}, T{1}), std::clamp(duty_c, T{0}, T{1})};
}

} // namespace wetmelon::control