#pragma once

#include <cmath>

#include "constexpr_math.hpp"

namespace wetmelon::control {

namespace online {

/**
 * @struct SMCResult
 * @brief Sliding Mode Control design result
 */
template<typename T = double>
struct SMCResult {
    T lambda{}; //< Sliding surface parameter
    T k{};      //< Switching gain
    T b0{};     //< Plant gain
    T Ts{};     //< Sampling time

    template<typename U>
    [[nodiscard]] constexpr auto as() const {
        return SMCResult<U>{lambda, k, b0, Ts};
    }
};

/**
 * @brief Sliding Mode Control design
 *
 * @param lambda Sliding surface parameter
 * @param k Switching gain
 * @param b0 Plant gain
 * @param Ts Sampling time
 *
 * @return SMCResult with computed gains
 */
template<typename T = double>
[[nodiscard]] constexpr SMCResult<T> smc(T lambda, T k, T b0, T Ts) {
    return SMCResult<T>{lambda, k, b0, Ts};
}
} // namespace online

namespace design {

/**
 * @struct SMCResult
 * @brief Sliding Mode Control design result
 */
template<typename T = double>
struct SMCResult {
    T lambda{}; //< Sliding surface parameter
    T k{};      //< Switching gain
    T b0{};     //< Plant gain
    T Ts{};     //< Sampling time

    template<typename U>
    [[nodiscard]] consteval auto as() const {
        return SMCResult<U>{lambda, k, b0, Ts};
    }
};

/**
 * @brief Sliding Mode Control design
 *
 * @param lambda Sliding surface parameter
 * @param k Switching gain
 * @param b0 Plant gain
 * @param Ts Sampling time
 *
 * @return SMCResult with computed gains
 */
template<typename T = double>
[[nodiscard]] consteval SMCResult<T> smc(T lambda, T k, T b0, T Ts) {
    return online::smc<T>(lambda, k, b0, Ts);
}

} // namespace design

/**
 * @ingroup discrete_controllers
 * @brief Sliding Mode Control (SMC)
 *
 * Discrete Sliding Mode Control for SISO systems.
 * Implements a basic sliding surface s = lambda * e + dot(e)
 * with control law u = -(k / b0) * sign(s)
 *
 * @tparam T Scalar type (default: float)
 */
template<typename T = float>
class SMCController {
    T lambda{}; //< Sliding surface parameter
    T k{};      //< Switching gain
    T b0{};     //< Plant gain
    T Ts{};     //< Sampling time

    T error_prev{}; //< Previous error for derivative calculation

public:
    constexpr SMCController() = default;
    consteval SMCController(const design::SMCResult<T>& result)
        : lambda(result.lambda), k(result.k), b0(result.b0), Ts(result.Ts) {}

    constexpr SMCController(const online::SMCResult<T>& result)
        : lambda(result.lambda), k(result.k), b0(result.b0), Ts(result.Ts) {}

    template<typename U>
    constexpr SMCController(const SMCController<U>& other)
        : lambda(other.lambda), k(other.k), b0(other.b0), Ts(other.Ts), error_prev(other.error_prev) {}

    /**
     * @brief Compute SMC control
     *
     * @param r Reference
     * @param y Measurement
     * @param phi Boundary layer thickness (default: 0, no boundary layer)
     *
     * @return Control output u
     */
    [[nodiscard]] constexpr T control(T r, T y, T phi = T{0.0}) {
        T error = r - y;

        // Compute derivative of error using backward difference
        T dot_e = (error - error_prev) / Ts;
        error_prev = error;

        // Define the "sliding surface"
        // Lambda weights error relative to its derivative.
        T s = lambda * error + dot_e;

        // Compute control using saturation function for boundary layer
        if (phi <= T{0.0}) {
            return -(k / b0) * wet::sgn(s);
        } else {
            return -(k / b0) * s / (phi + wet::abs(s));
        }
    }

    constexpr void reset() {
        error_prev = T{};
    }
};
} // namespace wetmelon::control