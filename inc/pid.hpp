#pragma once

#include <algorithm>
#include <cmath>

namespace wetmelon::control {

namespace online {

/**
 * @struct PIDResult
 * @brief Runtime PID design result (online namespace)
 */
template<typename T>
struct PIDResult {
    T Kp{};
    T Ki{};
    T Kd{};
    T Ts{};
    T u_min = -std::numeric_limits<T>::infinity();
    T u_max = std::numeric_limits<T>::infinity();
    T i_min = -std::numeric_limits<T>::infinity();
    T i_max = std::numeric_limits<T>::infinity();
    T Kbc = T{0}; // Back-calculation gain

    template<typename U>
    [[nodiscard]] constexpr auto as() const {
        return PIDResult<U>{Kp, Ki, Kd, Ts, u_min, u_max, i_min, i_max, Kbc};
    }
};

/**
 * @brief PID controller design (runtime version)
 *
 * @param Kp Proportional gain
 * @param Ki Integral gain
 * @param Kd Derivative gain
 * @param Ts Sampling time
 * @param u_min Minimum control output
 * @param u_max Maximum control output
 * @param i_min Minimum integrator value
 * @param i_max Maximum integrator value
 * @param Kbc Back-calculation gain
 *
 * @return PIDResult with the specified gains
 */
template<typename T = double>
[[nodiscard]] constexpr PIDResult<T> pid(T Kp, T Ki, T Kd, T Ts, T u_min = -std::numeric_limits<T>::infinity(), T u_max = std::numeric_limits<T>::infinity(), T i_min = -std::numeric_limits<T>::infinity(), T i_max = std::numeric_limits<T>::infinity(), T Kbc = T{0}) {
    return PIDResult<T>{Kp, Ki, Kd, Ts, u_min, u_max, i_min, i_max, Kbc};
}

} // namespace online

namespace design {

/**
 * @struct PIDResult
 * @brief PID controller design result
 */
template<typename T = double>
struct PIDResult {
    T Kp{};
    T Ki{};
    T Kd{};
    T Ts{};
    T u_min = -std::numeric_limits<T>::infinity();
    T u_max = std::numeric_limits<T>::infinity();
    T i_min = -std::numeric_limits<T>::infinity();
    T i_max = std::numeric_limits<T>::infinity();
    T Kbc = T{0}; // Back-calculation gain

    template<typename U>
    [[nodiscard]] consteval auto as() const {
        return PIDResult<U>{
            static_cast<U>(Kp), static_cast<U>(Ki), static_cast<U>(Kd), static_cast<U>(Ts),
            static_cast<U>(u_min), static_cast<U>(u_max), static_cast<U>(i_min), static_cast<U>(i_max),
            static_cast<U>(Kbc)
        };
    }
};

/**
 * @brief PID controller design
 *
 * @param Kp Proportional gain
 * @param Ki Integral gain
 * @param Kd Derivative gain
 * @param Ts Sampling time
 * @param u_min Minimum control output
 * @param u_max Maximum control output
 * @param i_min Minimum integrator value
 * @param i_max Maximum integrator value
 * @param Kbc Back-calculation gain
 *
 * @return PIDResult with the specified gains
 */
template<typename T = double>
[[nodiscard]] consteval PIDResult<T> pid(T Kp, T Ki, T Kd, T Ts, T u_min = -std::numeric_limits<T>::infinity(), T u_max = std::numeric_limits<T>::infinity(), T i_min = -std::numeric_limits<T>::infinity(), T i_max = std::numeric_limits<T>::infinity(), T Kbc = T{0}) {
    return PIDResult<T>{Kp, Ki, Kd, Ts, u_min, u_max, i_min, i_max, Kbc};
}
} // namespace design

/**
 * @ingroup discrete_controllers
 * @brief Discrete PID Controller
 *
 * Standard PID controller with anti-windup and output saturation.
 *
 * @tparam T Scalar type (default: float)
 */
template<typename T = float>
struct PIDController {
    T Kp{};
    T Ki{};
    T Kd{};
    T Ts{};
    T u_min = -std::numeric_limits<T>::infinity();
    T u_max = std::numeric_limits<T>::infinity();
    T i_min = -std::numeric_limits<T>::infinity();
    T i_max = std::numeric_limits<T>::infinity();
    T Kbc = T{0}; // Back-calculation gain
    T integral{0};
    T prev_error{0};

    constexpr PIDController() = default;
    consteval PIDController(const design::PIDResult<T>& result) : Kp(result.Kp), Ki(result.Ki), Kd(result.Kd), Ts(result.Ts), u_min(result.u_min), u_max(result.u_max), i_min(result.i_min), i_max(result.i_max), Kbc(result.Kbc) {}
    constexpr PIDController(const online::PIDResult<T>& result) : Kp(result.Kp), Ki(result.Ki), Kd(result.Kd), Ts(result.Ts), u_min(result.u_min), u_max(result.u_max), i_min(result.i_min), i_max(result.i_max), Kbc(result.Kbc) {}

    template<typename U>
    constexpr PIDController(const PIDController<U>& other) : Kp(other.Kp), Ki(other.Ki), Kd(other.Kd), Ts(other.Ts), u_min(other.u_min), u_max(other.u_max), i_min(other.i_min), i_max(other.i_max), Kbc(other.Kbc), integral(other.integral), prev_error(other.prev_error) {}

    /**
     * @brief Compute PID control law
     *
     * @param error Current error (r - y)
     * @return Control output u
     */
    [[nodiscard]] constexpr T control(T error) {
        T derivative = (error - prev_error) / Ts;
        T u_unsat = Kp * error + Ki * integral + Kd * derivative;
        T u = std::clamp(u_unsat, u_min, u_max);

        // Back-calculation anti-windup
        if (Kbc != T{0}) {
            integral += Ts * (error + (u - u_unsat) / Kbc);
        } else {
            integral += error * Ts;
        }

        // Separate integrator limiting
        integral = std::clamp(integral, i_min, i_max);

        prev_error = error;
        return u;
    }

    constexpr void reset() {
        integral = 0;
        prev_error = 0;
    }
};

} // namespace wetmelon::control
