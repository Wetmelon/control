#pragma once

#include <algorithm>
#include <cmath>

namespace wetmelon::control {
namespace design {

/**
 * @struct PIDResult
 * @brief 2-DOF PID controller design result
 *
 * Contains gains and setpoint weights for a two-degree-of-freedom PID controller.
 * The setpoint weights `b` and `c` decouple reference tracking from disturbance
 * rejection, allowing each to be tuned independently.
 *
 * Special cases:
 * - b=1, c=1: Standard PID (P and D on error)
 * - b=1, c=0: PI-D (D on measurement — no derivative kick)
 * - b=0, c=0: I-PD (P and D on measurement — no setpoint kick)
 *
 * @see Åström & Hägglund, "Advanced PID Control" (2006), §4.4
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
    T Kbc = T{0}; ///< Back-calculation anti-windup gain
    T b = T{1};   ///< Proportional setpoint weight (0=I-PD, 1=standard PID)
    T c = T{1};   ///< Derivative setpoint weight   (0=PI-D, 1=standard PID)

    template<typename U>
    [[nodiscard]] constexpr auto as() const {
        return PIDResult<U>{(U)Kp, (U)Ki, (U)Kd, (U)Ts, (U)u_min, (U)u_max, (U)i_min, (U)i_max, (U)Kbc, (U)b, (U)c};
    }
};

/**
 * @brief 2-DOF PID controller design
 *
 * Constructs a PIDResult with the given gains and optional setpoint weights.
 *
 * The control law is:
 *
 *     u = Kp(b·r − y) + Ki·∫(r−y)dt + Kd·d/dt(c·r − y)
 *
 * @param Kp    Proportional gain
 * @param Ki    Integral gain
 * @param Kd    Derivative gain
 * @param Ts    Sampling time
 * @param u_min Minimum control output
 * @param u_max Maximum control output
 * @param i_min Minimum integrator value
 * @param i_max Maximum integrator value
 * @param Kbc   Back-calculation anti-windup gain (0 = clamping only)
 * @param b     Proportional setpoint weight (default: 1 — standard PID)
 * @param c     Derivative setpoint weight   (default: 1 — standard PID)
 *
 * @return PIDResult with the specified parameters
 *
 * @see Åström & Hägglund, "Advanced PID Control" (2006), §4.4
 */
template<typename T = double>
[[nodiscard]] constexpr PIDResult<T> pid(
    T Kp, T Ki, T Kd, T Ts,
    T u_min = -std::numeric_limits<T>::infinity(),
    T u_max = std::numeric_limits<T>::infinity(),
    T i_min = -std::numeric_limits<T>::infinity(),
    T i_max = std::numeric_limits<T>::infinity(),
    T Kbc = T{0},
    T b = T{1},
    T c = T{1}
) {
    return PIDResult<T>{Kp, Ki, Kd, Ts, u_min, u_max, i_min, i_max, Kbc, b, c};
}
} // namespace design

/**
 * @ingroup discrete_controllers
 * @brief Discrete 2-DOF PID Controller
 *
 * Implements the two-degree-of-freedom PID control law:
 *
 *     u = Kp(b·r − y) + Ki·∫(r−y)dt + Kd·d/dt(c·r − y)
 *
 * The setpoint weights `b` and `c` decouple reference tracking from disturbance
 * rejection. Common configurations:
 * - b=1, c=1: Standard PID
 * - b=1, c=0: PI-D (no derivative kick on reference steps)
 * - b=0, c=0: I-PD (no proportional or derivative kick on reference steps)
 *
 * Call `control(r, y)` with the reference and measurement separately.
 *
 * @tparam T Scalar type (default: float)
 *
 * @see Åström & Hägglund, "Advanced PID Control" (2006), §4.4
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
    T Kbc = T{0}; ///< Back-calculation anti-windup gain
    T b = T{1};   ///< Proportional setpoint weight
    T c = T{1};   ///< Derivative setpoint weight

    T integral = T{0};        ///< Integrator state
    T prev_cr_minus_y = T{0}; ///< Previous value of (c·r − y) for derivative

    constexpr PIDController() = default;

    constexpr PIDController(const design::PIDResult<T>& result)
        : Kp(result.Kp), Ki(result.Ki), Kd(result.Kd), Ts(result.Ts), u_min(result.u_min), u_max(result.u_max), i_min(result.i_min), i_max(result.i_max), Kbc(result.Kbc), b(result.b), c(result.c) {}

    template<typename U>
    constexpr PIDController(const PIDController<U>& other)
        : Kp(other.Kp), Ki(other.Ki), Kd(other.Kd), Ts(other.Ts), u_min(other.u_min), u_max(other.u_max), i_min(other.i_min), i_max(other.i_max), Kbc(other.Kbc), b(other.b), c(other.c), integral(other.integral), prev_cr_minus_y(other.prev_cr_minus_y) {}

    /**
     * @brief Compute 2-DOF PID control output
     *
     * @param r Reference (setpoint)
     * @param y Measurement (process variable)
     * @return Control output u
     */
    [[nodiscard]] constexpr T control(T r, T y) {
        T e = r - y;
        T cr_minus_y = c * r - y;
        T derivative = (cr_minus_y - prev_cr_minus_y) / Ts;
        T u_unsat = Kp * (b * r - y) + Ki * integral + Kd * derivative;
        T u = std::clamp(u_unsat, u_min, u_max);

        // Back-calculation anti-windup
        if (Kbc != T{0}) {
            integral += Ts * (e + (u - u_unsat) / Kbc);
        } else {
            integral += e * Ts;
        }

        // Separate integrator limiting
        integral = std::clamp(integral, i_min, i_max);

        prev_cr_minus_y = cr_minus_y;
        return u;
    }

    constexpr void reset() {
        integral = T{0};
        prev_cr_minus_y = T{0};
    }
};

} // namespace wetmelon::control
