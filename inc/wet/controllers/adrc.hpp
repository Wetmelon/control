#pragma once

#include <array>
#include <cstddef>

#include "wet/math/wetmelon_math.hpp"
#include "wet/matrix/matrix.hpp"

namespace wet {

namespace design {

/**
 * @struct ADRCResult
 * @brief Active Disturbance Rejection Control design result
 *
 * Contains the computed observer and controller gains for ADRC.
 * Use .as<U>() to convert for type conversion (e.g., double to float).
 *
 * @see "From PID to Active Disturbance Rejection Control" (Han, 1998)
 */
template<size_t NX, typename T = double>
struct ADRCResult {
    T wc{}; //< Controller bandwidth
    T wo{}; //< Observer bandwidth
    T b0{}; //< Plant gain

    std::array<T, NX + 1> beta{}; //< ESO gains

    T Kp{}; //< Proportional gain
    T Kd{}; //< Derivative gain
    T Ts{}; //< Sampling time

    template<typename U>
    [[nodiscard]] constexpr auto as() const {
        return ADRCResult<NX, U>{wc, wo, b0, beta, Kp, Kd, Ts};
    }
};

/**
 * @brief Active Disturbance Rejection Control design
 *
 * Designs ESO gains using pole placement for observer poles at -wo.
 * Places all poles at -wo for the extended system.
 * Controller gains are computed based on system order:
 * - 1st order: Kp = wc/b0, Kd = 0
 * - 2nd order: Kp = wc²/b0, Kd = 2*wc/b0
 *
 * @param wc Controller bandwidth
 * @param wo Observer bandwidth
 * @param b0 Plant gain
 * @param Ts Sampling time
 *
 * @return ADRCResult with computed gains
 *
 * @see "From PID to Active Disturbance Rejection Control" (Han, 1998)
 */
template<size_t NX, typename T = double>
[[nodiscard]] constexpr ADRCResult<NX, T> adrc(T wc, T wo, T b0, T Ts) {
    std::array<T, NX + 1> beta{};
    size_t                n = NX + 1;
    for (size_t i = 1; i <= n; ++i) {
        T binom = T{1};
        for (size_t k = 1; k <= i; ++k) {
            binom *= static_cast<T>(n - k + 1);
            binom /= static_cast<T>(k);
        }
        beta[i - 1] = binom * wet::pow(wo, static_cast<int>(i));
    }

    T Kp{};
    T Kd{};
    if constexpr (NX == 1) {
        Kp = wc / b0;
        Kd = T{0};
    } else if constexpr (NX == 2) {
        Kp = wc * wc / b0;
        Kd = 2 * wc / b0;
    } else {
        // For higher order systems, use 2nd order gains as default
        Kp = wc * wc / b0;
        Kd = 2 * wc / b0;
    }

    return ADRCResult<NX, T>{wc, wo, b0, beta, Kp, Kd, Ts};
}

} // namespace design

/**
 * @ingroup discrete_controllers
 * @brief Active Disturbance Rejection Control (ADRC)
 *
 * Discrete ADRC with Extended State Observer (ESO) for SISO systems.
 * Supports both 1st-order and 2nd-order plant models.
 * Based on the work of Jingqing Han and Zhiqiang Gao.
 *
 * References:
 * - Han, J. (1998). From PID to Active Disturbance Rejection Control. IEEE Transactions on Industrial Electronics, 45(5), 900-906.
 * - Gao, Z. (2006). Active Disturbance Rejection Control: A Paradigm Shift in Feedback Control System Design. American Control Conference.
 *
 * @tparam NX Number of plant states (1 for 1st-order, 2 for 2nd-order systems)
 * @tparam T Scalar type (default: float)
 */
template<size_t NX, typename T = float>
class ADRCController {
    T b0{1.0f}; //< input gain
    T Kp{1.0f}; //< proportional gain
    T Kd{1.0f}; //< derivative gain
    T Ts{1.0f}; //< sampling time

    std::array<T, NX + 1> beta{};             //< ESO gains [β1, β2, ..., β_{NX+1}]
    ColVec<NX + 1, T>     z{};                //< ESO state: [z1, z2, ..., z_{NX+1}]
    T                     u_prev_{T{0}};      //< Last applied command (post-saturation)
    T                     ki_factor_{T{1.0}}; //< Integrator gain-scheduling factor

public:
    constexpr ADRCController() = default;
    constexpr ADRCController(const design::ADRCResult<NX, T>& result)
        : b0(result.b0), Kp(result.Kp), Kd(result.Kd), Ts(result.Ts), beta(result.beta) {}

    template<typename U>
    constexpr ADRCController(const ADRCController<NX, U>& other)
        : b0(other.b0), Kp(other.Kp), Kd(other.Kd), Ts(other.Ts), beta(other.beta), z(other.z), u_prev_(other.u_prev_), ki_factor_(other.ki_factor_) {}

    /**
     * @brief Set the integrator gain-scheduling factor used by the 2-arg
     *        @ref control overload. Defaults to 1.0.
     */
    constexpr void set_ki_factor(T factor) { ki_factor_ = factor; }

    /**
     * @brief Reference-tracking overload satisfying @ref SISOController.
     *
     * The ESO uses the previously-applied command (stored in `u_prev_`) for
     * its update. After computing a new command this overload optimistically
     * stores the unsaturated value in `u_prev_`; if a downstream stage clamps
     * the command, the caller should follow up with `back_calculate(u_unsat,
     * u_sat)` so the ESO's next tick uses what was actually applied.
     */
    [[nodiscard]] constexpr T control(T r, T y) {
        const T u = control(r, y, u_prev_, ki_factor_);
        u_prev_ = u;
        return u;
    }

    /**
     * @brief Compute ADRC control with explicit u_prev / ki_factor (legacy).
     *
     * @param r        Reference
     * @param y        Measurement
     * @param u_prev   Previous control command after any external saturation
     * @param ki_factor Integrator gain-scheduling factor (default: 1.0)
     * @return Control output u
     *
     * Prefer the 2-arg overload + `back_calculate` for new code.
     */
    [[nodiscard]] constexpr T control(T r, T y, T u_prev, T ki_factor = T{1.0}) {
        T e = y - z[0]; // Innovation

        // Error State Observer update
        z[0] += Ts * (z[1] + beta[0] * e + b0 * u_prev);
        for (size_t i = 1; i < NX; ++i) {
            z[i] += Ts * (z[i + 1] + beta[i] * e);
        }
        z[NX] += Ts * beta[NX] * ki_factor * e; // Gain-scheduled Disturbance estimate

        // Controller (order-dependent)
        T u{};
        if constexpr (NX == 1) {
            // 1st order: P control + disturbance rejection
            u = Kp * (r - z[0]) - (z[1] / b0);
        } else {
            // 2nd+ order: PD control + disturbance rejection
            u = Kp * (r - z[0]) - (Kd * z[1]) - (z[NX] / b0);
        }
        return u;
    }

    /**
     * @brief Anti-windup hook for cascade-level saturation propagation.
     *
     * Overrides the stored `u_prev_` with the actually-realized command so
     * the next ESO tick reflects what the plant received, not what the
     * controller wanted. This is the natural ADRC anti-windup behavior:
     * the ESO's disturbance estimate self-corrects once it sees the true
     * input, without an explicit integrator unwind step.
     */
    constexpr void back_calculate(T u_unsat, T u_sat) {
        (void)u_unsat;
        u_prev_ = u_sat;
    }

    constexpr void reset() {
        z = ColVec<NX + 1, T>{};
        u_prev_ = T{0};
    }
};
} // namespace wet