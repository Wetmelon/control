#pragma once

/**
 * @defgroup lead_lag Lead-Lag Compensator
 * @brief Frequency-domain compensator design for loop shaping
 *
 * A lead-lag compensator has the transfer function:
 *
 *   C(s) = K * (s + z) / (s + p)
 *
 * - Lead (z < p): Adds phase at a target frequency. Used to increase
 *   phase margin and improve transient response. Analogous to adding
 *   derivative action in a bounded frequency range.
 *
 * - Lag (z > p): Adds low-frequency gain without disturbing crossover.
 *   Used to reduce steady-state error. Analogous to adding integral
 *   action in a bounded frequency range.
 *
 * Design functions compute zero/pole locations and gain from intuitive
 * specifications (desired phase boost, crossover frequency, etc.).
 *
 * Common applications:
 * - Voltage/current loop compensation in DC-DC converters
 * - Phase margin improvement in feedback loops
 * - Gain shaping for disturbance rejection
 * - Cascade compensator design (lead + lag sections)
 */

#include <cmath>

#include "constexpr_math.hpp"
#include "discretization.hpp"
#include "state_space.hpp"
#include "transfer_function.hpp"

namespace wetmelon::control {

namespace design {

/**
 * @struct LeadLagResult
 * @brief Lead-lag compensator design result
 */
template<typename T = double>
struct LeadLagResult {
    T K{};  //!< DC gain of the compensator
    T z{};  //!< Zero location (rad/s, positive value; actual zero at s = -z)
    T p{};  //!< Pole location (rad/s, positive value; actual pole at s = -p)
    T Ts{}; //!< Sampling time (seconds), 0 = continuous

    template<typename U>
    [[nodiscard]] consteval LeadLagResult<U> as() const {
        return {static_cast<U>(K), static_cast<U>(z), static_cast<U>(p), static_cast<U>(Ts)};
    }

    /**
     * @brief Convert to continuous-time transfer function
     *
     * Returns C(s) = K * (s + z) / (s + p)
     * Numerator and denominator in ascending powers of s: {const, s}
     */
    [[nodiscard]] consteval TransferFunction<2, 2, T> to_tf() const {
        // C(s) = K * (s + z) / (s + p)
        // Numerator:   K*z + K*s   = {K*z, K}
        // Denominator: p + s       = {p, 1}
        return TransferFunction<2, 2, T>{
            .num = {K * z, K},
            .den = {p, T{1}},
        };
    }

    /**
     * @brief Convert to continuous-time state-space (1st order SISO)
     */
    [[nodiscard]] consteval StateSpace<1, 1, 1, 0, 0, T> to_ss() const {
        // From C(s) = K*(s+z)/(s+p), controllable canonical form:
        // A = [-p], B = [1], C = [K*(z-p)], D = [K]
        return StateSpace<1, 1, 1, 0, 0, T>{
            .A = Matrix<1, 1, T>{{-p}},
            .B = Matrix<1, 1, T>{{T{1}}},
            .C = Matrix<1, 1, T>{{K * (z - p)}},
            .D = Matrix<1, 1, T>{{K}},
        };
    }

    /**
     * @brief Convert to discrete-time state-space
     *
     * @param method Discretization method (default: Tustin)
     */
    [[nodiscard]] consteval StateSpace<1, 1, 1, 0, 0, T>
    to_discrete_ss(DiscretizationMethod method = DiscretizationMethod::Tustin) const {
        return discretize(to_ss(), Ts, method);
    }
};

/**
 * @brief Design a lead compensator from desired phase boost at a target frequency
 *
 * Places zero and pole symmetrically (in log-frequency) around wc so that
 * maximum phase advance occurs exactly at wc.
 *
 * The phase boost at wc is: phi_max = atan(1/sqrt(alpha)) - atan(sqrt(alpha))
 * where alpha = z/p < 1 for a lead compensator.
 *
 * Gain is set so that |C(j*wc)| = 1 (unity gain at crossover), meaning
 * the compensator adds phase without shifting the gain crossover frequency.
 *
 * @param phi_max  Desired maximum phase boost (radians, 0 < phi_max < pi/2)
 * @param wc       Target crossover frequency (rad/s)
 * @param Ts       Sampling time (seconds), 0 = continuous-time design
 * @return LeadLagResult with computed K, zero, and pole locations
 */
template<typename T = double>
[[nodiscard]] consteval LeadLagResult<T> lead(T phi_max, T wc, T Ts = T{0}) {
    // alpha = (1 - sin(phi)) / (1 + sin(phi))
    // For a lead: alpha < 1
    T s = wet::sin(phi_max);
    T alpha = (T{1} - s) / (T{1} + s);

    // Zero and pole placed symmetrically around wc in log-frequency:
    // z = wc * sqrt(alpha),  p = wc / sqrt(alpha)
    // So geometric mean = wc, and z < p (lead).
    T sqrt_alpha = wet::sqrt(alpha);
    T z = wc * sqrt_alpha;
    T p = wc / sqrt_alpha;

    // Gain set for unity magnitude at wc: |C(j*wc)| = 1
    // |C(j*wc)| = K * sqrt(wc²+z²) / sqrt(wc²+p²)
    //           = K * sqrt(1 + alpha) / sqrt(1 + 1/alpha)
    //           = K * sqrt(alpha)
    // So K = 1/sqrt(alpha) for unity gain at crossover.
    T K = T{1} / sqrt_alpha;

    return LeadLagResult<T>{K, z, p, Ts};
}

/**
 * @brief Design a lag compensator from desired low-frequency gain boost
 *
 * A lag compensator adds gain at low frequencies (below wc) while
 * introducing minimal phase lag at and above the crossover frequency wc.
 *
 * The zero is placed at wc/gain_factor, and the pole at wc/(gain_factor²)
 * so the lag section is well below crossover.  DC gain boost ≈ gain_factor.
 *
 * @param dc_gain_boost  Desired DC gain increase factor (> 1, e.g. 10 for +20 dB)
 * @param wc             Crossover frequency to stay below (rad/s)
 * @param margin_factor  How far below wc to place the compensator (default: 10).
 *                       Higher values = less phase disturbance at wc.
 * @param Ts             Sampling time (seconds), 0 = continuous
 * @return LeadLagResult with computed K, zero, and pole locations
 */
template<typename T = double>
[[nodiscard]] consteval LeadLagResult<T> lag(T dc_gain_boost, T wc, T margin_factor = T{10}, T Ts = T{0}) {
    // Place zero at wc / margin_factor
    // Place pole at zero / dc_gain_boost (further below, so z > p → lag)
    T z = wc / margin_factor;
    T p = z / dc_gain_boost;

    // DC gain of (s+z)/(s+p) → z/p = dc_gain_boost
    // Set K = 1 so the total DC gain is exactly dc_gain_boost
    T K = T{1};

    return LeadLagResult<T>{K, z, p, Ts};
}

/**
 * @brief Design a lead-lag compensator (cascade of lead + lag sections)
 *
 * Returns a 2nd-order system combining a lead section (for phase margin)
 * and a lag section (for DC gain). The two sections are cascaded:
 *
 *   C(s) = K_lead * (s+z_lead)/(s+p_lead) * (s+z_lag)/(s+p_lag)
 *
 * @param phi_max        Desired phase boost from lead section (radians)
 * @param wc             Target crossover frequency (rad/s)
 * @param dc_gain_boost  Desired DC gain increase from lag section (> 1)
 * @param margin_factor  How far below wc to place lag section (default: 10)
 * @param Ts             Sampling time (seconds), 0 = continuous
 * @return StateSpace<2,1,1> — 2nd order SISO compensator
 */
template<typename T = double>
[[nodiscard]] consteval StateSpace<2, 1, 1, 0, 0, T>
lead_lag(T phi_max, T wc, T dc_gain_boost, T margin_factor = T{10}, T Ts = T{0}) {
    auto lead_r = lead(phi_max, wc);
    auto lag_r = lag(dc_gain_boost, wc, margin_factor);

    // Cascade: series connection of lead and lag state-space representations
    auto lead_ss = lead_r.to_ss();
    auto lag_ss = lag_r.to_ss();
    auto combined = series(lead_ss, lag_ss);

    if (Ts > T{0}) {
        return discretize(combined, Ts, DiscretizationMethod::Tustin);
    }
    return combined;
}

/**
 * @brief Direct lead-lag specification from zero/pole locations
 *
 * For users who already know their zero and pole locations.
 *
 * @param K   Gain
 * @param z   Zero location (positive; actual zero at s = -z)
 * @param p   Pole location (positive; actual pole at s = -p)
 * @param Ts  Sampling time (seconds), 0 = continuous
 * @return LeadLagResult
 */
template<typename T = double>
[[nodiscard]] consteval LeadLagResult<T> lead_lag_direct(T K, T z, T p, T Ts = T{0}) {
    return LeadLagResult<T>{K, z, p, Ts};
}

} // namespace design

namespace online {

/**
 * @struct LeadLagResult
 * @brief Lead-lag compensator design result (runtime)
 */
template<typename T = double>
struct LeadLagResult {
    T K{};
    T z{};
    T p{};
    T Ts{};

    template<typename U>
    [[nodiscard]] constexpr LeadLagResult<U> as() const {
        return {static_cast<U>(K), static_cast<U>(z), static_cast<U>(p), static_cast<U>(Ts)};
    }

    [[nodiscard]] constexpr TransferFunction<2, 2, T> to_tf() const {
        return TransferFunction<2, 2, T>{
            .num = {K * z, K},
            .den = {p, T{1}},
        };
    }

    [[nodiscard]] constexpr StateSpace<1, 1, 1, 0, 0, T> to_ss() const {
        return StateSpace<1, 1, 1, 0, 0, T>{
            .A = Matrix<1, 1, T>{{-p}},
            .B = Matrix<1, 1, T>{{T{1}}},
            .C = Matrix<1, 1, T>{{K * (z - p)}},
            .D = Matrix<1, 1, T>{{K}},
        };
    }

    [[nodiscard]] constexpr StateSpace<1, 1, 1, 0, 0, T>
    to_discrete_ss(DiscretizationMethod method = DiscretizationMethod::Tustin) const {
        return discretize(to_ss(), Ts, method);
    }
};

/**
 * @brief Design a lead compensator (runtime)
 */
template<typename T = double>
[[nodiscard]] constexpr LeadLagResult<T> lead(T phi_max, T wc, T Ts = T{0}) {
    T s = wet::sin(phi_max);
    T alpha = (T{1} - s) / (T{1} + s);
    T sqrt_alpha = wet::sqrt(alpha);
    T z = wc * sqrt_alpha;
    T p = wc / sqrt_alpha;
    T K = T{1} / sqrt_alpha;
    return LeadLagResult<T>{K, z, p, Ts};
}

/**
 * @brief Design a lag compensator (runtime)
 */
template<typename T = double>
[[nodiscard]] constexpr LeadLagResult<T> lag(T dc_gain_boost, T wc, T margin_factor = T{10}, T Ts = T{0}) {
    T z = wc / margin_factor;
    T p = z / dc_gain_boost;
    T K = T{1};
    return LeadLagResult<T>{K, z, p, Ts};
}

/**
 * @brief Direct lead-lag specification (runtime)
 */
template<typename T = double>
[[nodiscard]] constexpr LeadLagResult<T> lead_lag_direct(T K, T z, T p, T Ts = T{0}) {
    return LeadLagResult<T>{K, z, p, Ts};
}

} // namespace online

/**
 * @ingroup discrete_controllers
 * @brief Discrete Lead-Lag Compensator
 *
 * Implements C(s) = K*(s+z)/(s+p) discretized via Tustin transform.
 * This is a 1st-order IIR filter.
 *
 * @tparam T Scalar type (default: float)
 */
template<typename T = float>
struct LeadLagController {
    T K{};
    T z{};
    T p{};
    T Ts{};

    // Pre-computed IIR coefficients (1st-order: y[n] = b0*u[n] + b1*u[n-1] - a1*y[n-1])
    T b0{0};
    T b1{0};
    T a1{0};

    // State
    T u_prev{0};
    T y_prev{0};

    constexpr LeadLagController() = default;

    consteval LeadLagController(const design::LeadLagResult<T>& result)
        : K(result.K), z(result.z), p(result.p), Ts(result.Ts) {
        compute_coefficients();
    }

    constexpr LeadLagController(const online::LeadLagResult<T>& result)
        : K(result.K), z(result.z), p(result.p), Ts(result.Ts) {
        compute_coefficients();
    }

    template<typename U>
    constexpr LeadLagController(const LeadLagController<U>& other)
        : K(other.K), z(other.z), p(other.p), Ts(other.Ts), b0(other.b0), b1(other.b1), a1(other.a1), u_prev(other.u_prev), y_prev(other.y_prev) {}

    /**
     * @brief Compute compensator output
     *
     * @param u Input signal (typically error signal)
     * @return Compensated output
     */
    [[nodiscard]] constexpr T control(T u) {
        T y = b0 * u + b1 * u_prev - a1 * y_prev;
        u_prev = u;
        y_prev = y;
        return y;
    }

    constexpr void reset() {
        u_prev = T{0};
        y_prev = T{0};
    }

private:
    /**
     * @brief Compute Tustin-discretized IIR coefficients
     *
     * C(s) = K*(s+z)/(s+p)
     * Tustin: s = (2/Ts)*(z-1)/(z+1)
     *
     * Numerator:   K * ((2/Ts)*(z-1)/(z+1) + z_loc)
     *            = K * ((2/Ts + z_loc)*z + (z_loc - 2/Ts)) / (z+1)
     *
     * Denominator: (2/Ts)*(z-1)/(z+1) + p_loc
     *            = ((2/Ts + p_loc)*z + (p_loc - 2/Ts)) / (z+1)
     */
    constexpr void compute_coefficients() {
        T k = T{2} / Ts; // Tustin substitution factor

        // Denominator coefficients (before normalization)
        T a0_raw = k + p;
        T a1_raw = p - k;

        // Numerator coefficients (before normalization)
        T b0_raw = K * (k + z);
        T b1_raw = K * (z - k);

        // Normalize by a0
        b0 = b0_raw / a0_raw;
        b1 = b1_raw / a0_raw;
        a1 = a1_raw / a0_raw;
    }
};

} // namespace wetmelon::control
