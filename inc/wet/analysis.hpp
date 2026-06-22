#pragma once

/**
 * @defgroup analysis System Analysis
 * @brief Frequency-domain and structural analysis for LTI systems
 *
 * Provides Bode plot data, gain/phase margins, bandwidth, DC gain,
 * controllability matrix, observability matrix, and controllability/observability
 * rank tests. All functions are constexpr-compatible.
 */

#include <cstddef>
#include <limits>
#include <vector>

#include "wet/math/complex.hpp"
#include "wet/math/math.hpp"
#include "wet/matrix/eigen.hpp"
#include "wet/matrix/matrix.hpp"
#include "wet/systems/discretization.hpp"
#include "wet/systems/state_space.hpp"
#include "wet/systems/transfer_function.hpp"
#include "wet/systems/zpk.hpp"

namespace wet {
namespace analysis {

template<typename T = double>
    requires std::is_floating_point_v<T>
constexpr std::vector<T> linspace(T start, T end, size_t num) {
    std::vector<T> result;
    result.reserve(num);
    if (num == 1) {
        result.push_back(start);
    } else {
        T step = (end - start) / static_cast<T>(num - 1);
        for (size_t i = 0; i < num; ++i) {
            result.push_back(start + (static_cast<T>(i) * step));
        }
    }
    return result;
}

template<typename T = double>
    requires std::is_floating_point_v<T>
constexpr std::vector<T> linspace(const wet::pair<T, T>& span, size_t num) {
    return linspace(span.first, span.second, num);
}

template<typename T = double>
    requires std::is_floating_point_v<T>
constexpr std::vector<T> logspace(T start, T end, size_t num, T base = T{10}) {
    std::vector<T> result;
    result.reserve(num);
    if (num == 1) {
        result.push_back(start);
    } else {
        // Compute logarithms in the requested base
        const T log_base = wet::log(base);
        T       log_start = wet::log(start) / log_base;
        T       log_end = wet::log(end) / log_base;
        T       step = (log_end - log_start) / static_cast<T>(num - 1);
        for (size_t i = 0; i < num; ++i) {
            result.push_back(wet::pow(base, log_start + (static_cast<T>(i) * step)));
        }
    }
    return result;
}

template<typename T = double>
    requires std::is_floating_point_v<T>
constexpr std::vector<T> logspace(const wet::pair<T, T>& span, size_t num, T base = T{10}) {
    return logspace(span.first, span.second, num, base);
}

// ============================================================================
// Frequency Response Data
// ============================================================================

/**
 * @brief Single-point frequency response result
 */
template<typename T = double>
struct FrequencyPoint {
    T omega{};        //!< Frequency (rad/s)
    T magnitude{};    //!< Magnitude (absolute)
    T magnitude_db{}; //!< Magnitude (dB)
    T phase_deg{};    //!< Phase (degrees)
};

/**
 * @brief Bode plot data for a SISO system
 */
template<typename T = double>
struct BodeResult {
    std::vector<FrequencyPoint<T>> points; //!< Frequency response data points

    /**
     * @brief Find gain margin (dB above 1.0 at -180° phase crossing)
     *
     * Gain margin is 1/|G(jω)| at the frequency where phase crosses -180°.
     * Positive values indicate stability.
     *
     * @return {gain_margin_dB, crossover_frequency} or nullopt if no -180° crossing
     */
    [[nodiscard]] constexpr wet::optional<wet::pair<T, T>> gain_margin() const {
        for (size_t i = 1; i < points.size(); ++i) {
            T phase_prev = points[i - 1].phase_deg;
            T phase_curr = points[i].phase_deg;
            // Looking for phase crossing -180°
            if ((phase_prev > T{-180} && phase_curr <= T{-180}) || (phase_prev < T{-180} && phase_curr >= T{-180})) {
                // Linear interpolation for crossing frequency
                T frac = (T{-180} - phase_prev) / (phase_curr - phase_prev);
                T omega_cross = points[i - 1].omega + frac * (points[i].omega - points[i - 1].omega);
                T mag_db_cross = points[i - 1].magnitude_db + frac * (points[i].magnitude_db - points[i - 1].magnitude_db);
                T gm = -mag_db_cross; // Gain margin in dB
                return wet::pair{gm, omega_cross};
            }
        }
        return wet::nullopt;
    }

    /**
     * @brief Find phase margin (degrees above -180° at 0dB gain crossing)
     *
     * Phase margin is 180° + phase(G(jω)) at the frequency where |G(jω)| = 1 (0 dB).
     * Positive values indicate stability.
     *
     * @return {phase_margin_deg, crossover_frequency} or nullopt if no 0dB crossing
     */
    [[nodiscard]] constexpr wet::optional<wet::pair<T, T>> phase_margin() const {
        for (size_t i = 1; i < points.size(); ++i) {
            T mag_prev = points[i - 1].magnitude_db;
            T mag_curr = points[i].magnitude_db;
            // Looking for magnitude crossing 0 dB (from above)
            if (mag_prev >= T{0} && mag_curr < T{0}) {
                T frac = (T{0} - mag_prev) / (mag_curr - mag_prev);
                T omega_cross = points[i - 1].omega + frac * (points[i].omega - points[i - 1].omega);
                T phase_cross = points[i - 1].phase_deg + frac * (points[i].phase_deg - points[i - 1].phase_deg);
                T pm = T{180} + phase_cross;
                return wet::pair{pm, omega_cross};
            }
        }
        return wet::nullopt;
    }

    /**
     * @brief Find -3dB bandwidth
     *
     * The frequency at which the magnitude drops 3 dB below the DC (or peak) value.
     *
     * @return Bandwidth in rad/s, or nullopt if not found
     */
    [[nodiscard]] constexpr wet::optional<T> bandwidth() const {
        if (points.empty()) {
            return wet::nullopt;
        }
        T dc_db = points[0].magnitude_db;
        T threshold = dc_db - T{3};
        for (size_t i = 1; i < points.size(); ++i) {
            if (points[i].magnitude_db < threshold) {
                // Interpolate
                T frac = (threshold - points[i - 1].magnitude_db) / (points[i].magnitude_db - points[i - 1].magnitude_db);
                return points[i - 1].omega + frac * (points[i].omega - points[i - 1].omega);
            }
        }
        return wet::nullopt;
    }
};

/**
 * @brief Unwrap phase data in degrees to avoid +/-180 discontinuities
 *
 * Given wrapped phase samples (typically in [-180, 180]), produces a
 * continuous phase trajectory by adding/subtracting 360 deg at jumps.
 *
 * @param phase_deg Wrapped phase samples in degrees
 * @return Unwrapped phase samples in degrees
 */
template<typename T>
[[nodiscard]] constexpr std::vector<T> unwrap_phase_deg(const std::vector<T>& phase_deg) {
    if (phase_deg.empty()) {
        return {};
    }

    std::vector<T> unwrapped = phase_deg;
    for (size_t i = 1; i < unwrapped.size(); ++i) {
        const T delta = unwrapped[i] - unwrapped[i - 1];
        // Range-reduce to nearest equivalent increment in [-180, 180).
        const T n = wet::floor(delta / T{360} + T{0.5});
        const T delta_reduced = delta - T{360} * n;
        unwrapped[i] = unwrapped[i - 1] + delta_reduced;
    }
    return unwrapped;
}

/**
 * @brief Normalize phase margin to (-180, 180]
 *
 * @param pm_deg Raw phase margin in degrees
 * @return Canonical phase margin in (-180, 180]
 */
template<typename T>
[[nodiscard]] constexpr T canonical_phase_margin(T pm_deg) {
    // Reuse shared wrapping helper, then remap to (-180, 180].
    const T wrapped = wrap(pm_deg, T{-180}, T{180});
    if (wrapped <= T{-180}) {
        return wrapped + T{360};
    }
    return wrapped;
}

/**
 * @brief Find phase margin using unwrapped phase trajectory
 *
 * Uses 0 dB crossing from Bode magnitude and computes PM as 180 + phase(wc),
 * then maps to canonical range (-180, 180].
 *
 * @return {phase_margin_deg, crossover_frequency} or nullopt if no 0dB crossing
 */
template<typename T>
[[nodiscard]] constexpr wet::optional<wet::pair<T, T>> phase_margin_unwrapped(const BodeResult<T>& result) {
    if (result.points.size() < 2) {
        return wet::nullopt;
    }

    std::vector<T> wrapped_phase;
    wrapped_phase.reserve(result.points.size());
    for (const auto& pt : result.points) {
        wrapped_phase.push_back(pt.phase_deg);
    }
    const auto unwrapped = unwrap_phase_deg(wrapped_phase);

    for (size_t i = 1; i < result.points.size(); ++i) {
        const T mag_prev = result.points[i - 1].magnitude_db;
        const T mag_curr = result.points[i].magnitude_db;
        if (mag_prev >= T{0} && mag_curr < T{0}) {
            const T frac = (T{0} - mag_prev) / (mag_curr - mag_prev);
            const T omega_cross = result.points[i - 1].omega + frac * (result.points[i].omega - result.points[i - 1].omega);
            const T phase_cross = unwrapped[i - 1] + frac * (unwrapped[i] - unwrapped[i - 1]);
            const T pm = canonical_phase_margin(T{180} + phase_cross);
            return wet::pair{pm, omega_cross};
        }
    }

    return wet::nullopt;
}

/**
 * @brief Find gain margin using unwrapped phase trajectory
 *
 * Finds the first -180 deg crossing on the unwrapped phase trajectory and
 * computes gain margin as -|L|_dB at that crossing.
 *
 * @return {gain_margin_dB, crossover_frequency} or nullopt if no -180 crossing
 */
template<typename T>
[[nodiscard]] constexpr wet::optional<wet::pair<T, T>> gain_margin_unwrapped(const BodeResult<T>& result) {
    if (result.points.size() < 2) {
        return wet::nullopt;
    }

    std::vector<T> wrapped_phase;
    wrapped_phase.reserve(result.points.size());
    for (const auto& pt : result.points) {
        wrapped_phase.push_back(pt.phase_deg);
    }
    const auto unwrapped = unwrap_phase_deg(wrapped_phase);

    for (size_t i = 1; i < result.points.size(); ++i) {
        const T p0 = unwrapped[i - 1] + T{180};
        const T p1 = unwrapped[i] + T{180};

        if ((p0 >= T{0} && p1 < T{0}) || (p0 <= T{0} && p1 > T{0})) {
            const T frac = (T{0} - p0) / (p1 - p0);
            const T omega_cross = result.points[i - 1].omega + frac * (result.points[i].omega - result.points[i - 1].omega);
            const T mag_cross = result.points[i - 1].magnitude_db + frac * (result.points[i].magnitude_db - result.points[i - 1].magnitude_db);
            return wet::pair{-mag_cross, omega_cross};
        }
    }

    return wet::nullopt;
}

/**
 * @brief Compute Bode plot data for a SISO state-space system
 *
 * Continuous-time systems evaluate G(jω) = C(jωI - A)^{-1}B + D.
 * Discrete-time systems (Ts > 0) evaluate G(z) on the unit circle
 * z = e^{jωTs}. Dispatch is automatic based on sys.is_discrete().
 *
 * @param sys   SISO state-space system (continuous or discrete)
 * @param omega Vector of frequencies (rad/s)
 * @return BodeResult with magnitude and phase at each frequency
 */
template<size_t NX, size_t NW, size_t NV, typename T>
[[nodiscard]] constexpr BodeResult<T> bode(
    const StateSpace<NX, 1, 1, NW, NV, T>& sys,
    const std::vector<T>&                  omega
) {
    using C = wet::complex<T>;
    BodeResult<T> result;
    result.points.reserve(omega.size());

    for (const auto& w : omega) {
        const C s_or_z = sys.is_discrete()
                           ? C{wet::cos(w * sys.Ts), wet::sin(w * sys.Ts)}
                           : C{T{0}, w};
        auto    G_frf = eval_frf(sys, s_or_z);
        C       G = G_frf(0, 0);
        T       mag = wet::abs(G);
        T       phase_rad = wet::arg(G);
        T       mag_db = mag > T{0} ? T{20} * wet::log10(mag) : T{-300};
        T       phase_deg = phase_rad * T{180} / wet::numbers::pi_v<T>;
        result.points.push_back({w, mag, mag_db, phase_deg});
    }
    return result;
}

/**
 * @brief Compute Bode plot data for a SISO transfer function
 *
 * Evaluates H(jω) = num(jω)/den(jω) over a vector of frequencies.
 *
 * @param num   Numerator coefficients (ascending powers of s)
 * @param den   Denominator coefficients (ascending powers of s)
 * @param omega Vector of frequencies (rad/s)
 * @return BodeResult with magnitude and phase at each frequency
 */
template<size_t Nnum, size_t Nden, typename T>
[[nodiscard]] constexpr BodeResult<T> bode(
    const wet::array<T, Nnum>& num,
    const wet::array<T, Nden>& den,
    const std::vector<T>&      omega
) {
    using C = wet::complex<T>;
    BodeResult<T> result;
    result.points.reserve(omega.size());

    for (const auto& w : omega) {
        C jw{T{0}, w};

        // Evaluate numerator polynomial at jw (ascending powers)
        C num_val{T{0}, T{0}};
        C jw_power{T{1}, T{0}};
        for (size_t i = 0; i < Nnum; ++i) {
            num_val = num_val + num[i] * jw_power;
            jw_power = jw_power * jw;
        }

        // Evaluate denominator polynomial at jw (ascending powers)
        C den_val{T{0}, T{0}};
        jw_power = C{T{1}, T{0}};
        for (size_t i = 0; i < Nden; ++i) {
            den_val = den_val + den[i] * jw_power;
            jw_power = jw_power * jw;
        }

        C G = num_val / den_val;
        T mag = wet::abs(G);
        T phase_rad = wet::arg(G);
        T mag_db = mag > T{0} ? T{20} * wet::log10(mag) : T{-300};
        T phase_deg = phase_rad * T{180} / wet::numbers::pi_v<T>;
        result.points.push_back({w, mag, mag_db, phase_deg});
    }
    return result;
}

/**
 * @brief Compute Bode plot data for a SISO transfer function object
 */
template<size_t Nnum, size_t Nden, typename T>
[[nodiscard]] constexpr BodeResult<T> bode(
    const TransferFunction<Nnum, Nden, T>& tf,
    const std::vector<T>&                  omega
) {
    return bode(tf.num, tf.den, omega);
}

/**
 * @brief Compute Bode plot data for a discrete-time SISO state-space system
 *
 * Evaluates G(z) on the unit circle z = e^(j*omega*Ts), where omega is in rad/s.
 *
 * @note bode() now auto-dispatches on sys.is_discrete(); this alias is retained
 *       for discoverability and call sites that want to be explicit.
 *
 * @param sys   Discrete-time SISO state-space system
 * @param omega Vector of frequencies (rad/s)
 * @return BodeResult with magnitude and phase at each frequency
 */
template<size_t NX, size_t NW, size_t NV, typename T>
[[nodiscard]] constexpr BodeResult<T> bode_discrete(
    const StateSpace<NX, 1, 1, NW, NV, T>& sys,
    const std::vector<T>&                  omega
) {
    return bode(sys, omega);
}

/**
 * @brief Single-point Nyquist response data
 */
template<typename T = double>
struct NyquistPoint {
    T               omega{};                 //!< Frequency (rad/s)
    wet::complex<T> value{};                 //!< Complex loop value (L(jω) or L(e^{jω Ts}))
    T               real{};                  //!< Real part
    T               imag{};                  //!< Imaginary part
    T               distance_to_minus_one{}; //!< |1 + L|
};

/**
 * @brief Nyquist response data across a frequency sweep
 */
template<typename T = double>
struct NyquistResult {
    std::vector<NyquistPoint<T>> points;

    /**
     * @brief Minimum Nyquist distance to the critical point -1 + j0
     * @return {minimum_distance, frequency} or nullopt when empty
     */
    [[nodiscard]] constexpr wet::optional<wet::pair<T, T>> min_distance_to_minus_one() const {
        if (points.empty()) {
            return wet::nullopt;
        }

        T min_dist = points[0].distance_to_minus_one;
        T at_omega = points[0].omega;
        for (size_t i = 1; i < points.size(); ++i) {
            if (points[i].distance_to_minus_one < min_dist) {
                min_dist = points[i].distance_to_minus_one;
                at_omega = points[i].omega;
            }
        }
        return wet::pair{min_dist, at_omega};
    }
};

/**
 * @brief Open-loop and closed-loop frequency response package
 *
 * For a loop transfer L, includes:
 * - open_loop: L
 * - sensitivity: S = 1 / (1 + L)
 * - complementary_sensitivity: T = L / (1 + L)
 * - nyquist: complex Nyquist response of L
 */
template<typename T = double>
struct LoopResponseResult {
    BodeResult<T>    open_loop;
    BodeResult<T>    sensitivity;
    BodeResult<T>    complementary_sensitivity;
    NyquistResult<T> nyquist;

    [[nodiscard]] constexpr wet::optional<wet::pair<T, T>> phase_margin_unwrapped() const {
        return analysis::phase_margin_unwrapped(open_loop);
    }

    [[nodiscard]] constexpr wet::optional<wet::pair<T, T>> gain_margin_unwrapped() const {
        return analysis::gain_margin_unwrapped(open_loop);
    }

    [[nodiscard]] constexpr wet::optional<T> closed_loop_bandwidth() const {
        return complementary_sensitivity.bandwidth();
    }
};

template<size_t NX, size_t NW, size_t NV, typename T>
[[nodiscard]] constexpr LoopResponseResult<T> loop_response(
    const StateSpace<NX, 1, 1, NW, NV, T>& loop,
    const std::vector<T>&                  omega
);

/**
 * @brief Compact loop summary metrics for quick stability/robustness checks
 *
 * Designed as a single result payload similar to what users expect from
 * MATLAB/control-toolbox workflows: margins, bandwidth, Nyquist distance,
 * and peak sensitivity in one object.
 */
template<typename T = double>
struct LoopSummary {
    wet::optional<wet::pair<T, T>> phase_margin;                                             //!< {PM [deg], gain crossover omega [rad/s]}
    wet::optional<wet::pair<T, T>> gain_margin;                                              //!< {GM [dB], phase crossover omega [rad/s]}
    wet::optional<T>               bandwidth;                                                //!< Closed-loop bandwidth from T=L/(1+L), rad/s
    wet::optional<wet::pair<T, T>> min_nyquist_distance;                                     //!< {min|1+L|, omega [rad/s]}
    T                              peak_sensitivity_db{-std::numeric_limits<T>::infinity()}; //!< max 20*log10|S|
};

/**
 * @brief Summarize loop_response() results into one compact metrics struct
 */
template<typename T>
[[nodiscard]] constexpr LoopSummary<T> summarize_loop_response(const LoopResponseResult<T>& response) {
    LoopSummary<T> summary{};
    summary.phase_margin = response.phase_margin_unwrapped();
    summary.gain_margin = response.gain_margin_unwrapped();
    summary.bandwidth = response.closed_loop_bandwidth();
    summary.min_nyquist_distance = response.nyquist.min_distance_to_minus_one();

    for (const auto& pt : response.sensitivity.points) {
        if (pt.magnitude_db > summary.peak_sensitivity_db) {
            summary.peak_sensitivity_db = pt.magnitude_db;
        }
    }

    return summary;
}

/**
 * @brief One-call loop analysis: compute L/S/T response and return compact metrics
 */
template<size_t NX, size_t NW, size_t NV, typename T>
[[nodiscard]] constexpr LoopSummary<T> loop_metrics(
    const StateSpace<NX, 1, 1, NW, NV, T>& loop,
    const std::vector<T>&                  omega
) {
    return summarize_loop_response(loop_response(loop, omega));
}

/**
 * @brief Compute Nyquist data for a SISO state-space system
 *
 * Continuous-time systems use s = jω.
 * Discrete-time systems use z = e^{jωTs}.
 */
template<size_t NX, size_t NW, size_t NV, typename T>
[[nodiscard]] constexpr NyquistResult<T> nyquist(
    const StateSpace<NX, 1, 1, NW, NV, T>& sys,
    const std::vector<T>&                  omega
) {
    using C = wet::complex<T>;
    NyquistResult<T> result;
    result.points.reserve(omega.size());

    for (const auto& w : omega) {
        const C         s_or_z = sys.is_discrete()
                                   ? C{wet::cos(w * sys.Ts), wet::sin(w * sys.Ts)}
                                   : C{T{0}, w};
        const auto      G_frf = eval_frf(sys, s_or_z);
        const C         G = G_frf(0, 0);
        const T         dist = wet::abs(C{T{1}, T{0}} + G);
        NyquistPoint<T> point{};
        point.omega = w;
        point.value = G;
        point.real = G.real();
        point.imag = G.imag();
        point.distance_to_minus_one = dist;
        result.points.push_back(point);
    }

    return result;
}

/**
 * @brief Compute Nyquist data for a SISO transfer function
 */
template<size_t Nnum, size_t Nden, typename T>
[[nodiscard]] constexpr NyquistResult<T> nyquist(
    const TransferFunction<Nnum, Nden, T>& tf,
    const std::vector<T>&                  omega
) {
    return nyquist(tf.to_state_space(), omega);
}

/**
 * @brief Compute open-loop L, sensitivity S, complementary sensitivity T, and Nyquist data
 *
 * For each frequency point:
 *   S = 1/(1+L),  T = L/(1+L)
 *
 * Continuous-time systems use s = jω.
 * Discrete-time systems use z = e^{jωTs}.
 */
template<size_t NX, size_t NW, size_t NV, typename T>
[[nodiscard]] constexpr LoopResponseResult<T> loop_response(
    const StateSpace<NX, 1, 1, NW, NV, T>& loop,
    const std::vector<T>&                  omega
) {
    using C = wet::complex<T>;

    LoopResponseResult<T> result;
    result.open_loop.points.reserve(omega.size());
    result.sensitivity.points.reserve(omega.size());
    result.complementary_sensitivity.points.reserve(omega.size());
    result.nyquist.points.reserve(omega.size());

    for (const auto& w : omega) {
        const C s_or_z = loop.is_discrete()
                           ? C{wet::cos(w * loop.Ts), wet::sin(w * loop.Ts)}
                           : C{T{0}, w};

        const auto L_frf = eval_frf(loop, s_or_z);
        const C    L = L_frf(0, 0);
        const C    one{T{1}, T{0}};
        const C    S = one / (one + L);
        const C    Tresp = L / (one + L);

        const T L_mag = wet::abs(L);
        const T L_mag_db = L_mag > T{0} ? T{20} * wet::log10(L_mag) : T{-300};
        const T L_phase = wet::arg(L) * T{180} / wet::numbers::pi_v<T>;
        result.open_loop.points.push_back({w, L_mag, L_mag_db, L_phase});

        const T S_mag = wet::abs(S);
        const T S_mag_db = S_mag > T{0} ? T{20} * wet::log10(S_mag) : T{-300};
        const T S_phase = wet::arg(S) * T{180} / wet::numbers::pi_v<T>;
        result.sensitivity.points.push_back({w, S_mag, S_mag_db, S_phase});

        const T T_mag = wet::abs(Tresp);
        const T T_mag_db = T_mag > T{0} ? T{20} * wet::log10(T_mag) : T{-300};
        const T T_phase = wet::arg(Tresp) * T{180} / wet::numbers::pi_v<T>;
        result.complementary_sensitivity.points.push_back({w, T_mag, T_mag_db, T_phase});

        const T         dist = wet::abs(one + L);
        NyquistPoint<T> point{};
        point.omega = w;
        point.value = L;
        point.real = L.real();
        point.imag = L.imag();
        point.distance_to_minus_one = dist;
        result.nyquist.points.push_back(point);
    }

    return result;
}

/**
 * @brief Compute open-loop L, sensitivity S, complementary sensitivity T, and Nyquist data for a transfer function
 */
template<size_t Nnum, size_t Nden, typename T>
[[nodiscard]] constexpr LoopResponseResult<T> loop_response(
    const TransferFunction<Nnum, Nden, T>& loop,
    const std::vector<T>&                  omega
) {
    return loop_response(loop.to_state_space(), omega);
}

/**
 * @brief One-call loop analysis for transfer-function loops
 */
template<size_t Nnum, size_t Nden, typename T>
[[nodiscard]] constexpr LoopSummary<T> loop_metrics(
    const TransferFunction<Nnum, Nden, T>& loop,
    const std::vector<T>&                  omega
) {
    return summarize_loop_response(loop_response(loop, omega));
}

/**
 * @brief Compute DC gain of a continuous-time system
 *
 * DC gain = C * (-A)^{-1} * B + D (for continuous-time systems)
 * DC gain = C * (I - A)^{-1} * B + D (for discrete-time systems)
 *
 * @return DC gain matrix, or nullopt if A is singular at the evaluation point
 */
template<size_t NX, size_t NU, size_t NY, size_t NW, size_t NV, typename T>
[[nodiscard]] constexpr wet::optional<Matrix<NY, NU, T>> dcgain(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys
) noexcept {
    Matrix<NX, NX, T> M;
    if (sys.is_continuous()) {
        // G(0) = C * A^{-1} * B + D, but we want -A^{-1} since G(s) = C(sI-A)^{-1}B + D at s=0
        M = -sys.A;
    } else {
        // G(1) = C * (I - A)^{-1} * B + D
        M = Matrix<NX, NX, T>::identity() - sys.A;
    }
    // M⁻¹·B by solving M·X = B (more accurate than forming the inverse).
    auto X = mat::solve(M, sys.B);
    if (!X) {
        return wet::nullopt;
    }
    return sys.C * (*X) + sys.D;
}

/**
 * @brief Compute open-loop poles (eigenvalues of A matrix)
 *
 * @param A State matrix
 * @return Vector of pole locations as complex numbers
 */
template<size_t NX, typename T = double>
[[nodiscard]] constexpr ColVec<NX, wet::complex<T>> poles(const Matrix<NX, NX, T>& A) {
    auto eigen = mat::compute_eigenvalues(A);
    return eigen.values;
}

/**
 * @brief Check continuous-time stability
 *
 * A continuous system is stable if all eigenvalues have Re(λ) < 0.
 *
 * @param A State matrix
 * @return true if all eigenvalues in the left half-plane
 */
template<size_t NX, typename T = double>
[[nodiscard]] constexpr bool is_stable_continuous(const Matrix<NX, NX, T>& A) {
    auto eigen = mat::compute_eigenvalues(A);
    if (!eigen.converged) {
        return false;
    }
    for (size_t i = 0; i < NX; ++i) {
        if (eigen.values[i].real() >= T{0}) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Natural frequency and damping ratio for each pole
 */
template<typename T = double>
struct PoleInfo {
    wet::complex<T> location{};
    T               natural_freq{};  //!< ωn = |pole| (rad/s)
    T               damping_ratio{}; //!< ζ = -Re(pole)/|pole|
    T               time_constant{}; //!< τ = -1/Re(pole) (seconds, for stable poles)
};

/**
 * @brief Compute natural frequency and damping for each pole
 *
 * @param A State matrix
 * @return Array of PoleInfo with ωn, ζ, τ for each pole
 */
template<size_t NX, typename T = double>
[[nodiscard]] constexpr wet::array<PoleInfo<T>, NX>
damp(const Matrix<NX, NX, T>& A) {
    auto                        eigen = mat::compute_eigenvalues(A);
    wet::array<PoleInfo<T>, NX> info{};
    for (size_t i = 0; i < NX; ++i) {
        auto p = eigen.values[i];
        T    wn = wet::abs(p);
        T    zeta = (wn > T{0}) ? -p.real() / wn : T{0};
        T    tau = (p.real() < T{0}) ? T{-1} / p.real() : std::numeric_limits<T>::infinity();
        info[i] = PoleInfo<T>{p, wn, zeta, tau};
    }
    return info;
}

// ============================================================================
// Time-Domain Responses
// ============================================================================

/**
 * @brief SISO time-domain response: output y(t) sampled on a time grid
 */
template<typename T = double>
struct TimeResponse {
    std::vector<T> t; //!< Time points (s)
    std::vector<T> y; //!< Output at each time point
};

namespace detail {

//! Iterate a discrete SISO system y[k]=Cx+Du, x[k+1]=Ax+Bu with constant input u.
template<size_t NX, size_t NW, size_t NV, typename T>
[[nodiscard]] TimeResponse<T> time_response_impl(
    const StateSpace<NX, 1, 1, NW, NV, T>& dsys,
    const Matrix<NX, 1, T>&                x0,
    T                                      u,
    const std::vector<T>&                  time
) {
    TimeResponse<T> r;
    r.t = time;
    r.y.reserve(time.size());
    Matrix<NX, 1, T> x = x0;
    for (size_t k = 0; k < time.size(); ++k) {
        r.y.push_back((dsys.C * x)(0, 0) + dsys.D(0, 0) * u);
        x = dsys.A * x + dsys.B * u;
    }
    return r;
}

//! ZOH-discretize a continuous system onto a uniform time grid (pass-through if already discrete).
template<size_t NX, size_t NW, size_t NV, typename T>
[[nodiscard]] StateSpace<NX, 1, 1, NW, NV, T> discretize_on_grid(
    const StateSpace<NX, 1, 1, NW, NV, T>& sys,
    const std::vector<T>&                  time
) {
    if (sys.is_discrete()) {
        return sys;
    }
    const T dt = (time.size() > 1) ? time[1] - time[0] : T{1};
    return wet::discretize(sys, dt, DiscretizationMethod::ZOH);
}

} // namespace detail

/**
 * @brief Unit step response of a SISO system
 *
 * Computes y(t) for a unit step input u(t)=1, t≥0, from zero initial state.
 * Continuous systems are ZOH-discretized on the (uniform) time grid; discrete
 * systems are iterated directly. MATLAB equivalent: `step(sys, t)`.
 *
 * @param sys  SISO state-space system (continuous or discrete)
 * @param time Uniformly spaced time vector (e.g. analysis::linspace(0, tf, n))
 * @return TimeResponse with time and output history
 */
template<size_t NX, size_t NW, size_t NV, typename T>
[[nodiscard]] TimeResponse<T> step(
    const StateSpace<NX, 1, 1, NW, NV, T>& sys,
    const std::vector<T>&                  time
) {
    auto dsys = detail::discretize_on_grid(sys, time);
    return detail::time_response_impl(dsys, Matrix<NX, 1, T>{}, T{1}, time);
}

/**
 * @brief Impulse response of a SISO system
 *
 * Computes the response to a unit impulse. Implemented as the free response
 * from initial state x₀ = B, giving y(t) = C·e^{At}·B (the strictly-proper
 * part; the D·δ(t) feedthrough term is not plotted). MATLAB equivalent:
 * `impulse(sys, t)`.
 *
 * @param sys  SISO state-space system
 * @param time Uniformly spaced time vector
 * @return TimeResponse with time and output history
 */
template<size_t NX, size_t NW, size_t NV, typename T>
[[nodiscard]] TimeResponse<T> impulse(
    const StateSpace<NX, 1, 1, NW, NV, T>& sys,
    const std::vector<T>&                  time
) {
    auto dsys = detail::discretize_on_grid(sys, time);
    return detail::time_response_impl(dsys, sys.B, T{0}, time);
}

/**
 * @brief Initial-condition (free) response of a SISO system
 *
 * Computes y(t) = C·e^{At}·x₀ for the unforced system (u=0). MATLAB
 * equivalent: `initial(sys, x0, t)`.
 *
 * @param sys  SISO state-space system
 * @param x0   Initial state
 * @param time Uniformly spaced time vector
 * @return TimeResponse with time and output history
 */
template<size_t NX, size_t NW, size_t NV, typename T>
[[nodiscard]] TimeResponse<T> initial(
    const StateSpace<NX, 1, 1, NW, NV, T>& sys,
    const Matrix<NX, 1, T>&                x0,
    const std::vector<T>&                  time
) {
    auto dsys = detail::discretize_on_grid(sys, time);
    return detail::time_response_impl(dsys, x0, T{0}, time);
}

/**
 * @brief Step response of a SISO transfer function (MATLAB `step(tf, t)`)
 */
template<size_t Nnum, size_t Nden, typename T>
[[nodiscard]] TimeResponse<T> step(
    const TransferFunction<Nnum, Nden, T>& tf,
    const std::vector<T>&                  time
) {
    return step(tf.to_state_space(), time);
}

/**
 * @brief Impulse response of a SISO transfer function (MATLAB `impulse(tf, t)`)
 */
template<size_t Nnum, size_t Nden, typename T>
[[nodiscard]] TimeResponse<T> impulse(
    const TransferFunction<Nnum, Nden, T>& tf,
    const std::vector<T>&                  time
) {
    return impulse(tf.to_state_space(), time);
}

// ============================================================================
// Impedance / Middlebrook Stability Analysis
// ============================================================================

/**
 * @brief Result of impedance frequency response evaluation
 *
 * Contains complex impedance at each frequency point, plus
 * magnitude/phase representation for Bode-style plotting.
 */
template<typename T = double>
struct ImpedanceResult {
    struct Point {
        T               omega{};        //!< Frequency (rad/s)
        wet::complex<T> Z{};            //!< Complex impedance
        T               magnitude{};    //!< |Z| (ohms)
        T               magnitude_db{}; //!< |Z| in dB
        T               phase_deg{};    //!< Phase of Z (degrees)
    };
    std::vector<Point> points;
};

/**
 * @brief Result of Middlebrook minor loop gain analysis
 *
 * The minor loop gain is T_m(s) = Z_s(s) / Z_L(s).
 * The interconnected system is stable if T_m satisfies the Nyquist criterion,
 * which for a stable T_m reduces to:
 *   - Gain margin: |T_m| should be well below 0 dB at -180° phase crossing
 *   - Phase margin: phase(T_m) should be well above -180° at 0 dB crossing
 *
 * A sufficient (conservative) condition: |T_m(jω)| < 1 for all ω,
 * i.e. |Z_s| < |Z_L| at every frequency.
 */
template<typename T = double>
struct MiddlebrookResult {
    BodeResult<T> minor_loop_gain; //!< Bode data for T_m = Z_s / Z_L

    ImpedanceResult<T> source_impedance; //!< Z_s frequency response
    ImpedanceResult<T> load_impedance;   //!< Z_L frequency response

    /**
     * @brief Check if the sufficient stability condition holds
     *
     * Returns true if |Z_s(jω)| < |Z_L(jω)| at every frequency point,
     * i.e. the minor loop gain magnitude is strictly below 0 dB everywhere.
     * This is a conservative criterion (sufficient but not necessary).
     */
    [[nodiscard]] constexpr bool is_stable_sufficient() const {
        for (const auto& pt : minor_loop_gain.points) {
            if (pt.magnitude_db >= T{0}) {
                return false;
            }
        }
        return !minor_loop_gain.points.empty();
    }

    /**
     * @brief Gain margin of the minor loop gain
     *
     * Gain margin at the -180° phase crossing of T_m.
     * Positive dB = stable.
     *
     * @return {gain_margin_dB, frequency} or nullopt
     */
    [[nodiscard]] constexpr wet::optional<wet::pair<T, T>> gain_margin() const {
        return minor_loop_gain.gain_margin();
    }

    /**
     * @brief Phase margin of the minor loop gain
     *
     * Phase margin at the 0 dB crossing of T_m.
     * Positive degrees = stable.
     *
     * @return {phase_margin_deg, frequency} or nullopt
     */
    [[nodiscard]] constexpr wet::optional<wet::pair<T, T>> phase_margin() const {
        return minor_loop_gain.phase_margin();
    }

    /**
     * @brief Find the worst-case (smallest) impedance ratio across all frequencies
     *
     * Returns the minimum |Z_L|/|Z_s| ratio, i.e. how close the system
     * is to violating the sufficient condition. Values > 1 mean the
     * sufficient condition holds at all frequencies.
     *
     * @return {min_ratio, frequency_of_worst_case}
     */
    [[nodiscard]] constexpr wet::pair<T, T> worst_case_margin() const {
        T min_ratio = std::numeric_limits<T>::infinity();
        T worst_freq = T{0};
        for (size_t i = 0; i < minor_loop_gain.points.size(); ++i) {
            T ratio = minor_loop_gain.points[i].magnitude;
            if (ratio > T{0}) {
                T inv_ratio = T{1} / ratio; // |Z_L|/|Z_s|
                if (inv_ratio < min_ratio) {
                    min_ratio = inv_ratio;
                    worst_freq = minor_loop_gain.points[i].omega;
                }
            }
        }
        return {min_ratio, worst_freq};
    }
};

/**
 * @brief Compute impedance frequency response from a SISO admittance system
 *
 * Given a system G(s) = I(s)/V(s) (admittance: current out per voltage in),
 * computes Z(s) = 1/G(s) = V(s)/I(s) at each frequency.
 *
 * @param admittance_sys  SISO state-space system where output = current, input = voltage
 * @param omega           Vector of frequencies (rad/s)
 * @return ImpedanceResult with Z(jω) at each frequency
 */
template<size_t NX, size_t NW, size_t NV, typename T>
[[nodiscard]] constexpr ImpedanceResult<T> impedance(
    const StateSpace<NX, 1, 1, NW, NV, T>& admittance_sys,
    const std::vector<T>&                  omega
) {
    using C = wet::complex<T>;
    ImpedanceResult<T> result;
    result.points.reserve(omega.size());

    for (const auto& w : omega) {
        C    jw{T{0}, w};
        auto G = eval_frf(admittance_sys, jw);
        C    Y = G(0, 0);           // Admittance Y(jω) = I/V
        C    Z = C{T{1}, T{0}} / Y; // Impedance Z = 1/Y

        T mag = wet::abs(Z);
        T phase_rad = wet::arg(Z);
        T mag_db = mag > T{0} ? T{20} * wet::log10(mag) : T{-300};
        T phase_deg = phase_rad * T{180} / wet::numbers::pi_v<T>;

        result.points.push_back({w, Z, mag, mag_db, phase_deg});
    }
    return result;
}

/**
 * @brief Compute impedance frequency response from a SISO impedance transfer function
 *
 * Given Z(s) directly as a state-space system (voltage out per current in),
 * evaluates the frequency response.
 *
 * @param impedance_sys  SISO state-space system where output = voltage, input = current
 * @param omega          Vector of frequencies (rad/s)
 * @return ImpedanceResult with Z(jω) at each frequency
 */
template<size_t NX, size_t NW, size_t NV, typename T>
[[nodiscard]] constexpr ImpedanceResult<T> impedance_direct(
    const StateSpace<NX, 1, 1, NW, NV, T>& impedance_sys,
    const std::vector<T>&                  omega
) {
    using C = wet::complex<T>;
    ImpedanceResult<T> result;
    result.points.reserve(omega.size());

    for (const auto& w : omega) {
        C    jw{T{0}, w};
        auto G = eval_frf(impedance_sys, jw);
        C    Z = G(0, 0);

        T mag = wet::abs(Z);
        T phase_rad = wet::arg(Z);
        T mag_db = mag > T{0} ? T{20} * wet::log10(mag) : T{-300};
        T phase_deg = phase_rad * T{180} / wet::numbers::pi_v<T>;

        result.points.push_back({w, Z, mag, mag_db, phase_deg});
    }
    return result;
}

/**
 * @brief Middlebrook stability analysis for cascaded source-load systems
 *
 * Evaluates the minor loop gain T_m(jω) = Z_s(jω) / Z_L(jω) over a
 * range of frequencies and returns stability margins.
 *
 * The source and load are specified as admittance systems (current/voltage):
 *   - source_admittance: G_s(s) = I_s(s)/V_s(s), so Z_s = 1/G_s
 *   - load_admittance:   G_L(s) = I_L(s)/V_L(s), so Z_L = 1/G_L
 *
 * @param source_admittance  Source admittance SISO system
 * @param load_admittance    Load admittance SISO system
 * @param omega              Vector of frequencies (rad/s)
 * @return MiddlebrookResult with minor loop gain Bode data and margins
 */
template<size_t NX_S, size_t NW_S, size_t NV_S, size_t NX_L, size_t NW_L, size_t NV_L, typename T>
[[nodiscard]] constexpr MiddlebrookResult<T> middlebrook(
    const StateSpace<NX_S, 1, 1, NW_S, NV_S, T>& source_admittance,
    const StateSpace<NX_L, 1, 1, NW_L, NV_L, T>& load_admittance,
    const std::vector<T>&                        omega
) {
    using C = wet::complex<T>;
    MiddlebrookResult<T> result;
    result.minor_loop_gain.points.reserve(omega.size());
    result.source_impedance.points.reserve(omega.size());
    result.load_impedance.points.reserve(omega.size());

    for (const auto& w : omega) {
        C jw{T{0}, w};

        // Source impedance: Z_s = 1/Y_s
        auto G_s = eval_frf(source_admittance, jw);
        C    Y_s = G_s(0, 0);
        C    Z_s = C{T{1}, T{0}} / Y_s;

        // Load impedance: Z_L = 1/Y_L
        auto G_L = eval_frf(load_admittance, jw);
        C    Y_L = G_L(0, 0);
        C    Z_L = C{T{1}, T{0}} / Y_L;

        // Minor loop gain: T_m = Z_s / Z_L = Y_L / Y_s
        C Tm = Z_s / Z_L;

        // Source impedance point
        T zs_mag = wet::abs(Z_s);
        T zs_phase = wet::arg(Z_s) * T{180} / wet::numbers::pi_v<T>;
        T zs_db = zs_mag > T{0} ? T{20} * wet::log10(zs_mag) : T{-300};
        result.source_impedance.points.push_back({w, Z_s, zs_mag, zs_db, zs_phase});

        // Load impedance point
        T zl_mag = wet::abs(Z_L);
        T zl_phase = wet::arg(Z_L) * T{180} / wet::numbers::pi_v<T>;
        T zl_db = zl_mag > T{0} ? T{20} * wet::log10(zl_mag) : T{-300};
        result.load_impedance.points.push_back({w, Z_L, zl_mag, zl_db, zl_phase});

        // Minor loop gain point
        T tm_mag = wet::abs(Tm);
        T tm_phase = wet::arg(Tm) * T{180} / wet::numbers::pi_v<T>;
        T tm_db = tm_mag > T{0} ? T{20} * wet::log10(tm_mag) : T{-300};
        result.minor_loop_gain.points.push_back({w, tm_mag, tm_db, tm_phase});
    }

    return result;
}

/**
 * @brief Middlebrook analysis from pre-computed impedance data
 *
 * For cases where Z_s and Z_L are already known (e.g., from measurement
 * or from separate impedance models).
 *
 * @param Z_source  Source impedance frequency response
 * @param Z_load    Load impedance frequency response
 * @return MiddlebrookResult with minor loop gain and margins
 */
template<typename T>
[[nodiscard]] constexpr MiddlebrookResult<T> middlebrook(
    const ImpedanceResult<T>& Z_source,
    const ImpedanceResult<T>& Z_load
) {
    MiddlebrookResult<T> result;
    result.source_impedance = Z_source;
    result.load_impedance = Z_load;

    size_t n = wet::min(Z_source.points.size(), Z_load.points.size());
    result.minor_loop_gain.points.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        auto& zs = Z_source.points[i];
        auto& zl = Z_load.points[i];

        // T_m = Z_s / Z_L
        auto Tm = zs.Z / zl.Z;
        T    tm_mag = wet::abs(Tm);
        T    tm_phase = wet::arg(Tm) * T{180} / wet::numbers::pi_v<T>;
        T    tm_db = tm_mag > T{0} ? T{20} * wet::log10(tm_mag) : T{-300};

        result.minor_loop_gain.points.push_back({zs.omega, tm_mag, tm_db, tm_phase});
    }

    return result;
}

} // namespace analysis
} // namespace wet
