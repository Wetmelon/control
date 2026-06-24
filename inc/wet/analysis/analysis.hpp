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

#include "wet/backend.hpp"
#include "wet/math/complex.hpp"
#include "wet/math/math.hpp"
#include "wet/matrix/eigen.hpp"
#include "wet/matrix/matrix.hpp"
#include "wet/systems/discretization.hpp"
#include "wet/systems/state_space.hpp"
#include "wet/systems/transfer_function.hpp"

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
    using Cplx = wet::complex<T>;
    BodeResult<T> result;
    result.points.reserve(omega.size());

    for (const auto& w : omega) {
        const Cplx s_or_z = sys.is_discrete()
                              ? Cplx{wet::cos(w * sys.Ts), wet::sin(w * sys.Ts)}
                              : Cplx{T{0}, w};
        auto       G_frf = eval_frf(sys, s_or_z);
        Cplx       G = G_frf(0, 0);
        T          mag = wet::abs(G);
        T          phase_rad = wet::arg(G);
        T          mag_db = mag > T{0} ? T{20} * wet::log10(mag) : T{-300};
        T          phase_deg = phase_rad * T{180} / wet::numbers::pi_v<T>;
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
    using Cplx = wet::complex<T>;
    BodeResult<T> result;
    result.points.reserve(omega.size());

    for (const auto& w : omega) {
        Cplx jw{T{0}, w};

        // Evaluate numerator polynomial at jw (ascending powers)
        Cplx num_val{T{0}, T{0}};
        Cplx jw_power{T{1}, T{0}};
        for (size_t i = 0; i < Nnum; ++i) {
            num_val = num_val + num[i] * jw_power;
            jw_power = jw_power * jw;
        }

        // Evaluate denominator polynomial at jw (ascending powers)
        Cplx den_val{T{0}, T{0}};
        jw_power = Cplx{T{1}, T{0}};
        for (size_t i = 0; i < Nden; ++i) {
            den_val = den_val + den[i] * jw_power;
            jw_power = jw_power * jw;
        }

        Cplx G = num_val / den_val;
        T    mag = wet::abs(G);
        T    phase_rad = wet::arg(G);
        T    mag_db = mag > T{0} ? T{20} * wet::log10(mag) : T{-300};
        T    phase_deg = phase_rad * T{180} / wet::numbers::pi_v<T>;
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
    wet::optional<wet::pair<T, T>> phase_margin;                                        //!< {PM [deg], gain crossover omega [rad/s]}
    wet::optional<wet::pair<T, T>> gain_margin;                                         //!< {GM [dB], phase crossover omega [rad/s]}
    wet::optional<T>               bandwidth;                                           //!< Closed-loop bandwidth from T=L/(1+L), rad/s
    wet::optional<wet::pair<T, T>> min_nyquist_distance;                                //!< {min|1+L|, omega [rad/s]}
    T                              peak_sensitivity_db{-std::numeric_limits<T>::max()}; //!< max 20*log10|S|
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
    using Cplx = wet::complex<T>;
    NyquistResult<T> result;
    result.points.reserve(omega.size());

    for (const auto& w : omega) {
        const Cplx      s_or_z = sys.is_discrete()
                                   ? Cplx{wet::cos(w * sys.Ts), wet::sin(w * sys.Ts)}
                                   : Cplx{T{0}, w};
        const auto      G_frf = eval_frf(sys, s_or_z);
        const Cplx      G = G_frf(0, 0);
        const T         dist = wet::abs(Cplx{T{1}, T{0}} + G);
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
    using Cplx = wet::complex<T>;

    LoopResponseResult<T> result;
    result.open_loop.points.reserve(omega.size());
    result.sensitivity.points.reserve(omega.size());
    result.complementary_sensitivity.points.reserve(omega.size());
    result.nyquist.points.reserve(omega.size());

    for (const auto& w : omega) {
        const Cplx s_or_z = loop.is_discrete()
                              ? Cplx{wet::cos(w * loop.Ts), wet::sin(w * loop.Ts)}
                              : Cplx{T{0}, w};

        const auto L_frf = eval_frf(loop, s_or_z);
        const Cplx L = L_frf(0, 0);
        const Cplx one{T{1}, T{0}};
        const Cplx S = one / (one + L);
        const Cplx Tresp = L / (one + L);

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
        T    tau = (p.real() < T{0}) ? T{-1} / p.real() : std::numeric_limits<T>::max();
        info[i] = PoleInfo<T>{p, wn, zeta, tau};
    }
    return info;
}

// ============================================================================
// Time-Domain Responses
// ============================================================================

/**
 * @brief Multi-channel time-domain response sampled on a time grid
 *
 * For a system with NU inputs and NY outputs, `y[k](i, j)` is output i at time
 * step k in response to a canonical input (unit step / unit impulse) applied to
 * input channel j alone. This mirrors MATLAB's (Nt × Ny × Nu) response array.
 *
 * @tparam NY Number of outputs
 * @tparam NU Number of inputs
 */
template<size_t NY, size_t NU, typename T = double>
struct TimeResponse {
    std::vector<T>                 t; //!< Time points (s)
    std::vector<Matrix<NY, NU, T>> y; //!< y[k](i,j): output i from a canonical input on channel j
};

/**
 * @brief Result of a single-trajectory simulation: time, output, and state history
 *
 * Used by `lsim` (forced response to a given input) and `initial` (free
 * response). `y[k]` is the NY-vector output and `x[k]` the full state at step k.
 *
 * @tparam NX Number of states
 * @tparam NY Number of outputs
 */
template<size_t NX, size_t NY, typename T = double>
struct LsimResult {
    std::vector<T>             t; //!< Time points (s)
    std::vector<ColVec<NY, T>> y; //!< Output vector at each time point
    std::vector<ColVec<NX, T>> x; //!< State vector at each time point
};

namespace detail {

//! ZOH-discretize a continuous system onto a uniform time grid (pass-through if already discrete).
template<size_t NX, size_t NU, size_t NY, size_t NW, size_t NV, typename T>
[[nodiscard]] StateSpace<NX, NU, NY, NW, NV, T> discretize_on_grid(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const std::vector<T>&                    time
) {
    if (sys.is_discrete()) {
        return sys;
    }
    const T dt = (time.size() > 1) ? time[1] - time[0] : T{1};
    return wet::discretize(sys, dt, DiscretizationMethod::ZOH);
}

//! Iterate a discrete system from x0 with constant input u, collecting y[k]=Cx+Du for n steps.
template<size_t NX, size_t NU, size_t NY, size_t NW, size_t NV, typename T>
[[nodiscard]] std::vector<ColVec<NY, T>> const_input_response(
    const StateSpace<NX, NU, NY, NW, NV, T>& dsys,
    const ColVec<NX, T>&                     x0,
    const ColVec<NU, T>&                     u,
    size_t                                   n
) {
    std::vector<ColVec<NY, T>> y;
    y.reserve(n);
    ColVec<NX, T> x = x0;
    for (size_t k = 0; k < n; ++k) {
        y.push_back(ColVec<NY, T>(dsys.C * x + dsys.D * u));
        x = ColVec<NX, T>(dsys.A * x + dsys.B * u);
    }
    return y;
}

//! Scatter a single-channel response (column j) into a multi-channel TimeResponse.
template<size_t NY, size_t NU, typename T>
void set_column(TimeResponse<NY, NU, T>& r, size_t j, const std::vector<ColVec<NY, T>>& yj) {
    for (size_t k = 0; k < yj.size(); ++k) {
        for (size_t i = 0; i < NY; ++i) {
            r.y[k](i, j) = yj[k][i];
        }
    }
}

} // namespace detail

/**
 * @brief Step response of a (MIMO) state-space system
 *
 * Applies a unit step to each input channel in turn (others held at zero) from
 * zero initial state and records every output, so `y[k](i, j)` is output i at
 * step k due to a step on input j. Continuous systems are ZOH-discretized on the
 * (uniform) time grid; discrete systems are iterated directly. MATLAB
 * equivalent: `step(sys, t)`.
 *
 * @param sys  State-space system (continuous or discrete)
 * @param time Uniformly spaced time vector (e.g. analysis::linspace(0, tf, n))
 * @return TimeResponse with per-channel output history
 */
template<size_t NX, size_t NU, size_t NY, size_t NW, size_t NV, typename T>
[[nodiscard]] TimeResponse<NY, NU, T> step(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const std::vector<T>&                    time
) {
    const auto              dsys = detail::discretize_on_grid(sys, time);
    TimeResponse<NY, NU, T> r;
    r.t = time;
    r.y.assign(time.size(), Matrix<NY, NU, T>{});
    for (size_t j = 0; j < NU; ++j) {
        ColVec<NU, T> uj{};
        uj[j] = T{1};
        detail::set_column(r, j, detail::const_input_response(dsys, ColVec<NX, T>{}, uj, time.size()));
    }
    return r;
}

/**
 * @brief Impulse response of a (MIMO) state-space system
 *
 * For each input channel j, computes the free response from initial state
 * x₀ = B(:,j), giving y(t) = C·e^{At}·B(:,j) (the strictly-proper part; the
 * D·δ(t) feedthrough term is not plotted). MATLAB equivalent: `impulse(sys, t)`.
 *
 * @param sys  State-space system
 * @param time Uniformly spaced time vector
 * @return TimeResponse with per-channel output history
 */
template<size_t NX, size_t NU, size_t NY, size_t NW, size_t NV, typename T>
[[nodiscard]] TimeResponse<NY, NU, T> impulse(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const std::vector<T>&                    time
) {
    const auto              dsys = detail::discretize_on_grid(sys, time);
    TimeResponse<NY, NU, T> r;
    r.t = time;
    r.y.assign(time.size(), Matrix<NY, NU, T>{});
    for (size_t j = 0; j < NU; ++j) {
        ColVec<NX, T> x0j{};
        for (size_t i = 0; i < NX; ++i) {
            x0j[i] = sys.B(i, j);
        }
        detail::set_column(r, j, detail::const_input_response(dsys, x0j, ColVec<NU, T>{}, time.size()));
    }
    return r;
}

/**
 * @brief Initial-condition (free) response of a (MIMO) state-space system
 *
 * Computes y(t) = C·e^{At}·x₀ for the unforced system (u=0). MATLAB
 * equivalent: `initial(sys, x0, t)`.
 *
 * @param sys  State-space system
 * @param x0   Initial state
 * @param time Uniformly spaced time vector
 * @return LsimResult with time, output, and state history
 */
template<size_t NX, size_t NU, size_t NY, size_t NW, size_t NV, typename T>
[[nodiscard]] LsimResult<NX, NY, T> initial(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const ColVec<NX, T>&                     x0,
    const std::vector<T>&                    time
) {
    const auto            dsys = detail::discretize_on_grid(sys, time);
    LsimResult<NX, NY, T> r;
    r.t = time;
    r.y.reserve(time.size());
    r.x.reserve(time.size());
    ColVec<NX, T> x = x0;
    for (size_t k = 0; k < time.size(); ++k) {
        r.x.push_back(x);
        r.y.push_back(ColVec<NY, T>(dsys.C * x));
        x = ColVec<NX, T>(dsys.A * x);
    }
    return r;
}

/**
 * @brief Forced time response of a (MIMO) state-space system to an input signal
 *
 * Simulates ẋ = Ax + Bu, y = Cx + Du driven by the supplied input samples,
 * from initial state x₀. Continuous systems are ZOH-discretized on the (uniform)
 * time grid — exact at the sample points for piecewise-constant input — and
 * discrete systems are iterated directly. MATLAB equivalent: `lsim(sys, u, t, x0)`.
 *
 * @param sys  State-space system (continuous or discrete)
 * @param u    Input samples, one ColVec<NU> per time point (length == time.size())
 * @param time Uniformly spaced time vector
 * @param x0   Initial state (defaults to zero)
 * @return LsimResult with time, output, and state history
 */
template<size_t NX, size_t NU, size_t NY, size_t NW, size_t NV, typename T>
[[nodiscard]] LsimResult<NX, NY, T> lsim(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const std::vector<ColVec<NU, T>>&        u,
    const std::vector<T>&                    time,
    const ColVec<NX, T>&                     x0 = {}
) {
    const T    dt = (time.size() > 1) ? time[1] - time[0] : T{1};
    const auto dsys = sys.is_discrete() ? sys : wet::discretize(sys, dt, DiscretizationMethod::ZOH);

    LsimResult<NX, NY, T> r;
    r.t = time;
    r.y.reserve(time.size());
    r.x.reserve(time.size());

    ColVec<NX, T> x = x0;
    for (size_t k = 0; k < time.size(); ++k) {
        r.x.push_back(x);
        r.y.push_back(ColVec<NY, T>(dsys.C * x + dsys.D * u[k]));
        x = ColVec<NX, T>(dsys.A * x + dsys.B * u[k]);
    }
    return r;
}

/**
 * @brief Single-input convenience overload of lsim taking a scalar input signal
 *
 * MATLAB allows `lsim(sys, u, t)` with a plain vector u for single-input systems.
 */
template<size_t NX, size_t NY, size_t NW, size_t NV, typename T>
[[nodiscard]] LsimResult<NX, NY, T> lsim(
    const StateSpace<NX, 1, NY, NW, NV, T>& sys,
    const std::vector<T>&                   u,
    const std::vector<T>&                   time,
    const ColVec<NX, T>&                    x0 = {}
) {
    std::vector<ColVec<1, T>> uv;
    uv.reserve(u.size());
    for (const T& ui : u) {
        uv.push_back(ColVec<1, T>{ui});
    }
    return lsim(sys, uv, time, x0);
}

/**
 * @brief Step response of a SISO transfer function (MATLAB `step(tf, t)`)
 */
template<size_t Nnum, size_t Nden, typename T>
[[nodiscard]] TimeResponse<1, 1, T> step(
    const TransferFunction<Nnum, Nden, T>& tf,
    const std::vector<T>&                  time
) {
    return step(tf.to_state_space(), time);
}

/**
 * @brief Impulse response of a SISO transfer function (MATLAB `impulse(tf, t)`)
 */
template<size_t Nnum, size_t Nden, typename T>
[[nodiscard]] TimeResponse<1, 1, T> impulse(
    const TransferFunction<Nnum, Nden, T>& tf,
    const std::vector<T>&                  time
) {
    return impulse(tf.to_state_space(), time);
}

// ============================================================================
// Response Characteristics
// ============================================================================

/**
 * @brief Step-response characteristics of a single output signal
 *
 * Sample-resolution metrics (no sub-sample interpolation) measured against the
 * initial value y(0) and the supplied steady-state value. MATLAB equivalent:
 * `stepinfo`.
 */
template<typename T = double>
struct StepInfo {
    T rise_time{};     //!< Time to go from 10% to 90% of the total change
    T settling_time{}; //!< Last time the response stays within the settling band of yfinal
    T settling_min{};  //!< Minimum value once the response first enters the settling band
    T settling_max{};  //!< Maximum value once the response first enters the settling band
    T overshoot{};     //!< Percent overshoot beyond yfinal (0 if none)
    T undershoot{};    //!< Percent undershoot below the initial value (0 if none)
    T peak{};          //!< Peak absolute value of the response
    T peak_time{};     //!< Time of the peak absolute value
};

/**
 * @brief Compute step-response characteristics from an output/time signal
 *
 * @param y           Output samples
 * @param t           Matching time samples (same length as y)
 * @param yfinal      Steady-state value the response settles to
 * @param settle_frac Settling band as a fraction of |yfinal - y0| (default 0.02 → ±2%)
 */
template<typename T = double>
[[nodiscard]] StepInfo<T> stepinfo(
    const std::vector<T>& y,
    const std::vector<T>& t,
    T                     yfinal,
    T                     settle_frac = T(0.02)
) {
    StepInfo<T> info;
    if (y.empty()) {
        return info;
    }
    const T y0 = y.front();
    const T span = yfinal - y0;
    const T aspan = wet::abs(span);

    // Rise time: first crossing of 10% then 90% of the total change.
    const T lo = y0 + T(0.1) * span;
    const T hi = y0 + T(0.9) * span;
    T       t_lo = t.front();
    T       t_hi = t.front();
    bool    got_lo = false;
    bool    got_hi = false;
    for (size_t k = 0; k < y.size(); ++k) {
        const T reach_lo = (span >= T(0)) ? (y[k] >= lo) : (y[k] <= lo);
        const T reach_hi = (span >= T(0)) ? (y[k] >= hi) : (y[k] <= hi);
        if (!got_lo && reach_lo) {
            t_lo = t[k];
            got_lo = true;
        }
        if (!got_hi && reach_hi) {
            t_hi = t[k];
            got_hi = true;
            break;
        }
    }
    info.rise_time = t_hi - t_lo;

    // Settling time: last instant the response is outside the ±settle_frac band.
    const T band = settle_frac * aspan;
    info.settling_time = t.front();
    for (size_t k = 0; k < y.size(); ++k) {
        if (wet::abs(y[k] - yfinal) > band) {
            info.settling_time = (k + 1 < t.size()) ? t[k + 1] : t[k];
        }
    }
    // settling_min/max: extremes after first entering the band.
    info.settling_min = yfinal;
    info.settling_max = yfinal;
    bool entered = false;
    for (size_t k = 0; k < y.size(); ++k) {
        if (!entered && wet::abs(y[k] - yfinal) <= band) {
            entered = true;
            info.settling_min = y[k];
            info.settling_max = y[k];
        }
        if (entered) {
            info.settling_min = wet::min(info.settling_min, y[k]);
            info.settling_max = wet::max(info.settling_max, y[k]);
        }
    }

    // Peak (absolute), overshoot, undershoot.
    info.peak = wet::abs(y.front());
    info.peak_time = t.front();
    T ymax = y.front();
    T ymin = y.front();
    for (size_t k = 0; k < y.size(); ++k) {
        if (wet::abs(y[k]) > info.peak) {
            info.peak = wet::abs(y[k]);
            info.peak_time = t[k];
        }
        ymax = wet::max(ymax, y[k]);
        ymin = wet::min(ymin, y[k]);
    }
    if (aspan > T(0)) {
        const T over = (span >= T(0)) ? (ymax - yfinal) : (yfinal - ymin);
        const T under = (span >= T(0)) ? (y0 - ymin) : (ymax - y0);
        info.overshoot = wet::max(T(0), over / aspan * T(100));
        info.undershoot = wet::max(T(0), under / aspan * T(100));
    }
    return info;
}

/**
 * @brief Step-response characteristics of a SISO system (MATLAB `stepinfo(sys)`)
 *
 * Runs `step(sys, time)` and summarizes the single output. The steady-state
 * value is taken as the last sample. For MIMO systems, extract the desired
 * `y[k](i, j)` channel and call the signal overload.
 */
template<size_t NX, size_t NW, size_t NV, typename T>
[[nodiscard]] StepInfo<T> stepinfo(
    const StateSpace<NX, 1, 1, NW, NV, T>& sys,
    const std::vector<T>&                  time
) {
    const auto     resp = step(sys, time);
    std::vector<T> y;
    y.reserve(resp.y.size());
    for (const auto& yk : resp.y) {
        y.push_back(yk(0, 0));
    }
    return stepinfo(y, resp.t, y.empty() ? T(0) : y.back());
}

/**
 * @brief Transient characteristics of an arbitrary response signal
 *
 * Like `stepinfo` but makes no step assumption: reports settling relative to a
 * given final value plus the signal's extremes. MATLAB equivalent: `lsiminfo`.
 */
template<typename T = double>
struct LsimInfo {
    T settling_time{}; //!< Last time the response leaves the settling band of yfinal
    T settling_min{};  //!< Minimum after first entering the band
    T settling_max{};  //!< Maximum after first entering the band
    T min{};           //!< Global minimum of the signal
    T max{};           //!< Global maximum of the signal
    T min_time{};      //!< Time of the global minimum
    T max_time{};      //!< Time of the global maximum
};

/**
 * @brief Compute transient characteristics from an output/time signal
 *
 * @param y           Output samples
 * @param t           Matching time samples
 * @param yfinal      Reference value for the settling band
 * @param settle_frac Settling band as a fraction of |yfinal| (default 0.02)
 */
template<typename T = double>
[[nodiscard]] LsimInfo<T> lsiminfo(
    const std::vector<T>& y,
    const std::vector<T>& t,
    T                     yfinal,
    T                     settle_frac = T(0.02)
) {
    LsimInfo<T> info;
    if (y.empty()) {
        return info;
    }
    const T band = settle_frac * wet::abs(yfinal);

    info.min = y.front();
    info.max = y.front();
    info.min_time = t.front();
    info.max_time = t.front();
    info.settling_time = t.front();
    for (size_t k = 0; k < y.size(); ++k) {
        if (y[k] < info.min) {
            info.min = y[k];
            info.min_time = t[k];
        }
        if (y[k] > info.max) {
            info.max = y[k];
            info.max_time = t[k];
        }
        if (wet::abs(y[k] - yfinal) > band) {
            info.settling_time = (k + 1 < t.size()) ? t[k + 1] : t[k];
        }
    }

    info.settling_min = yfinal;
    info.settling_max = yfinal;
    bool entered = false;
    for (size_t k = 0; k < y.size(); ++k) {
        if (!entered && wet::abs(y[k] - yfinal) <= band) {
            entered = true;
            info.settling_min = y[k];
            info.settling_max = y[k];
        }
        if (entered) {
            info.settling_min = wet::min(info.settling_min, y[k]);
            info.settling_max = wet::max(info.settling_max, y[k]);
        }
    }
    return info;
}

// ============================================================================
// Pole-Zero Maps
// ============================================================================

/**
 * @brief Poles and zeros of a system, for pole-zero plotting
 *
 * Stored as runtime vectors of complex values (a pole-zero map feeds a scatter
 * plot, and the zero count is data-dependent). MATLAB equivalent: the data
 * returned by `pzmap`.
 */
template<typename T = double>
struct PoleZeroMap {
    std::vector<wet::complex<T>> poles; //!< Pole locations
    std::vector<wet::complex<T>> zeros; //!< Zero locations
};

/**
 * @brief Roots of a polynomial given in ascending powers (MATLAB `roots`, reversed order)
 *
 * For coefficients c[0] + c[1]·x + … + c[N-1]·x^{N-1}, returns the N-1 roots as
 * the eigenvalues of the companion matrix. The highest-order coefficient
 * c[N-1] must be nonzero (no trailing-zero padding).
 */
template<size_t N, typename T>
[[nodiscard]] std::vector<wet::complex<T>> poly_roots(const wet::array<T, N>& c) {
    std::vector<wet::complex<T>> out;
    if constexpr (N >= 2) {
        constexpr size_t M = N - 1;
        Matrix<M, M, T>  companion{};
        for (size_t i = 0; i + 1 < M; ++i) {
            companion(i, i + 1) = T{1};
        }
        for (size_t j = 0; j < M; ++j) {
            companion(M - 1, j) = -c[j] / c[N - 1];
        }
        const auto eig = mat::compute_eigenvalues(companion);
        out.reserve(M);
        for (size_t i = 0; i < M; ++i) {
            out.push_back(eig.values[i]);
        }
    }
    return out;
}

/**
 * @brief Pole-zero map of a SISO transfer function (MATLAB `pzmap(tf)`)
 *
 * Poles are the roots of the denominator, zeros the roots of the numerator.
 */
template<size_t Nnum, size_t Nden, typename T>
[[nodiscard]] PoleZeroMap<T> pzmap(const TransferFunction<Nnum, Nden, T>& tf) {
    return {poly_roots(tf.den), poly_roots(tf.num)};
}

/**
 * @brief Pole map of a state matrix (MATLAB `pzmap(sys)`, poles only)
 *
 * Returns the eigenvalues of A as poles. Transmission zeros are not computed
 * (they require a generalized/QZ eigensolver, not yet available).
 */
template<size_t NX, typename T>
[[nodiscard]] PoleZeroMap<T> pzmap(const Matrix<NX, NX, T>& A) {
    PoleZeroMap<T> r;
    const auto     eig = mat::compute_eigenvalues(A);
    r.poles.reserve(NX);
    for (size_t i = 0; i < NX; ++i) {
        r.poles.push_back(eig.values[i]);
    }
    return r;
}

/**
 * @brief Pole map of a state-space system (MATLAB `pzmap(sys)`, poles only)
 */
template<size_t NX, size_t NU, size_t NY, size_t NW, size_t NV, typename T>
[[nodiscard]] PoleZeroMap<T> pzmap(const StateSpace<NX, NU, NY, NW, NV, T>& sys) {
    return pzmap(sys.A);
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
        T min_ratio = std::numeric_limits<T>::max();
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
    using Cplx = wet::complex<T>;
    ImpedanceResult<T> result;
    result.points.reserve(omega.size());

    for (const auto& w : omega) {
        Cplx jw{T{0}, w};
        auto G = eval_frf(admittance_sys, jw);
        Cplx Y = G(0, 0);              // Admittance Y(jω) = I/V
        Cplx Z = Cplx{T{1}, T{0}} / Y; // Impedance Z = 1/Y

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
    using Cplx = wet::complex<T>;
    ImpedanceResult<T> result;
    result.points.reserve(omega.size());

    for (const auto& w : omega) {
        Cplx jw{T{0}, w};
        auto G = eval_frf(impedance_sys, jw);
        Cplx Z = G(0, 0);

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
    using Cplx = wet::complex<T>;
    MiddlebrookResult<T> result;
    result.minor_loop_gain.points.reserve(omega.size());
    result.source_impedance.points.reserve(omega.size());
    result.load_impedance.points.reserve(omega.size());

    for (const auto& w : omega) {
        Cplx jw{T{0}, w};

        // Source impedance: Z_s = 1/Y_s
        auto G_s = eval_frf(source_admittance, jw);
        Cplx Y_s = G_s(0, 0);
        Cplx Z_s = Cplx{T{1}, T{0}} / Y_s;

        // Load impedance: Z_L = 1/Y_L
        auto G_L = eval_frf(load_admittance, jw);
        Cplx Y_L = G_L(0, 0);
        Cplx Z_L = Cplx{T{1}, T{0}} / Y_L;

        // Minor loop gain: T_m = Z_s / Z_L = Y_L / Y_s
        Cplx Tm = Z_s / Z_L;

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
