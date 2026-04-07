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
#include <numbers>
#include <optional>
#include <vector>

#include "constexpr_complex.hpp"
#include "constexpr_math.hpp"
#include "eigen.hpp"
#include "matrix.hpp"
#include "state_space.hpp"
#include "utility.hpp"

namespace wetmelon::control {
namespace analysis {

// ============================================================================
// Structural Analysis (Controllability / Observability)
// ============================================================================

/**
 * @brief Compute the controllability matrix [B, AB, A²B, ..., A^(N-1)B]
 *
 * The system (A, B) is controllable iff ctrb(A, B) has full row rank.
 *
 * @tparam NX Number of states
 * @tparam NU Number of inputs
 * @tparam T  Scalar type
 * @param A   State matrix
 * @param B   Input matrix
 * @return Matrix<NX, NX*NU, T> controllability matrix
 */
template<size_t NX, size_t NU, typename T = double>
[[nodiscard]] constexpr Matrix<NX, NX * NU, T>
ctrb(const Matrix<NX, NX, T>& A, const Matrix<NX, NU, T>& B) noexcept {
    Matrix<NX, NX * NU, T> Co{};
    Matrix<NX, NU, T>      AB = B;
    for (size_t k = 0; k < NX; ++k) {
        for (size_t r = 0; r < NX; ++r) {
            for (size_t c = 0; c < NU; ++c) {
                Co(r, k * NU + c) = AB(r, c);
            }
        }
        if (k + 1 < NX) {
            AB = A * AB;
        }
    }
    return Co;
}

/**
 * @brief Compute the observability matrix [C; CA; CA²; ...; CA^(N-1)]
 *
 * The system (A, C) is observable iff obsv(A, C) has full column rank.
 *
 * @tparam NX Number of states
 * @tparam NY Number of outputs
 * @tparam T  Scalar type
 * @param A   State matrix
 * @param C   Output matrix
 * @return Matrix<NX*NY, NX, T> observability matrix
 */
template<size_t NX, size_t NY, typename T = double>
[[nodiscard]] constexpr Matrix<NX * NY, NX, T>
obsv(const Matrix<NX, NX, T>& A, const Matrix<NY, NX, T>& C) noexcept {
    Matrix<NX * NY, NX, T> Ob{};
    Matrix<NY, NX, T>      CA = C;
    for (size_t k = 0; k < NX; ++k) {
        for (size_t r = 0; r < NY; ++r) {
            for (size_t c = 0; c < NX; ++c) {
                Ob(k * NY + r, c) = CA(r, c);
            }
        }
        if (k + 1 < NX) {
            CA = CA * A;
        }
    }
    return Ob;
}

/**
 * @brief Compute rank of a matrix via Gaussian elimination with partial pivoting
 *
 * @param M  Input matrix
 * @param tol Tolerance for zero detection (default: 1e-10)
 * @return size_t rank of the matrix
 */
template<size_t R, size_t C, typename T>
[[nodiscard]] constexpr size_t rank(const Matrix<R, C, T>& M, T tol = T{1e-10}) noexcept {
    // Work on a copy
    Matrix<R, C, T> work = M;
    size_t          r = 0;
    for (size_t col = 0; col < C && r < R; ++col) {
        // Find pivot
        size_t pivot = r;
        T      max_val = wet::abs(work(r, col));
        for (size_t i = r + 1; i < R; ++i) {
            T val = wet::abs(work(i, col));
            if (val > max_val) {
                max_val = val;
                pivot = i;
            }
        }
        if (max_val < tol) {
            continue;
        }
        // Swap rows
        if (pivot != r) {
            for (size_t j = 0; j < C; ++j) {
                T tmp = work(r, j);
                work(r, j) = work(pivot, j);
                work(pivot, j) = tmp;
            }
        }
        // Eliminate below
        for (size_t i = r + 1; i < R; ++i) {
            T factor = work(i, col) / work(r, col);
            for (size_t j = col; j < C; ++j) {
                work(i, j) -= factor * work(r, j);
            }
        }
        ++r;
    }
    return r;
}

/**
 * @brief Check if a system is controllable
 *
 * @param A State matrix
 * @param B Input matrix
 * @return true if the controllability matrix has full rank (NX)
 */
template<size_t NX, size_t NU, typename T = double>
[[nodiscard]] constexpr bool is_controllable(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    T                        tol = T{1e-10}
) noexcept {
    auto Co = ctrb(A, B);
    return rank(Co, tol) == NX;
}

/**
 * @brief Check if a system is observable
 *
 * @param A State matrix
 * @param C Output matrix
 * @return true if the observability matrix has full rank (NX)
 */
template<size_t NX, size_t NY, typename T = double>
[[nodiscard]] constexpr bool is_observable(
    const Matrix<NX, NX, T>& A,
    const Matrix<NY, NX, T>& C,
    T                        tol = T{1e-10}
) noexcept {
    auto Ob = obsv(A, C);
    return rank(Ob, tol) == NX;
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
    [[nodiscard]] constexpr std::optional<std::pair<T, T>> gain_margin() const {
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
                return std::pair{gm, omega_cross};
            }
        }
        return std::nullopt;
    }

    /**
     * @brief Find phase margin (degrees above -180° at 0dB gain crossing)
     *
     * Phase margin is 180° + phase(G(jω)) at the frequency where |G(jω)| = 1 (0 dB).
     * Positive values indicate stability.
     *
     * @return {phase_margin_deg, crossover_frequency} or nullopt if no 0dB crossing
     */
    [[nodiscard]] constexpr std::optional<std::pair<T, T>> phase_margin() const {
        for (size_t i = 1; i < points.size(); ++i) {
            T mag_prev = points[i - 1].magnitude_db;
            T mag_curr = points[i].magnitude_db;
            // Looking for magnitude crossing 0 dB (from above)
            if (mag_prev >= T{0} && mag_curr < T{0}) {
                T frac = (T{0} - mag_prev) / (mag_curr - mag_prev);
                T omega_cross = points[i - 1].omega + frac * (points[i].omega - points[i - 1].omega);
                T phase_cross = points[i - 1].phase_deg + frac * (points[i].phase_deg - points[i - 1].phase_deg);
                T pm = T{180} + phase_cross;
                return std::pair{pm, omega_cross};
            }
        }
        return std::nullopt;
    }

    /**
     * @brief Find -3dB bandwidth
     *
     * The frequency at which the magnitude drops 3 dB below the DC (or peak) value.
     *
     * @return Bandwidth in rad/s, or nullopt if not found
     */
    [[nodiscard]] constexpr std::optional<T> bandwidth() const {
        if (points.empty()) {
            return std::nullopt;
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
        return std::nullopt;
    }
};

/**
 * @brief Compute Bode plot data for a SISO state-space system
 *
 * Evaluates G(jω) = C(jωI - A)^{-1}B + D over a vector of frequencies.
 *
 * @param sys   Continuous-time SISO state-space system
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
        C    jw{T{0}, w};
        auto G_frf = eval_frf(sys, jw);
        C    G = G_frf(0, 0);
        T    mag = wet::abs(G);
        T    phase_rad = wet::arg(G);
        T    mag_db = mag > T{0} ? T{20} * wet::log10(mag) : T{-300};
        T    phase_deg = phase_rad * T{180} / std::numbers::pi_v<T>;
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
    const std::array<T, Nnum>& num,
    const std::array<T, Nden>& den,
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
        T phase_deg = phase_rad * T{180} / std::numbers::pi_v<T>;
        result.points.push_back({w, mag, mag_db, phase_deg});
    }
    return result;
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
[[nodiscard]] constexpr std::optional<Matrix<NY, NU, T>> dcgain(
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
    auto inv = M.inverse();
    if (!inv) {
        return std::nullopt;
    }
    return sys.C * (*inv) * sys.B + sys.D;
}

/**
 * @brief Compute open-loop poles (eigenvalues of A matrix)
 *
 * @param A State matrix
 * @return Vector of pole locations as complex numbers
 */
template<size_t NX, typename T = double>
[[nodiscard]] constexpr ColVec<NX, wet::complex<T>> poles(const Matrix<NX, NX, T>& A) {
    static_assert(NX <= 4, "Pole computation only supported for systems up to 4 states");
    auto eigen = compute_eigenvalues(A);
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
    static_assert(NX <= 4, "Stability analysis only supported for systems up to 4 states");
    auto eigen = compute_eigenvalues(A);
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
[[nodiscard]] constexpr std::array<PoleInfo<T>, NX>
damp(const Matrix<NX, NX, T>& A) {
    static_assert(NX <= 4, "Damp only supported for systems up to 4 states");
    auto                        eigen = compute_eigenvalues(A);
    std::array<PoleInfo<T>, NX> info{};
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
    [[nodiscard]] constexpr std::optional<std::pair<T, T>> gain_margin() const {
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
    [[nodiscard]] constexpr std::optional<std::pair<T, T>> phase_margin() const {
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
    [[nodiscard]] constexpr std::pair<T, T> worst_case_margin() const {
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
        T phase_deg = phase_rad * T{180} / std::numbers::pi_v<T>;

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
        T phase_deg = phase_rad * T{180} / std::numbers::pi_v<T>;

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
        T zs_phase = wet::arg(Z_s) * T{180} / std::numbers::pi_v<T>;
        T zs_db = zs_mag > T{0} ? T{20} * wet::log10(zs_mag) : T{-300};
        result.source_impedance.points.push_back({w, Z_s, zs_mag, zs_db, zs_phase});

        // Load impedance point
        T zl_mag = wet::abs(Z_L);
        T zl_phase = wet::arg(Z_L) * T{180} / std::numbers::pi_v<T>;
        T zl_db = zl_mag > T{0} ? T{20} * wet::log10(zl_mag) : T{-300};
        result.load_impedance.points.push_back({w, Z_L, zl_mag, zl_db, zl_phase});

        // Minor loop gain point
        T tm_mag = wet::abs(Tm);
        T tm_phase = wet::arg(Tm) * T{180} / std::numbers::pi_v<T>;
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

    size_t n = std::min(Z_source.points.size(), Z_load.points.size());
    result.minor_loop_gain.points.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        auto& zs = Z_source.points[i];
        auto& zl = Z_load.points[i];

        // T_m = Z_s / Z_L
        auto Tm = zs.Z / zl.Z;
        T    tm_mag = wet::abs(Tm);
        T    tm_phase = wet::arg(Tm) * T{180} / std::numbers::pi_v<T>;
        T    tm_db = tm_mag > T{0} ? T{20} * wet::log10(tm_mag) : T{-300};

        result.minor_loop_gain.points.push_back({zs.omega, tm_mag, tm_db, tm_phase});
    }

    return result;
}

} // namespace analysis
} // namespace wetmelon::control
