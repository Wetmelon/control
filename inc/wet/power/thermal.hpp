#pragma once

#include <cstddef>
#include <limits>

#include "wet/backend.hpp"
#include "wet/matrix/colvec.hpp"
#include "wet/matrix/matrix.hpp"
#include "wet/systems/state_space.hpp"
#include "wet/toolbox/lookup.hpp"

namespace wet {

/**
 * @brief A two-breakpoint derating curve: 1 below @p derate_start, 0 at @p cutoff.
 *
 * Convenience for the common single-segment ramp; the general case is any
 * @ref Lut1D of (temperature, factor) breakpoints with a falling factor. Clamp
 * extrapolation holds 1 below the first point and 0 above the last.
 *
 * @param derate_start Temperature at which derating begins [°C].
 * @param cutoff       Temperature at which the factor reaches 0 [°C] (> derate_start).
 * @return Lut1D mapping temperature [°C] to a derating factor in [0, 1].
 */
template<typename T = double>
[[nodiscard]] constexpr Lut1D<2, T> derate_window(T derate_start, T cutoff) {
    return Lut1D<2, T>{.xs = {derate_start, cutoff}, .ys = {T{1}, T{0}}};
}

namespace design {

/**
 * @brief Continuous state-space model of a Foster RC thermal network.
 * @ingroup foc_design
 *
 * A Foster network is the datasheet form of a transient thermal impedance: a sum
 * of decoupled first-order sections @f$ Z_{th}(s) = \sum_i R_i/(1 + s\tau_i) @f$.
 * Each section is a state @f$ \tau_i\dot x_i = -x_i + R_i u @f$ with the junction
 * temperature rise the sum of the sections, giving the diagonal realization
 * @f[
 *   A = \mathrm{diag}(-1/\tau_i), \quad B_i = R_i/\tau_i, \quad C = [1\ \cdots\ 1],
 *   \quad D = 0.
 * @f]
 * The input is power [W] at the junction; the output is junction-to-reference
 * temperature rise [K]. Continuous (Ts = 0) — discretize with @ref discretize
 * (ZOH is exact for the piecewise-constant power assumption) for a runtime step.
 *
 * @note Foster section states are not physical node temperatures (only their sum
 *       is meaningful). Use @ref cauer_thermal_ss when you need physical nodes.
 *
 * @param R   [K/W] section thermal resistances (datasheet).
 * @param tau [s]   section time constants (datasheet).
 * @return Continuous @ref StateSpace<N,1,1>.
 */
template<std::size_t N, typename T = double>
[[nodiscard]] constexpr StateSpace<N, 1, 1, 0, 0, T> foster_thermal_ss(const wet::array<T, N>& R, const wet::array<T, N>& tau) {
    StateSpace<N, 1, 1, 0, 0, T> sys{};
    for (std::size_t i = 0; i < N; ++i) {
        sys.A(i, i) = -T{1} / tau[i];
        sys.B(i, 0) = R[i] / tau[i];
        sys.C(0, i) = T{1};
    }
    return sys;
}

/**
 * @brief Continuous state-space model of a physical Cauer RC thermal ladder.
 * @ingroup foc_design
 *
 * The physical ladder: series thermal resistance @f$ R_i @f$ from node @f$ i @f$
 * to node @f$ i+1 @f$ (or to the reference for @f$ i = N-1 @f$) and shunt
 * capacitance @f$ C_i @f$ at each node. The states are physical node temperatures,
 * node 0 the junction (heat injected, and the output), the reference a measured
 * case temperature:
 * @f[
 *   C_i\dot\vartheta_i = \frac{\vartheta_{i-1}-\vartheta_i}{R_{i-1}}
 *     - \frac{\vartheta_i - \vartheta_{i+1}}{R_i}\ (+\,u\ \text{at node 0}),
 * @f]
 * with @f$ \vartheta_N \equiv 0 @f$ (the reference). Steady state gives
 * @f$ \vartheta_0 = u\sum_i R_i @f$. Continuous (Ts = 0); discretize with
 * @ref discretize.
 *
 * @param R [K/W] stage resistances (R[N-1] returns to the reference).
 * @param C [J/K] node capacitances to the reference.
 * @return Continuous @ref StateSpace<N,1,1>.
 */
template<std::size_t N, typename T = double>
[[nodiscard]] constexpr StateSpace<N, 1, 1, 0, 0, T> cauer_thermal_ss(const wet::array<T, N>& R, const wet::array<T, N>& C) {
    StateSpace<N, 1, 1, 0, 0, T> sys{};
    // Couplings between adjacent nodes via R[i] (i = 0..N-2).
    for (std::size_t i = 0; i + 1 < N; ++i) {
        const T g = T{1} / R[i];
        sys.A(i, i) -= g / C[i];
        sys.A(i, i + 1) += g / C[i];
        sys.A(i + 1, i) += g / C[i + 1];
        sys.A(i + 1, i + 1) -= g / C[i + 1];
    }
    // Last node returns to the reference (ground) through R[N-1].
    sys.A(N - 1, N - 1) -= (T{1} / R[N - 1]) / C[N - 1];
    sys.B(0, 0) = T{1} / C[0]; // power injected at the junction node
    sys.C(0, 0) = T{1};        // output is the junction node temperature
    return sys;
}

} // namespace design

/**
 * @brief A loss model usable by @ref JunctionEstimator.
 *
 * Any type exposing @c loss(i_rms, Vdc, tj) returning power [W] qualifies —
 * @ref FetLossModel (full datasheet) or @ref ResistiveLossModel (one number).
 */
template<typename M, typename T>
concept ThermalLossModel = requires(const M& m, T x) { m.loss(x, x, x); };

/**
 * @brief First-order inverter FET loss model (conduction + switching).
 *
 * Estimates total inverter device loss from a representative phase RMS current
 * and the bus, to drive a @ref JunctionEstimator:
 * @f[
 *   P = n\left[\,I_{rms}^2\,R_{ds}(T_j) + f_{sw}\,E_{sw}\,
 *       \frac{V_{dc}}{V_{ref}}\frac{I_{rms}}{I_{ref}}\,\right],
 * @f]
 * with @f$ R_{ds}(T_j) = R_{ds,0}\,[1 + \alpha(T_j - T_{ref})] @f$. The switching
 * term scales the datasheet energy @f$ E_{sw} = E_{on}+E_{off} @f$ from its
 * reference operating point. The constants are hardware-specific calibration
 * knobs — a first-order estimate, not a substitute for measurement.
 *
 * @tparam T Scalar type.
 */
template<typename T = double>
struct FetLossModel {
    T rds_on{T{0}};        //!< [ohm] on-resistance at @ref t_ref
    T rds_on_tempco{T{0}}; //!< [1/°C] on-resistance temperature coefficient @f$ \alpha @f$
    T t_ref{T{25}};        //!< [°C] reference temperature for rds_on / Esw
    T sw_energy{T{0}};     //!< [J] E_on + E_off at (v_ref, i_ref)
    T v_ref{T{1}};         //!< [V] reference bus voltage for sw_energy (> 0)
    T i_ref{T{1}};         //!< [A] reference current for sw_energy (> 0)
    T f_sw{T{0}};          //!< [Hz] switching frequency
    T device_count{T{6}};  //!< effective number of devices dissipating

    /**
     * @brief Total FET loss at an operating point.
     * @param i_rms Representative phase RMS current [A].
     * @param Vdc   Bus voltage [V].
     * @param tj    Present junction temperature for the rds_on tempco [°C].
     * @return Total inverter device power loss [W].
     */
    [[nodiscard]] constexpr T loss(T i_rms, T Vdc, T tj) const {
        const T rds = rds_on * (T{1} + (rds_on_tempco * (tj - t_ref)));
        const T conduction = i_rms * i_rms * rds;
        const T switching = f_sw * sw_energy * (Vdc / v_ref) * (i_rms / i_ref);
        return device_count * (conduction + switching);
    }
};

/**
 * @brief Minimal conduction-only loss model for a weak datasheet.
 *
 * For when only a rough on-resistance is known: @f$ P = I_{rms}^2\,R @f$, ignoring
 * switching loss and the Tj coefficient. @p loss_resistance is the lumped effective
 * resistance of all conducting devices. The hobbyist counterpart to @ref FetLossModel.
 *
 * @tparam T Scalar type.
 */
template<typename T = double>
struct ResistiveLossModel {
    T loss_resistance{T{0}}; //!< [ohm] lumped conduction resistance

    [[nodiscard]] constexpr T loss(T i_rms, T /*Vdc*/, T /*tj*/) const {
        return i_rms * i_rms * loss_resistance;
    }
};

/**
 * @brief FET junction-temperature estimator: case temperature plus a thermal model.
 *
 * Steps a discrete junction-to-reference thermal @ref StateSpace (from
 * @ref design::foster_thermal_ss or @ref design::cauer_thermal_ss, discretized with
 * @ref discretize) driven by a loss model, and adds the measured case temperature:
 * @f$ T_j = T_{case} + (C\,x)_0 @f$. A measured case NTC lags the silicon, so
 * deriving the limit from @f$ T_j @f$ rather than the case reading catches fast load
 * transients. Running the discretized LTI model — rather than a hand-rolled Euler
 * step — makes the per-tick update an exact ZOH propagation, stable for any step.
 *
 * @tparam N    Thermal model order.
 * @tparam T    Scalar type.
 * @tparam Loss Loss model type (default @ref FetLossModel).
 */
template<std::size_t N, typename T = double, ThermalLossModel<T> Loss = FetLossModel<T>>
class JunctionEstimator {
public:
    constexpr JunctionEstimator() = default;

    /**
     * @param loss        Loss model producing power [W] from the operating point.
     * @param discrete_sys Discrete thermal model (junction rise per unit power), e.g.
     *                     `discretize(design::cauer_thermal_ss(R, C), Ts, ZOH)`.
     */
    constexpr JunctionEstimator(const Loss& loss, const StateSpace<N, 1, 1, 0, 0, T>& discrete_sys)
        : loss_(loss), sys_(discrete_sys) {}

    /**
     * @brief Advance the thermal state one step; return the junction temperature.
     * @param i_rms     Representative phase RMS current [A].
     * @param Vdc       Bus voltage [V].
     * @param case_temp Measured case/heatsink temperature [°C].
     * @return Estimated junction temperature [°C]. (Step period is baked into the
     *         discretized @p sys_.)
     */
    constexpr T step(T i_rms, T Vdc, T case_temp) {
        const T power = loss_.loss(i_rms, Vdc, tj_);
        x_ = sys_.A * x_ + sys_.B * power;     // exact ZOH propagation
        tj_ = case_temp + (sys_.C * x_)(0, 0); // junction rise above the case
        return tj_;
    }

    [[nodiscard]] constexpr T junction_temperature() const { return tj_; }

    constexpr void reset(T tj0 = T{25}) {
        x_ = ColVec<N, T>{};
        tj_ = tj0;
    }

private:
    Loss                         loss_{};
    StateSpace<N, 1, 1, 0, 0, T> sys_{};
    ColVec<N, T>                 x_{};
    T                            tj_{T{25}};
};

/**
 * @brief A derating curve plus a hard fault threshold.
 *
 * @ref derate is any falling (temperature → factor) @ref Lut1D; @ref fault_temp is
 * the temperature above which the caller must disarm. The default-constructed value
 * never derates and never faults.
 *
 * @tparam N Number of derating-curve breakpoints.
 * @tparam T Scalar type.
 */
template<std::size_t N = 2, typename T = double>
struct ThermalLimits {
    Lut1D<N, T> derate{};                                  //!< temperature [°C] → factor in [0,1]
    T           fault_temp{std::numeric_limits<T>::max()}; //!< [°C] disarm above this

    constexpr ThermalLimits() {
        // Default: a flat factor of 1 (no derate) so a zero-initialised Lut1D does
        // not silently clamp the current command to zero.
        for (std::size_t i = 0; i < N; ++i) {
            derate.xs[i] = static_cast<T>(i);
            derate.ys[i] = T{1};
        }
    }
    constexpr ThermalLimits(const Lut1D<N, T>& curve, T fault) : derate(curve), fault_temp(fault) {}
};

/**
 * @brief State from a @ref ThermalLimiter evaluation.
 * @tparam T Scalar type.
 */
template<typename T = double>
struct ThermalState {
    T    scale{T{1}}; //!< [-] in [0,1], multiply onto the current command
    bool ok{true};    //!< false above fault_temp (caller disarms)
};

/**
 * @brief Derates the current command from a temperature (Tj for FETs, winding for the motor).
 *
 * Looks the temperature up in a falling @ref Lut1D derating curve and raises a hard
 * fault above @ref ThermalLimits::fault_temp. Feed it @ref JunctionEstimator output
 * for the inverter, or a winding-temperature estimate for the motor.
 *
 * @tparam N Derating-curve breakpoints.
 * @tparam T Scalar type.
 */
template<std::size_t N = 2, typename T = double>
class ThermalLimiter {
public:
    constexpr ThermalLimiter() = default;
    constexpr explicit ThermalLimiter(const ThermalLimits<N, T>& limits) : limits_(limits) {}

    [[nodiscard]] constexpr ThermalState<T> evaluate(T temp) const {
        return ThermalState<T>{
            .scale = limits_.derate(temp),
            .ok = temp <= limits_.fault_temp,
        };
    }

    [[nodiscard]] constexpr const ThermalLimits<N, T>& limits() const { return limits_; }
    constexpr void                                     set_limits(const ThermalLimits<N, T>& limits) { limits_ = limits; }

private:
    ThermalLimits<N, T> limits_{};
};
} // namespace wet
