#pragma once

#include <cstddef>
#include <cstdint>

#include "wet/estimation/excitation.hpp"              // design::PRBSConfig, PRBS
#include "wet/estimation/recursive_least_squares.hpp" // RecursiveLeastSquaresVectorEstimator
#include "wet/math/math.hpp"                          // wet::log
#include "wet/matrix/colvec.hpp"

namespace wet {

/**
 * @brief Configuration for online phase resistance/inductance commissioning.
 *
 * @ref prbs_clock_s sets the excitation bandwidth; a chip period of a few control
 * steps excites the R–L pole. A chip near @f$ T_s @f$ or much slower than the
 * electrical time constant @f$ L/R @f$ excites mainly @f$ R @f$. @ref lambda = 1
 * (growing window) suits time-invariant parameters.
 *
 * @tparam T Scalar type.
 */
template<typename T = double>
struct PhaseCalibrationConfig {
    T             inject_voltage{T{1}};  //!< [V] PRBS amplitude applied on the d axis (> 0)
    T             duration_s{T{0.5}};    //!< [s] total injection window (> 0)
    std::size_t   prbs_order{10};        //!< LFSR order (excitation richness / period)
    T             prbs_clock_s{T{5e-4}}; //!< [s] PRBS chip period (excitation bandwidth, > 0)
    std::uint32_t seed{0x0ACE1u};        //!< LFSR seed (non-zero)
    T             lambda{T{1}};          //!< RLS forgetting factor, (0, 1]; 1 = growing window
    T             p0{T{1000}};           //!< RLS initial covariance (> 0; larger = faster start)

    [[nodiscard]] constexpr bool valid() const {
        return (inject_voltage > T{0}) && (duration_s > T{0}) && (prbs_clock_s > T{0})
            && (lambda > T{0}) && (lambda <= T{1}) && (p0 > T{0});
    }
};

/**
 * @brief One step's output from @ref PhaseParameterCalibrator.
 * @tparam T Scalar type.
 */
template<typename T = double>
struct PhaseCalibrationCommand {
    T    v_d{T{0}};   //!< [V] d-axis voltage to apply over the coming interval
    bool done{false}; //!< true once the injection window has elapsed
};

/**
 * @brief Online phase R/L identification by recursive least squares (PRBS injected).
 *
 * Commissions the per-phase resistance @f$ R @f$ and d-axis inductance @f$ L_d @f$
 * of a PMSM by injecting a PRBS voltage on the d axis (rotor held at a fixed
 * electrical angle, so the d axis is a pure series R–L circuit producing no
 * torque) and regressing the measured current.
 *
 * Fits the exact zero-order-hold discretization of @f$ V_d = R\,i_d + L_d\,\dot i_d @f$,
 * which uses no derivative of the measured current:
 * @f[
 *   i_d[k] = a\,i_d[k-1] + b\,V_d[k-1], \quad
 *   a = e^{-R T_s / L_d}, \quad b = \frac{1-a}{R}.
 * @f]
 * The regressor @f$ \varphi = [\,i_d[k-1],\ V_d[k-1]\,] @f$ and parameter vector
 * @f$ \theta = [\,a,\ b\,] @f$ feed a @ref estimation::RecursiveLeastSquaresVectorEstimator,
 * and the physical parameters fall out of the fitted @f$ (a,b) @f$:
 * @f[
 *   R = \frac{1-a}{b}, \qquad L_d = \frac{-R\,T_s}{\ln a}.
 * @f]
 *
 * @note Average-value model: @f$ i_d[k] @f$ must be the ripple-free average
 *       current, i.e. synchronous sampling at the carrier peak/trough of
 *       center-aligned PWM (the scheme @ref FOController already assumes). @f$ T_s @f$
 *       is the PWM period. Mid-ripple or sawtooth sampling biases the fit, @f$ L_d @f$ most.
 *
 * Usage — each control tick, feed the current measured from the previously
 * commanded voltage and apply the returned one:
 * @code
 * PhaseParameterCalibrator<float> cal{PhaseCalibrationConfig<float>{.inject_voltage = 2.0f}};
 * float v_d = 0.0f;
 * for (;;) {
 *     float i_d = measure_d_axis_current();      // response to the last v_d
 *     auto cmd = cal.step(i_d, Ts);
 *     apply_d_axis_voltage(cmd.v_d);             // hold over the next interval
 *     v_d = cmd.v_d;
 *     if (cmd.done) { break; }
 * }
 * if (cal.valid()) { foc.R = cal.resistance(); foc.Ldq = {cal.inductance(), cal.inductance()}; }
 * @endcode
 *
 * For a salient machine, repeat on the q axis (inject @f$ V_q @f$ at the same
 * fixed angle) to identify @f$ L_q @f$; an SPM may take @f$ L_q = L_d @f$.
 *
 * @see estimation::RecursiveLeastSquaresVectorEstimator — the RLS core.
 * @see design::synthesize_prbs — the PRBS excitation.
 * @see Ljung, "System Identification: Theory for the User" (2nd ed., 1999), §10
 *      — recursive least squares for ARX models.
 *
 * @tparam T Scalar type.
 */
template<typename T = double>
class PhaseParameterCalibrator {
public:
    constexpr PhaseParameterCalibrator() = default;

    constexpr explicit PhaseParameterCalibrator(const PhaseCalibrationConfig<T>& config)
        : config_(config) {
        const design::PRBSConfig<T> prbs_config{
            .amplitude = config.inject_voltage,
            .lfsr_order = config.prbs_order,
            .clock_period_s = config.prbs_clock_s,
            .seed = config.seed,
        };
        prbs_ = PRBS<T>(design::synthesize_prbs(prbs_config), config.prbs_clock_s);

        estimation::RecursiveLeastSquaresConfig<T> rls_config{};
        rls_config.lambda = config.lambda;
        rls_config.p0 = config.p0;
        rls_ = estimation::RecursiveLeastSquaresVectorEstimator<2, T>(rls_config);
    }

    /**
     * @brief Feed the latest measured d-axis current and get the next voltage.
     *
     * @param i_d_measured [A] d-axis current measured from the previously commanded voltage.
     * @param Ts           [s] Control step period (assumed fixed across the run).
     * @return The d-axis voltage to apply next, and whether the window has elapsed.
     */
    [[nodiscard]] constexpr PhaseCalibrationCommand<T> step(T i_d_measured, T Ts) {
        ts_ = Ts;

        if (has_prev_) {
            ColVec<2, T> phi{};
            phi[0] = i_prev_;                                  // i_d[k-1]
            phi[1] = v_prev_;                                  // V_d[k-1]
            static_cast<void>(rls_.update(phi, i_d_measured)); // y = i_d[k]
        }

        // Continuous PRBS via the stateless absolute-time form (wraps every period),
        // so excitation never stops short within the injection window.
        const T v_next = prbs_.step(t_);

        i_prev_ = i_d_measured;
        v_prev_ = v_next;
        has_prev_ = true;
        t_ += Ts;

        return PhaseCalibrationCommand<T>{.v_d = v_next, .done = t_ >= config_.duration_s};
    }

    /// Estimated phase resistance @f$ R = (1-a)/b @f$ [ohm].
    [[nodiscard]] constexpr T resistance() const {
        const auto& theta = rls_.state().theta;
        return (T{1} - theta[0]) / theta[1];
    }

    /// Estimated d-axis inductance @f$ L_d = -R T_s/\ln a @f$ [H].
    [[nodiscard]] constexpr T inductance() const {
        const auto& theta = rls_.state().theta;
        const T     a = theta[0];
        const T     R = (T{1} - a) / theta[1];
        return -R * ts_ / wet::log(a);
    }

    /**
     * @brief True if the fit yields a physically valid R–L model.
     *
     * Requires a stable discrete pole @f$ a \in (0,1) @f$ and a positive input
     * gain @f$ b > 0 @f$ — anything else means the regression has not yet
     * captured a real first-order response (insufficient excitation, no current,
     * or the window was too short).
     */
    [[nodiscard]] constexpr bool valid() const {
        const auto& theta = rls_.state().theta;
        const T     a = theta[0];
        const T     b = theta[1];
        return rls_.valid() && (a > T{0}) && (a < T{1}) && (b > T{0}) && (ts_ > T{0});
    }

    constexpr void reset() {
        prbs_.reset();
        rls_.reset();
        t_ = T{0};
        ts_ = T{0};
        i_prev_ = T{0};
        v_prev_ = T{0};
        has_prev_ = false;
    }

    [[nodiscard]] constexpr const auto& config() const { return config_; }

private:
    PhaseCalibrationConfig<T> config_{};
    PRBS<T>                   prbs_{};

    estimation::RecursiveLeastSquaresVectorEstimator<2, T> rls_{};

    T    t_{T{0}};      // elapsed injection time [s]
    T    ts_{T{0}};     // last control period [s]
    T    i_prev_{T{0}}; // i_d[k-1]
    T    v_prev_{T{0}}; // V_d[k-1]
    bool has_prev_{false};
};

} // namespace wet
