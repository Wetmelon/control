#pragma once

/**
 * @file relay_autotune.hpp
 * @brief Åström-Hägglund relay-feedback autotuner.
 *
 * Drives a plant with a hysteresis-relay element to induce a sustained limit
 * cycle, then estimates the ultimate gain Kᵤ and ultimate period Tᵤ from the
 * limit-cycle amplitude and period. The (Kᵤ, Tᵤ) pair feeds a tuning rule
 * (Ziegler-Nichols, Tyreus-Luyben, AMIGO, ...) to compute PID gains.
 *
 * Relay law (with hysteresis ε around the setpoint r):
 *
 *     u[k] = u_bias + d   if (y[k] − r) < −ε
 *     u[k] = u_bias − d   if (y[k] − r) > +ε
 *     u[k] = u[k−1]       otherwise   (hysteresis hold)
 *
 * Once the limit cycle settles (consecutive half-period durations within
 * @ref RelayAutotuneConfig::period_tolerance of one another):
 *
 *     Tᵤ = 2 · mean(half_period)
 *     Kᵤ = 4d / (π · √(a² − ε²))             (Åström-Hägglund, 1984)
 *
 * where `a` is half the peak-to-peak excursion of y about the setpoint. With
 * ε = 0 the expression reduces to Kᵤ = 4d / (π · a).
 *
 * Example: relay autotune for a discrete plant, then apply Ziegler-Nichols.
 * @code
 * #include "wet/design/relay_autotune.hpp"
 * #include "wet/design/pid_design.hpp"
 *
 * using namespace wet;
 *
 * constexpr float Ts = 0.02f;
 *
 * constexpr design::RelayAutotuneConfig<float> cfg{
 *     .amplitude       = 1.0f,
 *     .hysteresis      = 0.05f,
 *     .setpoint        = 0.0f,
 *     .warmup_cycles   = 2,
 *     .measure_cycles  = 6,
 *     .period_tolerance = 0.05f,
 *     .max_duration    = 30.0f,
 * };
 *
 * constexpr auto autotune = design::synthesize_relay_autotune(cfg);
 * static_assert(autotune.success);
 *
 * RelayAutotuner<float> tuner(autotune, Ts);
 *
 * // In the control loop, replace the controller with the tuner:
 * RelayAutotuneOutput<float> out = tuner.step(y);
 * float u = out.u;
 *
 * if (out.status == RelayAutotuneStatus::Done) {
 *     // Tyreus-Luyben (1992) is the recommended modern drop-in over the original
 *     // Ziegler-Nichols formulas: same (Kᵤ, Tᵤ) input, gentler gains, ~6 dB
 *     // gain margin, no quarter-decay oscillation. Use design::ziegler_nichols
 *     // only when matching a textbook or legacy controller.
 *     auto pid = design::tyreus_luyben(out.Ku, out.Tu, Ts);
 *     // hand pid off to a PIDController and resume normal control
 * }
 * @endcode
 *
 * The relay test produces a single Nyquist point at the ultimate frequency, so
 * any tuning rule downstream is operating on limited information. Kᵤ from a
 * symmetric-relay limit cycle relies on the relay's first-harmonic describing
 * function `N(a) = 4d/(πa)` and is typically accurate to ~15–25% on low-order
 * plants (higher harmonics not fully filtered); Tᵤ is accurate to a few percent
 * (just a zero-crossing measurement). Tighter identification needs an
 * asymmetric/biased relay (also recovers static gain Kₛ for AMIGO tuning) or a
 * dedicated FOPDT fit.
 *
 * @see "Automatic tuning of simple regulators with specifications on phase and
 *      amplitude margins" (Åström & Hägglund, 1984), Automatica 20(5).
 *      https://doi.org/10.1016/0005-1098(84)90014-1
 * @see "Advanced PID Control" (Åström & Hägglund, 2006), Chapter 8.
 * @see "Tuning PI Controllers for Integrator/Dead Time Processes"
 *      (Tyreus & Luyben, 1992), Ind. Eng. Chem. Res. 31(11).
 *      https://doi.org/10.1021/ie00011a019
 * @see pid_design.hpp for tuning rules that consume (Kᵤ, Tᵤ) — prefer
 *      `tyreus_luyben` over `ziegler_nichols` for new designs.
 *
 * @note The relay autotuner is a closed-loop *experiment driver*, not a
 *       tracking controller, and intentionally does not satisfy
 *       @ref SISOController. Its `step(T y)` returns a struct
 *       `{u, status, Ku, Tu}` and runs a lifecycle state machine, so it
 *       cannot drop into `Cascade<Outer, Inner>` -- it temporarily
 *       *replaces* the controller for the duration of the experiment.
 */

#include <cstddef>
#include <cstdint>
#include <limits>

#include "wet/math/math.hpp"

namespace wet {

namespace design {

/**
 * @brief Configuration for the relay-feedback autotuning experiment.
 *
 * The relay output applied to the plant is `u = u_bias ± amplitude`, clamped
 * to [u_min, u_max]. Hysteresis ε around the setpoint rejects measurement
 * noise: the relay flips only when |y − setpoint| exceeds ε.
 */
template<typename T = double>
struct RelayAutotuneConfig {
    /// Maximum supported `measure_cycles`. Lets the runtime keep a fixed
    /// state footprint without dynamic allocation.
    static constexpr std::size_t kMaxMeasureCycles = 32;

    T amplitude{T{1}};                            ///< Relay magnitude d (> 0)
    T hysteresis{T{0}};                           ///< ε around setpoint (≥ 0)
    T setpoint{T{0}};                             ///< Operating point r
    T u_bias{T{0}};                               ///< Baseline command (relay swings around this)
    T u_min{-std::numeric_limits<T>::infinity()}; ///< Output saturation (lower)
    T u_max{std::numeric_limits<T>::infinity()};  ///< Output saturation (upper)

    std::size_t warmup_cycles{2};  ///< Half-periods discarded as transient
    std::size_t measure_cycles{6}; ///< Half-periods averaged for Tᵤ (≥ 2)

    T period_tolerance{T{0.05}}; ///< Convergence: (max − min) / mean ≤ tol
    T max_duration{T{30}};       ///< Safety timeout (seconds, > 0)

    [[nodiscard]] constexpr bool valid() const {
        if (amplitude <= T{0}) {
            return false;
        }
        if (hysteresis < T{0}) {
            return false;
        }
        if (u_min >= u_max) {
            return false;
        }
        if (measure_cycles < std::size_t{2}) {
            return false;
        }
        if (measure_cycles > kMaxMeasureCycles) {
            return false;
        }
        if (period_tolerance <= T{0}) {
            return false;
        }
        if (max_duration <= T{0}) {
            return false;
        }
        return true;
    }
};

/**
 * @brief Relay-autotuner design payload.
 *
 * Validates the @ref RelayAutotuneConfig and carries it through `.as<U>()` for
 * conversion to the embedded scalar type.
 */
template<typename T = double>
struct RelayAutotuneResult {
    RelayAutotuneConfig<T> config{};
    bool                   success{false};

    template<typename U>
    [[nodiscard]] constexpr RelayAutotuneResult<U> as() const {
        return RelayAutotuneResult<U>{
            RelayAutotuneConfig<U>{
                static_cast<U>(config.amplitude),
                static_cast<U>(config.hysteresis),
                static_cast<U>(config.setpoint),
                static_cast<U>(config.u_bias),
                static_cast<U>(config.u_min),
                static_cast<U>(config.u_max),
                config.warmup_cycles,
                config.measure_cycles,
                static_cast<U>(config.period_tolerance),
                static_cast<U>(config.max_duration),
            },
            success,
        };
    }
};

/**
 * @brief Build a validated relay-autotune design payload.
 *
 * @param config Experiment configuration.
 * @return RelayAutotuneResult with `success = config.valid()`.
 */
template<typename T = double>
[[nodiscard]] constexpr RelayAutotuneResult<T>
synthesize_relay_autotune(const RelayAutotuneConfig<T>& config) {
    return RelayAutotuneResult<T>{config, config.valid()};
}

} // namespace design

/**
 * @brief Lifecycle state of a @ref RelayAutotuner.
 */
enum class RelayAutotuneStatus : std::uint8_t {
    Idle,      ///< Constructed, no samples processed yet
    Warmup,    ///< Discarding initial half-periods (transient)
    Measuring, ///< Collecting half-periods, checking convergence
    Done,      ///< Limit cycle settled; Kᵤ and Tᵤ are valid
    Failed,    ///< Invalid config, timeout, or amplitude ≤ hysteresis
};

/**
 * @brief Per-tick output of @ref RelayAutotuner::step.
 *
 * `u` is the relay command for the current sample; apply it to the plant
 * exactly as you would a controller output. `Ku` / `Tu` are populated only
 * once `status == Done`.
 */
template<typename T = float>
struct RelayAutotuneOutput {
    T                   u{T{0}};                           ///< Relay command
    RelayAutotuneStatus status{RelayAutotuneStatus::Idle}; ///< Lifecycle state
    T                   Ku{T{0}};                          ///< Ultimate gain (Done only)
    T                   Tu{T{0}};                          ///< Ultimate period in seconds (Done only)
};

/**
 * @brief Runtime relay-feedback autotuner.
 *
 * Replaces the closed-loop controller for the duration of the experiment.
 * Each `step(y)` applies the hysteresis-relay law, tracks limit-cycle period
 * and amplitude, and reports `(Ku, Tu)` when the period stabilizes.
 *
 * Allocation-free; suitable for ISR / RTOS use. Pair with the tuning rules in
 * @ref pid_design.hpp once `status == Done`.
 */
template<typename T = float>
class RelayAutotuner {
public:
    constexpr RelayAutotuner() = default;

    /**
     * @brief Construct from a validated design payload and the loop sample time.
     *
     * @param design Relay-autotune design payload.
     * @param Ts     Sample time in seconds (> 0).
     */
    constexpr RelayAutotuner(const design::RelayAutotuneResult<T>& design, T Ts)
        : config_(design.config), Ts_(Ts), valid_design_(design.success && Ts > T{0}), status_(valid_design_ ? RelayAutotuneStatus::Idle : RelayAutotuneStatus::Failed) {}

    /**
     * @brief Reset to the Idle state, discarding all collected data.
     */
    constexpr void reset() {
        status_ = valid_design_ ? RelayAutotuneStatus::Idle : RelayAutotuneStatus::Failed;
        elapsed_ = T{0};
        time_since_last_crossing_ = T{0};
        relay_sign_ = T{1};
        warmup_count_ = 0;
        measure_count_ = 0;
        period_sum_ = T{0};
        period_min_ = std::numeric_limits<T>::infinity();
        period_max_ = T{0};
        peak_pos_ = -std::numeric_limits<T>::infinity();
        peak_neg_ = std::numeric_limits<T>::infinity();
        cycle_peak_pos_ = -std::numeric_limits<T>::infinity();
        cycle_peak_neg_ = std::numeric_limits<T>::infinity();
        Ku_ = T{0};
        Tu_ = T{0};
    }

    /**
     * @brief Advance the experiment by one sample.
     *
     * @param y Plant output for this sample.
     * @return Relay command and current status; Kᵤ/Tᵤ valid when Done.
     */
    [[nodiscard]] constexpr RelayAutotuneOutput<T> step(T y) {
        if (status_ == RelayAutotuneStatus::Failed) {
            return RelayAutotuneOutput<T>{saturate(config_.u_bias), status_, Ku_, Tu_};
        }

        const T err = y - config_.setpoint;

        if (status_ == RelayAutotuneStatus::Idle) {
            status_ = (config_.warmup_cycles > 0) ? RelayAutotuneStatus::Warmup
                                                  : RelayAutotuneStatus::Measuring;
            // Kick the plant toward the setpoint on the first sample.
            relay_sign_ = (err >= T{0}) ? T{-1} : T{1};
            cycle_peak_pos_ = y;
            cycle_peak_neg_ = y;
        }

        if (y > cycle_peak_pos_) {
            cycle_peak_pos_ = y;
        }
        if (y < cycle_peak_neg_) {
            cycle_peak_neg_ = y;
        }

        // Hysteresis-relay decision: flip only when |err| exceeds ε.
        T desired_sign = relay_sign_;
        if (err > config_.hysteresis) {
            desired_sign = T{-1};
        } else if (err < -config_.hysteresis) {
            desired_sign = T{1};
        }

        const bool switched = (desired_sign != relay_sign_);
        relay_sign_ = desired_sign;

        if (switched && status_ != RelayAutotuneStatus::Done) {
            on_half_period_complete(time_since_last_crossing_, y);
            time_since_last_crossing_ = T{0};
        }

        time_since_last_crossing_ += Ts_;
        elapsed_ += Ts_;

        if (status_ != RelayAutotuneStatus::Done && elapsed_ > config_.max_duration) {
            status_ = RelayAutotuneStatus::Failed;
        }

        return RelayAutotuneOutput<T>{
            saturate(config_.u_bias + (relay_sign_ * config_.amplitude)),
            status_,
            Ku_,
            Tu_,
        };
    }

    [[nodiscard]] constexpr RelayAutotuneStatus status() const { return status_; }
    [[nodiscard]] constexpr T                   Ku() const { return Ku_; }
    [[nodiscard]] constexpr T                   Tu() const { return Tu_; }

private:
    constexpr T saturate(T u) const {
        if (u < config_.u_min) {
            return config_.u_min;
        }
        if (u > config_.u_max) {
            return config_.u_max;
        }
        return u;
    }

    constexpr void on_half_period_complete(T half_period, T y) {
        if (status_ == RelayAutotuneStatus::Warmup) {
            ++warmup_count_;
            if (warmup_count_ >= config_.warmup_cycles) {
                begin_measuring(y);
            } else {
                // Continue warmup; reset per-half-period peaks for the next swing.
                cycle_peak_pos_ = y;
                cycle_peak_neg_ = y;
            }
            return;
        }

        if (status_ == RelayAutotuneStatus::Measuring) {
            if (cycle_peak_pos_ > peak_pos_) {
                peak_pos_ = cycle_peak_pos_;
            }
            if (cycle_peak_neg_ < peak_neg_) {
                peak_neg_ = cycle_peak_neg_;
            }

            period_sum_ += half_period;
            if (half_period < period_min_) {
                period_min_ = half_period;
            }
            if (half_period > period_max_) {
                period_max_ = half_period;
            }
            ++measure_count_;

            cycle_peak_pos_ = y;
            cycle_peak_neg_ = y;

            if (measure_count_ >= config_.measure_cycles) {
                const T mean = period_sum_ / static_cast<T>(measure_count_);
                const T spread = (period_max_ - period_min_) / mean;
                if (spread <= config_.period_tolerance) {
                    finalize(mean);
                } else {
                    // Window did not settle; restart accumulation. The
                    // max_duration safety net catches stuck experiments.
                    begin_measuring(y);
                }
            }
        }
    }

    constexpr void begin_measuring(T y) {
        status_ = RelayAutotuneStatus::Measuring;
        measure_count_ = 0;
        period_sum_ = T{0};
        period_min_ = std::numeric_limits<T>::infinity();
        period_max_ = T{0};
        peak_pos_ = -std::numeric_limits<T>::infinity();
        peak_neg_ = std::numeric_limits<T>::infinity();
        cycle_peak_pos_ = y;
        cycle_peak_neg_ = y;
    }

    constexpr void finalize(T half_period_mean) {
        Tu_ = T{2} * half_period_mean;
        const T a = (peak_pos_ - peak_neg_) / T{2};
        const T eps = config_.hysteresis;
        // Use (a − ε)(a + ε) rather than a² − ε² so the cancellation-prone
        // subtraction never sits under the square root directly.
        const T arg = (a - eps) * (a + eps);
        if (arg <= T{0}) {
            status_ = RelayAutotuneStatus::Failed;
            return;
        }
        Ku_ = (T{4} * config_.amplitude) / (wet::numbers::pi_v<T> * wet::sqrt(arg));
        status_ = RelayAutotuneStatus::Done;
    }

    design::RelayAutotuneConfig<T> config_{};
    T                              Ts_{T{0}};
    bool                           valid_design_{false};

    RelayAutotuneStatus status_{RelayAutotuneStatus::Idle};
    T                   elapsed_{T{0}};
    T                   time_since_last_crossing_{T{0}};
    T                   relay_sign_{T{1}};

    std::size_t warmup_count_{0};
    std::size_t measure_count_{0};

    T period_sum_{T{0}};
    T period_min_{std::numeric_limits<T>::infinity()};
    T period_max_{T{0}};

    T peak_pos_{-std::numeric_limits<T>::infinity()};
    T peak_neg_{std::numeric_limits<T>::infinity()};
    T cycle_peak_pos_{-std::numeric_limits<T>::infinity()};
    T cycle_peak_neg_{std::numeric_limits<T>::infinity()};

    T Ku_{T{0}};
    T Tu_{T{0}};
};

} // namespace wet
