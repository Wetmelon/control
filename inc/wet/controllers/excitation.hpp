#pragma once

/**
 * @file excitation.hpp
 * @brief Allocation-free excitation generators for commissioning and system identification.
 *
 * Provides the embeddable signal sources used in the commissioning workflow:
 * apply a known input, log plant response, then identify and tune offline.
 *
 * Each signal follows the library's three-tier pattern:
 * 1) config + validation in `design::*Config`
 * 2) design payload in `design::*Result`
 * 3) runtime generator with `step(...)`, `done()`, and `reset()`
 *
 * The generators are allocation-free and constexpr-friendly. They are intended
 * for ISR/RTOS use on target, while host-side identification (FRF, tfest,
 * ssest, etc.) consumes the logged data.
 *
 * Signal definitions implemented here:
 *
 * - Chirp:
 *     u(t) = A * sin(phi(t))
 *
 * - Linear sweep (`f0 -> f1` in duration `T`):
 *     phi(t) = 2*pi*(f0*t + 0.5*k*t^2),  k = (f1-f0)/T
 *
 * - Log sweep:
 *     phi(t) = 2*pi*(f0/beta)*(exp(beta*t)-1),  beta = ln(f1/f0)/T
 *
 * - PRBS:
 *     u[k] in {+A, -A} from a maximal-length Galois LFSR.
 *
 * - Step train:
 *     u(t) alternates between +A and -A every hold interval.
 *
 * - Ramp:
 *     u(t) = sign(target)*rate*t until target, then optional hold.
 *
 * - Multi-sine:
 *     u(t) = sum_i Ai*sin(2*pi*fi*t + phii)
 *
 * Example: chirp excitation in a discrete control loop.
 * @code
 * #include "wet/controllers/excitation.hpp"
 *
 * using namespace wet;
 *
 * constexpr auto chirp_cfg = design::synthesize_chirp<double>({
 *     .amplitude = 1.0,
 *     .f_start_hz = 0.2,
 *     .f_end_hz = 30.0,
 *     .duration_s = 20.0,
 *     .mode = design::ChirpMode::Log,
 * });
 * static_assert(chirp_cfg.success);
 *
 * Chirp<double> exciter(chirp_cfg, 0.001);
 *
 * // In the loop: apply u, then log {t, u, y}
 * while (!exciter.done()) {
 *     const double u = exciter.step();
 *     // plant.apply(u); log.push(t, u, y);
 * }
 * @endcode
 *
 * @see Ljung, "System Identification: Theory for the User" (1999)
 * @see Van Overschee & De Moor, "Subspace Identification for Linear Systems" (1996)
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>

#include "wet/math/math.hpp"

namespace wet {

namespace detail {

[[nodiscard]] constexpr std::uint32_t prbs_bit_mask(std::size_t order) {
    if (order == 0 || order >= 32) {
        return 0u;
    }
    return (std::uint32_t{1u} << static_cast<unsigned>(order)) - std::uint32_t{1u};
}

// Galois-LFSR tap masks for maximal-length sequences (orders 2..16).
[[nodiscard]] constexpr std::uint32_t prbs_feedback_mask(std::size_t order) {
    switch (order) {
        case 2: return 0x3u;
        case 3: return 0x6u;
        case 4: return 0xCu;
        case 5: return 0x14u;
        case 6: return 0x30u;
        case 7: return 0x60u;
        case 8: return 0xB8u;
        case 9: return 0x110u;
        case 10: return 0x240u;
        case 11: return 0x500u;
        case 12: return 0xE08u;
        case 13: return 0x1C80u;
        case 14: return 0x3802u;
        case 15: return 0x6000u;
        case 16: return 0xD008u;
        default: return 0u;
    }
}

[[nodiscard]] constexpr std::uint32_t prbs_advance(
    std::uint32_t state,
    std::size_t   order
) {
    const std::uint32_t lsb = state & std::uint32_t{1u};
    state >>= 1u;
    if (lsb != 0u) {
        state ^= prbs_feedback_mask(order);
    }
    const std::uint32_t mask = prbs_bit_mask(order);
    state &= mask;
    if (state == 0u) {
        state = std::uint32_t{1u};
    }
    return state;
}

template<typename T>
[[nodiscard]] constexpr bool finite_positive(T x) {
    return wet::isfinite(x) && (x > T{0});
}

template<typename T>
[[nodiscard]] constexpr bool finite_non_negative(T x) {
    return wet::isfinite(x) && (x >= T{0});
}

template<typename T>
[[nodiscard]] constexpr T clamp_non_negative(T x) {
    return (x < T{0}) ? T{0} : x;
}

} // namespace detail

namespace design {

/**
 * @defgroup excitation_design Excitation Design Payloads
 * @brief Validated excitation configurations for runtime signal generators.
 */

/**
 * @brief Chirp sweep law.
 */
enum class ChirpMode : std::uint8_t {
    Linear, ///< Frequency increases linearly in time.
    Log,    ///< Frequency increases exponentially (constant ratio per unit time).
};

/**
 * @struct ChirpConfig
 * @brief Configuration for a sine chirp excitation.
 *
 * Defines `u(t) = amplitude * sin(phi(t))` with linear or logarithmic sweep.
 *
 * @tparam T Scalar type.
 */
template<typename T = double>
struct ChirpConfig {
    T         amplitude{T{1}};         ///< Signal amplitude (> 0)
    T         f_start_hz{T{1}};        ///< Start frequency in Hz (>= 0; > 0 for log mode)
    T         f_end_hz{T{10}};         ///< End frequency in Hz (>= 0; > 0 for log mode)
    T         duration_s{T{1}};        ///< Sweep duration in seconds (> 0)
    ChirpMode mode{ChirpMode::Linear}; ///< Sweep law (linear or logarithmic)

    /**
     * @brief Validate chirp configuration.
     * @return true if all parameters are finite and physically valid.
     */
    [[nodiscard]] constexpr bool valid() const {
        if (!detail::finite_positive(amplitude)) {
            return false;
        }
        if (!detail::finite_positive(duration_s)) {
            return false;
        }
        if (!detail::finite_non_negative(f_start_hz)) {
            return false;
        }
        if (!detail::finite_non_negative(f_end_hz)) {
            return false;
        }
        if (mode == ChirpMode::Log) {
            if (f_start_hz <= T{0} || f_end_hz <= T{0}) {
                return false;
            }
        }
        return true;
    }
};

/**
 * @struct ChirpResult
 * @brief Chirp design payload.
 *
 * Carries a validated chirp configuration into the runtime generator. Use
 * `.as<float>()` for embedded deployment after host-side design in double.
 *
 * @tparam T Scalar type.
 */
template<typename T = double>
struct ChirpResult {
    ChirpConfig<T> config{};       ///< Validated chirp configuration
    bool           success{false}; ///< true if `config.valid()`

    /**
     * @brief Convert design payload to another scalar type.
     * @tparam U Target scalar type.
     * @return ChirpResult<U> converted element-wise.
     */
    template<typename U>
    [[nodiscard]] constexpr ChirpResult<U> as() const {
        return ChirpResult<U>{
            ChirpConfig<U>{
                static_cast<U>(config.amplitude),
                static_cast<U>(config.f_start_hz),
                static_cast<U>(config.f_end_hz),
                static_cast<U>(config.duration_s),
                config.mode,
            },
            success,
        };
    }
};

/**
 * @brief Build a chirp design payload from a configuration.
 *
 * @param config Chirp configuration.
 * @return ChirpResult with `success = config.valid()`.
 */
template<typename T = double>
[[nodiscard]] constexpr ChirpResult<T>
synthesize_chirp(const ChirpConfig<T>& config) {
    return ChirpResult<T>{config, config.valid()};
}

/**
 * @struct PRBSConfig
 * @brief Configuration for maximal-length pseudo-random binary excitation.
 *
 * The runtime generator emits `u[k] in {+amplitude, -amplitude}` from a
 * Galois LFSR. For order `n`, one period has `(2^n - 1)` chips.
 *
 * @tparam T Scalar type.
 */
template<typename T = double>
struct PRBSConfig {
    static constexpr std::size_t kMinOrder = 2;
    static constexpr std::size_t kMaxOrder = 16;

    T             amplitude{T{1}};          ///< Output level (+/- amplitude), must be > 0
    std::size_t   lfsr_order{10};           ///< LFSR order in [kMinOrder, kMaxOrder]
    T             clock_period_s{T{0.001}}; ///< Chip period in seconds (> 0)
    std::uint32_t seed{1u};                 ///< Initial LFSR state (must not map to all-zero)

    /**
     * @brief Validate PRBS configuration.
     * @return true if order/seed/clock/amplitude are valid.
     */
    [[nodiscard]] constexpr bool valid() const {
        if (!detail::finite_positive(amplitude)) {
            return false;
        }
        if (!detail::finite_positive(clock_period_s)) {
            return false;
        }
        if (lfsr_order < kMinOrder || lfsr_order > kMaxOrder) {
            return false;
        }
        const std::uint32_t mask = detail::prbs_bit_mask(lfsr_order);
        if ((seed & mask) == 0u) {
            return false;
        }
        if (detail::prbs_feedback_mask(lfsr_order) == 0u) {
            return false;
        }
        return true;
    }
};

/**
 * @struct PRBSResult
 * @brief PRBS design payload.
 *
 * Includes the validated configuration and sequence period (`2^n - 1` chips).
 * Use `.as<float>()` for embedded deployment.
 *
 * @tparam T Scalar type.
 */
template<typename T = double>
struct PRBSResult {
    PRBSConfig<T> config{};       ///< Validated PRBS configuration
    std::size_t   period_bits{0}; ///< Sequence period in chips: `2^order - 1`
    bool          success{false}; ///< true if `config.valid()`

    /**
     * @brief Convert design payload to another scalar type.
     * @tparam U Target scalar type.
     * @return PRBSResult<U> converted element-wise.
     */
    template<typename U>
    [[nodiscard]] constexpr PRBSResult<U> as() const {
        return PRBSResult<U>{
            PRBSConfig<U>{
                static_cast<U>(config.amplitude),
                config.lfsr_order,
                static_cast<U>(config.clock_period_s),
                config.seed,
            },
            period_bits,
            success,
        };
    }
};

/**
 * @brief Build a PRBS design payload from a configuration.
 *
 * @param config PRBS configuration.
 * @return PRBSResult with `success = config.valid()` and period metadata.
 */
template<typename T = double>
[[nodiscard]] constexpr PRBSResult<T>
synthesize_prbs(const PRBSConfig<T>& config) {
    const bool        valid = config.valid();
    const std::size_t period_bits = valid
                                      ? ((std::size_t{1} << config.lfsr_order) - std::size_t{1})
                                      : std::size_t{0};
    return PRBSResult<T>{config, period_bits, valid};
}

/**
 * @struct StepTrainConfig
 * @brief Configuration for alternating +/- step excitation.
 *
 * One cycle is two plateaus: `+amplitude` for `hold_s`, then `-amplitude`
 * for `hold_s`.
 *
 * @tparam T Scalar type.
 */
template<typename T = double>
struct StepTrainConfig {
    T           amplitude{T{1}}; ///< Step magnitude (> 0)
    T           hold_s{T{0.1}};  ///< Hold time per plateau in seconds (> 0)
    std::size_t cycles{1};       ///< Number of +/- pairs

    /**
     * @brief Validate step-train configuration.
     * @return true if amplitude, hold, and cycle count are valid.
     */
    [[nodiscard]] constexpr bool valid() const {
        if (!detail::finite_positive(amplitude)) {
            return false;
        }
        if (!detail::finite_positive(hold_s)) {
            return false;
        }
        if (cycles == 0) {
            return false;
        }
        return true;
    }
};

/**
 * @struct StepTrainResult
 * @brief Step-train design payload.
 *
 * Carries validated step-train settings into the runtime generator.
 *
 * @tparam T Scalar type.
 */
template<typename T = double>
struct StepTrainResult {
    StepTrainConfig<T> config{};       ///< Validated step-train configuration
    bool               success{false}; ///< true if `config.valid()`

    /**
     * @brief Convert design payload to another scalar type.
     * @tparam U Target scalar type.
     * @return StepTrainResult<U> converted element-wise.
     */
    template<typename U>
    [[nodiscard]] constexpr StepTrainResult<U> as() const {
        return StepTrainResult<U>{
            StepTrainConfig<U>{
                static_cast<U>(config.amplitude),
                static_cast<U>(config.hold_s),
                config.cycles,
            },
            success,
        };
    }
};

/**
 * @brief Build a step-train design payload from a configuration.
 *
 * @param config Step-train configuration.
 * @return StepTrainResult with `success = config.valid()`.
 */
template<typename T = double>
[[nodiscard]] constexpr StepTrainResult<T>
synthesize_step_train(const StepTrainConfig<T>& config) {
    return StepTrainResult<T>{config, config.valid()};
}

/**
 * @struct RampConfig
 * @brief Configuration for a slew-rate-limited ramp excitation.
 *
 * Ramps from 0 toward `target` at fixed slope `rate`, then optionally holds
 * the final value for `hold_at_end_s`.
 *
 * @tparam T Scalar type.
 */
template<typename T = double>
struct RampConfig {
    T target{T{1}};        ///< Final target value (finite)
    T rate{T{1}};          ///< Ramp rate in units/second (> 0)
    T hold_at_end_s{T{0}}; ///< Post-ramp hold duration in seconds (>= 0)

    /**
     * @brief Validate ramp configuration.
     * @return true if target/rate/hold are finite and valid.
     */
    [[nodiscard]] constexpr bool valid() const {
        if (!wet::isfinite(target)) {
            return false;
        }
        if (!detail::finite_positive(rate)) {
            return false;
        }
        if (!detail::finite_non_negative(hold_at_end_s)) {
            return false;
        }
        return true;
    }
};

/**
 * @struct RampResult
 * @brief Ramp design payload.
 *
 * Carries validated ramp settings into the runtime generator.
 *
 * @tparam T Scalar type.
 */
template<typename T = double>
struct RampResult {
    RampConfig<T> config{};       ///< Validated ramp configuration
    bool          success{false}; ///< true if `config.valid()`

    /**
     * @brief Convert design payload to another scalar type.
     * @tparam U Target scalar type.
     * @return RampResult<U> converted element-wise.
     */
    template<typename U>
    [[nodiscard]] constexpr RampResult<U> as() const {
        return RampResult<U>{
            RampConfig<U>{
                static_cast<U>(config.target),
                static_cast<U>(config.rate),
                static_cast<U>(config.hold_at_end_s),
            },
            success,
        };
    }
};

/**
 * @brief Build a ramp design payload from a configuration.
 *
 * @param config Ramp configuration.
 * @return RampResult with `success = config.valid()`.
 */
template<typename T = double>
[[nodiscard]] constexpr RampResult<T>
synthesize_ramp(const RampConfig<T>& config) {
    return RampResult<T>{config, config.valid()};
}

/**
 * @struct Tone
 * @brief One sinusoidal component in a multi-sine excitation.
 *
 * @tparam T Scalar type.
 */
template<typename T = double>
struct Tone {
    T amplitude{T{0}}; ///< Tone amplitude (>= 0)
    T freq_hz{T{0}};   ///< Tone frequency in Hz (>= 0)
    T phase_rad{T{0}}; ///< Tone phase offset in radians (finite)

    /**
     * @brief Convert tone to another scalar type.
     * @tparam U Target scalar type.
     * @return Tone<U> converted element-wise.
     */
    template<typename U>
    [[nodiscard]] constexpr Tone<U> as() const {
        return Tone<U>{
            static_cast<U>(amplitude),
            static_cast<U>(freq_hz),
            static_cast<U>(phase_rad),
        };
    }
};

/**
 * @struct MultiSineConfig
 * @brief Configuration for fixed-component multi-sine excitation.
 *
 * Defines:
 *     u(t) = sum_i Ai * sin(2*pi*fi*t + phii)
 *
 * At least one tone must have positive amplitude.
 *
 * @tparam NTones Number of tones.
 * @tparam T      Scalar type.
 */
template<std::size_t NTones, typename T = double>
struct MultiSineConfig {
    std::array<Tone<T>, NTones> tones{}; ///< Fixed tone table

    /**
     * @brief Validate multi-sine configuration.
     * @return true if all tones are finite/non-negative and at least one tone has energy.
     */
    [[nodiscard]] constexpr bool valid() const {
        if constexpr (NTones == 0) {
            return false;
        }

        bool has_energy = false;
        for (std::size_t i = 0; i < NTones; ++i) {
            const auto& tone = tones[i];
            if (!detail::finite_non_negative(tone.amplitude)) {
                return false;
            }
            if (!detail::finite_non_negative(tone.freq_hz)) {
                return false;
            }
            if (!wet::isfinite(tone.phase_rad)) {
                return false;
            }
            if (tone.amplitude > T{0}) {
                has_energy = true;
            }
        }
        return has_energy;
    }
};

/**
 * @struct MultiSineResult
 * @brief Multi-sine design payload.
 *
 * Carries validated tone definitions into the runtime generator.
 *
 * @tparam NTones Number of tones.
 * @tparam T      Scalar type.
 */
template<std::size_t NTones, typename T = double>
struct MultiSineResult {
    MultiSineConfig<NTones, T> config{};       ///< Validated multi-sine configuration
    bool                       success{false}; ///< true if `config.valid()`

    /**
     * @brief Convert design payload to another scalar type.
     * @tparam U Target scalar type.
     * @return MultiSineResult<NTones, U> converted element-wise.
     */
    template<typename U>
    [[nodiscard]] constexpr MultiSineResult<NTones, U> as() const {
        MultiSineConfig<NTones, U> cfg_u{};
        for (std::size_t i = 0; i < NTones; ++i) {
            cfg_u.tones[i] = config.tones[i].template as<U>();
        }
        return MultiSineResult<NTones, U>{cfg_u, success};
    }
};

/**
 * @brief Build a multi-sine design payload from a configuration.
 *
 * @tparam NTones Number of tones.
 * @tparam T      Scalar type.
 * @param config  Multi-sine configuration.
 * @return MultiSineResult with `success = config.valid()`.
 */
template<std::size_t NTones, typename T = double>
[[nodiscard]] constexpr MultiSineResult<NTones, T>
synthesize_multi_sine(const MultiSineConfig<NTones, T>& config) {
    return MultiSineResult<NTones, T>{config, config.valid()};
}

} // namespace design

/**
 * @ingroup discrete_controllers
 * @brief Linear or logarithmic chirp runtime generator.
 *
 * Generates a bounded-time sinusoid with either linear or logarithmic
 * instantaneous frequency sweep. Supports absolute-time evaluation (`step(t)`)
 * and internal-sample-clock evaluation (`step()`).
 *
 * @tparam T Scalar type.
 */
template<typename T = float>
class Chirp {
public:
    constexpr Chirp() = default;

    /**
     * @brief Construct from a chirp design payload.
     * @param design Validated design payload.
     * @param Ts     Optional sample period for `step()` mode. If `Ts <= 0`,
     *               `step()` evaluates at a fixed internal time.
     */
    constexpr explicit Chirp(const design::ChirpResult<T>& design, T Ts = T{0})
        : config_(design.config), Ts_(Ts), valid_(design.success) {}

    /**
     * @brief Evaluate chirp at absolute time.
     * @param t Time in seconds.
     * @return Excitation value at clamped time `t in [0, duration]`.
     */
    [[nodiscard]] constexpr T step(T t) const {
        if (!valid_) {
            return T{0};
        }

        const T tc = clamp_time(t);
        const T phase = (config_.mode == design::ChirpMode::Linear)
                          ? linear_phase(tc)
                          : log_phase(tc);
        return config_.amplitude * wet::sin(phase);
    }

    /**
     * @brief Evaluate chirp at internal time and advance by `Ts`.
     * @return Excitation value at current internal time.
     */
    [[nodiscard]] constexpr T step() {
        const T out = step(t_);
        if (Ts_ > T{0} && !done()) {
            t_ += Ts_;
        }
        return out;
    }

    /**
     * @brief Query whether the chirp duration has elapsed.
     * @param t Time in seconds.
     * @return true if generator is invalid or `t >= duration_s`.
     */
    [[nodiscard]] constexpr bool done(T t) const {
        if (!valid_) {
            return true;
        }
        return t >= config_.duration_s;
    }

    /**
     * @brief Query whether internal time has reached completion.
     * @return true if internal time has elapsed.
     */
    [[nodiscard]] constexpr bool done() const {
        return done(t_);
    }

    /**
     * @brief Reset internal time to zero.
     */
    constexpr void reset() {
        t_ = T{0};
    }

    /**
     * @brief Read current internal time.
     * @return Internal time in seconds.
     */
    [[nodiscard]] constexpr T time() const {
        return t_;
    }

private:
    [[nodiscard]] constexpr T clamp_time(T t) const {
        const T non_negative = detail::clamp_non_negative(t);
        if (non_negative > config_.duration_s) {
            return config_.duration_s;
        }
        return non_negative;
    }

    [[nodiscard]] constexpr T linear_phase(T t) const {
        const T k = (config_.f_end_hz - config_.f_start_hz) / config_.duration_s;
        const T phase_cycles = (config_.f_start_hz * t) + (T{0.5} * k * t * t);
        return T{2} * std::numbers::pi_v<T> * phase_cycles;
    }

    [[nodiscard]] constexpr T log_phase(T t) const {
        const T ratio = config_.f_end_hz / config_.f_start_hz;
        const T beta = wet::log(ratio) / config_.duration_s;
        if (wet::abs(beta) <= std::numeric_limits<T>::epsilon()) {
            return T{2} * std::numbers::pi_v<T> * config_.f_start_hz * t;
        }
        const T phase_cycles = (config_.f_start_hz / beta) * (wet::exp(beta * t) - T{1});
        return T{2} * std::numbers::pi_v<T> * phase_cycles;
    }

    design::ChirpConfig<T> config_{};
    T                      Ts_{T{0}};
    T                      t_{T{0}};
    bool                   valid_{false};
};

/**
 * @ingroup discrete_controllers
 * @brief Maximal-length PRBS runtime generator.
 *
 * Generates a deterministic binary sequence using a Galois LFSR, mapped to
 * output levels `+amplitude` and `-amplitude`.
 *
 * The sequence period is `(2^order - 1)` chips and never enters the all-zero
 * LFSR state.
 *
 * @tparam T Scalar type.
 */
template<typename T = float>
class PRBS {
public:
    constexpr PRBS() = default;

    /**
     * @brief Construct from a PRBS design payload.
     * @param design Validated design payload.
     * @param Ts     Sample period for internal `step()` mode.
     */
    constexpr explicit PRBS(const design::PRBSResult<T>& design, T Ts)
        : config_(design.config), Ts_(Ts), valid_(design.success && Ts > T{0}) {
        reset();
    }

    /**
     * @brief Evaluate PRBS output at absolute time.
     * @param t Time in seconds.
     * @return Sequence value (`+amplitude` or `-amplitude`) at chip index `floor(t/clock_period)`.
     */
    [[nodiscard]] constexpr T step(T t) const {
        if (!config_.valid()) {
            return T{0};
        }

        const T           tc = detail::clamp_non_negative(t);
        const std::size_t period = period_bits();
        auto              chips = static_cast<std::size_t>(wet::floor(tc / config_.clock_period_s));
        if (period > 0) {
            chips %= period;
        }

        std::uint32_t state = seeded_state();
        for (std::size_t i = 0; i < chips; ++i) {
            state = detail::prbs_advance(state, config_.lfsr_order);
        }

        return output_from_state(state);
    }

    /**
     * @brief Evaluate PRBS output at internal time and advance by `Ts`.
     * @return Current sequence value.
     */
    [[nodiscard]] constexpr T step() {
        if (!valid_) {
            return T{0};
        }
        if (done()) {
            return T{0};
        }

        const T out = output_from_state(state_);
        elapsed_chip_time_ += Ts_;

        while (elapsed_chip_time_ >= config_.clock_period_s && !done()) {
            elapsed_chip_time_ -= config_.clock_period_s;
            state_ = detail::prbs_advance(state_, config_.lfsr_order);
            ++chips_generated_;
        }

        return out;
    }

    /**
     * @brief Query whether one full PRBS period has elapsed at absolute time.
     * @param t Time in seconds.
     * @return true if `t >= period_bits * clock_period_s` or config invalid.
     */
    [[nodiscard]] constexpr bool done(T t) const {
        if (!config_.valid()) {
            return true;
        }
        const T period_duration = static_cast<T>(period_bits()) * config_.clock_period_s;
        return t >= period_duration;
    }

    /**
     * @brief Query whether one full PRBS period has been generated internally.
     * @return true once `period_bits()` chips have been emitted.
     */
    [[nodiscard]] constexpr bool done() const {
        return chips_generated_ >= period_bits();
    }

    /**
     * @brief Reset LFSR and internal chip timing to initial state.
     */
    constexpr void reset() {
        state_ = seeded_state();
        chips_generated_ = 0;
        elapsed_chip_time_ = T{0};
    }

    /**
     * @brief Return PRBS period in chips.
     * @return `2^lfsr_order - 1` for valid config, 0 otherwise.
     */
    [[nodiscard]] constexpr std::size_t period_bits() const {
        if (!config_.valid()) {
            return 0;
        }
        return (std::size_t{1} << config_.lfsr_order) - std::size_t{1};
    }

    /**
     * @brief Return current internal LFSR state.
     * @return Raw LFSR register value.
     */
    [[nodiscard]] constexpr std::uint32_t state() const {
        return state_;
    }

private:
    [[nodiscard]] constexpr std::uint32_t seeded_state() const {
        const std::uint32_t mask = detail::prbs_bit_mask(config_.lfsr_order);
        const std::uint32_t seeded = config_.seed & mask;
        if (seeded == 0u) {
            return 1u;
        }
        return seeded;
    }

    [[nodiscard]] constexpr T output_from_state(std::uint32_t state) const {
        return ((state & std::uint32_t{1u}) != 0u) ? config_.amplitude : -config_.amplitude;
    }

    design::PRBSConfig<T> config_{};
    T                     Ts_{T{0}};
    T                     elapsed_chip_time_{T{0}};
    std::size_t           chips_generated_{0};
    std::uint32_t         state_{1u};
    bool                  valid_{false};
};

/**
 * @ingroup discrete_controllers
 * @brief Alternating +/- step train runtime generator.
 *
 * Emits piecewise-constant plateaus of `+amplitude` and `-amplitude` with
 * hold time `hold_s`, for `cycles` complete +/- pairs.
 *
 * @tparam T Scalar type.
 */
template<typename T = float>
class StepTrain {
public:
    constexpr StepTrain() = default;

    /**
     * @brief Construct from a step-train design payload.
     * @param design Validated design payload.
     * @param Ts     Optional sample period for internal `step()` mode.
     */
    constexpr explicit StepTrain(const design::StepTrainResult<T>& design, T Ts = T{0})
        : config_(design.config), Ts_(Ts), valid_(design.success) {}

    /**
     * @brief Evaluate step train at absolute time.
     * @param t Time in seconds.
     * @return +/- amplitude while active, or 0 after completion.
     */
    [[nodiscard]] constexpr T step(T t) const {
        if (!valid_) {
            return T{0};
        }

        const T tc = detail::clamp_non_negative(t);
        if (done(tc)) {
            return T{0};
        }

        const auto segment = static_cast<std::size_t>(wet::floor(tc / config_.hold_s));
        return (segment % 2u == 0u) ? config_.amplitude : -config_.amplitude;
    }

    /**
     * @brief Evaluate step train at internal time and advance by `Ts`.
     * @return Excitation value at current internal time.
     */
    [[nodiscard]] constexpr T step() {
        const T out = step(t_);
        if (Ts_ > T{0} && !done()) {
            t_ += Ts_;
        }
        return out;
    }

    /**
     * @brief Query whether configured cycles are complete at absolute time.
     * @param t Time in seconds.
     * @return true if invalid or `t >= 2*cycles*hold_s`.
     */
    [[nodiscard]] constexpr bool done(T t) const {
        if (!valid_) {
            return true;
        }
        const T total = static_cast<T>(2u * config_.cycles) * config_.hold_s;
        return t >= total;
    }

    /**
     * @brief Query completion at internal time.
     * @return true if internal sequence is complete.
     */
    [[nodiscard]] constexpr bool done() const {
        return done(t_);
    }

    /**
     * @brief Reset internal time to zero.
     */
    constexpr void reset() {
        t_ = T{0};
    }

private:
    design::StepTrainConfig<T> config_{};
    T                          Ts_{T{0}};
    T                          t_{T{0}};
    bool                       valid_{false};
};

/**
 * @ingroup discrete_controllers
 * @brief Rate-limited ramp runtime generator.
 *
 * Ramps from 0 to `target` at magnitude `rate` (units/s), then holds the
 * target for `hold_at_end_s` before reporting done.
 *
 * @tparam T Scalar type.
 */
template<typename T = float>
class Ramp {
public:
    constexpr Ramp() = default;

    /**
     * @brief Construct from a ramp design payload.
     * @param design Validated design payload.
     * @param Ts     Optional sample period for internal `step()` mode.
     */
    constexpr explicit Ramp(const design::RampResult<T>& design, T Ts = T{0})
        : config_(design.config), Ts_(Ts), valid_(design.success) {}

    /**
     * @brief Evaluate ramp at absolute time.
     * @param t Time in seconds.
     * @return Slew-limited value, saturated at target after ramp duration.
     */
    [[nodiscard]] constexpr T step(T t) const {
        if (!valid_) {
            return T{0};
        }

        const T tc = detail::clamp_non_negative(t);
        const T tr = ramp_duration();
        if (tc >= tr) {
            return config_.target;
        }

        const T slope = wet::copysign(config_.rate, config_.target);
        return slope * tc;
    }

    /**
     * @brief Evaluate ramp at internal time and advance by `Ts`.
     * @return Excitation value at current internal time.
     */
    [[nodiscard]] constexpr T step() {
        const T out = step(t_);
        if (Ts_ > T{0} && !done()) {
            t_ += Ts_;
        }
        return out;
    }

    /**
     * @brief Query whether ramp and hold interval are complete at absolute time.
     * @param t Time in seconds.
     * @return true if invalid or `t >= ramp_duration() + hold_at_end_s`.
     */
    [[nodiscard]] constexpr bool done(T t) const {
        if (!valid_) {
            return true;
        }
        return t >= (ramp_duration() + config_.hold_at_end_s);
    }

    /**
     * @brief Query completion at internal time.
     * @return true if internal sequence is complete.
     */
    [[nodiscard]] constexpr bool done() const {
        return done(t_);
    }

    /**
     * @brief Reset internal time to zero.
     */
    constexpr void reset() {
        t_ = T{0};
    }

    /**
     * @brief Return the ramp-only duration.
     * @return `abs(target)/rate` for valid configuration.
     */
    [[nodiscard]] constexpr T ramp_duration() const {
        if (!valid_) {
            return T{0};
        }
        return wet::abs(config_.target) / config_.rate;
    }

private:
    design::RampConfig<T> config_{};
    T                     Ts_{T{0}};
    T                     t_{T{0}};
    bool                  valid_{false};
};

/**
 * @ingroup discrete_controllers
 * @brief Sum-of-tones multi-sine runtime generator.
 *
 * Emits:
 *     u(t) = sum_i Ai * sin(2*pi*fi*t + phii)
 *
 * Unlike finite-duration signals, MultiSine is open-ended (`done()` is false
 * for valid configuration).
 *
 * @tparam NTones Number of tones.
 * @tparam T      Scalar type.
 */
template<std::size_t NTones, typename T = float>
class MultiSine {
public:
    constexpr MultiSine() = default;

    /**
     * @brief Construct from a multi-sine design payload.
     * @param design Validated design payload.
     * @param Ts     Optional sample period for internal `step()` mode.
     */
    constexpr explicit MultiSine(const design::MultiSineResult<NTones, T>& design, T Ts = T{0})
        : config_(design.config), Ts_(Ts), valid_(design.success) {}

    /**
     * @brief Evaluate multi-sine at absolute time.
     * @param t Time in seconds.
     * @return Sum of all configured tones at time `t`.
     */
    [[nodiscard]] constexpr T step(T t) const {
        if (!valid_) {
            return T{0};
        }

        const T tc = detail::clamp_non_negative(t);
        T       y = T{0};
        for (std::size_t i = 0; i < NTones; ++i) {
            const auto& tone = config_.tones[i];
            const T     omega_t = (T{2} * std::numbers::pi_v<T> * tone.freq_hz * tc) + tone.phase_rad;
            y += tone.amplitude * wet::sin(omega_t);
        }
        return y;
    }

    /**
     * @brief Evaluate multi-sine at internal time and advance by `Ts`.
     * @return Excitation value at current internal time.
     */
    [[nodiscard]] constexpr T step() {
        const T out = step(t_);
        if (Ts_ > T{0}) {
            t_ += Ts_;
        }
        return out;
    }

    /**
     * @brief Query completion at absolute time.
     * @param t Time in seconds (ignored).
     * @return true only when configuration is invalid.
     */
    [[nodiscard]] constexpr bool done(T t) const {
        (void)t;
        return !valid_;
    }

    /**
     * @brief Query completion for internal-time mode.
     * @return true only when configuration is invalid.
     */
    [[nodiscard]] constexpr bool done() const {
        return !valid_;
    }

    /**
     * @brief Reset internal time to zero.
     */
    constexpr void reset() {
        t_ = T{0};
    }

private:
    design::MultiSineConfig<NTones, T> config_{};
    T                                  Ts_{T{0}};
    T                                  t_{T{0}};
    bool                               valid_{false};
};

} // namespace wet
