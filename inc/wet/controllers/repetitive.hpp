#pragma once

/**
 * @file repetitive.hpp
 * @brief Plug-in repetitive controller — internal-model rejection of a periodic
 *        reference/disturbance and *all* its harmonics with one delay loop.
 *
 * Repetitive control embeds the internal model of a periodic signal: a positive-
 * feedback delay line of one fundamental period N = round(fs / f0). That single
 * model puts (near-)infinite loop gain at the fundamental *and every harmonic*
 * simultaneously — the reason one repetitive block out-rejects a stack of
 * individual resonators. Targets: grid-tied inverters (track/clean a sinusoidal
 * reference, reject periodic load), precision motion with periodic trajectories,
 * rotating-machinery ripple.
 *
 * It is a *plug-in*: it sits alongside any base controller and adds a correction.
 *
 *     e   = r - y;                 // tracking error
 *     u   = base.control(r, y) + rc.step(e);
 *
 * Standard discrete realization (scalar robustness filter Q, integer phase lead m):
 *
 *     w[k]    = Q · w[k-N] + e[k]       (internal-model memory, length-N buffer)
 *     u_rc[k] = k_rc · w[k - N + m]     (the learned, phase-advanced correction)
 *
 * Q ∈ (0,1] trades steady-state rejection for robustness (rolls the model off the
 * high harmonics where plant phase is uncertain); m pre-applies the correction a
 * few samples early to offset plant transport lag.
 *
 * @see S. Hara, Y. Yamamoto, T. Omata, M. Nakano, "Repetitive Control System: A
 *      New Type Servo System for Periodic Exogenous Signals," IEEE TAC 33(7),
 *      1988, https://doi.org/10.1109/9.1274
 */

#include <cstddef>

#include "wet/backend.hpp"              // wet::array
#include "wet/matrix/matrix_traits.hpp" // scalar_type_t

namespace wet {

namespace design {

/**
 * @brief Repetitive-controller tuning + period.
 * @tparam T Scalar type
 */
template<typename T = double>
struct RepetitiveConfig {
    using value_type = std::remove_const_t<T>;
    using scalar = scalar_type_t<value_type>;

    size_t period{0};           //!< N = round(fs / f0), samples per fundamental period
    scalar gain{scalar{1}};     //!< k_rc, repetitive (learning) gain — 0 < k_rc ≲ 2 (plant-dependent)
    scalar q_filter{scalar{1}}; //!< Q ∈ (0,1], robustness roll-off (1 = full rejection, no roll-off)
    size_t lead{0};             //!< m, phase-lead samples to offset plant lag (0 ≤ m < N)

    [[nodiscard]] constexpr bool valid() const {
        if (period < 2) {
            return false;
        }
        if (gain <= scalar{0} || gain > scalar{2}) {
            return false;
        }
        if (q_filter <= scalar{0} || q_filter > scalar{1}) {
            return false;
        }
        if (lead >= period) {
            return false;
        }
        return true;
    }
};

/**
 * @brief Design result for the repetitive controller.
 */
template<typename T = double>
struct RepetitiveResult {
    using value_type = std::remove_const_t<T>;

    RepetitiveConfig<value_type> config{};
    bool                         success{false};

    template<typename U>
    [[nodiscard]] constexpr RepetitiveResult<std::remove_const_t<U>> as() const {
        using out_t = std::remove_const_t<U>;
        return RepetitiveResult<out_t>{
            RepetitiveConfig<out_t>{
                config.period,
                static_cast<scalar_type_t<out_t>>(config.gain),
                static_cast<scalar_type_t<out_t>>(config.q_filter),
                config.lead,
            },
            success,
        };
    }
};

/**
 * @brief Synthesize a repetitive controller for fundamental @p f0_hz at rate @p fs_hz.
 *
 * Computes the period N = round(fs / f0) and validates the tuning. Returns
 * `success = false` (rather than asserting) on an invalid spec so it composes in
 * `constexpr` design code.
 *
 * @param fs_hz Sample rate [Hz]
 * @param f0_hz Fundamental frequency to reject/track [Hz]
 * @param gain  Repetitive (learning) gain k_rc (default 1)
 * @param q     Robustness filter Q ∈ (0,1] (default 1)
 * @param lead  Phase-lead samples m (default 0)
 */
template<typename T = double>
[[nodiscard]] constexpr RepetitiveResult<T> synthesize_repetitive(
    T      fs_hz,
    T      f0_hz,
    T      gain = T{1},
    T      q = T{1},
    size_t lead = 0
) {
    RepetitiveResult<T> result{};
    if (fs_hz <= T{0} || f0_hz <= T{0} || f0_hz >= fs_hz) {
        return result; // success stays false
    }
    const T      ratio = fs_hz / f0_hz;
    const size_t period = static_cast<size_t>(ratio + T{0.5}); // round to nearest

    result.config = RepetitiveConfig<T>{
        period,
        static_cast<scalar_type_t<T>>(gain),
        static_cast<scalar_type_t<T>>(q),
        lead,
    };
    result.success = result.config.valid();
    return result;
}

} // namespace design

/**
 * @ingroup controllers
 * @brief Plug-in repetitive controller runtime (fixed-size internal model).
 *
 * `MaxPeriod` bounds the internal-model buffer at compile time (allocation-free);
 * the active period N ≤ MaxPeriod is set from the design. `step(error)` returns
 * the repetitive correction to add to the base controller's command.
 *
 * @tparam MaxPeriod Buffer capacity (largest supported N = round(fs/f0))
 * @tparam T         Scalar type (float or double)
 */
template<size_t MaxPeriod, typename T = float>
class RepetitiveController {
public:
    using value_type = std::remove_const_t<T>;
    using scalar = scalar_type_t<value_type>;
    static_assert(MaxPeriod >= 2, "RepetitiveController needs MaxPeriod >= 2");

    constexpr RepetitiveController() = default;

    constexpr explicit RepetitiveController(const design::RepetitiveConfig<value_type>& config)
        : config_(config), valid_(config.valid()) {}

    constexpr explicit RepetitiveController(const design::RepetitiveResult<value_type>& design)
        : config_(design.config), valid_(design.success) {}

    /**
     * @brief Advance one tick with the current tracking error; returns the
     *        repetitive correction u_rc to add to the base command.
     */
    [[nodiscard]] constexpr value_type step(value_type error) {
        if (!valid_ || config_.period < 2 || config_.period > MaxPeriod) {
            return value_type{0};
        }
        const size_t n = config_.period;

        // w[k-N]: oldest sample in the length-N window ending at the write head.
        const size_t read_old = (write_idx_ + MaxPeriod - n) % MaxPeriod;
        // w[k-N+m]: phase-advanced output tap (m < N ⇒ already-stored sample).
        const size_t read_out = (write_idx_ + MaxPeriod - n + config_.lead) % MaxPeriod;

        const value_type w_kN = mem_[read_old];
        const value_type u_rc = static_cast<value_type>(config_.gain) * mem_[read_out];

        // Internal-model update w[k] = Q·w[k-N] + e[k], stored at the head.
        mem_[write_idx_] = (static_cast<value_type>(config_.q_filter) * w_kN) + error;
        write_idx_ = (write_idx_ + 1) % MaxPeriod;

        return u_rc;
    }

    constexpr void reset() {
        mem_ = {};
        write_idx_ = 0;
    }

    [[nodiscard]] constexpr const auto& config() const { return config_; }
    [[nodiscard]] constexpr bool        valid() const { return valid_; }

private:
    design::RepetitiveConfig<value_type> config_{};
    wet::array<value_type, MaxPeriod>    mem_{};
    size_t                               write_idx_{0};
    bool                                 valid_{false};
};

} // namespace wet
