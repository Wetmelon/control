#pragma once

/**
 * @file disturbance_observer.hpp
 * @brief Disturbance-observer estimation primitives and lightweight runtime.
 */

#include <type_traits>

#include "wet/math/wetmelon_math.hpp"
#include "wet/matrix/matrix_traits.hpp"

namespace wet::estimation {

/**
 * @brief Configuration for a first-order disturbance observer.
 *
 * The runtime update is:
 *
 *     innovation = y_measured - y_predicted
 *     d_hat[k+1] = (1 - leak) * d_hat[k] + gain * innovation
 */
template<typename T = double>
struct DisturbanceObserverConfig {
    using value_type = std::remove_const_t<T>;
    using scalar_type = scalar_type_t<value_type>;

    scalar_type gain{scalar_type{0.1}};
    scalar_type leak{scalar_type{0}};
    scalar_type innovation_deadband{scalar_type{0}};
    scalar_type max_disturbance_magnitude{scalar_type{0}};
    bool        clamp_enabled{false};

    [[nodiscard]] constexpr bool valid() const {
        if (gain < scalar_type{0}) {
            return false;
        }
        if (gain > scalar_type{1}) {
            return false;
        }
        if (leak < scalar_type{0}) {
            return false;
        }
        if (leak >= scalar_type{1}) {
            return false;
        }
        if (innovation_deadband < scalar_type{0}) {
            return false;
        }
        if (max_disturbance_magnitude < scalar_type{0}) {
            return false;
        }
        return true;
    }
};

template<typename T = double>
struct DisturbanceObserverResult {
    using value_type = std::remove_const_t<T>;
    using scalar_type = scalar_type_t<value_type>;

    DisturbanceObserverConfig<value_type> config{};
    value_type                            steady_state_gain{};
    bool                                  success{false};

    template<typename U>
    [[nodiscard]] constexpr auto as() const {
        using out_t = std::remove_const_t<U>;
        return DisturbanceObserverResult<out_t>{
            DisturbanceObserverConfig<out_t>{
                static_cast<scalar_type_t<out_t>>(config.gain),
                static_cast<scalar_type_t<out_t>>(config.leak),
                static_cast<scalar_type_t<out_t>>(config.innovation_deadband),
                static_cast<scalar_type_t<out_t>>(config.max_disturbance_magnitude),
                config.clamp_enabled,
            },
            static_cast<out_t>(steady_state_gain),
            success,
        };
    }
};

template<typename T = float>
struct DisturbanceObserverState {
    using value_type = std::remove_const_t<T>;

    value_type disturbance_hat{};
    value_type innovation{};
    bool       initialized{false};
};

namespace detail {

template<typename T>
[[nodiscard]] constexpr T limit_magnitude(const T& value, scalar_type_t<T> max_magnitude) {
    if (max_magnitude <= scalar_type_t<T>{0}) {
        return value;
    }

    const auto magnitude = wet::abs(value);
    if (magnitude <= max_magnitude) {
        return value;
    }
    if (magnitude <= default_tol<T>()) {
        return T{};
    }

    const auto scale = max_magnitude / magnitude;
    return value * scale;
}

} // namespace detail

/**
 * @brief Validate and package DOB configuration into a runtime-ready design result.
 */
template<typename T = double>
[[nodiscard]] constexpr DisturbanceObserverResult<T> synthesize_disturbance_observer(
    const DisturbanceObserverConfig<T>& config
) {
    DisturbanceObserverResult<T> result{};
    result.config = config;
    result.success = config.valid();

    if (!result.success) {
        return result;
    }

    const auto one_minus_leak = scalar_type_t<T>{1} - config.leak;
    if (one_minus_leak <= default_tol<T>()) {
        result.steady_state_gain = T{};
        return result;
    }

    // Constant-innovation steady-state gain: d_hat_ss = gain / leak * innovation.
    if (config.leak <= default_tol<T>()) {
        result.steady_state_gain = T{};
    } else {
        result.steady_state_gain = static_cast<T>(config.gain / config.leak);
    }
    return result;
}

/**
 * @brief Lightweight SISO disturbance observer runtime.
 */
template<typename T = float>
class DisturbanceObserver {
public:
    using value_type = std::remove_const_t<T>;
    using scalar_type = scalar_type_t<value_type>;

    constexpr DisturbanceObserver() = default;

    constexpr explicit DisturbanceObserver(const DisturbanceObserverConfig<value_type>& config)
        : config_(config) {}

    constexpr explicit DisturbanceObserver(const DisturbanceObserverResult<value_type>& design)
        : config_(design.config), valid_(design.success) {}

    /**
     * @brief Update disturbance estimate from predicted and measured outputs.
     * @return true when update succeeded.
     */
    [[nodiscard]] constexpr bool update(value_type y_predicted, value_type y_measured) {
        if (!config_.valid()) {
            valid_ = false;
            return false;
        }

        const scalar_type innovation_mag_deadband = config_.innovation_deadband;
        value_type        innovation = y_measured - y_predicted;

        if (innovation_mag_deadband > scalar_type{0} && wet::abs(innovation) < innovation_mag_deadband) {
            innovation = value_type{};
        }

        const scalar_type alpha = scalar_type{1} - config_.leak;
        state_.disturbance_hat = (static_cast<value_type>(alpha) * state_.disturbance_hat) + (static_cast<value_type>(config_.gain) * innovation);

        if (config_.clamp_enabled && config_.max_disturbance_magnitude > scalar_type{0}) {
            state_.disturbance_hat = detail::limit_magnitude(state_.disturbance_hat, config_.max_disturbance_magnitude);
        }

        state_.innovation = innovation;
        state_.initialized = true;
        valid_ = true;
        return true;
    }

    /**
     * @brief Disturbance-compensated command (u = u_nominal - d_hat).
     */
    [[nodiscard]] constexpr value_type compensate(const value_type& u_nominal) const {
        return u_nominal - state_.disturbance_hat;
    }

    constexpr void reset() {
        state_ = DisturbanceObserverState<value_type>{};
        valid_ = true;
    }

    [[nodiscard]] constexpr const auto& config() const { return config_; }
    [[nodiscard]] constexpr const auto& state() const { return state_; }
    [[nodiscard]] constexpr bool        valid() const { return valid_; }

private:
    DisturbanceObserverConfig<value_type> config_{};
    DisturbanceObserverState<value_type>  state_{};
    bool                                  valid_{true};
};

} // namespace wet::estimation
