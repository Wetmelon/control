#pragma once

/**
 * @file recursive_least_squares.hpp
 * @brief Recursive Least Squares (RLS) primitives for online identification.
 */

#include <cstddef>
#include <type_traits>

#include "wet/math/wetmelon_math.hpp"
#include "wet/matrix/matrix.hpp"
#include "wet/matrix/matrix_traits.hpp"

namespace wet::estimation {

/**
 * @brief Common RLS configuration.
 */
template<typename T = double>
struct RecursiveLeastSquaresConfig {
    using value_type = std::remove_const_t<T>;
    using scalar_type = scalar_type_t<value_type>;

    scalar_type lambda{scalar_type{1}};
    scalar_type p0{scalar_type{1}};
    bool        projection_enabled{false};
    value_type  theta_min{};
    value_type  theta_max{};

    [[nodiscard]] constexpr bool valid() const {
        if (lambda <= scalar_type{0}) {
            return false;
        }
        if (lambda > scalar_type{1}) {
            return false;
        }
        if (p0 <= scalar_type{0}) {
            return false;
        }
        return true;
    }
};

/**
 * @brief Scalar RLS runtime state.
 */
template<typename T = float>
struct RecursiveLeastSquaresState {
    using value_type = std::remove_const_t<T>;

    value_type theta{};
    value_type covariance{value_type{1}};
    value_type predicted_output{};
    value_type residual{};
    value_type gain{};
    bool       initialized{false};
};

/**
 * @brief Scalar RLS design payload.
 */
template<typename T = double>
struct RecursiveLeastSquaresResult {
    using value_type = std::remove_const_t<T>;

    RecursiveLeastSquaresConfig<T> config{};
    value_type                     theta0{};
    value_type                     covariance0{value_type{1}};
    bool                           success{false};

    template<typename U>
    [[nodiscard]] constexpr auto as() const {
        using out_t = std::remove_const_t<U>;
        return RecursiveLeastSquaresResult<out_t>{
            RecursiveLeastSquaresConfig<out_t>{
                static_cast<scalar_type_t<out_t>>(config.lambda),
                static_cast<scalar_type_t<out_t>>(config.p0),
                config.projection_enabled,
                static_cast<out_t>(config.theta_min),
                static_cast<out_t>(config.theta_max),
            },
            static_cast<out_t>(theta0),
            static_cast<out_t>(covariance0),
            success,
        };
    }
};

/**
 * @brief Vector RLS runtime state for N parameters.
 */
template<size_t NP, typename T = float>
struct RecursiveLeastSquaresVectorState {
    using value_type = std::remove_const_t<T>;

    ColVec<NP, value_type>     theta{};
    Matrix<NP, NP, value_type> covariance{Matrix<NP, NP, value_type>::identity()};
    value_type                 predicted_output{};
    value_type                 residual{};
    ColVec<NP, value_type>     gain{};
    bool                       initialized{false};
};

/**
 * @brief Vector RLS design payload for N parameters.
 */
template<size_t NP, typename T = double>
struct RecursiveLeastSquaresVectorResult {
    using value_type = std::remove_const_t<T>;

    RecursiveLeastSquaresConfig<T> config{};
    ColVec<NP, value_type>         theta0{};
    Matrix<NP, NP, value_type>     covariance0{Matrix<NP, NP, value_type>::identity()};
    bool                           success{false};

    template<typename U>
    [[nodiscard]] constexpr auto as() const {
        using out_t = std::remove_const_t<U>;
        return RecursiveLeastSquaresVectorResult<NP, out_t>{
            RecursiveLeastSquaresConfig<out_t>{
                static_cast<scalar_type_t<out_t>>(config.lambda),
                static_cast<scalar_type_t<out_t>>(config.p0),
                config.projection_enabled,
                static_cast<out_t>(config.theta_min),
                static_cast<out_t>(config.theta_max),
            },
            theta0.template as<out_t>(),
            covariance0.template as<out_t>(),
            success,
        };
    }
};

/**
 * @brief Build scalar RLS design payload.
 */
template<typename T = double>
[[nodiscard]] constexpr RecursiveLeastSquaresResult<T> synthesize_recursive_least_squares(
    const RecursiveLeastSquaresConfig<T>& config,
    T                                     theta0 = T{}
) {
    RecursiveLeastSquaresResult<T> result{};
    result.config = config;
    result.theta0 = theta0;
    result.covariance0 = static_cast<T>(config.p0);
    result.success = config.valid();
    return result;
}

/**
 * @brief Build vector RLS design payload.
 */
template<size_t NP, typename T = double>
[[nodiscard]] constexpr RecursiveLeastSquaresVectorResult<NP, T> synthesize_recursive_least_squares_vector(
    const RecursiveLeastSquaresConfig<T>& config,
    const ColVec<NP, T>&                  theta0 = ColVec<NP, T>{}
) {
    RecursiveLeastSquaresVectorResult<NP, T> result{};
    result.config = config;
    result.theta0 = theta0;
    result.success = config.valid();
    if (!result.success) {
        return result;
    }

    result.covariance0 = Matrix<NP, NP, T>::identity() * static_cast<T>(config.p0);
    return result;
}

/**
 * @brief Scalar runtime RLS estimator.
 */
template<typename T = float>
class RecursiveLeastSquaresEstimator {
public:
    using value_type = std::remove_const_t<T>;
    using scalar_type = scalar_type_t<value_type>;

    constexpr RecursiveLeastSquaresEstimator() = default;

    constexpr explicit RecursiveLeastSquaresEstimator(const RecursiveLeastSquaresConfig<value_type>& config)
        : config_(config) {
        state_.covariance = static_cast<value_type>(config_.p0);
    }

    constexpr explicit RecursiveLeastSquaresEstimator(const RecursiveLeastSquaresResult<value_type>& design)
        : config_(design.config), valid_(design.success) {
        state_.theta = design.theta0;
        state_.covariance = design.covariance0;
    }

    /**
     * @brief Update estimate using one regression sample y = phi * theta + noise.
     */
    [[nodiscard]] constexpr bool update(value_type phi, value_type y) {
        if (!config_.valid()) {
            valid_ = false;
            return false;
        }

        const value_type denom = static_cast<value_type>(config_.lambda) + phi * state_.covariance * phi;
        if (wet::abs(denom) <= default_tol<value_type>()) {
            valid_ = false;
            return false;
        }

        state_.gain = (state_.covariance * phi) / denom;
        state_.predicted_output = phi * state_.theta;
        state_.residual = y - state_.predicted_output;
        state_.theta = state_.theta + state_.gain * state_.residual;

        const value_type one_over_lambda = value_type{1} / static_cast<value_type>(config_.lambda);
        state_.covariance = (state_.covariance - state_.gain * phi * state_.covariance) * one_over_lambda;

        if constexpr (!is_complex_v<value_type>) {
            if (config_.projection_enabled) {
                if (state_.theta < config_.theta_min) {
                    state_.theta = config_.theta_min;
                }
                if (state_.theta > config_.theta_max) {
                    state_.theta = config_.theta_max;
                }
            }

            if (state_.covariance < value_type{0}) {
                state_.covariance = value_type{0};
            }
        }

        state_.initialized = true;
        valid_ = true;
        return true;
    }

    [[nodiscard]] constexpr value_type predict(value_type phi) const {
        return phi * state_.theta;
    }

    constexpr void reset(value_type theta0 = value_type{}) {
        state_ = RecursiveLeastSquaresState<value_type>{};
        state_.theta = theta0;
        state_.covariance = static_cast<value_type>(config_.p0);
        valid_ = true;
    }

    [[nodiscard]] constexpr const auto& config() const { return config_; }
    [[nodiscard]] constexpr const auto& state() const { return state_; }
    [[nodiscard]] constexpr bool        valid() const { return valid_; }

private:
    RecursiveLeastSquaresConfig<value_type> config_{};
    RecursiveLeastSquaresState<value_type>  state_{};
    bool                                    valid_{true};
};

/**
 * @brief Vector runtime RLS estimator (NP parameters).
 */
template<size_t NP, typename T = float>
class RecursiveLeastSquaresVectorEstimator {
public:
    using value_type = std::remove_const_t<T>;

    constexpr RecursiveLeastSquaresVectorEstimator() = default;

    constexpr explicit RecursiveLeastSquaresVectorEstimator(const RecursiveLeastSquaresConfig<value_type>& config)
        : config_(config) {
        state_.covariance = Matrix<NP, NP, value_type>::identity() * static_cast<value_type>(config_.p0);
    }

    constexpr explicit RecursiveLeastSquaresVectorEstimator(const RecursiveLeastSquaresVectorResult<NP, value_type>& design)
        : config_(design.config), valid_(design.success) {
        state_.theta = design.theta0;
        state_.covariance = design.covariance0;
    }

    /**
     * @brief Update estimate using one sample y = phi^H * theta + noise.
     */
    [[nodiscard]] constexpr bool update(const ColVec<NP, value_type>& phi, value_type y) {
        if (!config_.valid()) {
            valid_ = false;
            return false;
        }

        const ColVec<NP, value_type> p_phi = state_.covariance * phi;
        const value_type             denom = static_cast<value_type>(config_.lambda) + dot(phi, p_phi);
        if (wet::abs(denom) <= default_tol<value_type>()) {
            valid_ = false;
            return false;
        }

        state_.gain = p_phi / denom;
        state_.predicted_output = dot(phi, state_.theta);
        state_.residual = y - state_.predicted_output;
        state_.theta += state_.gain * state_.residual;

        const auto phi_h = phi.conjugate_transpose();
        const auto one_over_lambda = value_type{1} / static_cast<value_type>(config_.lambda);
        state_.covariance = (state_.covariance - (state_.gain * phi_h * state_.covariance)) * one_over_lambda;

        if constexpr (!is_complex_v<value_type>) {
            if (config_.projection_enabled) {
                for (size_t i = 0; i < NP; ++i) {
                    if (state_.theta[i] < config_.theta_min) {
                        state_.theta[i] = config_.theta_min;
                    }
                    if (state_.theta[i] > config_.theta_max) {
                        state_.theta[i] = config_.theta_max;
                    }
                }
            }
        }

        // Keep covariance numerically symmetric/Hermitian after finite precision updates.
        state_.covariance = (state_.covariance + state_.covariance.conjugate_transpose()) * value_type{0.5};
        state_.initialized = true;
        valid_ = true;
        return true;
    }

    [[nodiscard]] constexpr value_type predict(const ColVec<NP, value_type>& phi) const {
        return dot(phi, state_.theta);
    }

    constexpr void reset(const ColVec<NP, value_type>& theta0 = ColVec<NP, value_type>{}) {
        state_ = RecursiveLeastSquaresVectorState<NP, value_type>{};
        state_.theta = theta0;
        state_.covariance = Matrix<NP, NP, value_type>::identity() * static_cast<value_type>(config_.p0);
        valid_ = true;
    }

    [[nodiscard]] constexpr const auto& config() const { return config_; }
    [[nodiscard]] constexpr const auto& state() const { return state_; }
    [[nodiscard]] constexpr bool        valid() const { return valid_; }

private:
    RecursiveLeastSquaresConfig<value_type>          config_{};
    RecursiveLeastSquaresVectorState<NP, value_type> state_{};
    bool                                             valid_{true};
};

} // namespace wet::estimation
