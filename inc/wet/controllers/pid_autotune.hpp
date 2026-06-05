#pragma once

/**
 * @file pid_autotune.hpp
 * @brief Placeholder header for model-agnostic adaptive PID autotuning.
 */

#include <concepts>

#include "wet/estimation/identification.hpp"
#include "wet/estimation/recursive_least_squares.hpp"

namespace wet {

namespace design {

template<typename T = double>
using DefaultPIDAutotuneModel = estimation::GreyBoxIdentificationResult<T>;

template<typename T = double>
struct PIDGains {
    using scalar_type = T;

    T kp{};
    T ki{};
    T kd{};
};

template<typename Model>
concept ModelLike = estimation::IdentifiedModelLike<Model>;

template<typename Model, typename Fitness>
concept PIDFitnessLike = ModelLike<Model> && requires(const Fitness& fitness, const Model& model, const PIDGains<typename Model::scalar_type>& gains) {
    { fitness(model, gains) } -> std::convertible_to<typename Model::scalar_type>;
};

template<ModelLike Model>
struct PIDAutotuneResult {
    using scalar_type = typename Model::scalar_type;

    Model                                                selected_model{};
    estimation::ValidationResult<scalar_type>            validation{};
    estimation::RecursiveLeastSquaresResult<scalar_type> rls{};
    PIDGains<scalar_type>                                gains{};
    scalar_type                                          best_fitness{};
    bool                                                 has_validation{false};
    bool                                                 has_rls{false};
    bool                                                 success{false};
};

template<ModelLike Model, typename Fitness>
    requires PIDFitnessLike<Model, Fitness>
struct PIDAutotuneProblem {
    Model   model{};
    Fitness fitness{};
};

template<ModelLike Model, typename Fitness>
    requires PIDFitnessLike<Model, Fitness>
[[nodiscard]] constexpr auto score_candidate(
    const Model&                                 model,
    const PIDGains<typename Model::scalar_type>& gains,
    const Fitness&                               fitness
) -> typename Model::scalar_type {
    return static_cast<typename Model::scalar_type>(fitness(model, gains));
};

} // namespace design

template<typename T = float>
struct PIDAutotuneRuntime {
    estimation::RecursiveLeastSquaresState<T> rls{};

    constexpr void reset() {
        rls = estimation::RecursiveLeastSquaresState<T>{};
    }
};

} // namespace wet
