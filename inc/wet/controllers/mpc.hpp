#pragma once

/**
 * @file mpc.hpp
 * @brief Placeholder header for constrained MPC roadmap feature.
 */

namespace wetmelon::control {

namespace design {

template<typename T = double>
struct MPCResult {
    bool success{false};
};

template<typename T = double>
struct MPCArtifacts {
    MPCResult<T> design{};
    bool         success{false};
};

} // namespace design

template<typename T = float>
struct MPCRuntimeBundle {
    constexpr void reset() {}
};

} // namespace wetmelon::control
