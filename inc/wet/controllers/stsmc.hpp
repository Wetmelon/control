#pragma once

/**
 * @file stsmc.hpp
 * @brief Placeholder header for super-twisting SMC roadmap feature.
 */

namespace wetmelon::control {

namespace design {

template<typename T = double>
struct STSMCResult {
    bool success{false};
};

template<typename T = double>
struct STSMCArtifacts {
    STSMCResult<T> design{};
    bool           success{false};
};

} // namespace design

template<typename T = float>
struct STSMCRuntimeBundle {
    constexpr void reset() {}
};

} // namespace wetmelon::control
