#pragma once

/**
 * @file hinf.hpp
 * @brief Placeholder header for H-infinity roadmap feature.
 */

namespace wetmelon::control {

namespace design {

template<typename T = double>
struct HInfResult {
    bool success{false};
};

template<typename T = double>
struct HInfArtifacts {
    HInfResult<T> design{};
    bool          success{false};
};

} // namespace design

template<typename T = float>
struct HInfRuntimeBundle {
    constexpr void reset() {}
};

} // namespace wetmelon::control
