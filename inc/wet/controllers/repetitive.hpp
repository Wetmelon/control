#pragma once

/**
 * @file repetitive.hpp
 * @brief Placeholder header for repetitive control feature.
 */

namespace wet {

namespace design {

template<typename T = double>
struct RepetitiveResult {
    bool success{false};
};

template<typename T = double>
struct RepetitiveArtifacts {
    RepetitiveResult<T> design{};
    bool                success{false};
};

} // namespace design

template<typename T = float>
struct RepetitiveRuntimeBundle {
    constexpr void reset() {}
};

} // namespace wet
