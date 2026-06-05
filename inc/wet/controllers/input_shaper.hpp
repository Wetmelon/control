#pragma once

/**
 * @file input_shaper.hpp
 * @brief Placeholder header for input shaping feature.
 */

namespace wet {

namespace design {

template<typename T = double>
struct InputShaperResult {
    bool success{false};
};

} // namespace design

template<typename T = float>
struct InputShaperRuntime {
    constexpr void reset() {}
};

} // namespace wet
