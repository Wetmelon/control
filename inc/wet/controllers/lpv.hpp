#pragma once

/**
 * @file lpv.hpp
 * @brief Placeholder header for LPV gain-scheduled feature.
 */

namespace wet {

namespace design {

template<typename T = double>
struct LPVScheduleResult {
    bool success{false};
};

template<typename T = double>
struct LPVArtifacts {
    LPVScheduleResult<T> schedule{};
    bool                 success{false};
};

} // namespace design

template<typename T = float>
struct LPVRuntimeBundle {
    constexpr void reset() {}
};

} // namespace wet
