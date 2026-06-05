#pragma once

/**
 * @file dob.hpp
 * @brief Placeholder header for Disturbance Observer feature.
 */

#include "wet/estimation/disturbance_observer.hpp"

namespace wet {

namespace design {

template<typename T = double>
struct DOBResult {
    estimation::DisturbanceObserverResult<T> observer{};
    bool                                     success{false};
};

template<typename T = double>
struct DOBArtifacts {
    DOBResult<T> design{};
    bool         success{false};
};

} // namespace design

template<typename T = float>
struct DOBRuntimeBundle {
    estimation::DisturbanceObserverState<T> observer{};

    constexpr void reset() {
        observer = estimation::DisturbanceObserverState<T>{};
    }
};

} // namespace wet
