#pragma once

/**
 * @file harmonic_suppression.hpp
 * @brief Placeholder header for harmonic detection and suppression feature.
 */

#include "wet/estimation/harmonic_estimation.hpp"

namespace wetmelon::control {

namespace design {

template<typename T = double>
using HarmonicDetectionResult = estimation::HarmonicDetectionResult<T>;

template<typename T = double>
struct HarmonicSuppressorResult {
    HarmonicDetectionResult<T> detection{};
    bool                       success{false};
};

} // namespace design

template<typename T = float>
struct ChatterSuppressionRuntime {
    estimation::HarmonicTrackerState<T> tracker{};

    constexpr void reset() {
        tracker = estimation::HarmonicTrackerState<T>{};
    }
};

} // namespace wetmelon::control
