#pragma once

/**
 * @file harmonic_estimation.hpp
 * @brief Placeholder estimation primitives for harmonic detection workflows.
 */

namespace wet::estimation {

template<typename T = double>
struct HarmonicDetectionResult {
    bool success{false};
};

template<typename T = float>
struct HarmonicTrackerState {
    bool initialized{false};
};

} // namespace wet::estimation
