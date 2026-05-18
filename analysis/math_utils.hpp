/**
 * @file math_utils.hpp
 * @brief CMSIS-DSP-style LUT sin/cos (512-entry table + linear interpolation)
 *
 * Wrappers around the constexpr fast_sin_f32 in math_utils.cpp, with
 * external linkage so they show up in the disassembly for instruction-count
 * comparison against wet_trig and ti_arm_trig.
 */
#pragma once

#include <utility>

namespace lut {

float sin(float x);
float cos(float x);
std::pair<float, float> sincos_rev(float rev);

} // namespace lut
