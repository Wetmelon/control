/**
 * @file example_math_backend.cpp
 * @brief Pluggable math backend example
 *
 * Demonstrates how to select and extend a math backend for embedded deployment.
 * The library's wet:: math functions (sin, cos, sqrt, etc.) dispatch through
 * MathBackend<T> at runtime. By providing a wet_profile.hpp anywhere in your
 * project's include path, you route math operations to hardware-accelerated
 * implementations without changing any library or application code.
 *
 * The compile-time path (consteval design functions, static_assert checks)
 * always uses the built-in constexpr series expansions regardless of backend.
 * Backends only affect runtime execution.
 *
 * Backend selection is done via a user-created, macro-only wet_profile.hpp (see
 * wet/config.hpp for the full surface):
 *
 * @code
 * // wet_profile.hpp  — place anywhere in your project's include path (macros only)
 *
 * // Option 1: std:: backend (desktop / host builds — the default, no macro needed)
 *
 * // Option 2: built-in fast float math (wet/math/trig.hpp)
 * #define WET_MATH_BACKEND_WET
 *
 * // Option 3: custom platform backend — point at a header that defines it:
 * #define WET_MATH_BACKEND_HEADER "ti_arm_backend.hpp"
 *
 * // ...where ti_arm_backend.hpp inherits StdMathFallback<T> and overrides only
 * // the functions the platform provides:
 * namespace wet {
 * template<>
 * struct MathBackend<float> : StdMathFallback<float> {
 *     static float sin(float x)  { return my_platform_sinf(x); }
 *     static float cos(float x)  { return my_platform_cosf(x); }
 *     static float sqrt(float x) { return my_platform_sqrtf(x); }
 *     // Unoverridden functions (cbrt, exp, log, etc.) fall through to std::
 * };
 * } // namespace wet
 * @endcode
 *
 * If no wet_profile.hpp is found, the library uses the std:: backend and emits a
 * one-time #warning so the choice is never silent.
 */

// This example uses the std:: backend via examples/wet_profile.hpp.
#include <cstdio>
#include <numbers>

#include "wet/matrix/colvec.hpp"
#include "wet/power/transforms.hpp"

using namespace wet;

int main() {
    // Park/Clarke transforms — these call wet::sin and wet::cos at runtime,
    // so they dispatch through MathBackend<float>. With the TI ARM backend
    // (#include "backends/ti_arm_backend.hpp"), these route to fast polynomial
    // approximations optimized for ARMv7 cores.
    float theta = std::numbers::pi_v<float> / 4.0f; // 45° rotor angle

    // Simulate three-phase motor currents
    wet::ColVec<3, float> iabc = {1.0f, -0.5f, -0.5f};

    // ABC → αβ (Clarke) → dq (Park)
    auto [alpha, beta] = clarke_transform(iabc);
    auto [id, iq] = park_transform({alpha, beta}, theta);

    // dq → αβ (inverse Park) → ABC (inverse Clarke)
    auto [alpha2, beta2] = inverse_park_transform({id, iq}, theta);
    auto iabc2 = inverse_clarke_transform<float>({alpha2, beta2});

    std::printf("\n=== FOC transforms (uses wet::sin, wet::cos → MathBackend) ===\n");
    std::printf("ABC in:  [%.4f, %.4f, %.4f]\n", double(iabc[0]), double(iabc[1]), double(iabc[2]));
    std::printf("αβ:      [%.4f, %.4f]\n", double(alpha), double(beta));
    std::printf("dq:      [%.4f, %.4f]\n", double(id), double(iq));
    std::printf("ABC out: [%.4f, %.4f, %.4f]\n", double(iabc2[0]), double(iabc2[1]), double(iabc2[2]));

    return 0;
}
