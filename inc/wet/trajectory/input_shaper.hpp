#pragma once

/**
 * @file input_shaper.hpp
 * @brief Input shaping — feedforward command prefilters that cancel residual
 *        vibration of a lightly-damped mode (ZV / ZVD / ZVDD / EI shapers).
 *
 * A shaper convolves the reference command with a short sequence of impulses
 * timed to the plant's vibratory mode so the move finishes with (near) zero
 * residual oscillation — pure feedforward, no sensor or estimator. The classic
 * use is point-to-point motion of a flexible structure: 3D-printer/CNC gantries,
 * pick-and-place arms, crane payloads, scanning stages.
 *
 * For a second-order mode (natural frequency ωn, damping ζ):
 *
 *     ωd = ωn·√(1−ζ²),   Td = 2π/ωd,   K = exp(−ζπ/√(1−ζ²))
 *
 * The shaper impulses sit at multiples of Td/2 with amplitudes chosen so the
 * mode's residual vibration is zero at ωn:
 *
 *   ZV   (2 impulses): A ∝ [1, K]              — shortest (Td/2 delay), least robust
 *   ZVD  (3 impulses): A ∝ [1, 2K, K²]         — zero vibration *and* its derivative
 *   ZVDD (4 impulses): A ∝ [1, 3K, 3K², K³]    — even more robust to ωn error
 *   EI   (3 impulses): allows a small tolerable vibration V for wider insensitivity
 *
 * Amplitudes are normalized to sum to 1, so the shaper has unity DC gain — a
 * constant command passes through unchanged, only transients are shaped. The cost
 * is a command delay equal to the last impulse time (Td/2 … 3Td/2).
 *
 * @note ZV/ZVD/ZVDD are exact for any ζ ∈ [0,1). The EI amplitudes here are the
 *       textbook *undamped* closed form (with damped timing) — exact at ζ = 0 and
 *       a good approximation for light damping; use ZVD/ZVDD for robust damped
 *       designs.
 *
 * @see "Preshaping Command Inputs to Reduce System Vibration" (Singer & Seering,
 *      ASME JDSMC 1990), https://doi.org/10.1115/1.2894142
 * @see "Extra-Insensitive Input Shapers..." (Singhose et al., ASME JMD 1994),
 *      https://doi.org/10.1115/1.2919428
 */

#include <cstddef>
#include <type_traits>

#include "wet/backend.hpp" // wet::array, wet::numbers
#include "wet/math/math.hpp"
#include "wet/matrix/matrix_traits.hpp" // default_tol

namespace wet {

/// Input-shaper family.
enum class ShaperType {
    ZV,   ///< Zero-vibration (2 impulses)
    ZVD,  ///< Zero-vibration-and-derivative (3 impulses)
    ZVDD, ///< Zero-vibration + 2 derivatives (4 impulses)
    EI    ///< Extra-insensitive (3 impulses, tolerable-vibration V)
};

namespace design {

/**
 * @brief Input-shaper design result: impulse amplitudes and sample delays.
 * @tparam T Scalar type
 */
template<typename T = double>
struct InputShaperResult {

    static constexpr size_t MaxImpulses = 4;

    wet::array<T, MaxImpulses>      amplitudes{}; ///< Impulse amplitudes (sum to 1)
    wet::array<T, MaxImpulses>      times{};      ///< Impulse times [s]
    wet::array<size_t, MaxImpulses> delays{};     ///< Sample delays = round(times/Ts)

    size_t count{0}; ///< Number of active impulses
    T      Ts{T{0}};
    bool   success{false};

    /// Last impulse delay in samples (the command latency the shaper adds).
    [[nodiscard]] constexpr size_t max_delay() const { return (count == 0) ? 0 : delays[count - 1]; }

    template<typename U>
    [[nodiscard]] constexpr InputShaperResult<std::remove_const_t<U>> as() const {
        InputShaperResult<std::remove_const_t<U>> out{};
        using O = std::remove_const_t<U>;
        for (size_t i = 0; i < MaxImpulses; ++i) {
            out.amplitudes[i] = static_cast<O>(amplitudes[i]);
            out.times[i] = static_cast<O>(times[i]);
            out.delays[i] = delays[i];
        }
        out.count = count;
        out.Ts = static_cast<O>(Ts);
        out.success = success;
        return out;
    }
};

/**
 * @brief Synthesize an input shaper for a second-order mode.
 *
 * @tparam T Scalar type
 * @param natural_frequency_hz Mode natural frequency fn [Hz] (> 0)
 * @param damping_ratio        Mode damping ζ ∈ [0, 1)
 * @param Ts                   Command sample time [s] (> 0)
 * @param type                 Shaper family (ZV / ZVD / ZVDD / EI)
 * @param ei_tolerance         EI tolerable residual vibration V (0 < V < 1, EI only)
 * @return InputShaperResult with normalized amplitudes and integer sample delays
 */
template<typename T = double>
[[nodiscard]] constexpr InputShaperResult<T> synthesize_input_shaper(
    T          natural_frequency_hz,
    T          damping_ratio,
    T          Ts,
    ShaperType type,
    T          ei_tolerance = T{0.05}
) {
    InputShaperResult<T> result{};
    if (natural_frequency_hz <= T{0} || damping_ratio < T{0} || damping_ratio >= T{1} || Ts <= T{0}) {
        return result;
    }

    const T wn = T{2} * wet::numbers::pi_v<T> * natural_frequency_hz;
    const T root = wet::sqrt(T{1} - (damping_ratio * damping_ratio));
    const T wd = wn * root;
    const T half_period = wet::numbers::pi_v<T> / wd; // Td/2
    const T K = wet::exp(-damping_ratio * wet::numbers::pi_v<T> / root);

    switch (type) {
        case ShaperType::ZV: {
            result.count = 2;
            result.amplitudes = {T{1}, K, T{0}, T{0}};
            result.times = {T{0}, half_period, T{0}, T{0}};
            break;
        }
        case ShaperType::ZVD: {
            result.count = 3;
            result.amplitudes = {T{1}, T{2} * K, K * K, T{0}};
            result.times = {T{0}, half_period, T{2} * half_period, T{0}};
            break;
        }
        case ShaperType::ZVDD: {
            result.count = 4;
            result.amplitudes = {T{1}, T{3} * K, T{3} * K * K, K * K * K};
            result.times = {T{0}, half_period, T{2} * half_period, T{3} * half_period};
            break;
        }
        case ShaperType::EI: {
            if (ei_tolerance <= T{0} || ei_tolerance >= T{1}) {
                return result;
            }
            const T V = ei_tolerance;
            result.count = 3;
            result.amplitudes = {(T{1} + V) / T{4}, (T{1} - V) / T{2}, (T{1} + V) / T{4}, T{0}};
            result.times = {T{0}, half_period, T{2} * half_period, T{0}};
            break;
        }
    }

    // Normalize amplitudes to unity DC gain and compute integer sample delays.
    T sum = T{0};
    for (size_t i = 0; i < result.count; ++i) {
        sum += result.amplitudes[i];
    }
    if (sum <= default_tol<T>()) {
        return InputShaperResult<T>{};
    }
    for (size_t i = 0; i < result.count; ++i) {
        result.amplitudes[i] /= sum;
        result.delays[i] = static_cast<size_t>((result.times[i] / Ts) + T{0.5});
    }
    result.Ts = Ts;
    result.success = true;
    return result;
}

} // namespace design

/**
 * @ingroup controllers
 * @brief Input-shaper runtime — convolves a command stream with the shaper impulses.
 *
 * Feed the reference command each tick; it returns the shaped command. A delay
 * line of @p MaxDelay+1 samples holds the command history; the output is the
 * impulse-weighted sum of delayed commands. If the design's longest delay exceeds
 * @p MaxDelay (frequency too low for the buffer), the shaper is invalid and passes
 * the command through unchanged.
 *
 * @tparam MaxDelay Maximum impulse delay in samples the buffer can hold
 * @tparam T        Scalar type (float or double)
 */
template<size_t MaxDelay, typename T = float>
class InputShaper {
public:
    static constexpr size_t BufLen = MaxDelay + 1;

    constexpr InputShaper() = default;

    constexpr explicit InputShaper(const design::InputShaperResult<T>& design) {
        if (!design.success || design.max_delay() > MaxDelay) {
            return; // valid_ stays false -> pass-through
        }
        count_ = design.count;
        for (size_t i = 0; i < count_; ++i) {
            amplitudes_[i] = design.amplitudes[i];
            delays_[i] = design.delays[i];
        }
        valid_ = true;
    }

    /// Shape one command sample.
    constexpr T step(T command) {
        if (!valid_) {
            return command;
        }
        buffer_[head_] = command;
        T out = T{0};
        for (size_t i = 0; i < count_; ++i) {
            const size_t idx = (head_ + BufLen - delays_[i]) % BufLen;
            out += amplitudes_[i] * buffer_[idx];
        }
        head_ = (head_ + 1) % BufLen;
        return out;
    }

    [[nodiscard]] constexpr bool valid() const { return valid_; }

    constexpr void reset() {
        buffer_ = wet::array<T, BufLen>{};
        head_ = 0;
    }

private:
    wet::array<T, BufLen> buffer_{};
    wet::array<T, 4>      amplitudes_{};
    wet::array<size_t, 4> delays_{};
    size_t                count_{0};
    size_t                head_{0};
    bool                  valid_{false};
};

/**
 * @ingroup controllers
 * @brief Multi-axis input-shaper bank — one shaper per axis, shared buffer length.
 *
 * Convenience for multi-axis motion: shape each component of a command vector
 * with its own (possibly distinct) shaper. Each axis may target a different mode.
 *
 * @tparam NAxes    Number of axes
 * @tparam MaxDelay Per-axis buffer length bound (samples)
 * @tparam T        Scalar type
 */
template<size_t NAxes, size_t MaxDelay, typename T = float>
class InputShaperBank {
public:
    constexpr InputShaperBank() = default;

    /// Set the shaper for one axis.
    constexpr void set_axis(size_t axis, const design::InputShaperResult<T>& design) {
        shapers_[axis] = InputShaper<MaxDelay, T>(design);
    }

    /// Shape one command per axis.
    constexpr wet::array<T, NAxes> step(const wet::array<T, NAxes>& command) {
        wet::array<T, NAxes> out{};
        for (size_t a = 0; a < NAxes; ++a) {
            out[a] = shapers_[a].step(command[a]);
        }
        return out;
    }

    [[nodiscard]] constexpr InputShaper<MaxDelay, T>& axis(size_t a) { return shapers_[a]; }

    constexpr void reset() {
        for (size_t a = 0; a < NAxes; ++a) {
            shapers_[a].reset();
        }
    }

private:
    wet::array<InputShaper<MaxDelay, T>, NAxes> shapers_{};
};

} // namespace wet
