#pragma once

/**
 * @file utility/conditioning.hpp
 * @brief Nonlinear signal-shaping primitives for real control loops.
 *
 * The everyday command/feedback conditioning blocks that aren't filters and
 * aren't calibration math: dead zones, their inverse (friction/overlap
 * compensation), rate limiting, and a Schmitt-trigger hysteresis comparator.
 * All `constexpr` and allocation-free — drop them straight into an ISR.
 *
 * @see utility/scaling.hpp for stateless range/affine calibration helpers.
 * @see filters/filters.hpp for the linear filter runtimes.
 */

#include <cstdint>
#include <limits>

#include "wet/math/math.hpp" // wet::abs, wet::copysign

namespace wet {

/**
 * @brief Dead zone over `[lower, upper]`, matching Simulink's Dead Zone block.
 *
 * Output is 0 inside the band and the signed distance past the edge outside it
 * (`x − lower` below, `x − upper` above) — continuous through both edges. The
 * standard joystick / valve-command dead zone that suppresses noise around
 * neutral without introducing a step.
 *
 * @param x     Input.
 * @param lower Lower edge of the dead zone.
 * @param upper Upper edge of the dead zone (`upper ≥ lower`).
 * @return 0 within `[lower, upper]`, else `x − lower` / `x − upper`.
 */
template<typename T>
[[nodiscard]] constexpr T deadband(T x, T lower, T upper) {
    if (x > upper) {
        return x - upper;
    }
    if (x < lower) {
        return x - lower;
    }
    return T{0};
}

/// Symmetric dead zone of half-width @p band (i.e. `[−band, band]`).
template<typename T>
[[nodiscard]] constexpr T deadband(T x, T band) {
    return deadband(x, -band, band);
}

/**
 * @brief Inverse dead zone: add an offset to overcome a physical dead zone
 *        (valve overlap, static friction, motor stiction), with independent
 *        negative/positive offsets.
 *
 * A positive command is boosted by @p upper, a negative one by @p lower (use a
 * negative @p lower for a symmetric-feeling boost), so the actuator actually
 * starts moving. Commands within ±@p threshold are forced to 0 so sensor noise
 * doesn't cause jitter/dither. This is the compensating inverse of @ref deadband.
 *
 * @param x         Input command.
 * @param lower     Offset added to negative commands (typically ≤ 0).
 * @param upper     Offset added to positive commands (typically ≥ 0).
 * @param threshold Magnitude below which the output is forced to 0 (default 0).
 */
template<typename T>
[[nodiscard]] constexpr T inverse_deadband(T x, T lower, T upper, T threshold = T{0}) {
    if (wet::abs(x) <= threshold) {
        return T{0};
    }
    return x + (x > T{0} ? upper : lower);
}

/// Symmetric inverse dead zone: boost by @p band in the command's direction.
template<typename T>
[[nodiscard]] constexpr T inverse_deadband(T x, T band) {
    return inverse_deadband(x, -band, band, T{0});
}

/**
 * @brief Center dead zone that rescales the surviving range back to full span.
 *
 * Unlike @ref deadband (which returns the raw distance past the edge), this maps
 * the dead-zone edge to 0 and the input extreme ±1 back to ±1 — the joystick/RC
 * convention, so removing the dead zone doesn't shrink the usable range.
 *
 * @param x        Normalized input, expected in `[-1, 1]`.
 * @param deadzone Dead-zone half-width as a fraction in `[0, 1)`.
 */
template<typename T>
[[nodiscard]] constexpr T scaled_deadband(T x, T deadzone) {
    const T m = wet::abs(x);
    if (m <= deadzone) {
        return T{0};
    }
    return wet::copysign((m - deadzone) / (T{1} - deadzone), x);
}

/**
 * @brief Exponential response curve `y = (1−k)·x + k·x³` (RC "expo").
 *
 * Softens response near center for fine control while keeping full authority at
 * the extremes: `k = 0` is linear, `k = 1` is fully cubic. Endpoints (±1) and
 * sign are preserved and the curve is monotonic for `k ∈ [0, 1]`.
 *
 * @param x Normalized input, expected in `[-1, 1]`.
 * @param k Expo factor in `[0, 1]`.
 */
template<typename T>
[[nodiscard]] constexpr T expo(T x, T k) {
    return ((T{1} - k) * x) + (k * x * x * x);
}

/**
 * @brief Slew-rate limiter: bound how fast the output may follow the target.
 *
 * Caps the per-step change at `rate·dt`, with independent up/down rates so you
 * can, e.g., ramp a command on slowly but allow a fast retreat. Rates are
 * positive magnitudes in units/second; the default (∞) is pass-through. The
 * first sample seeds the state to the target (no start-up ramp).
 *
 * @tparam T Scalar type.
 */
template<typename T = float>
class SlewLimiter {
public:
    constexpr SlewLimiter() = default;

    /// Symmetric rate limit [units/s].
    constexpr explicit SlewLimiter(T rate) : rate_up_(rate), rate_down_(rate) {}

    /// Independent rising/falling rate limits [units/s].
    constexpr SlewLimiter(T rate_up, T rate_down) : rate_up_(rate_up), rate_down_(rate_down) {}

    /// Advance one step toward @p target over @p dt seconds.
    constexpr T operator()(T target, T dt) {
        if (!init_) {
            y_ = target;
            init_ = true;
            return y_;
        }
        const T max_up = rate_up_ * dt;
        const T max_down = rate_down_ * dt;
        const T delta = target - y_;
        if (delta > max_up) {
            y_ += max_up;
        } else if (delta < -max_down) {
            y_ -= max_down;
        } else {
            y_ = target;
        }
        return y_;
    }

    [[nodiscard]] constexpr T value() const { return y_; }

    constexpr void reset() {
        y_ = T{0};
        init_ = false;
    }

    /// Pre-seed the output (and mark initialized) for bumpless start.
    constexpr void reset(T y) {
        y_ = y;
        init_ = true;
    }

private:
    T    rate_up_{std::numeric_limits<T>::infinity()};
    T    rate_down_{std::numeric_limits<T>::infinity()};
    T    y_{T{0}};
    bool init_{false};
};

/**
 * @brief Hysteresis comparator (Schmitt trigger): bool output with separate
 *        on/off thresholds to reject chatter.
 *
 * Output latches true once the input rises above @p high and stays true until
 * it falls below @p low (`low ≤ high`); between the thresholds it holds. The
 * standard cure for relay chatter on a noisy threshold crossing.
 *
 * @tparam T Scalar type.
 */
template<typename T = float>
class Hysteresis {
public:
    constexpr Hysteresis() = default;

    /// @param low Falling (release) threshold. @param high Rising (trip) threshold.
    constexpr Hysteresis(T low, T high) : low_(low), high_(high) {}

    /// Update with one sample, return the latched state.
    constexpr bool operator()(T x) {
        if (state_) {
            if (x < low_) {
                state_ = false;
            }
        } else if (x > high_) {
            state_ = true;
        }
        return state_;
    }

    [[nodiscard]] constexpr bool state() const { return state_; }

    constexpr void reset(bool state = false) { state_ = state; }

private:
    T    low_{0};
    T    high_{0};
    bool state_{false};
};

/**
 * @brief Classification of an analog input against its valid/fault bands.
 *
 * Models NAMUR NE43-style signal-range monitoring: the live-zero margins of a
 * ratiometric (0.5–4.5 V) or 4–20 mA sensor let a genuine reading be told apart
 * from a wire fault.
 */
enum class SignalStatus : std::uint8_t {
    Valid,      //!< within the nominal measuring span
    UnderRange, //!< below span but not a wire fault (saturated low)
    OverRange,  //!< above span but not a wire fault (saturated high)
    FaultLow,   //!< below the fault floor — open circuit / short to ground
    FaultHigh,  //!< above the fault ceiling — short to supply
};

/// True only for the in-span status.
[[nodiscard]] constexpr bool is_valid(SignalStatus s) {
    return s == SignalStatus::Valid;
}

/// True for a wire fault (FaultLow/FaultHigh) — i.e. not a real reading at all.
[[nodiscard]] constexpr bool is_fault(SignalStatus s) {
    return s == SignalStatus::FaultLow || s == SignalStatus::FaultHigh;
}

/**
 * @brief Classify @p x against the four band edges `[fault_lo (valid_lo,
 *        valid_hi) fault_hi]` (assumed ordered, non-decreasing).
 *
 * Bands: `x < fault_lo` → FaultLow; `[fault_lo, valid_lo)` → UnderRange;
 * `[valid_lo, valid_hi]` → Valid; `(valid_hi, fault_hi]` → OverRange;
 * `x > fault_hi` → FaultHigh. Pure/stateless.
 */
template<typename T>
[[nodiscard]] constexpr SignalStatus
classify_range(T x, T fault_lo, T valid_lo, T valid_hi, T fault_hi) {
    if (x < fault_lo) {
        return SignalStatus::FaultLow;
    }
    if (x > fault_hi) {
        return SignalStatus::FaultHigh;
    }
    if (x < valid_lo) {
        return SignalStatus::UnderRange;
    }
    if (x > valid_hi) {
        return SignalStatus::OverRange;
    }
    return SignalStatus::Valid;
}

/**
 * @brief Analog-input range/fault monitor (NAMUR NE43 pattern).
 *
 * Classifies a reading into valid / out-of-range / wire-fault bands so a broken
 * wire (short to ground/supply) is distinguished from a real low/high reading —
 * the standard front-end check before scaling a 0.5–4.5 V or 4–20 mA sensor.
 *
 * With a nonzero @p hysteresis the current band is widened by that margin, so a
 * signal hovering on an edge must move `hysteresis` past it to re-classify (no
 * boundary chatter). `hysteresis = 0` (default) is a plain per-call classifier.
 * For *time*-based fault qualification, compose `faulted()` with a `Debounce`.
 *
 * @code
 * RangeMonitor<float> ai{0.25f, 0.5f, 4.5f, 4.75f}; // [0.25 (0.5 4.5) 4.75]
 * if (ai.faulted(v)) trip();   // broken wire, not a low reading
 * @endcode
 */
template<typename T = float>
class RangeMonitor {
public:
    constexpr RangeMonitor() = default;

    /// @param fault_lo,valid_lo,valid_hi,fault_hi The four band edges.
    /// @param hysteresis Re-classification margin (≥ 0; 0 = none).
    constexpr RangeMonitor(T fault_lo, T valid_lo, T valid_hi, T fault_hi, T hysteresis = T{0})
        : fault_lo_(fault_lo), valid_lo_(valid_lo), valid_hi_(valid_hi), fault_hi_(fault_hi), hyst_(hysteresis) {}

    /// Classify one sample (applies hysteresis around the previous status).
    constexpr SignalStatus operator()(T x) {
        // Bias the current band's edges outward by hyst_ so leaving it costs an
        // extra margin; entering bands keep their nominal edges.
        T lo_f = fault_lo_;
        T lo_v = valid_lo_;
        T hi_v = valid_hi_;
        T hi_f = fault_hi_;
        switch (status_) {
            case SignalStatus::Valid:
                lo_v -= hyst_;
                hi_v += hyst_;
                break;
            case SignalStatus::UnderRange:
                lo_f -= hyst_;
                lo_v += hyst_;
                break;
            case SignalStatus::OverRange:
                hi_v -= hyst_;
                hi_f += hyst_;
                break;
            case SignalStatus::FaultLow:
                lo_f += hyst_;
                break;
            case SignalStatus::FaultHigh:
                hi_f -= hyst_;
                break;
        }
        status_ = classify_range(x, lo_f, lo_v, hi_v, hi_f);
        return status_;
    }

    /// Convenience predicates that also advance the monitor with @p x.
    constexpr bool valid(T x) { return is_valid((*this)(x)); }
    constexpr bool faulted(T x) { return is_fault((*this)(x)); }

    [[nodiscard]] constexpr SignalStatus value() const { return status_; }
    constexpr void                       reset() { status_ = SignalStatus::Valid; }

private:
    T            fault_lo_{};
    T            valid_lo_{};
    T            valid_hi_{};
    T            fault_hi_{};
    T            hyst_{T{0}};
    SignalStatus status_{SignalStatus::Valid};
};

} // namespace wet
