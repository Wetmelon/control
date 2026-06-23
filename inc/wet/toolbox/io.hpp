#pragma once

/**
 * @file utility/io.hpp
 * @brief Tier-2 I/O appliances — batteries-included field-I/O channels.
 *
 * Where utility/conditioning.hpp, utility/logic.hpp, and utility/scaling.hpp
 * provide the primitives (range monitor, debounce, edge detect, affine cal),
 * this header composes them into the constructs you actually wire up: a
 * calibrated+validated analog input, a debounced push-button with edges and
 * hold, a debounced maintained switch. Live in `wet::io` to mark them as the
 * higher-level composites.
 *
 * @see utility/conditioning.hpp, utility/logic.hpp, utility/scaling.hpp (tier 1).
 */

#include "wet/toolbox/conditioning.hpp" // RangeMonitor, SignalStatus, is_valid/is_fault
#include "wet/toolbox/logic.hpp"        // Debounce, OnDelayTimer
#include "wet/toolbox/scaling.hpp"      // AffineCal

namespace wet::io {

/**
 * @brief A single analog input: range/fault check on the raw reading, then
 *        affine calibration to engineering units.
 *
 * The everyday "read one sensor" appliance. The raw value is classified first
 * (a wire fault is a property of the raw V/mA, not the scaled value), then
 * scaled. `update()` returns the engineering value; `status()`/`faulted()`
 * report the channel health from the same tick.
 *
 * @code
 * io::AnalogInput<float> level{
 *     two_point_cal(0.5f, 0.0f, 4.5f, 100.0f),   // 0.5–4.5 V → 0–100 %
 *     RangeMonitor<float>{0.25f, 0.5f, 4.5f, 4.75f}}; // NE43 fault bands
 * float pct = level.update(volts);
 * if (level.faulted()) flag_wire_break();
 * @endcode
 */
template<typename T = float>
class AnalogInput {
public:
    constexpr AnalogInput() = default;
    constexpr AnalogInput(const AffineCal<T>& cal, const RangeMonitor<T>& monitor)
        : cal_(cal), monitor_(monitor) {}

    /// Validate the raw reading, scale it, and store both results.
    constexpr T update(T raw) {
        status_ = monitor_(raw);
        value_ = cal_.apply(raw);
        return value_;
    }

    [[nodiscard]] constexpr T            value() const { return value_; }
    [[nodiscard]] constexpr SignalStatus status() const { return status_; }
    [[nodiscard]] constexpr bool         valid() const { return is_valid(status_); }
    [[nodiscard]] constexpr bool         faulted() const { return is_fault(status_); }

private:
    AffineCal<T>    cal_{};
    RangeMonitor<T> monitor_{};
    T               value_{};
    SignalStatus    status_{SignalStatus::Valid};
};

/**
 * @brief Operator-axis conditioning chain (joystick / RC stick → command).
 *
 * The teleop counterpart to @ref AnalogInput: instead of validating a sensor, it
 * *shapes an operator input*. Each `update(raw)` runs the standard chain —
 * affine cal to normalized `[-1, 1]` → center scaled dead zone → exponential
 * response curve → output scale (with optional inversion):
 *
 *   1. `cal.apply(raw)`        raw counts/volts → ~`[-1, 1]` (clamped before shaping)
 *   2. `scaled_deadband(·, dz)` ignore slop around center, full range preserved
 *   3. `expo(·, k)`            soften near center for fine control
 *   4. `· × scale` (± invert)  to the actuator command range
 *
 * @code
 * io::AxisInput<float> steer{
 *     two_point_cal(0.0f, -1.0f, 4095.0f, 1.0f), // 12-bit ADC → [-1, 1]
 *     0.05f,   // 5% center dead zone
 *     0.6f,    // expo
 *     1.0f};   // output scale
 * float cmd = steer.update(adc);
 * @endcode
 */
template<typename T = float>
class AxisInput {
public:
    constexpr AxisInput() = default;
    constexpr AxisInput(const AffineCal<T>& cal, T deadzone = T{0}, T expo_k = T{0}, T scale = T{1}, bool invert = false)
        : cal_(cal), deadzone_(deadzone), expo_(expo_k), scale_(scale), invert_(invert) {}

    constexpr T update(T raw) {
        T x = wet::clamp(cal_.apply(raw), T{-1}, T{1}); // normalize, guard cal overshoot
        x = scaled_deadband(x, deadzone_);
        x = expo(x, expo_);
        if (invert_) {
            x = -x;
        }
        value_ = scale_ * x;
        return value_;
    }

    [[nodiscard]] constexpr T value() const { return value_; }

private:
    AffineCal<T> cal_{};
    T            deadzone_{T{0}};
    T            expo_{T{0}};
    T            scale_{T{1}};
    bool         invert_{false};
    T            value_{T{0}};
};

/**
 * @brief Debounced momentary push-button with edge and long-press detection.
 *
 * `down()` is the debounced level; `pressed()`/`released()` are one-tick edges;
 * `held()` is true once the button has been down continuously past the hold
 * time. A zero hold time (default) makes `held()` track `down()`.
 *
 * @tparam T time type (float seconds, or integral with `dt = 1` for ticks).
 */
template<typename T = float>
class Button {
public:
    constexpr Button() = default;
    constexpr explicit Button(T debounce_time, T hold_time = T{0})
        : db_(debounce_time), hold_(hold_time) {}

    /// Advance one tick with the raw electrical state and elapsed time.
    constexpr void update(bool raw, T dt) {
        const bool s = db_(raw, dt);
        pressed_ = s && !prev_;
        released_ = !s && prev_;
        prev_ = s;
        held_ = hold_(s, dt);
    }

    [[nodiscard]] constexpr bool down() const { return prev_; }         //!< debounced level
    [[nodiscard]] constexpr bool pressed() const { return pressed_; }   //!< rising edge (one tick)
    [[nodiscard]] constexpr bool released() const { return released_; } //!< falling edge (one tick)
    [[nodiscard]] constexpr bool held() const { return held_; }         //!< down past the hold time

    constexpr void reset() {
        db_.reset();
        hold_.reset();
        prev_ = pressed_ = released_ = held_ = false;
    }

private:
    Debounce<T>     db_{};
    OnDelayTimer<T> hold_{};
    bool            prev_{false};
    bool            pressed_{false};
    bool            released_{false};
    bool            held_{false};
};

/**
 * @brief Debounced maintained switch (toggle/selector contact) with change flag.
 *
 * `on()` is the debounced level; `changed()` is true the tick the debounced
 * state flips. The smoothed contact for a panel switch or limit/proximity sensor.
 *
 * @tparam T time type (see @ref Button).
 */
template<typename T = float>
class Switch {
public:
    constexpr Switch() = default;
    constexpr explicit Switch(T debounce_time, bool initial = false)
        : db_(debounce_time, initial), prev_(initial) {}

    /// Advance one tick; returns the debounced state.
    constexpr bool update(bool raw, T dt) {
        const bool s = db_(raw, dt);
        changed_ = (s != prev_);
        prev_ = s;
        return s;
    }

    [[nodiscard]] constexpr bool on() const { return prev_; }
    [[nodiscard]] constexpr bool changed() const { return changed_; }

    constexpr void reset(bool initial = false) {
        db_.reset(initial);
        prev_ = initial;
        changed_ = false;
    }

private:
    Debounce<T> db_{};
    bool        prev_{false};
    bool        changed_{false};
};

} // namespace wet::io
