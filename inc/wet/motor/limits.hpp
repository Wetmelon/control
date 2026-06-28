#pragma once

#include <limits>

#include "wet/toolbox/bounds.hpp" // Bounds (interval limits + contains gate)
#include "wet/transforms.hpp"     // DirectQuadrature, instantaneous_power

namespace wet {

/**
 * @brief DC-bus current and voltage limits for an inverter.
 *
 * The current and voltage limits are @ref Bounds intervals (default unbounded via
 * the lowest()/max() sentinels, never @f$ \pm\infty @f$ under @c -ffinite-math-only);
 * @ref voltage doubles as the disarm gate through @ref Bounds::contains. Power is a
 * single motoring cap. Set only the limits the hardware imposes.
 *
 * @tparam T Scalar type.
 */
template<typename T = double>
struct DcBusLimits {
    Bounds<1, T> bus_current{};                                //!< [A] [regen floor, motoring cap], default unbounded
    T            bus_power_max{std::numeric_limits<T>::max()}; //!< [W] motoring bus-power cap (> 0)
    Bounds<1, T> voltage{T{0}, std::numeric_limits<T>::max()}; //!< [V] [undervoltage, overvoltage] gate
};

/**
 * @brief DC-bus state and the torque-current derate it implies.
 * @tparam T Scalar type.
 */
template<typename T = double>
struct DcBusState {
    T    bus_power{T{0}};   //!< [W] @f$ 1.5(v_d i_d + v_q i_q) @f$, positive = motoring
    T    bus_current{T{0}}; //!< [A] bus_power / Vdc
    T    scale{T{1}};       //!< [-] in [0,1], multiply onto the torque-current command
    bool ok{true};          //!< false when Vdc is outside [undervoltage, overvoltage]
};

/**
 * @brief Holds the inverter's torque current within DC-bus current/power limits.
 *
 * Computes the instantaneous bus power @f$ P_{bus} = 1.5(v_d i_d + v_q i_q) @f$
 * and bus current @f$ I_{bus} = P_{bus}/V_{dc} @f$ from the dq operating point,
 * and returns a multiplicative derate @f$ \in [0,1] @f$ for the torque-current
 * command so the bus stays within @ref DcBusLimits. Because bus power is
 * approximately proportional to the torque current at a fixed operating point,
 * the proportional derate @f$ P_{max}/P_{bus} @f$ settles the bus at its limit in
 * roughly one step. Under- and over-voltage produce a hard gate (@ref DcBusState::ok
 * false) for the caller to disarm on.
 *
 * @see instantaneous_power — the dq-frame power used here.
 *
 * @tparam T Scalar type.
 */
template<typename T = double>
class DcBusLimiter {
public:
    constexpr DcBusLimiter() = default;
    constexpr explicit DcBusLimiter(const DcBusLimits<T>& limits) : limits_(limits) {}

    /**
     * @brief Evaluate the bus state and torque-current derate at an operating point.
     *
     * @param Vdq dq voltage applied to the motor [V].
     * @param Idq dq current measured [A].
     * @param Vdc DC bus voltage [V].
     * @return Bus power/current, the derate scale, and the voltage-gate flag.
     */
    [[nodiscard]] constexpr DcBusState<T> evaluate(const DirectQuadrature<T>& Vdq, const DirectQuadrature<T>& Idq, T Vdc) const {
        DcBusState<T> state;
        state.bus_power = instantaneous_power(Vdq, Idq).p;
        state.bus_current = (Vdc > T{0}) ? state.bus_power / Vdc : T{0};

        const T current_max = limits_.bus_current.upper[0];
        const T current_min = limits_.bus_current.lower[0];

        // Motoring (positive power/current draw): scale down toward the binding cap.
        if (state.bus_power > T{0}) {
            if (state.bus_power > limits_.bus_power_max) {
                state.scale = wet::min(state.scale, limits_.bus_power_max / state.bus_power);
            }
            if (state.bus_current > current_max) {
                state.scale = wet::min(state.scale, current_max / state.bus_current);
            }
        }

        // Regen (bus current more negative than the floor): both negative, ratio in (0,1).
        if (state.bus_current < current_min && state.bus_current < T{0}) {
            state.scale = wet::min(state.scale, current_min / state.bus_current);
        }

        state.ok = limits_.voltage.contains(Vdc);
        return state;
    }

    [[nodiscard]] constexpr const DcBusLimits<T>& limits() const { return limits_; }
    constexpr void                                set_limits(const DcBusLimits<T>& limits) { limits_ = limits; }

private:
    DcBusLimits<T> limits_{};
};

} // namespace wet
