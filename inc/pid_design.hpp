#pragma once

/**
 * @defgroup pid_design Modelless PID Design
 * @brief Classical PID tuning methods that don't require a plant model
 *
 * These methods use experimentally-determined parameters (ultimate gain/period,
 * step response characteristics, or desired closed-loop bandwidth) to compute
 * PID gains. All functions return PIDResult structs compatible with PIDController.
 *
 * Includes:
 * - Ziegler-Nichols (ultimate gain method and step response method)
 * - Cohen-Coon (first-order plus dead-time models)
 * - Lambda tuning (setpoint tracking, robustness-focused)
 * - Bandwidth-based design (specify desired closed-loop bandwidth)
 * - Direct pole placement (specify desired closed-loop poles for discretized PID)
 * - SIMC (Skogestad IMC) rules for FOPDT/SOPDT models
 * - Tyreus-Luyben (conservative Ziegler-Nichols variant)
 */

#include <cmath>
#include <numbers>

#include "constexpr_math.hpp"
#include "pid.hpp"

namespace wetmelon::control {

namespace design {

/**
 * @brief PID controller type selection for tuning methods
 */
enum class PIDType { P,
                     PI,
                     PD,
                     PID };

// ============================================================================
// Ziegler-Nichols Ultimate Gain Method
// ============================================================================

/**
 * @brief Ziegler-Nichols tuning from ultimate gain and ultimate period
 *
 * The user experimentally determines:
 * - Ku: ultimate (critical) gain where system oscillates continuously
 * - Tu: ultimate period of those oscillations
 *
 * Classic formulas from Ziegler & Nichols (1942).
 *
 * @param Ku   Ultimate gain (gain at which system marginally oscillates)
 * @param Tu   Ultimate period of oscillation (seconds)
 * @param Ts   Sampling time (seconds)
 * @param type Controller type (P, PI, PD, or PID)
 * @return PIDResult with tuned gains
 */
template<typename T = double>
[[nodiscard]] consteval PIDResult<T>
ziegler_nichols(T Ku, T Tu, T Ts, PIDType type = PIDType::PID) {
    T Kp{}, Ki{}, Kd{};
    switch (type) {
        case PIDType::P:
            Kp = T{0.5} * Ku;
            break;
        case PIDType::PI:
            Kp = T{0.45} * Ku;
            Ki = Kp / (Tu / T{1.2});
            break;
        case PIDType::PD:
            Kp = T{0.8} * Ku;
            Kd = Kp * Tu / T{8};
            break;
        case PIDType::PID:
            Kp = T{0.6} * Ku;
            Ki = Kp / (Tu / T{2});
            Kd = Kp * Tu / T{8};
            break;
    }
    return PIDResult<T>{Kp, Ki, Kd, Ts};
}

/**
 * @brief Ziegler-Nichols step response method (reaction curve)
 *
 * Uses first-order-plus-dead-time (FOPDT) model parameters obtained
 * from a step response:
 * - K: static gain (output change / input change)
 * - L: apparent dead time (seconds)
 * - tau: time constant (seconds)
 *
 * @param K    Static gain
 * @param L    Apparent dead time (delay, seconds)
 * @param tau  Time constant (seconds)
 * @param Ts   Sampling time (seconds)
 * @param type Controller type (P, PI, or PID)
 * @return PIDResult with tuned gains
 */
template<typename T = double>
[[nodiscard]] consteval PIDResult<T>
ziegler_nichols_step(T K, T L, T tau, T Ts, PIDType type = PIDType::PID) {
    T Kp{}, Ki{}, Kd{};
    T a = tau / (K * L); // normalized gain
    switch (type) {
        case PIDType::P:
            Kp = a;
            break;
        case PIDType::PI:
            Kp = T{0.9} * a;
            Ki = Kp / (L * T{3.33});
            break;
        case PIDType::PD:
            Kp = a;
            Kd = Kp * L * T{0.5};
            break;
        case PIDType::PID:
            Kp = T{1.2} * a;
            Ki = Kp / (T{2} * L);
            Kd = Kp * T{0.5} * L;
            break;
    }
    return PIDResult<T>{Kp, Ki, Kd, Ts};
}

// ============================================================================
// Tyreus-Luyben (conservative Ziegler-Nichols variant)
// ============================================================================

/**
 * @brief Tyreus-Luyben tuning from ultimate gain and ultimate period
 *
 * A more conservative variant of Ziegler-Nichols that produces less aggressive
 * controllers with better robustness. Recommended for processes where
 * Ziegler-Nichols is too aggressive.
 *
 * @param Ku   Ultimate gain
 * @param Tu   Ultimate period (seconds)
 * @param Ts   Sampling time (seconds)
 * @param type Controller type (PI or PID only; others fall back to ZN)
 * @return PIDResult with tuned gains
 */
template<typename T = double>
[[nodiscard]] consteval PIDResult<T>
tyreus_luyben(T Ku, T Tu, T Ts, PIDType type = PIDType::PID) {
    T Kp{}, Ki{}, Kd{};
    switch (type) {
        case PIDType::PI:
            Kp = Ku / T{3.2};
            Ki = Kp / (T{2.2} * Tu);
            break;
        case PIDType::PID:
            Kp = Ku / T{2.2};
            Ki = Kp / (T{2.2} * Tu);
            Kd = Kp * Tu / T{6.3};
            break;
        default:
            // Fall back to Ziegler-Nichols for P and PD
            return ziegler_nichols(Ku, Tu, Ts, type);
    }
    return PIDResult<T>{Kp, Ki, Kd, Ts};
}

// ============================================================================
// Cohen-Coon (FOPDT model)
// ============================================================================

/**
 * @brief Cohen-Coon tuning from first-order-plus-dead-time model
 *
 * Better than Ziegler-Nichols for processes with large dead time relative
 * to the time constant (L/tau > 0.25). Uses FOPDT model parameters.
 *
 * @param K    Static gain
 * @param L    Dead time (seconds)
 * @param tau  Time constant (seconds)
 * @param Ts   Sampling time (seconds)
 * @param type Controller type (P, PI, PD, or PID)
 * @return PIDResult with tuned gains
 */
template<typename T = double>
[[nodiscard]] consteval PIDResult<T>
cohen_coon(T K, T L, T tau, T Ts, PIDType type = PIDType::PID) {
    T Kp{}, Ki{}, Kd{};
    T r = L / tau; // dead-time ratio
    T a = tau / (K * L);

    switch (type) {
        case PIDType::P:
            Kp = a * (T{1} + r / T{3});
            break;
        case PIDType::PI: {
            Kp = a * (T{0.9} + r / T{12});
            T Ti = L * (T{30} + T{3} * r) / (T{9} + T{20} * r);
            Ki = Kp / Ti;
        } break;
        case PIDType::PD: {
            Kp = a * (T{1.24} + r / T{5});
            T Td = L * T{6} / (T{22} + T{3} * r);
            Kd = Kp * Td;
        } break;
        case PIDType::PID: {
            Kp = a * (T{4} / T{3} + r / T{4});
            T Ti = L * (T{32} + T{6} * r) / (T{13} + T{8} * r);
            T Td = L * T{4} / (T{11} + T{2} * r);
            Ki = Kp / Ti;
            Kd = Kp * Td;
        } break;
    }
    return PIDResult<T>{Kp, Ki, Kd, Ts};
}

// ============================================================================
// SIMC (Skogestad IMC) Tuning Rules
// ============================================================================

/**
 * @brief SIMC (Skogestad Internal Model Control) tuning for FOPDT models
 *
 * Simple, robust tuning rules from Skogestad (2003). The user specifies
 * a single tuning parameter tau_c (desired closed-loop time constant).
 * Rule of thumb: tau_c = max(tau, 8*L) for good robustness.
 *
 * @param K     Static gain
 * @param L     Dead time (seconds)
 * @param tau   Time constant (seconds)
 * @param tau_c Desired closed-loop time constant (seconds). Larger = more robust.
 * @param Ts    Sampling time (seconds)
 * @param type  Controller type (PI or PID)
 * @return PIDResult with tuned gains
 */
template<typename T = double>
[[nodiscard]] consteval PIDResult<T>
simc(T K, T L, T tau, T tau_c, T Ts, PIDType type = PIDType::PI) {
    T Kp{}, Ki{}, Kd{};
    switch (type) {
        case PIDType::P:
            Kp = tau / (K * (tau_c + L));
            break;
        case PIDType::PI: {
            Kp = tau / (K * (tau_c + L));
            T Ti = (tau < T{4} * (tau_c + L)) ? tau : T{4} * (tau_c + L);
            Ki = Kp / Ti;
        } break;
        case PIDType::PD: {
            Kp = tau / (K * (tau_c + L));
            Kd = Kp * L / T{2};
        } break;
        case PIDType::PID: {
            Kp = tau / (K * (tau_c + L));
            T Ti = (tau < T{4} * (tau_c + L)) ? tau : T{4} * (tau_c + L);
            Ki = Kp / Ti;
            Kd = Kp * L / T{2};
        } break;
    }
    return PIDResult<T>{Kp, Ki, Kd, Ts};
}

// ============================================================================
// Lambda Tuning (IMC-based)
// ============================================================================

/**
 * @brief Lambda tuning for FOPDT model
 *
 * IMC-based approach where the user directly specifies the desired
 * closed-loop time constant (lambda). Produces non-oscillatory
 * setpoint tracking. Very robust, commonly used in process control.
 *
 * @param K      Static gain
 * @param L      Dead time (seconds)
 * @param tau    Time constant (seconds)
 * @param lambda Desired closed-loop time constant (seconds). Must be > L.
 * @param Ts     Sampling time (seconds)
 * @return PIDResult (PI controller gains)
 */
template<typename T = double>
[[nodiscard]] consteval PIDResult<T>
lambda_tuning(T K, T L, T tau, T lambda, T Ts) {
    T Kp = tau / (K * (lambda + L));
    T Ki = Kp / tau;
    return PIDResult<T>{Kp, Ki, T{0}, Ts};
}

// ============================================================================
// Bandwidth-Based PID Design (model-free)
// ============================================================================

/**
 * @brief Design PID from desired bandwidth and phase margin
 *
 * Given a desired closed-loop bandwidth ωbw and phase margin φm,
 * computes PID gains that achieve approximately those specifications.
 * This is a model-free method — the user specifies desired performance
 * without needing a plant model.
 *
 * The controller is C(s) = Kp + Ki/s + Kd*s with:
 * - Crossover at ωbw: |C(jωbw)| = 1 (assuming unit-gain plant at crossover)
 * - Phase margin φm at crossover
 *
 * @param wbw          Desired bandwidth (crossover frequency, rad/s)
 * @param phase_margin Desired phase margin (degrees, default: 60°)
 * @param Ts           Sampling time (seconds)
 * @param type         Controller type (PI or PID)
 * @return PIDResult with tuned gains
 */
template<typename T = double>
[[nodiscard]] consteval PIDResult<T>
pid_from_bandwidth(T wbw, T phase_margin, T Ts, PIDType type = PIDType::PID) {
    constexpr T pi = std::numbers::pi_v<T>;
    T           phi = phase_margin * pi / T{180}; // Convert to radians
    T           desired_phase = phi - pi;         // Phase of C(jω) at crossover

    T Kp{}, Ki{}, Kd{};

    switch (type) {
        case PIDType::P:
            Kp = T{1}; // Unit gain at crossover
            break;
        case PIDType::PI: {
            // PI controller: C(jω) = Kp - j·Ki/ω
            // Phase of PI at crossover = -atan(Ki/(Kp·ω))
            // We want total loop phase margin = phi_m
            // For unit-gain plant: PM = 90° + atan(Kp·ω/Ki) [since PI phase is in [-90°,0°]]
            // So atan(Kp·wbw/Ki) = phi_m - 90° = phi - π/2
            // With |C(jωbw)| = 1: Kp² + (Ki/wbw)² = 1
            T alpha = phi - pi / T{2}; // angle from the geometric relationship
            // If phi < π/2 (phase margin < 90°), alpha < 0, use |alpha|
            // tan(alpha) = Kp·wbw / Ki, and Kp² + Ki²/wbw² = 1
            // Let r = Ki/(Kp·wbw), then Kp²(1 + r²) = 1
            // r = 1/tan(alpha) = cos(alpha)/sin(alpha)
            if (alpha > T{0}) {
                T r = wet::cos(alpha) / wet::sin(alpha); // cot(alpha)
                Kp = T{1} / wet::sqrt(T{1} + r * r);
                Ki = Kp * r * wbw;
            } else {
                // Phase margin > 90°: PI alone overshoots; use simple heuristic
                Kp = wet::cos(phi);
                Ki = wet::sin(phi) * wbw;
            }
        } break;
        case PIDType::PD: {
            // C(jω) = Kp + Kd*jω
            // Phase = atan2(Kd*ω, Kp), Magnitude = sqrt(Kp² + (Kd*ω)²) = 1
            T tan_phi_d = wet::tan(desired_phase + pi);
            Kp = wet::cos(desired_phase + pi);
            Kd = Kp * tan_phi_d / wbw;
            if (Kp < T{0}) {
                Kp = -Kp;
                Kd = -Kd;
            }
        } break;
        case PIDType::PID: {
            // For PID, place the zero pair symmetrically around crossover
            // Ti = 1/(ωbw * tan(φ/2)), Td = Ti/4
            T half_phi = (phi - pi / T{2}) / T{2};
            if (half_phi <= T{0}) {
                half_phi = pi / T{12}; // minimum 15° half-angle
            }
            T Ti = T{1} / (wbw * wet::tan(half_phi));
            T Td = Ti / T{4};
            Kp = T{1}; // Normalized for unit plant gain at crossover
            Ki = Kp / Ti;
            Kd = Kp * Td;
        } break;
    }
    return PIDResult<T>{Kp, Ki, Kd, Ts};
}

// ============================================================================
// Direct Pole Placement PID
// ============================================================================

/**
 * @brief Direct PID pole placement for a first-order-plus-dead-time model
 *
 * Places the closed-loop poles at the specified locations by computing PID
 * gains for a discretized FOPDT plant model.
 *
 * The FOPDT model is: G(s) = K * exp(-Ls) / (tau*s + 1)
 * Discretized with ZOH: G(z) = K*(1 - a) / (z - a)  where a = exp(-Ts/tau)
 * Dead time approximated as d = round(L/Ts) pure delays: G(z) = K*(1-a)*z^{-d} / (z - a)
 *
 * For a PI controller with one integrator pole at z=1, the closed-loop
 * characteristic polynomial is degree 2+d. The user specifies the desired
 * closed-loop poles and the gains are computed.
 *
 * This function handles the d=0 (no dead-time) case, placing closed-loop
 * poles for the 2nd-order closed-loop system (plant pole + integrator).
 *
 * @param K     Static gain
 * @param tau   Time constant (seconds)
 * @param p1    Desired closed-loop pole 1 (z-domain, inside unit circle)
 * @param p2    Desired closed-loop pole 2 (z-domain, inside unit circle)
 * @param Ts    Sampling time (seconds)
 * @return PIDResult (PI gains via pole placement)
 */
template<typename T = double>
[[nodiscard]] consteval PIDResult<T>
pid_pole_placement(T K, T tau, T p1, T p2, T Ts) {
    // Discretize the first-order plant: G(z) = K*(1-a)/(z-a)
    T a = wet::exp(-Ts / tau);
    T b = K * (T{1} - a);

    // PI controller: C(z) = Kp + Ki*Ts/(z-1)
    // Closed-loop characteristic equation: (z - a)(z - 1) + b*(Kp*(z-1) + Ki*Ts) = 0
    // Expanding: z² - (a+1)z + a + b*Kp*z - b*Kp + b*Ki*Ts = 0
    // = z² + (b*Kp - a - 1)*z + (a - b*Kp + b*Ki*Ts) = 0
    //
    // Desired: (z - p1)(z - p2) = z² - (p1+p2)*z + p1*p2 = 0
    //
    // Matching coefficients:
    // b*Kp - a - 1 = -(p1 + p2)    => Kp = (a + 1 - p1 - p2) / b
    // a - b*Kp + b*Ki*Ts = p1*p2   => Ki = (p1*p2 - a + b*Kp) / (b*Ts)

    T Kp = (a + T{1} - p1 - p2) / b;
    T Ki = (p1 * p2 - a + b * Kp) / (b * Ts);

    return PIDResult<T>{Kp, Ki, T{0}, Ts};
}

/**
 * @brief PID pole placement for first-order plant with 3 desired poles
 *
 * For PID control of a first-order plant (no dead time), the closed-loop
 * has 3 poles (plant + integrator + derivative filter).
 *
 * @param K     Static gain
 * @param tau   Time constant (seconds)
 * @param p1    Desired closed-loop pole 1 (z-domain)
 * @param p2    Desired closed-loop pole 2 (z-domain)
 * @param p3    Desired closed-loop pole 3 (z-domain)
 * @param Ts    Sampling time (seconds)
 * @return PIDResult with PID gains
 */
template<typename T = double>
[[nodiscard]] consteval PIDResult<T>
pid_pole_placement(T K, T tau, T p1, T p2, T p3, T Ts) {
    // Discretize: G(z) = b/(z-a), a = exp(-Ts/tau), b = K*(1-a)
    T a = wet::exp(-Ts / tau);
    T b = K * (T{1} - a);

    // PID controller in z-domain: C(z) = (Kp*(z-1) + Ki*Ts + Kd*(z-1)²/Ts) / (z-1)
    // = (Kd/Ts * z² + (Kp - 2*Kd/Ts)*z + (-Kp + Kd/Ts + Ki*Ts)) / (z*(z-1))
    //
    // Open-loop: C(z)*G(z) = b * (Kd/Ts*z² + (Kp-2Kd/Ts)*z + (-Kp+Kd/Ts+Ki*Ts)) / (z*(z-1)*(z-a))
    //
    // Closed-loop char poly: z*(z-1)*(z-a) + b*(Kd/Ts*z² + (Kp-2Kd/Ts)*z + (-Kp+Kd/Ts+Ki*Ts)) = 0
    // = z³ - (1+a)*z² + a*z + b*Kd/Ts*z² + b*(Kp-2Kd/Ts)*z + b*(-Kp+Kd/Ts+Ki*Ts)
    // = z³ + (b*Kd/Ts - 1 - a)*z² + (a + b*Kp - 2*b*Kd/Ts)*z + b*(-Kp + Kd/Ts + Ki*Ts)
    //
    // Desired: (z-p1)(z-p2)(z-p3) = z³ - (p1+p2+p3)*z² + (p1p2+p1p3+p2p3)*z - p1*p2*p3
    //
    // Matching coefficients:
    T sum_p = p1 + p2 + p3;
    T sum_pp = p1 * p2 + p1 * p3 + p2 * p3;
    T prod_p = p1 * p2 * p3;

    // b*Kd/Ts - 1 - a = -sum_p => Kd = (1 + a - sum_p) * Ts / b
    T Kd = (T{1} + a - sum_p) * Ts / b;

    // a + b*Kp - 2*b*Kd/Ts = sum_pp => Kp = (sum_pp - a + 2*b*Kd/Ts) / b
    T Kp = (sum_pp - a + T{2} * b * Kd / Ts) / b;

    // b*(-Kp + Kd/Ts + Ki*Ts) = -prod_p => Ki = (-prod_p/b + Kp - Kd/Ts) / Ts
    T Ki = (-prod_p / b + Kp - Kd / Ts) / Ts;

    return PIDResult<T>{Kp, Ki, Kd, Ts};
}

} // namespace design

namespace online {

/**
 * @brief Ziegler-Nichols tuning (runtime version)
 */
template<typename T = double>
[[nodiscard]] constexpr PIDResult<T>
ziegler_nichols(T Ku, T Tu, T Ts, design::PIDType type = design::PIDType::PID) {
    T Kp{}, Ki{}, Kd{};
    switch (type) {
        case design::PIDType::P:
            Kp = T{0.5} * Ku;
            break;
        case design::PIDType::PI:
            Kp = T{0.45} * Ku;
            Ki = Kp / (Tu / T{1.2});
            break;
        case design::PIDType::PD:
            Kp = T{0.8} * Ku;
            Kd = Kp * Tu / T{8};
            break;
        case design::PIDType::PID:
            Kp = T{0.6} * Ku;
            Ki = Kp / (Tu / T{2});
            Kd = Kp * Tu / T{8};
            break;
    }
    return PIDResult<T>{Kp, Ki, Kd, Ts};
}

/**
 * @brief Ziegler-Nichols step response method (runtime version)
 */
template<typename T = double>
[[nodiscard]] constexpr PIDResult<T>
ziegler_nichols_step(T K, T L, T tau, T Ts, design::PIDType type = design::PIDType::PID) {
    T Kp{}, Ki{}, Kd{};
    T a = tau / (K * L);
    switch (type) {
        case design::PIDType::P:
            Kp = a;
            break;
        case design::PIDType::PI:
            Kp = T{0.9} * a;
            Ki = Kp / (L * T{3.33});
            break;
        case design::PIDType::PD:
            Kp = a;
            Kd = Kp * L * T{0.5};
            break;
        case design::PIDType::PID:
            Kp = T{1.2} * a;
            Ki = Kp / (T{2} * L);
            Kd = Kp * T{0.5} * L;
            break;
    }
    return PIDResult<T>{Kp, Ki, Kd, Ts};
}

/**
 * @brief Tyreus-Luyben tuning (runtime version)
 */
template<typename T = double>
[[nodiscard]] constexpr PIDResult<T>
tyreus_luyben(T Ku, T Tu, T Ts, design::PIDType type = design::PIDType::PID) {
    T Kp{}, Ki{}, Kd{};
    switch (type) {
        case design::PIDType::PI:
            Kp = Ku / T{3.2};
            Ki = Kp / (T{2.2} * Tu);
            break;
        case design::PIDType::PID:
            Kp = Ku / T{2.2};
            Ki = Kp / (T{2.2} * Tu);
            Kd = Kp * Tu / T{6.3};
            break;
        default:
            return ziegler_nichols(Ku, Tu, Ts, type);
    }
    return PIDResult<T>{Kp, Ki, Kd, Ts};
}

/**
 * @brief Cohen-Coon tuning (runtime version)
 */
template<typename T = double>
[[nodiscard]] constexpr PIDResult<T>
cohen_coon(T K, T L, T tau, T Ts, design::PIDType type = design::PIDType::PID) {
    T Kp{}, Ki{}, Kd{};
    T r = L / tau;
    T a = tau / (K * L);
    switch (type) {
        case design::PIDType::P:
            Kp = a * (T{1} + r / T{3});
            break;
        case design::PIDType::PI: {
            Kp = a * (T{0.9} + r / T{12});
            T Ti = L * (T{30} + T{3} * r) / (T{9} + T{20} * r);
            Ki = Kp / Ti;
        } break;
        case design::PIDType::PD: {
            Kp = a * (T{1.24} + r / T{5});
            T Td = L * T{6} / (T{22} + T{3} * r);
            Kd = Kp * Td;
        } break;
        case design::PIDType::PID: {
            Kp = a * (T{4} / T{3} + r / T{4});
            T Ti = L * (T{32} + T{6} * r) / (T{13} + T{8} * r);
            T Td = L * T{4} / (T{11} + T{2} * r);
            Ki = Kp / Ti;
            Kd = Kp * Td;
        } break;
    }
    return PIDResult<T>{Kp, Ki, Kd, Ts};
}

/**
 * @brief SIMC tuning (runtime version)
 */
template<typename T = double>
[[nodiscard]] constexpr PIDResult<T>
simc(T K, T L, T tau, T tau_c, T Ts, design::PIDType type = design::PIDType::PI) {
    T Kp{}, Ki{}, Kd{};
    switch (type) {
        case design::PIDType::P:
            Kp = tau / (K * (tau_c + L));
            break;
        case design::PIDType::PI: {
            Kp = tau / (K * (tau_c + L));
            T Ti = (tau < T{4} * (tau_c + L)) ? tau : T{4} * (tau_c + L);
            Ki = Kp / Ti;
        } break;
        case design::PIDType::PD: {
            Kp = tau / (K * (tau_c + L));
            Kd = Kp * L / T{2};
        } break;
        case design::PIDType::PID: {
            Kp = tau / (K * (tau_c + L));
            T Ti = (tau < T{4} * (tau_c + L)) ? tau : T{4} * (tau_c + L);
            Ki = Kp / Ti;
            Kd = Kp * L / T{2};
        } break;
    }
    return PIDResult<T>{Kp, Ki, Kd, Ts};
}

/**
 * @brief Lambda tuning (runtime version)
 */
template<typename T = double>
[[nodiscard]] constexpr PIDResult<T>
lambda_tuning(T K, T L, T tau, T lambda, T Ts) {
    T Kp = tau / (K * (lambda + L));
    T Ki = Kp / tau;
    return PIDResult<T>{Kp, Ki, T{0}, Ts};
}

/**
 * @brief Bandwidth-based PID design (runtime version)
 */
template<typename T = double>
[[nodiscard]] constexpr PIDResult<T>
pid_from_bandwidth(T wbw, T phase_margin, T Ts, design::PIDType type = design::PIDType::PID) {
    constexpr T pi = std::numbers::pi_v<T>;
    T           phi = phase_margin * pi / T{180};
    T           desired_phase = phi - pi;
    T           Kp{}, Ki{}, Kd{};

    switch (type) {
        case design::PIDType::P:
            Kp = T{1};
            break;
        case design::PIDType::PI: {
            T alpha = phi - pi / T{2};
            if (alpha > T{0}) {
                T r = wet::cos(alpha) / wet::sin(alpha);
                Kp = T{1} / wet::sqrt(T{1} + r * r);
                Ki = Kp * r * wbw;
            } else {
                Kp = wet::cos(phi);
                Ki = wet::sin(phi) * wbw;
            }
        } break;
        case design::PIDType::PD: {
            T tan_phi_d = wet::tan(desired_phase + pi);
            Kp = wet::cos(desired_phase + pi);
            Kd = Kp * tan_phi_d / wbw;
            if (Kp < T{0}) {
                Kp = -Kp;
                Kd = -Kd;
            }
        } break;
        case design::PIDType::PID: {
            T half_phi = (phi - pi / T{2}) / T{2};
            if (half_phi <= T{0}) {
                half_phi = pi / T{12};
            }
            T Ti = T{1} / (wbw * wet::tan(half_phi));
            T Td = Ti / T{4};
            Kp = T{1};
            Ki = Kp / Ti;
            Kd = Kp * Td;
        } break;
    }
    return PIDResult<T>{Kp, Ki, Kd, Ts};
}

/**
 * @brief PI pole placement (runtime version)
 */
template<typename T = double>
[[nodiscard]] constexpr PIDResult<T>
pid_pole_placement(T K, T tau, T p1, T p2, T Ts) {
    T a = wet::exp(-Ts / tau);
    T b = K * (T{1} - a);
    T Kp = (a + T{1} - p1 - p2) / b;
    T Ki = (p1 * p2 - a + b * Kp) / (b * Ts);
    return PIDResult<T>{Kp, Ki, T{0}, Ts};
}

/**
 * @brief PID pole placement (runtime version)
 */
template<typename T = double>
[[nodiscard]] constexpr PIDResult<T>
pid_pole_placement(T K, T tau, T p1, T p2, T p3, T Ts) {
    T a = wet::exp(-Ts / tau);
    T b = K * (T{1} - a);
    T sum_p = p1 + p2 + p3;
    T sum_pp = p1 * p2 + p1 * p3 + p2 * p3;
    T prod_p = p1 * p2 * p3;
    T Kd = (T{1} + a - sum_p) * Ts / b;
    T Kp = (sum_pp - a + T{2} * b * Kd / Ts) / b;
    T Ki = (-prod_p / b + Kp - Kd / Ts) / Ts;
    return PIDResult<T>{Kp, Ki, Kd, Ts};
}

} // namespace online
} // namespace wetmelon::control
