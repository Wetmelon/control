#pragma once

/**
 * @file trapezoidal.hpp
 * @brief Minimum-time asymmetric trapezoidal velocity profile (2nd-order, C¹).
 *
 * Plans the classic three-segment motion — accelerate / cruise / decelerate — from
 * (Xi, Vi) to (Xf, Vf) under asymmetric limits Amax/Dmax/Vmax. Acceleration is
 * piecewise-constant so jerk is unbounded; the commanded reference is C¹ (velocity
 * continuous). For jerk-limited (C²) motion use @ref scurve.hpp instead.
 *
 * **Usage pattern** (shared by all profile families):
 * @code
 * #include "wet/trajectory/trapezoidal.hpp"
 * using namespace wet;
 *
 * constexpr TrajectoryLimits<double> lim{.max_velocity = 1.0, .max_acceleration = 2.0, .max_deceleration = 2.0};
 * constexpr auto prof = design::synthesize_trapezoidal(0.0, 1.0, lim);  // rest-to-rest
 * static_assert(prof.success);
 *
 * TrapezoidalTrajectory<double> traj(prof);
 * while (!traj.done()) {
 *     auto [pos, vel, acc, jerk] = traj.step(dt);
 * }
 * @endcode
 *
 * @see L. Biagiotti & C. Melchiorri, "Trajectory Planning for Automatic Machines
 *      and Robots," Springer, 2008, §3.
 * @see R. Béarée, "FIR filter-based online jerk-constrained trajectory generation"
 *      (the asymmetric structure used here).
 * @see scurve.hpp for the jerk-limited (C²) generalization.
 */

#include "wet/matrix/matrix_traits.hpp"
#include "wet/trajectory/trajectory_types.hpp"

namespace wet {

namespace design {

/**
 * @brief Planned trapezoidal profile: the segment durations, reached values, and
 *        boundary state needed to evaluate the trajectory at any time.
 *
 * Time origin t = 0 is the start of the accel phase. Phase boundaries are at
 * `Ta`, `Ta + Tv`, and `Tf = Ta + Tv + Td`.
 *
 * @tparam T Scalar type
 */
template<typename T = double>
struct TrapezoidalProfile {

    T Xi{T{0}}; //!< Start position
    T Vi{T{0}}; //!< Start velocity

    T Xf{T{0}}; //!< Target position
    T Vf{T{0}}; //!< Target velocity

    T Ar{T{0}}; //!< Reached acceleration (signed) of accel phase
    T Vr{T{0}}; //!< Reached (cruise / peak) velocity (signed)
    T Dr{T{0}}; //!< Reached acceleration (signed) of decel phase

    T Ta{T{0}}; //!< Accel-phase duration [s]
    T Tv{T{0}}; //!< Cruise-phase duration [s]
    T Td{T{0}}; //!< Decel-phase duration [s]
    T Tf{T{0}}; //!< Total duration Ta + Tv + Td [s]

    T yAccel{T{0}}; //!< Position at the end of the accel phase

    bool success{false}; //!< False on invalid limits / infeasible request

    /// Evaluate the profile at time @p t (clamped to the initial/final state outside [0, Tf]).
    [[nodiscard]] constexpr TrajectoryState<T> eval(T t) const {
        TrajectoryState<T> s{};
        if (t <= T{0}) { // initial condition (and t == 0)
            s.position = Xi;
            s.velocity = Vi;
            s.acceleration = (Ta > T{0}) ? Ar : T{0};
        } else if (t < Ta) { // accelerating
            s.position = Xi + (Vi * t) + (T{0.5} * Ar * t * t);
            s.velocity = Vi + (Ar * t);
            s.acceleration = Ar;
        } else if (t < Ta + Tv) { // cruising
            s.position = yAccel + (Vr * (t - Ta));
            s.velocity = Vr;
            s.acceleration = T{0};
        } else if (t < Tf) {     // decelerating (measured back from the end)
            const T td = t - Tf; // ∈ [−Td, 0)
            s.position = Xf + (Vf * td) + (T{0.5} * Dr * td * td);
            s.velocity = Vf + (Dr * td);
            s.acceleration = Dr;
        } else { // final condition
            s.position = Xf;
            s.velocity = Vf;
            s.acceleration = T{0};
        }
        return s;
    }

    /// Rebind the profile to another scalar type (e.g. plan in double, run in float).
    template<typename U>
    [[nodiscard]] constexpr TrapezoidalProfile<U> as() const {
        return TrapezoidalProfile<U>{
            static_cast<U>(Xi),
            static_cast<U>(Vi),
            static_cast<U>(Xf),
            static_cast<U>(Vf),
            static_cast<U>(Ar),
            static_cast<U>(Vr),
            static_cast<U>(Dr),
            static_cast<U>(Ta),
            static_cast<U>(Tv),
            static_cast<U>(Td),
            static_cast<U>(Tf),
            static_cast<U>(yAccel),
            success,
        };
    }
};

namespace detail {

/// Candidate plan for one assumed cruise direction @p s ∈ {+1, −1}.
template<typename T>
struct TrapCandidate {
    T    Ar{0};
    T    Vr{0};
    T    Dr{0};
    T    Ta{0};
    T    Tv{0};
    T    Td{0};
    bool ok{false};
};

/**
 * @brief Plan the three-segment profile assuming the cruise velocity has sign @p s.
 *
 * Tries the long (trapezoidal) move first; if the cruise time would be negative
 * the move is short (triangular) and the peak velocity is solved in closed form.
 * Returns `ok == false` if no consistent profile exists for this direction.
 */
template<typename T>
constexpr TrapCandidate<T> plan_for_sign(int s, T dX, T Vi, T Vf, T Vmax, T Amax, T Dmax, T base, T c, T eps) {
    TrapCandidate<T> cand{};
    const T          Vr = static_cast<T>(s) * Vmax;

    // First ramp Vi → Vr (Amax), final ramp Vr → Vf (Dmax); signs point toward the target.
    T Ar = (Vr >= Vi) ? Amax : -Amax;
    T Ta = (Vr - Vi) / Ar; // ≥ 0
    T Dr = (Vf >= Vr) ? Dmax : -Dmax;
    T Td = (Vf - Vr) / Dr; // ≥ 0

    const T dX_ramps = (T{0.5} * (Vi + Vr) * Ta) + (T{0.5} * (Vr + Vf) * Td);
    const T Tv = (dX - dX_ramps) / Vr;
    if (Tv >= -eps) { // long move: cruise at ±Vmax
        cand = {Ar, Vr, Dr, Ta, (Tv < T{0}) ? T{0} : Tv, Td, true};
        return cand;
    }

    // Short move: never reaches Vmax. Solve the peak Vp (sign s, beyond both
    // boundaries) from accel-ramp + decel-ramp displacement == dX.
    const T Vp2 = (s > 0) ? ((dX + base) / c) : ((base - dX) / c);
    if (Vp2 < T{0}) {
        return cand; // ok == false
    }
    const T Vp = wet::copysign(wet::sqrt(Vp2), static_cast<T>(s));
    if (wet::abs(Vp) > Vmax + eps) {
        return cand; // inconsistent — would have been a long move
    }
    Ar = (Vp >= Vi) ? Amax : -Amax;
    Ta = (Vp - Vi) / Ar;
    Dr = (Vf >= Vp) ? Dmax : -Dmax;
    Td = (Vf - Vp) / Dr;
    if (Ta < -eps || Td < -eps) {
        return cand; // ok == false
    }
    cand = {Ar, Vp, Dr, (Ta < T{0}) ? T{0} : Ta, T{0}, (Td < T{0}) ? T{0} : Td, true};
    return cand;
}

} // namespace detail

/**
 * @brief Synthesize the minimum-time asymmetric trapezoidal profile from
 *        (Xi, Vi) to (Xf, Vf) under the given limits.
 *
 * The planner accepts any initial velocity, including over-speed (|Vi| > Vmax,
 * the "handbrake" case — it brakes down to the cruise speed first) and motion in
 * the wrong direction. The target velocity must satisfy |Vf| ≤ Vmax.
 *
 * @tparam T      Scalar type
 * @param Xi      Start position
 * @param Vi      Start velocity
 * @param Xf      Target position
 * @param Vf      Target velocity (|Vf| ≤ max_velocity)
 * @param limits  Asymmetric kinematic limits (Vmax, Amax, Dmax)
 * @return Planned profile with `success` set; `success == false` on invalid
 *         limits, |Vf| > Vmax, or an infeasible request.
 */
template<typename T = double>
[[nodiscard]] constexpr TrapezoidalProfile<T> synthesize_trapezoidal(
    T                          Xi,
    T                          Vi,
    T                          Xf,
    T                          Vf,
    const TrajectoryLimits<T>& limits
) {
    TrapezoidalProfile<T> p{};
    p.Xi = Xi;
    p.Vi = Vi;
    p.Xf = Xf;
    p.Vf = Vf;

    const T Vmax = limits.max_velocity;
    const T Amax = limits.max_acceleration;
    const T Dmax = limits.max_deceleration;
    const T eps = wet::default_tol<T>();
    if (!limits.valid() || wet::abs(Vf) > Vmax + eps) {
        return p; // success == false
    }

    const T dX = Xf - Xi;

    // Trivial: already at the target state with nothing to do.
    if (wet::abs(dX) <= eps && wet::abs(Vi - Vf) <= eps) {
        p.yAccel = Xi;
        p.success = true;
        return p;
    }

    const T base = ((Vi * Vi) / (T{2} * Amax)) + ((Vf * Vf) / (T{2} * Dmax));
    const T c = (T{1} / (T{2} * Amax)) + (T{1} / (T{2} * Dmax));

    const auto plus = detail::plan_for_sign<T>(+1, dX, Vi, Vf, Vmax, Amax, Dmax, base, c, eps);
    const auto minus = detail::plan_for_sign<T>(-1, dX, Vi, Vf, Vmax, Amax, Dmax, base, c, eps);

    detail::TrapCandidate<T> best{};
    if (plus.ok && minus.ok) {
        // Both directions feasible: pick the time-optimal (shortest) one.
        best = ((plus.Ta + plus.Tv + plus.Td) <= (minus.Ta + minus.Tv + minus.Td)) ? plus : minus;
    } else if (plus.ok) {
        best = plus;
    } else if (minus.ok) {
        best = minus;
    } else {
        return p; // success == false
    }

    p.Ar = best.Ar;
    p.Vr = best.Vr;
    p.Dr = best.Dr;
    p.Ta = best.Ta;
    p.Tv = best.Tv;
    p.Td = best.Td;
    p.Tf = best.Ta + best.Tv + best.Td;
    p.yAccel = Xi + (Vi * best.Ta) + (T{0.5} * best.Ar * best.Ta * best.Ta);
    p.success = true;
    return p;
}

/// Rest-to-rest convenience overload (Vi = Vf = 0), matching ODrive `planTrapezoidal`.
template<typename T = double>
[[nodiscard]] constexpr TrapezoidalProfile<T> synthesize_trapezoidal(
    T Xi, T Xf, const TrajectoryLimits<T>& limits
) {
    return synthesize_trapezoidal<T>(Xi, T{0}, Xf, T{0}, limits);
}

} // namespace design

/**
 * @ingroup trajectory
 * @brief Runtime evaluator for a precomputed trapezoidal profile.
 *
 * Holds a @ref design::TrapezoidalProfile and an internal clock. Call `step(dt)`
 * each control period to advance the clock and get the next commanded state, or
 * `eval(t)` for a stateless lookup at an arbitrary time. Allocation-free and
 * constexpr-constructible. Re-target by assigning a freshly synthesized profile.
 *
 * @tparam T Scalar type (float or double)
 */
template<typename T = float>
class TrapezoidalTrajectory {
public:
    constexpr TrapezoidalTrajectory() = default;

    /// Construct from a planned profile (typically `design::synthesize_trapezoidal(...).as<T>()`).
    constexpr explicit TrapezoidalTrajectory(const design::TrapezoidalProfile<T>& profile)
        : profile_(profile) {}

    /// Replace the profile and rewind the clock (re-targeting).
    constexpr void set_profile(const design::TrapezoidalProfile<T>& profile) {
        profile_ = profile;
        t_ = T{0};
    }

    /// Advance the clock by @p dt and return the commanded state at the new time.
    constexpr TrajectoryState<T> step(T dt) {
        t_ += dt;
        return profile_.eval(t_);
    }

    /// Stateless evaluation at absolute time @p t (does not move the clock).
    [[nodiscard]] constexpr TrajectoryState<T> eval(T t) const { return profile_.eval(t); }

    /// Rewind the internal clock to t = 0.
    constexpr void reset() { t_ = T{0}; }

    [[nodiscard]] constexpr T    time() const { return t_; }
    [[nodiscard]] constexpr T    duration() const { return profile_.Tf; }
    [[nodiscard]] constexpr bool done() const { return t_ >= profile_.Tf; }
    [[nodiscard]] constexpr bool valid() const { return profile_.success; }
    [[nodiscard]] constexpr T    peak_velocity() const { return profile_.Vr; }

    [[nodiscard]] constexpr auto profile() const -> const design::TrapezoidalProfile<T>& { return profile_; }

private:
    design::TrapezoidalProfile<T> profile_{};
    T                             t_{T{0}};
};

} // namespace wet
