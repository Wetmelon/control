#pragma once

/**
 * @file trajectory.hpp
 * @brief Point-to-point motion-profile generators — the feedforward
 *        position/velocity/acceleration(/jerk) reference an actuator or axis
 *        tracks. Three families, all precomputed (a constexpr `design::` stage
 *        solves the profile once with a `success` flag) and replayed by an
 *        allocation-free runtime via `step(dt)` / `eval(t)`:
 *
 *  1. **Trapezoidal** (velocity/acceleration-limited, 2nd order).
 *     @ref design::synthesize_trapezoidal → @ref TrapezoidalTrajectory.
 *     Minimum-time three-segment profile (accelerate / cruise / decelerate) from
 *     (Xi, Vi) to (Xf, Vf). *Asymmetric* limits: `Amax` bounds the first ramp,
 *     `Dmax` the last — the classic machine-motion convention (gentle accel, hard
 *     brake, or vice versa). Acceleration is piecewise-constant (jerk unbounded).
 *     Accepts any Vi (over-speed / wrong-direction). With `Amax == Dmax`, `Vf == 0`
 *     this is the symmetric ODrive `planTrapezoidal`.
 *
 *  2. **Jerk-limited double-S** (S-curve, 3rd order).
 *     @ref design::synthesize_scurve → @ref ScurveTrajectory. The seven-phase
 *     bounded-jerk profile (jerk held at ±Jmax or 0): the acceleration ramps
 *     instead of stepping, so the reference is C² (continuous acceleration) — no
 *     torque step to excite vibration. Same asymmetric Amax/Dmax convention plus a
 *     jerk bound Jmax; arbitrary Vi/Vf. Converges to the trapezoidal as Jmax → ∞.
 *
 *  3. **Polynomial, fixed-time / derivative-optimal.**
 *     @ref design::synthesize_poly_trajectory<Order> → @ref PolynomialTrajectory.
 *     Solves the Vandermonde boundary-value problem (via @ref mat::solve) for a
 *     cubic/quintic/septic matching position + derivatives at both ends over a
 *     *fixed* duration T. Convenience wrappers @ref design::min_accel /
 *     @ref design::min_jerk / @ref design::min_snap give the rest-to-rest
 *     derivative-optimal moves (Flash–Hogan minimum jerk, Mellinger–Kumar minimum
 *     snap). Evaluates {position, velocity, acceleration, jerk}.
 *
 * All planning is feed-forward and open-loop in time — re-plan when the target
 * changes. Pair with an input shaper (controllers/input_shaper.hpp) to suppress
 * residual vibration (most relevant for the jerk-discontinuous trapezoidal).
 *
 * @see L. Biagiotti & C. Melchiorri, "Trajectory Planning for Automatic Machines
 *      and Robots," Springer, 2008 (trapezoidal, double-S and polynomial — the
 *      canonical treatment).
 * @see R. Béarée, "FIR filter-based online jerk-constrained trajectory
 *      generation" — the asymmetric trapezoidal / double-S structure used here
 *      (also the basis for ODrive's symmetric rest-to-rest `planTrapezoidal`).
 * @see T. Flash & N. Hogan (1985, minimum jerk); D. Mellinger & V. Kumar (ICRA
 *      2011, minimum snap).
 */

#include <cstddef>
#include <type_traits>

#include "wet/math/math.hpp"
#include "wet/matrix/matrix.hpp"
#include "wet/matrix/matrix_traits.hpp"
#include "wet/matrix/solve.hpp"

namespace wet {

/**
 * @brief Asymmetric kinematic limits for a trapezoidal motion profile.
 *
 * @p max_acceleration bounds the approach-to-cruise ramp and @p max_deceleration
 * the approach-to-target ramp (they may differ). All three must be > 0.
 *
 * @tparam T Scalar type
 */
template<typename T = double>
struct TrajectoryLimits {

    T max_velocity{T{0}};     //!< |v| ≤ max_velocity (> 0)
    T max_acceleration{T{0}}; //!< accel-region bound Amax (> 0)
    T max_deceleration{T{0}}; //!< decel-region bound Dmax (> 0)
    T max_jerk{T{0}};         //!< |j| ≤ max_jerk (> 0; S-curve only, ignored by trapezoidal)

    /// Trapezoidal validity: positive v/a/d limits (jerk is not required).
    [[nodiscard]] constexpr bool valid() const {
        return (max_velocity > T{0}) && (max_acceleration > T{0}) && (max_deceleration > T{0});
    }

    /// S-curve validity: additionally requires a positive jerk limit.
    [[nodiscard]] constexpr bool valid_jerk_limited() const { return valid() && (max_jerk > T{0}); }
};

/**
 * @brief A point on a motion profile: commanded position, velocity, acceleration.
 * @tparam T Scalar type
 */
template<typename T = double>
struct TrajectoryState {

    T position{T{0}};
    T velocity{T{0}};
    T acceleration{T{0}};
    T jerk{T{0}}; //!< Only meaningful for jerk-continuous profiles (polynomial)
};

/**
 * @brief Boundary conditions at one endpoint of a polynomial trajectory: a
 *        position and its time derivatives.
 *
 * How many derivatives are honored depends on the polynomial order: cubic uses
 * {position, velocity}; quintic adds acceleration; septic adds jerk.
 *
 * @tparam T Scalar type
 */
template<typename T = double>
struct TrajectoryBoundary {

    T position{T{0}};
    T velocity{T{0}};
    T acceleration{T{0}};
    T jerk{T{0}};
};

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
    T Xi,
    T Vi,
    T Xf,
    T Vf,

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
 * @ingroup controllers
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

    [[nodiscard]] constexpr auto time() const -> T { return t_; }
    [[nodiscard]] constexpr auto duration() const -> T { return profile_.Tf; }
    [[nodiscard]] constexpr auto done() const -> bool { return t_ >= profile_.Tf; }
    [[nodiscard]] constexpr auto valid() const -> bool { return profile_.success; }
    [[nodiscard]] constexpr auto peak_velocity() const -> T { return profile_.Vr; }

    [[nodiscard]] constexpr auto profile() const -> const design::TrapezoidalProfile<T>& { return profile_; }

private:
    design::TrapezoidalProfile<T> profile_{};
    T                             t_{T{0}};
};

namespace design {

/// Maximum number of constant-jerk phases in a double-S profile.
inline constexpr size_t kScurveMaxSegments = 7;

/// One constant-jerk segment, valid for t ∈ [t0, t0 + duration), with the
/// position/velocity/acceleration cached at the segment start.
template<typename T = double>
struct ScurveSegment {
    T t0{0};
    T duration{0};
    T p0{0};
    T v0{0};
    T a0{0};
    T jerk{0};
};

/**
 * @brief A synthesized jerk-limited (double-S) profile: a sequence of
 *        constant-jerk segments, evaluated exactly (cubic in t within a segment).
 *
 * @tparam T Scalar type
 */
template<typename T = double>
struct ScurveProfile {

    wet::array<ScurveSegment<T>, kScurveMaxSegments> segments{};

    size_t count{0};
    T      duration{T{0}};
    T      Xi{T{0}};
    T      Vi{T{0}};
    T      Xf{T{0}};
    T      Vf{T{0}};
    bool   success{false};

    /// Evaluate position/velocity/acceleration/jerk at time @p t (clamped to [0, T]).
    [[nodiscard]] constexpr TrajectoryState<T> eval(T t) const {
        TrajectoryState<T> s{};
        if (t <= T{0} || count == 0) {
            s.position = Xi;
            s.velocity = Vi;
            return s;
        }
        if (t >= duration) {
            s.position = Xf;
            s.velocity = Vf;
            return s;
        }
        // Locate the active segment (≤ 7, so a linear scan is fine).
        size_t i = 0;
        while (i + 1 < count && t >= segments[i + 1].t0) {
            ++i;
        }
        const ScurveSegment<T>& seg = segments[i];
        const T                 tau = t - seg.t0;
        s.position = seg.p0 + (seg.v0 * tau) + (T{0.5} * seg.a0 * tau * tau) + (seg.jerk * tau * tau * tau / T{6});
        s.velocity = seg.v0 + (seg.a0 * tau) + (T{0.5} * seg.jerk * tau * tau);
        s.acceleration = seg.a0 + (seg.jerk * tau);
        s.jerk = seg.jerk;
        return s;
    }

    /// Rebind the profile to another scalar type (e.g. plan in double, run in float).
    template<typename U>
    [[nodiscard]] constexpr ScurveProfile<std::remove_const_t<U>> as() const {
        using O = std::remove_const_t<U>;
        ScurveProfile<O> out{};
        for (size_t i = 0; i < count; ++i) {
            out.segments[i] = ScurveSegment<O>{static_cast<O>(segments[i].t0), static_cast<O>(segments[i].duration), static_cast<O>(segments[i].p0), static_cast<O>(segments[i].v0), static_cast<O>(segments[i].a0), static_cast<O>(segments[i].jerk)};
        }
        out.count = count;
        out.duration = static_cast<O>(duration);
        out.Xi = static_cast<O>(Xi);
        out.Vi = static_cast<O>(Vi);
        out.Xf = static_cast<O>(Xf);
        out.Vf = static_cast<O>(Vf);
        out.success = success;
        return out;
    }
};

namespace detail {

/// Sub-phase times of one S-shaped velocity region taking v_start → v_end under a
/// peak-acceleration magnitude @p L and jerk magnitude @p J.
template<typename T>
struct ScurveRegion {
    T jerk_time{0};  //!< Tj — each of the two jerk ramps
    T const_time{0}; //!< the constant-acceleration plateau (0 if L is never reached)
    T total{0};      //!< Tj + const_time + Tj
};

/// Plan one acceleration/deceleration region. Works for v_end above or below
/// v_start; displacement over the region is exactly total·(v_start + v_end)/2.
template<typename T>
constexpr ScurveRegion<T> plan_scurve_region(T v_start, T v_end, T L, T J) {
    ScurveRegion<T> r{};
    const T         dv = v_end - v_start;
    const T         mag = (dv < T{0}) ? -dv : dv;
    if (mag * J <= L * L) { // L never reached: triangular acceleration
        r.jerk_time = wet::sqrt(mag / J);
        r.const_time = T{0};
    } else { // L reached: trapezoidal acceleration
        r.jerk_time = L / J;
        r.const_time = (mag / L) - (L / J);
    }
    r.total = (T{2} * r.jerk_time) + r.const_time;
    return r;
}

/// Total displacement of the two regions for a junction (peak/valley) velocity @p vp.
template<typename T>
constexpr T scurve_region_displacement(T v0, T vp, T v1, T Amax, T Dmax, T J) {
    const T t_acc = plan_scurve_region<T>(v0, vp, Amax, J).total;
    const T t_dec = plan_scurve_region<T>(vp, v1, Dmax, J).total;
    return ((v0 + vp) * T{0.5} * t_acc) + ((vp + v1) * T{0.5} * t_dec);
}

} // namespace detail

/**
 * @brief Synthesize a minimum-time jerk-limited (7-segment double-S) profile from
 *        (Xi, Vi) to (Xf, Vf) under asymmetric kinematic limits.
 *
 * Jerk is held at ±max_jerk (or 0) throughout, giving the seven phases
 * accel-jerk / accel-const / accel-jerk / cruise / decel-jerk / decel-const /
 * decel-jerk; any phase may vanish (acceleration plateau not reached, no cruise,
 * etc.). The accel region is bounded by `max_acceleration`, the decel region by
 * `max_deceleration`; both use `max_jerk`. The runtime integrates each
 * constant-jerk segment exactly, so position/velocity are continuous *and so is
 * acceleration* (C²) — the point of the S-curve. As max_jerk → ∞ it converges to
 * @ref synthesize_trapezoidal.
 *
 * The junction (peak) velocity is found by bisection on the region displacement,
 * which brackets a sign change over [−Vmax, +Vmax]; boundary conditions and limit
 * compliance hold by construction.
 *
 * @tparam T      Scalar type
 * @param Xi      Start position
 * @param Vi      Start velocity (|Vi| may exceed Vmax — it brakes into range)
 * @param Xf      Target position
 * @param Vf      Target velocity (|Vf| ≤ max_velocity)
 * @param limits  Limits (Vmax, Amax, Dmax, Jmax); requires `valid_jerk_limited()`
 * @return Profile with `success`; false on invalid limits, |Vf| > Vmax.
 */
template<typename T = double>
[[nodiscard]] constexpr ScurveProfile<T> synthesize_scurve(
    T Xi, T Vi, T Xf, T Vf, const TrajectoryLimits<T>& limits
) {
    ScurveProfile<T> prof{};
    prof.Xi = Xi;
    prof.Vi = Vi;
    prof.Xf = Xf;
    prof.Vf = Vf;

    const T Vmax = limits.max_velocity;
    const T Amax = limits.max_acceleration;
    const T Dmax = limits.max_deceleration;
    const T J = limits.max_jerk;
    const T eps = wet::default_tol<T>();
    if (!limits.valid_jerk_limited() || wet::abs(Vf) > Vmax + eps) {
        return prof; // success == false
    }

    const T h = Xf - Xi;
    if (wet::abs(h) <= eps && wet::abs(Vi - Vf) <= eps) {
        prof.success = true; // already at the target state; empty (zero-duration) profile
        return prof;
    }

    // Pick the junction velocity. Long move: cruise at ±Vmax (the regions alone
    // under/over-shoot the distance). Otherwise the regions must reproduce h with
    // no cruise — bisect the junction velocity over [−Vmax, +Vmax], where the
    // region displacement brackets a sign change.
    const T dX_pos = detail::scurve_region_displacement<T>(Vi, Vmax, Vf, Amax, Dmax, J);
    const T dX_neg = detail::scurve_region_displacement<T>(Vi, -Vmax, Vf, Amax, Dmax, J);
    T       peak{};
    T       Tv{};
    if (h >= dX_pos) { // cruise forward at +Vmax
        peak = Vmax;
        Tv = (h - dX_pos) / Vmax;
    } else if (h <= dX_neg) { // cruise backward at −Vmax
        peak = -Vmax;
        Tv = (h - dX_neg) / (-Vmax);
    } else {          // no cruise: solve the junction velocity by bisection
        T lo = -Vmax; // displacement(lo) − h < 0
        T hi = Vmax;  // displacement(hi) − h > 0
        for (int it = 0; it < 100; ++it) {
            const T mid = (lo + hi) * T{0.5};
            if (detail::scurve_region_displacement<T>(Vi, mid, Vf, Amax, Dmax, J) - h > T{0}) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        peak = (lo + hi) * T{0.5};
        Tv = T{0};
    }

    // Build the seven phases (durations and signed jerks), integrating the state.
    const detail::ScurveRegion<T> acc = detail::plan_scurve_region<T>(Vi, peak, Amax, J);
    const detail::ScurveRegion<T> dec = detail::plan_scurve_region<T>(peak, Vf, Dmax, J);
    const T                       ja = static_cast<T>(wet::sgn(peak - Vi)) * J;
    const T                       jd = static_cast<T>(wet::sgn(Vf - peak)) * J;
    const T                       durs[kScurveMaxSegments] = {acc.jerk_time, acc.const_time, acc.jerk_time, Tv, dec.jerk_time, dec.const_time, dec.jerk_time};
    const T                       jerks[kScurveMaxSegments] = {ja, T{0}, -ja, T{0}, jd, T{0}, -jd};

    T      p = Xi;
    T      v = Vi;
    T      a = T{0};
    T      tnow = T{0};
    size_t n = 0;
    for (size_t i = 0; i < kScurveMaxSegments; ++i) {
        const T tau = durs[i];
        if (tau <= eps) {
            continue; // skip negligible phases (state is unchanged)
        }
        const T jk = jerks[i];
        prof.segments[n] = ScurveSegment<T>{tnow, tau, p, v, a, jk};
        ++n;
        p = p + (v * tau) + (T{0.5} * a * tau * tau) + (jk * tau * tau * tau / T{6});
        v = v + (a * tau) + (T{0.5} * jk * tau * tau);
        a = a + (jk * tau);
        tnow += tau;
    }
    prof.count = n;
    prof.duration = tnow;
    prof.success = true;
    return prof;
}

/// Rest-to-rest convenience overload, matching the roadmap's
/// `synthesize_scurve(distance, v_max, a_max, j_max)` intent.
template<typename T = double>
[[nodiscard]] constexpr ScurveProfile<T> synthesize_scurve(T Xi, T Xf, const TrajectoryLimits<T>& limits) {
    return synthesize_scurve<T>(Xi, T{0}, Xf, T{0}, limits);
}

} // namespace design

/**
 * @ingroup controllers
 * @brief Runtime evaluator for a precomputed jerk-limited (double-S) profile.
 *
 * Holds a @ref design::ScurveProfile and an internal clock. `step(dt)` advances
 * and returns {position, velocity, acceleration, jerk}; `eval(t)` is a stateless
 * lookup. Allocation-free and constexpr-constructible.
 *
 * @tparam T Scalar type (float or double)
 */
template<typename T = float>
class ScurveTrajectory {
public:
    constexpr ScurveTrajectory() = default;

    constexpr explicit ScurveTrajectory(const design::ScurveProfile<T>& profile) : profile_(profile) {}

    /// Replace the profile and rewind the clock.
    constexpr void set_profile(const design::ScurveProfile<T>& profile) {
        profile_ = profile;
        t_ = T{0};
    }

    /// Advance the clock by @p dt and return the commanded state.
    constexpr TrajectoryState<T> step(T dt) {
        t_ += dt;
        return profile_.eval(t_);
    }

    /// Stateless evaluation at absolute time @p t (does not move the clock).
    [[nodiscard]] constexpr TrajectoryState<T> eval(T t) const { return profile_.eval(t); }

    constexpr void reset() { t_ = T{0}; }

    [[nodiscard]] constexpr T                               time() const { return t_; }
    [[nodiscard]] constexpr T                               duration() const { return profile_.duration; }
    [[nodiscard]] constexpr bool                            done() const { return t_ >= profile_.duration; }
    [[nodiscard]] constexpr bool                            valid() const { return profile_.success; }
    [[nodiscard]] constexpr const design::ScurveProfile<T>& profile() const { return profile_; }

private:
    design::ScurveProfile<T> profile_{};
    T                        t_{T{0}};
};

namespace design {

/**
 * @brief A synthesized polynomial trajectory: the coefficients of
 *        p(t) = Σ cᵢ·tⁱ over t ∈ [0, T], plus the duration.
 *
 * @tparam Order Polynomial order (3 cubic, 5 quintic, 7 septic).
 * @tparam T     Scalar type
 */
template<size_t Order, typename T = double>
struct PolyTrajectory {

    static constexpr size_t NumCoeffs = Order + 1;

    wet::array<T, NumCoeffs> coeffs{}; //!< c₀ … c_Order (ascending power)
    T                        duration{T{0}};
    bool                     success{false};

    /// Evaluate position/velocity/acceleration/jerk at time @p t (clamped to [0, T]).
    [[nodiscard]] constexpr TrajectoryState<T> eval(T t) const {
        T tc = t; // clamp to [0, T]; the polynomial extrapolates wildly outside
        if (tc < T{0}) {
            tc = T{0};
        } else if (tc > duration) {
            tc = duration;
        }
        // Horner-with-derivatives (synthetic division): b → p, b1 → p′,
        // b2 → p″/2!, b3 → p‴/3!, accumulated from the highest power down.
        T b = coeffs[NumCoeffs - 1];
        T b1{0};
        T b2{0};
        T b3{0};
        for (size_t i = NumCoeffs - 1; i-- > 0;) {
            b3 = (b3 * tc) + b2;
            b2 = (b2 * tc) + b1;
            b1 = (b1 * tc) + b;
            b = (b * tc) + coeffs[i];
        }
        TrajectoryState<T> s{};
        s.position = b;
        s.velocity = b1;
        s.acceleration = T{2} * b2;
        s.jerk = T{6} * b3;
        return s;
    }

    /// Rebind the profile to another scalar type (e.g. plan in double, run in float).
    template<typename U>
    [[nodiscard]] constexpr PolyTrajectory<Order, std::remove_const_t<U>> as() const {
        using O = std::remove_const_t<U>;
        PolyTrajectory<Order, O> out{};
        for (size_t i = 0; i < NumCoeffs; ++i) {
            out.coeffs[i] = static_cast<O>(coeffs[i]);
        }
        out.duration = static_cast<O>(duration);
        out.success = success;
        return out;
    }
};

namespace detail {

/// k! (k ≤ 7 here, fits any scalar exactly).
template<typename T>
constexpr T factorial(size_t k) {
    T f{1};
    for (size_t i = 2; i <= k; ++i) {
        f *= static_cast<T>(i);
    }
    return f;
}

/// Falling factorial i·(i−1)···(i−k+1) = i! / (i−k)! — the k-th derivative
/// coefficient of tⁱ. Zero when i < k.
template<typename T>
constexpr T falling_factorial(size_t i, size_t k) {
    if (i < k) {
        return T{0};
    }
    T f{1};
    for (size_t m = 0; m < k; ++m) {
        f *= static_cast<T>(i - m);
    }
    return f;
}

} // namespace detail

/**
 * @brief Synthesize a fixed-duration polynomial trajectory matching boundary
 *        conditions on position and its derivatives at both endpoints.
 *
 * Solves the Vandermonde-style boundary-value problem for the coefficients of
 * p(t) = Σ cᵢ·tⁱ on t ∈ [0, T]. The number of conditions per endpoint is
 * (Order+1)/2: cubic matches {p, v}, quintic {p, v, a}, septic {p, v, a, j}.
 * The (Order+1)×(Order+1) linear system is solved with @ref mat::solve.
 *
 * Derivative-optimality is implied by the boundary conditions: a quintic with
 * zero velocity/acceleration at both ends is the minimum-jerk profile
 * (Flash–Hogan); a cubic with zero end velocities is minimum-acceleration; a
 * septic with zero end jerk is minimum-snap (Mellinger–Kumar). See the
 * @ref min_jerk / @ref min_accel / @ref min_snap convenience wrappers.
 *
 * @tparam Order   Odd polynomial order: 3 (cubic), 5 (quintic) or 7 (septic).
 * @tparam T       Scalar type
 * @param bc_start Boundary conditions at t = 0
 * @param bc_end   Boundary conditions at t = T
 * @param duration Move duration T [s] (> 0)
 * @return Coefficients with `success`; `success == false` on T ≤ 0 or a
 *         singular system.
 */
template<size_t Order, typename T = double>
[[nodiscard]] constexpr PolyTrajectory<Order, T> synthesize_poly_trajectory(
    const TrajectoryBoundary<T>& bc_start,
    const TrajectoryBoundary<T>& bc_end,
    T                            duration
) {
    static_assert(Order % 2 == 1, "Order must be odd (3 cubic, 5 quintic, 7 septic)");
    static_assert(Order <= 7, "Boundary conditions go up to jerk, so Order ≤ 7 (septic)");
    constexpr size_t N = Order + 1;
    constexpr size_t D = N / 2; // conditions per endpoint (p plus D−1 derivatives)

    PolyTrajectory<Order, T> p{};
    p.duration = duration;
    if (duration <= T{0}) {
        return p; // success == false
    }

    const T s[4] = {bc_start.position, bc_start.velocity, bc_start.acceleration, bc_start.jerk};
    const T e[4] = {bc_end.position, bc_end.velocity, bc_end.acceleration, bc_end.jerk};

    // Row r is the r-th boundary condition; column i the coefficient cᵢ.
    // M(r, i) = dᵏ/dtᵏ (tⁱ) |_(t = endpoint). At t = 0 only i == k survives.
    Matrix<N, N, T> M{};
    Matrix<N, 1, T> rhs{};
    for (size_t k = 0; k < D; ++k) {
        M(k, k) = detail::factorial<T>(k); // start endpoint (t = 0)
        rhs(k, 0) = s[k];
        for (size_t i = k; i < N; ++i) { // end endpoint (t = T)
            M(D + k, i) = detail::falling_factorial<T>(i, k) * wet::pow(duration, static_cast<int>(i - k));
        }
        rhs(D + k, 0) = e[k];
    }

    const auto x = mat::solve(M, rhs);
    if (!x) {
        return p; // singular -> success == false
    }
    for (size_t i = 0; i < N; ++i) {
        p.coeffs[i] = x.value()(i, 0);
    }
    p.success = true;
    return p;
}

/// Minimum-jerk (quintic) rest-to-rest move p0 → pT over duration T (Flash–Hogan).
template<typename T = double>
[[nodiscard]] constexpr PolyTrajectory<5, T> min_jerk(T p0, T pT, T duration) {
    return synthesize_poly_trajectory<5, T>(TrajectoryBoundary<T>{p0}, TrajectoryBoundary<T>{pT}, duration);
}

/// Minimum-acceleration (cubic) rest-to-rest move p0 → pT over duration T.
template<typename T = double>
[[nodiscard]] constexpr PolyTrajectory<3, T> min_accel(T p0, T pT, T duration) {
    return synthesize_poly_trajectory<3, T>(TrajectoryBoundary<T>{p0}, TrajectoryBoundary<T>{pT}, duration);
}

/// Minimum-snap (septic) rest-to-rest move p0 → pT over duration T (Mellinger–Kumar).
template<typename T = double>
[[nodiscard]] constexpr PolyTrajectory<7, T> min_snap(T p0, T pT, T duration) {
    return synthesize_poly_trajectory<7, T>(TrajectoryBoundary<T>{p0}, TrajectoryBoundary<T>{pT}, duration);
}

} // namespace design

/**
 * @ingroup controllers
 * @brief Runtime evaluator for a precomputed polynomial trajectory.
 *
 * Holds a @ref design::PolyTrajectory and an internal clock. Call `step(dt)`
 * each control period to advance and read the next {position, velocity,
 * acceleration, jerk}, or `eval(t)` for a stateless lookup. Allocation-free and
 * constexpr-constructible.
 *
 * @tparam Order Polynomial order (3, 5 or 7).
 * @tparam T     Scalar type (float or double)
 */
template<size_t Order, typename T = float>
class PolynomialTrajectory {
public:
    constexpr PolynomialTrajectory() = default;

    constexpr explicit PolynomialTrajectory(const design::PolyTrajectory<Order, T>& profile)
        : profile_(profile) {}

    /// Replace the profile and rewind the clock.
    constexpr void set_profile(const design::PolyTrajectory<Order, T>& profile) {
        profile_ = profile;
        t_ = T{0};
    }

    /// Advance the clock by @p dt and return the commanded state.
    constexpr TrajectoryState<T> step(T dt) {
        t_ += dt;
        return profile_.eval(t_);
    }

    /// Stateless evaluation at absolute time @p t (does not move the clock).
    [[nodiscard]] constexpr TrajectoryState<T> eval(T t) const { return profile_.eval(t); }

    constexpr void reset() { t_ = T{0}; }

    [[nodiscard]] constexpr T                                       time() const { return t_; }
    [[nodiscard]] constexpr T                                       duration() const { return profile_.duration; }
    [[nodiscard]] constexpr bool                                    done() const { return t_ >= profile_.duration; }
    [[nodiscard]] constexpr bool                                    valid() const { return profile_.success; }
    [[nodiscard]] constexpr const design::PolyTrajectory<Order, T>& profile() const { return profile_; }

private:
    design::PolyTrajectory<Order, T> profile_{};
    T                                t_{T{0}};
};

} // namespace wet
