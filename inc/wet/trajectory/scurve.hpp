#pragma once

/**
 * @file scurve.hpp
 * @brief Minimum-time jerk-limited double-S velocity profile (3rd-order, C²).
 *
 * The seven-phase bounded-jerk profile (jerk held at ±Jmax or 0): the acceleration
 * ramps instead of stepping, so the commanded reference is C² (continuous
 * acceleration) — no torque step to excite mechanical resonance. Same asymmetric
 * Amax/Dmax convention as @ref trapezoidal.hpp, plus a jerk bound Jmax; arbitrary
 * Vi/Vf. Converges to the trapezoidal as Jmax → ∞.
 *
 * @code
 * #include "wet/trajectory/scurve.hpp"
 * using namespace wet;
 *
 * constexpr TrajectoryLimits<double> lim{
 *     .max_velocity = 1.0, .max_acceleration = 2.0,
 *     .max_deceleration = 2.0, .max_jerk = 10.0
 * };
 * constexpr auto prof = design::synthesize_scurve(0.0, 1.0, lim);
 * static_assert(prof.success);
 *
 * ScurveTrajectory<double> traj(prof);
 * while (!traj.done()) {
 *     auto [pos, vel, acc, jerk] = traj.step(dt);
 * }
 * @endcode
 *
 * @see L. Biagiotti & C. Melchiorri, "Trajectory Planning for Automatic Machines
 *      and Robots," Springer, 2008, §3.4.
 * @see trapezoidal.hpp for the jerk-discontinuous (C¹) version.
 */

#include <cstddef>
#include <type_traits>

#include "wet/matrix/matrix_traits.hpp"
#include "wet/trajectory/trajectory_types.hpp"

namespace wet {

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

/// Rest-to-rest convenience overload.
template<typename T = double>
[[nodiscard]] constexpr ScurveProfile<T> synthesize_scurve(T Xi, T Xf, const TrajectoryLimits<T>& limits) {
    return synthesize_scurve<T>(Xi, T{0}, Xf, T{0}, limits);
}

} // namespace design

/**
 * @ingroup trajectory
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

    [[nodiscard]] constexpr T    time() const { return t_; }
    [[nodiscard]] constexpr T    duration() const { return profile_.duration; }
    [[nodiscard]] constexpr bool done() const { return t_ >= profile_.duration; }
    [[nodiscard]] constexpr bool valid() const { return profile_.success; }

    [[nodiscard]] constexpr const design::ScurveProfile<T>& profile() const { return profile_; }

private:
    design::ScurveProfile<T> profile_{};
    T                        t_{T{0}};
};

} // namespace wet
