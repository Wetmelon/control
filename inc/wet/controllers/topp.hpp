#pragma once

/**
 * @file topp.hpp
 * @brief Time-Optimal Path Parameterization (TOPP) — the pointwise time-optimal
 *        re-timing of a fixed task-space path under per-joint velocity and
 *        acceleration limits.
 *
 * The time-optimal counterpart to @ref CartesianMove (controllers/cartesian_move.hpp):
 * both keep the geometric path `p(s)` exact and only choose the *speed* `ṡ(t)`
 * along it, but where `CartesianMove` slows the whole timeline by a single global
 * factor `K` (so the one worst point is at its limit and everywhere else is
 * under-driven), TOPP drives *every* point of the path to its own velocity /
 * acceleration limit, so the traversal is genuinely minimum-time. The Jacobian /
 * inverse-kinematics layer (kinematics/serial_arm.hpp, the motion maps) supplies
 * the joint sensitivities the re-timing needs.
 *
 * **Method — reachability on a path grid (TOPP-RA flavour).** Reparameterize by
 * the squared path speed `x(s) = ṡ²`; then `dx/ds = 2·s̈`, and the per-joint
 * limits become *linear* in `(x, s̈)` at each `s`:
 *   - velocity: `|q'ᵢ(s)|·√x ≤ v_maxᵢ`  ⇒  `x ≤ (v_maxᵢ/|q'ᵢ|)²`,
 *   - acceleration: `−a_maxᵢ ≤ q'ᵢ(s)·s̈ + q''ᵢ(s)·x ≤ a_maxᵢ`.
 * On a uniform grid `s₀…s_{N−1}` this gives a velocity Maximum-Velocity-Curve and,
 * since high `x` on a curved path can make the acceleration interval empty, an
 * acceleration MVC (found per point by a monotone bisection on feasibility). A
 * forward pass accelerates maximally up to the MVC ceiling and a backward pass
 * decelerates into it; the result is the minimum-time `x(s)` profile honouring the
 * boundary speeds `(ṡ₀, ṡ_f)`. Segment times integrate exactly under the
 * constant-`s̈` (linear-`ṡ`) model, `Δt = 2Δs/(ṡᵢ + ṡᵢ₊₁)`.
 *
 * Allocation-free (fixed `NGrid` template grid) and constexpr-constructible; the
 * grid solve runs in the constructor (the `design::` stage), the runtime samples
 * `q(s(t))` with the chain rule, exactly as @ref CartesianMove.
 *
 * @see J. Bobrow, S. Dubowsky & J. Gibson, "Time-optimal control of robotic
 *      manipulators along specified paths," IJRR 4(3), 1985.
 * @see H. Pham & Q.-C. Pham, "A new approach to time-optimal path parameterization
 *      based on reachability analysis," IEEE T-RO 34(3), 2018 (TOPP-RA),
 *      https://doi.org/10.1109/TRO.2018.2819195.
 * @see K. Kunz & M. Stilman, and J. Slotine & H. Yang for the numerical-integration
 *      lineage this specializes (box velocity/acceleration constraints).
 */

#include "wet/controllers/cartesian_move.hpp" // JointLimits
#include "wet/controllers/trajectory.hpp"     // TrajectoryState
#include "wet/math/math.hpp"                  // wet::abs, sqrt, min, max

namespace wet {

/**
 * @brief The scalar time-optimal path-timing produced by TOPP.
 *
 * Carries the grid of path coordinate `sᵢ`, path speed `ṡᵢ`, and arrival time
 * `tᵢ`. Independent of the joints — it is the minimum-time `s(t)` schedule for the
 * supplied path and limits. Evaluate `s`, `ṡ`, `s̈` at a time with @ref at.
 *
 * @tparam NGrid Number of grid points (≥ 2)
 * @tparam T     Scalar type
 */
template<size_t NGrid, typename T = double>
struct ToppProfile {
    static_assert(NGrid >= 2, "TOPP needs at least two grid points");

    wet::array<T, NGrid> s{};    //!< path coordinate grid (uniform, s₀=0 … s_{N-1}=length)
    wet::array<T, NGrid> sdot{}; //!< path speed ṡ at each grid point
    wet::array<T, NGrid> time{}; //!< arrival time at each grid point
    T                    length{T{0}};
    T                    duration{T{0}};
    bool                 success{false};

    /// Path coordinate, speed, and acceleration at time @p t (clamped to [0, T]).
    [[nodiscard]] constexpr TrajectoryState<T> at(T t) const {
        TrajectoryState<T> out{};
        if (!success) {
            return out;
        }
        if (t <= T{0}) {
            out.position = s[0];
            out.velocity = sdot[0];
            return out;
        }
        if (t >= duration) {
            out.position = length;
            out.velocity = sdot[NGrid - 1];
            return out;
        }
        // Locate the segment [k, k+1] containing t (linear scan; grids are modest).
        size_t k = 0;
        while (k + 1 < NGrid && time[k + 1] <= t) {
            ++k;
        }
        const T dt = time[k + 1] - time[k];
        const T tau = t - time[k];
        const T sddot = (dt > T{0}) ? (sdot[k + 1] - sdot[k]) / dt : T{0}; // constant s̈ on the segment
        out.position = s[k] + (sdot[k] * tau) + (T{0.5} * sddot * tau * tau);
        out.velocity = sdot[k] + (sddot * tau);
        out.acceleration = sddot;
        return out;
    }
};

/**
 * @ingroup controllers
 * @brief Time-optimal task-space move (path-preserving, pointwise minimum-time).
 *
 * Same interface family as @ref CartesianMove — generic over a path callable and
 * an inverse-kinematics callable — but re-timed by TOPP rather than a global slow
 * factor. `eval(t)` returns the per-joint `{position, velocity, acceleration}`.
 *
 * @tparam NJoints Number of joints/actuators
 * @tparam NGrid   Path-grid resolution (more points ⇒ tighter to the true optimum)
 * @tparam PathFn  Callable `T s -> TaskPoint` over `s ∈ [0, length]`
 * @tparam IkFn    Callable `TaskPoint -> wet::pair<wet::array<T,NJoints>, bool>`
 * @tparam T       Scalar type
 */
template<size_t NJoints, size_t NGrid, typename PathFn, typename IkFn, typename T = double>
class ToppMove {
public:
    using StateArray = wet::array<TrajectoryState<T>, NJoints>;

    /**
     * @param path         Path geometry `p(s)`, `s ∈ [0, length]`.
     * @param ik           Inverse kinematics `p -> {joints, reachable}`.
     * @param length       Path arc length.
     * @param joint_limits Per-joint velocity / acceleration caps.
     * @param sdot_start   Path speed at the start (`ṡ₀`, default rest).
     * @param sdot_end     Path speed at the end (`ṡ_f`, default rest).
     */
    constexpr ToppMove(PathFn path, IkFn ik, T length, const JointLimits<NJoints, T>& joint_limits, T sdot_start = T{0}, T sdot_end = T{0})
        : path_(path), ik_(ik), length_(length), step_s_(length * static_cast<T>(1e-5)) {
        if (step_s_ < T{1e-9}) {
            step_s_ = T{1e-9};
        }
        valid_ = joint_limits.valid() && (length > T{0});
        if (!valid_) {
            return;
        }
        solve(joint_limits, sdot_start, sdot_end);
    }

    /// The scalar time-optimal path schedule `s(t)`.
    [[nodiscard]] constexpr const ToppProfile<NGrid, T>& profile() const { return profile_; }

    /// Evaluate every joint at path time @p t ∈ [0, duration].
    [[nodiscard]] constexpr StateArray eval(T t) const {
        StateArray out{};
        if (!valid_ || !profile_.success) {
            return out;
        }
        const TrajectoryState<T> ps = profile_.at(t); // {s, ṡ, s̈}
        JointDeriv               d{};
        joint_derivatives(ps.position, d);
        for (size_t i = 0; i < NJoints; ++i) {
            out[i].position = d.q[i];
            out[i].velocity = d.dq[i] * ps.velocity;
            out[i].acceleration = (d.ddq[i] * ps.velocity * ps.velocity) + (d.dq[i] * ps.acceleration);
        }
        return out;
    }

    /// Advance the clock by @p dt and return the per-joint states.
    constexpr StateArray step(T dt) {
        t_ += dt;
        return eval(t_);
    }

    constexpr void reset() { t_ = T{0}; }

    [[nodiscard]] constexpr T    time() const { return t_; }
    [[nodiscard]] constexpr T    duration() const { return profile_.duration; }
    [[nodiscard]] constexpr bool done() const { return t_ >= profile_.duration; }
    [[nodiscard]] constexpr bool valid() const { return valid_ && profile_.success; }
    [[nodiscard]] constexpr bool reachable() const { return reachable_; }

private:
    struct JointDeriv {
        wet::array<T, NJoints> q{};
        wet::array<T, NJoints> dq{};  // dq/ds
        wet::array<T, NJoints> ddq{}; // d²q/ds²
        bool                   reachable{true};
    };

    struct UBounds {
        T    lo{T{0}};
        T    hi{T{0}};
        bool feasible{false};
    };

    // Feasible path-acceleration interval s̈ ∈ [lo, hi] at curvature (dq, ddq) and
    // squared speed x, from the box acceleration limits |q'ᵢ·s̈ + q''ᵢ·x| ≤ a_maxᵢ.
    static constexpr UBounds accel_bounds(const JointDeriv& d, T x, const wet::array<T, NJoints>& amax) {
        UBounds b{};
        T       lo = -kBig;
        T       hi = kBig;
        for (size_t i = 0; i < NJoints; ++i) {
            const T c = d.ddq[i] * x;
            if (wet::abs(d.dq[i]) > kTiny) {
                const T r1 = (-amax[i] - c) / d.dq[i];
                const T r2 = (amax[i] - c) / d.dq[i];
                lo = max_of(lo, min_of(r1, r2));
                hi = min_of(hi, max_of(r1, r2));
            } else if (wet::abs(c) > amax[i]) {
                return b; // joint can't be held within accel limit at this x — infeasible
            }
        }
        b.lo = lo;
        b.hi = hi;
        b.feasible = lo <= hi;
        return b;
    }

    constexpr void solve(const JointLimits<NJoints, T>& lim, T sdot_start, T sdot_end) {
        const T ds = length_ / static_cast<T>(NGrid - 1);
        for (size_t i = 0; i < NGrid; ++i) {
            profile_.s[i] = (i == NGrid - 1) ? length_ : (static_cast<T>(i) * ds);
        }
        profile_.length = length_;

        // Per-point curvature, velocity MVC, and acceleration MVC (bisection).
        wet::array<JointDeriv, NGrid> deriv{};
        wet::array<T, NGrid>          mvc{};
        for (size_t i = 0; i < NGrid; ++i) {
            joint_derivatives(profile_.s[i], deriv[i]);
            reachable_ = reachable_ && deriv[i].reachable;
            T xv = kBig;
            for (size_t j = 0; j < NJoints; ++j) {
                if (wet::abs(deriv[i].dq[j]) > kTiny) {
                    const T xj = (lim.max_velocity[j] * lim.max_velocity[j]) / (deriv[i].dq[j] * deriv[i].dq[j]);
                    xv = min_of(xv, xj);
                }
            }
            mvc[i] = accel_mvc(deriv[i], xv, lim.max_acceleration);
        }

        // x = ṡ²: start/end speeds, capped to the curve.
        wet::array<T, NGrid> x{};
        for (size_t i = 0; i < NGrid; ++i) {
            x[i] = mvc[i];
        }
        x[0] = min_of(x[0], sdot_start * sdot_start);
        x[NGrid - 1] = min_of(x[NGrid - 1], sdot_end * sdot_end);

        // Forward pass: accelerate maximally, never above the MVC ceiling.
        for (size_t i = 0; i + 1 < NGrid; ++i) {
            const UBounds b = accel_bounds(deriv[i], x[i], lim.max_acceleration);
            const T       cand = b.feasible ? (x[i] + (T{2} * b.hi * ds)) : x[i];
            x[i + 1] = clamp_nonneg(min_of(x[i + 1], cand));
        }
        // Backward pass: ensure each point can decelerate into its successor.
        x[NGrid - 1] = min_of(x[NGrid - 1], mvc[NGrid - 1]);
        x[NGrid - 1] = min_of(x[NGrid - 1], sdot_end * sdot_end);
        for (size_t i = NGrid - 1; i > 0; --i) {
            const UBounds b = accel_bounds(deriv[i], x[i], lim.max_acceleration);
            const T       cand = b.feasible ? (x[i] - (T{2} * b.lo * ds)) : kBig; // b.lo ≤ 0 ⇒ raises the bound
            x[i - 1] = clamp_nonneg(min_of(x[i - 1], cand));
        }

        // Speeds and segment times (exact under constant-s̈ / linear-ṡ).
        for (size_t i = 0; i < NGrid; ++i) {
            profile_.sdot[i] = wet::sqrt(x[i]);
        }
        profile_.time[0] = T{0};
        for (size_t i = 0; i + 1 < NGrid; ++i) {
            const T vsum = profile_.sdot[i] + profile_.sdot[i + 1];
            const T dt = (vsum > kTiny) ? (T{2} * ds / vsum) : kBig;
            profile_.time[i + 1] = profile_.time[i] + dt;
        }
        profile_.duration = profile_.time[NGrid - 1];
        profile_.success = profile_.duration < kBig && profile_.duration > T{0};
    }

    // Largest x ∈ [0, x_vel] with a non-empty acceleration interval (monotone
    // feasibility: x = 0 is always feasible, the curvature term shrinks it).
    static constexpr T accel_mvc(const JointDeriv& d, T x_vel, const wet::array<T, NJoints>& amax) {
        if (x_vel <= T{0}) {
            return T{0};
        }
        if (accel_bounds(d, x_vel, amax).feasible) {
            return x_vel;
        }
        T lo = T{0};
        T hi = x_vel;
        for (int it = 0; it < 50; ++it) {
            const T mid = T{0.5} * (lo + hi);
            if (accel_bounds(d, mid, amax).feasible) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    // q(s), q'(s), q''(s) via central differences of IK along the path. The
    // stencil centre is nudged inward so `[c−h, c+h]` stays within `[0, length]`:
    // clamping only one side while still dividing by the full `2h` would halve the
    // derivative magnitude at the endpoints — and at a path end that halving would
    // *double* the acceleration bound there. Since q'(s)/q''(s) are continuous, the
    // ≈`1e-5·length` centre shift is negligible.
    constexpr void joint_derivatives(T s, JointDeriv& d) const {
        const T h = step_s_;
        T       c = s;
        if (c < h) {
            c = h;
        }
        if (c > length_ - h) {
            c = length_ - h;
        }
        const auto qm = ik_(path_(c - h));
        const auto q0 = ik_(path_(c));
        const auto qp = ik_(path_(c + h));
        // Position is reported at the *true* s (the shifted centre only serves the
        // derivative stencil); only the endpoints incur the extra IK evaluation.
        const T    s_true = clamp_s(s);
        const auto qt = (c == s_true) ? q0 : ik_(path_(s_true));
        d.reachable = qm.second && q0.second && qp.second && qt.second;
        const T inv2h = T{1} / (T{2} * h);
        const T invhh = T{1} / (h * h);
        for (size_t i = 0; i < NJoints; ++i) {
            d.q[i] = qt.first[i];
            d.dq[i] = (qp.first[i] - qm.first[i]) * inv2h;
            d.ddq[i] = (qp.first[i] - (T{2} * q0.first[i]) + qm.first[i]) * invhh;
        }
    }

    [[nodiscard]] constexpr T clamp_s(T s) const {
        if (s < T{0}) {
            return T{0};
        }
        if (s > length_) {
            return length_;
        }
        return s;
    }

    [[nodiscard]] static constexpr T min_of(T a, T b) { return (a < b) ? a : b; }
    [[nodiscard]] static constexpr T max_of(T a, T b) { return (a > b) ? a : b; }
    [[nodiscard]] static constexpr T clamp_nonneg(T a) { return (a < T{0}) ? T{0} : a; }

    static constexpr T kBig = T{1e18};
    static constexpr T kTiny = T{1e-12};

    PathFn                path_;
    IkFn                  ik_;
    ToppProfile<NGrid, T> profile_{};
    T                     length_{T{0}};
    T                     step_s_{T{1e-6}};
    T                     t_{T{0}};
    bool                  valid_{false};
    bool                  reachable_{true};
};

/// Deduction-friendly factory (deduces NJoints from @p joint_limits; pick NGrid).
template<size_t NGrid, size_t NJoints, typename PathFn, typename IkFn, typename T>
[[nodiscard]] constexpr ToppMove<NJoints, NGrid, PathFn, IkFn, T> make_topp_move(
    PathFn path, IkFn ik, T length, const JointLimits<NJoints, T>& joint_limits,
    T sdot_start = T{0}, T sdot_end = T{0}
) {
    return ToppMove<NJoints, NGrid, PathFn, IkFn, T>(path, ik, length, joint_limits, sdot_start, sdot_end);
}

} // namespace wet
