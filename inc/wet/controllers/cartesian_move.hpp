#pragma once

/**
 * @file cartesian_move.hpp
 * @brief Path-preserving task-space ("Cartesian" / "LIN") move: drive every joint
 *        from a single shared path clock so the tool follows the commanded path
 *        exactly, globally time-scaled so no joint exceeds its limit.
 *
 * The contrast with @ref TrajectoryBank (joint-space PTP / "MoveJ"): the bank
 * plans each axis independently and only syncs their *durations*, so the tool
 * traces whatever curve interpolating the joints produces — the path is **not**
 * preserved. `CartesianMove` keeps the path by construction: one scalar profile
 * `s(t)` along the path arc length parameterizes a fixed geometry `p(s)`, and
 * every joint is `qᵢ(s) = IK(p(s))`. Re-timing `s(t)` only changes the speed
 * along the path, never the path itself.
 *
 * Limit handling is a single **global** time-scale `K` (no Jacobian needed):
 * plan a nominal `s(t)` from the path-space limits, sample the induced joint
 * velocities/accelerations along the path, and slow the whole timeline by
 * `K = max(1, max|q̇ᵢ|/v_maxᵢ, √(max|q̈ᵢ|/a_maxᵢ))`. Because time-scaling shrinks
 * joint velocity ∝ 1/K and acceleration ∝ 1/K², this makes the single
 * worst-constrained point exactly hit its limit and the path stay exact. It is
 * paced by that worst point (a straight `s`-line crawls through a singularity);
 * the time-optimal pointwise version is TOPP, a planned extension.
 *
 * `K` is estimated from a finite sweep of `samples` points, so a *narrow* joint-rate
 * peak (e.g. skimming a singularity) can fall between samples and slightly
 * under-derate; raise `samples` (or add margin to the joint limits) when the path
 * grazes a singularity. Far from singularities the default resolution is ample.
 *
 * Runtime evaluates `IK(p(s(t/K)))` each tick (the closed-form maps are cheap and
 * allocation-free); joint velocity/acceleration come from the chain rule
 * `q̇ = q'(s)·ṡ`, `q̈ = q''(s)·ṡ² + q'(s)·s̈`, with `q'`,`q''` finite-differenced
 * from the IK and `ṡ`,`s̈` taken analytically from the scalar profile.
 *
 * @see controllers/trajectory.hpp (the scalar `s(t)` generator + `TrajectoryBank`).
 * @see J. Bobrow et al. (1985) and Q.-C. Pham, TOPP-RA (2018) for the time-optimal
 *      pointwise re-timing this global-K version approximates.
 */

#include "wet/controllers/trajectory.hpp" // TrajectoryLimits, TrajectoryState, design::synthesize_scurve
#include "wet/math/math.hpp"              // wet::abs, wet::sqrt

namespace wet {

/**
 * @brief Per-joint velocity and acceleration limits for a task-space move.
 * @tparam NJoints Number of joints/actuators
 * @tparam T       Scalar type
 */
template<size_t NJoints, typename T = double>
struct JointLimits {
    wet::array<T, NJoints> max_velocity{};     //!< |q̇ᵢ| ≤ max_velocity[i] (> 0)
    wet::array<T, NJoints> max_acceleration{}; //!< |q̈ᵢ| ≤ max_acceleration[i] (> 0)

    [[nodiscard]] constexpr bool valid() const {
        for (size_t i = 0; i < NJoints; ++i) {
            if (max_velocity[i] <= T{0} || max_acceleration[i] <= T{0}) {
                return false;
            }
        }
        return NJoints > 0;
    }
};

/**
 * @ingroup controllers
 * @brief Path-preserving task-space move (Pipeline B / LIN).
 *
 * @tparam NJoints Number of joints
 * @tparam PathFn  Callable `T s -> TaskPoint` giving the path geometry over
 *                 `s ∈ [0, length]` (e.g. @ref LinearPath).
 * @tparam IkFn    Callable `TaskPoint -> wet::pair<wet::array<T,NJoints>, bool>`
 *                 returning {joint values, reachable}.
 * @tparam T       Scalar type
 */
template<size_t NJoints, typename PathFn, typename IkFn, typename T = double>
class CartesianMove {
public:
    using StateArray = wet::array<TrajectoryState<T>, NJoints>;

    /**
     * @param path         Path geometry `p(s)`, `s ∈ [0, length]`.
     * @param ik           Inverse kinematics `p -> {joints, reachable}`.
     * @param length       Path arc length.
     * @param path_limits  Nominal path-space limits (set the move's *shape* and
     *                     desired feed; jerk-limited, in task units).
     * @param joint_limits Hard per-joint velocity/acceleration caps (the real
     *                     constraint enforced by the global slow-down `K`).
     * @param samples      Sweep resolution for computing `K`.
     */
    constexpr CartesianMove(PathFn path, IkFn ik, T length, const TrajectoryLimits<T>& path_limits, const JointLimits<NJoints, T>& joint_limits, int samples = 256)
        : path_(path), ik_(ik), length_(length), step_s_(length * static_cast<T>(1e-5)) {
        if (step_s_ < T{1e-9}) {
            step_s_ = T{1e-9};
        }
        nominal_ = design::synthesize_scurve(T{0}, length, path_limits);
        valid_ = nominal_.success && joint_limits.valid() && (length > T{0});
        if (!valid_) {
            return;
        }

        // Sweep the nominal profile; find the worst joint velocity/acceleration
        // ratio against the limits, then K makes the binding one exactly active.
        const T   Tnom = nominal_.duration;
        T         r_v = T{0};
        T         r_a = T{0};
        const int M = (samples < 2) ? 2 : samples;
        for (int k = 0; k < M; ++k) {
            const T    tau = (Tnom * static_cast<T>(k)) / static_cast<T>(M - 1);
            const auto sp = nominal_.eval(tau);
            JointDeriv d{};
            joint_derivatives(sp.position, d);
            reachable_ = reachable_ && d.reachable;
            for (size_t i = 0; i < NJoints; ++i) {
                const T v = d.dq[i] * sp.velocity;
                const T a = (d.ddq[i] * sp.velocity * sp.velocity) + (d.dq[i] * sp.acceleration);
                r_v = max_of(r_v, wet::abs(v) / joint_limits.max_velocity[i]);
                r_a = max_of(r_a, wet::abs(a) / joint_limits.max_acceleration[i]);
            }
        }
        K_ = max_of(T{1}, max_of(r_v, wet::sqrt(r_a)));
        duration_ = K_ * Tnom;
    }

    /// Evaluate every joint at the common (synchronized) time @p t ∈ [0, duration].
    [[nodiscard]] constexpr StateArray eval(T t) const {
        StateArray out{};
        if (!valid_) {
            return out;
        }
        const T    tau = t / K_;
        const auto sp = nominal_.eval(tau);
        const T    sd = sp.velocity / K_;             // ṡ = s_nom'(t/K) / K
        const T    sdd = sp.acceleration / (K_ * K_); // s̈ = s_nom''(t/K) / K²
        JointDeriv d{};
        joint_derivatives(sp.position, d);
        for (size_t i = 0; i < NJoints; ++i) {
            out[i].position = d.q[i];
            out[i].velocity = d.dq[i] * sd;
            out[i].acceleration = (d.ddq[i] * sd * sd) + (d.dq[i] * sdd);
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
    [[nodiscard]] constexpr T    duration() const { return duration_; }
    [[nodiscard]] constexpr bool done() const { return t_ >= duration_; }
    [[nodiscard]] constexpr bool valid() const { return valid_; }
    [[nodiscard]] constexpr bool reachable() const { return reachable_; }
    [[nodiscard]] constexpr T    scale() const { return K_; } //!< the global slow-down factor K

private:
    struct JointDeriv {
        wet::array<T, NJoints> q{};   // joint positions at s
        wet::array<T, NJoints> dq{};  // dq/ds
        wet::array<T, NJoints> ddq{}; // d²q/ds²
        bool                   reachable{true};
    };

    // q(s), q'(s), q''(s) via central differences of IK along the path. The
    // stencil centre is nudged inward so `[c−h, c+h]` stays within `[0, length]`:
    // clamping one side while still dividing by the full `2h` would halve the
    // derivative magnitude at the endpoints. (Here the endpoints are at ṡ≈0 so the
    // effect is masked, but the correct stencil keeps it so.) q'/q'' are continuous,
    // so the ≈`1e-5·length` centre shift is negligible.
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

    [[nodiscard]] static constexpr T max_of(T a, T b) { return (a > b) ? a : b; }

    PathFn path_;
    IkFn   ik_;

    design::ScurveProfile<T> nominal_{};

    T length_{T{0}};
    T step_s_{T{1e-6}};
    T K_{T{1}};
    T duration_{T{0}};
    T t_{T{0}};

    bool valid_{false};
    bool reachable_{true};
};

/// Deduction-friendly factory (deduces NJoints from @p joint_limits, callables from args).
template<size_t NJoints, typename PathFn, typename IkFn, typename T>
[[nodiscard]] constexpr CartesianMove<NJoints, PathFn, IkFn, T> make_cartesian_move(
    PathFn path, IkFn ik, T length, const TrajectoryLimits<T>& path_limits,
    const JointLimits<NJoints, T>& joint_limits, int samples = 256
) {
    return CartesianMove<NJoints, PathFn, IkFn, T>(path, ik, length, path_limits, joint_limits, samples);
}

} // namespace wet
