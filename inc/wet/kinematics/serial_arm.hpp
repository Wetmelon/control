#pragma once

/**
 * @file serial_arm.hpp
 * @brief Serial N-DOF revolute manipulator kinematics (N ≤ 6).
 *
 * The series counterpart to the Stewart platform (kinematics/stewart.hpp): an
 * articulated revolute-joint arm — the canonical industrial robot. Where the
 * parallel Stewart platform has an easy closed-form inverse and an iterative
 * forward map, the serial arm is the mirror image: **trivial closed-form forward**
 * (stack the joint transforms) and a **multi-solution inverse**.
 *
 * Three-tier shape, embeddable, allocation-free. The kinematics emit joint
 * setpoints that the per-joint trajectory generators (controllers/trajectory.hpp)
 * then time-profile.
 *
 * **Geometry.** A @ref DhChain<N> of standard (distal) Denavit–Hartenberg
 * parameters `(aᵢ, αᵢ, dᵢ, θ_offsetᵢ)` per revolute joint, plus per-joint angle
 * limits `[q_minᵢ, q_maxᵢ]`. Each joint's transform is
 * `Aᵢ = Rot_z(θ)·Trans_z(d)·Trans_x(a)·Rot_x(α)`, `θ = qᵢ + θ_offsetᵢ`.
 *
 * **Forward kinematics (closed-form, any N).** `forward(q)` evaluates the chain
 * `T = A₁(q₁)·…·A_N(q_N)` and returns the end-effector @ref Pose. Exact, constexpr.
 *
 * **Inverse kinematics — numerical (damped least squares, any N ≤ 6).** This pass
 * ships the N-generic numerical solver: `inverse(target, seed, mask)` iterates
 * `Δq = Jᵀ(JJᵀ + λ²I)⁻¹·e` (Levenberg–Marquardt damping, `JJᵀ` is 6×6 for every
 * N) from a seed pose, reusing @ref mat::solve, warm-started, with a fixed
 * iteration budget and `converged`/`residual` reporting. An optional 6-DOF
 * **task mask** selects which Cartesian DOF to control, so an under-actuated
 * `N < 6` arm targets only the reachable DOF and least-squares the rest. The
 * spherical-wrist **closed-form (Pieper) branch enumeration** is a planned
 * follow-up; `synthesize_serial_arm` already flags whether the wrist is spherical
 * so it can be auto-selected later.
 *
 * **Jacobian / singularity.** `jacobian(q)` is the geometric 6×N Jacobian
 * (Cartesian↔joint velocity); `manipulability(q)` is Yoshikawa's measure
 * (`√det(JJᵀ)`, or `√det(JᵀJ)` when `N < 6`); `near_singular(q)` thresholds it.
 * Velocity/acceleration *limits* are not enforced here — that is the trajectory
 * generators' job (controllers/trajectory.hpp, controllers/cartesian_move.hpp).
 *
 * @see J. Denavit & R. S. Hartenberg, ASME J. Appl. Mech., 1955 (DH notation).
 * @see J. Craig, "Introduction to Robotics: Mechanics and Control," 3rd ed., 2005.
 * @see C. Wampler, "Manipulator inverse kinematic solutions based on vector
 *      formulations and damped least-squares methods," IEEE SMC, 1986 (DLS).
 * @see T. Yoshikawa, "Manipulability of robotic mechanisms," IJRR, 1985.
 * @see D. Pieper, PhD thesis, Stanford, 1968 (spherical-wrist decoupling — the
 *      closed-form path, planned follow-up).
 */

#include <cstdint> // std::uint8_t

#include "wet/kinematics/pose.hpp"  // Pose, Translation3, Vec3, Quaternion
#include "wet/math/math.hpp"        // wet::sin, cos, atan2, sqrt, abs, clamp
#include "wet/matrix/functions.hpp" // mat::det
#include "wet/matrix/matrix.hpp"    // Matrix, ColVec
#include "wet/matrix/solve.hpp"     // mat::solve

namespace wet {

/// Joint actuation type for a DH joint.
enum class JointType : std::uint8_t {
    Revolute,  //!< the joint variable adds to θ (rotates about zᵢ₋₁)
    Prismatic, //!< the joint variable adds to d (translates along zᵢ₋₁)
};

/**
 * @brief One joint's standard (distal) DH parameters and motion limits.
 *
 * Revolute by default; set @ref type to @ref JointType::Prismatic for a sliding
 * joint (then the joint variable extends `d` instead of `θ`, and `[q_min, q_max]`
 * are stroke limits rather than angle limits). Mixing the two builds, e.g., a
 * SCARA (RRPR) — see kinematics/scara.hpp.
 *
 * @tparam T Scalar type (floating point)
 */
template<typename T = double>
struct DhJoint {
    T         a{T{0}};                       //!< link length aᵢ (along xᵢ)
    T         alpha{T{0}};                   //!< link twist αᵢ [rad] (about xᵢ)
    T         d{T{0}};                       //!< link offset dᵢ (along zᵢ₋₁)
    T         theta_offset{T{0}};            //!< constant added to the joint variable [rad]
    T         q_min{-wet::numbers::pi_v<T>}; //!< lower limit (angle [rad] / stroke)
    T         q_max{wet::numbers::pi_v<T>};  //!< upper limit (angle [rad] / stroke)
    JointType type{JointType::Revolute};     //!< revolute (θ) or prismatic (d)

    [[nodiscard]] constexpr bool valid() const { return q_max > q_min; }
};

/**
 * @brief An N-joint DH chain (the arm geometry).
 * @tparam N Number of joints (1 ≤ N ≤ 6), each revolute or prismatic
 * @tparam T Scalar type
 */
template<size_t N, typename T = double>
struct DhChain {
    static_assert(N >= 1 && N <= 6, "SerialArm supports 1..6 joints");
    wet::array<DhJoint<T>, N> joints{};

    [[nodiscard]] constexpr bool valid() const {
        for (size_t i = 0; i < N; ++i) {
            if (!joints[i].valid()) {
                return false;
            }
        }
        return true;
    }

    /// Rebind to another scalar type.
    template<typename U>
    [[nodiscard]] constexpr DhChain<N, U> as() const {
        DhChain<N, U> out{};
        for (size_t i = 0; i < N; ++i) {
            out.joints[i] = DhJoint<U>{
                static_cast<U>(joints[i].a),
                static_cast<U>(joints[i].alpha),
                static_cast<U>(joints[i].d),
                static_cast<U>(joints[i].theta_offset),
                static_cast<U>(joints[i].q_min),
                static_cast<U>(joints[i].q_max),
                joints[i].type,
            };
        }
        return out;
    }
};

/**
 * @brief Which of the six Cartesian task DOF an IK solve should control.
 *
 * Index order `[x, y, z, rx, ry, rz]`: the first three are translation, the last
 * three the orientation-error rotation vector. Default controls all six.
 */
using TaskMask = wet::array<bool, 6>;

/// Control all six Cartesian DOF (full pose).
inline constexpr TaskMask task_full = {true, true, true, true, true, true};
/// Control position only (the natural target for a ≤3-DOF arm).
inline constexpr TaskMask task_position = {true, true, true, false, false, false};
/// Control position + yaw about z (the natural 4-DOF target for a SCARA).
inline constexpr TaskMask task_position_yaw = {true, true, true, false, false, true};

/// Result of a numerical inverse-kinematics solve.
template<size_t N, typename T = double>
struct ArmIkResult {
    wet::array<T, N> joints{};         //!< the solved joint angles [rad]
    bool             converged{false}; //!< masked task residual fell below tolerance
    T                residual{T{0}};   //!< final ‖masked task error‖
};

/**
 * @brief Validated serial-arm configuration (the @ref design payload).
 * @tparam N Number of joints
 * @tparam T Scalar type
 */
template<size_t N, typename T = double>
struct SerialArmConfig {
    DhChain<N, T> chain{};                //!< the validated DH chain
    bool          spherical_wrist{false}; //!< last three axes intersect (Pieper) — N == 6 only
    bool          success{false};         //!< chain valid

    template<typename U>
    [[nodiscard]] constexpr SerialArmConfig<N, U> as() const {
        return {chain.template as<U>(), spherical_wrist, success};
    }
};

/**
 * @brief Serial N-DOF revolute manipulator runtime.
 *
 * Construct from a @ref SerialArmConfig (or a @ref DhChain directly). All methods
 * are allocation-free and constexpr-constructible.
 *
 * Example:
 * @code
 * constexpr auto cfg = wet::design::arm_spherical_wrist<double>(0.4, 0.5, 0.4, 0.1);
 * static_assert(cfg.success && cfg.spherical_wrist);
 * const wet::SerialArm<6, double> arm(cfg);
 *
 * wet::array<double, 6> q{0.2, -0.4, 0.8, 0.1, 0.5, -0.3};
 * const wet::Pose<double> tcp = arm.forward(q);          // closed-form FK
 * const auto sol = arm.inverse(tcp, q);                  // numerical IK round-trip
 * @endcode
 *
 * @tparam N Number of joints (1 ≤ N ≤ 6)
 * @tparam T Scalar type
 */
template<size_t N, typename T = double>
class SerialArm {
public:
    using Frames = wet::array<Pose<T>, N + 1>; //!< cumulative frames {0}…{N}

    constexpr SerialArm() = default;
    constexpr explicit SerialArm(const DhChain<N, T>& chain) : chain_(chain) {}
    constexpr explicit SerialArm(const SerialArmConfig<N, T>& cfg) : chain_(cfg.chain) {}

    [[nodiscard]] constexpr const DhChain<N, T>& chain() const { return chain_; }

    /// Forward kinematics: joint angles → end-effector pose.
    [[nodiscard]] constexpr Pose<T> forward(const wet::array<T, N>& q) const { return frames(q)[N]; }

    /**
     * @brief Cumulative frames `{0}…{N}` in the base frame.
     *
     * `frames(q)[k]` is `T⁰ₖ = A₁·…·Aₖ` (frame `{0}` is the base identity, frame
     * `{N}` the end-effector). Shared by @ref forward and @ref jacobian.
     */
    [[nodiscard]] constexpr Frames frames(const wet::array<T, N>& q) const {
        Frames f{};
        f[0] = Pose<T>::identity();
        for (size_t i = 0; i < N; ++i) {
            f[i + 1] = f[i] * joint_pose(chain_.joints[i], q[i]);
        }
        return f;
    }

    /**
     * @brief Geometric 6×N Jacobian at @p q (spatial, base frame).
     *
     * Maps joint rates to the end-effector twist `[v; ω] = J·q̇`. For a revolute
     * joint `j` the column is `[ z_j × (o_N − o_j) ; z_j ]`; for a prismatic joint
     * it is `[ z_j ; 0 ]` (pure translation along the axis), with `z_j` the joint
     * axis and `o_j` the frame origin of `{j}` (both in base coords).
     */
    [[nodiscard]] constexpr Matrix<6, N, T> jacobian(const wet::array<T, N>& q) const {
        return jacobian_from_frames(frames(q));
    }

    /**
     * @brief Numerical inverse kinematics (damped least squares).
     *
     * Iterates `Δq = Jᵀ(JJᵀ + λ²I)⁻¹·e` from @p seed toward @p target, where `e`
     * is the masked 6-DOF task error (translation + orientation rotation vector).
     * Joint angles are clamped to their limits each step. Deterministic iteration
     * budget; reports `converged` and the final masked residual.
     *
     * @param target Desired end-effector pose.
     * @param seed   Warm-start joint angles (e.g. the current configuration).
     * @param mask   Which Cartesian DOF to control (default: full pose).
     * @param iters  Maximum iterations.
     * @param lambda Damping factor (larger ⇒ slower but singularity-robust).
     * @param tol    Convergence tolerance on the masked task residual.
     */
    [[nodiscard]] constexpr ArmIkResult<N, T> inverse(const Pose<T>& target, const wet::array<T, N>& seed, const TaskMask& mask = task_full, size_t iters = 100, T lambda = T{0.05}, T tol = T{1e-9}) const {
        ArmIkResult<N, T> out{};
        out.joints = seed;
        const T lam2 = lambda * lambda;

        for (size_t it = 0; it < iters; ++it) {
            const Frames    f = frames(out.joints);
            Matrix<6, 1, T> e = task_error(target, f[N]);
            apply_mask(e, mask);
            out.residual = vec_norm(e);
            if (out.residual <= tol) {
                out.converged = true;
                return out;
            }

            Matrix<6, N, T> J = jacobian_from_frames(f);
            mask_rows(J, mask);

            // y = (JJᵀ + λ²I)⁻¹ e   (6×6 solve), then Δq = Jᵀ y.
            Matrix<6, 6, T> A = J * J.transpose();
            for (size_t i = 0; i < 6; ++i) {
                A(i, i) += lam2;
            }
            const auto y_opt = mat::solve(A, e);
            if (!y_opt) {
                return out; // singular damped system (λ should preclude this)
            }
            const Matrix<N, 1, T> dq = J.transpose() * y_opt.value();
            for (size_t i = 0; i < N; ++i) {
                out.joints[i] = wet::clamp(out.joints[i] + dq(i, 0), chain_.joints[i].q_min, chain_.joints[i].q_max);
            }
        }

        // Final residual after the last update.
        Matrix<6, 1, T> e = task_error(target, forward(out.joints));
        apply_mask(e, mask);
        out.residual = vec_norm(e);
        out.converged = out.residual <= tol;
        return out;
    }

    /**
     * @brief Yoshikawa manipulability `w` at @p q.
     *
     * `w = √det(JJᵀ)` (6×6) when the arm can span the full task space (`N ≥ 6`),
     * else `√det(JᵀJ)` (N×N). Drops to zero at a kinematic singularity.
     */
    [[nodiscard]] constexpr T manipulability(const wet::array<T, N>& q) const {
        const Matrix<6, N, T> J = jacobian(q);
        T                     d{};
        if constexpr (N >= 6) {
            d = mat::det(Matrix<6, 6, T>(J * J.transpose()));
        } else {
            d = mat::det(Matrix<N, N, T>(J.transpose() * J));
        }
        return (d > T{0}) ? wet::sqrt(d) : T{0};
    }

    /// True when @p q is at/near a singularity (manipulability below @p eps).
    [[nodiscard]] constexpr bool near_singular(const wet::array<T, N>& q, T eps = T{1e-4}) const {
        return manipulability(q) < eps;
    }

private:
    /// Standard-DH per-joint transform `Aᵢ` as a Pose. The joint variable @p q
    /// drives `θ` for a revolute joint, or `d` for a prismatic one.
    [[nodiscard]] static constexpr Pose<T> joint_pose(const DhJoint<T>& j, T q) {
        const bool revolute = (j.type == JointType::Revolute);
        const T    theta = revolute ? (q + j.theta_offset) : j.theta_offset;
        const T    dval = revolute ? j.d : (j.d + q);
        const auto sct = wet::sincos(theta * T{0.5}); // {sin, cos} of θ/2
        const auto sca = wet::sincos(j.alpha * T{0.5});
        // qz = (cos θ/2, 0, 0, sin θ/2), qx = (cos α/2, sin α/2, 0, 0); R = Rz·Rx.
        const Quaternion<T> qz{sct.second, T{0}, T{0}, sct.first};
        const Quaternion<T> qx{sca.second, sca.first, T{0}, T{0}};
        Pose<T>             p;
        p.orientation = qz * qx;
        const auto sc = wet::sincos(theta); // translation (a cosθ, a sinθ, d)
        p.translation = Translation3<T>(j.a * sc.second, j.a * sc.first, dval);
        return p;
    }

    [[nodiscard]] constexpr Matrix<6, N, T> jacobian_from_frames(const Frames& f) const {
        Matrix<6, N, T> J{};
        const Vec3<T>   on = static_cast<const Vec3<T>&>(f[N].translation);
        const Vec3<T>   zhat{T{0}, T{0}, T{1}};
        for (size_t j = 0; j < N; ++j) {
            const Vec3<T> zj = f[j].orientation.rotate(zhat);
            if (chain_.joints[j].type == JointType::Revolute) {
                const Vec3<T> oj = static_cast<const Vec3<T>&>(f[j].translation);
                const Vec3<T> jv = zj.cross(on - oj);
                J(0, j) = jv[0];
                J(1, j) = jv[1];
                J(2, j) = jv[2];
                J(3, j) = zj[0];
                J(4, j) = zj[1];
                J(5, j) = zj[2];
            } else { // prismatic: pure translation along the joint axis
                J(0, j) = zj[0];
                J(1, j) = zj[1];
                J(2, j) = zj[2];
            }
        }
        return J;
    }

    /// Masked 6-DOF task error `[Δp ; rotvec(R_target·R_currentᵀ)]` (base frame).
    [[nodiscard]] static constexpr Matrix<6, 1, T> task_error(const Pose<T>& target, const Pose<T>& current) {
        Matrix<6, 1, T> e{};

        const Vec3<T> dp = static_cast<const Vec3<T>&>(target.translation) - static_cast<const Vec3<T>&>(current.translation);
        e(0, 0) = dp[0];
        e(1, 0) = dp[1];
        e(2, 0) = dp[2];

        const Vec3<T> w = rotation_vector(target.orientation * current.orientation.conjugate());
        e(3, 0) = w[0];
        e(4, 0) = w[1];
        e(5, 0) = w[2];

        return e;
    }

    /// Shortest-arc rotation vector (axis·angle) of a unit quaternion.
    [[nodiscard]] static constexpr Vec3<T> rotation_vector(const Quaternion<T>& q_in) {
        Quaternion<T> q = q_in;
        if (q.w() < T{0}) { // take the shorter rotation
            q = Quaternion<T>{-q.w(), -q.x(), -q.y(), -q.z()};
        }
        const Vec3<T> v{q.x(), q.y(), q.z()};
        const T       vn = v.norm();
        if (vn <= T{0}) {
            return Vec3<T>{T{0}, T{0}, T{0}};
        }
        const T angle = T{2} * wet::atan2(vn, q.w());
        return v * (angle / vn);
    }

    static constexpr void apply_mask(Matrix<6, 1, T>& e, const TaskMask& mask) {
        for (size_t i = 0; i < 6; ++i) {
            if (!mask[i]) {
                e(i, 0) = T{0};
            }
        }
    }

    static constexpr void mask_rows(Matrix<6, N, T>& J, const TaskMask& mask) {
        for (size_t i = 0; i < 6; ++i) {
            if (!mask[i]) {
                for (size_t j = 0; j < N; ++j) {
                    J(i, j) = T{0};
                }
            }
        }
    }

    [[nodiscard]] static constexpr T vec_norm(const Matrix<6, 1, T>& e) {
        T s{};
        for (size_t i = 0; i < 6; ++i) {
            s += e(i, 0) * e(i, 0);
        }
        return wet::sqrt(s);
    }

    DhChain<N, T> chain_{};
};

/**
 * @brief Pick the solution branch nearest a reference configuration.
 *
 * Among the first @p count candidate joint sets, returns the index minimizing the
 * sum of squared angle differences (wrapped to `(-π, π]`) to @p reference — the
 * continuity helper for tracking one branch across a trajectory. Returns `count`
 * (an invalid index) when `count == 0`.
 *
 * @tparam M Candidate array capacity
 * @tparam N Number of joints
 */
template<size_t M, size_t N, typename T>
[[nodiscard]] constexpr size_t select_nearest(const wet::array<wet::array<T, N>, M>& solutions, size_t count, const wet::array<T, N>& reference) {
    size_t best = count;
    T      best_d = T{0};
    for (size_t s = 0; s < count && s < M; ++s) {
        T d{};
        for (size_t i = 0; i < N; ++i) {
            const T diff = wet::wrap(solutions[s][i] - reference[i], -wet::numbers::pi_v<T>, wet::numbers::pi_v<T>);
            d += diff * diff;
        }
        if (best == count || d < best_d) {
            best = s;
            best_d = d;
        }
    }
    return best;
}

namespace design {

/// Spherical-wrist (Pieper) criterion for a 6R chain: axes 4-5-6 intersect, i.e.
/// `a₅ = a₆ = 0` and `d₅ = 0` (within @p eps).
template<size_t N, typename T>
[[nodiscard]] constexpr bool is_spherical_wrist(const DhChain<N, T>& chain, T eps = T{1e-9}) {
    if constexpr (N != 6) {
        return false;
    } else {
        return wet::abs(chain.joints[4].a) <= eps && wet::abs(chain.joints[5].a) <= eps && wet::abs(chain.joints[4].d) <= eps;
    }
}

/**
 * @brief Validate a serial-arm DH chain and flag a spherical wrist.
 * @return SerialArmConfig with `success` and `spherical_wrist` set.
 */
template<size_t N, typename T = double>
[[nodiscard]] constexpr SerialArmConfig<N, T> synthesize_serial_arm(const DhChain<N, T>& chain) {
    SerialArmConfig<N, T> cfg{};
    cfg.chain = chain;
    cfg.success = chain.valid();
    cfg.spherical_wrist = cfg.success && is_spherical_wrist(chain);
    return cfg;
}

/**
 * @brief Tier-2 builder for a standard 6R elbow arm with a spherical wrist.
 *
 * Builds a valid spherical-wrist DH table from intuitive link dimensions so users
 * need not hand-enter twelve parameters: a vertical base, a planar shoulder/elbow
 * (upper arm @p upper_arm, forearm @p forearm), and an intersecting roll-pitch-roll
 * wrist with tool offset @p tool. Angle limits default to `±limit`.
 *
 * @param base_height z-offset of joint 2 above the base (d₁).
 * @param upper_arm   shoulder→elbow link length (a₂).
 * @param forearm     elbow→wrist link length (d₄).
 * @param tool        wrist→tool-flange offset along the approach axis (d₆).
 * @param limit       symmetric joint-angle limit [rad].
 */
template<typename T = double>
[[nodiscard]] constexpr SerialArmConfig<6, T> arm_spherical_wrist(T base_height, T upper_arm, T forearm, T tool, T limit = wet::numbers::pi_v<T>) {
    const T       h = wet::numbers::pi_v<T> / T{2};
    DhChain<6, T> chain{};
    //          a            alpha   d            θ_off  q_min   q_max
    chain.joints[0] = {T{0}, h, base_height, T{0}, -limit, limit}; // base yaw
    chain.joints[1] = {upper_arm, T{0}, T{0}, -h, -limit, limit};  // shoulder
    chain.joints[2] = {T{0}, h, T{0}, T{0}, -limit, limit};        // elbow
    chain.joints[3] = {T{0}, -h, forearm, T{0}, -limit, limit};    // wrist roll
    chain.joints[4] = {T{0}, h, T{0}, T{0}, -limit, limit};        // wrist pitch
    chain.joints[5] = {T{0}, T{0}, tool, T{0}, -limit, limit};     // wrist roll/flange
    return synthesize_serial_arm(chain);
}

} // namespace design

} // namespace wet
