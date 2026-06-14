#pragma once

/**
 * @file scara.hpp
 * @brief SCARA kinematics — both the **series** SCARA (RRPR articulated arm) and
 *        the **parallel** SCARA (planar five-bar mechanism), with one-call builders.
 *
 * Two ways to build a SCARA:
 *
 * - **Series SCARA (RRPR).** A shoulder + elbow (two parallel vertical revolute
 *   axes giving planar X/Y reach), a vertical **prismatic** Z, and a revolute
 *   wrist. `design::scara_arm(...)` fills a 4-joint @ref DhChain (using the
 *   prismatic-joint support in kinematics/serial_arm.hpp) and returns a
 *   @ref SerialArmConfig you drop straight into @ref SerialArm — forward
 *   kinematics, the geometric Jacobian, and damped-least-squares IK all come for
 *   free. The natural 4-DOF task is position + yaw (@ref task_position_yaw).
 *
 * - **Parallel SCARA (five-bar).** Two base-mounted motors drive a closed planar
 *   five-bar linkage whose distal links meet at the end-effector — a 2-DOF
 *   translational parallel mechanism (the parallel cousin of the delta). @ref
 *   FiveBar gives closed-form inverse (a circle intersection per arm), closed-form
 *   forward (intersect the two distal circles), the 2×2 velocity Jacobian, and
 *   reachability / singularity flags. `design::five_bar_symmetric(...)` builds the
 *   common symmetric layout.
 *
 * All allocation-free and constexpr-constructible, in the `wet/control.hpp`
 * umbrella.
 *
 * @see kinematics/serial_arm.hpp (the RRPR backbone, prismatic joints).
 * @see J.-P. Merlet, "Parallel Robots," 2nd ed., Springer, 2006 (five-bar / planar
 *      parallel manipulators and their singularities).
 */

#include "wet/kinematics/serial_arm.hpp" // SerialArm, DhChain, JointType, design::synthesize_serial_arm
#include "wet/math/math.hpp"             // wet::sin, cos, atan2, sqrt, abs

namespace wet {

// ---- Parallel SCARA: planar five-bar ---------------------------------------

/**
 * @brief Symmetric five-bar geometry (two base motors, equal proximal/distal links).
 *
 * The two motor pivots sit on the x-axis at `(±base_separation/2, 0)`; each drives
 * a proximal link of length @ref proximal, and the two distal links of length
 * @ref distal meet at the end-effector. The end-effector works in the `y > 0`
 * half-plane by default.
 *
 * @tparam T Scalar type
 */
template<typename T = double>
struct FiveBarGeometry {
    T base_separation{T{0}}; //!< distance between the two motor pivots
    T proximal{T{0}};        //!< driven (proximal) link length, both arms
    T distal{T{0}};          //!< distal link length, both arms

    [[nodiscard]] constexpr bool valid() const {
        return (base_separation > T{0}) && (proximal > T{0}) && (distal > T{0});
    }
};

/// Inverse-kinematics result: the two motor angles [rad] + reachability.
template<typename T = double>
struct FiveBarInverse {
    wet::array<T, 2> angles{};         //!< {θ_left, θ_right} [rad]
    bool             reachable{false}; //!< both arms can reach the target
};

/// Forward-kinematics result: the end-effector point (x, y) + validity.
template<typename T = double>
struct FiveBarForward {
    wet::array<T, 2> point{};      //!< end-effector {x, y}
    bool             valid{false}; //!< the linkage closes (distal circles meet)
};

/**
 * @brief Planar five-bar parallel manipulator (parallel SCARA).
 *
 * Closed-form both directions. Branch signs select the assembly mode: `elbow1` /
 * `elbow2` pick each arm's elbow side in @ref inverse, `assembly` picks which of
 * the two distal-circle intersections is the end-effector in @ref forward. The
 * defaults give the common "elbows-out, end-effector-up" working mode.
 *
 * @tparam T Scalar type
 */
template<typename T = double>
class FiveBar {
public:
    constexpr FiveBar() = default;
    constexpr explicit FiveBar(const FiveBarGeometry<T>& g) : g_(g) {}

    [[nodiscard]] constexpr bool                      valid() const { return g_.valid(); }
    [[nodiscard]] constexpr const FiveBarGeometry<T>& geometry() const { return g_; }

    /// Left / right motor pivot positions.
    [[nodiscard]] constexpr wet::array<T, 2> base_left() const { return {-g_.base_separation / T{2}, T{0}}; }
    [[nodiscard]] constexpr wet::array<T, 2> base_right() const { return {g_.base_separation / T{2}, T{0}}; }

    /**
     * @brief Inverse kinematics: end-effector (x, y) → the two motor angles.
     *
     * Each arm independently intersects its proximal circle (radius `proximal`,
     * centred at the motor) with the distal circle (radius `distal`, centred at the
     * target). `reachable` is false if either intersection does not exist.
     */
    [[nodiscard]] constexpr FiveBarInverse<T> inverse(T x, T y, int elbow1 = +1, int elbow2 = -1) const {
        FiveBarInverse<T> out{};
        bool              ok1 = true;
        bool              ok2 = true;
        const auto        bl = base_left();
        const auto        br = base_right();
        const auto        e1 = circle_intersect(bl, g_.proximal, {x, y}, g_.distal, elbow1, ok1);
        const auto        e2 = circle_intersect(br, g_.proximal, {x, y}, g_.distal, elbow2, ok2);
        out.angles[0] = wet::atan2(e1[1] - bl[1], e1[0] - bl[0]);
        out.angles[1] = wet::atan2(e2[1] - br[1], e2[0] - br[0]);
        out.reachable = ok1 && ok2;
        return out;
    }

    [[nodiscard]] constexpr FiveBarInverse<T> inverse(const wet::array<T, 2>& p, int elbow1 = +1, int elbow2 = -1) const {
        return inverse(p[0], p[1], elbow1, elbow2);
    }

    /**
     * @brief Forward kinematics: the two motor angles → end-effector (x, y).
     *
     * Places the elbows from the proximal links, then intersects the two distal
     * circles (both radius `distal`). `valid` is false if the linkage cannot close.
     */
    [[nodiscard]] constexpr FiveBarForward<T> forward(T a1, T a2, int assembly = +1) const {
        FiveBarForward<T> out{};
        const auto        e1 = elbow(base_left(), a1);
        const auto        e2 = elbow(base_right(), a2);
        bool              ok = true;
        out.point = circle_intersect(e1, g_.distal, e2, g_.distal, assembly, ok);
        out.valid = ok;
        return out;
    }

    [[nodiscard]] constexpr FiveBarForward<T> forward(const wet::array<T, 2>& a, int assembly = +1) const {
        return forward(a[0], a[1], assembly);
    }

    /**
     * @brief Velocity Jacobian `J` (2×2) at motor angles @p a1, @p a2: `Ṗ = J·θ̇`.
     *
     * From the loop constraints `‖P − Eᵢ‖ = distal`: `A·Ṗ = B·θ̇` with rows of `A`
     * the distal-link directions `(P − Eᵢ)ᵀ` and `B` diagonal, so `J = A⁻¹B`. A
     * near-singular `A` (aligned distal links) is the parallel singularity; the
     * result is then ill-conditioned and @ref singular reports it.
     */
    [[nodiscard]] constexpr Matrix<2, 2, T> jacobian(T a1, T a2, int assembly = +1) const {
        Matrix<2, 2, T> J{};
        const auto      e1 = elbow(base_left(), a1);
        const auto      e2 = elbow(base_right(), a2);
        const auto      fwd = forward(a1, a2, assembly);
        if (!fwd.valid) {
            return J;
        }
        const wet::array<T, 2> p = fwd.point;
        // A·Ṗ = B·θ̇,  A rows = (P−Eᵢ)ᵀ,  Bᵢᵢ = proximal·(P−Eᵢ)·uᵢ⊥.
        const T    d1x = p[0] - e1[0];
        const T    d1y = p[1] - e1[1];
        const T    d2x = p[0] - e2[0];
        const T    d2y = p[1] - e2[1];
        const auto u1 = wet::sincos(a1); // {sin, cos}
        const auto u2 = wet::sincos(a2);
        const T    b1 = g_.proximal * ((d1x * -u1.first) + (d1y * u1.second));
        const T    b2 = g_.proximal * ((d2x * -u2.first) + (d2y * u2.second));
        const T    detA = (d1x * d2y) - (d1y * d2x);
        if (detA == T{0}) {
            return J; // parallel singularity — leave J zero
        }
        const T inv = T{1} / detA;
        // J = A⁻¹·B, with A⁻¹ = (1/detA)·[[d2y, −d1y], [−d2x, d1x]].
        J(0, 0) = inv * d2y * b1;
        J(0, 1) = inv * -d1y * b2;
        J(1, 0) = inv * -d2x * b1;
        J(1, 1) = inv * d1x * b2;
        return J;
    }

    /// True near a parallel singularity (distal links aligned) at @p a1, @p a2.
    [[nodiscard]] constexpr bool singular(T a1, T a2, T eps = T{1e-6}, int assembly = +1) const {
        const auto fwd = forward(a1, a2, assembly);
        if (!fwd.valid) {
            return true;
        }
        const auto e1 = elbow(base_left(), a1);
        const auto e2 = elbow(base_right(), a2);
        const T    d1x = fwd.point[0] - e1[0];
        const T    d1y = fwd.point[1] - e1[1];
        const T    d2x = fwd.point[0] - e2[0];
        const T    d2y = fwd.point[1] - e2[1];
        return wet::abs((d1x * d2y) - (d1y * d2x)) < eps;
    }

private:
    // Elbow position of one arm: base + proximal·(cos a, sin a).
    [[nodiscard]] constexpr wet::array<T, 2> elbow(const wet::array<T, 2>& base, T a) const {
        const auto sc = wet::sincos(a); // {sin, cos}
        return {base[0] + (g_.proximal * sc.second), base[1] + (g_.proximal * sc.first)};
    }

    // A point on circle(centre=ca, r=ra) that also lies on circle(centre=cb, r=rb).
    // `sign` (±1) selects which of the two intersections; `ok` flags existence.
    [[nodiscard]] static constexpr wet::array<T, 2> circle_intersect(const wet::array<T, 2>& ca, T ra, const wet::array<T, 2>& cb, T rb, int sign, bool& ok) {
        const T dx = cb[0] - ca[0];
        const T dy = cb[1] - ca[1];
        const T dist2 = (dx * dx) + (dy * dy);
        const T dist = wet::sqrt(dist2);
        if (dist <= T{0} || dist > ra + rb || dist < wet::abs(ra - rb)) {
            ok = false;
            return ca;
        }
        const T a = ((ra * ra) - (rb * rb) + dist2) / (T{2} * dist);
        const T h2 = (ra * ra) - (a * a);
        const T h = (h2 > T{0}) ? wet::sqrt(h2) : T{0};
        const T mx = ca[0] + (a * dx / dist);
        const T my = ca[1] + (a * dy / dist);
        // Perpendicular to (dx, dy), scaled by ±h.
        const T s = static_cast<T>(sign);
        return {mx + (s * h * -dy / dist), my + (s * h * dx / dist)};
    }

    FiveBarGeometry<T> g_{};
};

namespace design {

/**
 * @brief Build a symmetric five-bar parallel SCARA.
 * @tparam T Scalar type
 * @param base_separation Distance between the two motor pivots.
 * @param proximal        Driven-link length (both arms).
 * @param distal          Distal-link length (both arms).
 */
template<typename T = double>
[[nodiscard]] constexpr FiveBar<T> five_bar_symmetric(T base_separation, T proximal, T distal) {
    return FiveBar<T>(FiveBarGeometry<T>{base_separation, proximal, distal});
}

/**
 * @brief Build a series SCARA (RRPR) as a 4-joint DH chain.
 *
 * Two parallel vertical revolute axes (planar X/Y reach via links @p link1 and
 * @p link2), a vertical prismatic Z (stroke `[0, z_stroke]`, raising the tool
 * along **+z** from @p base_height), and a revolute wrist. Feed the result to a
 * `SerialArm<4>`; solve IK against @ref task_position_yaw (position + yaw).
 *
 * @tparam T Scalar type
 * @param link1       shoulder link length (a₁).
 * @param link2       elbow link length (a₂).
 * @param base_height height of the arm plane above the base (d₁).
 * @param z_stroke    prismatic travel (> 0).
 * @param tool        fixed tool offset along +z (d₄, default 0).
 * @param joint_limit symmetric revolute-angle limit [rad] (default π).
 */
template<typename T = double>
[[nodiscard]] constexpr SerialArmConfig<4, T> scara_arm(T link1, T link2, T base_height, T z_stroke, T tool = T{0}, T joint_limit = wet::numbers::pi_v<T>) {
    DhChain<4, T> c{};
    //                a       alpha  d            θ_off  q_min         q_max        type
    c.joints[0] = {link1, T{0}, base_height, T{0}, -joint_limit, joint_limit, JointType::Revolute};
    c.joints[1] = {link2, T{0}, T{0}, T{0}, -joint_limit, joint_limit, JointType::Revolute};
    c.joints[2] = {T{0}, T{0}, T{0}, T{0}, T{0}, z_stroke, JointType::Prismatic};
    c.joints[3] = {T{0}, T{0}, tool, T{0}, -joint_limit, joint_limit, JointType::Revolute};
    return synthesize_serial_arm(c);
}

} // namespace design

} // namespace wet
