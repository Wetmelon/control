#pragma once

/**
 * @file stewart.hpp
 * @brief Gough–Stewart platform kinematics (6-DOF parallel manipulator).
 *
 * Closed-form **inverse** and iterative **forward** kinematics for the six-leg
 * parallel manipulator behind motion-cueing rigs (flight/driving simulators),
 * hexapod fixtures, and precision pointing/isolation stages. Given a desired
 * platform pose the inverse map yields the six actuator lengths the controllers
 * drive; it is the geometric layer that turns a 6-DOF motion-cueing trajectory
 * into per-leg setpoints.
 *
 * Fits the three-tier pattern: a constexpr @ref design::synthesize_stewart stage
 * validates the geometry and precomputes the fixed base/platform anchor tables,
 * and an allocation-free @ref StewartPlatform runtime evaluates the per-tick
 * kinematics. Embeddable (`wet/control.hpp`).
 *
 * **Inverse (closed-form, the hot path).** With pose `(t ∈ ℝ³, R ∈ SO(3))` the
 * length of leg `i` is `Lᵢ = ‖ t + R·pᵢ − bᵢ ‖`, where `bᵢ` is the fixed base
 * anchor and `pᵢ` the platform anchor in the moving frame. The unit leg vector
 * `ûᵢ = (t + R·pᵢ − bᵢ)/Lᵢ` feeds the actuator Jacobian.
 *
 * **Forward (iterative).** No closed form exists for the general Gough–Stewart
 * platform; pose is recovered from six measured leg lengths by Newton–Raphson on
 * the 6×6 actuator Jacobian `J` (rows `ûᵢᵀ·[I, −[R·pᵢ]ₓ]`), reusing
 * @ref mat::solve, warm-started from the previous solved pose.
 *
 * @see D. Stewart, "A Platform with Six Degrees of Freedom," Proc. IMechE, 1965,
 *      https://doi.org/10.1243/PIME_PROC_1965_180_029_02
 * @see J.-P. Merlet, "Parallel Robots," 2nd ed., Springer, 2006 (forward
 *      kinematics, singularity analysis).
 * @see B. Dasgupta & T. S. Mruthyunjaya, "The Stewart platform manipulator: a
 *      review," Mech. Mach. Theory 35(1), 2000,
 *      https://doi.org/10.1016/S0094-114X(99)00006-3
 */

#include "wet/kinematics/pose.hpp" // Pose, Translation3, Vec3
#include "wet/math/math.hpp"       // wet::sin, cos, sqrt, abs
#include "wet/matrix/matrix.hpp"   // Matrix, Mat3
#include "wet/matrix/solve.hpp"    // mat::solve

namespace wet {

/// Number of legs in a Gough–Stewart platform.
inline constexpr size_t kStewartLegs = 6;

/**
 * @brief Rig geometry: the six fixed base anchors `bᵢ`, the six moving-platform
 *        anchors `pᵢ`, the actuator stroke limits, and the nominal home height.
 *
 * `base[i]` is expressed in the fixed (world) frame; `platform[i]` in the moving
 * platform frame. A leg's installed length must stay within `[stroke_min,
 * stroke_max]`. `home_height` is the platform-origin z at the neutral pose, used
 * by @ref design::synthesize_stewart to check the home pose is reachable.
 *
 * @tparam T Scalar type (floating point)
 */
template<typename T = double>
struct StewartGeometry {
    wet::array<Vec3<T>, kStewartLegs> base{};     //!< fixed base anchors bᵢ (world frame)
    wet::array<Vec3<T>, kStewartLegs> platform{}; //!< platform anchors pᵢ (moving frame)

    T stroke_min{T{0}};  //!< minimum installed leg length
    T stroke_max{T{0}};  //!< maximum installed leg length
    T home_height{T{0}}; //!< platform-origin z at the neutral pose

    /// Basic structural validity (positive, ordered stroke window).
    [[nodiscard]] constexpr bool valid() const {
        if (!(stroke_max > stroke_min) || !(stroke_min >= T{0})) {
            return false;
        }
        // Anchors must not be coincident within a ring (degenerate geometry).
        for (size_t i = 0; i < kStewartLegs; ++i) {
            for (size_t j = i + 1; j < kStewartLegs; ++j) {
                if ((base[i] - base[j]).norm() <= T{0}) {
                    return false;
                }
                if ((platform[i] - platform[j]).norm() <= T{0}) {
                    return false;
                }
            }
        }
        return true;
    }
};

/// Result of an inverse solve: the six leg lengths + a stroke-window flag.
template<typename T = double>
struct StewartInverse {
    wet::array<T, kStewartLegs> lengths{};        //!< Lᵢ = ‖t + R·pᵢ − bᵢ‖
    bool                        reachable{false}; //!< all Lᵢ within [stroke_min, stroke_max]
};

/// Result of a forward (Newton–Raphson) solve.
template<typename T = double>
struct StewartForward {
    Pose<T> pose{};           //!< recovered platform pose
    bool    converged{false}; //!< residual fell below tolerance within the iteration budget
    T       residual{T{0}};   //!< final ‖L(pose) − L_measured‖
};

/**
 * @brief Validated Stewart configuration (the @ref design payload).
 * @tparam T Scalar type (floating point)
 */
template<typename T = double>
struct StewartConfig {
    StewartGeometry<T> geometry{};     //!< the validated rig geometry
    bool               success{false}; //!< geometry valid and home pose reachable

    /// Rebind to another scalar type.
    template<typename U>
    [[nodiscard]] constexpr StewartConfig<U> as() const {
        StewartConfig<U> out{};
        for (size_t i = 0; i < kStewartLegs; ++i) {
            out.geometry.base[i] = Vec3<U>{
                static_cast<U>(geometry.base[i][0]),
                static_cast<U>(geometry.base[i][1]),
                static_cast<U>(geometry.base[i][2]),
            };
            out.geometry.platform[i] = Vec3<U>{
                static_cast<U>(geometry.platform[i][0]),
                static_cast<U>(geometry.platform[i][1]),
                static_cast<U>(geometry.platform[i][2]),
            };
        }
        out.geometry.stroke_min = static_cast<U>(geometry.stroke_min);
        out.geometry.stroke_max = static_cast<U>(geometry.stroke_max);
        out.geometry.home_height = static_cast<U>(geometry.home_height);
        out.success = success;
        return out;
    }
};

/**
 * @brief Gough–Stewart platform runtime — closed-form inverse, Newton forward.
 *
 * Construct from a @ref StewartConfig (or directly from a @ref StewartGeometry).
 * `inverse(pose)` runs every control tick; `forward(lengths, guess)` is the
 * optional feedback path when leg encoders are present. Allocation-free and
 * constexpr-constructible.
 *
 * Example:
 * @code
 * // base_radius, platform_radius, base/platform half-angles, home_height, stroke_min/max
 * constexpr auto cfg = wet::design::stewart_symmetric<double>(
 *     1.0, 0.5, 0.2, 0.2, 1.2, 0.8, 1.8);
 * static_assert(cfg.success);
 * const wet::StewartPlatform<double> rig(cfg);
 *
 * wet::Pose<double> target;
 * target.translation = wet::Translation3<double>(0.05, 0.0, 1.25);
 * const auto legs = rig.inverse(target);   // {lengths, reachable}
 * @endcode
 *
 * @tparam T Scalar type (floating point)
 */
template<typename T = double>
class StewartPlatform {
public:
    constexpr StewartPlatform() = default;
    constexpr explicit StewartPlatform(const StewartGeometry<T>& g) : g_(g) {}
    constexpr explicit StewartPlatform(const StewartConfig<T>& c) : g_(c.geometry) {}

    [[nodiscard]] constexpr const StewartGeometry<T>& geometry() const { return g_; }

    /**
     * @brief Inverse kinematics: pose → six leg lengths.
     *
     * `Lᵢ = ‖ t + R·pᵢ − bᵢ ‖`. `reachable` is true iff every leg lies inside
     * the configured stroke window.
     */
    [[nodiscard]] constexpr StewartInverse<T> inverse(const Pose<T>& pose) const {
        StewartInverse<T> out{};
        out.reachable = true;
        for (size_t i = 0; i < kStewartLegs; ++i) {
            const T L = leg_vector(pose, i).norm();
            out.lengths[i] = L;
            if (L < g_.stroke_min || L > g_.stroke_max) {
                out.reachable = false;
            }
        }
        return out;
    }

    /**
     * @brief Actuator Jacobian `J` at @p pose (rows `ûᵢᵀ·[I, −[R·pᵢ]ₓ]`).
     *
     * Maps a spatial platform twist `[v; ω]` (world frame) to leg-length rates
     * `L̇ = J·[v; ω]`. The first three columns of each row are `ûᵢ`; the last
     * three are `(R·pᵢ) × ûᵢ`. Used by @ref forward and for singularity checks.
     */
    [[nodiscard]] constexpr Matrix<kStewartLegs, kStewartLegs, T> jacobian(const Pose<T>& pose) const {
        Matrix<kStewartLegs, kStewartLegs, T> J{};
        for (size_t i = 0; i < kStewartLegs; ++i) {
            const Vec3<T> rp = pose.orientation.rotate(g_.platform[i]);
            const Vec3<T> lvec = static_cast<const Vec3<T>&>(pose.translation) + rp - g_.base[i];

            const T L = lvec.norm();

            const Vec3<T> uhat = (L > T{0}) ? Vec3<T>(lvec / L) : Vec3<T>{};
            J(i, 0) = uhat[0];
            J(i, 1) = uhat[1];
            J(i, 2) = uhat[2];

            const Vec3<T> ang = rp.cross(uhat);
            J(i, 3) = ang[0];
            J(i, 4) = ang[1];
            J(i, 5) = ang[2];
        }
        return J;
    }

    /**
     * @brief Forward kinematics: six measured leg lengths → pose (Newton–Raphson).
     *
     * Solves `L(pose) = lengths` from a seed pose (the previous solved pose in a
     * real loop). Fixed iteration budget for deterministic runtime; reports
     * `converged` and the final length residual. Returns `converged=false` (with
     * the best pose found) on a singular Jacobian or non-convergence.
     *
     * @param lengths Measured leg lengths.
     * @param guess   Seed pose (warm start).
     * @param iters   Maximum Newton iterations.
     * @param tol     Convergence tolerance on `‖L(pose) − lengths‖`. With a
     *                non-singular Jacobian the residual converges quadratically to
     *                machine precision in a handful of iterations.
     */
    [[nodiscard]] constexpr StewartForward<T> forward(const wet::array<T, kStewartLegs>& lengths, const Pose<T>& guess, size_t iters = 30, T tol = T{1e-9}) const {
        StewartForward<T> out{};
        out.pose = guess;

        for (size_t it = 0; it < iters; ++it) {
            // Length residual e = L(pose) − L_measured.
            Matrix<kStewartLegs, 1, T> e{};
            T                          res2 = T{0};
            for (size_t i = 0; i < kStewartLegs; ++i) {
                const T Li = leg_vector(out.pose, i).norm();
                const T ei = Li - lengths[i];
                e(i, 0) = ei;
                res2 += ei * ei;
            }
            out.residual = wet::sqrt(res2);
            if (out.residual <= tol) {
                out.converged = true;
                return out;
            }

            // Newton step: J·δ = −e, twist δ = [v; ω] (world frame).
            const auto delta_opt = mat::solve(jacobian(out.pose), Matrix<kStewartLegs, 1, T>(e * T{-1}));
            if (!delta_opt) {
                return out; // singular Jacobian — converged stays false
            }
            const auto& d = delta_opt.value();
            apply_twist(out.pose, Vec3<T>{d(0, 0), d(1, 0), d(2, 0)}, Vec3<T>{d(3, 0), d(4, 0), d(5, 0)});
        }

        // Final residual check after the last update.
        T res2 = T{0};
        for (size_t i = 0; i < kStewartLegs; ++i) {
            const T ei = leg_vector(out.pose, i).norm() - lengths[i];
            res2 += ei * ei;
        }
        out.residual = wet::sqrt(res2);
        out.converged = out.residual <= tol;
        return out;
    }

private:
    /// Leg vector `t + R·pᵢ − bᵢ` for leg @p i at @p pose.
    [[nodiscard]] constexpr Vec3<T> leg_vector(const Pose<T>& pose, size_t i) const {
        return static_cast<const Vec3<T>&>(pose.translation) + pose.orientation.rotate(g_.platform[i]) - g_.base[i];
    }

    /// Apply a spatial twist increment (world-frame translation @p v + rotation
    /// vector @p w) to @p pose.
    static constexpr void apply_twist(Pose<T>& pose, const Vec3<T>& v, const Vec3<T>& w) {
        pose.translation = Translation3<T>(static_cast<const Vec3<T>&>(pose.translation) + v);
        const T angle = w.norm();
        if (angle > T{0}) {
            // Build the incremental quaternion exp(½·w) from the rotation vector
            // directly — no optional to unwrap, no separate axis normalization, and
            // exact for the small steps Newton takes near convergence.
            const T             half = angle * T{0.5};
            const T             s = wet::sin(half) / angle; // → ½ as angle → 0
            const Quaternion<T> dq{wet::cos(half), s * w[0], s * w[1], s * w[2]};
            // World-frame (spatial) rotation pre-multiplies the orientation.
            pose.orientation = (dq * pose.orientation).normalized();
        }
    }

    StewartGeometry<T> g_{};
};

namespace design {

/**
 * @brief Validate a hand-entered Stewart geometry and confirm the home pose is
 *        reachable.
 *
 * Checks structural validity (ordered stroke window, non-degenerate anchors) and
 * that the neutral pose `(t = [0, 0, home_height], R = I)` places every leg
 * inside the stroke window.
 *
 * @tparam T Scalar type
 * @param geometry The rig geometry.
 * @return StewartConfig with `success` set accordingly.
 */
template<typename T = double>
[[nodiscard]] constexpr StewartConfig<T> synthesize_stewart(const StewartGeometry<T>& geometry) {
    StewartConfig<T> cfg{geometry, false};
    if (!geometry.valid()) {
        return cfg;
    }

    Pose<T> home;
    home.translation = Translation3<T>(T{0}, T{0}, geometry.home_height);

    const StewartPlatform<T> rig(geometry);
    cfg.success = rig.inverse(home).reachable;

    return cfg;
}

/**
 * @brief Tier-2 builder for the common symmetric hexagonal layout.
 *
 * Places the six base anchors on a circle of radius @p base_radius and the six
 * platform anchors on a circle of radius @p platform_radius, grouped in three
 * pairs spaced 120° apart. Within each triad the two anchors straddle the triad
 * direction by `±half_angle`. The base triads are centred on 0°/120°/240°; the
 * platform triads are rotated by 60° so the legs cross (the classic 6-6 rig).
 *
 * @tparam T Scalar type
 * @param base_radius          Base anchor circle radius.
 * @param platform_radius      Platform anchor circle radius.
 * @param base_half_angle      Half the angular split of a base anchor pair [rad].
 * @param platform_half_angle  Half the angular split of a platform anchor pair [rad].
 * @param home_height          Platform-origin z at the neutral pose.
 * @param stroke_min           Minimum installed leg length.
 * @param stroke_max           Maximum installed leg length.
 * @return Validated StewartConfig (`success` reflects reachability of home).
 */
template<typename T = double>
[[nodiscard]] constexpr StewartConfig<T> stewart_symmetric(T base_radius, T platform_radius, T base_half_angle, T platform_half_angle, T home_height, T stroke_min, T stroke_max) {
    StewartGeometry<T> g{};
    g.stroke_min = stroke_min;
    g.stroke_max = stroke_max;
    g.home_height = home_height;

    const T third = T{2} * wet::numbers::pi_v<T> / T{3}; // 120°
    const T offset = wet::numbers::pi_v<T> / T{3};       // 60° platform rotation

    for (size_t k = 0; k < 3; ++k) {
        const T cb = static_cast<T>(k) * third; // base triad centre
        const T cp = cb + offset;               // platform triad centre
        // Base pair: centre ∓ half_angle.
        const auto sc_bm = wet::sincos(cb - base_half_angle);
        const auto sc_bp = wet::sincos(cb + base_half_angle);
        g.base[(2 * k)] = Vec3<T>{base_radius * sc_bm.second, base_radius * sc_bm.first, T{0}};
        g.base[(2 * k) + 1] = Vec3<T>{base_radius * sc_bp.second, base_radius * sc_bp.first, T{0}};
        // Platform pair: centre ∓ half_angle.
        const auto sc_pm = wet::sincos(cp - platform_half_angle);
        const auto sc_pp = wet::sincos(cp + platform_half_angle);
        g.platform[(2 * k)] = Vec3<T>{platform_radius * sc_pm.second, platform_radius * sc_pm.first, T{0}};
        g.platform[(2 * k) + 1] = Vec3<T>{platform_radius * sc_pp.second, platform_radius * sc_pp.first, T{0}};
    }

    return synthesize_stewart(g);
}

} // namespace design

} // namespace wet
