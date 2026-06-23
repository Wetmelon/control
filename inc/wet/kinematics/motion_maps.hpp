#pragma once

/**
 * @file motion_maps.hpp
 * @brief Closed-form kinematic maps for the common non-articulated motion
 *        architectures: Cartesian gantry, CoreXY, polar (r-θ), and delta
 *        (rotary and linear). Each maps machine actuators ↔ task space; like the
 *        manipulator solvers it emits per-actuator setpoints that the trajectory
 *        generators (controllers/trajectory.hpp) then time-profile.
 *
 * `forward(actuators) → task` and `inverse(task) → actuators`. The trivial maps
 * (Cartesian / CoreXY / polar) are exact and unconditional; the deltas are true
 * parallel mechanisms and report a reachability/validity flag.
 *
 * Convention: these are *geometric* maps in engineering units (mm, rad). Per-axis
 * counts↔mm scaling is the separate concern of utility/scaling.hpp.
 *
 * @see R. Clavel, delta robot (EPFL, 1991); CoreXY mechanism, https://corexy.com.
 */

#include "wet/kinematics/pose.hpp" // Pose, Translation3, Vec3
#include "wet/math/complex.hpp"
#include "wet/math/math.hpp" // wet::sqrt, sin, cos, atan2

namespace wet {

/**
 * @brief Cartesian gantry: independent per-axis affine map `task = scale·act +
 *        offset` (the "kinematics" is the identity, exposed for a uniform
 *        forward/inverse interface).
 * @tparam N Number of axes
 * @tparam T Scalar type
 */
template<size_t N, typename T = double>
struct CartesianMap {
    wet::array<T, N> scale{};  //!< task units per actuator unit (must be nonzero)
    wet::array<T, N> offset{}; //!< task value at actuator zero

    [[nodiscard]] constexpr wet::array<T, N> forward(const wet::array<T, N>& act) const {
        wet::array<T, N> task{};
        for (size_t i = 0; i < N; ++i) {
            task[i] = (scale[i] * act[i]) + offset[i];
        }
        return task;
    }
    [[nodiscard]] constexpr wet::array<T, N> inverse(const wet::array<T, N>& task) const {
        wet::array<T, N> act{};
        for (size_t i = 0; i < N; ++i) {
            act[i] = (task[i] - offset[i]) / scale[i];
        }
        return act;
    }
};

/**
 * @brief CoreXY belt mapping (2 motors A/B → Cartesian X/Y).
 *
 * The two motors move the head along the belt-sum/difference: `X = ½(A+B)`,
 * `Y = ½(A−B)`; inversely `A = X+Y`, `B = X−Y` (motor travel in the same length
 * units as X/Y).
 *
 * @tparam T Scalar type
 */
template<typename T = double>
struct CoreXY {
    struct Motors {
        T a, b;
    };

    struct Point {
        T x, y;
    };

    [[nodiscard]] static constexpr Motors inverse(T x, T y) { return Motors{x + y, x - y}; }
    [[nodiscard]] static constexpr Point  forward(T a, T b) { return Point{(a + b) / T{2}, (a - b) / T{2}}; }
};

/**
 * @brief Polar / R-θ mapping (radius + angle ↔ Cartesian X/Y).
 * @tparam T Scalar type
 */
template<typename T = double>
struct PolarMap {
    struct Axes {
        T r, theta; //!< radius, angle [rad]
    };

    struct Point {
        T x, y;
    };

    [[nodiscard]] static constexpr Axes inverse(T x, T y) {
        return Axes{wet::hypot(x, y), wet::atan2(y, x)};
    }

    [[nodiscard]] static constexpr Point forward(T r, T theta) {
        const auto sc = wet::sincos(theta); // {sin, cos}
        return Point{r * sc.second, r * sc.first};
    }
};

// ---- Delta robots ----------------------------------------------------------

/// Result of a delta inverse solve: the three actuator values + reachability.
template<typename T = double>
struct DeltaInverse {
    wet::array<T, 3> actuators{};
    bool             reachable{false};
};

/// Result of a delta forward solve: the end-effector pose (orientation fixed to
/// identity — deltas are 3-DOF translational) + validity.
template<typename T = double>
struct DeltaForward {
    Pose<T> pose{};
    bool    valid{false};
};

/**
 * @brief Rotary delta geometry (three base servos, parallelogram arms).
 *
 * @tparam T Scalar type
 */
template<typename T = double>
struct RotaryDeltaGeometry {
    T base_tri{0};     //!< f — base equilateral-triangle side
    T effector_tri{0}; //!< e — moving-platform equilateral-triangle side
    T bicep{0};        //!< rf — upper-arm (servo horn) length
    T forearm{0};      //!< re — lower-arm (parallelogram) length

    [[nodiscard]] constexpr bool valid() const {
        return (base_tri > T{0}) && (effector_tri > T{0}) && (bicep > T{0}) && (forearm > T{0});
    }
};

/**
 * @brief Rotary delta robot — closed-form inverse, quadratic-intersection forward.
 *
 * Angles are servo-horn angles [rad] measured from the horizontal base plane;
 * the standard delta frame has the platform below the base (z < 0). Inverse is
 * exact and per-arm; forward intersects the three bicep-tip spheres.
 *
 * @tparam T Scalar type
 */
template<typename T = double>
class RotaryDelta {
public:
    constexpr RotaryDelta() = default;
    constexpr explicit RotaryDelta(const RotaryDeltaGeometry<T>& g) : g_(g) {}

    [[nodiscard]] constexpr bool valid() const { return g_.valid(); }

    /// Inverse: end-effector (x, y, z) → three servo angles [rad].
    [[nodiscard]] constexpr DeltaInverse<T> inverse(T x, T y, T z) const {
        const T sqrt3 = wet::sqrt(T{3});
        const T sin120 = sqrt3 / T{2};
        const T cos120 = T{-1} / T{2};
        bool    ok = true;

        // Rotate the target into each arm's plane (±120°).
        const auto a1 = arm_angle(x, y, z, ok);
        const auto a2 = arm_angle((x * cos120) + (y * sin120), (y * cos120) - (x * sin120), z, ok);
        const auto a3 = arm_angle((x * cos120) - (y * sin120), (y * cos120) + (x * sin120), z, ok);

        return {.actuators = {a1, a2, a3}, .reachable = ok};
    }

    [[nodiscard]] constexpr DeltaInverse<T> inverse(const Vec3<T>& p) const { return inverse(p[0], p[1], p[2]); }

    /// Forward: three servo angles [rad] → end-effector pose.
    [[nodiscard]] constexpr DeltaForward<T> forward(T t1, T t2, T t3) const {
        const T sqrt3 = wet::sqrt(T{3});
        const T sin30 = T{1} / T{2};
        const T tan60 = sqrt3;
        const T tan30 = T{1} / sqrt3;
        const T t = (g_.base_tri - g_.effector_tri) * tan30 / T{2};

        const T y1 = -(t + (g_.bicep * wet::cos(t1)));
        const T z1 = -g_.bicep * wet::sin(t1);
        const T y2 = (t + (g_.bicep * wet::cos(t2))) * sin30;
        const T x2 = y2 * tan60;
        const T z2 = -g_.bicep * wet::sin(t2);
        const T y3 = (t + (g_.bicep * wet::cos(t3))) * sin30;
        const T x3 = -y3 * tan60;
        const T z3 = -g_.bicep * wet::sin(t3);

        const T dnm = ((y2 - y1) * x3) - ((y3 - y1) * x2);
        const T w1 = (y1 * y1) + (z1 * z1);
        const T w2 = (x2 * x2) + (y2 * y2) + (z2 * z2);
        const T w3 = (x3 * x3) + (y3 * y3) + (z3 * z3);

        const T a1 = ((z2 - z1) * (y3 - y1)) - ((z3 - z1) * (y2 - y1));
        const T b1 = -(((w2 - w1) * (y3 - y1)) - ((w3 - w1) * (y2 - y1))) / T{2};
        const T a2 = -((z2 - z1) * x3) + ((z3 - z1) * x2);
        const T b2 = (((w2 - w1) * x3) - ((w3 - w1) * x2)) / T{2};

        const T aa = (a1 * a1) + (a2 * a2) + (dnm * dnm);
        const T bb = T{2} * ((a1 * b1) + (a2 * (b2 - (y1 * dnm))) - (z1 * dnm * dnm));
        const T cc = ((b2 - (y1 * dnm)) * (b2 - (y1 * dnm))) + (b1 * b1) + (dnm * dnm * ((z1 * z1) - (g_.forearm * g_.forearm)));

        const T         disc = (bb * bb) - (T{4} * aa * cc);
        DeltaForward<T> out{};
        if (disc < T{0} || dnm == T{0}) {
            return out; // valid == false
        }
        const T z0 = T{-1} / T{2} * (bb + wet::sqrt(disc)) / aa;
        const T x0 = ((a1 * z0) + b1) / dnm;
        const T y0 = ((a2 * z0) + b2) / dnm;
        out.pose.translation = Translation3<T>(x0, y0, z0);
        out.valid = true;
        return out;
    }

    [[nodiscard]] constexpr DeltaForward<T> forward(const wet::array<T, 3>& angles) const {
        return forward(angles[0], angles[1], angles[2]);
    }

private:
    // Servo angle for one arm in its own y-z plane (the others are rotated ±120°).
    [[nodiscard]] constexpr T arm_angle(T x0, T y0, T z0, bool& ok) const {
        const T tan30 = T{1} / wet::sqrt(T{3});
        const T y1 = -T{1} / T{2} * tan30 * g_.base_tri; // base joint, projected
        y0 -= T{1} / T{2} * tan30 * g_.effector_tri;     // shift to platform edge
        const T a = ((x0 * x0) + (y0 * y0) + (z0 * z0) + (g_.bicep * g_.bicep) - (g_.forearm * g_.forearm) - (y1 * y1)) / (T{2} * z0);
        const T b = (y1 - y0) / z0;
        const T disc = -((a + (b * y1)) * (a + (b * y1))) + (g_.bicep * (((b * b) * g_.bicep) + g_.bicep));
        if (disc < T{0}) {
            ok = false;
            return T{0};
        }
        const T yj = (y1 - (a * b) - wet::sqrt(disc)) / ((b * b) + T{1});
        const T zj = a + (b * yj);
        return wet::atan2(-zj, y1 - yj);
    }

    RotaryDeltaGeometry<T> g_{};
};

/**
 * @brief Linear delta geometry (three vertical carriages, fixed-length rods).
 * @tparam T Scalar type
 */
template<typename T = double>
struct LinearDeltaGeometry {
    T base_radius{0};     //!< carriage circle radius
    T effector_radius{0}; //!< platform joint circle radius
    T rod_length{0};      //!< parallelogram rod length

    [[nodiscard]] constexpr bool valid() const {
        return (base_radius > T{0}) && (effector_radius >= T{0}) && (rod_length > T{0});
    }
};

/**
 * @brief Linear delta robot — per-carriage closed-form inverse, sphere-
 *        trilateration forward. Towers at 90°, 210°, 330°.
 * @tparam T Scalar type
 */
template<typename T = double>
class LinearDelta {
public:
    constexpr LinearDelta() = default;
    constexpr explicit LinearDelta(const LinearDeltaGeometry<T>& g) : g_(g) { init_towers(); }

    [[nodiscard]] constexpr bool valid() const { return g_.valid(); }

    /// Inverse: end-effector (x, y, z) → three carriage heights.
    [[nodiscard]] constexpr DeltaInverse<T> inverse(T x, T y, T z) const {
        DeltaInverse<T> out{};
        out.reachable = true;
        const T rv = g_.base_radius - g_.effector_radius; // virtual tower radius
        for (size_t i = 0; i < 3; ++i) {
            const T dx = x - (rv * cos_[i]);
            const T dy = y - (rv * sin_[i]);
            const T horiz2 = (dx * dx) + (dy * dy);
            const T rem = (g_.rod_length * g_.rod_length) - horiz2;
            if (rem < T{0}) {
                out.reachable = false;
                out.actuators[i] = T{0};
            } else {
                out.actuators[i] = z + wet::sqrt(rem); // carriage above the joint
            }
        }
        return out;
    }

    [[nodiscard]] constexpr DeltaInverse<T> inverse(const Vec3<T>& p) const { return inverse(p[0], p[1], p[2]); }

    /// Forward: three carriage heights → end-effector pose (lower intersection).
    [[nodiscard]] constexpr DeltaForward<T> forward(T h0, T h1, T h2) const {
        const T rv = g_.base_radius - g_.effector_radius;
        // Virtual sphere centers (carriage shifted in by the effector radius).
        const Vec3<T> s0{rv * cos_[0], rv * sin_[0], h0};
        const Vec3<T> s1{rv * cos_[1], rv * sin_[1], h1};
        const Vec3<T> s2{rv * cos_[2], rv * sin_[2], h2};
        return trilaterate(s0, s1, s2, g_.rod_length);
    }

    [[nodiscard]] constexpr DeltaForward<T> forward(const wet::array<T, 3>& h) const {
        return forward(h[0], h[1], h[2]);
    }

private:
    constexpr void init_towers() {
        const T sqrt3 = wet::sqrt(T{3});
        // 90°, 210°, 330°: cos/sin precomputed exactly.
        cos_ = {T{0}, -sqrt3 / T{2}, sqrt3 / T{2}};
        sin_ = {T{1}, T{-1} / T{2}, T{-1} / T{2}};
    }

    // Intersect three equal-radius spheres; pick the lower-z solution.
    [[nodiscard]] static constexpr DeltaForward<T> trilaterate(const Vec3<T>& s0, const Vec3<T>& s1, const Vec3<T>& s2, T L) {
        DeltaForward<T> out{};
        // Pairwise subtraction -> two planes: P·(sj-s0) = (|sj|²-|s0|²)/2.
        const T dx1 = s1[0] - s0[0];
        const T dy1 = s1[1] - s0[1];
        const T dz1 = s1[2] - s0[2];

        const T dx2 = s2[0] - s0[0];
        const T dy2 = s2[1] - s0[1];
        const T dz2 = s2[2] - s0[2];

        const T n0 = (s0[0] * s0[0]) + (s0[1] * s0[1]) + (s0[2] * s0[2]);
        const T n1 = (s1[0] * s1[0]) + (s1[1] * s1[1]) + (s1[2] * s1[2]);
        const T n2 = (s2[0] * s2[0]) + (s2[1] * s2[1]) + (s2[2] * s2[2]);

        const T k1 = (n1 - n0) / T{2};
        const T k2 = (n2 - n0) / T{2};
        const T det = (dx1 * dy2) - (dy1 * dx2);
        if (det == T{0}) {
            return out; // degenerate tower layout
        }

        // Solve x, y as affine functions of z: x = px + qx·z, y = py + qy·z.
        const T px = ((dy2 * k1) - (dy1 * k2)) / det;
        const T qx = ((dy1 * dz2) - (dy2 * dz1)) / det;

        const T py = ((dx1 * k2) - (dx2 * k1)) / det;
        const T qy = ((dx2 * dz1) - (dx1 * dz2)) / det;

        // Substitute into sphere 0: A·z² + B·z + C = 0.
        const T ex = px - s0[0];
        const T ey = py - s0[1];

        const T A = (qx * qx) + (qy * qy) + T{1};
        const T B = T{2} * ((ex * qx) + (ey * qy) - s0[2]);
        const T C = (ex * ex) + (ey * ey) + (s0[2] * s0[2]) - (L * L);

        const T disc = (B * B) - (T{4} * A * C);
        if (disc < T{0}) {
            return out;
        }

        const T z = -(B + wet::sqrt(disc)) / (T{2} * A); // lower root
        out.pose.translation = Translation3<T>(px + (qx * z), py + (qy * z), z);
        out.valid = true;
        return out;
    }

    LinearDeltaGeometry<T> g_{};
    wet::array<T, 3>       cos_{};
    wet::array<T, 3>       sin_{};
};

} // namespace wet
