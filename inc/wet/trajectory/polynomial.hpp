#pragma once

/**
 * @file polynomial.hpp
 * @brief Fixed-time polynomial trajectory generators and multi-axis coordination.
 *
 * Two related tools in one header:
 *
 * **1. Polynomial trajectory (fixed-duration, derivative-optimal)**
 *
 * @ref design::synthesize_poly_trajectory<Order> solves the Vandermonde BVP for a
 * cubic/quintic/septic matching position + derivatives at both endpoints over a
 * fixed duration T. Convenience wrappers:
 * - @ref design::min_accel — cubic (C¹ profile, minimizes integrated acceleration²)
 * - @ref design::min_jerk  — quintic (Flash–Hogan, minimizes integrated jerk²)
 * - @ref design::min_snap  — septic (Mellinger–Kumar, minimizes integrated snap²)
 *
 * @ref PolynomialTrajectory is the runtime evaluator.
 *
 * **2. Trajectory bank (multi-axis time-synchronization)**
 *
 * @ref TrajectoryBank wraps any homogeneous array of per-axis runtime trajectories
 * (trapezoidal, S-curve, or polynomial) and time-scales each axis so they all
 * start and finish together ("coordinated joint move"). Works with any runtime that
 * exposes `eval(t)`, `duration()`, and `valid()`.
 *
 * @code
 * #include "wet/trajectory/polynomial.hpp"
 * using namespace wet;
 *
 * constexpr auto p = design::min_jerk(0.0, 1.0, 0.5);  // quintic, 0.5 s
 * static_assert(p.success);
 *
 * PolynomialTrajectory<5, double> traj(p);
 * while (!traj.done()) {
 *     auto [pos, vel, acc, jerk] = traj.step(dt);
 * }
 * @endcode
 *
 * @see T. Flash & N. Hogan, "The coordination of arm movements: an experimentally
 *      confirmed mathematical model," J. Neurosci. 5(7), 1985 (minimum jerk).
 * @see D. Mellinger & V. Kumar, "Minimum snap trajectory generation and control
 *      for quadrotors," ICRA 2011 (minimum snap).
 * @see L. Biagiotti & C. Melchiorri, "Trajectory Planning for Automatic Machines
 *      and Robots," Springer, 2008, §4 (polynomial boundary-value formulation).
 */

#include <cstddef>
#include <type_traits>

#include "wet/matrix/matrix.hpp"
#include "wet/matrix/solve.hpp"
#include "wet/trajectory/trajectory_types.hpp"

namespace wet {

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
 * @ingroup trajectory
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

    [[nodiscard]] constexpr T    time() const { return t_; }
    [[nodiscard]] constexpr T    duration() const { return profile_.duration; }
    [[nodiscard]] constexpr bool done() const { return t_ >= profile_.duration; }
    [[nodiscard]] constexpr bool valid() const { return profile_.success; }

    [[nodiscard]] constexpr const design::PolyTrajectory<Order, T>& profile() const { return profile_; }

private:
    design::PolyTrajectory<Order, T> profile_{};
    T                                t_{T{0}};
};

/**
 * @ingroup trajectory
 * @brief Multi-axis coordination: time-scale each axis's profile to the slowest
 *        so a multi-DOF move starts and finishes synchronized ("linear" /
 *        coordinated joint moves — the feedforward reference for a manipulator).
 *
 * Each axis is planned independently (its own minimum-time profile). The bank
 * takes the synchronized duration `T_sync = max_i duration_i` and **time-scales**
 * every axis to it: axis `i` is evaluated at internal time `t·kᵢ` with
 * `kᵢ = duration_i / T_sync ≤ 1`, and its derivatives are scaled accordingly
 * (velocity ·kᵢ, acceleration ·kᵢ², jerk ·kᵢ³). Time-scaling preserves each path's
 * shape and can only *shrink* velocity/acceleration/jerk, so per-axis limits stay
 * satisfied; the slowest axis runs at its native min-time (kᵢ = 1) and the rest are
 * slowed to land together at `t = T_sync`.
 *
 * Works with any per-axis runtime that exposes `eval(t)`, `duration()` and
 * `valid()` — @ref TrapezoidalTrajectory, @ref ScurveTrajectory or
 * @ref PolynomialTrajectory (homogeneous across the bank).
 *
 * @note Time-scaling scales the *boundary* velocities too, so an axis's nonzero
 *       final velocity becomes `Vf·kᵢ`. Coordinated moves are normally
 *       rest-to-rest (Vi = Vf = 0), where this is exact; only the slowest axis
 *       retains a nonzero Vf unchanged.
 *
 * @tparam NAxes      Number of axes
 * @tparam Trajectory Per-axis runtime trajectory type
 */
template<size_t NAxes, typename Trajectory>
class TrajectoryBank {
public:
    using value_type = decltype(Trajectory{}.duration());
    using State = TrajectoryState<value_type>;
    using StateArray = wet::array<State, NAxes>;

    constexpr TrajectoryBank() = default;

    /// Construct from the per-axis trajectories (each already planned, min-time).
    constexpr explicit TrajectoryBank(const wet::array<Trajectory, NAxes>& axes) : axes_(axes) { sync(); }

    /// Replace all axes and rewind the clock.
    constexpr void set_axes(const wet::array<Trajectory, NAxes>& axes) {
        axes_ = axes;
        t_ = value_type{0};
        sync();
    }

    /// Replace one axis and rewind the clock (re-synchronizes the bank).
    constexpr void set_axis(size_t i, const Trajectory& traj) {
        axes_[i] = traj;
        t_ = value_type{0};
        sync();
    }

    /// Evaluate all axes at the common (synchronized) time @p t ∈ [0, T_sync].
    [[nodiscard]] constexpr StateArray eval(value_type t) const {
        StateArray out{};
        for (size_t i = 0; i < NAxes; ++i) {
            const value_type k = scale_[i];
            State            s = axes_[i].eval(t * k);
            s.velocity *= k;
            s.acceleration *= k * k;
            s.jerk *= k * k * k;
            out[i] = s;
        }
        return out;
    }

    /// Advance the common clock by @p dt and return the per-axis states.
    constexpr StateArray step(value_type dt) {
        t_ += dt;
        return eval(t_);
    }

    constexpr void reset() { t_ = value_type{0}; }

    [[nodiscard]] constexpr value_type        time() const { return t_; }
    [[nodiscard]] constexpr value_type        duration() const { return T_sync_; } //!< synchronized total time
    [[nodiscard]] constexpr bool              done() const { return t_ >= T_sync_; }
    [[nodiscard]] constexpr bool              valid() const { return valid_; }
    [[nodiscard]] constexpr value_type        scale(size_t i) const { return scale_[i]; }
    [[nodiscard]] constexpr const Trajectory& axis(size_t i) const { return axes_[i]; }

private:
    constexpr void sync() {
        valid_ = (NAxes > 0);
        T_sync_ = value_type{0};
        for (size_t i = 0; i < NAxes; ++i) {
            valid_ = valid_ && axes_[i].valid();
            const value_type d = axes_[i].duration();
            if (d > T_sync_) {
                T_sync_ = d;
            }
        }
        for (size_t i = 0; i < NAxes; ++i) {
            scale_[i] = (T_sync_ > value_type{0}) ? (axes_[i].duration() / T_sync_) : value_type{1};
        }
    }

    wet::array<Trajectory, NAxes> axes_{};
    wet::array<value_type, NAxes> scale_{};
    value_type                    T_sync_{value_type{0}};
    value_type                    t_{value_type{0}};
    bool                          valid_{false};
};

} // namespace wet
