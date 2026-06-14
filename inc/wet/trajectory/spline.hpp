#pragma once

/**
 * @file spline.hpp
 * @brief Multi-waypoint piecewise-polynomial splines (C² cubic, C⁴ quintic).
 *
 * Interpolates a list of waypoints with one piecewise polynomial per interval,
 * joined so the whole curve is globally smooth. Two families:
 * - **Cubic (Order 3)** — C² (continuous acceleration). Solved by matching
 *   position at both ends of each segment, continuity of velocity and acceleration
 *   at interior knots, and clamped end velocities.
 * - **Quintic (Order 5)** — C⁴ (continuous jerk *and* snap). Adds continuity of
 *   jerk and snap at interior knots, and clamped end velocity + acceleration.
 *
 * The global coefficient system (`(Order+1)(NPts−1)` unknowns) is assembled in one
 * shot and solved with @ref mat::solve — the multi-segment generalization of
 * @ref design::synthesize_poly_trajectory (which is the NPts == 2 single-segment
 * special case).
 *
 * @code
 * #include "wet/trajectory/spline.hpp"
 * using namespace wet;
 *
 * constexpr wet::array<double, 4> t = {0.0, 1.0, 2.5, 4.0};
 * constexpr wet::array<double, 4> p = {0.0, 1.0, 0.5, 2.0};
 * constexpr auto prof = design::cubic_spline(t, p);  // clamped v = 0 at ends
 * static_assert(prof.success);
 *
 * SplineTrajectory<4, 3, double> traj(prof);
 * while (!traj.done()) {
 *     auto [pos, vel, acc, jerk] = traj.step(dt);
 * }
 * @endcode
 *
 * @see L. Biagiotti & C. Melchiorri, "Trajectory Planning for Automatic Machines
 *      and Robots," Springer, 2008, §4.4 (cubic/quintic multi-segment splines).
 * @see polynomial.hpp for the single-segment case and the min-jerk / min-snap
 *      derivative-optimal wrappers.
 */

#include <cstddef>
#include <type_traits>

#include "wet/matrix/matrix.hpp"
#include "wet/matrix/solve.hpp"
#include "wet/trajectory/trajectory_types.hpp"

namespace wet {

namespace design {

/**
 * @brief A synthesized multi-waypoint spline: per-segment polynomial coefficients
 *        (ascending power, in segment-local time) plus the knot times.
 *
 * @tparam NPts  Number of waypoints (≥ 2)
 * @tparam Order Segment polynomial order: 3 (cubic, C²) or 5 (quintic, C⁴)
 * @tparam T     Scalar type
 */
template<size_t NPts, size_t Order, typename T = double>
struct SplineProfile {
    static_assert(NPts >= 2, "a spline needs at least two waypoints");
    static_assert(Order == 3 || Order == 5, "spline Order is 3 (cubic, C²) or 5 (quintic, C⁴)");

    static constexpr size_t NSeg = NPts - 1;
    static constexpr size_t NumCoeffs = Order + 1;

    wet::array<T, NPts>                        knots{};  //!< knot times, shifted so knots[0] = 0
    wet::array<wet::array<T, NumCoeffs>, NSeg> coeffs{}; //!< per-segment coeffs, local time τ = t − knots[k]
    T                                          duration{T{0}};
    bool                                       success{false};

    /// Evaluate position/velocity/acceleration/jerk at time @p t (clamped to [0, T]).
    [[nodiscard]] constexpr TrajectoryState<T> eval(T t) const {
        TrajectoryState<T> s{};
        if (!success) {
            return s;
        }
        T tc = t;
        if (tc < T{0}) {
            tc = T{0};
        } else if (tc > duration) {
            tc = duration;
        }
        size_t k = 0;
        while (k + 1 < NSeg && knots[k + 1] <= tc) {
            ++k;
        }
        const T     tau = tc - knots[k];
        const auto& c = coeffs[k];
        // Horner-with-derivatives (synthetic division), as in PolyTrajectory.
        T b = c[NumCoeffs - 1];
        T b1{0};
        T b2{0};
        T b3{0};
        for (size_t i = NumCoeffs - 1; i-- > 0;) {
            b3 = (b3 * tau) + b2;
            b2 = (b2 * tau) + b1;
            b1 = (b1 * tau) + b;
            b = (b * tau) + c[i];
        }
        s.position = b;
        s.velocity = b1;
        s.acceleration = T{2} * b2;
        s.jerk = T{6} * b3;
        return s;
    }

    /// Rebind to another scalar type.
    template<typename U>
    [[nodiscard]] constexpr SplineProfile<NPts, Order, std::remove_const_t<U>> as() const {
        using O = std::remove_const_t<U>;
        SplineProfile<NPts, Order, O> out{};
        for (size_t i = 0; i < NPts; ++i) {
            out.knots[i] = static_cast<O>(knots[i]);
        }
        for (size_t k = 0; k < NSeg; ++k) {
            for (size_t j = 0; j < NumCoeffs; ++j) {
                out.coeffs[k][j] = static_cast<O>(coeffs[k][j]);
            }
        }
        out.duration = static_cast<O>(duration);
        out.success = success;
        return out;
    }
};

/**
 * @brief Synthesize a multi-waypoint spline through @p points at @p times.
 *
 * Each of the `NPts − 1` segments is a degree-`Order` polynomial; the global
 * system enforces interpolation at both ends of every segment, continuity of
 * derivatives `1 … Order−1` at the interior knots, and the boundary conditions
 * (clamped end velocity for cubic; end velocity + acceleration for quintic). The
 * `((Order+1)(NPts−1))²` system is solved with @ref mat::solve.
 *
 * @tparam NPts  Number of waypoints
 * @tparam Order 3 (cubic, C²) or 5 (quintic, C⁴ — jerk- and snap-continuous)
 * @param times    Strictly increasing knot times.
 * @param points   Waypoint positions.
 * @param bc_start End-condition derivatives at the first knot (velocity; + accel for quintic).
 * @param bc_end   End-condition derivatives at the last knot.
 * @return Profile with `success`; `success == false` on non-increasing times or a
 *         singular system.
 */
template<size_t NPts, size_t Order, typename T = double>
[[nodiscard]] constexpr SplineProfile<NPts, Order, T> synthesize_spline(
    const wet::array<T, NPts>&   times,
    const wet::array<T, NPts>&   points,
    const TrajectoryBoundary<T>& bc_start = TrajectoryBoundary<T>{},
    const TrajectoryBoundary<T>& bc_end = TrajectoryBoundary<T>{}
) {
    static_assert(Order == 3 || Order == 5, "spline Order is 3 (cubic) or 5 (quintic)");
    constexpr size_t NSeg = NPts - 1;
    constexpr size_t NC = Order + 1;
    constexpr size_t Ncoef = NC * NSeg;

    SplineProfile<NPts, Order, T> sp{};
    for (size_t i = 0; i < NPts; ++i) {
        sp.knots[i] = times[i] - times[0]; // shift so knots[0] = 0
    }
    for (size_t i = 0; i + 1 < NPts; ++i) {
        if (!(sp.knots[i + 1] > sp.knots[i])) {
            return sp; // non-increasing times -> success == false
        }
    }
    sp.duration = sp.knots[NPts - 1];

    const auto              col = [](size_t k, size_t j) { return (k * NC) + j; };
    Matrix<Ncoef, Ncoef, T> A{};
    Matrix<Ncoef, 1, T>     rhs{};
    size_t                  row = 0;

    // Interpolation: every segment hits its two endpoint waypoints.
    for (size_t k = 0; k < NSeg; ++k) {
        const T h = sp.knots[k + 1] - sp.knots[k];
        A(row, col(k, 0)) = T{1}; // s_k(0) = p_k
        rhs(row, 0) = points[k];
        ++row;
        for (size_t j = 0; j < NC; ++j) { // s_k(h) = p_{k+1}
            A(row, col(k, j)) = wet::pow(h, static_cast<int>(j));
        }
        rhs(row, 0) = points[k + 1];
        ++row;
    }

    // Continuity of derivatives 1 … Order−1 at each interior knot.
    for (size_t k = 1; k < NSeg; ++k) {
        const T h = sp.knots[k] - sp.knots[k - 1];
        for (size_t d = 1; d < Order; ++d) {
            for (size_t j = d; j < NC; ++j) {
                A(row, col(k - 1, j)) = detail::falling_factorial<T>(j, d) * wet::pow(h, static_cast<int>(j - d));
            }
            A(row, col(k, d)) -= detail::factorial<T>(d); // − s_k^{(d)}(0)
            ++row;
        }
    }

    // Boundary conditions: velocity (cubic) or velocity + acceleration (quintic).
    const size_t last = NSeg - 1;
    const T      hL = sp.knots[NPts - 1] - sp.knots[NPts - 2];
    const T      bs[3] = {T{0}, bc_start.velocity, bc_start.acceleration};
    const T      be[3] = {T{0}, bc_end.velocity, bc_end.acceleration};
    const size_t Dbc = (Order == 3) ? 1 : 2;
    for (size_t d = 1; d <= Dbc; ++d) {
        A(row, col(0, d)) = detail::factorial<T>(d); // s_0^{(d)}(0)
        rhs(row, 0) = bs[d];
        ++row;
        for (size_t j = d; j < NC; ++j) { // s_last^{(d)}(hL)
            A(row, col(last, j)) = detail::falling_factorial<T>(j, d) * wet::pow(hL, static_cast<int>(j - d));
        }
        rhs(row, 0) = be[d];
        ++row;
    }

    const auto x = mat::solve(A, rhs);
    if (!x) {
        return sp; // singular -> success == false
    }
    for (size_t k = 0; k < NSeg; ++k) {
        for (size_t j = 0; j < NC; ++j) {
            sp.coeffs[k][j] = x.value()(col(k, j), 0);
        }
    }
    sp.success = true;
    return sp;
}

/// Cubic (C²) spline through @p points at @p times; clamped end velocities.
template<size_t NPts, typename T = double>
[[nodiscard]] constexpr SplineProfile<NPts, 3, T> cubic_spline(
    const wet::array<T, NPts>& times, const wet::array<T, NPts>& points, T v_start = T{0}, T v_end = T{0}
) {
    TrajectoryBoundary<T> s{};
    TrajectoryBoundary<T> e{};
    s.velocity = v_start;
    e.velocity = v_end;
    return synthesize_spline<NPts, 3, T>(times, points, s, e);
}

/// Quintic (C⁴ — jerk- and snap-continuous) spline; clamped end velocity + accel.
template<size_t NPts, typename T = double>
[[nodiscard]] constexpr SplineProfile<NPts, 5, T> quintic_spline(
    const wet::array<T, NPts>& times, const wet::array<T, NPts>& points,
    T v_start = T{0}, T a_start = T{0}, T v_end = T{0}, T a_end = T{0}
) {
    TrajectoryBoundary<T> s{};
    TrajectoryBoundary<T> e{};
    s.velocity = v_start;
    s.acceleration = a_start;
    e.velocity = v_end;
    e.acceleration = a_end;
    return synthesize_spline<NPts, 5, T>(times, points, s, e);
}

} // namespace design

/**
 * @ingroup trajectory
 * @brief Runtime player for a multi-waypoint spline (@ref design::SplineProfile).
 *
 * Plays a synthesized spline with the usual `eval(t)` / `step(dt)` interface, so
 * it drops into a @ref TrajectoryBank like the other generators.
 *
 * @tparam NPts  Number of waypoints
 * @tparam Order 3 (cubic) or 5 (quintic)
 * @tparam T     Scalar type
 */
template<size_t NPts, size_t Order, typename T = double>
class SplineTrajectory {
public:
    constexpr SplineTrajectory() = default;
    constexpr explicit SplineTrajectory(const design::SplineProfile<NPts, Order, T>& profile) : profile_(profile) {}

    constexpr TrajectoryState<T> step(T dt) {
        t_ += dt;
        return profile_.eval(t_);
    }

    [[nodiscard]] constexpr TrajectoryState<T> eval(T t) const { return profile_.eval(t); }
    constexpr void                             reset() { t_ = T{0}; }

    [[nodiscard]] constexpr T    time() const { return t_; }
    [[nodiscard]] constexpr T    duration() const { return profile_.duration; }
    [[nodiscard]] constexpr bool done() const { return t_ >= profile_.duration; }
    [[nodiscard]] constexpr bool valid() const { return profile_.success; }

    [[nodiscard]] constexpr const design::SplineProfile<NPts, Order, T>& profile() const { return profile_; }

private:
    design::SplineProfile<NPts, Order, T> profile_{};
    T                                     t_{T{0}};
};

} // namespace wet
