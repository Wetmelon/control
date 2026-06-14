#pragma once

/**
 * @file trajectory.hpp
 * @brief Umbrella header for all point-to-point motion-profile generators.
 *
 * Includes every trajectory family in one shot. Include this header for the full
 * surface; include the individual sub-headers when you only need one family:
 *
 * | Sub-header              | Contents                                                      |
 * |-------------------------|---------------------------------------------------------------|
 * | trajectory_types.hpp    | Shared types: TrajectoryLimits, TrajectoryState, TrajectoryBoundary |
 * | trapezoidal.hpp         | TrapezoidalProfile, synthesize_trapezoidal, TrapezoidalTrajectory |
 * | scurve.hpp              | ScurveProfile, synthesize_scurve, ScurveTrajectory            |
 * | polynomial.hpp          | PolyTrajectory, synthesize_poly_trajectory, min_jerk/accel/snap, PolynomialTrajectory, TrajectoryBank |
 * | spline.hpp              | SplineProfile, synthesize_spline, cubic_spline, quintic_spline, SplineTrajectory |
 *
 * All profiles share the design pattern: a constexpr @p design:: function
 * synthesizes and validates the profile (returning it with a `success` flag), and a
 * runtime class holds the profile + an internal clock for @p step(dt) / @p eval(t).
 *
 * @see cartesian_move.hpp, topp.hpp for task-space path-following that uses these
 *      scalar profiles to parameterize a fixed Cartesian path.
 * @see input_shaper.hpp for residual-vibration suppression of the jerk-discontinuous
 *      trapezoidal output.
 */

#include "wet/trajectory/cartesian_move.hpp" // IWYU pragma: export
#include "wet/trajectory/input_shaper.hpp"   // IWYU pragma: export
#include "wet/trajectory/polynomial.hpp"     // IWYU pragma: export
#include "wet/trajectory/scurve.hpp"         // IWYU pragma: export
#include "wet/trajectory/spline.hpp"         // IWYU pragma: export
#include "wet/trajectory/topp.hpp"           // IWYU pragma: export
#include "wet/trajectory/trapezoidal.hpp"    // IWYU pragma: export
