#pragma once

/**
 * @file control.hpp
 * @brief Umbrella header for the embeddable core of wet.
 *
 * Includes everything that is allocation-free and safe to compile for an
 * embedded target: the linear-algebra core, LTI system types, all runtime
 * controllers, the constexpr `design::` synthesis functions, estimators,
 * filters, fixed-step integration, and the domain helpers.
 *
 * Nothing reachable from this header allocates on the heap or pulls a
 * third-party dependency. Anything that does — frequency-domain analysis
 * (Bode/Nyquist/sweeps), simulation harnesses, plotting, and the MATLAB-style
 * aliases — lives behind @ref workbench.hpp instead and is opt-in for host builds.
 *
 * @code
 * #include "wet/control.hpp"          // one include, embedded-safe
 * using namespace wet;
 *
 * constexpr auto art = design::synthesize_lqi(sys_d, Q_aug, R);
 * static_assert(art.success);
 * constinit LQI controller{art.runtime.controller};
 * @endcode
 *
 * @see workbench.hpp for the host-side analysis/simulation/plotting superset.
 */

// --- Linear algebra + math core ---------------------------------------------
#include "wet/matrix/matrix.hpp" // IWYU pragma: export

// --- LTI system types -------------------------------------------------------
#include "wet/systems/discretization.hpp"    // IWYU pragma: export
#include "wet/systems/state_space.hpp"       // IWYU pragma: export
#include "wet/systems/transfer_function.hpp" // IWYU pragma: export
#include "wet/systems/zpk.hpp"               // IWYU pragma: export

// --- Structural analysis (allocation-free) ----------------------------------
#include "wet/design/linearization.hpp" // IWYU pragma: export
#include "wet/design/stability.hpp"     // IWYU pragma: export

// --- Controllers ------------------------------------------------------------
#include "wet/controllers/adrc.hpp"                 // IWYU pragma: export
#include "wet/controllers/esc.hpp"                  // IWYU pragma: export
#include "wet/controllers/harmonic_suppression.hpp" // IWYU pragma: export
#include "wet/controllers/lead_lag.hpp"             // IWYU pragma: export
#include "wet/controllers/lqg.hpp"                  // IWYU pragma: export
#include "wet/controllers/lqgi.hpp"                 // IWYU pragma: export
#include "wet/controllers/lqi.hpp"                  // IWYU pragma: export
#include "wet/controllers/lqr.hpp"                  // IWYU pragma: export
#include "wet/controllers/pid.hpp"                  // IWYU pragma: export
#include "wet/controllers/pr.hpp"                   // IWYU pragma: export
#include "wet/controllers/repetitive.hpp"           // IWYU pragma: export
#include "wet/controllers/smc.hpp"                  // IWYU pragma: export
#include "wet/design/riccati.hpp"                   // IWYU pragma: export

// --- Trajectory + motion planning -------------------------------------------
#include "wet/trajectory/cartesian_move.hpp" // IWYU pragma: export
#include "wet/trajectory/input_shaper.hpp"   // IWYU pragma: export
#include "wet/trajectory/polynomial.hpp"     // IWYU pragma: export
#include "wet/trajectory/scurve.hpp"         // IWYU pragma: export
#include "wet/trajectory/spline.hpp"         // IWYU pragma: export
#include "wet/trajectory/topp.hpp"           // IWYU pragma: export
#include "wet/trajectory/trapezoidal.hpp"    // IWYU pragma: export

// --- Controller design tools ------------------------------------------------
#include "wet/design/pid_design.hpp"     // IWYU pragma: export
#include "wet/design/pole_placement.hpp" // IWYU pragma: export
#include "wet/design/synthesis.hpp"      // IWYU pragma: export

// --- Estimators -------------------------------------------------------------
#include "wet/estimation/disturbance_observer.hpp"    // IWYU pragma: export
#include "wet/estimation/ekf.hpp"                     // IWYU pragma: export
#include "wet/estimation/eskf.hpp"                    // IWYU pragma: export
#include "wet/estimation/excitation.hpp"              // IWYU pragma: export
#include "wet/estimation/kalman.hpp"                  // IWYU pragma: export
#include "wet/estimation/observer.hpp"                // IWYU pragma: export
#include "wet/estimation/recursive_least_squares.hpp" // IWYU pragma: export
#include "wet/estimation/relay_autotune.hpp"          // IWYU pragma: export
#include "wet/estimation/sensor_fusion.hpp"           // IWYU pragma: export
#include "wet/estimation/ukf.hpp"                     // IWYU pragma: export

// --- Kinematics (allocation-free) -------------------------------------------
#include "wet/kinematics/motion_maps.hpp" // IWYU pragma: export
#include "wet/kinematics/pose.hpp"        // IWYU pragma: export
#include "wet/kinematics/scara.hpp"       // IWYU pragma: export
#include "wet/kinematics/serial_arm.hpp"  // IWYU pragma: export
#include "wet/kinematics/stewart.hpp"     // IWYU pragma: export

// --- Signal filters ---------------------------------------------------------
#include "wet/filters/differentiator.hpp" // IWYU pragma: export
#include "wet/filters/filters.hpp"        // IWYU pragma: export
#include "wet/filters/pll.hpp"            // IWYU pragma: export
#include "wet/filters/sogi.hpp"           // IWYU pragma: export
#include "wet/filters/spectral.hpp"       // IWYU pragma: export

// --- Fixed-step integration (allocation-free) -------------------------------
#include "wet/simulation/integrator.hpp" // IWYU pragma: export

// --- 3D math types ----------------------------------------------------------
#include "wet/math/geometry.hpp" // IWYU pragma: export

// --- Motor / power-electronics drives ---------------------------------------
#include "wet/power/calibration.hpp"          // IWYU pragma: export
#include "wet/power/foc.hpp"                  // IWYU pragma: export
#include "wet/power/limits.hpp"               // IWYU pragma: export
#include "wet/power/mechanical_estimator.hpp" // IWYU pragma: export
#include "wet/power/modulation.hpp"           // IWYU pragma: export
#include "wet/power/servo.hpp"                // IWYU pragma: export
#include "wet/power/thermal.hpp"              // IWYU pragma: export
#include "wet/power/transforms.hpp"           // IWYU pragma: export

// --- Toolbox: allocation-free embedded primitives every controls project ----
// reaches for. Controls/DSP-specific helpers only; generic plumbing
// (containers, CRC, FIFOs) is out of scope — pair with ETL for those.
#include "wet/toolbox/actuator.hpp"     // IWYU pragma: export
#include "wet/toolbox/bounds.hpp"       // IWYU pragma: export
#include "wet/toolbox/conditioning.hpp" // IWYU pragma: export
#include "wet/toolbox/encoder.hpp"      // IWYU pragma: export
#include "wet/toolbox/iec61131.hpp"     // IWYU pragma: export
#include "wet/toolbox/io.hpp"           // IWYU pragma: export
#include "wet/toolbox/logic.hpp"        // IWYU pragma: export
#include "wet/toolbox/lookup.hpp"       // IWYU pragma: export
#include "wet/toolbox/scaling.hpp"      // IWYU pragma: export
#include "wet/toolbox/timing.hpp"       // IWYU pragma: export
