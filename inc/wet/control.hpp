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
 * aliases — lives behind @ref toolbox.hpp instead and is opt-in for host builds.
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
 * @see toolbox.hpp for the host-side analysis/simulation/plotting superset.
 */

// --- Linear algebra + math core ---------------------------------------------
#include "wet/matrix/matrix.hpp" // IWYU pragma: keep

// --- LTI system types -------------------------------------------------------
#include "wet/systems/discretization.hpp"    // IWYU pragma: keep
#include "wet/systems/state_space.hpp"       // IWYU pragma: keep
#include "wet/systems/transfer_function.hpp" // IWYU pragma: keep

// --- Structural analysis (allocation-free) ----------------------------------
#include "wet/analysis/linearization.hpp" // IWYU pragma: keep
#include "wet/analysis/stability.hpp"     // IWYU pragma: keep

// --- Controllers + design synthesis -----------------------------------------
#include "wet/analysis/riccati.hpp"                 // IWYU pragma: keep
#include "wet/controllers/adrc.hpp"                 // IWYU pragma: keep
#include "wet/controllers/cartesian_move.hpp"       // IWYU pragma: keep
#include "wet/controllers/cascade.hpp"              // IWYU pragma: keep
#include "wet/controllers/esc.hpp"                  // IWYU pragma: keep
#include "wet/controllers/excitation.hpp"           // IWYU pragma: keep
#include "wet/controllers/harmonic_suppression.hpp" // IWYU pragma: keep
#include "wet/controllers/input_shaper.hpp"         // IWYU pragma: keep
#include "wet/controllers/lead_lag.hpp"             // IWYU pragma: keep
#include "wet/controllers/lqg.hpp"                  // IWYU pragma: keep
#include "wet/controllers/lqgi.hpp"                 // IWYU pragma: keep
#include "wet/controllers/lqi.hpp"                  // IWYU pragma: keep
#include "wet/controllers/lqr.hpp"                  // IWYU pragma: keep
#include "wet/controllers/pid.hpp"                  // IWYU pragma: keep
#include "wet/controllers/pid_design.hpp"           // IWYU pragma: keep
#include "wet/controllers/pll.hpp"                  // IWYU pragma: keep
#include "wet/controllers/pr.hpp"                   // IWYU pragma: keep
#include "wet/controllers/relay_autotune.hpp"       // IWYU pragma: keep
#include "wet/controllers/repetitive.hpp"           // IWYU pragma: keep
#include "wet/controllers/smc.hpp"                  // IWYU pragma: keep
#include "wet/controllers/synthesis.hpp"            // IWYU pragma: keep
#include "wet/controllers/topp.hpp"                 // IWYU pragma: keep
#include "wet/controllers/trajectory.hpp"           // IWYU pragma: keep

// --- Estimators -------------------------------------------------------------
#include "wet/estimation/disturbance_observer.hpp"    // IWYU pragma: keep
#include "wet/estimation/ekf.hpp"                     // IWYU pragma: keep
#include "wet/estimation/eskf.hpp"                    // IWYU pragma: keep
#include "wet/estimation/kalman.hpp"                  // IWYU pragma: keep
#include "wet/estimation/observer.hpp"                // IWYU pragma: keep
#include "wet/estimation/recursive_least_squares.hpp" // IWYU pragma: keep
#include "wet/estimation/sensor_fusion.hpp"           // IWYU pragma: keep
#include "wet/estimation/ukf.hpp"                     // IWYU pragma: keep

// NOTE: estimation/harmonic_estimation.hpp and estimation/identification.hpp are
// intentionally NOT included here. They are placeholders for planned features (harmonic
// detection, system identification) that expose only result-struct skeletons,
// not working APIs. Keeping them out of the umbrella avoids advertising unshipped
// surface as part of the embeddable contract; include them directly if you are
// developing those features.

// --- Kinematics (allocation-free) -------------------------------------------
#include "wet/kinematics/motion_maps.hpp" // IWYU pragma: keep
#include "wet/kinematics/pose.hpp"        // IWYU pragma: keep
#include "wet/kinematics/scara.hpp"       // IWYU pragma: keep
#include "wet/kinematics/serial_arm.hpp"  // IWYU pragma: keep
#include "wet/kinematics/stewart.hpp"     // IWYU pragma: keep

// --- Signal filters ---------------------------------------------------------
#include "wet/filters/differentiator.hpp" // IWYU pragma: keep
#include "wet/filters/filters.hpp"        // IWYU pragma: keep
#include "wet/filters/sogi.hpp"           // IWYU pragma: keep
#include "wet/filters/spectral.hpp"       // IWYU pragma: keep

// --- Fixed-step integration (allocation-free) -------------------------------
#include "wet/simulation/integrator.hpp" // IWYU pragma: keep

// --- Domain helpers ---------------------------------------------------------
#include "wet/utility/geometry.hpp"      // IWYU pragma: keep
#include "wet/utility/iec61131.hpp"      // IWYU pragma: keep
#include "wet/utility/motor_control.hpp" // IWYU pragma: keep

// --- Embedded firmware primitives (allocation-free leaf utilities) ----------
// Controls/DSP-specific helpers only; generic plumbing (containers, CRC, FIFOs)
// is out of scope — pair with ETL for those.
#include "wet/utility/encoder.hpp" // IWYU pragma: keep
#include "wet/utility/lookup.hpp"  // IWYU pragma: keep
#include "wet/utility/scaling.hpp" // IWYU pragma: keep
#include "wet/utility/timing.hpp"  // IWYU pragma: keep
