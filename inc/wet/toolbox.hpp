#pragma once

/**
 * @file toolbox.hpp
 * @brief Host-side superset of wetmelon::control for design & analysis work.
 *
 * Includes everything in @ref control.hpp plus the tools you use while
 * designing on a workstation: frequency-domain analysis (Bode, Nyquist,
 * margins, `linspace`/`logspace` sweeps), the ODE solvers and closed-loop
 * simulation harness, simple plotting, and the MATLAB-style aliases.
 *
 * These extras allocate on the heap (`std::vector`), so this header is **not**
 * meant for embedded targets — deploy with @ref control.hpp and keep the
 * analysis here on the host.
 *
 * @note The Plotly/SVG backend (`wet/plotting/plot_plotly.hpp`) is intentionally
 *       not pulled in here: it requires plotlypp and nlohmann-json on the
 *       include path. Include it directly when you need it.
 *
 * @code
 * #include "wet/toolbox.hpp"          // host build: design + analyze + simulate
 * using namespace wetmelon::control;
 *
 * const auto omega = analysis::logspace(1.0, 1000.0, 200);
 * const auto metrics = analysis::loop_metrics(loop, omega);
 * @endcode
 *
 * @see control.hpp for the embeddable subset.
 */

#include "wet/control.hpp" // IWYU pragma: keep

// --- Host-only design & analysis tooling (heap-allocating) ------------------
#include "wet/analysis/analysis.hpp"   // IWYU pragma: keep  (Bode/Nyquist/margins/sweeps)
#include "wet/matlab.hpp"              // IWYU pragma: keep  (MATLAB-style aliases)
#include "wet/plotting/plot.hpp"       // IWYU pragma: keep  (text/console plotting)
#include "wet/simulation/simulate.hpp" // IWYU pragma: keep  (closed-loop simulation)
#include "wet/simulation/solver.hpp"   // IWYU pragma: keep  (RK4/RK45 solvers)
