# Compile-Time Control Design System

## Overview

A header-only C++20 library for control system design targeting embedded systems. All controllers, observers, and matrix operations can be computed at compile time with zero dynamic allocation.

**Key design choices:**

- **Fixed-size `Matrix<N,M,T>`** — stack-allocated, fully `constexpr`
- **`.as<float>()`** — design in `double`, convert results to `float` for embedded deployment
- **`std::optional`** — returned for operations that can fail (matrix inversion, DARE convergence)
- **Operator overloads** — `sys1 * sys2` (series), `sys1 + sys2` (parallel), `sys1 / sys2` (feedback)

Supports `float`, `double`, `wet::complex<float>`, and `wet::complex<double>`.

## Library Contents

For a flat index of every public type and function — categorized,
one row each with a one-line description — see **[REFERENCE.md](REFERENCE.md)**
(generated from the headers by `python tools/gen_reference.py`). The prose below is the
guided tour.

Every header follows the three-tier pattern (constexpr `design::` → result struct with
`.as<U>()` → allocation-free runtime). See [decisions.md](inc/wet/decisions.md) for the
design rationale and [inc/wet/roadmap.md](inc/wet/roadmap.md) for planned work.

### Core modeling and math

- Fixed-size `constexpr` matrix algebra: arithmetic, blocks, views, LU/QR/Cholesky/eigen decompositions, linear solve, matrix functions (`matrix/*`)
- Constexpr scalar math and `wet::complex` (`math/*`); selectable stdlib/freestanding math backend
- State-space models and interconnections — series, parallel, feedback (`systems/state_space.hpp`)
- Transfer functions and block-diagram arithmetic (`systems/transfer_function.hpp`)
- Continuous-to-discrete conversion — Forward Euler, ZOH, Tustin (`systems/discretization.hpp`)

### Runtime controllers (`controllers/`)

- PID with anti-windup (`pid.hpp`)
- Proportional-resonant and multi-harmonic PR (`pr.hpp`)
- LQR / LQI / LQG / LQGI optimal state-feedback (`lqr.hpp`, `lqi.hpp`, `lqg.hpp`, `lqgi.hpp`)
- Lead-lag compensator (`lead_lag.hpp`)
- ADRC — active disturbance rejection with extended-state observer (`adrc.hpp`)
- Sliding-mode control — first-order with boundary layer (`smc.hpp`) and super-twisting / generalized STA (`stsmc.hpp`)
- Extremum-seeking control + MPPT (`esc.hpp`)
- Repetitive control — internal-model periodic-disturbance rejection (`repetitive.hpp`)
- Selective harmonic suppression — parallel PR resonator bank (`harmonic_suppression.hpp`)

### Design / synthesis (`design/`)

- Robust MIMO pole placement (`design::place`, Kautsky–Nichols–Van Dooren) + single-input Ackermann (`pole_placement.hpp`)
- Riccati / DARE solvers backing the LQ family (`riccati.hpp`)
- PID tuning rules: Ziegler-Nichols, Tyreus-Luyben, Cohen-Coon, SIMC, lambda, bandwidth, pole placement, plus settling-time/overshoot spec synthesis (`pid_design.hpp`)
- Relay (Åström–Hägglund) autotuner runtime (`relay_autotune.hpp`)
- Nonlinear operating-point linearization to A/B/C/D (`linearization.hpp`)
- Stability/structural analysis primitives (`stability.hpp`)
- High-level workflow artifacts — combined design + analysis + runtime bundle (`synthesis.hpp`)

### Observers and estimators (`estimation/`)

- Linear Kalman filter (`kalman.hpp`), Extended (`ekf.hpp`), Error-State (`eskf.hpp`), Unscented / sigma-point (`ukf.hpp`)
- Luenberger full-order and reduced-order (Gopinath) observers (`observer.hpp`)
- Disturbance observers — scalar innovation-based + classical Pn⁻¹·Q (Ohnishi) bolt-on (`disturbance_observer.hpp`)
- Sensor-fusion filters: Complementary, Madgwick, Mahony, ESKF orientation (`sensor_fusion.hpp`)
- Recursive least squares (`recursive_least_squares.hpp`)
- Excitation generators — Chirp, PRBS, StepTrain, Ramp, MultiSine (`excitation.hpp`)
- Identification model types — FOPDT/SOPDT/ARX, fit/validation metrics (`identification.hpp`)

### Filters and signal conditioning (`filters/`)

- Biquad family (RBJ notch/bandpass/peaking/shelf) + utility runtime blocks (moving average, RMS, median, rate limiter, dirty derivative, clamped integrator, deadband, hysteresis, EWMA, peak/envelope) (`filters.hpp`)
- Spectral primitives — Goertzel single-bin DFT + harmonic analyzer/THD (`spectral.hpp`)
- Robust exact (Levant super-twisting) differentiator (`differentiator.hpp`)
- SOGI / MSTOGI and SOGI-FLL self-tuning tracker (`sogi.hpp`)
- PLLs and sensorless PMSM flux/position estimator, incl. three-phase DSOGI sequence PLL (`pll.hpp`)

### Trajectory and motion planning (`trajectory/`)

- Trapezoidal and S-curve time-optimal profiles, arbitrary boundary velocities + asymmetric limits (`trapezoidal.hpp`, `scurve.hpp`)
- Polynomial BVP trajectories — min-jerk/accel/snap + `TrajectoryBank` multi-axis coordination (`polynomial.hpp`)
- Multi-waypoint C²/C⁴ splines (`spline.hpp`)
- Input shaping — ZV/ZVD/ZVDD/EI command prefilter (`input_shaper.hpp`)
- Cartesian / task-space path-preserving moves (`cartesian_move.hpp`) and time-optimal path parameterization / TOPP (`topp.hpp`)

### Kinematics (`kinematics/`)

- `Pose` (quaternion + translation) interchange type (`pose.hpp`)
- Motion-system maps — Cartesian, CoreXY, polar, rotary/linear delta (`motion_maps.hpp`)
- Stewart platform — 6-DOF parallel manipulator, closed-form IK + Newton FK (`stewart.hpp`)
- Serial N-DOF arm (N≤6) — DH chains, geometric Jacobian, damped-least-squares IK (`serial_arm.hpp`)
- SCARA — series RRPR + parallel five-bar (`scara.hpp`)

### Utilities (`utility/`)

- Scaling/calibration (`scaling.hpp`), interpolation LUTs (`lookup.hpp`), software timers (`timing.hpp`)
- Quadrature encoder + tachometer (`encoder.hpp`), thermistor linearization (`thermistor.hpp`), actuator models (`actuator.hpp`)
- Geometry/attitude — DCM, Quaternion, Euler, Vec3 (`geometry.hpp`)
- Motor-control: Clarke/Park transforms (`transforms.hpp`), SVPWM modulation (`modulation.hpp`), field-oriented control (`foc.hpp`)
- PMAC servo drive — `PmacServo` integrates the FOC current loop, bandwidth-tuned velocity/position cascade, a `[θ,ω,τ_load]` Kalman estimator (encoder / sensorless / load-accel channels), RLS R/L commissioning (`calibration.hpp`), DC-bus + Cauer-network junction-temperature limiting (`limits.hpp`, `thermal.hpp`); `{Iabc,Vdc,θ}` in, duties out (`servo.hpp`). See [examples/example_pmac_servo.cpp](examples/example_pmac_servo.cpp)
- IEC 61131-3 function blocks for PLC-style logic (`iec61131.hpp`)

### Analysis, simulation, interop (host-only behind `toolbox.hpp`)

- Frequency-domain analysis — Bode, Nyquist, margins, bandwidth, loop metrics (`analysis.hpp`)
- ODE solvers and closed-loop simulation (`simulation/{solver,integrator,simulate}.hpp`)
- Plotting — text/console and Plotly/SVG (`plotting/{plot,plot_plotly}.hpp`)
- MATLAB-style short-name wrappers (`matlab.hpp`)

## Quick Start

```cpp
#include "wet/control.hpp"   // one embeddable umbrella

using namespace wet;

// Double integrator: states = [position, velocity], input = acceleration
constexpr StateSpace sys{
    .A = Matrix<2, 2>{{0.0, 1.0}, {0.0, 0.0}},
    .B = Matrix<2, 1>{{0.0}, {1.0}},
    .C = Matrix<1, 2>{{1.0, 0.0}}
};

constexpr auto Q  = Matrix<2, 2>::identity();
constexpr auto R  = Matrix<1, 1>{{0.1}};
constexpr auto Ts = 0.01;  // 100 Hz

// Design at compile time, convert to float for embedded use
constexpr auto result = lqrd(sys.A, sys.B, Q, R, Ts).as<float>();
static_assert(result.success);

// constinit ensures gain matrix K is computed at compile time
constinit LQR lqr = result;
ColVec<1> u = lqr.control(x);  // u = -K*x
```

## Using in Your Project

Header-only — copy the `inc/` directory into your project. A single include path is all you need:

```bash
g++ -std=c++20 -I path/to/inc your_code.cpp
```

### Two umbrellas

| Header | Use it for | Allocates? |
| ------ | ---------- | ---------- |
| `wet/control.hpp` | The embeddable library: linear algebra, LTI types, every runtime controller, the constexpr `design::` synthesis, estimators, filters, fixed-step integration, domain helpers. | No — nothing reachable from it touches the heap. Safe on an MCU. |
| `wet/toolbox.hpp` | Host/desktop superset: everything above **plus** frequency-domain analysis (Bode/Nyquist/margins/sweeps), ODE solvers, closed-loop simulation, plotting, and the MATLAB-style aliases. | Yes (`std::vector`) — for workstation design work, not the target. |

```cpp
#include "wet/control.hpp"   // firmware: heap-free, one include
// ... or ...
#include "wet/toolbox.hpp"   // host: design + analyze + simulate
```

Or reach for a single feature directly:

```cpp
#include "wet/controllers/lqr.hpp"      // LQR/LQI design + controller classes
#include "wet/estimation/kalman.hpp"    // Kalman filter
#include "wet/controllers/pid.hpp"      // PID with anti-windup
#include "wet/filters/filters.hpp"      // LowPass and delay/filter helpers
#include "wet/utility/geometry.hpp"             // DCM, Quaternion, Euler, Transform4
#include "wet/systems/state_space.hpp"  // StateSpace type and interconnections
```

### Layout

```
inc/wet/
  control.hpp      embeddable umbrella        toolbox.hpp   host superset
  config.hpp · backend.hpp                    backend profile (stdlib / ETL)
  math/            constexpr math, complex, backends
  matrix/          fixed-size linear algebra, decompositions, solve, eigen
  systems/         state_space, transfer_function, discretization
  controllers/     pid, pr, lqr, lqi, lqg, lqgi, lead_lag, adrc,
                   smc, stsmc, esc, repetitive, harmonic_suppression
  design/          pole_placement, riccati, pid_design, relay_autotune,
                   linearization, stability, synthesis
  estimation/      kalman, ekf, eskf, ukf, observer, disturbance_observer,
                   sensor_fusion, recursive_least_squares, excitation, identification
  filters/         filters, spectral, differentiator, sogi, pll (incl. DSOGI + sensorless)
  trajectory/      trapezoidal, scurve, polynomial, spline, input_shaper,
                   cartesian_move, topp
  kinematics/      pose, motion_maps, stewart, serial_arm, scara
  utility/         scaling, lookup, timing, encoder, thermistor, actuator,
                   geometry, transforms, modulation, foc, iec61131
  analysis.hpp · matlab.hpp                    (host)
  simulation/      integrator                  + solver, simulate (host)
  plotting/        plot, plot_plotly           (host)
```

Planning docs live alongside the code: [inc/wet/roadmap.md](inc/wet/roadmap.md)
(what's next) and [inc/wet/decisions.md](inc/wet/decisions.md) (why it's built this way).

The one-call synthesis helpers live in `design::` (e.g. `design::synthesize_lqgi`) — design + analysis models + a ready-to-deploy runtime bundle in a single constexpr call.

No build step, no linking, no dependencies beyond a C++20 compiler (GCC 10+, Clang 12+, MSVC 2022+). The Plotly plotting backend (`wet/simulation/plot_plotly.hpp`) additionally needs plotlypp and nlohmann-json.

## Building Examples

The build system uses [tup](https://gittup.org/tup/). From the repo root:

```bash
make          # format, build, and run all tests
make tests    # build and run tests only
```

Examples are built by the default `make` (`tup`) build. To compile examples directly, run `tup --quiet examples`.
Outputs go to `examples/build/`.

## Servo Drive GUI

An interactive PMSM servo drive simulator with real-time plots and tunable parameters:

```bash
make gui
```

This builds a C++ simulation DLL (`servo_sim.dll`) and launches a Python GUI using [dearpygui](https://github.com/hoffstadt/DearPyGui). Features:

- P-PI-PI cascade control (position → speed → current) with bandwidth-based current loop design
- Two-mass mechanical model (motor + flexible coupling + load with Coulomb friction)
- Live-adjustable motor, coupling, load, and controller parameters
- Real-time plots (position, speed, torque, current, voltage)
- Auto plant sub-stepping for numerical stability
- Real-time lock mode with color-coded performance indicator

**Requirements:** Python 3 with `dearpygui` (`pip install dearpygui`)
