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

### Core Modeling and Math

- Fixed-size matrix algebra with compile-time dimensions (`matrix.hpp`, `matrix/*`)
- State-space models and interconnections (series, parallel, feedback) (`state_space.hpp`)
- Transfer functions and block-diagram arithmetic (`transfer_function.hpp`)
- Continuous-to-discrete conversion methods (Forward Euler, ZOH, Tustin) (`discretization.hpp`)
- Stability and structural analysis primitives (`stability.hpp`, `analysis.hpp`)

### Runtime Controllers

- PID (`pid.hpp`)
- PR and multi-harmonic PR (`pr.hpp`)
- LQR (`lqr.hpp`)
- LQI (`lqi.hpp`)
- LQG (`lqg.hpp`)
- LQGI (`lqgi.hpp`)
- Lead-lag compensator (`lead_lag.hpp`)
- ADRC (`adrc.hpp`)
- Sliding Mode Control (SMC) (`smc.hpp`)
- Single-phase PLL (`pll.hpp`)

### Design/Tuning APIs

- LQR/LQI/LQG/LQGI synthesis result APIs (`lqr.hpp`, `lqi.hpp`, `lqg.hpp`, `lqgi.hpp`)
- PID tuning methods: Ziegler-Nichols, Tyreus-Luyben, Cohen-Coon, SIMC, lambda, bandwidth-based, pole placement (`pid_design.hpp`)
- PID performance-spec synthesis helpers from settling time and overshoot (`pid_design.hpp`)
- PR/ADRC/SMC/lead-lag design helpers (`pr.hpp`, `adrc.hpp`, `smc.hpp`, `lead_lag.hpp`)

### Observers and Estimators

- Linear Kalman filter (`kalman.hpp`)
- Extended Kalman Filter (EKF) (`ekf.hpp`)
- Error-State Kalman Filter (ESKF) (`eskf.hpp`)
- Sensor-fusion filters: Complementary, Madgwick, Mahony, ESKF orientation (`sensor_fusion.hpp`)

### Analysis and Simulation

- Frequency-domain analysis: Bode, Nyquist, margins, bandwidth, loop metrics (`analysis.hpp`)
- Nonlinear operating-point linearization to A/B/C/D (`linearization.hpp`)
- ODE solvers and closed-loop simulation helpers (`solver.hpp`, `integrator.hpp`, `simulate.hpp`)

### Signal Conditioning and Utilities

- SOGI and MSTOGI system models/runtime SOGI block (`sogi.hpp`)
- First/second-order and Butterworth low-pass design, delay approximations, runtime low-pass and delay blocks (`filters.hpp`)
- Geometry and attitude utilities: DCM, Quaternion, Euler (`geometry.hpp`)
- Motor-control transforms: Clarke/Park and inverse transforms (`motor_control.hpp`)
- IEC 61131-3 function blocks for PLC-style control logic (`iec61131.hpp`)

### Interop and High-Level Glue

- MATLAB-style short-name wrappers (`matlab.hpp`)
- High-level workflow artifacts for combined design + analysis + runtime assembly (`workflow.hpp`)

## Quick Start

```cpp
#include "lqr.hpp"
#include "state_space.hpp"

using namespace wetmelon::control;

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

Header-only — copy the `inc/` directory into your project and add it to your include path:

```bash
g++ -std=c++20 -I path/to/inc -I path/to/inc/matrix your_code.cpp
```

Include only what you need:

```cpp
#include "lqr.hpp"          // LQR/LQI design and controller classes
#include "kalman.hpp"       // Kalman filter design
#include "pid.hpp"          // PID controller with anti-windup
#include "filters.hpp"      // LowPass and delay/filter design helpers
#include "geometry.hpp"     // DCM, Quaternion, Euler angles, Transform4
#include "state_space.hpp"  // StateSpace type and interconnections
```

No build step, no linking, no dependencies beyond a C++20 compiler (GCC 10+, Clang 12+, MSVC 2022+).

## Building Examples

The build system uses [tup](https://gittup.org/tup/). From the repo root:

```bash
make          # format, build, and run all tests
make tests    # build and run tests only
```

Examples are built alongside tests. Outputs go to `examples/build/`.

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
