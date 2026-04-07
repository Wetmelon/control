# Compile-Time Control Design System

## Overview

A header-only C++20 library for control system design targeting embedded systems. All controllers, observers, and matrix operations can be computed at compile time with zero dynamic allocation.

**Key design choices:**

- **Fixed-size `Matrix<N,M,T>`** — stack-allocated, fully `constexpr`
- **`design::` namespace** — `consteval` functions that execute at compile time only
- **`online::` namespace** — `constexpr` functions that can also run at runtime
- **`.as<float>()`** — design in `double`, convert results to `float` for embedded deployment
- **`std::optional`** — returned for operations that can fail (matrix inversion, DARE convergence)
- **Operator overloads** — `sys1 * sys2` (series), `sys1 + sys2` (parallel), `sys1 / sys2` (feedback)

Supports `float`, `double`, `wet::complex<float>`, and `wet::complex<double>`.

**What's included:**

- LQR, LQI, LQG, LQGI, Kalman filter design (DARE via Structure-preserving Doubling Algorithm)
- PID tuning (Ziegler-Nichols, Cohen-Coon, Lambda, SIMC, pole placement)
- Lead-lag compensators, proportional-resonant controllers, sliding mode, ADRC
- Frequency-domain analysis (Bode, margins, bandwidth, controllability/observability)
- ODE solvers (RK4, RK45 with event detection), closed-loop simulation
- Digital filters (LowPass, SOGI, notch), IEC 61131-3 function blocks
- Rotations (DCM, Quaternion, Euler), sensor fusion (ESKF, EKF, Madgwick, Mahony)
- Motor control (Clarke/Park transforms, FOC)
- Discretization (Forward Euler, ZOH, Tustin)

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
auto result = design::lqrd(sys.A, sys.B, Q, R, Ts).as<float>();
static_assert(result.success);

// Runtime control loop
LQR controller(result);
ColVec<1> u = controller.control(x);  // u = -K*x
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
#include "filters.hpp"      // LowPass, SOGI, notch filters
#include "rotation.hpp"     // DCM, Quaternion, Euler angles
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
