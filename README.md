# Compile-Time Control Design System

## Overview

A header-only C++20 library for designing optimal controllers and observers **at compile time**, with extensive linear algebra, rotation utilities, digital filters, IEC 61131-3 function blocks, and embedded control algorithms. Two evaluation modes:

- **`design::`** — `consteval` functions that **must** execute at compile time
- **`online::`** — `constexpr` functions that **can** execute at runtime

All designs produce fixed-size, stack/statically-allocated artifacts suitable for embedded systems with zero dynamic allocation.

## Quick Start

```cpp

#include "lqr.hpp"
#include "state_space.hpp"

// Define a continuous double-integrator system
// dx/dt = A*x + B*u,  y = C*x
// States: [position, velocity]
constexpr StateSpace sys{
    .A = Matrix<2, 2>{{0.0, 1.0}, {0.0, 0.0}},  // A: pos' = vel, vel' = accel
    .B = Matrix<2, 1>{{0.0}, {1.0}},            // B: acceleration = input
    .C = Matrix<1, 2>{{1.0, 0.0}}               // C: measure position
};

constexpr auto Q = Matrix<2, 2>::identity(); // Penalize position and velocity
constexpr auto R = Matrix<1, 1>{{0.1}};      // Penalize control effort

constexpr auto Ts = 0.01; // 100Hz control loop

// Design discrete-time single-precision LQR at compile time (constexpr)
auto lqr_result = design::lqrd(sys.A, sys.B, Q, R, Ts).as<float>();

// Use the controller
LQR controller(lqr_result);
ColVec<1> u = controller.control(x);  // u = -K*x
```

## Features

### Design API

| Function | Shorthand | Description |
|----------|-------------| --- |
| `discrete_lqr` |  `dlqr(A, B, Q, R)` | Discrete LQR from state matrices |
| `discrete_lqr` |  `dlqr(A, B, Q, R, N)` | Discrete LQR with cross-term cost |
| `discrete_lqr_from_continuous` |  `lqrd(A, B, Q, R, Ts)` | Discrete LQR from continuous plant |
| `lqr_with_integral` |  `lqi(sys, Q_aug, R)` | LQI with integral action for tracking |
| `lqg_regulator` |  `lqg(sys, Q_lqr, R_lqr, Q_kf, R_kf)` | LQG regulator (LQR + Kalman) |
| `lqg_servo` |  `lqgtrack(sys, Q_aug, R, Q_kf, R_kf)` | LQG servo with integral action |
| `lqg_from_parts` |  `lqgreg(kalman_result, lqr_result)` | Combine separate designs |
| `kalman` |  `kalman(sys, Q, R)` | Steady-state LTI Kalman filter design |

All functions available in both `design::` and `online::` namespaces.

### Additional Capabilities

- **Frequency-Domain Analysis**: Bode plots, gain/phase margins, bandwidth, DC gain, controllability/observability matrices and rank tests
- **PID Design**: Ziegler-Nichols (ultimate gain & step response), Cohen-Coon, Lambda tuning, SIMC, Tyreus-Luyben, bandwidth-based, and pole placement methods
- **PID Controller**: Runtime `PIDController` class with anti-windup, output/integrator saturation, and configurable gains
- **Lead-Lag Compensators**: Phase lead/lag design from crossover specs, with `LeadLagController` runtime class (Tustin-discretized IIR)
- **Proportional-Resonant Control**: Ideal and non-ideal PR for AC reference tracking, multi-harmonic compensation, frequency adaptation
- **Sliding Mode Control**: Configurable sliding surface, switching gain, and plant gain for robust nonlinear control
- **Active Disturbance Rejection**: ESO-based disturbance estimation and PD feedback with pole placement
- **Middlebrook Impedance Analysis**: Input/output impedance stability via gain/phase margin criteria
- **Digital Filters**: LowPass, SOGI, and notch filters with compile-time design
- **IEC 61131-3 Function Blocks**: PLC-standard bistables, edge detectors, timers, and counters
- **Motor Control**: Clarke/Park transforms and Field-Oriented Control (FOC)
- **Sensor Fusion**: ESKF, EKF, Madgwick, and Mahony orientation estimation
- **ODE Solvers**: Fixed-step (RK4) and adaptive (RK45) with event detection and zero-crossing
- **Simulation**: Closed-loop simulation of nonlinear plants with linear/nonlinear controllers
- **Visualization**: CSV/Gnuplot export and interactive HTML plots via plotlypp
- **Extended Math**: Comprehensive constexpr math library (exp, log, trig, floor, ceil)
- **Utility Wrappers**: `c2d()`, `blkdiag()`, `diag()`

### State-Space System Interconnections

Interconnect systems using intuitive operators or explicit functions:

| Operation | Operator | Function | Description |
|-----------|----------|----------|-------------|
| Series | `sys1 * sys2` | `series(sys1, sys2)` | Cascade: sys1 → sys2 |
| Parallel | `sys1 + sys2` | `parallel(sys1, sys2)` | Summing: outputs add |
| Subtract | `sys1 - sys2` | `subtract(sys1, sys2)` | Differencing: outputs subtract |
| Feedback | `sys1 / sys2` | `feedback(sys1, sys2)` | Negative feedback loop |

All interconnections preserve noise matrices and are fully `constexpr`.

### Linear Algebra & Rotation Utilities

**Compile-time Matrix Operations:**

- `Matrix<N, M, T>` with arithmetic, transpose, trace, determinant, inversion
- Block views via `.block<Rows, Cols>(r, c)` for non-owning sub-matrices
- Vector types `ColVec<N>` and `RowVec<N>` with dot/cross products, norms
- Optional returns for singular cases (inversion, axis-angle)

**Rotations:**

- `DCM<T>` (3×3 Direction Cosine Matrix) with axis-specific constructors
- `Quaternion<T>` with SLERP, Hamilton product, axis-angle conversion
- `Euler<T, Order>` for ZYX (yaw-pitch-roll) and XYZ (roll-pitch-yaw) orders
- Gimbal-lock aware conversions between representations
- `Vec3` operations for 3D vectors (cross product, rotation)

### Result Types with Type Conversion

```cpp
// Design in double precision at compile time
constexpr auto result_d = design::dlqr(Ad, Bd, Q, R);

// Convert to float for embedded use
auto result_f = result_d.as<float>();
```

Result types: `LQRResult`, `LQIResult`, `KalmanResult`, `LQGResult`, `LQGIResult`

### Controller Classes

| Class | Description |
|-------|-------------|
| `LQR<NX, NU, T>` | State feedback: `u = -K*x` |
| `LQI<NX, NU, NY, T>` | Integral action: `u = -Kx*x - Ki*∫(r-y)` |
| `LQG<NX, NU, NY, T>` | LQR + Kalman filter for output feedback |
| `LQGI<NX, NU, NY, T>` | LQG + integral action for servo tracking |

### Discretization Methods

```cpp
// Zero-Order Hold (default)
auto sys_d = discretize(sys_c, Ts, DiscretizationMethod::ZOH);

// Tustin (bilinear transform)
auto sys_d = discretize(sys_c, Ts, DiscretizationMethod::Tustin);
```

### DARE with Cross-Term Support

The Discrete Algebraic Riccati Equation solver supports cross-term costs:

```
J = Σ [x'Qx + u'Ru + 2x'Nu]
```

```cpp
// With cross-term N
auto result = online::dlqr(A, B, Q, R, N);
```

### IEC 61131-3 Function Blocks

Complete implementation of PLC-standard function blocks for industrial control:

```cpp
#include "iec61131.hpp"

using namespace wetmelon::control::plc;

// Bistables
SR sr;  // Set-dominant latch
RS rs;  // Reset-dominant latch

// Edge detectors
R_TRIG rtrig;  // Rising edge detector
F_TRIG ftrig;  // Falling edge detector

// Timers
TON ton;  // On-delay timer
TOF tof;  // Off-delay timer
TP  tp;   // Pulse timer

// Counters
CTU ctu;  // Up counter
CTD ctd;  // Down counter
CTUD ctud; // Up-down counter

// Usage
bool edge = rtrig(signal);  // Detects rising edges
bool pulse = tp.PT(1.0f);   // 1 second pulse
```

### Digital Filters

Compile-time and runtime filter design:

```cpp
#include "filters.hpp"

// Design-time filter design
constexpr auto lpf = design::lowpass(100.0, 10.0);  // 10Hz cutoff at 100Hz sample rate

// Runtime filter usage
LowPass lpf_rt(lpf);
float filtered = lpf_rt.update(raw_signal);
```

Available filters: LowPass, SOGI (Second-Order Generalized Integrator), notch filters.

### Motor Control Transforms

Three-phase motor control utilities:

```cpp
#include "motor_control.hpp"

// Clarke transform (ABC → αβ)
auto clarke = clarke_transform(ia, ib, ic);

// Park transform (αβ → dq)
auto park = park_transform(clarke.alpha, clarke.beta, theta);

// Field-Oriented Control
FOC foc;
auto voltages = foc.control(iq_ref, id_ref, iq_meas, id_meas, omega);
```

### Sensor Fusion

Orientation estimation algorithms:

```cpp
#include "sensor_fusion.hpp"

// Error-State Kalman Filter
ESKF eskff;
eskf.predict(dt);
eskf.update(gyro, accel, mag);

// Madgwick gradient descent filter
Madgwick madgwick;
Quaternion q = madgwick.update(gyro, accel, mag, dt);
```

### Stability Analysis

```cpp
bool stable = stability::is_stable_discrete(A_cl);
double margin = stability::stability_margin_discrete(A_cl);
```

## Design Workflow

### Compile-Time Design for Embedded Systems

```cpp
// 1. Define continuous plant
constexpr StateSpace sys_c{
    Matrix<3, 3>{{-20.0, -500.0, 0.0},
                 {1.0, 0.0, -1.0},
                 {250.0, 0.0, -5.0}},
    Matrix<3, 1>{{10.0}, {0.0}, {0.0}},
    Matrix<3, 3>::identity()
};

// 2. Design at compile time
constexpr auto ctrl = design::discrete_lqr_from_continuous(
    sys_c, Q, R, 0.01  // 100 Hz
);

// 3. Verify properties at compile time
static_assert(ctrl.success);
static_assert(ctrl.K(0, 0) > 0.0);

// 4. Use in control loop (zero runtime overhead)
void control_loop() {
    u = -ctrl.K * x;  // Gains are compile-time constants
}
```

### Runtime Design for Adaptive Control

```cpp
// Design can adapt to runtime parameters
void reconfigure(double new_weight) {
    Matrix<2, 2> Q_new = Matrix<2, 2>::identity() * new_weight;
    auto new_ctrl = online::dlqr(sys.A, sys.B, Q_new, R);
    if (new_ctrl.success) {
        controller = LQR(new_ctrl);
    }
}
```

### Interconnecting Systems

```cpp
// Series connection: cascade G1 then G2
constexpr auto G = series(G1, G2);  // or: G1 * G2

// Parallel: sensor fusion
constexpr auto fused = sensor1 + sensor2;

// Differencing: error signal
constexpr auto error = reference - measurement;

// Feedback: closed-loop with observer
constexpr auto closed_loop = plant / observer;

// Complex networks at compile time
constexpr auto system = (plant * controller) / observer + disturbance_model;
```

## API Reference

### Result Structures

#### `LQRResult<NX, NU, T>`

- `.K` — Feedback gain (NU × NX)
- `.S` — DARE solution / cost-to-go matrix (NX × NX)
- `.success` — Convergence flag
- `.as<U>()` — Convert to type U

#### `LQIResult<NX, NU, NY, T>`

- `.Kx` — State gain (NU × NX)
- `.Ki` — Integral gain (NU × NY)
- `.S` — Augmented DARE solution
- `.success` — Convergence flag

#### `KalmanResult<NX, NY, T>`

- `.L` — Kalman gain (NX × NY)
- `.P` — Steady-state covariance (NX × NX)
- `.success` — Convergence flag

#### `LQGResult<NX, NU, NY, T>`

- `.lqr` — LQRResult
- `.kalman` — KalmanResult
- `.sys` — System copy
- `.success` — Combined success flag

#### `LQGIResult<NX, NU, NY, T>`

- `.lqi` — LQIResult
- `.kalman` — KalmanResult
- `.sys` — System copy
- `.success` — Combined success flag

## Testing

Tests are organized by functionality:

| File | Purpose |
|------|---------|
| `test_matrix.cpp` | Matrix arithmetic, block views, operations |
| `test_vector.cpp` | Vector types, dot/cross products, norms |
| `test_views.cpp` | Block, diagonal, triangular views |
| `test_cholesky.cpp` | Cholesky decomposition |
| `test_matrix_functions.cpp` | Matrix exp, sin, cos, sqrt, pow |
| `test_rotation.cpp` | DCM, Quaternion, Euler angle conversions |
| `test_quaternion.cpp` | Quaternion SLERP, integration, axis-angle |
| `test_state_space.cpp` | System interconnections (series, parallel, feedback, subtract) |
| `test_transfer_function.cpp` | Transfer function representations |
| `test_discretization.cpp` | ZOH, Tustin discretization methods |
| `test_analysis.cpp` | Bode, margins, controllability, observability |
| `test_analysis_extended.cpp` | Extended frequency-domain analysis |
| `test_middlebrook.cpp` | Middlebrook impedance stability |
| `test_api.cpp` | Design API (dlqr, lqi, lqg) |
| `test_design.cpp` | Compile-time `design::` verification |
| `test_online.cpp` | Runtime `online::` controller construction |
| `test_lqi.cpp` | LQI integral action |
| `test_kalman.cpp` | Kalman filter design |
| `test_eigen.cpp` | Eigenvalue/eigenvector computation |
| `test_eskf.cpp` | Error-state Kalman filter |
| `test_pid_design.cpp` | PID tuning methods |
| `test_lead_lag.cpp` | Lead-lag compensator design |
| `test_pr.cpp` | Proportional-resonant controller |
| `test_smc.cpp` | Sliding mode control |
| `test_adrc.cpp` | Active disturbance rejection |
| `test_filters.cpp` | Digital filters (LowPass, SOGI, notch) |
| `test_iec61131.cpp` | IEC 61131-3 PLC function blocks |
| `test_motor_control.cpp` | Clarke/Park transforms, FOC |
| `test_sensor_fusion.cpp` | ESKF, Madgwick, Mahony filters |
| `test_solver.cpp` | ODE solvers (RK4, RK45, events) |
| `test_simulate.cpp` | Closed-loop simulation |
| `test_matlab.cpp` | Utility wrappers (c2d, blkdiag, diag) |
| `test_constexpr_math.cpp` | Compile-time math functions |
| `test_4x4_systems.cpp` | Larger 4×4 system tests |
| `test_integration.cpp` | Motor-coupling-mass system example |

## Header Organization

| Header | Contents |
|--------|----------|
| `matrix.hpp` | Core `Matrix<N,M,T>` type with block views, `Vec/ColVec/RowVec` |
| `matrix_functions.hpp` | Matrix exponential, norms, sin, cos, sqrt, pow |
| `cholesky.hpp` | Cholesky decomposition and triangular solves |
| `state_space.hpp` | `StateSpace<NX,NU,NY,T>`, system interconnections |
| `transfer_function.hpp` | Transfer function representations |
| `discretization.hpp` | `discretize()`: Forward Euler, ZOH, Tustin methods |
| `ricatti.hpp` | `dare()` (SDA), `care()` solvers |
| `analysis.hpp` | Bode plots, margins, bandwidth, DC gain, controllability, observability |
| `stability.hpp` | Eigenvalue-based stability checking |
| `eigen.hpp` | Eigenvalue/eigenvector computation (QR iteration) |
| `lqr.hpp` | `LQR` controller class, `design::dlqr`, `design::lqrd` |
| `lqi.hpp` | `LQI` controller class with integral action |
| `lqg.hpp` | `LQG` regulator (LQR + Kalman) |
| `lqgi.hpp` | `LQGI` servo (LQG + integral action) |
| `kalman.hpp` | Steady-state Kalman filter design |
| `ekf.hpp` | Extended Kalman Filter for nonlinear systems |
| `eskf.hpp` | Error-State Kalman Filter for attitude estimation |
| `pid.hpp` | `PIDController` runtime class with anti-windup |
| `pid_design.hpp` | Modelless PID tuning: Z-N, Cohen-Coon, Lambda, SIMC, pole placement |
| `lead_lag.hpp` | Lead-lag compensator design and `LeadLagController` class |
| `pr.hpp` | Proportional-Resonant controller, multi-harmonic compensation |
| `smc.hpp` | Sliding Mode Control with configurable surface and switching gain |
| `adrc.hpp` | Active Disturbance Rejection Control with ESO |
| `filters.hpp` | Digital filters: LowPass, SOGI, notch |
| `iec61131.hpp` | IEC 61131-3 function blocks: SR, RS, R_TRIG, F_TRIG, timers, counters |
| `motor_control.hpp` | Clarke/Park transforms, Field-Oriented Control (FOC) |
| `rotation.hpp` | `DCM<T>`, `Quaternion<T>`, `Euler<T, Order>` rotations |
| `sensor_fusion.hpp` | ESKF, Madgwick, Mahony orientation estimation |
| `integrator.hpp` | Low-level integrators: Discrete, Exact (LTI), RK4/RK45 |
| `solver.hpp` | `FixedStepSolver`, `AdaptiveStepSolver` with event detection |
| `simulate.hpp` | Closed-loop nonlinear plant simulation |
| `plot.hpp` | CSV/Gnuplot export, step/impulse response generation |
| `plot_plotly.hpp` | Interactive HTML plotting via plotlypp |
| `matlab.hpp` | Utility wrappers: `c2d()`, `blkdiag()`, `diag()` |
| `constexpr_math.hpp` | Compile-time math: exp, log, sin, cos, tan, sqrt, pow, floor, ceil |
| `constexpr_complex.hpp` | Complex number operations for eigenvalue computations |
| `utility.hpp` | Utility functions: linspace, conversions |

## Design Constraints

- **Fixed-size only**: No dynamic allocation, all sizes known at compile time
- **Stack-allocated**: Suitable for embedded systems without heap
- **Arithmetic types**: Supports `float`, `double`, `wet::complex<float>`, `wet::complex<double>`
- **Constexpr-compatible**: All algorithms work at compile time
- **No Dynamic STL containers**: Uses `std::array` and `std::optional` only

## Numerical Methods

- **Matrix Operations**: Padé approximation exponential, Gauss-Jordan inversion, Cholesky decomposition, QR decomposition
- **DARE**: Structure-preserving Doubling Algorithm (SDA) with quadratic convergence — works for open-loop unstable plants
- **CARE**: Continuous Algebraic Riccati Equation solver
- **Eigenvalues**: QR iteration for stability analysis
- **ODE Integration**: RK4 (fixed-step), RK45 (adaptive with error control), event detection with zero-crossing
- **Rotations**: DCM/quaternion/Euler conversions with gimbal-lock handling, SLERP interpolation
- **System Interconnections**: Block-based state augmentation with noise matrix propagation
- **Discretization**: Forward Euler, ZOH (matrix exponential), Tustin (bilinear transform)
- **Filters**: Bilinear transform design, SOGI implementation, PR resonant controllers
- **IEC 61131-3**: Standard PLC timing and counting logic
- **Sensor Fusion**: Extended Kalman filtering, gradient descent orientation estimation

## Examples

| Example | Description |
|---------|-------------|
| `example_servo_drive.cpp` | PMSM servo drive with P-PI-PI cascade (position/speed/current), flexible coupling, load disturbance |
| `servo_drive/` | Interactive real-time GUI simulator (C++ DLL + Python/dearpygui) with live parameter tuning |
| `example_buck_converter.cpp` | DC-DC buck converter LQR control from continuous LC filter dynamics |
| `example_cart_pole.cpp` | Inverted pendulum-on-cart LQR stabilization of nonlinear Lagrangian system |
| `example_lqr_pendulum.cpp` | LQR pendulum swing-up and balance |
| `example_pendulum_sim.cpp` | Pendulum simulation with adaptive ODE solver |
| `example_lpf.cpp` | Low-pass filter design and application |
| `example_eskf_arduino.cpp` | ESKF attitude estimation targeting Arduino |

## Formula References

- Franklin, Powell & Workman: "Digital Control of Dynamic Systems"
- Anderson & Moore: "Optimal Control: Linear Quadratic Methods"
- Simon: "Optimal State Estimation" (Kalman filtering)

## Build Requirements

- **C++ Standard**: C++20 (consteval, constexpr, concepts)
- **Compilers**: GCC 10+, Clang 12+, MSVC 2022+
