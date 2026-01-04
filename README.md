# Compile-Time Control Design System

## Overview

A header-only C++20 library for designing optimal controllers and observers **at compile time**, with extensive linear algebra and rotation utilities. Features optional MATLAB®-style APIs (`dlqr`, `lqi`, `lqg`, `kalman`) with two evaluation modes:

- **`design::`** — `consteval` functions that **must** execute at compile time
- **`online::`** — `constexpr` functions that **can** execute at runtime

All designs produce fixed-size, stack/statically-allocated artifacts suitable for embedded systems with zero dynamic allocation.

## Quick Start

```cpp
#include "control_design.hpp"
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

| Function | MATLAB® compatible | Description |
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

Note: MATLAB® compatible API does not mean it has the full features of MATLAB®.

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
| `test_rotation.cpp` | DCM, Quaternion, Euler angle conversions |
| `test_quaternion.cpp` | Quaternion SLERP, integration, axis-angle |
| `test_state_space.cpp` | System interconnections (series, parallel, feedback, subtract) |
| `test_discretization.cpp` | ZOH, Tustin discretization methods |
| `test_api.cpp` | MATLAB®-style API (dlqr, lqi, lqg, etc.) |
| `test_design.cpp` | Compile-time `design::` verification |
| `test_online.cpp` | Runtime `online::` controller construction |
| `test_integration.cpp` | Motor-coupling-mass system example |
| `test_analysis.cpp` | Stability analysis functions |
| `test_matrix_functions.cpp` | Matrix exp, sin, cos, sqrt, pow |
| `test_constexpr_math.cpp` | Compile-time math functions |
| `test_4x4_systems.cpp` | Larger 4×4 system tests |
| `test_kalman.cpp` | Kalman filter design |
| `test_eigen.cpp` | Eigenvalue/eigenvector computation |
| `test_eskf.cpp` | Error-state Kalman filter |

**Test Results:** 184 test cases, 1000+ assertions, all passing ✓

## Header Organization

| Header | Contents |
|--------|----------|
| `matrix.hpp` | Core `Matrix<N,M,T>` type with block views, `Vec/ColVec/RowVec` |
| `vector.hpp` | Vector operations (dot, cross, norm, normalization) |
| `rotation.hpp` | `DCM<T>`, `Quaternion<T>`, `Euler<T, Order>` types |
| `state_space.hpp` | `StateSpace<NX,NU,NY,T>`, system interconnections (series, parallel, feedback, subtract) |
| `discretization.hpp` | `discretize()`, ZOH/Tustin methods |
| `ricatti.hpp` | `dare()`, `care()` solvers |
| `control_design.hpp` | `design::` and `online::` APIs, result types |
| `lqr.hpp` | `LQR`, `LQI`, `LQG`, `LQGI` controller classes |
| `kalman.hpp` | `KalmanFilter`, `ExtendedKalmanFilter`, `ErrorStateKalmanFilter` classes |
| `constexpr_math.hpp` | Compile-time math (exp, sin, sqrt, pow) |
| `eigen.hpp` | Eigenvalue computation for stability |
| `transfer_function.hpp` | Transfer function representations (placeholder) |

## Design Constraints

- **Fixed-size only**: No dynamic allocation, all sizes known at compile time
- **Stack-allocated**: Suitable for embedded systems without heap
- **Arithmetic types**: Supports `float` and `double`
- **Maximum tested**: 4×4 systems (extensible to larger)
- **Constexpr-compatible**: All algorithms work at compile time
- **No Dynamic STL containers**: Uses `std::array` and `std::span` only

## Numerical Methods

- **Matrix Exponential**: Padé approximation with scaling/squaring
- **DARE/CARE**: Kleinman iteration with Joseph-form covariance update
- **Matrix Inversion**: Gauss-Jordan with partial pivoting (up to 6×6)
- **Eigenvalues**: QR iteration for stability analysis
- **Rotations**: Rotation matrix/quaternion/Euler conversions with gimbal-lock handling
- **System Interconnections**: Block-based state augmentation with noise matrix propagation

## Formula References

- Franklin, Powell & Workman: "Digital Control of Dynamic Systems"
- Anderson & Moore: "Optimal Control: Linear Quadratic Methods"
- Simon: "Optimal State Estimation" (Kalman filtering)

## Build Requirements

- **C++ Standard**: C++20 (consteval, constexpr, concepts)
- **Compilers**: GCC 10+, Clang 12+, MSVC 2022+
