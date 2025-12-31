# Control Library Documentation

## Overview

The Control library is a comprehensive C++ library for control systems analysis and design, built on top of Eigen for matrix operations. It provides classes and functions for representing and manipulating Linear Time-Invariant (LTI) systems, performing system analysis, and implementing observers and controllers.

## Main Header: control.hpp

The `control.hpp` header serves as the main entry point for the library, including all necessary headers and providing free functions for system conversions and operations.

### Includes

```cpp
#include "LTI.hpp"         // Base LTI system classes
#include "format.hpp"      // Formatting utilities
#include "integrator.hpp"  // ODE integration methods
#include "observer.hpp"    // Observer implementations
#include "solver.hpp"      // System solvers
#include "ss.hpp"          // State-space representations
#include "tf.hpp"          // Transfer function representations
#include "types.hpp"       // Type definitions
#include "utility.hpp"     // Utility functions
#include "zpk.hpp"         // Zero-pole-gain representations
```

### Concepts

The library defines several C++20 concepts for type checking:

#### SSConvertible
```cpp
template <class T>
concept SSConvertible = requires(const T& t) { { t.toStateSpace() }; };
```
Types that can be converted to StateSpace representation.

#### TFConvertible
```cpp
template <class T>
concept TFConvertible = requires(const T& t) { { t.toTransferFunction() }; };
```
Types that can be converted to TransferFunction representation.

#### ZPKConvertible
```cpp
template <class T>
concept ZPKConvertible = requires(const T& t) { { t.toZeroPoleGain() }; };
```
Types that can be converted to ZeroPoleGain representation.

### Free Functions

#### System Conversion Functions

##### ss()
```cpp
template <SSConvertible T>
StateSpace ss(const T& sys) {
    return sys.toStateSpace();
}
```
Converts any SSConvertible system to StateSpace representation.

##### tf()
```cpp
template <TFConvertible T>
TransferFunction tf(const T& sys) {
    return sys.toTransferFunction();
}

// MIMO overload
inline TransferFunction tf(const StateSpace& sys, int output_idx, int input_idx) {
    return sys.toTransferFunction(output_idx, input_idx);
}
```
Converts systems to TransferFunction representation. The MIMO overload extracts a specific input-output pair from a StateSpace system.

##### zpk()
```cpp
template <ZPKConvertible T>
ZeroPoleGain zpk(const T& sys) {
    return sys.toZeroPoleGain();
}
```
Converts systems to ZeroPoleGain representation.

##### pade()
```cpp
template <SSConvertible A>
A pade(const A& sys, int order) {
    return A(pade(sys.toStateSpace(), order));
}
```
Applies Pade approximation for time delay to a system. The result type matches the input type.

### System Operations

The library provides extensive operations for combining and analyzing LTI systems.

#### System Interconnection

##### series() / operator*
```cpp
template <SSConvertible A, SSConvertible B>
StateSpace series(const A& a, const B& b) {
    return series(a.toStateSpace(), b.toStateSpace());
}

// Type-preserving versions
StateSpace       series(const StateSpace& sys1, const StateSpace& sys2);
TransferFunction series(const TransferFunction& sys1, const TransferFunction& sys2);
ZeroPoleGain     series(const ZeroPoleGain& sys1, const ZeroPoleGain& sys2);
```
Series connection of two systems (multiplication).

##### parallel() / operator+
```cpp
template <SSConvertible A, SSConvertible B>
StateSpace parallel(const A& a, const B& b) {
    return parallel(a.toStateSpace(), b.toStateSpace());
}

// Type-preserving versions
StateSpace       parallel(const StateSpace& sys1, const StateSpace& sys2);
TransferFunction parallel(const TransferFunction& sys1, const TransferFunction& sys2);
ZeroPoleGain     parallel(const ZeroPoleGain& sys1, const ZeroPoleGain& sys2);
```
Parallel connection of two systems (addition).

##### feedback() / operator/
```cpp
template <SSConvertible A, SSConvertible B>
StateSpace feedback(const A& a, const B& b, int sign = -1) {
    return feedback(a.toStateSpace(), b.toStateSpace(), sign);
}

// Type-preserving versions
StateSpace       feedback(const StateSpace& sys_forward, const StateSpace& sys_feedback, int sign = -1);
TransferFunction feedback(const TransferFunction& sys_forward, const TransferFunction& sys_feedback, int sign = -1);
ZeroPoleGain     feedback(const ZeroPoleGain& sys_forward, const ZeroPoleGain& sys_feedback, int sign = -1);
```
Feedback connection. Default is negative feedback (sign = -1).

##### operator-
```cpp
template <SSConvertible A, SSConvertible B>
StateSpace operator-(const A& a, const B& b) {
    StateSpace neg_b = b.toStateSpace();
    neg_b.C          = -neg_b.C;
    neg_b.D          = -neg_b.D;
    return parallel(a.toStateSpace(), neg_b);
}
```
Difference of two systems.

#### Discretization

##### c2d()
```cpp
StateSpace c2d(const StateSpace& sys, double Ts, DiscretizationMethod method = DiscretizationMethod::ZOH, std::optional<double> prewarp = std::nullopt);

template <SSConvertible T>
StateSpace c2d(const T& sys, double Ts, DiscretizationMethod method = DiscretizationMethod::ZOH, std::optional<double> prewarp = std::nullopt) {
    return sys.toStateSpace().discretize(Ts, method, prewarp);
}
```
Convert continuous-time system to discrete-time.

**Methods:**
- `ZOH`: Zero-order hold
- `Bilinear`/`Tustin`: Bilinear transform with optional pre-warping
- `ForwardEuler`: Forward Euler
- `BackwardEuler`: Backward Euler

#### Time Delays

##### pade() / delay()
```cpp
StateSpace       pade(const StateSpace& sys, double delay, int order = 3);
TransferFunction pade(const TransferFunction& tf, double delay, int order = 3);
ZeroPoleGain     pade(const ZeroPoleGain& zpk_sys, double delay, int order = 3);

// Aliases
StateSpace       delay(const StateSpace& sys, double delay, int order = 3);
TransferFunction delay(const TransferFunction& tf, double delay, int order = 3);
ZeroPoleGain     delay(const ZeroPoleGain& zpk_sys, double delay, int order = 3);
```
Approximate time delays using Pade approximation.

### System Analysis

#### Stability Analysis

##### is_stable()
```cpp
bool is_stable(const StateSpace& sys);
bool is_stable(const ZeroPoleGain& sys);
```
Check if system is stable (all poles in left half-plane for continuous, inside unit circle for discrete).

##### poles() / zeros()
```cpp
std::vector<Pole> poles(const StateSpace& sys);
std::vector<Zero> zeros(const StateSpace& sys);
```
Compute poles and zeros of the system.

#### Time Response

##### step()
```cpp
StepResponse step(const StateSpace& sys, double tStart = 0.0, double tEnd = 10.0, ColVec uStep = ColVec::Ones(1));
```
Compute step response of the system.

##### impulse()
```cpp
ImpulseResponse impulse(const StateSpace& sys, double tStart = 0.0, double tEnd = 10.0);
```
Compute impulse response of the system.

#### Frequency Response

##### bode()
```cpp
BodeResponse bode(const StateSpace& sys, double fStart = 0.1, double fEnd = 1.0e4, size_t maxPoints = 500);
```
Compute Bode plot data (magnitude and phase vs frequency).

##### nyquist()
```cpp
NyquistResponse nyquist(const StateSpace& sys, double fStart = 0.1, double fEnd = 1.0e4, size_t maxPoints = 500);
```
Compute Nyquist plot data.

##### freqresp()
```cpp
FrequencyResponse freqresp(const StateSpace& sys, const std::vector<double>& frequencies);
```
Compute frequency response at specified frequencies.

##### margin()
```cpp
MarginInfo margin(const StateSpace& sys);
```
Compute gain and phase margins.

#### Root Locus

##### rlocus()
```cpp
RootLocusResponse rlocus(const StateSpace& sys, double kMin = 0.0, double kMax = 100.0, size_t numPoints = 500);
RootLocusResponse rlocus(const TransferFunction& sys, double kMin = 0.0, double kMax = 100.0, size_t numPoints = 500);
```
Compute root locus data.

#### System Properties

##### damp()
```cpp
DampingInfo damp(const StateSpace& sys);
```
Compute natural frequencies and damping ratios.

##### stepinfo()
```cpp
StepInfo stepinfo(const StateSpace& sys);
```
Compute step response characteristics (rise time, settling time, overshoot, etc.).

##### observability() / controllability()
```cpp
ObservabilityInfo   observability(const StateSpace& sys);
ControllabilityInfo controllability(const StateSpace& sys);
```
Check observability and controllability properties.

##### gramian()
```cpp
Matrix gramian(const StateSpace& sys, GramianType type);
```
Compute controllability or observability Gramian.

##### ctrb() / obsv()
```cpp
Matrix ctrb(const StateSpace& sys);
Matrix obsv(const StateSpace& sys);
```
Compute controllability and observability matrices.

##### norm()
```cpp
double norm(const StateSpace& sys, const std::string& type = "inf");
```
Compute system norms (H-infinity, H-2, etc.).

#### Model Reduction

##### minreal()
```cpp
StateSpace minreal(const StateSpace& sys, double tol = 1e-9);
```
Minimal realization (remove uncontrollable/unobservable states).

##### balred()
```cpp
StateSpace balred(const StateSpace& sys, size_t r);
```
Balanced model reduction.

### Advanced Functions

#### Lyapunov Equations

##### lyap() / dlyap()
```cpp
Matrix lyap(const Matrix& A, const Matrix& Q);   // Continuous: A*X + X*A^T + Q = 0
Matrix dlyap(const Matrix& A, const Matrix& Q);  // Discrete: A*X*A^T - X + Q = 0
```
Solve Lyapunov equations.

#### Riccati Equations

##### care() / dare()
```cpp
Matrix care(const Matrix& A, const Matrix& B, const Matrix& Q, const Matrix& R);  // Continuous ARE
Matrix dare(const Matrix& A, const Matrix& B, const Matrix& Q, const Matrix& R);  // Discrete ARE
```
Solve Algebraic Riccati Equations.

#### Optimal Control

##### lqr() / dlqr() / lqrd()
```cpp
LQRResult lqr(const Matrix& A, const Matrix& B, const Matrix& Q, const Matrix& R, const Matrix& N = Matrix());   // Continuous LQR
LQRResult dlqr(const Matrix& A, const Matrix& B, const Matrix& Q, const Matrix& R, const Matrix& N = Matrix());  // Discrete LQR
LQRResult lqrd(const Matrix& A, const Matrix& B, const Matrix& Q, const Matrix& R, double Ts, const Matrix& N = Matrix()); // Discretized LQR
```
Linear Quadratic Regulator design.

##### lqi()
```cpp
LQRResult lqi(const StateSpace& sys, const Matrix& Q, const Matrix& R, const Matrix& N = Matrix());
```
Linear Quadratic Integrator (with integral action).

#### Kalman Filter Synthesis

##### kalman()
```cpp
KalmanResult kalman(const StateSpace& sys, const Matrix& Qn, const Matrix& Rn, const Matrix& N = Matrix());
```
Design Kalman filter for optimal state estimation.

##### predict()
```cpp
struct PredictResult {
    ColVec x_pred;  // Predicted state
    Matrix P_pred;  // Predicted covariance
};

ColVec        predict(LuenbergerObserver& obs, const ColVec& u, double dt = 0.0);
PredictResult predict(KalmanFilter& kf, const ColVec& u, double dt);
PredictResult predict(ExtendedKalmanFilter& kf, double t, const ColVec& u, double dt);
```
Convenience functions for observer prediction steps.

### Helper Functions

#### System Creation

##### tf2ss() / ss2tf()
```cpp
inline StateSpace tf2ss(std::vector<double> num, std::vector<double> den, std::optional<double> Ts = std::nullopt) {
    return TransferFunction{std::move(num), std::move(den), Ts}.toStateSpace();
}

inline TransferFunction ss2tf(Matrix A, Matrix B, Matrix C, Matrix D, std::optional<double> Ts = std::nullopt) {
    return StateSpace{std::move(A), std::move(B), std::move(C), std::move(D), Ts}.toTransferFunction();
}
```
Convert between transfer function and state-space representations.

##### tf() / zpk()
```cpp
inline TransferFunction tf(std::vector<double> num, std::vector<double> den, std::optional<double> Ts = std::nullopt) {
    return TransferFunction{std::move(num), std::move(den), Ts};
}

inline ZeroPoleGain zpk(const std::vector<Zero>& zeros,
                        const std::vector<Pole>& poles,
                        double                   gain,
                        std::optional<double>    Ts = std::nullopt) {
    return ZeroPoleGain{zeros, poles, gain, Ts};
}
```
Create systems from coefficients or poles/zeros.

## System Classes

The library provides three main representations for Linear Time-Invariant (LTI) systems, all inheriting from the base `LTI` class:

### StateSpace

The fundamental time-domain representation using state-space matrices.

```cpp
class StateSpace : public LTI {
public:
    Matrix A, B, C, D;  // State-space matrices
    std::optional<double> Ts;  // Sampling time (nullopt for continuous)

    // Constructors
    StateSpace(Matrix A, Matrix B, Matrix C, Matrix D, std::optional<double> Ts = std::nullopt);

    // Conversion methods
    TransferFunction toTransferFunction(int output_idx = 0, int input_idx = 0) const;
    ZeroPoleGain toZeroPoleGain() const;

    // System operations
    StateSpace discretize(double Ts, DiscretizationMethod method = DiscretizationMethod::ZOH) const;
    StateSpace& operator+=(const StateSpace& other);
    StateSpace& operator*=(const StateSpace& other);

    // Analysis methods
    std::vector<Pole> poles() const;
    std::vector<Zero> zeros() const;
    bool isStable() const;
    bool isDiscrete() const { return Ts.has_value(); }
};
```

### TransferFunction

Frequency-domain representation using rational transfer functions.

```cpp
class TransferFunction : public LTI {
public:
    std::vector<double> num, den;  // Numerator and denominator coefficients
    std::optional<double> Ts;      // Sampling time

    // Constructors
    TransferFunction(std::vector<double> num, std::vector<double> den, std::optional<double> Ts = std::nullopt);

    // Conversion methods
    StateSpace toStateSpace() const;
    ZeroPoleGain toZeroPoleGain() const;

    // Operations
    TransferFunction& operator+=(const TransferFunction& other);
    TransferFunction& operator*=(const TransferFunction& other);
};
```

### ZeroPoleGain

Pole-zero representation with DC gain.

```cpp
class ZeroPoleGain : public LTI {
public:
    std::vector<Zero> zeros;
    std::vector<Pole> poles;
    double gain;
    std::optional<double> Ts;

    // Constructors
    ZeroPoleGain(std::vector<Zero> zeros, std::vector<Pole> poles, double gain, std::optional<double> Ts = std::nullopt);

    // Conversion methods
    StateSpace toStateSpace() const;
    TransferFunction toTransferFunction() const;
};
```

## Observers

The library provides standalone observer implementations without forced inheritance:

### LuenbergerObserver

Full-state observer for LTI systems.

```cpp
class LuenbergerObserver {
public:
    LuenbergerObserver(const StateSpace& sys, const Matrix& L);

    // Predict state forward by one step
    ColVec predict(const ColVec& u, double dt = 0.0);
    ColVec update(const ColVec& y);

    ColVec state() const;
    ColVec output() const;

    StateSpace getModel() const { return model_; }

private:
    StateSpace model_;  // State-space model
    ColVec x_hat_;      // State estimate
    ColVec y_;          // Measurement residual
    Matrix L_;          // Observer gain
};
```

### KalmanFilter

Optimal state estimator for linear systems with process and measurement noise.

```cpp
class KalmanFilter {
public:
    KalmanFilter(const StateSpace& sys, const Matrix& Q, const Matrix& R);

    // Predict state forward by one step
    ColVec predict(const ColVec& u, double dt = 0.0);
    ColVec update(const ColVec& z);

    ColVec state() const;
    ColVec output() const;
    ColVec residual() const;

    StateSpace getModel() const { return model_; }

    const ColVec& getStateEstimate() const { return x_hat_; }
    const Matrix& getCovariance() const { return P_; }
    const Matrix& getProcessNoiseCovariance() const { return Q_; }
    const Matrix& getMeasurementNoiseCovariance() const { return R_; }
    const Matrix& getKalmanGain() const { return K_; }

private:
    StateSpace model_;  // State-space model
    ColVec x_hat_;      // State estimate
    ColVec y_;          // Measurement residual
    ColVec S_;          // Measurement residual covariance
    Matrix P_;          // Estimate covariance
    Matrix Q_;          // Process noise covariance
    Matrix R_;          // Measurement noise covariance
    Matrix K_;          // Kalman gain
};
```

### ExtendedKalmanFilter

Nonlinear state estimator using linearized dynamics.

```cpp
class ExtendedKalmanFilter {
public:
    using StateTransitionFcn  = std::function<ColVec(double, const ColVec&, const ColVec&)>;
    using MeasurementFcn      = std::function<ColVec(double, const ColVec&)>;
    using StateJacobian       = std::function<Matrix(double, const ColVec&, const ColVec&)>;
    using ObservationJacobian = std::function<Matrix(double, const ColVec&)>;

    // Compute Jacobians using Finite Differences, default covariances
    ExtendedKalmanFilter(StateTransitionFcn f, MeasurementFcn h, ColVec x0);

    // Automatically compute Jacobians using Finite Differences
    ExtendedKalmanFilter(StateTransitionFcn f, MeasurementFcn h, ColVec x0, Matrix Q, Matrix R);

    // User-provided Jacobians, default covariances
    ExtendedKalmanFilter(StateTransitionFcn f, MeasurementFcn h, StateJacobian F, ObservationJacobian H, ColVec x0);

    // User-provided jacobians
    ExtendedKalmanFilter(StateTransitionFcn f, MeasurementFcn h, StateJacobian F, ObservationJacobian H, ColVec x0, Matrix Q, Matrix R);

    ColVec predict(double t, const ColVec& u);
    ColVec update(double t, const ColVec& z);

    void setStateJacobian(StateJacobian F) { F_ = std::move(F); }
    void setMeasurementJacobian(ObservationJacobian H) { H_ = std::move(H); }

    const ColVec& state() const { return x_; }
    const Matrix& covariance() const { return P_; }

    // Finite Difference jacobian computations
    Matrix computeJacobian(const StateTransitionFcn& f, double t, const ColVec& x, const ColVec& u) const;
    Matrix computeJacobian(const MeasurementFcn& h, double t, const ColVec& x) const;

private:
    StateTransitionFcn f_;
    MeasurementFcn h_;
    StateJacobian F_;
    ObservationJacobian H_;

    ColVec x_;  // State estimate
    ColVec y_;  // Measurement residual
    ColVec K_;  // Kalman gain
    ColVec S_;  // Innovation covariance
    Matrix Q_;  // Process noise covariance
    Matrix R_;  // Measurement noise covariance
    Matrix P_;  // Estimate covariance
};
```

## Solvers and Integrators

### ExactSolver

Analytical solution for LTI systems with constant inputs.

```cpp
class ExactSolver {
public:
    static IntegrationResult solve(const StateSpace& sys, double t0, double dt, const ColVec& x0, const ColVec& u);
};
```

### Integrators

The library provides various ODE integration methods wrapped in a `std::variant`:

```cpp
using Integrator = std::variant<RK4, EulerForward, EulerBackward, Discrete>;
```

#### RK4 (Runge-Kutta 4th Order)

```cpp
struct RK4 {
    IntegrationResult integrate(const std::function<ColVec(double, const ColVec&, const ColVec&)>& f,
                               double t0, double dt, const ColVec& x0, const ColVec& u) const;
};
```

#### Euler Methods

```cpp
struct EulerForward {
    IntegrationResult integrate(/* same signature as RK4 */);
};

struct EulerBackward {
    IntegrationResult integrate(/* same signature as RK4 */);
};
```

#### Discrete

```cpp
struct Discrete {
    IntegrationResult integrate(/* same signature as RK4 */);
};
```

## Utilities

### Types and Constants

```cpp
using Matrix = Eigen::MatrixXd;
using ColVec = Eigen::VectorXd;
using RowVec = Eigen::RowVectorXd;

using Pole = std::complex<double>;
using Zero = std::complex<double>;
```

### Response Types

```cpp
struct StepResponse {
    std::vector<double> t;
    Matrix y;
    ColVec u;
};

struct BodeResponse {
    std::vector<double> freq;
    Matrix mag, phase;
};

struct IntegrationResult {
    ColVec x;      // Final state
    Matrix states; // State trajectory (optional)
    double t;      // Final time
};
```

### Enums

```cpp
enum class DiscretizationMethod { ZOH, Bilinear, ForwardEuler, BackwardEuler };
enum class GramianType { Controllability, Observability };
```

### Mathematical Constants

```cpp
constexpr double PI = 3.14159265358979323846;
```

## Usage Examples

### Basic System Creation and Conversion

```cpp
#include "control.hpp"

// Create a transfer function: G(s) = 1/(s^2 + 2s + 1)
TransferFunction G({1.0}, {1.0, 2.0, 1.0});

// Convert to state space
StateSpace sys = ss(G);

// Create from state-space matrices
StateSpace sys2(Matrix::Identity(2,2), Matrix::Ones(2,1), 
                Matrix::Ones(1,2), Matrix::Zero(1,1));

// Convert back to transfer function
TransferFunction G2 = tf(sys2);

// Create from poles and zeros
ZeroPoleGain zpk_sys({}, {-1.0, -1.0}, 1.0);
```

### System Operations

```cpp
// Create two systems
TransferFunction G1({1.0}, {1.0, 1.0});      // 1/(s+1)
TransferFunction G2({1.0}, {1.0, 0.1});      // 1/(s+0.1)

// Series connection (multiplication)
TransferFunction series_sys = G1 * G2;

// Parallel connection (addition)
TransferFunction parallel_sys = G1 + G2;

// Feedback connection
TransferFunction closed_loop = feedback(G1, G2);  // Negative feedback by default

// Using operators
auto result = G1 / G2;  // Feedback
auto diff = G1 - G2;    // Difference
```

### Discretization

```cpp
// Continuous system
StateSpace continuous_sys(A, B, C, D);

// Discretize with zero-order hold
StateSpace discrete_sys = c2d(continuous_sys, 0.01, DiscretizationMethod::ZOH);

// With pre-warping for bilinear transform
StateSpace discrete_sys2 = c2d(continuous_sys, 0.01, DiscretizationMethod::Bilinear, 10.0);
```

### System Analysis

```cpp
// Check stability
bool stable = is_stable(sys);

// Compute poles and zeros
auto poles = sys.poles();
auto zeros = sys.zeros();

// Frequency response
auto bode_data = bode(sys, 0.1, 1000.0, 100);

// Time response
auto step_resp = step(sys, 0.0, 10.0);
auto impulse_resp = impulse(sys, 0.0, 10.0);

// Root locus
auto rl_data = rlocus(sys, 0.0, 100.0, 100);

// System properties
auto damp_info = damp(sys);
auto step_info = stepinfo(sys);
```

### Observer Design and Simulation

```cpp
// System matrices
Matrix A = Matrix::Zero(2,2), B = Matrix::Ones(2,1), C = Matrix::Ones(1,2);

// Create system
StateSpace sys(A, B, C, Matrix::Zero(1,1));

// Design Luenberger observer
Matrix L = place(sys.A, sys.C, {-2.0, -3.0});  // Place poles at -2, -3
LuenbergerObserver observer(sys, L);

// Simulate
ColVec x_true = ColVec::Zero(2);  // True state
ColVec x_hat = ColVec::Zero(2);   // Initial estimate
// Note: Observer state is initialized in constructor, no setState method

ColVec u = ColVec::Ones(1);  // Input
double dt = 0.01;

// Prediction step
x_hat = observer.predict(u, dt);

// Measurement update
ColVec y = sys.C * x_true;  // Measurement
observer.update(y);

// Get updated state
x_hat = observer.state();
```

### Kalman Filter

```cpp
// System with noise
Matrix Q = 0.1 * Matrix::Identity(2,2);  // Process noise
Matrix R = 0.01 * Matrix::Identity(1,1); // Measurement noise

KalmanFilter kf(sys, Q, R);

// Prediction
ColVec x_pred = kf.predict(u, dt);

// Update with measurement
ColVec y_measured = sys.C * x_true + 0.01 * ColVec::Random(1);  // Noisy measurement
kf.update(y_measured);

// Get estimated state
ColVec x_estimated = kf.state();
```

### Extended Kalman Filter for Nonlinear Systems

```cpp
// Nonlinear system: pendulum
auto f = [](double t, const ColVec& x, const ColVec& u) -> ColVec {
    return ColVec({x(1), -9.81*sin(x(0)) + u(0)});  // [theta_dot, omega_dot]
};

auto F = [](double t, const ColVec& x, const ColVec& u) -> Matrix {  // Jacobian
    Matrix J(2,2);
    J << 0, 1, -9.81*cos(x(0)), 0;
    return J;
};

auto h = [](double t, const ColVec& x) -> ColVec {  // Measurement: position only
    return ColVec({x(0)});
};

auto H = [](double t, const ColVec& x) -> Matrix {  // Measurement Jacobian
    return Matrix::Identity(1,2).row(0);  // d/dx [x(0)] = [1, 0]
};

Matrix Q = 0.01 * Matrix::Identity(2,2);
Matrix R = 0.1 * Matrix::Identity(1,1);
ColVec x0 = ColVec({0.1, 0.0});  // Initial state

ExtendedKalmanFilter ekf(f, h, x0, Q, R);  // Automatic Jacobian computation

// Or with user-provided Jacobians:
// ExtendedKalmanFilter ekf(f, h, F, H, x0, Q, R);

// Prediction
double t = 0.0;
ColVec x_pred = ekf.predict(t, u);

// Update
ColVec y_measured = h(t, x_true) + 0.1 * ColVec::Random(1);
ekf.update(t, y_measured);
```

### Optimal Control Design

```cpp
// LQR design
Matrix Q = Matrix::Identity(2,2);  // State weighting
Matrix R = Matrix::Identity(1,1);  // Input weighting

auto lqr_result = lqr(sys.A, sys.B, Q, R);
Matrix K = lqr_result.K;  // Optimal gain matrix

// Closed-loop system
StateSpace cl_sys = feedback(sys, StateSpace(Matrix::Zero(2,2), Matrix::Zero(2,1), K, Matrix::Zero(1,1)));
```

### Frequency Domain Analysis

```cpp
// Bode plot
auto [freq, mag, phase] = bode(sys, 0.01, 100.0, 1000);

// Nyquist plot
auto nyquist_data = nyquist(sys, 0.01, 100.0, 1000);

// Gain and phase margins
auto margins = margin(sys);

// Frequency response at specific frequencies
std::vector<double> freqs = {0.1, 1.0, 10.0, 100.0};
auto freq_resp = freqresp(sys, freqs);
```

### Model Reduction

```cpp
// Remove uncontrollable/unobservable states
StateSpace reduced_sys = minreal(sys, 1e-9);

// Balanced reduction to 1 state
StateSpace bal_reduced = balred(sys, 1);
```

## Dependencies

- **Eigen**: Matrix operations and linear algebra
- **fmt**: String formatting
- **C++23**: Modern C++ features including concepts and ranges

## Building

The library uses Tup as the build system. Run `make` to build all components and run tests.

## Testing

Comprehensive test suite with doctest framework. Run `make test` to execute all tests.