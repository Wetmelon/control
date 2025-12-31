#pragma once

#include <stdexcept>

#include "LTI.hpp"         // IWYU pragma: keep
#include "format.hpp"      // IWYU pragma: keep
#include "integrator.hpp"  // IWYU pragma: keep
#include "nonlinear.hpp"   // IWYU pragma: keep
#include "observer.hpp"    // IWYU pragma: keep
#include "solver.hpp"      // IWYU pragma: keep
#include "ss.hpp"          // IWYU pragma: keep
#include "tf.hpp"          // IWYU pragma: keep
#include "types.hpp"       // IWYU pragma: keep
#include "utility.hpp"     // IWYU pragma: keep
#include "zpk.hpp"         // IWYU pragma: keep

// Free functions for creating LTI systems and performing operations
namespace control {

template <class T>
concept SSConvertible = requires(const T& t) { { t.toStateSpace() }; };

template <class T>
concept TFConvertible = requires(const T& t) { { t.toTransferFunction() }; };

template <class T>
concept ZPKConvertible = requires(const T& t) { { t.toZeroPoleGain() }; };

template <SSConvertible T>
StateSpace ss(const T& sys) {
    return sys.toStateSpace();
}

template <TFConvertible T>
TransferFunction tf(const T& sys) {
    return sys.toTransferFunction();
}

// Handle MIMO case with specified input/output indices
inline TransferFunction tf(const StateSpace& sys, int output_idx, int input_idx) {
    return sys.toTransferFunction(output_idx, input_idx);
}

inline TransferFunction tf(std::vector<double> num, std::vector<double> den, std::optional<double> Ts = std::nullopt) {
    return TransferFunction{std::move(num), std::move(den), Ts};
}

inline StateSpace tf2ss(std::vector<double> num, std::vector<double> den, std::optional<double> Ts = std::nullopt) {
    return TransferFunction{std::move(num), std::move(den), Ts}.toStateSpace();
}

inline TransferFunction ss2tf(Matrix A, Matrix B, Matrix C, Matrix D, std::optional<double> Ts = std::nullopt) {
    return StateSpace{std::move(A), std::move(B), std::move(C), std::move(D), Ts}.toTransferFunction();
}

template <ZPKConvertible T>
ZeroPoleGain zpk(const T& sys) {
    return sys.toZeroPoleGain();
}

inline ZeroPoleGain zpk(const StateSpace& sys, int output_idx, int input_idx) {
    return sys.toTransferFunction(output_idx, input_idx).toZeroPoleGain();
}

inline ZeroPoleGain zpk(const std::vector<Zero>& zeros,
                        const std::vector<Pole>& poles,
                        double                   gain,
                        std::optional<double>    Ts = std::nullopt) {
    return ZeroPoleGain{zeros, poles, gain, Ts};
}

/**
 * @brief Convert a continuous-time LTI system to discrete-time using specified method.
 *
 * @param sys           Continuous-time LTI system
 * @param Ts            Sampling time
 * @param method        Discretization method (default: ZOH)
 * @param prewarp       Optional pre-warp frequency for Tustin method
 *
 * @return T   Discrete-time system of the same type as input
 */
StateSpace c2d(const StateSpace& sys, double Ts, DiscretizationMethod method = DiscretizationMethod::ZOH, std::optional<double> prewarp = std::nullopt);

template <SSConvertible T>
StateSpace c2d(const T& sys, double Ts, DiscretizationMethod method = DiscretizationMethod::ZOH, std::optional<double> prewarp = std::nullopt) {
    return sys.toStateSpace().discretize(Ts, method, prewarp);
}

// Continuous to discrete conversion for raw matrices (A, B)
std::pair<Matrix, Matrix> c2d(const Matrix& A, const Matrix& B, double Ts, DiscretizationMethod method = DiscretizationMethod::ZOH, std::optional<double> prewarp = std::nullopt);

// ============================================================================
// LTI System Arithmetic Operations
// ============================================================================
// These operators enable combining LTI systems to create complex control
// systems from simple building blocks (controllers, plants, sensors).
//
// Usage Examples:
//   auto open_loop    = controller * plant;          // Series connection
//   auto parallel_sys = sys1 + sys2;                 // Parallel (sum)
//   auto error_sys    = reference - measurement;     // Parallel (difference)
//   auto closed_loop  = feedback(fwd_path, fb_path); // Negative feedback
//   auto closed_loop  = fwd_path / fb_path;          // Negative feedback (same as above)
//
// Control System Construction:
//   1. Create individual components (Controller C, Plant G, Sensor H)
//   2. Combine them: T = feedback(C * G, H)
//   3. This creates closed-loop: T(s) = C*G / (1 + C*G*H)
// ============================================================================

// LTI operations on mixed types always return StateSpace representation
template <SSConvertible A, SSConvertible B>
StateSpace series(const A& a, const B& b) {
    return series(a.toStateSpace(), b.toStateSpace());
}

template <SSConvertible A, SSConvertible B>
StateSpace parallel(const A& a, const B& b) {
    return parallel(a.toStateSpace(), b.toStateSpace());
}

template <SSConvertible A, SSConvertible B>
StateSpace feedback(const A& a, const B& b, int sign = -1) {
    return feedback(a.toStateSpace(), b.toStateSpace(), sign);
}

// Convenience overload: feedback(sys) -> feedback(sys, unity_sensor)
template <SSConvertible A>
StateSpace feedback(const A& a, int sign = -1) {
    auto fs = a.toStateSpace();
    // Forward system dimensions: p outputs, m inputs
    const auto p = static_cast<int>(fs.C.rows());
    const auto m = static_cast<int>(fs.B.cols());
    if (m != p) {
        throw std::invalid_argument("feedback(sys): unity feedback requires system with equal number of inputs and outputs");
    }

    StateSpace unity{Matrix::Zero(0, 0), Matrix::Zero(0, m), Matrix::Zero(p, 0), Matrix::Identity(p, m)};
    return feedback(fs, unity, sign);
}

template <SSConvertible A, SSConvertible B>
StateSpace operator*(const A& a, const B& b) {
    return series(a.toStateSpace(), b.toStateSpace());
}

template <SSConvertible A, SSConvertible B>
StateSpace operator+(const A& a, const B& b) {
    return parallel(a.toStateSpace(), b.toStateSpace());
}

template <SSConvertible A, SSConvertible B>
StateSpace operator-(const A& a, const B& b) {
    StateSpace neg_b = b.toStateSpace();
    neg_b.C          = -neg_b.C;
    neg_b.D          = -neg_b.D;

    return parallel(a.toStateSpace(), neg_b);
}

template <SSConvertible A, SSConvertible B>
StateSpace operator/(const A& a, const B& b) {
    return feedback(a.toStateSpace(), b.toStateSpace(), -1);
}

// Type-preserving series connections
StateSpace       series(const StateSpace& sys1, const StateSpace& sys2);
TransferFunction series(const TransferFunction& sys1, const TransferFunction& sys2);
ZeroPoleGain     series(const ZeroPoleGain& sys1, const ZeroPoleGain& sys2);

// Type-preserving parallel connections
StateSpace       parallel(const StateSpace& sys1, const StateSpace& sys2);
TransferFunction parallel(const TransferFunction& sys1, const TransferFunction& sys2);
ZeroPoleGain     parallel(const ZeroPoleGain& sys1, const ZeroPoleGain& sys2);

// Type-preserving feedback connections
StateSpace       feedback(const StateSpace& sys_forward, const StateSpace& sys_feedback, int sign = -1);
TransferFunction feedback(const TransferFunction& sys_forward, const TransferFunction& sys_feedback, int sign = -1);
ZeroPoleGain     feedback(const ZeroPoleGain& sys_forward, const ZeroPoleGain& sys_feedback, int sign = -1);

// Pade approximation for time delays
StateSpace       pade(const StateSpace& sys, double delay, int order = 3);
TransferFunction pade(const TransferFunction& tf, double delay, int order = 3);
ZeroPoleGain     pade(const ZeroPoleGain& zpk_sys, double delay, int order = 3);

// Pade approximation for time delays
StateSpace       delay(const StateSpace& sys, double delay, int order = 3);
TransferFunction delay(const TransferFunction& tf, double delay, int order = 3);
ZeroPoleGain     delay(const ZeroPoleGain& zpk_sys, double delay, int order = 3);

// Type-preserving series connection operators
StateSpace       operator*(const StateSpace& sys1, const StateSpace& sys2);
TransferFunction operator*(const TransferFunction& sys1, const TransferFunction& sys2);
ZeroPoleGain     operator*(const ZeroPoleGain& sys1, const ZeroPoleGain& sys2);

// Type-preserving parallel connection operators
StateSpace       operator+(const StateSpace& sys1, const StateSpace& sys2);
TransferFunction operator+(const TransferFunction& sys1, const TransferFunction& sys2);
ZeroPoleGain     operator+(const ZeroPoleGain& sys1, const ZeroPoleGain& sys2);

// Type-preserving difference operators
StateSpace       operator-(const StateSpace& sys1, const StateSpace& sys2);
TransferFunction operator-(const TransferFunction& sys1, const TransferFunction& sys2);
ZeroPoleGain     operator-(const ZeroPoleGain& sys1, const ZeroPoleGain& sys2);

// Feedback connection operators: sys_forward / sys_feedback
StateSpace       operator/(const StateSpace& sys_forward, const StateSpace& sys_feedback);
TransferFunction operator/(const TransferFunction& sys_forward, const TransferFunction& sys_feedback);
ZeroPoleGain     operator/(const ZeroPoleGain& sys_forward, const ZeroPoleGain& sys_feedback);

// Free function implementations for LTI operations
bool is_stable(const StateSpace& sys);
bool is_stable(const ZeroPoleGain& sys);

std::vector<Pole> poles(const StateSpace& sys);
std::vector<Zero> zeros(const StateSpace& sys);

StepResponse    step(const StateSpace& sys, double tStart = 0.0, double tEnd = 10.0, ColVec uStep = ColVec::Ones(1));
ImpulseResponse impulse(const StateSpace& sys, double tStart = 0.0, double tEnd = 10.0);

BodeResponse      bode(const StateSpace& sys, double fStart = 0.1, double fEnd = 1.0e4, size_t maxPoints = 500);
NyquistResponse   nyquist(const StateSpace& sys, double fStart = 0.1, double fEnd = 1.0e4, size_t maxPoints = 500);
RootLocusResponse rlocus(const StateSpace& sys, double kMin = 0.0, double kMax = 100.0, size_t numPoints = 500);
RootLocusResponse rlocus(const TransferFunction& sys, double kMin = 0.0, double kMax = 100.0, size_t numPoints = 500);

MarginInfo        margin(const StateSpace& sys);
FrequencyResponse freqresp(const StateSpace& sys, const std::vector<double>& frequencies);

DampingInfo damp(const StateSpace& sys);
StepInfo    stepinfo(const StateSpace& sys);

ObservabilityInfo   observability(const StateSpace& sys);
ControllabilityInfo controllability(const StateSpace& sys);

Matrix     gramian(const StateSpace& sys, GramianType type);
StateSpace minreal(const StateSpace& sys, double tol = 1e-9);
StateSpace balred(const StateSpace& sys, size_t r);

// Matrix computations
Matrix ctrb(const StateSpace& sys);
Matrix ctrb(const Matrix& A, const Matrix& B);
Matrix obsv(const StateSpace& sys);
Matrix obsv(const Matrix& C, const Matrix& A);
double norm(const Matrix& A, const Matrix& B, const Matrix& C, const Matrix& D, const std::string& type = "inf");
double norm(const StateSpace& sys, const std::string& type = "inf");

// Solve continuous Lyapunov equation A*X + X*A^T + Q = 0 for X
// Primary solver: Schur-based Bartels–Stewart (complex Schur variant)
// Fallback: numerical integral approximation for large/stiff problems
Matrix lyap(const Matrix& A, const Matrix& Q);
Matrix dlyap(const Matrix& A, const Matrix& Q);

// Solve continuous/discrete Algebraic Riccati Equations
Matrix care(const Matrix& A, const Matrix& B, const Matrix& Q, const Matrix& R);
Matrix dare(const Matrix& A, const Matrix& B, const Matrix& Q, const Matrix& R);

// Linear Quadratic Regulator
struct LQRResult {
    Matrix            K;  // State-feedback gain matrix
    Matrix            S;  // Riccati solution matrix
    std::vector<Pole> P;  // Closed-loop eigenvalues
};

// Continuous-time Linear Quadratic Regulator for continuous system x_dot = A*x + B*u
LQRResult lqr(const Matrix& A, const Matrix& B, const Matrix& Q, const Matrix& R, const Matrix& N = {});
LQRResult lqr(const StateSpace& sys, const Matrix& Q, const Matrix& R, const Matrix& N = {});

// Continuous-time Linear Quadratic Regulator for continuous system x_hat = f(x, u), linearized at (x0, u0)
LQRResult lqr(const NonlinearSystem& sys, const ColVec& x0, const ColVec& u0,
              const Matrix& Q, const Matrix& R, const Matrix& N = {});

// Linear Quadratic Integrator for continuous system with integral action
LQRResult lqi(const StateSpace& sys, const Matrix& Q, const Matrix& R, const Matrix& N = {});

// Discrete-time Linear Quadratic Regulator for discrete system x[k+1] = A*x[k] + B*u[k]
LQRResult dlqr(const Matrix& A, const Matrix& B, const Matrix& Q, const Matrix& R, const Matrix& N = {});

// Discrete-time Linear Quadratic Regulator for continuous system discretized with Ts
LQRResult lqrd(const Matrix& A, const Matrix& B, const Matrix& Q, const Matrix& R, double Ts, const Matrix& N = {});

// LQR with output weighting (Q = C^T * Qy * C)
LQRResult lqry(const StateSpace& sys, const Matrix& Qy, const Matrix& Ru, const Matrix& N = {});

// Linear Quadratic Gaussian (LQG) Controller
struct LQGResult {
    Matrix       K;       // LQR gain matrix
    KalmanFilter filter;  // Kalman filter for state estimation
    Matrix       S;       // LQR Riccati solution
    Matrix       P;       // Kalman error covariance
};

// Continuous-time LQG for continuous system
LQGResult lqg(const Matrix& A, const Matrix& B, const Matrix& C, const Matrix& Q, const Matrix& R, const Matrix& Qn, const Matrix& Rn, const Matrix& N = {}, const Matrix& Nn = {});
LQGResult lqg(const StateSpace& sys, const Matrix& Q, const Matrix& R, const Matrix& Qn, const Matrix& Rn, const Matrix& N = {}, const Matrix& Nn = {});
LQGResult lqg(const NonlinearSystem& sys, const ColVec& x0, const ColVec& u0,
              const Matrix& Q, const Matrix& R, const Matrix& Qn, const Matrix& Rn,
              const Matrix& N = {}, const Matrix& Nn = {});

// Discrete-time LQG for discrete system
LQGResult dlqg(const StateSpace& sys, const Matrix& Q, const Matrix& R, const Matrix& Qn, const Matrix& Rn, const Matrix& N = {}, const Matrix& Nn = {});

// LQG for continuous system discretized with Ts
LQGResult lqgd(const StateSpace& sys, const Matrix& Q, const Matrix& R, const Matrix& Qn, const Matrix& Rn, double Ts, const Matrix& N = {}, const Matrix& Nn = {});

// LQG servo controller for tracking
LQGResult lqgtrack(const StateSpace& sys, const Matrix& Q, const Matrix& R, const Matrix& Qn, const Matrix& Rn, const Matrix& N = {}, const Matrix& Nn = {});

// Pole placement design
Matrix place(const Matrix& A, const Matrix& B, const std::vector<Pole>& poles);

// Regulator design - combines state feedback with observer
StateSpace reg(const StateSpace& sys, const Matrix& K, const Matrix& L);

struct KalmanResult {
    KalmanFilter          filter;  // Kalman filter object
    Matrix                L;       // Kalman gain
    Matrix                P;       // Error covariance matrix
    std::optional<Matrix> Mx;      // Innovation gain matrix (discrete systems only)
    std::optional<Matrix> Z;       // Error covariance matrix for time update (discrete systems only)
    std::optional<Matrix> My;      // Additional innovation gain matrix (discrete systems only)
};

// Kalman Filter synthesis for continuous system
KalmanResult kalman(const StateSpace& sys, const Matrix& Qn, const Matrix& Rn, const Matrix& N = {});

// Design discrete Kalman estimator for continuous plant
KalmanResult kalmd(const StateSpace& sys, const Matrix& Qn, const Matrix& Rn, double Ts, const Matrix& N = {});

struct PredictResult {
    ColVec x_pred;  // Predicted state
    Matrix P_pred;  // Predicted covariance
};

ColVec        predict(LuenbergerObserver& obs, const ColVec& u, double dt = 0.0);
PredictResult predict(KalmanFilter& kf, const ColVec& u, double dt);
PredictResult predict(ExtendedKalmanFilter& kf, const ColVec& u);

// Controllability and observability matrices
// System norms

// Append state vector to output vector
StateSpace augstate(const StateSpace& sys);

// Augment system with disturbance states for unmeasured disturbances
StateSpace augd(const StateSpace& sys);

// Relative Gain Array (RGA)
Matrix rga(const StateSpace& sys);

// Sampling and holding blocks
StateSpace zoh();
StateSpace foh();

}  // namespace control
