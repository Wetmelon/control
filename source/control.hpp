#pragma once

#include "LTI.hpp"         // IWYU pragma: keep
#include "format.hpp"      // IWYU pragma: keep
#include "integrator.hpp"  // IWYU pragma: keep
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

template <ZPKConvertible T>
ZeroPoleGain zpk(const T& sys) {
    return sys.toZeroPoleGain();
}

template <SSConvertible A>
A pade(const A& sys, int order) {
    return A(pade(sys.toStateSpace(), order));
}

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
template <SSConvertible T>
StateSpace c2d(const T& sys, double Ts, DiscretizationMethod method = DiscretizationMethod::ZOH, std::optional<double> prewarp = std::nullopt) {
    return sys.toStateSpace().discretize(Ts, method, prewarp);
}

inline StateSpace tf2ss(std::vector<double> num, std::vector<double> den, std::optional<double> Ts = std::nullopt) {
    return TransferFunction{std::move(num), std::move(den), Ts}.toStateSpace();
}

inline TransferFunction ss2tf(Matrix A, Matrix B, Matrix C, Matrix D, std::optional<double> Ts = std::nullopt) {
    return StateSpace{std::move(A), std::move(B), std::move(C), std::move(D), Ts}.toTransferFunction();
}

inline TransferFunction tf(std::vector<double> num, std::vector<double> den, std::optional<double> Ts = std::nullopt) {
    return TransferFunction{std::move(num), std::move(den), Ts};
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

}  // namespace control
