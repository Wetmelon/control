#pragma once

#include <optional>

#include "types.hpp"

namespace control {

// Forward declarations
class StateSpace;
class TransferFunction;
class ZeroPoleGain;

// Data structures for frequency and time responses
struct FrequencyResponse {
    std::vector<std::complex<double>> response;  // Complex frequency response
    std::vector<double>               freq;      // Frequency points in Hz
};

struct BodeResponse {
    std::vector<double> freq;       // Frequency in Hz
    std::vector<double> magnitude;  // Magnitude in dB
    std::vector<double> phase;      // Phase in degrees
};

struct StepResponse {
    std::vector<double> time;
    std::vector<ColVec> output;
};

struct ImpulseResponse {
    std::vector<double> time;
    std::vector<ColVec> output;
};

struct NyquistResponse {
    std::vector<std::complex<double>> response;  // Complex frequency response
    std::vector<double>               freq;      // Frequency points in Hz
};

struct RootLocusResponse {
    std::vector<std::vector<std::complex<double>>> branches;  // Each branch is a vector of pole locations
    std::vector<double>                            gains;     // Corresponding gain values
};

struct ObservabilityInfo {
    size_t rank;
    bool   isObservable;
};

struct ControllabilityInfo {
    size_t rank;
    bool   isControllable;
};

struct MarginInfo {
    double gainMargin;      // Gain margin in dB
    double phaseMargin;     // Phase margin in degrees
    double gainCrossover;   // Gain crossover frequency in Hz
    double phaseCrossover;  // Phase crossover frequency in Hz
};

struct DampingInfo {
    double naturalFrequency;  // Natural frequency (rad/s)
    double dampingRatio;      // Damping ratio
};

struct StepInfo {
    double riseTime;          // Time to rise from 10% to 90% of final value
    double settlingTime;      // Time to settle within 2% of final value
    double overshoot;         // Percentage overshoot
    double steadyStateError;  // Steady-state error
    double peak;              // Peak value of the response
    double peakTime;          // Time at which peak occurs
};

enum class DiscretizationMethod {
    ZOH,
    FOH,
    Bilinear,
    Tustin,
};

enum class SystemType {
    Continuous,
    Discrete,
};

enum class GramianType {
    Controllability,
    Observability,
};

/**
 * @brief Abstract base class for all LTI systems (Linear Time-Invariant).
 *
 * Provides a common interface for transfer functions and state-space representations,
 * supporting both continuous and discrete systems.
 */
class LTI {
   public:
    virtual ~LTI() = default;

    /**
     * @brief  Compute gain and phase margins of the system.
     *
     * @return MarginInfo  Gain and phase margin information
     */
    virtual MarginInfo margin() const = 0;

    /**
     * @brief  Compute damping information of the system.
     *
     * @return DampingInfo  Natural frequency and damping ratio
     */
    virtual DampingInfo damp() const = 0;

    virtual StepInfo stepinfo() const = 0;

    virtual FrequencyResponse freqresp(const std::vector<double>& frequencies) const = 0;

    /**
     * @brief  Check if the system is stable.
     *
     * A system is stable if all poles have negative real parts (continuous) or lie inside the unit circle (discrete).
     */
    virtual bool is_stable() const = 0;

    /**
     * @brief  Get the complex poles of the system.
     *
     * @return Eigen::VectorXcd
     */
    virtual std::vector<Pole> poles() const = 0;

    /**
     * @brief  Get the complex zeros of the system.
     *
     * @return Eigen::VectorXcd
     */
    virtual std::vector<Zero> zeros() const = 0;

    /**
     * @brief  Compute the step response of the system.
     *
     * @param tStart          Start time
     * @param tEnd            End time
     * @param uStep           Step input vector (default is ones)
     * @return StepResponse
     */
    virtual StepResponse step(double tStart = 0.0, double tEnd = 10.0, ColVec uStep = ColVec::Ones(1)) const = 0;

    /**
     * @brief  Compute the impulse response of the system.
     *
     * @param tStart      Start time
     * @param tEnd        End time
     *
     * @return ImpulseResponse
     */
    virtual ImpulseResponse impulse(double tStart = 0.0, double tEnd = 10.0) const = 0;

    /**
     * @brief  Compute the Bode plot data of the system.
     *
     * @param fStart        Start frequency in Hz
     * @param fEnd          End frequency in Hz
     * @param maxPoints     Maximum number of frequency points
     *
     * @return FrequencyResponse
     */
    virtual BodeResponse bode(double fStart = 0.1, double fEnd = 1.0e4, size_t maxPoints = 500) const = 0;

    /**
     * @brief  Compute the Nyquist plot data of the system.
     *
     * @param fStart        Start frequency in Hz
     * @param fEnd          End frequency in Hz
     * @param maxPoints     Maximum number of frequency points
     *
     * @return NyquistResponse
     */
    virtual NyquistResponse nyquist(double fStart = 0.1, double fEnd = 1.0e4, size_t maxPoints = 500) const = 0;

    /**
     * @brief  Compute the Root Locus data of the system.
     *
     * @param kMin          Minimum gain
     * @param kMax          Maximum gain
     * @param numPoints     Number of gain points
     *
     * @return RootLocusResponse
     */
    virtual RootLocusResponse rlocus(double kMin = 0.0, double kMax = 100.0, size_t numPoints = 500) const = 0;

    /**
     * @brief  Discretize the LTI system
     *
     * @param Ts            Sampling time
     * @param method        Discretization method (ZOH, FOH, Bilinear, Tustin)
     * @param prewarp       Prewarp frequency for bilinear/Tustin method (optional)
     * @return StateSpace   Discretized StateSpace system
     */
    virtual StateSpace discretize(double Ts, DiscretizationMethod method = DiscretizationMethod::ZOH, std::optional<double> prewarp = std::nullopt) const = 0;

    /**
     * @brief  Get the observability information of the system.
     *
     * @return ObservabilityInfo
     */
    virtual ObservabilityInfo observability() const = 0;

    /**
     * @brief  Get the controllability information of the system.
     *
     * @return ControllabilityInfo
     */
    virtual ControllabilityInfo controllability() const = 0;

    virtual Matrix     gramian(GramianType type) const  = 0;
    virtual StateSpace minreal(double tol = 1e-9) const = 0;
    virtual StateSpace balred(size_t r) const           = 0;

    /**
     * @brief  Get the System Type object (Continuous or Discrete)
     *
     * @return SystemType
     */
    SystemType systemType() const { return Ts.has_value() ? SystemType::Discrete : SystemType::Continuous; }

    virtual StateSpace       toStateSpace() const       = 0;
    virtual TransferFunction toTransferFunction() const = 0;
    virtual ZeroPoleGain     toZeroPoleGain() const     = 0;

    bool isDiscrete() const { return Ts.has_value(); }
    bool isContinuous() const { return !Ts.has_value(); }

    std::optional<double> Ts;  // Sampling time; nullopt for continuous, value for discrete
};

StateSpace ss(Matrix A, Matrix B, Matrix C, Matrix D, std::optional<double> Ts = std::nullopt);
StateSpace ss(Matrix D);
StateSpace ss(const TransferFunction& tf);
StateSpace ss(TransferFunction&& tf);
StateSpace ss(const ZeroPoleGain& zpk_sys);
StateSpace ss(ZeroPoleGain&& zpk_sys);

TransferFunction tf(std::vector<double> num, std::vector<double> den, std::optional<double> Ts = std::nullopt);
TransferFunction tf(const ZeroPoleGain& zpk_sys);
TransferFunction tf(ZeroPoleGain&& zpk_sys);
TransferFunction tf(const StateSpace& sys);
TransferFunction tf(StateSpace&& sys);
TransferFunction tf(const StateSpace& sys, int output_idx, int input_idx);
TransferFunction tf(StateSpace&& sys, int output_idx, int input_idx);

ZeroPoleGain zpk(const StateSpace& sys);
ZeroPoleGain zpk(StateSpace&& sys);
ZeroPoleGain zpk(const StateSpace& sys, int output_idx, int input_idx);
ZeroPoleGain zpk(TransferFunction&& tf);
ZeroPoleGain zpk(const TransferFunction& tf);
ZeroPoleGain zpk(const std::vector<Zero>& zeros,
                 const std::vector<Pole>& poles,
                 double                   gain,
                 std::optional<double>    Ts = std::nullopt);

/**
 * @brief Convert a continuous-time LTI system to discrete-time using specified method.
 *
 * @param sys           Continuous-time LTI system
 * @param Ts            Sampling time
 * @param method        Discretization method (default: ZOH)
 * @param prewarp       Optional pre-warp frequency for Tustin method
 * @return StateSpace   Discrete-time StateSpace system
 */
StateSpace c2d(const LTI& sys, double Ts, DiscretizationMethod method = DiscretizationMethod::ZOH, std::optional<double> prewarp = std::nullopt);

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

};  // namespace control
