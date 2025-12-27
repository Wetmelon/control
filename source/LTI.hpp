#pragma once

#include <optional>

#include "types.hpp"
#include "unsupported/Eigen/MatrixFunctions"  // IWYU pragma: keep

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
    Matrix gramian;
    size_t rank;
    bool   isObservable;
};

struct ControllabilityInfo {
    Matrix gramian;
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

template <class T>
concept SSConvertible = requires(const T& t) { { t.toStateSpace() }; };

/**
 * @brief Abstract base class for all LTI systems (Linear Time-Invariant).
 *
 * Provides a common interface for transfer functions and state-space representations,
 * supporting both continuous and discrete systems.
 */
class LTI {
   public:
    virtual ~LTI() = default;

    /* TODO:
        - Implement margin(), stepinfo(), freqresp(), etc. with appropriate return structures
    */

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
    // virtual DampingInfo damp() const = 0;

    // virtual StepInfo stepinfo() const = 0;

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

/**
 * @brief Unified state-space LTI system (continuous or discrete).
 *
 * If Ts is set, it's discrete; otherwise, continuous.
 */
class StateSpace : public LTI {
   public:
    Matrix A, B, C, D;

    bool is_stable() const override;

    StateSpace discretize(double Ts, DiscretizationMethod method = DiscretizationMethod::ZOH, std::optional<double> prewarp = std::nullopt) const override;

    StepResponse      step(double tStart = 0.0, double tEnd = 10.0, ColVec uStep = ColVec::Ones(1)) const override;
    ImpulseResponse   impulse(double tStart = 0.0, double tEnd = 10.0) const override;
    BodeResponse      bode(double fStart = 0.1, double fEnd = 1.0e4, size_t maxPoints = 500) const override;
    NyquistResponse   nyquist(double fStart = 0.1, double fEnd = 1.0e4, size_t maxPoints = 500) const override;
    RootLocusResponse rlocus(double kMin = 0.0, double kMax = 100.0, size_t numPoints = 500) const override;
    MarginInfo        margin() const override;
    FrequencyResponse freqresp(const std::vector<double>& frequencies) const override;

    ObservabilityInfo   observability() const override;
    ControllabilityInfo controllability() const override;

    std::vector<Pole> poles() const override;
    std::vector<Zero> zeros() const override;

    StateSpace       toStateSpace() const override;
    TransferFunction toTransferFunction() const override;
    ZeroPoleGain     toZeroPoleGain() const override;

    StateSpace(const Matrix& A, const Matrix& B, const Matrix& C, const Matrix& D, std::optional<double> Ts = std::nullopt);
    StateSpace(Matrix&& A, Matrix&& B, Matrix&& C, Matrix&& D, std::optional<double> Ts = std::nullopt);

    StateSpace(const StateSpace& other);
    StateSpace(const TransferFunction& tf);
    StateSpace(const ZeroPoleGain& zpk);

    StateSpace(StateSpace&& other) noexcept;
    StateSpace(TransferFunction&& tf) noexcept;
    StateSpace(ZeroPoleGain&& zpk) noexcept;

    StateSpace& operator=(const StateSpace& other);
    StateSpace& operator=(const TransferFunction& tf);
    StateSpace& operator=(const ZeroPoleGain& zpk);

    StateSpace& operator=(StateSpace&& other) noexcept;
    StateSpace& operator=(TransferFunction&& tf) noexcept;
    StateSpace& operator=(ZeroPoleGain&& zpk) noexcept;

    // State-space output equation: y = Cx + Du
    ColVec output(const ColVec& x, const ColVec& u) const { return C * x + D * u; }

    // Equality comparison
    bool operator==(const StateSpace& other) const {
        return A.isApprox(other.A) && B.isApprox(other.B) &&
               C.isApprox(other.C) && D.isApprox(other.D) && Ts == other.Ts;
    }

   private:
    Eigen::VectorXcd eigenvalues() const { return A.eigenvalues(); }
};

/**
 * @brief Unified transfer function LTI system (continuous or discrete).
 *
 * If Ts is set, it's discrete; otherwise, continuous.
 */
class TransferFunction : public LTI {
   public:
    std::vector<double> num, den;

    bool is_stable() const override;

    StepResponse      step(double tStart = 0.0, double tEnd = 10.0, ColVec uStep = ColVec::Ones(1)) const override;
    ImpulseResponse   impulse(double tStart = 0.0, double tEnd = 10.0) const override;
    BodeResponse      bode(double fStart = 0.1, double fEnd = 1.0e4, size_t maxPoints = 500) const override;
    NyquistResponse   nyquist(double fStart = 0.1, double fEnd = 1.0e4, size_t maxPoints = 500) const override;
    RootLocusResponse rlocus(double kMin = 0.0, double kMax = 100.0, size_t numPoints = 500) const override;
    MarginInfo        margin() const override;
    FrequencyResponse freqresp(const std::vector<double>& frequencies) const override;

    ObservabilityInfo   observability() const override;
    ControllabilityInfo controllability() const override;

    std::vector<Pole> poles() const override;
    std::vector<Zero> zeros() const override;

    StateSpace discretize(double Ts, DiscretizationMethod method = DiscretizationMethod::ZOH, std::optional<double> prewarp = std::nullopt) const override;

    StateSpace       toStateSpace() const override;
    TransferFunction toTransferFunction() const override;
    ZeroPoleGain     toZeroPoleGain() const override;

    // Default constructor - creates a zero transfer function
    TransferFunction()
        : num({0.0}), den({1.0}) {}

    TransferFunction(std::vector<double>   num,
                     std::vector<double>   den,
                     std::optional<double> Ts = std::nullopt);

    TransferFunction(const TransferFunction& other);
    TransferFunction(const StateSpace& ss);
    TransferFunction(const ZeroPoleGain& zpk);

    TransferFunction(TransferFunction&& other) noexcept;
    TransferFunction(StateSpace&& ss) noexcept;
    TransferFunction(ZeroPoleGain&& zpk) noexcept;

    // Copy and Move Assignment Operators
    TransferFunction& operator=(const TransferFunction& other);
    TransferFunction& operator=(const StateSpace& ss);
    TransferFunction& operator=(const ZeroPoleGain& zpk);

    TransferFunction& operator=(TransferFunction&& other) noexcept;
    TransferFunction& operator=(StateSpace&& ss) noexcept;
    TransferFunction& operator=(ZeroPoleGain&& zpk) noexcept;

   private:
    // Caching of state-space representation to avoid repeated conversions
    mutable std::optional<StateSpace> ss_cache_;
};

class ZeroPoleGain : public LTI {
   public:
    std::vector<Zero> zeros_;
    std::vector<Pole> poles_;
    double            gain_;

    bool is_stable() const override;

    StepResponse      step(double tStart = 0.0, double tEnd = 10.0, ColVec uStep = ColVec::Ones(1)) const override;
    ImpulseResponse   impulse(double tStart = 0.0, double tEnd = 10.0) const override;
    BodeResponse      bode(double fStart = 0.1, double fEnd = 1.0e4, size_t maxPoints = 500) const override;
    NyquistResponse   nyquist(double fStart = 0.1, double fEnd = 1.0e4, size_t maxPoints = 500) const override;
    RootLocusResponse rlocus(double kMin = 0.0, double kMax = 100.0, size_t numPoints = 500) const override;
    MarginInfo        margin() const override;
    FrequencyResponse freqresp(const std::vector<double>& frequencies) const override;

    ObservabilityInfo   observability() const override;
    ControllabilityInfo controllability() const override;

    std::vector<Pole> poles() const override { return poles_; };
    std::vector<Zero> zeros() const override { return zeros_; };
    double            gain() const { return gain_; }

    StateSpace discretize(double                Ts,
                          DiscretizationMethod  method  = DiscretizationMethod::ZOH,
                          std::optional<double> prewarp = std::nullopt) const override;

    StateSpace       toStateSpace() const override;
    TransferFunction toTransferFunction() const override;
    ZeroPoleGain     toZeroPoleGain() const override;

    ZeroPoleGain(const StateSpace& ss);
    ZeroPoleGain(const TransferFunction& tf);

    ZeroPoleGain(std::vector<Zero>     zeros,
                 std::vector<Pole>     poles,
                 double                gain,
                 std::optional<double> Ts = std::nullopt)
        : zeros_(std::move(zeros)), poles_(std::move(poles)), gain_(gain) {
        this->Ts = Ts;
    }

    ZeroPoleGain(const ZeroPoleGain& other);
    ZeroPoleGain(ZeroPoleGain&& other) noexcept;

    ZeroPoleGain& operator=(const ZeroPoleGain& other);
    ZeroPoleGain& operator=(const StateSpace& ss);
    ZeroPoleGain& operator=(const TransferFunction& tf);

    ZeroPoleGain& operator=(ZeroPoleGain&& other) noexcept;
    ZeroPoleGain& operator=(StateSpace&& ss) noexcept;
    ZeroPoleGain& operator=(TransferFunction&& tf) noexcept;

   private:
    // Caching of state-space representation to avoid repeated conversions
    mutable std::optional<StateSpace> ss_cache_;

    // Caching of transfer function representation to avoid repeated conversions
    mutable std::optional<TransferFunction> tf_cache_;
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

// Model reduction utilities (operate on StateSpace representation)
StateSpace minreal(const StateSpace& sys, double tol = 1e-9);
StateSpace balred(const StateSpace& sys, size_t r);

template <SSConvertible T>
StateSpace minreal(const T& t, double tol = 1e-9) {
    return minreal(t.toStateSpace(), tol);
}

template <SSConvertible T>
StateSpace balred(const T& t, size_t r) {
    return balred(t.toStateSpace(), r);
}

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

// ---------------------------------------------------------------------------
// LTI operations for mixed types always return StateSpace representation
// ---------------------------------------------------------------------------

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

};  // namespace control
