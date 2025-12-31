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
    std::vector<double> naturalFrequency;  // Natural frequency (rad/s) per mode
    std::vector<double> dampingRatio;      // Damping ratio per mode
};

struct StepInfo {
    std::vector<double> riseTime;          // Time to rise from 10% to 90% of final value (per output)
    std::vector<double> settlingTime;      // Time to settle within 2% of final value (per output)
    std::vector<double> overshoot;         // Percentage overshoot (per output)
    std::vector<double> steadyStateError;  // Steady-state error (per output)
    std::vector<double> peak;              // Peak value of the response (per output)
    std::vector<double> peakTime;          // Time at which peak occurs (per output)
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

struct Input {};
struct Output {};

/**
 * @brief Abstract base class for all LTI systems (Linear Time-Invariant).
 *
 * Provides a common interface for transfer functions and state-space representations,
 * supporting both continuous and discrete systems.
 */
class LTI {
   public:
    // virtual size_t inputs() const;
    // virtual size_t outputs() const;

    virtual ~LTI() = default;

    /**
     * @brief  Compute gain and phase margins of the system.
     *
     * @return MarginInfo  Gain and phase margin information
     */
    virtual MarginInfo margin() const;

    /**
     * @brief  Compute damping information of the system.
     *
     * @return DampingInfo  Natural frequency and damping ratio
     */
    virtual DampingInfo damp() const;

    virtual StepInfo stepinfo() const;

    virtual FrequencyResponse freqresp(const std::vector<double>& frequencies) const;

    /**
     * @brief  Check if the system is stable.
     *
     * A system is stable if all poles have negative real parts (continuous) or lie inside the unit circle (discrete).
     */
    virtual bool is_stable() const;

    /**
     * @brief  Get the complex poles of the system.
     *
     * @return Eigen::VectorXcd
     */
    virtual std::vector<Pole> poles() const;

    /**
     * @brief  Get the complex zeros of the system.
     *
     * @return Eigen::VectorXcd
     */
    virtual std::vector<Zero> zeros() const;

    /**
     * @brief  Compute the step response of the system.
     *
     * @param tStart          Start time
     * @param tEnd            End time
     * @param uStep           Step input vector (default is ones)
     * @return StepResponse
     */
    virtual StepResponse step(double tStart = 0.0, double tEnd = 10.0, ColVec uStep = ColVec::Ones(1)) const;

    /**
     * @brief  Compute the impulse response of the system.
     *
     * @param tStart      Start time
     * @param tEnd        End time
     *
     * @return ImpulseResponse
     */
    virtual ImpulseResponse impulse(double tStart = 0.0, double tEnd = 10.0) const;

    /**
     * @brief  Compute the Bode plot data of the system.
     *
     * @param fStart        Start frequency in Hz
     * @param fEnd          End frequency in Hz
     * @param maxPoints     Maximum number of frequency points
     *
     * @return FrequencyResponse
     */
    virtual BodeResponse bode(double fStart = 0.1, double fEnd = 1.0e4, size_t maxPoints = 500) const;

    /**
     * @brief  Compute the Nyquist plot data of the system.
     *
     * @param fStart        Start frequency in Hz
     * @param fEnd          End frequency in Hz
     * @param maxPoints     Maximum number of frequency points
     *
     * @return NyquistResponse
     */
    virtual NyquistResponse nyquist(double fStart = 0.1, double fEnd = 1.0e4, size_t maxPoints = 500) const;

    /**
     * @brief  Compute the Root Locus data of the system.
     *
     * @param kMin          Minimum gain
     * @param kMax          Maximum gain
     * @param numPoints     Number of gain points
     *
     * @return RootLocusResponse
     */
    virtual RootLocusResponse rlocus(double kMin = 0.0, double kMax = 100.0, size_t numPoints = 500) const;

    /**
     * @brief  Discretize the LTI system
     *
     * @param Ts            Sampling time
     * @param method        Discretization method (ZOH, FOH, Bilinear, Tustin)
     * @param prewarp       Prewarp frequency for bilinear/Tustin method (optional)
     * @return StateSpace   Discretized StateSpace system
     */
    virtual StateSpace discretize(double Ts, DiscretizationMethod method = DiscretizationMethod::ZOH, std::optional<double> prewarp = std::nullopt) const;

    /**
     * @brief  Get the observability information of the system.
     *
     * @return ObservabilityInfo
     */
    virtual ObservabilityInfo observability() const;

    /**
     * @brief  Get the controllability information of the system.
     *
     * @return ControllabilityInfo
     */
    virtual ControllabilityInfo controllability() const;

    virtual Matrix     gramian(GramianType type) const;
    virtual StateSpace minreal(double tol = 1e-9) const;
    virtual StateSpace balred(size_t r) const;

    /**
     * @brief  Get the System Type object (Continuous or Discrete)
     *
     * @return SystemType
     */
    SystemType systemType() const { return Ts.has_value() ? SystemType::Discrete : SystemType::Continuous; }

    virtual StateSpace toStateSpace() const = 0;

    bool isDiscrete() const { return Ts.has_value(); }
    bool isContinuous() const { return !Ts.has_value(); }

    std::optional<double> Ts = {};  // Sampling time; nullopt for continuous, value for discrete
};

};  // namespace control
