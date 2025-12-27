#pragma once

#include "LTI.hpp"
#include "types.hpp"

namespace control {

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

    Matrix     gramian(GramianType type) const override;
    StateSpace minreal(double tol = 1e-9) const override;
    StateSpace balred(size_t r) const override;

    std::vector<Pole> poles() const override;
    std::vector<Zero> zeros() const override;

    StateSpace discretize(double Ts, DiscretizationMethod method = DiscretizationMethod::ZOH, std::optional<double> prewarp = std::nullopt) const override;

    StateSpace       toStateSpace() const override;
    TransferFunction toTransferFunction() const;
    ZeroPoleGain     toZeroPoleGain() const;

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

    DampingInfo damp() const override;
    StepInfo    stepinfo() const override;
};
}  // namespace control