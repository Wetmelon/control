#pragma once

#include "LTI.hpp"
#include "ss.hpp"
#include "tf.hpp"
#include "types.hpp"

namespace control {

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
    DampingInfo       damp() const override;
    StepInfo          stepinfo() const override;

    ObservabilityInfo   observability() const override;
    ControllabilityInfo controllability() const override;

    Matrix     gramian(GramianType type) const override;
    StateSpace minreal(double tol = 1e-9) const override;
    StateSpace balred(size_t r) const override;

    std::vector<Pole> poles() const override { return poles_; };
    std::vector<Zero> zeros() const override { return zeros_; };
    double            gain() const { return gain_; }

    StateSpace discretize(double                Ts,
                          DiscretizationMethod  method  = DiscretizationMethod::ZOH,
                          std::optional<double> prewarp = std::nullopt) const override;

    StateSpace       toStateSpace() const override;
    TransferFunction toTransferFunction() const;
    ZeroPoleGain     toZeroPoleGain() const;

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
};

}  // namespace control