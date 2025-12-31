#pragma once

#include "LTI.hpp"
#include "types.hpp"

namespace control {

/**
 * @brief Unified state-space LTI system (continuous or discrete).
 *
 * If Ts is set, it's discrete; otherwise, continuous.
 */
class StateSpace : public LTI {
   public:
    Matrix A = {}, B = {}, C = {}, D = {};
    Matrix G = {}, H = {};  // Noise input matrices: x_dot = A*x + B*u + G*w, y = C*x + D*u + H*w + v

    bool is_stable() const override;

    StateSpace discretize(double Ts, DiscretizationMethod method = DiscretizationMethod::ZOH, std::optional<double> prewarp = std::nullopt) const override;

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
    StateSpace balreal(size_t r) const;

    std::vector<Pole> poles() const override;
    std::vector<Zero> zeros() const override;

    StateSpace       toStateSpace() const override;
    TransferFunction toTransferFunction(int output_idx = 0, int input_idx = 0) const;
    ZeroPoleGain     toZeroPoleGain() const;

    StateSpace() = default;
    StateSpace(const Matrix& A, const Matrix& B, const Matrix& C, const Matrix& D, std::optional<double> Ts = std::nullopt);
    StateSpace(Matrix&& A, Matrix&& B, Matrix&& C, Matrix&& D, std::optional<double> Ts = std::nullopt);
    StateSpace(const Matrix& A, const Matrix& B, const Matrix& C, const Matrix& D, const Matrix& G, const Matrix& H, std::optional<double> Ts = std::nullopt);
    StateSpace(Matrix&& A, Matrix&& B, Matrix&& C, Matrix&& D, Matrix&& G, Matrix&& H, std::optional<double> Ts = std::nullopt);

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
               C.isApprox(other.C) && D.isApprox(other.D) &&
               G.isApprox(other.G) && H.isApprox(other.H) && Ts == other.Ts;
    }

   private:
    Eigen::VectorXcd eigenvalues() const { return A.eigenvalues(); }
};

};  // namespace control