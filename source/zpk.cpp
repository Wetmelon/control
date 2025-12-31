#include "zpk.hpp"

#include "LTI.hpp"
#include "control.hpp"
#include "types.hpp"

namespace control {

ZeroPoleGain::ZeroPoleGain(const StateSpace& ss) {
    *this = ss.toZeroPoleGain();  // Delegate to method
}

ZeroPoleGain::ZeroPoleGain(const TransferFunction& tf) {
    *this = tf.toZeroPoleGain();  // Delegate to method
}

ZeroPoleGain::ZeroPoleGain(const ZeroPoleGain& other)
    : zeros_(other.zeros_), poles_(other.poles_), gain_(other.gain_) {
    this->Ts = other.Ts;
}

ZeroPoleGain::ZeroPoleGain(ZeroPoleGain&& other) noexcept
    : zeros_(std::move(other.zeros_)), poles_(std::move(other.poles_)), gain_(other.gain_) {
    this->Ts = other.Ts;
}

ZeroPoleGain& ZeroPoleGain::operator=(const ZeroPoleGain& other) {
    if (this != &other) {
        zeros_ = other.zeros_;
        poles_ = other.poles_;
        gain_  = other.gain_;
        Ts     = other.Ts;
    }
    return *this;
}

ZeroPoleGain& ZeroPoleGain::operator=(ZeroPoleGain&& other) noexcept {
    if (this != &other) {
        zeros_ = std::move(other.zeros_);
        poles_ = std::move(other.poles_);
        gain_  = other.gain_;
        Ts     = other.Ts;
    }
    return *this;
}

ZeroPoleGain& ZeroPoleGain::operator=(const StateSpace& ss) {
    *this = ZeroPoleGain(ss);
    return *this;
}

ZeroPoleGain& ZeroPoleGain::operator=(StateSpace&& ss) noexcept {
    *this = ZeroPoleGain(std::move(ss));
    return *this;
}

ZeroPoleGain& ZeroPoleGain::operator=(const TransferFunction& tf) {
    *this = ZeroPoleGain(tf);
    return *this;
}

ZeroPoleGain& ZeroPoleGain::operator=(TransferFunction&& tf) noexcept {
    *this = ZeroPoleGain(std::move(tf));
    return *this;
}

bool ZeroPoleGain::is_stable() const {
    if (Ts.has_value()) {
        // Discrete: unstable if any |pole| >= 1
        for (const auto& p : poles_) {
            if (std::abs(p) >= 1.0) {
                return false;
            }
        }
    } else {
        // Continuous: unstable if any Re(pole) >= 0
        for (const auto& p : poles_) {
            if (p.real() >= 0.0) {
                return false;
            }
        }
    }
    return true;
}

StepResponse ZeroPoleGain::step(double tStart, double tEnd, ColVec uStep) const {
    return control::step(toStateSpace(), tStart, tEnd, uStep);
}

ImpulseResponse ZeroPoleGain::impulse(double tStart, double tEnd) const {
    return control::impulse(toStateSpace(), tStart, tEnd);
}

BodeResponse ZeroPoleGain::bode(double fStart, double fEnd, size_t maxPoints) const {
    return control::bode(toStateSpace(), fStart, fEnd, maxPoints);
}

NyquistResponse ZeroPoleGain::nyquist(double fStart, double fEnd, size_t maxPoints) const {
    return control::nyquist(toStateSpace(), fStart, fEnd, maxPoints);
}

RootLocusResponse ZeroPoleGain::rlocus(double kMin, double kMax, size_t numPoints) const {
    return control::rlocus(toStateSpace(), kMin, kMax, numPoints);
}

MarginInfo ZeroPoleGain::margin() const {
    return control::margin(toStateSpace());
}

DampingInfo ZeroPoleGain::damp() const {
    return control::damp(toStateSpace());
}

StepInfo ZeroPoleGain::stepinfo() const {
    return control::stepinfo(toStateSpace());
}

FrequencyResponse ZeroPoleGain::freqresp(const std::vector<double>& frequencies) const {
    return control::freqresp(toStateSpace(), frequencies);
}

ObservabilityInfo ZeroPoleGain::observability() const {
    return control::observability(toStateSpace());
}

ControllabilityInfo ZeroPoleGain::controllability() const {
    return control::controllability(toStateSpace());
}

Matrix ZeroPoleGain::gramian(GramianType type) const {
    return control::gramian(toStateSpace(), type);
}

StateSpace ZeroPoleGain::minreal(double tol) const {
    return control::minreal(toStateSpace(), tol);
}

StateSpace ZeroPoleGain::balred(size_t r) const {
    return control::balred(toStateSpace(), r);
}

// Build state-space from zeros, poles, and gain
StateSpace ZeroPoleGain::toStateSpace() const {
    return this->toTransferFunction().toStateSpace();
}

// Convert to TransferFunction
TransferFunction ZeroPoleGain::toTransferFunction() const {
    // Build transfer function from zeros, poles, and gain
    // G(s) = K * (s - z1)(s - z2)... / (s - p1)(s - p2)...

    // Helper lambda to expand polynomial from roots
    auto expand_poly = [](const std::vector<std::complex<double>>& roots) -> std::vector<double> {
        if (roots.empty()) {
            return {1.0};
        }

        std::vector<std::complex<double>> coeffs = {1.0};

        for (const auto& root : roots) {
            std::vector<std::complex<double>> new_coeffs(coeffs.size() + 1, 0.0);
            for (size_t i = 0; i < coeffs.size(); ++i) {
                new_coeffs[i] += coeffs[i];
                new_coeffs[i + 1] -= coeffs[i] * root;
            }
            coeffs = new_coeffs;
        }

        // Convert to real coefficients (imaginary parts should be negligible)
        std::vector<double> real_coeffs;
        real_coeffs.reserve(coeffs.size());
        for (const auto& c : coeffs) {
            real_coeffs.push_back(c.real());
        }
        return real_coeffs;
    };

    // Expand numerator and denominator
    std::vector<double> num_coeffs = expand_poly(zeros_);
    std::vector<double> den_coeffs = expand_poly(poles_);

    // Multiply numerator by gain
    for (auto& c : num_coeffs) {
        c *= gain_;
    }

    return TransferFunction(num_coeffs, den_coeffs, Ts);
}

StateSpace ZeroPoleGain::discretize(double                Ts,
                                    DiscretizationMethod  method,
                                    std::optional<double> prewarp) const {
    return control::c2d(toStateSpace(), Ts, method, prewarp);
}

ZeroPoleGain ZeroPoleGain::toZeroPoleGain() const {
    return *this;
}

}  // namespace control