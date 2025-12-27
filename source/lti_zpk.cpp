#include "LTI.hpp"
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
    StateSpace ss = this->toStateSpace();
    return ss.step(tStart, tEnd, uStep);
}

ImpulseResponse ZeroPoleGain::impulse(double tStart, double tEnd) const {
    StateSpace ss = this->toStateSpace();
    return ss.impulse(tStart, tEnd);
}

BodeResponse ZeroPoleGain::bode(double fStart, double fEnd, size_t maxPoints) const {
    StateSpace ss = this->toStateSpace();
    return ss.bode(fStart, fEnd, maxPoints);
}

NyquistResponse ZeroPoleGain::nyquist(double fStart, double fEnd, size_t maxPoints) const {
    StateSpace ss = this->toStateSpace();
    return ss.nyquist(fStart, fEnd, maxPoints);
}

RootLocusResponse ZeroPoleGain::rlocus(double kMin, double kMax, size_t numPoints) const {
    StateSpace ss = this->toStateSpace();
    return ss.rlocus(kMin, kMax, numPoints);
}

MarginInfo ZeroPoleGain::margin() const {
    StateSpace ss = this->toStateSpace();
    return ss.margin();
}

DampingInfo ZeroPoleGain::damp() const {
    StateSpace ss = this->toStateSpace();
    return ss.damp();
}

StepInfo ZeroPoleGain::stepinfo() const {
    StateSpace ss = this->toStateSpace();
    return ss.stepinfo();
}

FrequencyResponse ZeroPoleGain::freqresp(const std::vector<double>& frequencies) const {
    StateSpace ss = this->toStateSpace();
    return ss.freqresp(frequencies);
}

ObservabilityInfo ZeroPoleGain::observability() const {
    StateSpace ss = this->toStateSpace();
    return ss.observability();
}

ControllabilityInfo ZeroPoleGain::controllability() const {
    StateSpace ss = this->toStateSpace();
    return ss.controllability();
}

// Convert to StateSpace (with caching)
StateSpace ZeroPoleGain::toStateSpace() const {
    if (ss_cache_.has_value()) {
        return ss_cache_.value();
    }

    // Build state-space from zeros, poles, and gain
    ss_cache_ = this->toTransferFunction().toStateSpace();
    return ss_cache_.value();
}

// Convert to TransferFunction (with caching)
TransferFunction ZeroPoleGain::toTransferFunction() const {
    if (tf_cache_.has_value()) {
        return tf_cache_.value();
    }

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

    tf_cache_ = TransferFunction(num_coeffs, den_coeffs, Ts);
    return tf_cache_.value();
}

StateSpace ZeroPoleGain::discretize(double                Ts,
                                    DiscretizationMethod  method,
                                    std::optional<double> prewarp) const {
    return toStateSpace().discretize(Ts, method, prewarp);
}

ZeroPoleGain ZeroPoleGain::toZeroPoleGain() const {
    return *this;
}

}  // namespace control