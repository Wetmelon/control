#include "LTI.hpp"

#include <cmath>
#include <optional>

#include "ss.hpp"
#include "types.hpp"

namespace control {

/* LTI Member Function Definitions - Default to StateSpace dispatch */
bool LTI::is_stable() const {
    return toStateSpace().is_stable();
}
MarginInfo LTI::margin() const {
    return toStateSpace().margin();
}
DampingInfo LTI::damp() const {
    return toStateSpace().damp();
}
StepInfo LTI::stepinfo() const {
    return toStateSpace().stepinfo();
}
FrequencyResponse LTI::freqresp(const std::vector<double>& frequencies) const {
    return toStateSpace().freqresp(frequencies);
}
std::vector<Pole> LTI::poles() const {
    return toStateSpace().poles();
}
std::vector<Zero> LTI::zeros() const {
    return toStateSpace().zeros();
}
StepResponse LTI::step(double tStart, double tEnd, ColVec uStep) const {
    return toStateSpace().step(tStart, tEnd, uStep);
}
ImpulseResponse LTI::impulse(double tStart, double tEnd) const {
    return toStateSpace().impulse(tStart, tEnd);
}
BodeResponse LTI::bode(double fStart, double fEnd, size_t maxPoints) const {
    return toStateSpace().bode(fStart, fEnd, maxPoints);
}
NyquistResponse LTI::nyquist(double fStart, double fEnd, size_t maxPoints) const {
    return toStateSpace().nyquist(fStart, fEnd, maxPoints);
}
RootLocusResponse LTI::rlocus(double kMin, double kMax, size_t numPoints) const {
    return toStateSpace().rlocus(kMin, kMax, numPoints);
}
StateSpace LTI::discretize(double Ts, DiscretizationMethod method, std::optional<double> prewarp) const {
    return toStateSpace().discretize(Ts, method, prewarp);
}
ObservabilityInfo LTI::observability() const {
    return toStateSpace().observability();
}
ControllabilityInfo LTI::controllability() const {
    return toStateSpace().controllability();
}
Matrix LTI::gramian(GramianType type) const {
    return toStateSpace().gramian(type);
}
StateSpace LTI::minreal(double tol) const {
    return toStateSpace().minreal(tol);
}
StateSpace LTI::balred(size_t r) const {
    return toStateSpace().balred(r);
}

}  // namespace control
