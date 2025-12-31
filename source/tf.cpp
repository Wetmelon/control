#include "tf.hpp"

#include <shlobj.h>

#include "LTI.hpp"
#include "control.hpp"
#include "ss.hpp"
#include "types.hpp"
#include "zpk.hpp"

namespace control {

static void validateTransferFunctionVectors(const std::vector<double>& num, const std::vector<double>& den) {
    if (den.empty() || std::abs(den[0]) < 1e-15) {
        throw std::invalid_argument("TransferFunction: Denominator must have nonzero leading coefficient");
    }
    if (num.empty()) {
        throw std::invalid_argument("TransferFunction: Numerator must not be empty");
    }
    if (std::abs(num[0]) < 1e-15) {
        throw std::invalid_argument("TransferFunction: Numerator must have nonzero leading coefficient");
    }
}

TransferFunction::TransferFunction(std::vector<double>   num,
                                   std::vector<double>   den,
                                   std::optional<double> Ts)
    : num(std::move(num)), den(std::move(den)) {
    validateTransferFunctionVectors(this->num, this->den);
    this->Ts = Ts;
}

TransferFunction::TransferFunction(const TransferFunction& other)
    : num(other.num), den(other.den) {
    Ts = other.Ts;
}

TransferFunction::TransferFunction(TransferFunction&& other) noexcept
    : num(std::move(other.num)), den(std::move(other.den)) {
    Ts = other.Ts;
}

TransferFunction::TransferFunction(const StateSpace& ss)
    : TransferFunction(ss.toTransferFunction()) {}

TransferFunction::TransferFunction(const ZeroPoleGain& zpk)
    : TransferFunction(zpk.toTransferFunction()) {}

TransferFunction::TransferFunction(StateSpace&& ss) noexcept
    : TransferFunction(ss.toTransferFunction()) {}

TransferFunction::TransferFunction(ZeroPoleGain&& zpk) noexcept
    : TransferFunction(zpk.toTransferFunction()) {}

// Copy and Move Assignment Operators
TransferFunction& TransferFunction::operator=(const TransferFunction& other) {
    if (this != &other) {
        num = other.num;
        den = other.den;
        Ts  = other.Ts;
    }
    return *this;
}

TransferFunction& TransferFunction::operator=(TransferFunction&& other) noexcept {
    if (this != &other) {
        num = std::move(other.num);
        den = std::move(other.den);
        Ts  = other.Ts;
    }
    return *this;
}

TransferFunction& TransferFunction::operator=(const StateSpace& ss) {
    *this = TransferFunction(ss.toTransferFunction());
    return *this;
}

TransferFunction& TransferFunction::operator=(StateSpace&& ss) noexcept {
    *this = TransferFunction(ss.toTransferFunction());
    return *this;
}

TransferFunction& TransferFunction::operator=(const ZeroPoleGain& zpk) {
    *this = TransferFunction(zpk);
    return *this;
}

TransferFunction& TransferFunction::operator=(ZeroPoleGain&& zpk) noexcept {
    *this = TransferFunction(std::move(zpk));
    return *this;
}

// Convert to StateSpace
StateSpace TransferFunction::toStateSpace() const {
    // Uses companion matrix form (controllable canonical form), which is numerically stable
    // for transfer function to state-space conversion

    const int n = static_cast<int>(den.size()) - 1;  // Order of the system
    const int m = static_cast<int>(num.size()) - 1;  // Order of the numerator

    // Normalize coefficients by leading denominator coefficient
    double den_lead = den[0];
    if (std::abs(den_lead) < 1e-15) {
        throw std::runtime_error("Leading coefficient of denominator is zero");
    }

    std::vector<double> norm_num = num;
    std::vector<double> norm_den = den;

    for (auto& c : norm_num) {
        c /= den_lead;
    }

    for (auto& c : norm_den) {
        c /= den_lead;
    }

    // Handle edge case: pure gain (no dynamics)
    if (n == 0) {
        return StateSpace{Matrix::Zero(0, 0), Matrix::Zero(0, 1), Matrix::Zero(1, 0),
                          Matrix::Constant(1, 1, norm_num[0]), Ts};
    }

    Matrix A = Matrix::Zero(n, n);
    Matrix B = Matrix::Zero(n, 1);
    Matrix C = Matrix::Zero(1, n);
    Matrix D = Matrix::Zero(1, 1);

    // Fill A matrix (companion form)
    for (int i = 0; i < n - 1; ++i) {
        A(i, i + 1) = 1.0;
    }
    for (int i = 0; i < n; ++i) {
        A(n - 1, i) = -norm_den[n - i];
    }

    // Fill B matrix
    B(n - 1, 0) = 1.0;

    // Fill C and D matrices
    // If m < n (proper), D = 0 and C comes from numerator
    // If m >= n (improper or bi-proper), need polynomial division
    if (m < n) {
        // Proper: D = 0
        D(0, 0) = 0.0;
        // C matrix: numerator coefficients followed by zeros
        for (int i = 0; i < n; ++i) {
            if (i <= m) {
                C(0, i) = norm_num[i];
            } else {
                C(0, i) = 0.0;
            }
        }
    } else {
        // m >= n: improper or bi-proper
        // Direct feedthrough
        D(0, 0) = norm_num[0];
        // C matrix gets the proper part
        for (int i = 0; i < n; ++i) {
            C(0, i) = (norm_num[i + 1] - D(0, 0) * norm_den[i + 1]);
        }
    }

    return StateSpace{A, B, C, D, Ts};
}

bool TransferFunction::is_stable() const {
    return toStateSpace().is_stable();
}

std::vector<Pole> TransferFunction::poles() const {
    // Poles are the roots of the denominator polynomial
    const int n = static_cast<int>(den.size()) - 1;  // Order of denominator

    if (n <= 0) {
        // Constant denominator (no poles) or zero denominator
        return std::vector<Pole>();
    }

    // Check if leading coefficient is zero - this is an error
    if (std::abs(den[0]) < 1e-15) {
        throw std::runtime_error("Leading coefficient of denominator is zero");
    }

    // Build companion matrix for denominator polynomial
    // For polynomial a_0*s^n + a_1*s^(n-1) + ... + a_n = 0
    // Companion matrix has the form:
    // [  -a_1/a_0   -a_2/a_0  ...  -a_n/a_0 ]
    // [    1          0       ...     0     ]
    // [    0          1       ...     0     ]
    // [   ...                               ]
    // [    0          0       ...     1   0 ]

    Matrix companion = Matrix::Zero(n, n);

    // Fill first row with normalized coefficients
    for (int i = 0; i < n; ++i) {
        companion(0, i) = -den[i + 1] / den[0];
    }

    // Fill subdiagonal with 1s
    for (int i = 1; i < n; ++i) {
        companion(i, i - 1) = 1.0;
    }

    // Compute eigenvalues (these are the poles)
    const auto& eigenvalues = companion.eigenvalues();
    return std::vector<Pole>(eigenvalues.data(), eigenvalues.data() + eigenvalues.size());
}

std::vector<Zero> TransferFunction::zeros() const {
    // Zeros are the roots of the numerator polynomial
    const int m = static_cast<int>(num.size()) - 1;  // Order of numerator

    if (m <= 0) {
        // Constant numerator (no zeros) or zero numerator
        return std::vector<Zero>();
    }

    // Check if leading coefficient is zero - this is an error
    if (std::abs(num[0]) < 1e-15) {
        throw std::runtime_error("Leading coefficient of numerator is zero");
    }

    // Build companion matrix for numerator polynomial
    // For polynomial a_0*s^m + a_1*s^(m-1) + ... + a_m = 0
    // Companion matrix has the form:
    // [  -a_1/a_0   -a_2/a_0  ...  -a_m/a_0 ]
    // [    1          0       ...     0     ]
    // [    0          1       ...     0     ]
    // [   ...                               ]
    // [    0          0       ...     1   0 ]

    Matrix companion = Matrix::Zero(m, m);

    // Fill first row with normalized coefficients
    for (int i = 0; i < m; ++i) {
        companion(0, i) = -num[i + 1] / num[0];
    }

    // Fill subdiagonal with 1s
    for (int i = 1; i < m; ++i) {
        companion(i, i - 1) = 1.0;
    }

    // Compute eigenvalues (these are the zeros)
    const auto& eigenvalues = companion.eigenvalues();
    return std::vector<Zero>(eigenvalues.data(), eigenvalues.data() + eigenvalues.size());
}

StepResponse TransferFunction::step(double tStart, double tEnd, ColVec uStep) const {
    return control::step(toStateSpace(), tStart, tEnd, uStep);
}

ImpulseResponse TransferFunction::impulse(double tStart, double tEnd) const {
    return control::impulse(toStateSpace(), tStart, tEnd);
}

BodeResponse TransferFunction::bode(double fStart, double fEnd, size_t maxPoints) const {
    return control::bode(toStateSpace(), fStart, fEnd, maxPoints);
}

NyquistResponse TransferFunction::nyquist(double fStart, double fEnd, size_t maxPoints) const {
    return control::nyquist(toStateSpace(), fStart, fEnd, maxPoints);
}

FrequencyResponse TransferFunction::freqresp(const std::vector<double>& frequencies) const {
    FrequencyResponse response;
    response.freq.reserve(frequencies.size());
    response.response.reserve(frequencies.size());

    // Evaluate transfer function at each frequency
    for (double freq : frequencies) {
        const double omega = 2.0 * std::numbers::pi * freq;

        std::complex<double> s_or_z;
        if (Ts.has_value()) {
            // Discrete: z = exp(j * omega * Ts)
            s_or_z = std::exp(std::complex<double>(0.0, omega * (*Ts)));
        } else {
            // Continuous: s = j * omega
            s_or_z = std::complex<double>(0.0, omega);
        }

        // Evaluate numerator and denominator polynomials using Horner's method
        std::complex<double> num_val = num[0];
        for (size_t k = 1; k < num.size(); ++k) {
            num_val = num_val * s_or_z + num[k];
        }

        std::complex<double> den_val = den[0];
        for (size_t k = 1; k < den.size(); ++k) {
            den_val = den_val * s_or_z + den[k];
        }

        // Compute H(s) or H(z)
        const std::complex<double> H = num_val / den_val;

        response.freq.push_back(freq);
        response.response.push_back(H);
    }

    return response;
}

RootLocusResponse TransferFunction::rlocus(double kMin, double kMax, size_t numPoints) const {
    if (Ts.has_value()) {
        throw std::runtime_error("Root locus not yet implemented for discrete systems");
    }

    RootLocusResponse response;
    response.gains.reserve(numPoints);

    // Generate linearly spaced gain values
    const double kStep = (kMax - kMin) / (numPoints - 1);

    // For root locus, we solve the characteristic equation: 1 + k*G(s) = 0
    // Which becomes: k*num(s) + den(s) = 0
    // Or: k*num(s) = -den(s)

    std::vector<Eigen::VectorXcd> all_poles;
    all_poles.reserve(numPoints);

    for (size_t i = 0; i < numPoints; ++i) {
        const double k = kMin + i * kStep;
        response.gains.push_back(k);

        // Build the characteristic polynomial: k*num(s) + den(s)
        std::vector<double> char_poly;
        char_poly.reserve(std::max(num.size(), den.size()));

        // Add k*num(s) and den(s), padding with zeros as needed
        const size_t max_order = std::max(num.size(), den.size()) - 1;
        for (size_t order = 0; order <= max_order; ++order) {
            double coeff = 0.0;
            // Add den[s] coefficient (if exists)
            if (order < den.size()) {
                coeff += den[den.size() - 1 - order];
            }
            // Add k*num[s] coefficient (if exists)
            if (order < num.size()) {
                coeff += k * num[num.size() - 1 - order];
            }
            char_poly.push_back(coeff);
        }

        // Build companion matrix for the characteristic polynomial
        const int n         = static_cast<int>(char_poly.size()) - 1;
        Matrix    companion = Matrix::Zero(n, n);

        // First row: normalized coefficients with negative sign
        for (int col = 0; col < n; ++col) {
            companion(0, col) = -char_poly[col + 1] / char_poly[0];
        }

        // Subdiagonal: 1s
        for (int row = 1; row < n; ++row) {
            companion(row, row - 1) = 1.0;
        }

        // Compute poles (eigenvalues of companion matrix)
        Eigen::VectorXcd cl_poles = companion.eigenvalues();
        all_poles.push_back(cl_poles);
    }

    // Organize poles into branches (track pole movement)
    if (!all_poles.empty()) {
        const int n_poles = all_poles[0].size();
        response.branches.resize(n_poles);

        for (int pole_idx = 0; pole_idx < n_poles; ++pole_idx) {
            response.branches[pole_idx].reserve(numPoints);
            for (size_t k_idx = 0; k_idx < numPoints; ++k_idx) {
                response.branches[pole_idx].push_back(all_poles[k_idx](pole_idx));
            }
        }
    }

    return response;
}

MarginInfo TransferFunction::margin() const {
    return control::margin(toStateSpace());
}

DampingInfo TransferFunction::damp() const {
    return control::damp(toStateSpace());
}

StepInfo TransferFunction::stepinfo() const {
    return control::stepinfo(toStateSpace());
}

ObservabilityInfo TransferFunction::observability() const {
    return control::observability(toStateSpace());
}

ControllabilityInfo TransferFunction::controllability() const {
    return control::controllability(toStateSpace());
}

Matrix TransferFunction::gramian(GramianType type) const {
    return control::gramian(toStateSpace(), type);
}

StateSpace TransferFunction::minreal(double tol) const {
    return control::minreal(toStateSpace(), tol);
}

StateSpace TransferFunction::balred(size_t r) const {
    return control::balred(toStateSpace(), r);
}

StateSpace TransferFunction::discretize(double Ts, DiscretizationMethod method, std::optional<double> prewarp) const {
    return control::c2d(toStateSpace(), Ts, method, prewarp);
}

TransferFunction TransferFunction::toTransferFunction() const {
    return *this;
}

ZeroPoleGain TransferFunction::toZeroPoleGain() const {
    // Normalize the transfer function so denominator is monic (leading coefficient = 1)
    double den_lead = den[0];
    if (std::abs(den_lead) < 1e-15) {
        throw std::invalid_argument("TransferFunction denominator has zero leading coefficient");
    }

    // Normalize coefficients
    std::vector<double> norm_num = num;
    std::vector<double> norm_den = den;
    for (auto& c : norm_num) c /= den_lead;
    for (auto& c : norm_den) c /= den_lead;
    norm_den[0] = 1.0;  // Ensure monic

    // Create normalized transfer function
    TransferFunction norm_tf(norm_num, norm_den, Ts);

    // Extract gain (leading coefficient of normalized numerator)
    double gain = norm_num[0];

    // Extract zeros and poles
    std::vector<Zero> zeros_vec = norm_tf.zeros();
    std::vector<Pole> poles_vec = norm_tf.poles();

    // Return ZeroPoleGain
    return ZeroPoleGain(std::move(zeros_vec), std::move(poles_vec), gain, Ts);
}
}  // namespace control