#include <numbers>

#include "LTI.hpp"
#include "solver.hpp"
#include "types.hpp"
#include "utility.hpp"

namespace control {

// Unwrap phase to ensure continuity (remove 2π jumps)
static inline void unwrap_phase(std::vector<double>& phases) {
    for (size_t i = 1; i < phases.size(); ++i) {
        double diff = phases[i] - phases[i - 1];
        while (diff > 180.0) {
            phases[i] -= 360.0;
            diff -= 360.0;
        }
        while (diff < -180.0) {
            phases[i] += 360.0;
            diff += 360.0;
        }
    }
}

static void validateStateSpaceMatrices(const Matrix& A, const Matrix& B, const Matrix& C, const Matrix& D) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("StateSpace: A must be square");
    }
    if (B.rows() != A.rows()) {
        throw std::invalid_argument("StateSpace: B.rows() must match A.rows()");
    }
    if (C.cols() != A.cols()) {
        throw std::invalid_argument("StateSpace: C.cols() must match A.cols()");
    }
    if (D.rows() != C.rows() || D.cols() != B.cols()) {
        throw std::invalid_argument("StateSpace: D shape must be (C.rows(), B.cols())");
    }
}

StateSpace::StateSpace(const Matrix& A, const Matrix& B, const Matrix& C, const Matrix& D, std::optional<double> Ts)
    : A(A), B(B), C(C), D(D) {
    validateStateSpaceMatrices(A, B, C, D);
    this->Ts = Ts;
}

StateSpace::StateSpace(Matrix&& A, Matrix&& B, Matrix&& C, Matrix&& D, std::optional<double> Ts)
    : A(std::move(A)), B(std::move(B)), C(std::move(C)), D(std::move(D)) {
    validateStateSpaceMatrices(this->A, this->B, this->C, this->D);
    this->Ts = Ts;
}

StateSpace::StateSpace(const TransferFunction& tf)
    : StateSpace(ss(tf)) {}

StateSpace::StateSpace(TransferFunction&& tf) noexcept
    : StateSpace(ss(std::move(tf))) {}

StateSpace::StateSpace(const StateSpace& other)
    : A(other.A), B(other.B), C(other.C), D(other.D) {
    validateStateSpaceMatrices(this->A, this->B, this->C, this->D);
    this->Ts = other.Ts;
}

StateSpace::StateSpace(StateSpace&& other) noexcept
    : A(std::move(other.A)), B(std::move(other.B)), C(std::move(other.C)), D(std::move(other.D)) {
    validateStateSpaceMatrices(this->A, this->B, this->C, this->D);
    this->Ts = other.Ts;
}

// Copy assignment
StateSpace& StateSpace::operator=(const StateSpace& other) {
    if (this != &other) {
        A  = other.A;
        B  = other.B;
        C  = other.C;
        D  = other.D;
        Ts = other.Ts;
    }
    return *this;
}

// Move assignment
StateSpace& StateSpace::operator=(StateSpace&& other) noexcept {
    if (this != &other) {
        A  = std::move(other.A);
        B  = std::move(other.B);
        C  = std::move(other.C);
        D  = std::move(other.D);
        Ts = other.Ts;
    }
    return *this;
}

// Assignment from TransferFunction
StateSpace& StateSpace::operator=(const TransferFunction& tf) {
    *this = ss(tf);
    return *this;
}

// Move assignment from TransferFunction
StateSpace& StateSpace::operator=(TransferFunction&& tf) noexcept {
    *this = ss(std::move(tf));
    return *this;
}

bool StateSpace::is_stable() const {
    Eigen::EigenSolver<Matrix> solver(A);
    const auto&                eigenvalues = solver.eigenvalues();

    if (Ts.has_value()) {
        // Discrete: unstable if any |eigenvalue| >= 1
        for (int i = 0; i < eigenvalues.size(); ++i) {
            if (std::abs(eigenvalues[i]) >= 1.0) {
                return false;
            }
        }
    } else {
        // Continuous: unstable if any Re(eigenvalue) >= 0
        for (int i = 0; i < eigenvalues.size(); ++i) {
            if (eigenvalues[i].real() >= 0.0) {
                return false;
            }
        }
    }
    return true;
}

std::vector<Pole> StateSpace::poles() const {
    const auto& eigenvalues = A.eigenvalues();

    std::vector<Pole> poles;
    poles.reserve(eigenvalues.size());
    for (int i = 0; i < eigenvalues.size(); ++i) {
        poles.push_back(eigenvalues[i]);
    }
    return poles;
}

std::vector<Zero> StateSpace::zeros() const {
    // For state-space systems, zeros are the roots of the numerator of the transfer function
    // Convert to transfer function and compute zeros from there
    if (B.cols() != 1 || C.rows() != 1) {
        throw std::invalid_argument("zeros() only works for SISO systems");
    }

    auto tf_sys = toTransferFunction();
    return tf_sys.zeros();
}

MarginInfo StateSpace::margin() const {
    // For discrete systems, adjust frequency range to avoid aliasing
    double fStart = 1e-3;  // 0.001 Hz
    double fEnd   = 1e4;   // 10000 Hz

    if (Ts.has_value()) {
        fEnd = 1.0 / (2.0 * (*Ts));  // Nyquist frequency
    }

    // Compute Bode plot over appropriate frequency range
    const size_t numPoints = 1000;
    auto         bode_resp = bode(fStart, fEnd, numPoints);

    // Find gain crossover frequency (where magnitude is closest to 0 dB)
    double gainCrossover = 0.0;
    double minMagDiff    = std::numeric_limits<double>::max();

    for (size_t i = 0; i < bode_resp.freq.size(); ++i) {
        double magDiff = std::abs(bode_resp.magnitude[i]);
        if (magDiff < minMagDiff) {
            minMagDiff    = magDiff;
            gainCrossover = bode_resp.freq[i];
        }
    }

    // Find phase crossover frequency (where phase is closest to -180°)
    double phaseCrossover = 0.0;
    double minPhaseDiff   = std::numeric_limits<double>::max();

    for (size_t i = 0; i < bode_resp.freq.size(); ++i) {
        double phaseDiff = std::abs(bode_resp.phase[i] - (-180.0));
        if (phaseDiff < minPhaseDiff) {
            minPhaseDiff   = phaseDiff;
            phaseCrossover = bode_resp.freq[i];
        }
    }

    // Compute gain margin: -magnitude at phase crossover frequency
    double gainMargin = 0.0;
    if (phaseCrossover > 0.0) {
        // Find magnitude at phase crossover
        for (size_t i = 0; i < bode_resp.freq.size(); ++i) {
            if (std::abs(bode_resp.freq[i] - phaseCrossover) < 1e-6) {
                gainMargin = -bode_resp.magnitude[i];
                break;
            }
        }
    }

    // Compute phase margin: 180° + phase at gain crossover frequency
    double phaseMargin = 0.0;
    if (gainCrossover > 0.0) {
        // Find phase at gain crossover
        for (size_t i = 0; i < bode_resp.freq.size(); ++i) {
            if (std::abs(bode_resp.freq[i] - gainCrossover) < 1e-6) {
                phaseMargin = 180.0 + bode_resp.phase[i];
                break;
            }
        }
    }

    return MarginInfo{
        .gainMargin     = gainMargin,
        .phaseMargin    = phaseMargin,
        .gainCrossover  = gainCrossover,
        .phaseCrossover = phaseCrossover};
}

StepResponse StateSpace::step(double tStart, double tEnd, ColVec uStep) const {
    if (uStep.size() == 0) {
        uStep = ColVec::Ones(B.cols(), 1);
    }

    if (Ts.has_value()) {
        // Discrete step response
        size_t numPoints = static_cast<size_t>((tEnd - tStart) / (*Ts)) + 1;

        StepResponse response;
        response.time.reserve(numPoints);
        response.output.reserve(numPoints);

        ColVec x = ColVec::Zero(A.rows(), 1);  // Start from zero initial conditions
        for (size_t i = 0; i < numPoints; ++i) {
            double t = tStart + i * (*Ts);
            response.time.push_back(t);
            response.output.push_back(C * x + D * uStep);

            // Discrete-time state update: x[k+1] = A*x[k] + B*u[k]
            x = A * x + B * uStep;
        }
        return response;
    } else {
        // Continuous step response
        const std::optional<double> timestep  = std::nullopt;
        const double                dt        = timestep.value_or(0.01);  // Use configured timestep or default 10ms steps
        size_t                      numPoints = static_cast<size_t>((tEnd - tStart) / dt) + 1;

        // Generate time evaluation points
        std::vector<double> t_eval;
        t_eval.reserve(numPoints);
        for (size_t i = 0; i < numPoints; ++i) {
            t_eval.push_back(tStart + i * dt);
        }

        // Use exact solver for LTI system with constant input
        ColVec x0     = ColVec::Zero(A.rows());  // Start from zero initial conditions
        auto   result = ExactSolver().solve(A, B, x0, uStep, {tStart, tEnd}, t_eval);

        // Convert SolveResult to StepResponse
        StepResponse response;
        response.time.reserve(result.t.size());
        response.output.reserve(result.x.size());

        for (size_t i = 0; i < result.x.size(); ++i) {
            response.time.push_back(result.t[i]);
            response.output.push_back(output(result.x[i], uStep));
        }

        return response;
    }
}

ImpulseResponse StateSpace::impulse(double tStart, double tEnd) const {
    if (Ts.has_value()) {
        // Discrete impulse response
        size_t numPoints = static_cast<size_t>((tEnd - tStart) / (*Ts)) + 1;

        ImpulseResponse response;
        response.time.reserve(numPoints);
        response.output.reserve(numPoints);

        ColVec x = ColVec::Zero(A.rows(), 1);  // Start from zero initial conditions
        for (size_t i = 0; i < numPoints; ++i) {
            double t = tStart + i * (*Ts);
            response.time.push_back(t);

            // For impulse, we apply B at the first time step only
            ColVec u_impulse;
            if (i == 0) {
                u_impulse = ColVec::Ones(B.cols(), 1);  // Unit impulse
            } else {
                u_impulse = ColVec::Zero(B.cols(), 1);
            }
            response.output.push_back(C * x + D * u_impulse);

            // Discrete-time state update: x[k+1] = A*x[k] + B*u[k]
            x = A * x + B * u_impulse;
        }
        return response;
    } else {
        // Continuous impulse response
        // For continuous systems, impulse response is the derivative of step response
        // Or equivalently: h(t) = C*e^(At)*B for initial impulse at t=0
        const std::optional<double> timestep  = std::nullopt;
        const double                dt        = timestep.value_or(0.01);
        size_t                      numPoints = static_cast<size_t>((tEnd - tStart) / dt) + 1;

        ImpulseResponse response;
        response.time.reserve(numPoints);
        response.output.reserve(numPoints);

        for (size_t i = 0; i < numPoints; ++i) {
            double t = tStart + i * dt;
            response.time.push_back(t);

            // Compute e^(At)
            Matrix expAt = (A * t).exp();

            // Impulse response: h(t) = C*e^(At)*B (+ D*delta(t), but we ignore the delta term)
            ColVec h = C * expAt * B;
            response.output.push_back(h);
        }

        return response;
    }
}

BodeResponse StateSpace::bode(double fStart, double fEnd, size_t maxPoints) const {
    // Generate logarithmically spaced frequency points
    const double logStart = std::log10(fStart);
    const double logEnd   = std::log10(fEnd);
    const double logStep  = (logEnd - logStart) / (maxPoints - 1);

    std::vector<double> freqs;
    freqs.reserve(maxPoints);
    for (size_t i = 0; i < maxPoints; ++i) {
        freqs.push_back(std::pow(10.0, logStart + i * logStep));
    }

    // Get frequency response
    auto freq_resp = freqresp(freqs);

    // Convert to magnitude (dB) and phase (degrees)
    std::vector<double> mags, phases;
    mags.reserve(maxPoints);
    phases.reserve(maxPoints);

    for (const auto& H : freq_resp.response) {
        const double magnitude = 20.0 * std::log10(std::abs(H));
        const double phase     = std::arg(H) * 180.0 / std::numbers::pi;
        mags.push_back(magnitude);
        phases.push_back(phase);
    }

    // Unwrap phase to ensure continuity
    unwrap_phase(phases);

    return BodeResponse{
        .freq      = std::move(freqs),
        .magnitude = std::move(mags),
        .phase     = std::move(phases)};
}

NyquistResponse StateSpace::nyquist(double fStart, double fEnd, size_t maxPoints) const {
    // Generate logarithmically spaced frequency points
    const double logStart = std::log10(fStart);
    const double logEnd   = std::log10(fEnd);
    const double logStep  = (logEnd - logStart) / (maxPoints - 1);

    std::vector<double> freqs;
    freqs.reserve(maxPoints);
    for (size_t i = 0; i < maxPoints; ++i) {
        freqs.push_back(std::pow(10.0, logStart + i * logStep));
    }

    // Get frequency response
    auto freq_resp = freqresp(freqs);

    return NyquistResponse{
        .response = std::move(freq_resp.response),
        .freq     = std::move(freqs)};
}

FrequencyResponse StateSpace::freqresp(const std::vector<double>& frequencies) const {
    FrequencyResponse response;
    response.freq.reserve(frequencies.size());
    response.response.reserve(frequencies.size());

    const int n = A.rows();

    // Pre-compute identity matrix and zero blocks to avoid repeated allocations
    const Matrix I = Matrix::Identity(n, n);
    const Matrix Z = Matrix::Zero(n, n);

    // Pre-allocate system matrix and RHS (reused for each frequency)
    Matrix real_sys(2 * n, 2 * n);
    Matrix real_rhs(2 * n, B.cols());

    // Set up the constant parts of RHS: [B; 0] since B is real
    real_rhs.block(0, 0, n, B.cols()) = B;
    real_rhs.block(n, 0, n, B.cols()) = Matrix::Zero(n, B.cols());

    // Evaluate transfer function at each frequency
    for (double freq : frequencies) {
        const double omega = 2.0 * std::numbers::pi * freq;

        // Form s = jω (continuous) or z = e^(jωTs) (discrete)
        std::complex<double> s_or_z;
        if (Ts.has_value()) {
            s_or_z = std::exp(std::complex<double>(0.0, omega * (*Ts)));
        } else {
            s_or_z = std::complex<double>(0.0, omega);
        }

        // Build the real block matrix for (sI - A)
        const double re_s = s_or_z.real();
        const double im_s = s_or_z.imag();

        real_sys.block(0, 0, n, n) = re_s * I - A;  // Top-left block
        real_sys.block(0, n, n, n) = -im_s * I;     // Top-right block
        real_sys.block(n, 0, n, n) = im_s * I;      // Bottom-left block
        real_sys.block(n, n, n, n) = re_s * I - A;  // Bottom-right block

        // Solve the 2n x 2n system
        const Matrix real_sol = real_sys.colPivHouseholderQr().solve(real_rhs);

        // Extract complex solution: x = real_sol[0:n] + j*real_sol[n:2n]
        const ColVec               re_x = real_sol.block(0, 0, n, 1);
        const ColVec               im_x = real_sol.block(n, 0, n, 1);
        const std::complex<double> y    = (C * re_x)(0) + std::complex<double>(0.0, 1.0) * (C * im_x)(0) + std::complex<double>(D(0, 0));  // Assume SISO

        response.freq.push_back(freq);
        response.response.push_back(y);
    }

    return response;
}

RootLocusResponse StateSpace::rlocus(double kMin, double kMax, size_t numPoints) const {
    // Convert to transfer function and compute root locus from there
    auto tf_sys = toTransferFunction();
    return tf_sys.rlocus(kMin, kMax, numPoints);
}

// Discretize (only if continuous)
StateSpace StateSpace::discretize(double Ts, DiscretizationMethod method, std::optional<double> prewarp) const {
    if (this->Ts.has_value()) {
        throw std::runtime_error("Cannot discretize an already discrete system.");
    }

    const auto I    = decltype(A)::Identity(A.rows(), A.cols());
    const auto E    = (A * Ts).exp();
    const auto Ainv = A.inverse();
    const auto I1   = Ainv * (E - I);
    const auto I2   = Ainv * (E * Ts - I1);

    switch (method) {
        case DiscretizationMethod::ZOH: {
            return StateSpace{
                E,       // A
                I1 * B,  // B
                C,       // C
                D,       // D
                Ts       // Ts
            };
        }
        case DiscretizationMethod::FOH: {
            const auto Q = I1 - (I2 / Ts);
            const auto P = I1 - Q;
            return StateSpace{
                E,                  // A
                (P + (E * Q)) * B,  // B
                C,                  // C
                C * Q * B + D,      // D
                Ts                  // Ts
            };
        }
        case DiscretizationMethod::Tustin:  // Fallthrough
        case DiscretizationMethod::Bilinear: {
            double k = 2.0 / Ts;
            if (prewarp.has_value()) {
                k = prewarp.value() / std::tan(prewarp.value() * Ts / 2.0);
            }

            const auto Q = (k * I - A).inverse();
            return StateSpace{
                Q * (k * I + A),  // A
                (I + A) * Q * B,  // B
                C,                // C
                C * Q * B + D,    // D
                Ts                // Ts
            };
        }
        default:
            // Default to ZOH
            return StateSpace{
                E,       // A
                I1 * B,  // B
                C,       // C
                D,       // D
                Ts       // Ts
            };
    }
}

ObservabilityInfo StateSpace::observability() const {
    // Structural observability: compute observability matrix and its rank.
    const int n = A.rows();
    const int p = static_cast<int>(C.rows());
    if (n == 0) {
        return ObservabilityInfo{.rank = 0, .isObservable = true};
    }

    Matrix Ob = Matrix::Zero(p * n, n);
    Matrix Ak = Matrix::Identity(n, n);
    for (int k = 0; k < n; ++k) {
        Ob.block(k * p, 0, p, n) = C * Ak;
        Ak                       = A * Ak;
    }

    Eigen::ColPivHouseholderQR<Matrix> qr(Ob);
    size_t                             rank         = qr.rank();
    bool                               isObservable = (rank == static_cast<size_t>(n));

    // Do not compute the gramian here; return empty (zero) gramian to avoid expensive or unstable ops.
    return ObservabilityInfo{.rank = rank, .isObservable = isObservable};
}

ControllabilityInfo StateSpace::controllability() const {
    // Structural controllability: compute controllability matrix and its rank.
    const int n = A.rows();
    const int m = static_cast<int>(B.cols());
    if (n == 0) {
        return ControllabilityInfo{.rank = 0, .isControllable = true};
    }

    Matrix Ctrb = Matrix::Zero(n, n * std::max(1, m));
    Matrix Ak2  = Matrix::Identity(n, n);
    for (int k = 0; k < n; ++k) {
        Ctrb.block(0, k * m, n, m) = Ak2 * B;
        Ak2                        = A * Ak2;
    }

    Eigen::ColPivHouseholderQR<Matrix> qr(Ctrb);
    size_t                             rank           = qr.rank();
    bool                               isControllable = (rank == static_cast<size_t>(n));

    // Do not compute the gramian here; return empty (zero) gramian to avoid expensive or unstable ops.
    return ControllabilityInfo{.rank = rank, .isControllable = isControllable};
}

// Compute Gramian matrices for continuous-time systems using iterative series
Matrix gramian(const StateSpace& sys, GramianType type) {
    if (sys.isDiscrete()) {
        throw std::runtime_error("gramian: discrete-time systems are not yet supported");
    }

    const Matrix A = sys.A;
    const Matrix B = sys.B;
    const Matrix C = sys.C;

    const int n = static_cast<int>(A.rows());
    if (n == 0) return Matrix::Zero(0, 0);

    Matrix    W        = Matrix::Zero(n, n);
    const int max_iter = 1000;
    if (type == GramianType::Observability) {
        Matrix term = C.transpose() * C;
        W += term;
        for (int k = 1; k < max_iter; ++k) {
            term = A.transpose() * term * A;
            W += term;
            if (term.norm() < 1e-12 * W.norm()) break;
        }
    } else {  // Controllability
        Matrix term = B * B.transpose();
        W += term;
        for (int k = 1; k < max_iter; ++k) {
            term = A * term * A.transpose();
            W += term;
            if (term.norm() < 1e-12 * W.norm()) break;
        }
    }

    return W;
}

TransferFunction StateSpace::toTransferFunction() const {
    // Check that system is SISO
    if (B.cols() != 1 || C.rows() != 1) {
        throw std::invalid_argument(
            "toTransferFunction() only works for SISO systems (1 input, 1 output). "
            "For MIMO systems, use tf(sys, output_idx, input_idx) to extract individual transfer functions.");
    }

    // For SISO systems, delegate to the indexed version
    return tf(*this, 0, 0);
}

StateSpace StateSpace::toStateSpace() const {
    return *this;
}

ZeroPoleGain StateSpace::toZeroPoleGain() const {
    // Convert via SS→TF→ZPK path
    TransferFunction tf_sys = this->toTransferFunction();
    return tf_sys.toZeroPoleGain();
}

}  // namespace control