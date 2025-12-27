#include "ss.hpp"

#include <numbers>

#include "LTI.hpp"
#include "solver.hpp"
#include "tf.hpp"
#include "types.hpp"
#include "zpk.hpp"

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
    : StateSpace(tf.toStateSpace()) {}

StateSpace::StateSpace(TransferFunction&& tf) noexcept
    : StateSpace(std::move(tf).toStateSpace()) {}

StateSpace::StateSpace(const StateSpace& other)
    : A(other.A), B(other.B), C(other.C), D(other.D) {
    this->Ts = other.Ts;
}

StateSpace::StateSpace(StateSpace&& other) noexcept
    : A(std::move(other.A)), B(std::move(other.B)), C(std::move(other.C)), D(std::move(other.D)) {
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
    *this = tf.toStateSpace();
    return *this;
}

// Move assignment from TransferFunction
StateSpace& StateSpace::operator=(TransferFunction&& tf) noexcept {
    *this = std::move(tf).toStateSpace();
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

DampingInfo StateSpace::damp() const {
    auto                 poles_vec     = this->poles();
    std::complex<double> dominant_pole = 0.0;
    double               max_imag      = 0.0;
    for (auto p : poles_vec) {
        if (std::abs(p.imag()) > max_imag) {
            max_imag      = std::abs(p.imag());
            dominant_pole = p;
        }
    }
    if (max_imag > 1e-6) {
        double abs_p = std::abs(dominant_pole);
        double zeta  = -dominant_pole.real() / abs_p;
        double wn    = abs_p;
        return {wn, zeta};
    } else {
        // no complex poles, find real pole closest to imaginary axis
        double max_re = -1e9;
        for (auto p : poles_vec) {
            if (std::abs(p.imag()) < 1e-6 && p.real() > max_re) {
                max_re        = p.real();
                dominant_pole = p;
            }
        }
        if (max_re > -1e9) {
            return {-dominant_pole.real(), 1.0};  // ζ=1, ω_n = -Re
        } else {
            return {0.0, 1.0};  // no poles?
        }
    }
}

StepInfo StateSpace::stepinfo() const {
    // Assume SISO
    if (B.cols() != 1 || C.rows() != 1) {
        throw std::invalid_argument("stepinfo() only works for SISO systems");
    }
    auto                step_resp  = this->step(0.0, 10.0, ColVec::Ones(1));
    const auto&         time       = step_resp.time;
    const auto&         output_vec = step_resp.output;
    std::vector<double> y;
    for (const auto& out : output_vec) {
        y.push_back(out(0));
    }
    double y_ss             = y.back();
    double steadyStateError = 1.0 - y_ss;
    // Find peak
    auto   max_it    = std::max_element(y.begin(), y.end());
    double peak      = *max_it;
    size_t peak_idx  = std::distance(y.begin(), max_it);
    double peakTime  = time[peak_idx];
    double overshoot = (std::abs(y_ss) > 1e-6) ? (peak - y_ss) / std::abs(y_ss) * 100.0 : 0.0;
    // Rise time: 10% to 90%
    double y10      = 0.1 * y_ss;
    double y90      = 0.9 * y_ss;
    double riseTime = 0.0;
    size_t i10 = 0, i90 = 0;
    bool   found10 = false, found90 = false;
    for (size_t i = 0; i < y.size(); ++i) {
        if (!found10 && y[i] >= y10) {
            i10     = i;
            found10 = true;
        }
        if (!found90 && y[i] >= y90) {
            i90     = i;
            found90 = true;
            break;
        }
    }
    if (found10 && found90 && i90 > i10) {
        riseTime = time[i90] - time[i10];
    }
    // Settling time: within 2% of y_ss
    double tol          = 0.02 * std::abs(y_ss);
    double settlingTime = time.back();
    for (size_t i = y.size() - 1; i > 0; --i) {
        bool settled = true;
        for (size_t j = i; j < y.size(); ++j) {
            if (std::abs(y[j] - y_ss) > tol) {
                settled = false;
                break;
            }
        }
        if (settled) {
            settlingTime = time[i];
            break;
        }
    }
    return {riseTime, settlingTime, overshoot, steadyStateError, peak, peakTime};
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

    size_t rank         = qr.rank();
    bool   isObservable = (rank == static_cast<size_t>(n));

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

    size_t rank           = qr.rank();
    bool   isControllable = (rank == static_cast<size_t>(n));

    // Do not compute the gramian here; return empty (zero) gramian to avoid expensive or unstable ops.
    return ControllabilityInfo{.rank = rank, .isControllable = isControllable};
}

// Compute Gramian matrices for continuous-time systems using iterative series
Matrix StateSpace::gramian(GramianType type) const {
    if (isDiscrete()) {
        throw std::runtime_error("gramian: discrete-time systems are not yet supported");
    }

    const int n = static_cast<int>(A.rows());
    if (n == 0) return Matrix::Zero(0, 0);

    // Use the Schur-based Lyapunov solver for robustness and reuse
    if (type == GramianType::Observability) {
        // Observability Gramian Q solves: A^T*Q + Q*A + C^T*C = 0
        Matrix CtC = C.transpose() * C;
        Matrix Q   = lyap(A.transpose(), CtC);
        return (Q + Q.transpose()) * 0.5;
    } else {  // Controllability
        // Controllability Gramian P solves: A*P + P*A^T + B*B^T = 0
        Matrix BBt = B * B.transpose();
        Matrix P   = lyap(A, BBt);
        return (P + P.transpose()) * 0.5;
    }
}

/**
 * @brief Extract a SISO transfer function from a MIMO StateSpace system.
 *
 * For MIMO systems, extracts the transfer function from a specific input to a specific output.
 * This creates a SISO subsystem: G_ij(s) = C_i(sI-A)^(-1)B_j + D_ij
 * where i is the output index and j is the input index (0-based).
 *
 * @param output_idx   Output index (0-based, must be < number of outputs)
 * @param input_idx    Input index (0-based, must be < number of inputs)
 * @return TransferFunction  Transfer function from input_idx to output_idx
 * @throws std::out_of_range if indices are out of bounds
 */
TransferFunction StateSpace::toTransferFunction(int output_idx, int input_idx) const {
    // Validate indices
    int num_outputs = C.rows();
    int num_inputs  = B.cols();

    if (output_idx < 0 || output_idx >= num_outputs) {
        throw std::out_of_range("Output index " + std::to_string(output_idx) +
                                " is out of range [0, " + std::to_string(num_outputs - 1) + "]");
    }

    if (input_idx < 0 || input_idx >= num_inputs) {
        throw std::out_of_range("Input index " + std::to_string(input_idx) +
                                " is out of range [0, " + std::to_string(num_inputs - 1) + "]");
    }

    int n = A.rows();  // State dimension

    // Extract SISO subsystem: C_row(i), B_col(j), D(i,j)
    RowVec C_i  = C.row(output_idx);
    ColVec B_j  = B.col(input_idx);
    double D_ij = D(output_idx, input_idx);

    // For very simple cases, handle directly
    if (n == 0) {
        // Pure gain system (no states)
        return TransferFunction({D_ij}, {1.0}, Ts);
    }

    // Compute transfer function using Faddeev-LeVerrier algorithm
    // This is numerically stable for high-order systems

    // Step 1: Compute characteristic polynomial using Faddeev-LeVerrier
    // det(sI - A) = s^n + a_{n-1}s^{n-1} + ... + a_0
    std::vector<double> p(n + 1, 0.0);  // p[0] = 0
    std::vector<Matrix> H(n + 1);
    H[0] = Matrix::Identity(n, n);

    for (int k = 1; k <= n; ++k) {
        H[k] = A * H[k - 1];
        p[k] = H[k].trace() / static_cast<double>(k);
    }

    // Build denominator polynomial: s^n + a_{n-1}s^{n-1} + ... + a_0
    std::vector<double> den(n + 1);
    den[0] = 1.0;  // Leading coefficient s^n
    for (int i = 1; i <= n; ++i) {
        den[i] = (i % 2 == 1 ? -p[i] : p[i]);
    }

    // Step 2: Compute numerator coefficients
    // For the transfer function G(s) = [C*adj(sI-A)*B + D*det(sI-A)] / det(sI-A)
    std::vector<double> num(n + 1, 0.0);

    // Add D*det(sI-A) term
    for (int i = 0; i <= n; ++i) {
        num[i] += D_ij * den[i];
    }

    // Add C*adj(sI-A)*B term using the Faddeev-LeVerrier matrices
    // The adjoint matrix adj(sI-A) = sum_{k=0}^{n-1} s^{n-1-k} * F_k
    // where F_k satisfy a similar recurrence
    if (n > 0) {
        std::vector<Matrix> F(n);
        F[0] = Matrix::Identity(n, n);

        for (int k = 1; k < n; ++k) {
            F[k] = A * F[k - 1] - p[k] * Matrix::Identity(n, n);
        }

        // Compute C * F_k * B for each k and add to appropriate power of s
        for (int k = 0; k < n; ++k) {
            double coeff = C_i.dot(F[k] * B_j);
            num[k + 1] += coeff;
        }
    }

    // Normalize denominator to have leading coefficient 1
    double den_leading = den[0];
    if (std::abs(den_leading) > 1e-10) {
        for (double& coeff : den) {
            coeff /= den_leading;
        }
        for (double& coeff : num) {
            coeff /= den_leading;
        }
    }

    // Remove leading zeros from numerator (but keep at least one coefficient)
    while (num.size() > 1 && std::abs(num[0]) < 1e-10) {
        num.erase(num.begin());
    }

    // Create and return transfer function
    TransferFunction result(num, den, Ts);
    return result;
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