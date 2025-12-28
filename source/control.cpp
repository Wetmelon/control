#include "control.hpp"

#include "LTI.hpp"
#include "ss.hpp"
#include "tf.hpp"
#include "types.hpp"
#include "zpk.hpp"

// Free Function Interface
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

bool is_stable(const StateSpace& sys) {
    Eigen::EigenSolver<Matrix> solver(sys.A);
    const auto&                eigenvalues = solver.eigenvalues();

    if (sys.Ts.has_value()) {
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

std::vector<Pole> poles(const StateSpace& sys) {
    const auto& eigenvalues = sys.A.eigenvalues();

    std::vector<Pole> poles;
    poles.reserve(eigenvalues.size());
    for (int i = 0; i < eigenvalues.size(); ++i) {
        poles.push_back(eigenvalues[i]);
    }
    return poles;
}

std::vector<Zero> zeros(const StateSpace& sys) {
    // For state-space systems, zeros are the roots of the numerator of the transfer function
    // Convert to transfer function and compute zeros from there
    if (sys.B.cols() != 1 || sys.C.rows() != 1) {
        throw std::invalid_argument("zeros() only works for SISO systems");
    }

    auto tf_sys = sys.toTransferFunction();
    return tf_sys.zeros();
}

MarginInfo margin(const StateSpace& sys) {
    // For discrete systems, adjust frequency range to avoid aliasing
    double fStart = 1e-3;  // 0.001 Hz
    double fEnd   = 1e4;   // 10000 Hz

    if (sys.Ts.has_value()) {
        fEnd = 1.0 / (2.0 * (*sys.Ts));  // Nyquist frequency
    }

    // Compute Bode plot over appropriate frequency range
    const size_t numPoints = 1000;
    auto         bode_resp = bode(sys, fStart, fEnd, numPoints);

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

DampingInfo damp(const StateSpace& sys) {
    auto                 poles_vec     = poles(sys);
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

StepInfo stepinfo(const StateSpace& sys) {
    // Assume SISO
    if (sys.B.cols() != 1 || sys.C.rows() != 1) {
        throw std::invalid_argument("stepinfo() only works for SISO systems");
    }
    auto                step_resp  = step(sys, 0.0, 10.0, ColVec::Ones(1));
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

StepResponse step(const StateSpace& sys, double tStart, double tEnd, ColVec uStep) {
    if (uStep.size() == 0) {
        uStep = ColVec::Ones(sys.B.cols(), 1);
    }

    if (sys.Ts.has_value()) {
        // Discrete step response
        size_t numPoints = static_cast<size_t>((tEnd - tStart) / (*sys.Ts)) + 1;

        StepResponse response;
        response.time.reserve(numPoints);
        response.output.reserve(numPoints);

        ColVec x = ColVec::Zero(sys.A.rows(), 1);  // Start from zero initial conditions
        for (size_t i = 0; i < numPoints; ++i) {
            double t = tStart + i * (*sys.Ts);
            response.time.push_back(t);
            response.output.push_back(sys.C * x + sys.D * uStep);

            // Discrete-time state update: x[k+1] = A*x[k] + B*u[k]
            x = sys.A * x + sys.B * uStep;
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
        ColVec x0     = ColVec::Zero(sys.A.rows());  // Start from zero initial conditions
        auto   result = ExactSolver().solve(sys.A, sys.B, x0, uStep, {tStart, tEnd}, t_eval);

        // Convert SolveResult to StepResponse
        StepResponse response;
        response.time.reserve(result.t.size());
        response.output.reserve(result.x.size());

        for (size_t i = 0; i < result.x.size(); ++i) {
            response.time.push_back(result.t[i]);
            response.output.push_back(sys.C * result.x[i] + sys.D * uStep);
        }

        return response;
    }
}

ImpulseResponse impulse(const StateSpace& sys, double tStart, double tEnd) {
    if (sys.Ts.has_value()) {
        // Discrete impulse response
        size_t numPoints = static_cast<size_t>((tEnd - tStart) / (*sys.Ts)) + 1;

        ImpulseResponse response;
        response.time.reserve(numPoints);
        response.output.reserve(numPoints);

        ColVec x = ColVec::Zero(sys.A.rows(), 1);  // Start from zero initial conditions
        for (size_t i = 0; i < numPoints; ++i) {
            double t = tStart + i * (*sys.Ts);
            response.time.push_back(t);

            // For impulse, we apply B at the first time step only
            ColVec u_impulse;
            if (i == 0) {
                u_impulse = ColVec::Ones(sys.B.cols(), 1);  // Unit impulse
            } else {
                u_impulse = ColVec::Zero(sys.B.cols(), 1);
            }
            response.output.push_back(sys.C * x + sys.D * u_impulse);

            // Discrete-time state update: x[k+1] = A*x[k] + B*u[k]
            x = sys.A * x + sys.B * u_impulse;
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
            Matrix expAt = (sys.A * t).exp();

            // Impulse response: h(t) = C*e^(At)*B (+ D*delta(t), but we ignore the delta term)
            ColVec h = sys.C * expAt * sys.B;
            response.output.push_back(h);
        }

        return response;
    }
}

BodeResponse bode(const StateSpace& sys, double fStart, double fEnd, size_t maxPoints) {
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
    auto freq_resp = freqresp(sys, freqs);

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

NyquistResponse nyquist(const StateSpace& sys, double fStart, double fEnd, size_t maxPoints) {
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
    auto freq_resp = freqresp(sys, freqs);

    return NyquistResponse{
        .response = std::move(freq_resp.response),
        .freq     = std::move(freqs)};
}

FrequencyResponse freqresp(const StateSpace& sys, const std::vector<double>& frequencies) {
    FrequencyResponse response;
    response.freq.reserve(frequencies.size());
    response.response.reserve(frequencies.size());

    const int n = sys.A.rows();

    // Pre-compute identity matrix and zero blocks to avoid repeated allocations
    const Matrix I = Matrix::Identity(n, n);
    const Matrix Z = Matrix::Zero(n, n);

    // Pre-allocate system matrix and RHS (reused for each frequency)
    Matrix real_sys(2 * n, 2 * n);
    Matrix real_rhs(2 * n, sys.B.cols());

    // Set up the constant parts of RHS: [B; 0] since B is real
    real_rhs.block(0, 0, n, sys.B.cols()) = sys.B;
    real_rhs.block(n, 0, n, sys.B.cols()) = Matrix::Zero(n, sys.B.cols());

    // Evaluate transfer function at each frequency
    for (double freq : frequencies) {
        const double omega = 2.0 * std::numbers::pi * freq;

        // Form s = jω (continuous) or z = e^(jωTs) (discrete)
        std::complex<double> s_or_z;
        if (sys.Ts.has_value()) {
            s_or_z = std::exp(std::complex<double>(0.0, omega * (*sys.Ts)));
        } else {
            s_or_z = std::complex<double>(0.0, omega);
        }

        // Build the real block matrix for (sI - A)
        const double re_s = s_or_z.real();
        const double im_s = s_or_z.imag();

        real_sys.block(0, 0, n, n) = re_s * I - sys.A;  // Top-left block
        real_sys.block(0, n, n, n) = -im_s * I;         // Top-right block
        real_sys.block(n, 0, n, n) = im_s * I;          // Bottom-left block
        real_sys.block(n, n, n, n) = re_s * I - sys.A;  // Bottom-right block

        // Solve the 2n x 2n system
        const Matrix real_sol = real_sys.colPivHouseholderQr().solve(real_rhs);

        // Extract complex solution: x = real_sol[0:n] + j*real_sol[n:2n]
        const ColVec               re_x = real_sol.block(0, 0, n, 1);
        const ColVec               im_x = real_sol.block(n, 0, n, 1);
        const std::complex<double> y    = (sys.C * re_x)(0) + std::complex<double>(0.0, 1.0) * (sys.C * im_x)(0) + std::complex<double>(sys.D(0, 0));  // Assume SISO

        response.freq.push_back(freq);
        response.response.push_back(y);
    }

    return response;
}

RootLocusResponse rlocus(const StateSpace& sys, double kMin, double kMax, size_t numPoints) {
    // Convert to transfer function and compute root locus from there
    auto tf_sys = sys.toTransferFunction();
    return rlocus(tf_sys, kMin, kMax, numPoints);
}

// Discretize (only if continuous)
StateSpace discretize(const StateSpace& sys, double Ts, DiscretizationMethod method, std::optional<double> prewarp) {
    if (sys.Ts.has_value()) {
        throw std::runtime_error("Cannot discretize an already discrete system.");
    }

    const auto I    = decltype(sys.A)::Identity(sys.A.rows(), sys.A.cols());
    const auto E    = (sys.A * Ts).exp();
    const auto Ainv = sys.A.inverse();
    const auto I1   = Ainv * (E - I);
    const auto I2   = Ainv * (E * Ts - I1);

    switch (method) {
        case DiscretizationMethod::ZOH: {
            return StateSpace{
                E,           // A
                I1 * sys.B,  // B
                sys.C,       // C
                sys.D,       // D
                Ts           // Ts
            };
        }
        case DiscretizationMethod::FOH: {
            const auto Q = I1 - (I2 / Ts);
            const auto P = I1 - Q;
            return StateSpace{
                E,                          // A
                (P + (E * Q)) * sys.B,      // B
                sys.C,                      // C
                sys.C * Q * sys.B + sys.D,  // D
                Ts                          // Ts
            };
        }
        case DiscretizationMethod::Tustin:  // Fallthrough
        case DiscretizationMethod::Bilinear: {
            double k = 2.0 / Ts;
            if (prewarp.has_value()) {
                k = prewarp.value() / std::tan(prewarp.value() * Ts / 2.0);
            }

            const auto Q = (k * I - sys.A).inverse();
            return StateSpace{
                Q * (k * I + sys.A),        // A
                (I + sys.A) * Q * sys.B,    // B
                sys.C,                      // C
                sys.C * Q * sys.B + sys.D,  // D
                Ts                          // Ts
            };
        }
        default:
            // Default to ZOH
            return StateSpace{
                E,           // A
                I1 * sys.B,  // B
                sys.C,       // C
                sys.D,       // D
                Ts           // Ts
            };
    }
}

ObservabilityInfo observability(const StateSpace& sys) {
    // Structural observability: compute observability matrix and its rank.
    const int n = sys.A.rows();
    const int p = static_cast<int>(sys.C.rows());
    if (n == 0) {
        return ObservabilityInfo{.rank = 0, .isObservable = true};
    }

    Matrix Ob = Matrix::Zero(p * n, n);
    Matrix Ak = Matrix::Identity(n, n);
    for (int k = 0; k < n; ++k) {
        Ob.block(k * p, 0, p, n) = sys.C * Ak;
        Ak                       = sys.A * Ak;
    }

    Eigen::ColPivHouseholderQR<Matrix> qr(Ob);

    size_t rank         = qr.rank();
    bool   isObservable = (rank == static_cast<size_t>(n));

    return ObservabilityInfo{.rank = rank, .isObservable = isObservable};
}

ControllabilityInfo controllability(const StateSpace& sys) {
    // Structural controllability: compute controllability matrix and its rank.
    const int n = sys.A.rows();
    const int m = static_cast<int>(sys.B.cols());
    if (n == 0) {
        return ControllabilityInfo{.rank = 0, .isControllable = true};
    }

    Matrix Ctrb = Matrix::Zero(n, n * std::max(1, m));
    Matrix Ak2  = Matrix::Identity(n, n);
    for (int k = 0; k < n; ++k) {
        Ctrb.block(0, k * m, n, m) = Ak2 * sys.B;
        Ak2                        = sys.A * Ak2;
    }

    Eigen::ColPivHouseholderQR<Matrix> qr(Ctrb);

    size_t rank           = qr.rank();
    bool   isControllable = (rank == static_cast<size_t>(n));

    // Do not compute the gramian here; return empty (zero) gramian to avoid expensive or unstable ops.
    return ControllabilityInfo{.rank = rank, .isControllable = isControllable};
}

// Compute Gramian matrices for continuous-time systems using iterative series
Matrix gramian(const StateSpace& sys, GramianType type) {
    if (sys.isDiscrete()) {
        throw std::runtime_error("gramian: discrete-time systems are not yet supported");
    }

    const int n = static_cast<int>(sys.A.rows());
    if (n == 0) return Matrix::Zero(0, 0);

    // Use the Schur-based Lyapunov solver for robustness and reuse
    if (type == GramianType::Observability) {
        // Observability Gramian Q solves: A^T*Q + Q*A + C^T*C = 0
        Matrix CtC = sys.C.transpose() * sys.C;
        Matrix Q   = lyap(sys.A.transpose(), CtC);
        return (Q + Q.transpose()) * 0.5;
    } else {  // Controllability
        // Controllability Gramian P solves: A*P + P*A^T + B*B^T = 0
        Matrix BBt = sys.B * sys.B.transpose();
        Matrix P   = lyap(sys.A, BBt);
        return (P + P.transpose()) * 0.5;
    }
}

StateSpace minreal(const StateSpace& sys, double tol) {
    if (sys.isDiscrete()) {
        throw std::runtime_error("minreal: discrete-time systems are not yet supported");
    }

    const int n = static_cast<int>(sys.A.rows());
    if (n == 0) {
        return sys;  // nothing to do
    }

    // Helper: compute orthonormal basis for column space of M using SVD
    auto colspace_basis = [&](const Matrix& M, double atol) -> Matrix {
        if (M.size() == 0) return Matrix::Zero(M.rows(), 0);
        Eigen::JacobiSVD<Matrix> svd(M, Eigen::ComputeThinU);
        Eigen::VectorXd          s    = svd.singularValues();
        int                      rank = 0;
        double                   smax = s.size() ? s(0) : 0.0;
        for (int i = 0; i < s.size(); ++i) {
            if (s(i) > atol * std::max(1.0, smax)) ++rank;
        }
        if (rank == 0) return Matrix::Zero(M.rows(), 0);
        return svd.matrixU().leftCols(rank);
    };

    double atol = tol;

    // Build controllability matrix [B, A*B, A^2*B, ...]
    int m = static_cast<int>(sys.B.cols());
    if (m == 0) {
        // no inputs -> no controllable states
        return StateSpace(Matrix::Zero(0, 0), Matrix::Zero(0, sys.B.cols()), Matrix::Zero(sys.C.rows(), 0), sys.D, sys.Ts);
    }
    Matrix Ctrb = Matrix::Zero(n, n * m);
    Matrix Ak   = Matrix::Identity(n, n);
    for (int k = 0; k < n; ++k) {
        Matrix block               = Ak * sys.B;  // n x m
        Ctrb.block(0, k * m, n, m) = block;
        Ak                         = sys.A * Ak;
    }

    Matrix Uc = colspace_basis(Ctrb, atol);
    int    rc = static_cast<int>(Uc.cols());

    if (rc == 0) {
        // No controllable dynamics -> return D-only
        return StateSpace(Matrix::Zero(0, 0), Matrix::Zero(0, sys.B.cols()), Matrix::Zero(sys.C.rows(), 0), sys.D, sys.Ts);
    }

    // Build orthonormal complement of Uc via SVD of (I - Uc*Uc^T)
    Matrix P       = Uc * Uc.transpose();
    Matrix I       = Matrix::Identity(n, n);
    Matrix Mcomp   = I - P;
    Matrix Uc_perp = colspace_basis(Mcomp, atol);

    // Form orthonormal transform Q = [Uc, Uc_perp]
    const int nc2 = static_cast<int>(Uc_perp.cols());
    Matrix    Q   = Matrix::Zero(n, rc + nc2);
    if (rc > 0) Q.leftCols(rc) = Uc;
    if (nc2 > 0) Q.rightCols(nc2) = Uc_perp;

    // Transform coordinates (Q is orthonormal if Uc and Uc_perp are orthonormal and orthogonal)
    Matrix Qt  = Q.transpose();
    Matrix A_t = Qt * sys.A * Q;
    Matrix B_t = Qt * sys.B;
    Matrix C_t = sys.C * Q;

    // Keep controllable block (first rc states)
    Matrix A_c = A_t.topLeftCorner(rc, rc);
    Matrix B_c = B_t.topRows(rc);
    Matrix C_c = C_t.leftCols(rc);

    // Now remove unobservable states from the controllable subsystem
    // Build observability matrix for (A_c, C_c): [C_c; C_c*A_c; ...]
    Matrix Ob  = Matrix::Zero(rc * C_c.rows(), rc);
    Matrix Ak2 = Matrix::Identity(rc, rc);
    for (int k = 0; k < rc; ++k) {
        Matrix rowblock                             = C_c * Ak2;  // p x rc
        Ob.block(k * C_c.rows(), 0, C_c.rows(), rc) = rowblock;
        Ak2                                         = A_c * Ak2;
    }

    Matrix Uo = colspace_basis(Ob.transpose(), atol);
    int    ro = static_cast<int>(Uo.cols());

    if (ro == 0) {
        // No observable states -> return D-only
        return StateSpace(Matrix::Zero(0, 0), Matrix::Zero(0, sys.B.cols()), Matrix::Zero(sys.C.rows(), 0), sys.D, sys.Ts);
    }

    // Form final transform for controllable subsystem using observable basis
    // Uo is rc x ro (basis in controllable coordinates). We need a selector S of
    // size (rc + nc2) x ro where the top rc rows are Uo and bottom nc2 rows are zero.
    int    rfinal = ro;
    Matrix S      = Matrix::Zero(rc + nc2, std::max(1, rfinal));
    if (rfinal > 0) S.topRows(rc).leftCols(rfinal) = Uo.leftCols(rfinal);

    // Build final transform from original coordinates: n x ro
    Matrix T_final = Q * S;
    Matrix Tfin_t  = T_final.transpose();

    Matrix A_f = Tfin_t * sys.A * T_final;
    Matrix B_f = Tfin_t * sys.B;
    Matrix C_f = sys.C * T_final;

    // Keep top-left ro x ro
    Matrix A_min = A_f.topLeftCorner(ro, ro);
    Matrix B_min = B_f.topRows(ro);
    Matrix C_min = C_f.leftCols(ro);

    StateSpace red(A_min, B_min, C_min, sys.D, sys.Ts);
    return red;
}

StateSpace balred(const StateSpace& sys, size_t r) {
    if (sys.isDiscrete()) {
        throw std::runtime_error("balred: discrete-time systems are not yet supported");
    }

    const int n = static_cast<int>(sys.A.rows());
    if (n == 0 || r == 0) {
        // Truncate to a pure D matrix
        return StateSpace(Matrix::Zero(0, 0), Matrix::Zero(0, sys.B.cols()), Matrix::Zero(sys.C.rows(), 0), sys.D, sys.Ts);
    }
    if (r >= static_cast<size_t>(n)) {
        return sys;  // nothing to do
    }

    // Compute controllability and observability gramians
    Matrix BBt = sys.B * sys.B.transpose();
    Matrix CtC = sys.C.transpose() * sys.C;

    Matrix P = lyap(sys.A, BBt);
    Matrix Q = lyap(sys.A.transpose(), CtC);

    // Ensure symmetry
    P = (P + P.transpose()) * 0.5;
    Q = (Q + Q.transpose()) * 0.5;

    // Cholesky (fallback to eigen if not positive definite)
    Eigen::LLT<Matrix> lltP(P);
    Eigen::LLT<Matrix> lltQ(Q);
    Matrix             Rp, Rq;
    if (lltP.info() == Eigen::Success && lltQ.info() == Eigen::Success) {
        // P = Lp * Lp^T where Lp is lower-triangular
        Rp = lltP.matrixL();
        Rq = lltQ.matrixL();
    } else {
        // Fallback: symmetric eigen decomposition to get sqrt
        Eigen::SelfAdjointEigenSolver<Matrix> esP(P);
        Eigen::SelfAdjointEigenSolver<Matrix> esQ(Q);
        if (esP.info() != Eigen::Success || esQ.info() != Eigen::Success) {
            throw std::runtime_error("balred: failed to factor gramians");
        }
        Matrix sqrtP = esP.operatorSqrt();
        Matrix sqrtQ = esQ.operatorSqrt();
        Rp           = sqrtP;
        Rq           = sqrtQ;
    }

    // Compute SVD of Rp^T * Rq
    Matrix                   M = Rp.transpose() * Rq;
    Eigen::JacobiSVD<Matrix> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Matrix                   U = svd.matrixU();
    Matrix                   V = svd.matrixV();
    Eigen::VectorXd          s = svd.singularValues();

    // Form balancing transform T and its inverse
    // Use Sigma = diag(s)
    Eigen::VectorXd sqrt_s     = s.array().sqrt();
    Eigen::VectorXd inv_sqrt_s = sqrt_s.array().inverse();

    // Compute T = Rp.inverse() * U * sqrt(S)
    // For triangular Rp, solving is better than inverting
    Matrix U_sqrt = U * sqrt_s.asDiagonal();
    Matrix T      = Rp.triangularView<Eigen::Lower>().solve(U_sqrt);

    // Compute Tinv = (inv_sqrt(S) * V.transpose()) * Rq.transpose().inverse()
    Matrix Vt        = V.transpose();
    Matrix invsqrtVt = inv_sqrt_s.asDiagonal() * Vt;
    Matrix Tinv      = Rq.triangularView<Eigen::Lower>().transpose().solve(invsqrtVt.transpose()).transpose();

    // Transform system to balanced coordinates
    Matrix A_bal = Tinv * sys.A * T;
    Matrix B_bal = Tinv * sys.B;
    Matrix C_bal = sys.C * T;

    // Partition and truncate
    int    rint = static_cast<int>(r);
    Matrix A11  = A_bal.topLeftCorner(rint, rint);
    Matrix A12  = A_bal.topRightCorner(rint, n - rint);
    Matrix A21  = A_bal.bottomLeftCorner(n - rint, rint);
    Matrix A22  = A_bal.bottomRightCorner(n - rint, n - rint);

    Matrix B1 = B_bal.topRows(rint);
    Matrix B2 = B_bal.bottomRows(n - rint);

    Matrix C1 = C_bal.leftCols(rint);
    Matrix C2 = C_bal.rightCols(n - rint);

    // Truncated reduced-order model
    StateSpace red(A11, B1, C1, sys.D, sys.Ts);
    return red;
}

// TransferFunction specific
std::vector<Pole> poles(const TransferFunction& sys) {
    // Poles are the roots of the denominator polynomial
    const int n = static_cast<int>(sys.den.size()) - 1;  // Order of denominator

    if (n <= 0) {
        // Constant denominator (no poles) or zero denominator
        return std::vector<Pole>();
    }

    // Check if leading coefficient is zero - this is an error
    if (std::abs(sys.den[0]) < 1e-15) {
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
        companion(0, i) = -sys.den[i + 1] / sys.den[0];
    }

    // Fill subdiagonal with 1s
    for (int i = 1; i < n; ++i) {
        companion(i, i - 1) = 1.0;
    }

    // Compute eigenvalues (these are the poles)
    const auto& eigenvalues = companion.eigenvalues();
    return std::vector<Pole>(eigenvalues.data(), eigenvalues.data() + eigenvalues.size());
}

RootLocusResponse rlocus(const TransferFunction& sys, double kMin, double kMax, size_t numPoints) {
    // Simple root locus implementation
    // For each gain k, solve den(s) + k*num(s) = 0
    // This is a placeholder - a full implementation would use more sophisticated methods

    RootLocusResponse response;
    response.gains.reserve(numPoints);
    response.branches.reserve(sys.den.size() - 1);  // Number of poles

    // Initialize branches
    auto poles_at_zero = poles(sys);
    for (size_t i = 0; i < poles_at_zero.size(); ++i) {
        response.branches.emplace_back();
        response.branches.back().reserve(numPoints);
    }

    double kStep = (kMax - kMin) / (numPoints - 1);
    for (size_t i = 0; i < numPoints; ++i) {
        double k = kMin + i * kStep;
        response.gains.push_back(k);

        // For simplicity, assume the poles don't move much, just add k to the real part or something
        // A proper implementation would solve the polynomial equation
        for (size_t j = 0; j < response.branches.size(); ++j) {
            // Placeholder: just shift the pole
            std::complex<double> pole = poles_at_zero[j] - k * 0.1;  // Dummy shift
            response.branches[j].push_back(pole);
        }
    }

    return response;
}

// ZeroPoleGain specific
bool is_stable(const ZeroPoleGain& sys) {
    if (sys.Ts.has_value()) {
        // Discrete: unstable if any |pole| >= 1
        for (const auto& p : sys.poles_) {
            if (std::abs(p) >= 1.0) {
                return false;
            }
        }
    } else {
        // Continuous: unstable if any Re(pole) >= 0
        for (const auto& p : sys.poles_) {
            if (p.real() >= 0.0) {
                return false;
            }
        }
    }
    return true;
}

// ============================================================================
// LTI System Arithmetic Operations
// ============================================================================
// These operators enable combining LTI systems to create complex control
// systems from simple building blocks (controllers, plants, sensors).
//
// Usage Examples:
//   auto open_loop    = controller * plant;          // Series connection
//   auto parallel_sys = sys1 + sys2;                 // Parallel (sum)
//   auto error_sys    = reference - measurement;     // Parallel (difference)
//   auto closed_loop  = feedback(fwd_path, fb_path); // Negative feedback
//   auto closed_loop  = fwd_path / fb_path;          // Negative feedback (same as above)
//
// Control System Construction:
//   1. Create individual components (Controller C, Plant G, Sensor H)
//   2. Combine them: T = feedback(C * G, H)
//   3. This creates closed-loop: T(s) = C*G / (1 + C*G*H)
// ============================================================================

StateSpace series(const StateSpace& sys1, const StateSpace& sys2) {
    if (sys1.systemType() != sys2.systemType()) {
        throw std::runtime_error("Cannot combine continuous and discrete systems. Use discretize() or c2d() first.");
    }

    if (sys1.Ts != sys2.Ts) {
        throw std::runtime_error("Sampling times do not match for discrete systems.");
    }

    // Series connection: sys2 follows sys1 (sys1 -> sys2)
    // State space representation:
    // x = [x1; x2]
    // A = [A1,    0  ]    B = [B1]
    //     [B2*C1, A2 ]        [B2*D1]
    // C = [D2*C1, C2]    D = [D2*D1]

    const int n1 = sys1.A.rows();
    const int n2 = sys2.A.rows();
    const int m  = sys1.B.cols();
    const int p  = sys2.C.rows();

    Matrix A = Matrix::Zero(n1 + n2, n1 + n2);
    Matrix B = Matrix::Zero(n1 + n2, m);
    Matrix C = Matrix::Zero(p, n1 + n2);
    Matrix D = Matrix::Zero(p, m);

    // Fill A matrix
    A.block(0, 0, n1, n1)   = sys1.A;
    A.block(n1, 0, n2, n1)  = sys2.B * sys1.C;
    A.block(n1, n1, n2, n2) = sys2.A;

    // Fill B matrix
    B.block(0, 0, n1, m)  = sys1.B;
    B.block(n1, 0, n2, m) = sys2.B * sys1.D;

    // Fill C matrix
    C.block(0, 0, p, n1)  = sys2.D * sys1.C;
    C.block(0, n1, p, n2) = sys2.C;

    // Fill D matrix
    D = sys2.D * sys1.D;

    return StateSpace{std::move(A), std::move(B), std::move(C), std::move(D), sys1.Ts};
}

TransferFunction series(const TransferFunction& sys1, const TransferFunction& sys2) {
    if (sys1.systemType() != sys2.systemType()) {
        throw std::runtime_error("Cannot combine continuous and discrete systems. Use discretize() or c2d() first.");
    }
    if (sys1.Ts != sys2.Ts) {
        throw std::runtime_error("Sampling times do not match for discrete systems.");
    }

    // Numerical-robust path: convert to StateSpace, perform series, reduce, convert back
    StateSpace ss1 = sys1.toStateSpace();
    StateSpace ss2 = sys2.toStateSpace();
    StateSpace ssr = series(ss1, ss2);
    return ssr.toTransferFunction();
}

ZeroPoleGain series(const ZeroPoleGain& sys1, const ZeroPoleGain& sys2) {
    if (sys1.systemType() != sys2.systemType()) {
        throw std::runtime_error("Cannot combine continuous and discrete systems. Use discretize() or c2d() first.");
    }
    if (sys1.Ts != sys2.Ts) {
        throw std::runtime_error("Sampling times do not match for discrete systems.");
    }

    // Convert to TF, multiply, convert back
    StateSpace ss1       = sys1.toStateSpace();
    StateSpace ss2       = sys2.toStateSpace();
    StateSpace result_ss = series(ss1, ss2);
    return result_ss.toZeroPoleGain();
}

StateSpace operator*(const StateSpace& sys1, const StateSpace& sys2) {
    return series(sys1, sys2);
}

TransferFunction operator*(const TransferFunction& sys1, const TransferFunction& sys2) {
    return series(sys1, sys2);
}

ZeroPoleGain operator*(const ZeroPoleGain& sys1, const ZeroPoleGain& sys2) {
    return series(sys1, sys2);
}

/*  Parallel Connections */
StateSpace parallel(const StateSpace& sys1, const StateSpace& sys2) {
    StateSpace ss1 = sys1.toStateSpace();
    StateSpace ss2 = sys2.toStateSpace();

    if (ss1.systemType() != ss2.systemType()) {
        throw std::runtime_error("Cannot combine continuous and discrete systems. Use discretize() or c2d() first.");
    }

    if (ss1.Ts != ss2.Ts) {
        throw std::runtime_error("Sampling times do not match for discrete systems.");
    }

    // Parallel connection: outputs are added
    // State space representation:
    // x = [x1; x2]
    // A = [A1, 0 ]    B = [B1]
    //     [0,  A2]        [B2]
    // C = [C1, C2]    D = [D1 + D2]

    const int n1 = ss1.A.rows();
    const int n2 = ss2.A.rows();
    const int m  = ss1.B.cols();
    const int p  = ss1.C.rows();

    Matrix A = Matrix::Zero(n1 + n2, n1 + n2);
    Matrix B = Matrix::Zero(n1 + n2, m);
    Matrix C = Matrix::Zero(p, n1 + n2);
    Matrix D = Matrix::Zero(p, m);

    // Fill A matrix (block diagonal)
    A.block(0, 0, n1, n1)   = ss1.A;
    A.block(n1, n1, n2, n2) = ss2.A;

    // Fill B matrix (stacked)
    B.block(0, 0, n1, m)  = ss1.B;
    B.block(n1, 0, n2, m) = ss2.B;

    // Fill C matrix (concatenated)
    C.block(0, 0, p, n1)  = ss1.C;
    C.block(0, n1, p, n2) = ss2.C;

    // Fill D matrix (sum)
    D = ss1.D + ss2.D;

    return StateSpace{std::move(A), std::move(B), std::move(C), std::move(D), ss1.Ts};
}
TransferFunction parallel(const TransferFunction& sys1, const TransferFunction& sys2) {
    if (sys1.systemType() != sys2.systemType()) {
        throw std::runtime_error("Cannot combine continuous and discrete systems. Use discretize() or c2d() first.");
    }
    if (sys1.Ts != sys2.Ts) {
        throw std::runtime_error("Sampling times do not match for discrete systems.");
    }

    // Robust path: convert to SS, perform parallel, reduce, convert back
    StateSpace       ss1 = sys1.toStateSpace();
    StateSpace       ss2 = sys2.toStateSpace();
    StateSpace       ssr = parallel(ss1, ss2);
    TransferFunction tf  = ssr.toTransferFunction();
    return tf;
}

ZeroPoleGain parallel(const ZeroPoleGain& sys1, const ZeroPoleGain& sys2) {
    if (sys1.systemType() != sys2.systemType()) {
        throw std::runtime_error("Cannot combine continuous and discrete systems. Use discretize() or c2d() first.");
    }
    if (sys1.Ts != sys2.Ts) {
        throw std::runtime_error("Sampling times do not match for discrete systems.");
    }

    // Convert to TF, add, convert back
    TransferFunction tf1       = sys1.toTransferFunction();
    TransferFunction tf2       = sys2.toTransferFunction();
    TransferFunction result_tf = parallel(tf1, tf2);
    return result_tf.toZeroPoleGain();
}

StateSpace operator+(const StateSpace& sys1, const StateSpace& sys2) {
    return parallel(sys1, sys2);
}

TransferFunction operator+(const TransferFunction& sys1, const TransferFunction& sys2) {
    return parallel(sys1, sys2);
}

ZeroPoleGain operator+(const ZeroPoleGain& sys1, const ZeroPoleGain& sys2) {
    return parallel(sys1, sys2);
}

StateSpace operator-(const StateSpace& sys1, const StateSpace& sys2) {
    StateSpace neg_sys2 = sys2;
    neg_sys2.C          = -neg_sys2.C;
    neg_sys2.D          = -neg_sys2.D;

    return parallel(sys1, neg_sys2);
}

TransferFunction operator-(const TransferFunction& sys1, const TransferFunction& sys2) {
    TransferFunction neg_sys2 = sys2;
    for (double& coeff : neg_sys2.num) {
        coeff = -coeff;
    }

    return parallel(sys1, neg_sys2);
}

ZeroPoleGain operator-(const ZeroPoleGain& sys1, const ZeroPoleGain& sys2) {
    ZeroPoleGain neg_sys2 = sys2;
    neg_sys2.gain_        = -neg_sys2.gain_;

    return parallel(sys1, neg_sys2);
}

/* Feedback Connections */
StateSpace feedback(const StateSpace& sys_forward, const StateSpace& sys_feedback, int sign) {
    StateSpace G = sys_forward.toStateSpace();
    StateSpace H = sys_feedback.toStateSpace();

    if (G.systemType() != H.systemType()) {
        throw std::runtime_error("Cannot combine continuous and discrete systems. Use discretize() or c2d() first.");
    }

    if (G.Ts != H.Ts) {
        throw std::runtime_error("Sampling times do not match for discrete systems.");
    }

    // Feedback connection:
    // Closed-loop transfer function: G_cl = G / (1 - sign * G * H)
    // where G is forward path, H is feedback path
    // sign = -1 for negative feedback (default), +1 for positive feedback
    //
    // State space representation:
    // A_cl = [A_G + sign*B_G*(I - sign*D_H*D_G)^-1*D_H*C_G,  sign*B_G*(I - sign*D_H*D_G)^-1*C_H]
    //        [B_H*(I - sign*D_G*D_H)^-1*C_G,                   A_H + sign*B_H*(I - sign*D_G*D_H)^-1*D_G*C_H]
    // B_cl = [B_G*(I - sign*D_H*D_G)^-1]
    //        [sign*B_H*(I - sign*D_G*D_H)^-1*D_G]
    // C_cl = [(I - sign*D_H*D_G)^-1*C_G,  sign*(I - sign*D_H*D_G)^-1*D_H*C_H]
    // D_cl = (I - sign*D_H*D_G)^-1*D_G

    const int nG = G.A.rows();
    const int nH = H.A.rows();
    const int m  = G.B.cols();
    const int p  = G.C.rows();

    // Calculate the inverses we need
    const auto   I_p = Matrix::Identity(p, p);
    const auto   I_m = Matrix::Identity(m, m);
    const double s   = static_cast<double>(sign);

    // (I - sign*D_H*D_G)^-1
    const auto inv1 = (I_p - s * H.D * G.D).inverse();
    // (I - sign*D_G*D_H)^-1
    const auto inv2 = (I_m - s * G.D * H.D).inverse();

    Matrix A_cl = Matrix::Zero(nG + nH, nG + nH);
    Matrix B_cl = Matrix::Zero(nG + nH, m);
    Matrix C_cl = Matrix::Zero(p, nG + nH);
    Matrix D_cl = Matrix::Zero(p, m);

    // Fill A_cl matrix
    A_cl.block(0, 0, nG, nG)   = G.A + s * G.B * inv1 * H.D * G.C;
    A_cl.block(0, nG, nG, nH)  = s * G.B * inv1 * H.C;
    A_cl.block(nG, 0, nH, nG)  = H.B * inv2 * G.C;
    A_cl.block(nG, nG, nH, nH) = H.A + s * H.B * inv2 * G.D * H.C;

    // Fill B_cl matrix
    B_cl.block(0, 0, nG, m)  = G.B * inv1;
    B_cl.block(nG, 0, nH, m) = s * H.B * inv2 * G.D;

    // Fill C_cl matrix
    C_cl.block(0, 0, p, nG)  = inv1 * G.C;
    C_cl.block(0, nG, p, nH) = s * inv1 * H.D * H.C;

    // Fill D_cl matrix
    D_cl = inv1 * G.D;

    return StateSpace{std::move(A_cl), std::move(B_cl), std::move(C_cl), std::move(D_cl), G.Ts};
}

TransferFunction feedback(const TransferFunction& sys_forward, const TransferFunction& sys_feedback, int sign) {
    if (sys_forward.systemType() != sys_feedback.systemType()) {
        throw std::runtime_error("Cannot combine continuous and discrete systems. Use discretize() or c2d() first.");
    }
    if (sys_forward.Ts != sys_feedback.Ts) {
        throw std::runtime_error("Sampling times do not match for discrete systems.");
    }
    // Use numerically robust path: convert to StateSpace, perform feedback, reduce, convert back
    StateSpace       G   = sys_forward.toStateSpace();
    StateSpace       H   = sys_feedback.toStateSpace();
    StateSpace       ssr = feedback(G, H, sign);
    TransferFunction tf  = ssr.toTransferFunction();
    return tf;
}

ZeroPoleGain feedback(const ZeroPoleGain& sys_forward, const ZeroPoleGain& sys_feedback, int sign) {
    if (sys_forward.systemType() != sys_feedback.systemType()) {
        throw std::runtime_error("Cannot combine continuous and discrete systems. Use discretize() or c2d() first.");
    }
    if (sys_forward.Ts != sys_feedback.Ts) {
        throw std::runtime_error("Sampling times do not match for discrete systems.");
    }

    // Convert to TF, perform feedback, convert back
    TransferFunction G = sys_forward.toTransferFunction();
    TransferFunction H = sys_feedback.toTransferFunction();

    TransferFunction closed_loop_tf = feedback(G, H, sign);
    return closed_loop_tf.toZeroPoleGain();
}

StateSpace operator/(const StateSpace& sys_forward, const StateSpace& sys_feedback) {
    return feedback(sys_forward, sys_feedback, -1);
}

TransferFunction operator/(const TransferFunction& sys_forward, const TransferFunction& sys_feedback) {
    return feedback(sys_forward, sys_feedback, -1);
}

ZeroPoleGain operator/(const ZeroPoleGain& sys_forward, const ZeroPoleGain& sys_feedback) {
    return feedback(sys_forward, sys_feedback, -1);
}

// Pade approximation for time delays
StateSpace pade(const StateSpace& sys, double delay, int order) {
    if (order < 1) {
        throw std::invalid_argument("Pade order must be at least 1");
    }
    if (delay < 0) {
        throw std::invalid_argument("Delay must be non-negative");
    }

    // Compute Pade [order/order] coefficients for e^{-z} where z = s*delay
    std::vector<double> num(order + 1);
    std::vector<double> den(order + 1);

    for (int k = 0; k <= order; ++k) {
        double binom = std::tgamma(2 * order - k + 1) /
                       (std::tgamma(k + 1) * std::tgamma(order - k + 1) * std::tgamma(order + 1));
        num[k] = std::pow(-1.0, k) * binom;
        den[k] = binom;
    }

    // Scale by powers of delay
    std::vector<double> num_scaled(order + 1);
    std::vector<double> den_scaled(order + 1);
    for (int k = 0; k <= order; ++k) {
        num_scaled[k] = num[k] * std::pow(delay, k);
        den_scaled[k] = den[k] * std::pow(delay, k);
    }

    // Create transfer function for the delay approximation
    TransferFunction delay_tf(num_scaled, den_scaled, sys.Ts);
    StateSpace       delay_ss = delay_tf.toStateSpace();

    // Series connect with the original system
    return series(sys, delay_ss);
}

TransferFunction pade(const TransferFunction& tf, double delay, int order) {
    StateSpace ss = pade(tf.toStateSpace(), delay, order);
    return ss.toTransferFunction();
}

ZeroPoleGain pade(const ZeroPoleGain& zpk_sys, double delay, int order) {
    StateSpace ss = pade(zpk_sys.toStateSpace(), delay, order);
    return ss.toZeroPoleGain();
}

StateSpace delay(const StateSpace& sys, double delay, int order) {
    auto I = StateSpace{Matrix::Identity(0, 0), Matrix::Zero(0, 1), Matrix::Zero(1, 0), Matrix::Identity(1, 1), sys.Ts};

    if (sys.Ts.has_value()) {
        // Discrete delay: exact implementation using shift register
        int num_states = static_cast<int>(delay / *sys.Ts);
        if (num_states < 1) {
            // No delay or fractional - return identity
            return I;
        }

        // Create shift register: x(k+1) = [0, 1, 0; 0, 0, 1; ...; 0, 0, 0] * x(k) + [1; 0; 0] * u(k)
        // y(k) = [0, 0, ..., 1] * x(k)
        Matrix A = Matrix::Zero(num_states, num_states);
        for (int i = 0; i < num_states - 1; ++i) {
            A(i, i + 1) = 1.0;
        }
        Matrix B             = Matrix::Zero(num_states, 1);
        B(0, 0)              = 1.0;
        Matrix C             = Matrix::Zero(1, num_states);
        C(0, num_states - 1) = 1.0;
        Matrix D             = Matrix::Zero(1, 1);

        return StateSpace{std::move(A), std::move(B), std::move(C), std::move(D), sys.Ts};
    } else {
        // Continuous delay: use Pade approximation
        return pade(I, delay, order);
    }
}

TransferFunction delay(const TransferFunction& tf, double delay, int order) {
    StateSpace ss = control::delay(tf.toStateSpace(), delay, order);
    return ss.toTransferFunction();
}

ZeroPoleGain delay(const ZeroPoleGain& zpk_sys, double delay, int order) {
    StateSpace ss = control::delay(zpk_sys.toStateSpace(), delay, order);
    return ss.toZeroPoleGain();
}

}  // namespace control