#include "LTI.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>

#include "unsupported/Eigen/MatrixFunctions"  // IWYU pragma: keep

namespace control {

IntegrationResult Solver::evolveDiscrete(const Matrix& A, const Matrix& B, const Matrix& x, const Matrix& u) const {
    return {A * x + B * u, 0.0};
}

IntegrationResult Solver::evolveForwardEuler(const Matrix& A, const Matrix& B, const Matrix& x, const Matrix& u, double h) const {
    return {x + h * (A * x + B * u), 0.0};
}

IntegrationResult Solver::evolveBackwardEuler(const Matrix& A, const Matrix& B, const Matrix& x, const Matrix& u, double h) const {
    const auto I   = Matrix::Identity(A.rows(), A.cols());
    const auto lhs = I - h * A;
    const auto rhs = x + h * B * u;
    return {lhs.colPivHouseholderQr().solve(rhs), 0.0};
}

IntegrationResult Solver::evolveTrapezoidal(const Matrix& A, const Matrix& B, const Matrix& x, const Matrix& u, double h) const {
    const auto I   = Matrix::Identity(A.rows(), A.cols());
    const auto lhs = I - (h / 2.0) * A;
    const auto rhs = (I + (h / 2.0) * A) * x + (h / 2.0) * B * u;
    return {lhs.colPivHouseholderQr().solve(rhs), 0.0};
}

IntegrationResult Solver::evolveRK4(const Matrix& A, const Matrix& B, const Matrix& x, const Matrix& u, double h) const {
    Matrix k1 = A * x + B * u;
    Matrix k2 = A * (x + h / 2.0 * k1) + B * u;
    Matrix k3 = A * (x + h / 2.0 * k2) + B * u;
    Matrix k4 = A * (x + h * k3) + B * u;
    return {x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 0.0};
}

IntegrationResult Solver::evolveRK45(const Matrix& A, const Matrix& B, const Matrix& x, const Matrix& u, double h) const {
    auto f = [&](const Matrix& x) { return A * x + B * u; };

    Matrix k1 = f(x);
    Matrix k2 = f(x + h * (1.0 / 4.0) * k1);
    Matrix k3 = f(x + h * (3.0 / 32.0 * k1 + 9.0 / 32.0 * k2));
    Matrix k4 = f(x + h * (1932.0 / 2197.0 * k1 - 7200.0 / 2197.0 * k2 + 7296.0 / 2197.0 * k3));
    Matrix k5 = f(x + h * (439.0 / 216.0 * k1 - 8.0 * k2 + 3680.0 / 513.0 * k3 - 845.0 / 4104.0 * k4));
    Matrix k6 = f(x + h * (-8.0 / 27.0 * k1 + 2.0 * k2 - 3544.0 / 2565.0 * k3 + 1859.0 / 4104.0 * k4 - 11.0 / 40.0 * k5));

    Matrix x4 = x + h * (25.0 / 216.0 * k1 + 1408.0 / 2565.0 * k3 + 2197.0 / 4104.0 * k4 - 1.0 / 5.0 * k5);
    Matrix x5 = x + h * (16.0 / 135.0 * k1 + 6656.0 / 12825.0 * k3 + 28561.0 / 56430.0 * k4 - 9.0 / 50.0 * k5 + 2.0 / 55.0 * k6);

    double error = (x5 - x4).norm();
    return {x5, error};
}

IntegrationResult Solver::evolveFixedTimestep(const Matrix& A, const Matrix& B, const Matrix& x, const Matrix& u, double h) const {
    switch (method_) {
        case IntegrationMethod::ForwardEuler:
            return evolveForwardEuler(A, B, x, u, h);
        case IntegrationMethod::BackwardEuler:
            return evolveBackwardEuler(A, B, x, u, h);
        case IntegrationMethod::Trapezoidal:
            return evolveTrapezoidal(A, B, x, u, h);
        case IntegrationMethod::RK4:
            return evolveRK4(A, B, x, u, h);
        case IntegrationMethod::RK45:
            return evolveRK45(A, B, x, u, h);
        default:
            return {x, 0.0};  // No change
    }
}

AdaptiveIntegrationResult Solver::evolveAdaptiveTimestep(const Matrix& A, const Matrix& B, const Matrix& x, const Matrix& u, double hInitial, double tol) {
    double h        = h_prev_;
    Matrix xCurrent = x;
    while (true) {
        auto res1 = evolveFixedTimestep(A, B, xCurrent, u, h);
        auto res2 = evolveFixedTimestep(A, B, xCurrent, u, h / 2.0);
        res2      = evolveFixedTimestep(A, B, res2.x, u, h / 2.0);

        double error = (res1.x - res2.x).norm();

        if (error < tol) {
            // Accept step
            h_prev_ = h * 1.5;
            return {.result = res2, .hNext = h_prev_};
        } else {
            // Reduce step size
            h *= 0.5;
            if (h < 1e-10) {
                // Prevent excessively small step sizes
                h = 1e-10;
            }
        }
    }
}

DiscreteStateSpace ContinuousStateSpace::discretize(double Ts, DiscretizationMethod method, std::optional<double> prewarp) const {
    const auto I    = decltype(A)::Identity(A.rows(), A.cols());
    const auto E    = (A * Ts).exp();
    const auto Ainv = A.inverse();
    const auto I1   = Ainv * (E - I);
    const auto I2   = Ainv * (E * Ts - I1);

    switch (method) {
        case DiscretizationMethod::ZOH: {
            return DiscreteStateSpace{
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
            return DiscreteStateSpace{
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
            return DiscreteStateSpace{
                Q * (k * I + A),  // A
                (I + A) * Q * B,  // B
                C,                // C
                C * Q * B + D,    // D
                Ts                // Ts
            };
        }
        default:
            // Default to ZOH
            return DiscreteStateSpace{
                E,       // A
                I1 * B,  // B
                C,       // C
                D,       // D
                Ts       // Ts
            };
    }
}

FrequencyResponse ContinuousStateSpace::bodeImpl(double fStart, double fEnd, size_t numFreq) const {
    auto response = FrequencyResponse{
        .freq      = std::vector<double>(numFreq),
        .magnitude = std::vector<double>(numFreq),
        .phase     = std::vector<double>(numFreq)};

    // Generate logarithmically spaced frequencies
    const auto [logStart, logEnd] = std::tuple{std::log10(fStart), std::log10(fEnd)};
    const auto logStep            = (logEnd - logStart) / (numFreq - 1);

    for (size_t i = 0; i < numFreq; ++i) {
        response.freq[i] = std::pow(10.0, logStart + i * logStep);     // Hz
        const auto w     = 2.0 * std::numbers::pi * response.freq[i];  // Convert to rad/s for calculations

        const auto s = std::complex<double>(0, w);  // s = jω

        // Calculate transfer function H(s) = C(sI - A)^(-1)B + D
        const auto I = control::Matrix::Identity(A.rows(), A.cols());
        const auto H = (C * (((s * I) - A).inverse()) * B) + D;

        // Store magnitude in dB and phase in degrees
        response.magnitude[i] = 20.0 * std::log10(std::abs(H(0, 0)));
        response.phase[i]     = std::arg(H(0, 0)) * 180.0 / std::numbers::pi;
    }

    return response;
}

FrequencyResponse DiscreteStateSpace::bodeImpl(double fStart, double fEnd, size_t numFreq) const {
    auto response = FrequencyResponse{
        .freq      = std::vector<double>(numFreq),
        .magnitude = std::vector<double>(numFreq),
        .phase     = std::vector<double>(numFreq)};

    // Generate logarithmically spaced frequencies
    const auto [logStart, logEnd] = std::tuple{std::log10(fStart), std::log10(fEnd)};
    const auto logStep            = (logEnd - logStart) / (numFreq - 1);

    for (size_t i = 0; i < numFreq; ++i) {
        response.freq[i] = std::pow(10.0, logStart + (i * logStep));  // Hz

        // Discrete system: z = exp(j * 2π * f * Ts)
        double omega_d = 2.0 * std::numbers::pi * response.freq[i] * Ts;
        auto   z       = std::exp(std::complex<double>(0, omega_d));

        // Calculate transfer function H(z) = C(zI - A)^(-1)B + D
        const auto I = control::Matrix::Identity(A.rows(), A.cols());
        const auto H = (C * ((z * I) - A).inverse() * B) + D;

        // Store magnitude in dB and phase in degrees
        response.magnitude[i] = 20.0 * std::log10(std::abs(H(0, 0)));
        response.phase[i]     = std::arg(H(0, 0)) * 180.0 / std::numbers::pi;
    }

    return response;
}

StepResponse ContinuousStateSpace::stepImpl(double tStart, double tEnd, Matrix uStep) const {
    Solver solver{IntegrationMethod::RK45, std::nullopt};

    const double dt        = solver.getTimestep().value_or(0.01);  // Use configured timestep or default 10ms steps
    size_t       numPoints = static_cast<size_t>((tEnd - tStart) / dt) + 1;

    StepResponse response;
    Matrix       x = Matrix::Zero(A.rows(), 1);  // Start from zero initial conditions
    if (!solver.getTimestep().has_value()) {
        // Adaptive timestepping for RK45 when no fixed timestep is set
        std::vector<AdaptiveIntegrationResult> integration_data;

        double time = tStart;
        while (time < tEnd) {
            auto res = solver.evolveAdaptiveTimestep(A, B, x, uStep, dt, 1e-6);
            x        = res.result.x;

            time += res.hNext;
            response.time.push_back(time);
            response.output.push_back(output(x, uStep)(0, 0));
        }

    } else {
        // Fixed timestep if set
        for (size_t i = 0; i < numPoints; ++i) {
            response.time.push_back(tStart + i * dt);
            response.output.push_back(output(x, uStep)(0, 0));

            x = solver.evolveFixedTimestep(A, B, x, uStep, dt).x;
        }
    }
    return response;
}

StepResponse DiscreteStateSpace::stepImpl(double tStart, double tEnd, Matrix uStep) const {
    Solver solver{IntegrationMethod::RK45, Ts};

    // Number of discrete time points
    size_t numPoints = static_cast<size_t>((tEnd - tStart) / Ts) + 1;

    StepResponse response;
    response.time.resize(numPoints);
    response.output.resize(numPoints);
    Matrix x = Matrix::Zero(A.rows(), 1);  // Start from zero initial conditions
    for (size_t i = 0; i < numPoints; ++i) {
        response.time[i]   = tStart + i * Ts;
        response.output[i] = (C * x + D * uStep)(0, 0);
        auto res           = solver.evolveDiscrete(A, B, x, uStep);
        x                  = res.x;
    }
    return response;
}

// Explicit instantiations
template class StateSpaceBase<ContinuousStateSpace>;
template class StateSpaceBase<DiscreteStateSpace>;

}  // namespace control