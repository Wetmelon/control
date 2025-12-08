#include "LTI.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>

#include "solver.hpp"
#include "unsupported/Eigen/MatrixFunctions"  // IWYU pragma: keep

namespace control {

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
        std::vector<AdaptiveStepResult> integration_data;

        double time = tStart;
        while (time < tEnd) {
            auto res = solver.evolveAdaptiveTimestep(A, B, x, uStep, dt, 1e-6);
            x        = res.x;

            time += res.step_size;
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