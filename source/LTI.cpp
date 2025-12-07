#include "LTI.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>

#include "unsupported/Eigen/MatrixFunctions"  //IWYU pragma: keep

namespace control {

// Integration methods for evolving the state
IntegrationResult ContinuousStateSpace::evolveForwardEuler(const Matrix& x, const Matrix& u, double h) const {
    return {x + h * (A * x + B * u), 0.0};
}

IntegrationResult ContinuousStateSpace::evolveBackwardEuler(const Matrix& x, const Matrix& u, double h) const {
    const auto I   = Matrix::Identity(A.rows(), A.cols());
    const auto lhs = I - h * A;
    const auto rhs = x + h * B * u;
    return {lhs.colPivHouseholderQr().solve(rhs), 0.0};
}

IntegrationResult ContinuousStateSpace::evolveTrapezoidal(const Matrix& x, const Matrix& u, double h) const {
    const auto I   = Matrix::Identity(A.rows(), A.cols());
    const auto lhs = I - (h / 2.0) * A;
    const auto rhs = (I + (h / 2.0) * A) * x + (h / 2.0) * B * u;
    return {lhs.colPivHouseholderQr().solve(rhs), 0.0};
}

IntegrationResult ContinuousStateSpace::evolveRK4(const Matrix& x, const Matrix& u, double h) const {
    Matrix k1 = A * x + B * u;
    Matrix k2 = A * (x + h / 2.0 * k1) + B * u;
    Matrix k3 = A * (x + h / 2.0 * k2) + B * u;
    Matrix k4 = A * (x + h * k3) + B * u;
    return {x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 0.0};
}

IntegrationResult ContinuousStateSpace::evolveRK45(const Matrix& x, const Matrix& u, double h) const {
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

IntegrationResult ContinuousStateSpace::evolve(const Matrix& x, const Matrix& u, double h, IntegrationMethod method) const {
    switch (method) {
        case IntegrationMethod::ForwardEuler:
            return evolveForwardEuler(x, u, h);
        case IntegrationMethod::BackwardEuler:
            return evolveBackwardEuler(x, u, h);
        case IntegrationMethod::Trapezoidal:
            return evolveTrapezoidal(x, u, h);
        case IntegrationMethod::RK4:
            return evolveRK4(x, u, h);
        case IntegrationMethod::RK45:
            return evolveRK45(x, u, h);
        default:
            return {x, 0.0};  // No change
    }
}

DiscreteStateSpace ContinuousStateSpace::c2d(double Ts, DiscretizationMethod method, std::optional<double> prewarp) const {
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

template <typename Derived>
FrequencyResponse StateSpaceBase<Derived>::generateFrequencyResponse(double fStart, double fEnd, int numFreq) const {
    auto response = FrequencyResponse{
        .freq      = std::vector<double>(numFreq),
        .magnitude = std::vector<double>(numFreq),
        .phase     = std::vector<double>(numFreq)};

    // Generate logarithmically spaced frequencies
    const auto [logStart, logEnd] = std::tuple{std::log10(fStart), std::log10(fEnd)};
    const auto logStep            = (logEnd - logStart) / (numFreq - 1);

    for (int i = 0; i < numFreq; ++i) {
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

template <typename Derived>
StepResponse StateSpaceBase<Derived>::generateStepResponse(double tStart, double tEnd, Matrix x0, Matrix uStep, IntegrationMethod method) const {
    if (x0.size() == 0) x0 = Matrix::Zero(A.rows(), 1);
    if (uStep.size() == 0) uStep = Matrix::Ones(B.cols(), 1);

    return static_cast<const Derived*>(this)->generateStepResponseImpl(tStart, tEnd, x0, uStep, method);
}

StepResponse DiscreteStateSpace::generateStepResponseImpl(double tStart, double tEnd, Matrix x0, Matrix uStep, IntegrationMethod /*method*/) const {
    StepResponse response;

    const size_t actualNumPoints = static_cast<size_t>((tEnd - tStart) / Ts) + 1ULL;
    response.time.resize(actualNumPoints);
    response.output.resize(actualNumPoints);
    Matrix x = x0;
    for (size_t i = 0; i < actualNumPoints; ++i) {
        response.time[i]   = tStart + i * Ts;
        response.output[i] = output(x, uStep)(0, 0);
        x                  = step(x, uStep);
    }
    return response;
}

StepResponse ContinuousStateSpace::generateStepResponseImpl(double tStart, double tEnd, Matrix x0, Matrix uStep, IntegrationMethod method) const {
    StepResponse response;
    const double dt = 0.01;  // 10ms steps

    size_t numPoints = static_cast<size_t>((tEnd - tStart) / dt) + 1;
    response.time.resize(numPoints);
    response.output.resize(numPoints);
    Matrix x = x0;
    if (method == IntegrationMethod::RK45) {
        // Adaptive timestepping for RK45
        const double tol = 1e-6;
        double       h   = 0.01;
        double       t   = tStart;
        for (size_t i = 0; i < numPoints; ++i) {
            double target_t = tStart + i * dt;
            while (t < target_t) {
                double remaining = target_t - t;
                double step_h    = std::min(h, remaining);

                const auto res = evolve(x, uStep, step_h, IntegrationMethod::RK45);

                double error  = res.error;
                double factor = std::pow(tol / (error + 1e-16), 0.2);
                if (error <= tol) {
                    x = res.x;
                    t += step_h;
                    h = std::clamp(h * factor, 1e-6, 1.0);
                } else {
                    h = std::clamp(h * std::max(0.1, factor), 1e-6, 1.0);
                }
            }
            response.time[i]   = target_t;
            response.output[i] = output(x, uStep)(0, 0);
        }
    } else {
        // Fixed timestep for other methods
        for (size_t i = 0; i < numPoints; ++i) {
            response.time[i]   = tStart + i * dt;
            response.output[i] = output(x, uStep)(0, 0);
            auto res           = evolve(x, uStep, dt, method);
            x                  = res.x;
        }
    }
    return response;
}

// Explicit instantiations
template class StateSpaceBase<ContinuousStateSpace>;
template class StateSpaceBase<DiscreteStateSpace>;

}  // namespace control