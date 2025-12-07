#include "LTI.hpp"

#include <cassert>
#include "unsupported/Eigen/MatrixFunctions"

namespace control {

// Integration methods for evolving the state
Matrix ContinuousStateSpace::evolveForwardEuler(const Matrix &x, const Matrix &u, double h) const {
    return x + h * (A * x + B * u);
}

Matrix ContinuousStateSpace::evolveBackwardEuler(const Matrix &x, const Matrix &u, double h) const {
    const auto I   = Matrix::Identity(A.rows(), A.cols());
    const auto lhs = I - h * A;
    const auto rhs = x + h * B * u;
    return lhs.colPivHouseholderQr().solve(rhs);
}

Matrix ContinuousStateSpace::evolveTrapezoidal(const Matrix &x, const Matrix &u, double h) const {
    const auto I   = Matrix::Identity(A.rows(), A.cols());
    const auto lhs = I - (h / 2.0) * A;
    const auto rhs = (I + (h / 2.0) * A) * x + (h / 2.0) * B * u;
    return lhs.colPivHouseholderQr().solve(rhs);
}

Matrix ContinuousStateSpace::evolve(const Matrix &x, const Matrix &u, double h, IntegrationMethod method) const {
    switch (method) {
    case IntegrationMethod::ForwardEuler:
        return evolveForwardEuler(x, u, h);
    case IntegrationMethod::BackwardEuler:
        return evolveBackwardEuler(x, u, h);
    case IntegrationMethod::Trapezoidal:
        return evolveTrapezoidal(x, u, h);
    default:
        return x; // No change
    }
}

DiscreteStateSpace ContinuousStateSpace::c2d(double Ts, Method method, std::optional<double> prewarp) const {
    const auto I    = decltype(A)::Identity(A.rows(), A.cols());
    const auto E    = (A * Ts).exp();
    const auto Ainv = A.inverse();
    const auto I1   = Ainv * (E - I);
    const auto I2   = Ainv * (E * Ts - I1);

    switch (method) {
    case Method::ZOH: {
        return DiscreteStateSpace{
            E,      // A
            I1 * B, // B
            C,      // C
            D,      // D
            Ts      // Ts
        };
    }
    case Method::FOH: {
        const auto Q = I1 - (I2 / Ts);
        const auto P = I1 - Q;
        return DiscreteStateSpace{
            E,                 // A
            (P + (E * Q)) * B, // B
            C,                 // C
            C * Q * B + D,     // D
            Ts                 // Ts
        };
    }
    case Method::Tustin: // Fallthrough
    case Method::Bilinear: {
        double k = 2.0 / Ts;
        if (prewarp.has_value()) {
            k = prewarp.value() / std::tan(prewarp.value() * Ts / 2.0);
        }

        const auto Q = (k * I - A).inverse();
        return DiscreteStateSpace{
            Q * (k * I + A), // A
            (I + A) * Q * B, // B
            C,               // C
            C * Q * B + D,   // D
            Ts               // Ts
        };
    }
    default:
        // Default to ZOH
        return DiscreteStateSpace{
            E,      // A
            I1 * B, // B
            C,      // C
            D,      // D
            Ts      // Ts
        };
    }
}

} // namespace control