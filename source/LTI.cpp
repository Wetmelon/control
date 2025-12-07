#include "LTI.hpp"

#include "unsupported/Eigen/MatrixFunctions"

namespace control {
StateSpace StateSpace::c2d(const double Ts, const Method method, const std::optional<double> prewarp) const {
    if (this->Ts.has_value()) {
        return *this;
    }

    const auto I    = decltype(A)::Identity(A.rows(), A.cols());
    const auto E    = (A * Ts).exp();
    const auto Ainv = A.inverse();
    const auto I1   = Ainv * (E - I);
    const auto I2   = Ainv * (E * Ts - I1);

    switch (method) {
        case Method::ZOH: {
            return StateSpace{
                E,       // A
                I1 * B,  // B
                C,       // C
                D,       // D
                Ts,      // Ts
            };
        }
        case Method::FOH: {
            const auto Q = I1 - (I2 / Ts);
            const auto P = I1 - Q;
            return StateSpace{
                E,                  // A
                (P + (E * Q)) * B,  // B
                C,                  // C
                C * Q * B + D,      // D
                Ts,                 // Ts
            };
        }
        case Method::Tustin:  // Fallthrough
        case Method::Bilinear: {
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
                Ts,               // Ts
            };
        }
        default:
            return *this;
    }
}

}  // namespace control