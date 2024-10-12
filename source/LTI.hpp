#pragma once

#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <span>

#include "Eigen/Dense"
#include "src/Core/Matrix.h"
#include "unsupported/Eigen/MatrixFunctions"

namespace control {

enum class DiscretizationMethod {
    ZOH,
    FOH,
    Tustin,
};

struct LTI {};

struct TransferFunction {
    Eigen::MatrixXd num, den;
    double Ts;
};

struct StateSpace {
    Eigen::MatrixXd A, B, C, D;

    StateSpace minimal() const { return *this; }
};

static inline auto c2d(const StateSpace& sys, const double Ts, const DiscretizationMethod method = DiscretizationMethod::Tustin, std::optional<double> prewarp = std::nullopt) {
    const auto I = Eigen::MatrixXd::Identity(sys.A.rows(), sys.A.cols());
    const auto E = (sys.A * Ts).exp();
    const auto Ainv = sys.A.inverse();
    const auto I1 = Ainv * (E - I);
    const auto I2 = Ainv * (E * Ts - I1);

    if (method == DiscretizationMethod::ZOH) {
        return StateSpace{
            .A = E,
            .B = I1 * sys.B,
            .C = sys.C,
            .D = sys.D,
        };
    } else if (method == DiscretizationMethod::FOH) {
        const auto Q = I1 - (I2 / Ts);
        const auto P = I1 - Q;
        return StateSpace{
            .A = E,
            .B = (P + (E * Q)) * sys.B,
            .C = sys.C,
            .D = sys.C * Q * sys.B + sys.D,
        };
    } else {
        double k = 2.0 / Ts;
        if (prewarp.has_value()) {
            std::cout << "Has Value\n" << std::endl;
            k = prewarp.value() / std::tan(prewarp.value() * Ts / 2.0);
        }

        const auto Q = (k * I - sys.A).inverse();
        return StateSpace{
            .A = Q * (k * I + sys.A),
            .B = (I + sys.A) * Q * sys.B,
            .C = sys.C,
            .D = sys.C * Q * sys.B + sys.D,
        };
    }
}

};  // namespace control
