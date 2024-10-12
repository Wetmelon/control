#pragma once

#include <cmath>
#include <optional>

#include "Eigen"

namespace control {
using mat = Eigen::MatrixXd;

enum class DiscretizationMethod {
    ZOH,
    FOH,
    Tustin,
    Bilinear,
};

struct TransferFunction {
    Eigen::MatrixXd num, den;
};

struct StateSpace {
    StateSpace(const mat& A, const mat& B, const mat& C, const mat& D, std::optional<double> Ts = std::nullopt) : A(A), B(B), C(C), D(D), Ts(Ts){};
    StateSpace c2d(const double Ts, const DiscretizationMethod method = DiscretizationMethod::ZOH, std::optional<double> prewarp = std::nullopt) const;

    Eigen::MatrixXd       A, B, C, D;
    std::optional<double> Ts;
};

};  // namespace control
