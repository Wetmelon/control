#pragma once

#include <cmath>
#include <optional>
#include <ostream>

#include "Eigen/Dense"
#include "Eigen/src/Core/util/Meta.h"

namespace control {
using mat = Eigen::MatrixXd;

enum class DiscretizationMethod {
    ZOH,
    FOH,
    Bilinear,
    Tustin,
};

struct TransferFunction {
    Eigen::MatrixXd num, den;
};

struct StateSpace {
    StateSpace(const mat& A, const mat& B, const mat& C, const mat& D, const std::optional<double>& Ts = std::nullopt,
               const std::optional<DiscretizationMethod>& method = std::nullopt, const std::optional<double>& prewarp = std::nullopt)
        : A(A), B(B), C(C), D(D), Ts(Ts), method(method), prewarp(prewarp){};

    StateSpace c2d(const double Ts, const DiscretizationMethod method = DiscretizationMethod::ZOH, std::optional<double> prewarp = std::nullopt) const;

    const Eigen::MatrixXd A = {}, B = {}, C = {}, D = {};

    const std::optional<double>               Ts      = std::nullopt;
    const std::optional<DiscretizationMethod> method  = std::nullopt;
    const std::optional<double>               prewarp = std::nullopt;

    friend std::ostream& operator<<(std::ostream& os, const StateSpace& sys) {
        os << "A = \n" << sys.A << '\n' << '\n';
        os << "B = \n" << sys.B << '\n' << '\n';
        os << "C = \n" << sys.C << '\n' << '\n';
        os << "D = \n" << sys.D << '\n' << '\n';

        return os;
    }
};

};  // namespace control
