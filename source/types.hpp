#pragma once

#include <initializer_list>

#include "Eigen/Dense"
#include "Eigen/src/Core/Matrix.h"

namespace control {

using Matrix    = Eigen::MatrixXd;
using ColVector = Eigen::VectorXd;
using RowVector = Eigen::RowVectorXd;

using Pole = std::complex<double>;
using Zero = std::complex<double>;

// Wrapper classes that support initializer list syntax
struct ColVec : Eigen::VectorXd {
    using Eigen::VectorXd::VectorXd;  // Inherit constructors

    ColVec(std::initializer_list<double> list)
        : Eigen::VectorXd(list.size()) {
        size_t i = 0;
        for (double val : list) (*this)[i++] = val;
    }
};

struct RowVec : Eigen::RowVectorXd {
    using Eigen::RowVectorXd::RowVectorXd;  // Inherit constructors

    RowVec(std::initializer_list<double> list)
        : Eigen::RowVectorXd(list.size()) {
        size_t i = 0;
        for (double val : list) (*this)[i++] = val;
    }
};

}  // namespace control
