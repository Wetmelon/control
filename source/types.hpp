#pragma once

#include <initializer_list>

#include "Eigen/Dense"
#include "Eigen/src/Core/Matrix.h"

namespace control {

using Matrix = Eigen::MatrixXd;

using Pole = std::complex<double>;
using Zero = std::complex<double>;

// Wrapper classes that support initializer list syntax
struct ColVec : public Eigen::VectorXd {
    using Eigen::VectorXd::VectorXd;  // Inherit constructors

    ColVec(std::initializer_list<double> list)
        : Eigen::VectorXd(list.size()) {
        size_t i = 0;
        for (double val : list) (*this)[i++] = val;
    }
};

struct RowVec : public Eigen::RowVectorXd {
    using Eigen::RowVectorXd::RowVectorXd;  // Inherit constructors

    RowVec(std::initializer_list<double> list)
        : Eigen::RowVectorXd(list.size()) {
        size_t i = 0;
        for (double val : list) (*this)[i++] = val;
    }
};

}  // namespace control

namespace Eigen {
namespace internal {
    template <>
    struct traits<control::ColVec> : traits<Eigen::VectorXd> {};

    template <>
    struct traits<control::RowVec> : traits<Eigen::RowVectorXd> {};
}  // namespace internal

}  // namespace Eigen
