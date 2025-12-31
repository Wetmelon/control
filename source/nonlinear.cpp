#include "nonlinear.hpp"

#include <Eigen/Dense>

#include "ss.hpp"
#include "types.hpp"

namespace control {

StateSpace NonlinearSystem::linearize(const ColVec& x0, const ColVec& u0) const {
    // Compute A = df/dx at (x0, u0)
    auto   f_x = [&](const ColVec& x) { return f_(x, u0); };
    Matrix A   = numericalJacobian(f_x, x0);

    // Compute B = df/du at (x0, u0)
    auto   f_u = [&](const ColVec& u) { return f_(x0, u); };
    Matrix B   = numericalJacobian(f_u, u0);

    // Compute C = dh/dx at (x0, u0)
    auto   h_x = [&](const ColVec& x) { return h_(x, u0); };
    Matrix C   = numericalJacobian(h_x, x0);

    // Compute D = dh/du at (x0, u0)
    auto   h_u = [&](const ColVec& u) { return h_(x0, u); };
    Matrix D   = numericalJacobian(h_u, u0);

    return StateSpace(A, B, C, D);
}

}  // namespace control