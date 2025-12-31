#pragma once

#include <functional>

#include "ss.hpp"
#include "types.hpp"

namespace control {

/**
 * @brief Compute numerical Jacobian using finite differences
 *
 * @tparam Func Callable type that returns a ColVec
 * @tparam Args Types of additional arguments to pass to func
 * @param func Function to differentiate: ColVec func(ColVec, Args...)
 * @param x Point at which to compute the Jacobian (with respect to this variable)
 * @param eps Finite difference step size
 * @param args Additional arguments to pass to func
 * @return Matrix Jacobian matrix (m x n) where m = func(x, args...).size(), n = x.size()
 */
template <typename Func, typename... Args>
Matrix numericalJacobian(Func&& func, const ColVec& x, double eps = 1e-8, Args&&... args) {
    // Evaluate function at base point
    ColVec f0 = func(x, std::forward<Args>(args)...);
    size_t n  = x.size();
    size_t m  = f0.size();

    Matrix J(m, n);

    // Compute finite differences for each column
    for (size_t j = 0; j < n; ++j) {
        ColVec x_plus = x;
        x_plus(j) += eps;
        ColVec f_plus = func(x_plus, std::forward<Args>(args)...);
        J.col(j)      = (f_plus - f0) / eps;
    }

    return J;
}

/**
 * @brief Representation of a nonlinear system
 *
 * dx/dt = f(x, u)
 * y = h(x, u)
 */
class NonlinearSystem {
   public:
    // Function signatures
    using StateTransitionFcn = std::function<ColVec(const ColVec& x, const ColVec& u)>;
    using MeasurementFcn     = std::function<ColVec(const ColVec& x, const ColVec& u)>;

    NonlinearSystem(StateTransitionFcn f, MeasurementFcn h, size_t nx, size_t nu, size_t ny)
        : f_(std::move(f)), h_(std::move(h)), nx_(nx), nu_(nu), ny_(ny) {}

    // Linearize around operating point
    StateSpace linearize(const ColVec& x0, const ColVec& u0) const;

    // Get system dimensions
    size_t getNumStates() const { return nx_; }
    size_t getNumInputs() const { return nu_; }
    size_t getNumOutputs() const { return ny_; }

    // Get function objects
    const StateTransitionFcn& getStateTransitionFcn() const { return f_; }
    const MeasurementFcn&     getMeasurementFcn() const { return h_; }

   private:
    StateTransitionFcn f_;
    MeasurementFcn     h_;
    size_t             nx_, nu_, ny_;
};

}  // namespace control