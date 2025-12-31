#include "observer.hpp"

#include <functional>

#include "LTI.hpp"
#include "integrator.hpp"
#include "nonlinear.hpp"
#include "ss.hpp"
#include "types.hpp"

namespace control {
LuenbergerObserver::LuenbergerObserver(StateSpace sys, Matrix L)
    : model_(std::move(sys)), L_(std::move(L)) {
    x_hat_ = ColVec::Zero(model_.A.rows());
}

ColVec LuenbergerObserver::predict(const ColVec& u, double dt) {
    if (model_.Ts.has_value()) {
        // Discrete-time system, one step prediction
        x_hat_ = model_.A * x_hat_ + model_.B * u;
        return x_hat_;
    } else {
        // Continuous-time system, integrate over dt
        x_hat_ = Exact{}.evolve(model_.A, model_.B, x_hat_, u, dt).x;
        return x_hat_;
    }
}

ColVec LuenbergerObserver::update(const ColVec& y) {
    y_     = y - model_.C * x_hat_;
    x_hat_ = x_hat_ + (L_ * y_);
    return x_hat_;
}

ColVec LuenbergerObserver::state() const {
    return x_hat_;
}

ColVec LuenbergerObserver::output() const {
    return model_.C * x_hat_;
}

KalmanFilter::KalmanFilter(StateSpace sys, Matrix Q, Matrix R)
    : model_(std::move(sys)), Q_(std::move(Q)), R_(std::move(R)) {
    P_     = Matrix::Identity(model_.A.rows(), model_.A.rows());
    x_hat_ = ColVec::Zero(model_.A.rows());
}

ColVec KalmanFilter::predict(const ColVec& u, double dt) {
    if (model_.Ts.has_value()) {
        // Discrete-time system, one step prediction
        x_hat_ = model_.A * x_hat_ + model_.B * u;

        // Covariance propagation
        P_ = model_.A * P_ * model_.A.transpose() + Q_;
        return x_hat_;
    } else {
        // Continuous system: exact state propagation
        x_hat_ = Exact{}.evolve(model_.A, model_.B, x_hat_, u, dt).x;

        // Covariance propagation using continuous-time Riccati equation
        P_ = P_ + dt * (model_.A * P_ + P_ * model_.A.transpose() + Q_);
        return x_hat_;
    }
}

ColVec KalmanFilter::update(const ColVec& z) {
    const Matrix  I  = Matrix::Identity(P_.rows(), P_.cols());
    const Matrix& H  = model_.C;  // Measurement matrix
    const Matrix& HT = H.transpose();

    y_     = z - H * x_hat_;          // Measurement residual y = z - H x̂
    S_     = H * P_ * HT + R_;        // Innovation covariance S = H P H^T + R
    K_     = P_ * HT * S_.inverse();  // Kalman Gain K = P H^T S^-1
    x_hat_ = x_hat_ + K_ * y_;        // State update x̂ = x̂ + K * y

    // Joseph form for numerical stability: P = (I - K H) P (I - K H)^T + K R K^T
    Matrix M = I - K_ * H;
    P_       = M * P_ * M.transpose() + K_ * R_ * K_.transpose();

    return x_hat_;
}

ColVec KalmanFilter::state() const {
    return x_hat_;
}

ColVec KalmanFilter::output() const {
    return model_.C * x_hat_;
}

// Compute Jacobians using Finite Differences, default covariances
ExtendedKalmanFilter::ExtendedKalmanFilter(StateTransitionFcn f,
                                           MeasurementFcn     h,
                                           ColVec             x0)
    : f_{std::move(f)},
      h_{std::move(h)},
      x_{std::move(x0)},
      Q_{Matrix::Identity(x_.size(), x_.size())},
      R_{Matrix::Identity(h_(x_).size(), h_(x_).size())},
      P_{Matrix::Identity(x_.size(), x_.size())} {}

// Automatically compute Jacobians using Finite Differences
ExtendedKalmanFilter::ExtendedKalmanFilter(StateTransitionFcn f,
                                           MeasurementFcn     h,
                                           ColVec             x0,
                                           Matrix             Q,
                                           Matrix             R)
    : f_{std::move(f)},
      h_{std::move(h)},
      x_{std::move(x0)},
      Q_{std::move(Q)},
      R_{std::move(R)},
      P_{Matrix::Identity(x_.size(), x_.size())} {}

// NonlinearSystem-based constructor with automatic jacobian computation
ExtendedKalmanFilter::ExtendedKalmanFilter(const NonlinearSystem& system, ColVec x0)
    : f_{[&system](const ColVec& x, const ColVec& u) { return system.getStateTransitionFcn()(x, u); }},
      h_{[&system](const ColVec& x) { return system.getMeasurementFcn()(x, ColVec::Zero(system.getNumInputs())); }},
      x_{std::move(x0)},
      Q_{Matrix::Identity(x_.size(), x_.size())},
      R_{Matrix::Identity(system.getNumOutputs(), system.getNumOutputs())},
      P_{Matrix::Identity(x_.size(), x_.size())} {}

// NonlinearSystem-based constructor with custom covariances
ExtendedKalmanFilter::ExtendedKalmanFilter(const NonlinearSystem& system, ColVec x0, Matrix Q, Matrix R)
    : f_{[&system](const ColVec& x, const ColVec& u) { return system.getStateTransitionFcn()(x, u); }},
      h_{[&system](const ColVec& x) { return system.getMeasurementFcn()(x, ColVec::Zero(system.getNumInputs())); }},
      x_{std::move(x0)},
      Q_{std::move(Q)},
      R_{std::move(R)},
      P_{Matrix::Identity(x_.size(), x_.size())} {}

// User-provided jacobians
ExtendedKalmanFilter::ExtendedKalmanFilter(StateTransitionFcn  f,
                                           MeasurementFcn      h,
                                           StateJacobian       F,
                                           ObservationJacobian H,
                                           ColVec              x0)
    : f_{std::move(f)},
      h_{std::move(h)},
      F_{std::move(F)},
      H_{std::move(H)},
      x_{std::move(x0)},
      Q_{Matrix::Identity(x_.size(), x_.size())},
      R_{Matrix::Identity(h_(x_).size(), h_(x_).size())},
      P_{Matrix::Identity(x_.size(), x_.size())} {}

ExtendedKalmanFilter::ExtendedKalmanFilter(StateTransitionFcn  f,
                                           MeasurementFcn      h,
                                           StateJacobian       F,
                                           ObservationJacobian H,
                                           ColVec              x0,
                                           Matrix              Q,
                                           Matrix              R)
    : f_{std::move(f)},
      h_{std::move(h)},
      F_{std::move(F)},
      H_{std::move(H)},
      x_{std::move(x0)},
      Q_{std::move(Q)},
      R_{std::move(R)},
      P_{Matrix::Identity(x_.size(), x_.size())} {}

ColVec ExtendedKalmanFilter::predict(const ColVec& u) {
    Matrix F = (F_) ? F_(x_, u) : computeJacobian(f_, x_, u);
    x_       = f_(x_, u);
    P_       = F * P_ * F.transpose() + Q_;

    return x_;
}

ColVec ExtendedKalmanFilter::update(const ColVec& z) {
    const Matrix I = Matrix::Identity(P_.rows(), P_.cols());

    const Matrix H  = (H_) ? H_(x_) : computeJacobian(h_, x_);
    const Matrix HT = H.transpose();

    y_ = z - h_(x_);              // Measurement residual
    S_ = H * P_ * HT + R_;        // Innovation covariance
    K_ = P_ * HT * S_.inverse();  // Kalman Gain
    x_ = x_ + K_ * y_;            // State update

    // Joseph form for numerical stability: P = (I - K H) P (I - K H)^T + K R K^T
    Matrix M = I - K_ * H;
    P_       = M * P_ * M.transpose() + K_ * R_ * K_.transpose();

    return x_;
}

// Finite Difference for state function (3-arg)
Matrix ExtendedKalmanFilter::computeJacobian(const StateTransitionFcn& f, const ColVec& x, const ColVec& u) const {
    return numericalJacobian(f, x, 1e-8, u);
}

// Finite Difference for measurement function (2-arg)
Matrix ExtendedKalmanFilter::computeJacobian(const MeasurementFcn& h, const ColVec& x) const {
    return numericalJacobian(h, x);
}

}  // namespace control