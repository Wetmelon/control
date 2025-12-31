#pragma once

#include "LTI.hpp"
#include "integrator.hpp"  // Added for EKF integration
#include "nonlinear.hpp"
#include "ss.hpp"
#include "types.hpp"

namespace control {

/**
 * @brief Luenberger Observer for state estimation
 *
 * Given a continuous or discrete-time state-space model:
 *      x_dot = A x + B u
 *      y     = C x + D u
 * this class implements the Luenberger observer algorithm to estimate the state x
 * from the measurements y and inputs u.
 */
class LuenbergerObserver {
   public:
    LuenbergerObserver(StateSpace sys, Matrix L);

    // Predict state forward by one step
    // TODO:  Handle continuous vs discrete time properly, currently assumes dt=0.0 for continuous.  Should throw if dt=0.0 for continuous
    ColVec predict(const ColVec& u, double dt = 0.0);
    ColVec update(const ColVec& y);

    ColVec state() const;
    ColVec output() const;

    StateSpace getModel() const { return model_; }

   private:
    StateSpace model_ = {};  // Continuous or Discrete time StateSpace model
    ColVec     x_hat_ = {};  // State estimate
    ColVec     y_     = {};  // Measurement residual
    Matrix     L_     = {};  // Observer gain
};

/**
 * @brief Linear Kalman Filter (for linear systems)
 *
 * Given a discrete-time state-space model:
 *      x[k+1] = A x[k] + B u[k] + w[k],   w ~ N(0, Q)
 *      y[k]   = C x[k] + D u[k] + v[k],   v ~ N(0, R)
 *      where w and v are process and measurement noise respectively,
 *      this class implements the Kalman filter algorithm to estimate the state x[k]
 *      from the measurements y[k] and inputs u[k].
 */
class KalmanFilter {
   public:
    KalmanFilter(StateSpace sys, Matrix Q, Matrix R);

    // Predict state forward by one step
    // TODO:  Handle continuous vs discrete time properly, currently assumes dt=0.0 for continuous.  Should throw if dt=0.0 for continuous
    ColVec predict(const ColVec& u, double dt = 0.0);
    ColVec update(const ColVec& z);

    ColVec state() const;
    ColVec output() const;
    ColVec residual() const;

    StateSpace getModel() const { return model_; }

    const ColVec& getStateEstimate() const { return x_hat_; }
    const Matrix& getCovariance() const { return P_; }
    const Matrix& getProcessNoiseCovariance() const { return Q_; }
    const Matrix& getMeasurementNoiseCovariance() const { return R_; }
    const Matrix& getKalmanGain() const { return K_; }

    const ColVec& getX() const { return x_hat_; }
    const Matrix& getP() const { return P_; }
    const Matrix& getQ() const { return Q_; }
    const Matrix& getR() const { return R_; }
    const Matrix& getK() const { return K_; }

   private:
    StateSpace model_ = {};  // Continuous or Discrete time StateSpace model
    ColVec     x_hat_ = {};  // State estimate
    ColVec     y_     = {};  // Measurement residual
    ColVec     S_     = {};  // Measurement residual Covariance
    Matrix     P_     = {};  // Estimate covariance
    Matrix     Q_     = {};  // Process noise covariance
    Matrix     R_     = {};  // Measurement noise covariance
    Matrix     K_     = {};  // Kalman Gain
};

/**
 * @brief  Extended Kalman Filter for discrete nonlinear systems
 *
 * The Extended Kalman Filter (EKF) is used for state estimation in nonlinear systems.
 * It linearizes the nonlinear state transition and measurement functions around
 * the current state estimate to apply the Kalman filter equations.
 * The EKF requires the user to provide the state transition function, measurement function,
 * and their respective Jacobians.

 * The system is defined as:
 *      x[k+1] = f(t, x[k], u[k]) + w[k],   w ~ N(0, Q)
 *      y[k]   = h(t, x[k]) + v[k],         v ~ N(0, R)
 * where f is the nonlinear state transition function, h is the nonlinear measurement function,
 * and w and v are process and measurement noise respectively.
 *
 * Integration is performed according to the provided Integrator object.
 */
class ExtendedKalmanFilter {
   public:
    using StateTransitionFcn  = std::function<ColVec(const ColVec&, const ColVec&)>;
    using MeasurementFcn      = std::function<ColVec(const ColVec&)>;
    using StateJacobian       = std::function<Matrix(const ColVec&, const ColVec&)>;
    using ObservationJacobian = std::function<Matrix(const ColVec&)>;

    // Compute Jacobians using Finite Differences, default covariances
    ExtendedKalmanFilter(StateTransitionFcn f,
                         MeasurementFcn     h,
                         ColVec             x0);

    // NonlinearSystem-based constructor with automatic jacobian computation
    ExtendedKalmanFilter(const NonlinearSystem& system, ColVec x0);

    // NonlinearSystem-based constructor with custom covariances
    ExtendedKalmanFilter(const NonlinearSystem& system, ColVec x0, Matrix Q, Matrix R);

    // Automatically compute Jacobians using Finite Differences
    ExtendedKalmanFilter(StateTransitionFcn f,
                         MeasurementFcn     h,
                         ColVec             x0,
                         Matrix             Q,
                         Matrix             R);

    // User-provided Jacobians, default covariances
    ExtendedKalmanFilter(StateTransitionFcn  f,
                         MeasurementFcn      h,
                         StateJacobian       F,
                         ObservationJacobian H,
                         ColVec              x0);

    // User-provided jacobians
    ExtendedKalmanFilter(StateTransitionFcn  f,
                         MeasurementFcn      h,
                         StateJacobian       F,
                         ObservationJacobian H,
                         ColVec              x0,
                         Matrix              Q,
                         Matrix              R);

    ColVec predict(const ColVec& u);
    ColVec update(const ColVec& z);

    void setStateJacobian(StateJacobian F) { F_ = std::move(F); }
    void setMeasurementJacobian(ObservationJacobian H) { H_ = std::move(H); }

    const ColVec& state() const { return x_; }
    const Matrix& covariance() const { return P_; }

    // Finite Difference jacobian computations
    Matrix computeJacobian(const StateTransitionFcn& f, const ColVec& x, const ColVec& u) const;
    Matrix computeJacobian(const MeasurementFcn& h, const ColVec& x) const;

    const auto& getState() const { return x_; }
    const auto& getCovariance() const { return P_; }
    const auto& getProcessNoiseCovariance() const { return Q_; }
    const auto& getMeasurementNoiseCovariance() const { return R_; }
    const auto& getStateJacobian() const { return F_; }
    const auto& getMeasurementJacobian() const { return H_; }

    const auto& getX() const { return x_; }
    const auto& getP() const { return P_; }
    const auto& getQ() const { return Q_; }
    const auto& getR() const { return R_; }
    const auto& getF() const { return F_; }
    const auto& getH() const { return H_; }

   private:
    StateTransitionFcn  f_ = {};
    MeasurementFcn      h_ = {};
    StateJacobian       F_ = {};
    ObservationJacobian H_ = {};

    ColVec x_ = {};  // State estimate
    ColVec y_ = {};  // Measurement residual
    ColVec K_ = {};  // Kalman Gain
    ColVec S_ = {};  // Innovation covariance
    Matrix Q_ = {};  // Process noise covariance
    Matrix R_ = {};  // Measurement noise covariance
    Matrix P_ = {};  // Estimate covariance
};

}  // namespace control
