#pragma once

#include <cstddef>

#include "control_design.hpp"
#include "kalman.hpp"
#include "matrix.hpp"

/**
 * @defgroup discrete_controllers Discrete-Time Controllers
 * @brief Ready-to-use discrete-time controllers for embedded systems (ISR/RTOS)
 *
 * Use design:: namespace functions to create these from continuous or discrete models.
 * Use online:: namespace for fast runtime linearization and system identification.
 */

/**
 * @ingroup discrete_controllers
 * @brief Discrete Linear-Quadratic Regulator (LQR)
 *
 * State-feedback controller u = -K*x that minimizes a quadratic cost function.
 * Designed for regulation (driving state to zero) or tracking (following state reference).
 *
 * @tparam NX Number of states
 * @tparam NU Number of control inputs
 * @tparam T  Scalar type (default: double)
 */
template<size_t NX, size_t NU, typename T = double>
struct LQR {
private:
    Matrix<NU, NX, T> K{};

public:
    constexpr LQR() = default;
    constexpr explicit LQR(const Matrix<NU, NX, T>& K_) : K(K_) {}
    constexpr LQR(const design::LQRResult<NX, NU, T>& result) : K(result.K) {}

    template<typename U>
    constexpr LQR(const LQR<NX, NU, U>& other) : K(other.getK()) {}

    /**
     * @brief Compute regulator control law
     *
     * Drives state to zero: u = -K*x
     *
     * @param x Current state vector
     * @return Control input vector u
     */
    [[nodiscard]] constexpr ColVec<NU, T> control(const ColVec<NX, T>& x) const {
        return ColVec<NU, T>(-K * x);
    }

    /**
     * @brief Compute servo control law
     *
     * Tracks state reference: u = -K*(x - x_ref)
     *
     * @param x     Current state vector
     * @param x_ref Reference state vector
     * @return Control input vector u
     */
    [[nodiscard]] constexpr ColVec<NU, T> control(const ColVec<NX, T>& x, const ColVec<NX, T>& x_ref) const {
        return ColVec<NU, T>(-K * (x - x_ref));
    }

    [[nodiscard]] constexpr const Matrix<NU, NX, T>& getK() const { return K; }
};

/**
 * @ingroup discrete_controllers
 * @brief Linear-Quadratic-Integral (LQI) controller
 *
 * Output tracking controller with integral action: u = -Kx*x - Ki*xi
 * where xi integrates the output error (r - y).
 * Provides zero steady-state error for constant references and disturbances.
 *
 * @tparam NX Number of states
 * @tparam NU Number of control inputs
 * @tparam NY Number of outputs
 * @tparam T  Scalar type (default: double)
 */
template<size_t NX, size_t NU, size_t NY, typename T = double>
struct LQI {
private:
    Matrix<NU, NX, T> Kx{};
    Matrix<NU, NY, T> Ki{};
    ColVec<NY, T>     xi{};

public:
    constexpr LQI() = default;
    constexpr LQI(const Matrix<NU, NX, T>& Kx_, const Matrix<NU, NY, T>& Ki_) : Kx(Kx_), Ki(Ki_) {}
    constexpr LQI(const design::LQIResult<NX, NU, NY, T>& result) : Kx(result.Kx), Ki(result.Ki) {}

    template<typename U>
    constexpr LQI(const LQI<NX, NU, NY, U>& other)
        : Kx(other.getKx()), Ki(other.getKi()), xi(other.getIntegratorState()) {}

    /**
     * @brief Compute control with integral action
     *
     * Computes u = -Kx*x - Ki*xi where xi integrates (r - y)
     *
     * @param x Current state vector
     * @param r Reference output vector
     * @param y Current output vector
     * @return Control input vector u
     */
    [[nodiscard]] constexpr ColVec<NU, T> control(const ColVec<NX, T>& x, const ColVec<NY, T>& r, const ColVec<NY, T>& y) {
        xi = ColVec<NY, T>(xi + (r - y));
        return ColVec<NU, T>(-Kx * x - Ki * xi);
    }

    constexpr void reset() { xi = ColVec<NY, T>{}; }

    [[nodiscard]] constexpr const Matrix<NU, NX, T>& getKx() const { return Kx; }
    [[nodiscard]] constexpr const Matrix<NU, NY, T>& getKi() const { return Ki; }
    [[nodiscard]] constexpr const ColVec<NY, T>&     getIntegratorState() const { return xi; }
};

/**
 * @ingroup discrete_controllers
 * @brief Linear-Quadratic-Gaussian (LQG) controller
 *
 * Combines LQR optimal control with Kalman filter state estimation.
 * Implements separation principle: estimate state with Kalman filter,
 * then apply LQR control to estimated state.
 *
 * @tparam NX Number of states
 * @tparam NU Number of control inputs
 * @tparam NY Number of outputs
 * @tparam NW Number of process noise inputs (default: NX)
 * @tparam NV Number of measurement noise inputs (default: NY)
 * @tparam T  Scalar type (default: double)
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
struct LQG {
    LQR<NX, NU, T>                      lqr{};
    KalmanFilter<NX, NU, NY, NW, NV, T> kf{};

    constexpr LQG() = default;

    constexpr LQG(const LQR<NX, NU, T>& lqr_, const KalmanFilter<NX, NU, NY, NW, NV, T>& kf_)
        : lqr(lqr_), kf(kf_) {}

    constexpr LQG(const design::LQGResult<NX, NU, NY, NW, NV, T>& result)
        : lqr(result.lqr), kf(result.kalman.sys, result.kalman.Q, result.kalman.R) {}

    template<typename U>
    constexpr LQG(const LQG<NX, NU, NY, NW, NV, U>& other) : lqr(other.lqr), kf(other.kf) {}

    constexpr void predict(const ColVec<NU, T>& u = ColVec<NU, T>{}) { kf.predict(u); }
    constexpr bool update(const ColVec<NY, T>& z, const ColVec<NU, T>& u = ColVec<NU, T>{}) { return kf.update(z, u); }

    [[nodiscard]] constexpr ColVec<NU, T> control() const { return lqr.control(kf.state()); }
    [[nodiscard]] constexpr ColVec<NU, T> control(const ColVec<NX, T>& x_ref) const { return lqr.control(kf.state(), x_ref); }
};

/**
 * @ingroup discrete_controllers
 * @brief Linear-Quadratic-Gaussian-Integral (LQGI) controller
 *
 * Combines LQI output tracking controller with Kalman filter state estimation.
 * Provides optimal control with integral action and state estimation.
 *
 * @tparam NX Number of states
 * @tparam NU Number of control inputs
 * @tparam NY Number of outputs
 * @tparam NW Number of process noise inputs (default: NX)
 * @tparam NV Number of measurement noise inputs (default: NY)
 * @tparam T  Scalar type (default: double)
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
struct LQGI {
    LQI<NX, NU, NY, T>                  lqi{};
    KalmanFilter<NX, NU, NY, NW, NV, T> kf{};

    constexpr LQGI() = default;

    constexpr LQGI(const LQI<NX, NU, NY, T>& lqi_, const KalmanFilter<NX, NU, NY, NW, NV, T>& kf_)
        : lqi(lqi_), kf(kf_) {}

    constexpr LQGI(const design::LQGIResult<NX, NU, NY, NW, NV, T>& result)
        : lqi(result.lqi), kf(result.kalman.sys, result.kalman.Q, result.kalman.R) {}

    template<typename U>
    constexpr LQGI(const LQGI<NX, NU, NY, NW, NV, U>& other) : lqi(other.lqi), kf(other.kf) {}

    constexpr void predict(const ColVec<NU, T>& u = ColVec<NU, T>{}) { kf.predict(u); }
    constexpr bool update(const ColVec<NY, T>& z, const ColVec<NU, T>& u = ColVec<NU, T>{}) { return kf.update(z, u); }

    [[nodiscard]] constexpr ColVec<NU, T> control(const ColVec<NY, T>& r) { return lqi.control(kf.state(), r, kf.innovation()); }

    constexpr void reset() { lqi.reset(); }
};
