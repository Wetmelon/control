#pragma once

#include "kalman.hpp"
#include "lqi.hpp"
#include "matrix.hpp"
#include "state_space.hpp"

namespace wetmelon::control {

namespace online {

/**
 * @struct LQGIResult
 * @brief Runtime LQGI design result (online namespace)
 */
template<size_t NX, size_t NU, size_t NY, size_t NW, size_t NV, typename T>
struct LQGIResult {
    LQIResult<NX, NU, NY, T>            lqi{};
    KalmanResult<NX, NU, NY, NW, NV, T> kalman{};
    bool                                success{false};

    template<typename U>
    [[nodiscard]] constexpr auto as() const {
        return LQGIResult<NX, NU, NY, NW, NV, U>{
            lqi.template as<U>(),
            kalman.template as<U>(),
            success
        };
    }
};

/**
 * @brief Linear-Quadratic-Gaussian with integral action for tracking (runtime version)
 *
 * @param sys      State-space system
 * @param Q_aug    Augmented state cost (state + integral error)
 * @param R        Input cost matrix
 * @param Q_kf     Process noise covariance for Kalman filter
 * @param R_kf     Measurement noise covariance for Kalman filter
 * @param dof      Servo degrees of freedom (1DOF or 2DOF)
 *
 * @return LQGIResult with integral action for tracking
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = 0, size_t NV = 0, typename T = double>
[[nodiscard]] constexpr LQGIResult<NX, NU, NY, NW, NV, T> lqgtrack(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NX + NY, NX + NY, T>&       Q_aug,
    const Matrix<NU, NU, T>&                 R,
    const Matrix<NW, NW, T>&                 Q_kf,
    const Matrix<NV, NV, T>&                 R_kf
) {
    const auto lqi_result = lqi(sys, Q_aug, R);
    const auto kalman_result = kalman(sys, Q_kf, R_kf);
    return LQGIResult<NX, NU, NY, NW, NV, T>{lqi_result, kalman_result, lqi_result.success && kalman_result.success};
}

/**
 * @brief LQG with integral action for tracking
 *
 * @param sys    State-space system
 * @param Q_aug  Augmented state cost (state + integral)
 * @param R      Input cost
 * @param Q_kf   Process noise covariance
 * @param R_kf   Measurement noise covariance
 * @param dof    Servo degrees of freedom (1DOF or 2DOF)
 *
 * @return  LQGIResult with integral action for tracking
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
[[nodiscard]] constexpr LQGIResult<NX, NU, NY, NW, NV, T> lqg_servo(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NX + NY, NX + NY, T>&       Q_aug,
    const Matrix<NU, NU, T>&                 R,
    const Matrix<NW, NW, T>&                 Q_kf,
    const Matrix<NV, NV, T>&                 R_kf
) {
    return lqgtrack(sys, Q_aug, R, Q_kf, R_kf);
}

} // namespace online

namespace design {

/**
 * @struct LQGIResult
 * @brief Linear-Quadratic-Gaussian Integral controller design result
 *
 * Combines LQI and Kalman filter designs for servo control with state estimation.
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
struct LQGIResult {
    LQIResult<NX, NU, NY, T>            lqi{};          //!< LQI design result
    KalmanResult<NX, NU, NY, NW, NV, T> kalman{};       //!< Kalman filter result
    bool                                success{false}; //!< Indicates combined design success

    template<typename U>
    [[nodiscard]] consteval auto as() const {
        return LQGIResult<NX, NU, NY, NW, NV, U>{
            lqi.template as<U>(),
            kalman.template as<U>(),
            success
        };
    }
};

/**
 * @brief Linear-Quadratic-Gaussian with integral action for tracking
 *
 * @param sys      State-space system
 * @param Q_aug    Augmented state cost (state + integral error)
 * @param R        Input cost matrix
 * @param Q_kf     Process noise covariance for Kalman filter
 * @param R_kf     Measurement noise covariance for Kalman filter
 * @param dof      Servo degrees of freedom (1DOF or 2DOF)
 *
 * @return LQGIResult with integral action for tracking
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = 0, size_t NV = 0, typename T = double>
[[nodiscard]] consteval LQGIResult<NX, NU, NY, NW, NV, T> lqgtrack(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NX + NY, NX + NY, T>&       Q_aug, // Augmented state cost (state + integral)
    const Matrix<NU, NU, T>&                 R,     // Input cost
    const Matrix<NW, NW, T>&                 Q_kf,  // Process noise covariance
    const Matrix<NV, NV, T>&                 R_kf   // Measurement noise covariance
) {
    const auto lqi_result = lqi(sys, Q_aug, R);
    const auto kalman_result = kalman(sys, Q_kf, R_kf);
    return LQGIResult<NX, NU, NY, NW, NV, T>{lqi_result, kalman_result, lqi_result.success && kalman_result.success};
}

/**
 * @brief Alias for LQG with integral action for servo tracking (consteval version)
 *
 * @param sys      State-space system
 * @param Q_aug    Augmented state cost (state + integral error)
 * @param R        Input cost matrix
 * @param Q_kf     Process noise covariance for Kalman filter
 * @param R_kf     Measurement noise covariance for Kalman filter
 * @param dof      Servo degrees of freedom (1DOF or 2DOF)
 *
 * @return LQGIResult with integral action for tracking
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
[[nodiscard]] consteval LQGIResult<NX, NU, NY, NW, NV, T> lqg_servo(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NX + NY, NX + NY, T>&       Q_aug,
    const Matrix<NU, NU, T>&                 R,
    const Matrix<NW, NW, T>&                 Q_kf,
    const Matrix<NV, NV, T>&                 R_kf
) {
    return lqgtrack(sys, Q_aug, R, Q_kf, R_kf);
}

} // namespace design

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

    // Compile-time only constructor for design:: results
    consteval LQGI(const design::LQGIResult<NX, NU, NY, NW, NV, T>& result)
        : lqi(result.lqi),
          kf(
              result.kalman.sys,
              result.kalman.Q,
              result.kalman.R,
              ColVec<NX, T>{},
              result.kalman.success ? result.kalman.P : Matrix<NX, NX, T>::identity()
          ) {}

    // Runtime constructor for online:: results
    constexpr LQGI(const online::LQGIResult<NX, NU, NY, NW, NV, T>& result)
        : lqi(result.lqi),
          kf(
              result.kalman.sys,
              result.kalman.Q,
              result.kalman.R,
              ColVec<NX, T>{},
              result.kalman.success ? result.kalman.P : Matrix<NX, NX, T>::identity()
          ) {}

    template<typename U>
    constexpr LQGI(const LQGI<NX, NU, NY, NW, NV, U>& other) : lqi(other.lqi), kf(other.kf) {}

    constexpr void predict(const ColVec<NU, T>& u = ColVec<NU, T>{}) { kf.predict(u); }
    constexpr bool update(const ColVec<NY, T>& y, const ColVec<NU, T>& u = ColVec<NU, T>{}) { return kf.update(y, u); }

    [[nodiscard]] constexpr ColVec<NU, T> control(const ColVec<NY, T>& x_aug) {
        return lqi.control(x_aug);
    }
};

} // namespace wetmelon::control