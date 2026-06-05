#pragma once

#include "lqi.hpp"
#include "wet/estimation/kalman.hpp"
#include "wet/matrix/matrix.hpp"
#include "wet/systems/state_space.hpp"

namespace wet {
namespace design {
/**
 * @struct LQGIResult
 * @brief LQGI design result
 *
 * Combines LQI and Kalman filter designs for servo control with state estimation.
 * Use `.as<float>()` to convert for embedded deployment.
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
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

    constexpr LQGI(const design::LQGIResult<NX, NU, NY, NW, NV, T>& result) // NOLINT
        : lqi(result.lqi), kf(result.kalman) {}

    template<typename U>
    constexpr LQGI(const LQGI<NX, NU, NY, NW, NV, U>& other) : lqi(other.lqi), kf(other.kf) {} // NOLINT

    constexpr void predict(const ColVec<NU, T>& u = ColVec<NU, T>{}) { kf.predict(u); }
    constexpr bool update(const ColVec<NY, T>& y, const ColVec<NU, T>& u = ColVec<NU, T>{}) { return kf.update(y, u); }

    /**
     * @brief Compute control with integral action from the augmented state.
     *
     * @param x_aug Augmented state [x̂; xi] — estimated plant state stacked on
     *              the integral-of-tracking-error state (size NX + NY).
     * @return Control input u = −K·x_aug
     */
    [[nodiscard]] constexpr ColVec<NU, T> control(const ColVec<NX + NY, T>& x_aug) {
        return lqi.control(x_aug);
    }
};

} // namespace wet