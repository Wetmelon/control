#pragma once

#include <cstddef>

#include "kalman.hpp"
#include "lqr.hpp"
#include "matrix.hpp"
#include "state_space.hpp"

namespace wetmelon::control {

namespace online {

/**
 * @struct LQGResult
 * @brief Runtime LQG design result (online namespace)
 */
template<size_t NX, size_t NU, size_t NY, size_t NW, size_t NV, typename T>
struct LQGResult {
    LQRResult<NX, NU, T>                lqr{};
    KalmanResult<NX, NU, NY, NW, NV, T> kalman{};
    bool                                success{false};

    template<typename U>
    [[nodiscard]] constexpr auto as() const {
        return LQGResult<NX, NU, NY, NW, NV, U>{
            lqr.template as<U>(),
            kalman.template as<U>(),
            success
        };
    }
};

/**
 * @brief  LQG regulator design combining LQR and Kalman filter
 *
 * @param sys    State-space system
 * @param Q_lqr  State cost for LQR
 * @param R_lqr  Input cost for LQR
 * @param Q_kf   Process noise covariance for Kalman filter
 * @param R_kf   Measurement noise covariance for Kalman filter
 * @param N      (optional) Cross-term cost matrix
 *
 * @return LQGResult combining LQR and Kalman filter designs
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
[[nodiscard]] constexpr LQGResult<NX, NU, NY, NW, NV, T> lqg_regulator(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NX, NX, T>&                 Q_lqr,
    const Matrix<NU, NU, T>&                 R_lqr,
    const Matrix<NW, NW, T>&                 Q_kf,
    const Matrix<NV, NV, T>&                 R_kf,
    const Matrix<NX, NU, T>&                 N = Matrix<NX, NU, T>{}
) {
    return lqg(sys, Q_lqr, R_lqr, Q_kf, R_kf, N);
}

/**
 * @brief Combine separate Kalman and LQR results into an LQG controller
 *
 * @param kest          Kalman filter design result
 * @param lqr_result    LQR controller design result
 *
 * @return LQG Regulator combining the provided Kalman and LQR results
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
[[nodiscard]] constexpr LQGResult<NX, NU, NY, NW, NV, T> lqg_from_parts(
    const KalmanResult<NX, NU, NY, NW, NV, T>& kest,
    const LQRResult<NX, NU, T>&                lqr_result
) {
    return lqgreg(kest, lqr_result);
}

/**
 * @brief Linear-Quadratic-Gaussian regulator design combining LQR and Kalman filter (runtime version)
 *
 * @param sys     State-space system
 * @param Q_lqr   State cost for LQR
 * @param R_lqr   Input cost for LQR
 * @param Q_kf    Process noise covariance for Kalman filter
 * @param R_kf    Measurement noise covariance for Kalman filter
 * @param N       (optional) Cross-term cost matrix
 *
 * @return LQGResult combining LQR and Kalman filter designs
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = 0, size_t NV = 0, typename T = double>
[[nodiscard]] constexpr LQGResult<NX, NU, NY, NW, NV, T> lqg(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NX, NX, T>&                 Q_lqr,
    const Matrix<NU, NU, T>&                 R_lqr,
    const Matrix<NW, NW, T>&                 Q_kf,
    const Matrix<NV, NV, T>&                 R_kf,
    const Matrix<NX, NU, T>&                 N = Matrix<NX, NU, T>{}
) {
    const auto lqr_result = dlqr(sys.A, sys.B, Q_lqr, R_lqr, N);
    const auto kalman_result = kalman(sys, Q_kf, R_kf);
    return LQGResult<NX, NU, NY, NW, NV, T>{lqr_result, kalman_result, lqr_result.success && kalman_result.success};
}

/**
 * @brief Combine separate Kalman filter and LQR designs into an LQG controller (runtime version)
 *
 * @param kest        Kalman filter design result
 * @param lqr_result  LQR controller design result
 *
 * @return LQGResult combining the provided Kalman and LQR designs
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
[[nodiscard]] constexpr LQGResult<NX, NU, NY, NW, NV, T> lqgreg(
    const KalmanResult<NX, NU, NY, NW, NV, T>& kest,
    const LQRResult<NX, NU, T>&                lqr_result
) {
    return LQGResult<NX, NU, NY, NW, NV, T>{lqr_result, kest, lqr_result.success && kest.success};
}

} // namespace online

namespace design {

/**
 * @struct LQGResult
 * @brief Linear-Quadratic-Gaussian controller design result
 *
 * Combines LQR and Kalman filter designs for separation principle-based control.
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
struct LQGResult {
    LQRResult<NX, NU, T>                lqr{};          //!< LQR design result
    KalmanResult<NX, NU, NY, NW, NV, T> kalman{};       //!< Kalman filter result
    bool                                success{false}; //!< Indicates combined design success

    template<typename U>
    [[nodiscard]] consteval auto as() const {
        return LQGResult<NX, NU, NY, NW, NV, U>{
            lqr.template as<U>(),
            kalman.template as<U>(),
            success
        };
    }
};

/**
 * @brief Linear-Quadratic-Gaussian regulator design combining LQR and Kalman filter
 *
 * @param sys     State-space system
 * @param Q_lqr   State cost for LQR
 * @param R_lqr   Input cost for LQR
 * @param Q_kf    Process noise covariance for Kalman filter
 * @param R_kf    Measurement noise covariance for Kalman filter
 * @param N       (optional) Cross-term cost matrix
 *
 * @return LQGResult combining LQR and Kalman filter designs
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = 0, size_t NV = 0, typename T = double>
[[nodiscard]] consteval LQGResult<NX, NU, NY, NW, NV, T> lqg(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NX, NX, T>&                 Q_lqr, // State cost
    const Matrix<NU, NU, T>&                 R_lqr, // Input cost
    const Matrix<NW, NW, T>&                 Q_kf,  // Process noise covariance
    const Matrix<NV, NV, T>&                 R_kf,  // Measurement noise covariance
    const Matrix<NX, NU, T>&                 N = Matrix<NX, NU, T>{}
) {
    const auto lqr_result = dlqr(sys.A, sys.B, Q_lqr, R_lqr, N);
    const auto kalman_result = kalman(sys, Q_kf, R_kf);
    return LQGResult<NX, NU, NY, NW, NV, T>{lqr_result, kalman_result, lqr_result.success && kalman_result.success};
}

/**
 * @brief Alias for LQG regulator design combining LQR and Kalman filter (consteval version)
 *
 * @param sys     State-space system
 * @param Q_lqr   State cost for LQR
 * @param R_lqr   Input cost for LQR
 * @param Q_kf    Process noise covariance for Kalman filter
 * @param R_kf    Measurement noise covariance for Kalman filter
 * @param N       (optional) Cross-term cost matrix
 *
 * @return LQGResult combining LQR and Kalman filter designs
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
[[nodiscard]] consteval LQGResult<NX, NU, NY, NW, NV, T> lqg_regulator(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NX, NX, T>&                 Q_lqr,
    const Matrix<NU, NU, T>&                 R_lqr,
    const Matrix<NW, NW, T>&                 Q_kf,
    const Matrix<NV, NV, T>&                 R_kf,
    const Matrix<NX, NU, T>&                 N = Matrix<NX, NU, T>{}
) {
    return lqg(sys, Q_lqr, R_lqr, Q_kf, R_kf, N);
}

/**
 * @brief Combine separate Kalman filter and LQR designs into an LQG controller
 *
 * @param kest        Kalman filter design result
 * @param lqr_result  LQR controller design result
 *
 * @return LQGResult combining the provided Kalman and LQR designs
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
[[nodiscard]] consteval LQGResult<NX, NU, NY, NW, NV, T> lqgreg(
    const KalmanResult<NX, NU, NY, NW, NV, T>& kest,
    const LQRResult<NX, NU, T>&                lqr_result
) {
    return LQGResult<NX, NU, NY, NW, NV, T>{lqr_result, kest, lqr_result.success && kest.success};
}

/**
 * @brief Combine separate Kalman and LQR results into an LQG controller
 *
 * @param kest        Kalman filter result
 * @param lqr_result  LQR controller result
 *
 * @return LQG controller result structure
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
[[nodiscard]] consteval LQGResult<NX, NU, NY, NW, NV, T> lqg_from_parts(
    const KalmanResult<NX, NU, NY, NW, NV, T>& kest,
    const LQRResult<NX, NU, T>&                lqr_result
) {
    return lqgreg(kest, lqr_result);
}

} // namespace design

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

    // Compile-time only constructor for design:: results
    consteval LQG(const design::LQGResult<NX, NU, NY, NW, NV, T>& result)
        : lqr(result.lqr),
          kf(
              result.kalman.sys,
              result.kalman.Q,
              result.kalman.R,
              ColVec<NX, T>{},
              result.kalman.success ? result.kalman.P : Matrix<NX, NX, T>::identity()
          ) {}

    // Runtime constructor for online:: results
    constexpr LQG(const online::LQGResult<NX, NU, NY, NW, NV, T>& result)
        : lqr(result.lqr),
          kf(
              result.kalman.sys,
              result.kalman.Q,
              result.kalman.R,
              ColVec<NX, T>{},
              result.kalman.success ? result.kalman.P : Matrix<NX, NX, T>::identity()
          ) {}

    template<typename U>
    constexpr LQG(const LQG<NX, NU, NY, NW, NV, U>& other) : lqr(other.lqr), kf(other.kf) {}

    constexpr void predict(const ColVec<NU, T>& u = ColVec<NU, T>{}) { kf.predict(u); }
    constexpr bool update(const ColVec<NY, T>& z, const ColVec<NU, T>& u = ColVec<NU, T>{}) { return kf.update(z, u); }

    [[nodiscard]] constexpr ColVec<NU, T> control() const { return lqr.control(kf.state()); }
    [[nodiscard]] constexpr ColVec<NU, T> control(const ColVec<NX, T>& x_ref) const { return lqr.control(kf.state(), x_ref); }
};

} // namespace wetmelon::control