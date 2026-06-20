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

    /**
     * @brief Convert the LQGI compensator to a discrete state-space block
     *
     * Realizes the dynamic compensator mapping the exogenous inputs [r; y] to the
     * control u, with internal state [x̂; xi] (estimator state stacked on the
     * integral-of-error state). Lets the closed loop drop into Bode/`feedback`/
     * `series` analysis. The prediction-form estimator uses the steady-state
     * Kalman gain L; the integrator advances xi[k+1] = xi[k] + (r − y).
     *
     * Partition K = [Kx | Ki]. With Bl = B − L·D:
     * @f[
     *   A_c = \begin{bmatrix} A - LC - B_l K_x & -B_l K_i \\ 0 & I \end{bmatrix},\;
     *   B_c = \begin{bmatrix} 0 & L \\ I & -I \end{bmatrix},\;
     *   C_c = [-K_x\; -K_i],\; D_c = 0.
     * @f]
     *
     * @return StateSpace with NX+NY states, 2·NY inputs ([r; y]), NU outputs (u)
     */
    [[nodiscard]] constexpr StateSpace<NX + NY, 2 * NY, NU, 0, 0, T> to_ss() const {
        const auto& A = kalman.sys.A;
        const auto& B = kalman.sys.B;
        const auto& C = kalman.sys.C;
        const auto& D = kalman.sys.D;
        const auto& L = kalman.L;

        const Matrix<NU, NX, T> Kx = lqi.K.template block<NU, NX>(0, 0);
        const Matrix<NU, NY, T> Ki = lqi.K.template block<NU, NY>(0, NX);
        const Matrix<NX, NU, T> Bl = B - L * D;

        StateSpace<NX + NY, 2 * NY, NU, 0, 0, T> ss{};
        ss.A.template block<NX, NX>(0, 0) = A - L * C - Bl * Kx;
        ss.A.template block<NX, NY>(0, NX) = -(Bl * Ki);
        ss.A.template block<NY, NY>(NX, NX) = Matrix<NY, NY, T>::identity();

        ss.B.template block<NX, NY>(0, NY) = L;                               // y → x̂
        ss.B.template block<NY, NY>(NX, 0) = Matrix<NY, NY, T>::identity();   // r → xi
        ss.B.template block<NY, NY>(NX, NY) = -Matrix<NY, NY, T>::identity(); // y → xi

        ss.C.template block<NU, NX>(0, 0) = -Kx;
        ss.C.template block<NU, NY>(0, NX) = -Ki;

        ss.Ts = kalman.sys.Ts;
        return ss;
    }
};

/**
 * @brief Linear-Quadratic-Gaussian with integral action for tracking
 *
 * @note Compare with MATLAB's lqgtrack(...) — exposed as the lqgtrack() alias in matlab.hpp.
 *
 * @param sys      State-space system
 * @param Q_aug    Augmented state cost (state + integral error)
 * @param R        Input cost matrix
 * @param Q_kf     Process noise covariance for Kalman filter
 * @param R_kf     Measurement noise covariance for Kalman filter
 *
 * @return LQGIResult with integral action for tracking
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = 0, size_t NV = 0, typename T = double>
[[nodiscard]] constexpr LQGIResult<NX, NU, NY, NW, NV, T> discrete_lqgi(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NX + NY, NX + NY, T>&       Q_aug,
    const Matrix<NU, NU, T>&                 R,
    const Matrix<NW, NW, T>&                 Q_kf,
    const Matrix<NV, NV, T>&                 R_kf
) {
    const auto lqi_result = discrete_lqi(sys, Q_aug, R);
    const auto kalman_result = kalman(sys, Q_kf, R_kf);
    return LQGIResult<NX, NU, NY, NW, NV, T>{lqi_result, kalman_result, lqi_result.success && kalman_result.success};
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
    [[nodiscard]] constexpr ColVec<NU, T> control(const ColVec<NX + NY, T>& x_aug) const {
        return lqi.control(x_aug);
    }

    /**
     * @brief Compute control using the estimate and the controller's own integrator
     *
     * Pulls the plant estimate from the Kalman filter and the integral state from
     * the embedded LQI, so the caller only supplies the reference and measurement:
     * u = -[Kx Ki]·[x̂; xi], then xi advances by (r − y).
     *
     * @param r Output reference
     * @param y Measured output
     * @return Control input u
     */
    [[nodiscard]] constexpr ColVec<NU, T> control(const ColVec<NY, T>& r, const ColVec<NY, T>& y) {
        return lqi.control(kf.state(), r, y);
    }

    /// Clear the integral state (the Kalman estimate is left untouched).
    constexpr void reset() { lqi.reset(); }
};

} // namespace wet