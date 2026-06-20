#pragma once

#include <cstddef>

#include "lqr.hpp"
#include "wet/estimation/kalman.hpp"
#include "wet/matrix/matrix.hpp"
#include "wet/systems/state_space.hpp"

namespace wet {

namespace design {

/**
 * @struct LQGResult
 * @brief LQG design result
 *
 * Combines LQR and Kalman filter designs for separation principle-based control.
 * Use `.as<float>()` to convert for embedded deployment.
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
struct LQGResult {
    LQRResult<NX, NU, T>                lqr{};          ///< LQR design result
    KalmanResult<NX, NU, NY, NW, NV, T> kalman{};       ///< Kalman filter result
    bool                                success{false}; ///< true if both designs succeeded

    template<typename U>
    [[nodiscard]] constexpr auto as() const {
        return LQGResult<NX, NU, NY, NW, NV, U>{
            lqr.template as<U>(),
            kalman.template as<U>(),
            success
        };
    }

    /**
     * @brief Convert the LQG regulator to a discrete state-space block
     *
     * Realizes the dynamic output-feedback compensator mapping measurement y to
     * control u, with estimator state x̂. Lets the regulator drop into Bode/
     * `feedback`/`series` analysis. Prediction-form estimator with the
     * steady-state Kalman gain L and feedback u = −K·x̂:
     * @f[
     *   A_c = A - BK - LC + LDK,\quad B_c = L,\quad C_c = -K,\quad D_c = 0.
     * @f]
     *
     * @return StateSpace with NX states, NY inputs (y), NU outputs (u)
     */
    [[nodiscard]] constexpr StateSpace<NX, NY, NU, 0, 0, T> to_ss() const {
        const auto& A = kalman.sys.A;
        const auto& B = kalman.sys.B;
        const auto& C = kalman.sys.C;
        const auto& D = kalman.sys.D;
        const auto& L = kalman.L;
        const auto& K = lqr.K;

        return StateSpace<NX, NY, NU, 0, 0, T>{
            .A = A - (B * K) - (L * C) + (L * D * K),
            .B = L,
            .C = -K,
            .D = Matrix<NU, NY, T>{},
            .Ts = kalman.sys.Ts,
        };
    }
};

/**
 * @brief Linear-Quadratic-Gaussian regulator design combining LQR and Kalman filter
 *
 * @note Compare with MATLAB's lqg(sys, ...) — exposed as the lqg() alias in matlab.hpp.
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
[[nodiscard]] constexpr LQGResult<NX, NU, NY, NW, NV, T> discrete_lqg(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NX, NX, T>&                 Q_lqr,
    const Matrix<NU, NU, T>&                 R_lqr,
    const Matrix<NW, NW, T>&                 Q_kf,
    const Matrix<NV, NV, T>&                 R_kf,
    const Matrix<NX, NU, T>&                 N = Matrix<NX, NU, T>{}
) {
    const auto lqr_result = discrete_lqr(sys.A, sys.B, Q_lqr, R_lqr, N);
    const auto kalman_result = kalman(sys, Q_kf, R_kf);
    return LQGResult<NX, NU, NY, NW, NV, T>{lqr_result, kalman_result, lqr_result.success && kalman_result.success};
}

/**
 * @brief Combine separate Kalman filter and LQR designs into an LQG controller
 *
 * @note Compare with MATLAB's lqgreg(kest, k) — exposed as the lqgreg() alias in matlab.hpp.
 *
 * @param kest        Kalman filter design result
 * @param lqr_result  LQR controller design result
 *
 * @return LQGResult combining the provided Kalman and LQR designs
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
[[nodiscard]] constexpr LQGResult<NX, NU, NY, NW, NV, T> lqg_from_parts(
    const KalmanResult<NX, NU, NY, NW, NV, T>& kest,
    const LQRResult<NX, NU, T>&                lqr_result
) {
    return LQGResult<NX, NU, NY, NW, NV, T>{lqr_result, kest, lqr_result.success && kest.success};
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

    constexpr LQG(const design::LQGResult<NX, NU, NY, NW, NV, T>& result) // NOLINT
        : lqr(result.lqr), kf(result.kalman) {}

    template<typename U>
    constexpr LQG(const LQG<NX, NU, NY, NW, NV, U>& other) : lqr(other.lqr), kf(other.kf) {} // NOLINT

    constexpr void predict(const ColVec<NU, T>& u = ColVec<NU, T>{}) { kf.predict(u); }
    constexpr bool update(const ColVec<NY, T>& y, const ColVec<NU, T>& u = ColVec<NU, T>{}) { return kf.update(y, u); }

    [[nodiscard]] constexpr ColVec<NU, T> control() const { return lqr.control(kf.state()); }
    [[nodiscard]] constexpr ColVec<NU, T> control(const ColVec<NX, T>& x_ref) const { return lqr.control(kf.state(), x_ref); }
};

} // namespace wet