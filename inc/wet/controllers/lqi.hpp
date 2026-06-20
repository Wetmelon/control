#pragma once

#include <cstddef>

#include "wet/design/riccati.hpp"
#include "wet/design/stability.hpp"
#include "wet/matrix/matrix.hpp"
#include "wet/systems/state_space.hpp"

namespace wet {
namespace design {

/**
 * @struct LQIResult
 * @brief LQI design result
 *
 * Contains the optimal gain for integral tracking control.
 * Use `.as<float>()` to convert for embedded deployment.
 */
template<size_t NX, size_t NU, size_t NY, typename T = double>
struct LQIResult {
    Matrix<NU, NX + NY, T>           K{};            ///< Optimal gain: u = -K*[x; xi]
    Matrix<NX + NY, NX + NY, T>      S{};            ///< Riccati equation solution
    ColVec<NX + NY, wet::complex<T>> e{};            ///< Closed-loop poles
    bool                             success{false}; ///< true if DARE converged

    template<typename U>
    [[nodiscard]] constexpr auto as() const {
        return LQIResult<NX, NU, NY, U>{
            K.template as<U>(),
            S.template as<U>(),
            e.template as<wet::complex<U>>(),
            success
        };
    }
};

/**
 * @brief Linear-Quadratic Integral design for tracking with servo action
 *
 * @note Compare with MATLAB's lqi(sys, Q, R) — exposed as the lqi() alias in matlab.hpp.
 *
 * @param sys  State-space system
 * @param Q    Augmented state cost matrix (state + integral error)
 * @param R    Input cost matrix
 *
 * @return LQIResult containing state and integral gains
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = 0, size_t NV = 0, typename T = double>
[[nodiscard]] constexpr LQIResult<NX, NU, NY, T> discrete_lqi(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NX + NY, NX + NY, T>&       Q,
    const Matrix<NU, NU, T>&                 R
) {
    // Build augmented system
    Matrix<NX + NY, NX + NY, T> A_aug{};
    Matrix<NX + NY, NU, T>      B_aug{};

    /* Canonical form for LQI augmentation:
     *
     * A_aug = |  A      0 |
     *         | -C      I |
     *
     * B_aug = | B |
     *         | 0 |
     */

    // Top-left block: original A
    A_aug.template block<NX, NX>(0, 0) = sys.A;

    // Top-right block: C matrix for integral action
    A_aug.template block<NY, NX>(NX, 0) = -sys.C;

    // Top block of B_aug: original B
    B_aug.template block<NX, NU>(0, 0) = sys.B;

    // Bottom-right block: integrator memory
    A_aug.template block<NY, NY>(NX, NX) = Matrix<NY, NY, T>::identity();

    // Solve DARE for augmented system
    const auto dare_opt = dare(A_aug, B_aug, Q, R);
    if (!dare_opt) {
        return LQIResult<NX, NU, NY, T>{};
    }
    Matrix<NX + NY, NX + NY, T> P_aug = dare_opt.value();

    const auto K_opt = lqr_gain(A_aug, B_aug, P_aug, R);
    if (!K_opt) {
        return LQIResult<NX, NU, NY, T>{};
    }
    Matrix<NU, NX + NY, T> K_aug = K_opt.value();

    ColVec<NX + NY, wet::complex<T>> poles = stability::closed_loop_poles(A_aug, B_aug, K_aug);
    return LQIResult<NX, NU, NY, T>{K_aug, P_aug, poles, true};
}
} // namespace design

/**
 * @ingroup discrete_controllers
 * @brief Linear-Quadratic-Integral (LQI) controller
 *
 * Output tracking controller with integral action: u = -K * [x; xi]
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
    Matrix<NU, NX + NY, T> K{};  ///< Optimal gain: u = -K*[x; xi]
    ColVec<NY, T>          xi{}; ///< Integral of tracking error, owned by the controller

    constexpr LQI() = default;
    constexpr LQI(const Matrix<NU, NX + NY, T>& K_) : K(K_) {} // NOLINT

    constexpr LQI(const design::LQIResult<NX, NU, NY, T>& result) : K(result.K) {} // NOLINT

    template<typename U>
    constexpr LQI(const LQI<NX, NU, NY, U>& other) : K(other.K.template as<T>()), xi(other.xi.template as<T>()) {} // NOLINT

    /**
     * @brief Compute control from a caller-supplied augmented state
     *
     * Computes u = -K * x_aug where x_aug = [x; xi]. Use this when you maintain
     * the integral state yourself; it does not touch the internal integrator.
     *
     * @param x_aug Augmented state vector [x; xi]
     * @return Control input vector u
     */
    [[nodiscard]] constexpr ColVec<NU, T> control(const ColVec<NX + NY, T>& x_aug) const {
        return ColVec<NU, T>(-K * x_aug);
    }

    /**
     * @brief Compute control with the controller's own integral action
     *
     * Uses the internal integrator state: u = -[Kx Ki]·[x; xi] with the current
     * xi, then advances xi[k+1] = xi[k] + (r - y) (matching the −C augmentation
     * sign convention). Gives zero steady-state error to constant references
     * without the caller tracking the integral.
     *
     * @param x Current (estimated or measured) plant state
     * @param r Output reference
     * @param y Measured output
     * @return Control input vector u
     */
    [[nodiscard]] constexpr ColVec<NU, T> control(const ColVec<NX, T>& x, const ColVec<NY, T>& r, const ColVec<NY, T>& y) {
        const auto    Kx = K.template block<NU, NX>(0, 0);
        const auto    Ki = K.template block<NU, NY>(0, NX);
        ColVec<NU, T> u = ColVec<NU, T>(-(Kx * x + Ki * xi));
        xi = ColVec<NY, T>(xi + (r - y));
        return u;
    }

    /// Clear the integral state.
    constexpr void reset() { xi = ColVec<NY, T>{}; }

    [[nodiscard]] constexpr const Matrix<NU, NX + NY, T>& getK() const { return K; }
};

} // namespace wet