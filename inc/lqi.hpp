#pragma once

#include <cstddef>

#include "matrix.hpp"
#include "ricatti.hpp"
#include "stability.hpp"
#include "state_space.hpp"

namespace wetmelon::control {

// Result type goes here

namespace online {

/**
 * @struct LQIResult
 * @brief Runtime LQI design result (online namespace)
 */
template<size_t NX, size_t NU, size_t NY, typename T>
struct LQIResult {
    Matrix<NU, NX + NY, T>           K{};
    Matrix<NX + NY, NX + NY, T>      S{};
    ColVec<NX + NY, wet::complex<T>> e{};
    bool                             success{false};

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
 * @brief Linear-Quadratic Integral design for tracking with servo action (runtime version)
 *
 * @param sys  State-space system
 * @param Q    Augmented state cost matrix (state + integral error)
 * @param R    Input cost matrix
 * @param dof  Servo degrees of freedom (1DOF or 2DOF)
 *
 * @return LQIResult containing state and integral gains
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = 0, size_t NV = 0, typename T = double>
[[nodiscard]] constexpr LQIResult<NX, NU, NY, T> lqi(
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

    // Compute augmented gain
    const Matrix denom = R + B_aug.transpose() * P_aug * B_aug;
    const auto   denom_inv = denom.inverse();

    if (!denom_inv) {
        return LQIResult<NX, NU, NY, T>{};
    }
    Matrix<NU, NX + NY, T> K_aug{};
    K_aug = denom_inv.value() * B_aug.transpose() * P_aug * A_aug;

    ColVec<NX + NY, wet::complex<T>> poles = stability::closed_loop_poles(A_aug, B_aug, K_aug);
    return LQIResult<NX, NU, NY, T>{K_aug, P_aug, poles, true};
}

/**
 * @brief Design continuous LQR with integral action for state-space system
 *
 * @param sys State-space system
 * @param Q   Augmented state cost matrix (state + integral)
 * @param R   Input cost matrix
 * @param dof Servo degrees of freedom (1DOF or 2DOF)
 *
 * @return LQIResult containing gains and solution to DARE
 */
template<size_t NX, size_t NU, size_t NY, typename T = double>
[[nodiscard]] constexpr LQIResult<NX, NU, NY, T> lqr_with_integral(
    const StateSpace<NX, NU, NY, NX, NY, T>& sys,
    const Matrix<NX + NY, NX + NY, T>&       Q,
    const Matrix<NU, NU, T>&                 R
) {
    return lqi(sys, Q, R);
}

} // namespace online

namespace design {

/**
 * @struct LQIResult
 * @brief Linear-Quadratic Integral controller design result
 *
 * Contains gains and Riccati solution for servo control with integral action.
 */
template<size_t NX, size_t NU, size_t NY, typename T = double>
struct LQIResult {
    Matrix<NU, NX + NY, T>           K{};
    Matrix<NX + NY, NX + NY, T>      S{};
    ColVec<NX + NY, wet::complex<T>> e{};
    bool                             success{false};

    template<typename U>
    [[nodiscard]] consteval auto as() const {
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
 * @param sys  State-space system
 * @param Q    Augmented state cost matrix (state + integral error)
 * @param R    Input cost matrix
 * @return LQIResult containing state and integral gains
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = 0, size_t NV = 0, typename T = double>
[[nodiscard]] consteval LQIResult<NX, NU, NY, T> lqi(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NX + NY, NX + NY, T>&       Q,
    const Matrix<NU, NU, T>&                 R
) {
    auto result = online::lqi(sys, Q, R);
    return LQIResult<NX, NU, NY, T>{
        result.K,
        result.S,
        result.e,
        result.success
    };
}

/**
 * @brief Alias for LQR with integral action for tracking (consteval version)
 *
 * @param sys  State-space system
 * @param Q    Augmented state cost matrix (state + integral error)
 * @param R    Input cost matrix
 * @param dof  Servo degrees of freedom (1DOF or 2DOF)
 *
 * @return LQIResult containing state and integral gains
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = 0, size_t NV = 0, typename T = double>
[[nodiscard]] consteval LQIResult<NX, NU, NY, T> lqr_with_integral(
    const StateSpace<NX, NU, NY, NX, NY, T>& sys,
    const Matrix<NX + NY, NX + NY, T>&       Q,
    const Matrix<NU, NU, T>&                 R
) {
    return lqi(sys, Q, R);
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
private:
    Matrix<NU, NX + NY, T> K{};

public:
    constexpr LQI() = default;
    constexpr LQI(const Matrix<NU, NX + NY, T>& K_) : K(K_) {}

    // Compile-time only constructor for design:: results
    consteval LQI(const design::LQIResult<NX, NU, NY, T>& result) : K(result.K) {}

    // Runtime constructor for online:: results
    constexpr LQI(const online::LQIResult<NX, NU, NY, T>& result) : K(result.K) {}

    template<typename U>
    constexpr LQI(const LQI<NX, NU, NY, U>& other) : K(other.getK()) {}

    /**
     * @brief Compute control with integral action
     *
     * Computes u = -K * x_aug where x_aug = [x; xi]
     *
     * @param x_aug Augmented state vector [x; xi]
     * @return Control input vector u
     */
    [[nodiscard]] constexpr ColVec<NU, T> control(const ColVec<NX + NY, T>& x_aug) {
        return ColVec<NU, T>(-K * x_aug);
    }

    [[nodiscard]] constexpr const Matrix<NU, NX + NY, T>& getK() const { return K; }
};

} // namespace wetmelon::control