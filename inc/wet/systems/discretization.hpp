#pragma once

#include <cmath>

#include "state_space.hpp"
#include "wet/matrix/matrix.hpp"

namespace wet {

/**
 * @brief Discretization methods for continuous-time state-space systems
 *
 * @see "Feedback Control of Dynamic Systems" (Franklin et al., 2015), Chapter 8
 */
enum class DiscretizationMethod {
    ForwardEuler, //!< Explicit Euler: Ad = I + ATs (first-order, simple, low overhead)
    ZOH,          //!< Zero-Order Hold: Ad = e^(ATs), exact for piecewise-constant inputs
    Tustin,       //!< Bilinear transform: s → (2/Ts)(z−1)/(z+1), preserves stability
};

namespace detail {

/**
 * @brief Discretize using Forward Euler (explicit Euler)
 *
 *     A_d = I + ATs,  B_d = BTs,  C_d = C,  D_d = D
 *
 * First-order approximation of the matrix exponential. Simple and fast but
 * only accurate when ‖A‖��Ts ≪ 1. Does not preserve stability for stiff systems.
 *
 * @see "Feedback Control of Dynamic Systems" (Franklin et al., 2015), §8.3
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
[[nodiscard]] constexpr StateSpace<NX, NU, NY, NW, NV, T> discretize_forward_euler_impl(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    T                                        sampling_time
) {
    StateSpace<NX, NU, NY, NW, NV, T> sys_d{};

    // A_d = I + A*Ts
    sys_d.A = Matrix<NX, NX, T>::identity() + sys.A * sampling_time;

    // B_d = B*Ts
    sys_d.B = sys.B * sampling_time;

    // C_d = C (unchanged)
    sys_d.C = sys.C;

    // D_d = D (unchanged)
    sys_d.D = sys.D;

    // Set sampling time
    sys_d.Ts = sampling_time;

    // Process noise matrices (if present)
    if constexpr (NW > 0) {
        sys_d.G = sys.G * sampling_time;
    }

    // Measurement noise matrices (if present)
    if constexpr (NV > 0) {
        sys_d.H = sys.H;
    }

    return sys_d;
}

/**
 * @brief Discretize using Zero-Order Hold (ZOH)
 *
 *     A_d = e^(ATs)
 *     B_d = A⁻¹(e^(ATs) − I)B   [or series expansion if A is singular]
 *
 * Exact discretization assuming piecewise-constant input between samples.
 * Uses matrix exponential for A_d and LU solve for B_d (avoids forming A⁻¹).
 * Falls back to Taylor series B_d ≈ (ITs + ATs²/2! + A²Ts³/3! + ⋯)B when
 * A is singular.
 *
 * @note Compare with MATLAB's c2d(sys, Ts, 'zoh').
 * @see "Feedback Control of Dynamic Systems" (Franklin et al., 2015), §8.3
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
[[nodiscard]] constexpr StateSpace<NX, NU, NY, NW, NV, T> discretize_zoh_impl(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    T                                        sampling_time
) {
    // Compute A_d = exp(A * Ts)
    const Matrix A_scaled = sys.A * sampling_time;
    const Matrix exp_A_Ts = mat::expm(A_scaled);

    // Compute B_d = A⁻¹ * (exp(A*Ts) - I) * B  via solve: A * B_d = (exp(A*Ts) - I) * B
    const Matrix I = Matrix<NX, NX, T>::identity();
    const Matrix rhs = (exp_A_Ts - I) * sys.B;
    const auto   B_d_opt = mat::lu_solve(sys.A, rhs);

    Matrix<NX, NU, T> B_d;
    if (B_d_opt) {
        B_d = B_d_opt.value();
    } else {
        //! Fallback: Use series expansion directly (A may be singular)
        //! B_d ≈ (I*Ts + A*Ts²/2 + A²*Ts³/6 + ...) * B
        B_d = sys.B * sampling_time;
        Matrix A_power = sys.A;

        for (size_t n = 2; n <= 10; ++n) {
            T coeff = T{1};
            for (size_t i = 1; i <= n; ++i) {
                coeff *= (sampling_time / static_cast<T>(i));
            }
            B_d += A_power * sys.B * coeff;
            if (n < 10) {
                A_power = A_power * sys.A;
            }
        }
    }

    //! C_d = C, D_d = D (output equation is the same)
    const Matrix C_d = sys.C;
    const Matrix D_d = sys.D;

    //! G_d and H_d can be transformed similarly, but for simplicity use continuous versions
    //! (noise models don't discretize as simply without differential equations)
    const Matrix G_d = sys.G * sampling_time; //! Approximate
    const Matrix H_d = sys.H;                 //! Direct

    return StateSpace{exp_A_Ts, B_d, C_d, D_d, G_d, H_d, sampling_time};
}

/**
 * @brief Discretize a continuous-time state-space system using Tustin method
 *
 * Tustin (Bilinear Transform) implementation
 *    Maps: s → (2/Ts) · (z − 1) / (z + 1)
 *
 *    Let L = (I − A·Ts/2). All results are computed via LU solve
 *    against L rather than forming L⁻¹ explicitly:
 *      A_d:  solve L · A_d = (I + A·Ts/2)
 *      B_d:  solve L · X = B,  then B_d = Ts · X
 *      C_d:  solve Lᵀ · X = Cᵀ, then C_d = Xᵀ
 *      D_d = D + (Ts/2) · C_d · B
 *
 * @note Compare with MATLAB's c2d(sys, Ts, 'tustin').
 * @see "Feedback Control of Dynamic Systems" (Franklin et al., 2015), §8.6
 *
 * @param sys             Continuous-time state-space model
 * @param sampling_time   Desired sampling period for discrete system
 * @return Discretized state-space system
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
[[nodiscard]] constexpr StateSpace<NX, NU, NY, NW, NV, T> discretize_tustin_impl(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    T                                        sampling_time
) {
    const T      ts_half = sampling_time / T{2};
    const Matrix I = Matrix<NX, NX, T>::identity();
    const Matrix L = I - sys.A * ts_half;

    //! A_d: solve L · A_d = (I + A·Ts/2)
    const auto A_d_opt = mat::lu_solve(L, I + sys.A * ts_half);
    if (!A_d_opt) {
        return discretize_zoh_impl(sys, sampling_time);
    }
    const Matrix A_d = A_d_opt.value();

    //! B_d: solve L · X = B, then B_d = Ts · X
    const auto B_d_opt = mat::lu_solve(L, sys.B);
    if (!B_d_opt) {
        return discretize_zoh_impl(sys, sampling_time);
    }
    const Matrix B_d = B_d_opt.value() * sampling_time;

    //! C_d = C · L⁻¹ → solve Lᵀ · X = Cᵀ, then C_d = Xᵀ
    const auto C_d_t_opt = mat::lu_solve(L.transpose(), sys.C.transpose());
    if (!C_d_t_opt) {
        return discretize_zoh_impl(sys, sampling_time);
    }
    const Matrix C_d = C_d_t_opt.value().transpose();

    //! D_d = D + (Ts/2) · C_d · B
    const Matrix D_d = sys.D + C_d * sys.B * ts_half;

    //! G_d: solve L · X = G, then G_d = Ts · X (same transform as B)
    const auto G_d_opt = mat::lu_solve(L, sys.G);
    if (!G_d_opt) {
        return discretize_zoh_impl(sys, sampling_time);
    }
    const Matrix G_d = G_d_opt.value() * sampling_time;
    const Matrix H_d = sys.H;

    return StateSpace{A_d, B_d, C_d, D_d, G_d, H_d, sampling_time};
}

} // namespace detail

/**
 * @brief Discretize a continuous-time state-space system
 *
 * Converts ẋ = Ax + Bu to x[k+1] = A_d x[k] + B_d u[k] using the
 * specified discretization method.
 *
 * @note Compare with MATLAB's c2d(sys, Ts, method).
 * @see "Feedback Control of Dynamic Systems" (Franklin et al., 2015), Chapter 8
 *
 * @param sys           Continuous-time state-space model (Ts = 0)
 * @param sampling_time Desired sampling period [s]
 * @param method        Discretization method (ForwardEuler, ZOH, or Tustin)
 * @return Discrete-time state-space system with Ts set
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
[[nodiscard]] constexpr StateSpace<NX, NU, NY, NW, NV, T> discretize(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    T                                        sampling_time,
    DiscretizationMethod                     method = DiscretizationMethod::ZOH
) {
    if (sampling_time <= T{0}) {
        return sys; //! No discretization needed
    }

    if (sys.Ts > T{0}) {
        return sys; //! Already discrete
    }

    switch (method) {
        case DiscretizationMethod::ForwardEuler:
            return detail::discretize_forward_euler_impl(sys, sampling_time);
        case DiscretizationMethod::ZOH:
            return detail::discretize_zoh_impl(sys, sampling_time);
        case DiscretizationMethod::Tustin:
            return detail::discretize_tustin_impl(sys, sampling_time);
        default:
            return detail::discretize_forward_euler_impl(sys, sampling_time);
    }
}
} // namespace wet