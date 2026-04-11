#pragma once

#include <cmath>

#include "matrix.hpp"
#include "matrix/cholesky.hpp"
#include "state_space.hpp"

namespace wetmelon::control {
/**
 * @brief Discretization methods for continuous-time state-space systems
 */
enum class DiscretizationMethod {
    ForwardEuler, //!< Explicit Euler method (simple, low overhead)
    ZOH,          //!< Zero-Order Hold (exact for piecewise constant inputs)
    Tustin,       //!< Bilinear Transform (preserves stability, good for filters)
};

namespace detail {

/**
 * @brief Discretize a continuous-time state-space system using Forward Euler method
 *
 * Forward Euler implementation where:
 * ```latex
 * A_d = I + A*Ts
 * B_d = B*Ts
 * C_d = C
 * D_d = D
 * ```
 * @tparam NX number of states
 * @tparam NU number of inputs
 * @tparam NY number of outputs
 * @tparam NW number of process noise inputs
 * @tparam NV number of measurement noise inputs
 * @tparam T scalar type
 * @param sys continuous-time state-space system
 * @param sampling_time sampling period for discrete system
 * @return constexpr StateSpace<NX, NU, NY, NW, NV, T>
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
 * @brief Discretize a continuous-time state-space system using Zero-Order Hold (ZOH) method
 *
 * Zero-Order Hold (ZOH) implementation where:
 * ```latex
 * A_d = exp(A*Ts)
 * B_d = integral from 0 to Ts of exp(A*tau)*B*dtau
 *     = A^{-1} * (exp(A*Ts) - I) * B  [if A is invertible]
 *     = (A*Ts + (A*Ts)^2/2! + ...) * B [via series expansion]
 * ```
 * @tparam NX number of states
 * @tparam NU number of inputs
 * @tparam NY number of outputs
 * @tparam NW number of process noise inputs
 * @tparam NV number of measurement noise inputs
 * @tparam T scalar type
 * @param sys continuous-time state-space system
 * @param sampling_time sampling period for discrete system
 * @return constexpr StateSpace<NX, NU, NY, NW, NV, T>
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
 *    Maps: `s -> (2/Ts) * (z - 1) / (z + 1)`
 *    For A, B, C, D matrices (let M = (I - A*Ts/2)⁻¹):
 *      `A_d = M * (I + A*Ts/2)`
 *      `B_d = Ts * M * B`
 *      `C_d = C * M`
 *      `D_d = D + (Ts/2) * C * M * B`
 *
 * @param sys             Continuous-time state-space model
 * @param sampling_time   Desired sampling period for discrete system
 * @return constexpr StateSpace<NX, NU, NY, NW, NV, T>
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
[[nodiscard]] constexpr StateSpace<NX, NU, NY, NW, NV, T> discretize_tustin_impl(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    T                                        sampling_time
) {
    const T      ts_half = sampling_time / T{2};
    const Matrix I = Matrix<NX, NX, T>::identity();

    //! Compute M = (I - A*Ts/2)⁻¹
    const Matrix I_minus_A_ts2 = I - sys.A * ts_half;
    const auto   M = I_minus_A_ts2.inverse();

    if (!M) {
        //! Fallback to ZOH if Tustin matrix is singular
        return discretize_zoh_impl(sys, sampling_time);
    }

    //! A_d = M * (I + A*Ts/2)
    const Matrix I_plus_A_ts2 = I + sys.A * ts_half;
    const Matrix A_d = M.value() * I_plus_A_ts2;

    //! B_d = Ts * M * B
    const Matrix B_d = M.value() * sys.B * sampling_time;

    //! C_d = C * M
    const Matrix C_d = sys.C * M.value();

    //! D_d = D + (Ts/2) * C_d * B  (where C_d already contains M)
    const Matrix D_d = sys.D + C_d * sys.B * ts_half;

    //! Noise input: G_d = Ts * M * G  (same transformation as B)
    const Matrix G_d = M.value() * sys.G * sampling_time;
    const Matrix H_d = sys.H;

    return StateSpace{A_d, B_d, C_d, D_d, G_d, H_d, sampling_time};
}

} // namespace detail

/**
 * @brief Discretize a continuous-time state-space system
 *
 * @param sys           Continuous-time state-space model (Ts should be 0)
 * @param sampling_time Desired sampling period for discrete system
 * @param method        Discretization method (ZOH or Tustin)
 * @return constexpr StateSpace<NX, NU, NY, NW, NV, T>
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
} // namespace wetmelon::control