#pragma once

#include <cmath>

#include "matrix.hpp"
#include "matrix_functions.hpp"
#include "state_space.hpp"

/**
 * @brief Discretization methods for continuous-time state-space systems
 */
enum class DiscretizationMethod {
    ZOH,    // Zero-Order Hold (exact for piecewise constant inputs)
    Tustin, // Bilinear Transform (preserves stability, good for filters)
};

namespace detail {

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
    const Matrix exp_A_Ts = mat::exp(A_scaled);

    // Compute B_d using: B_d = A^{-1} * (exp(A*Ts) - I) * B
    // For numerical stability, use the integral formula when A is near singular
    const auto A_inv = sys.A.inverse();

    Matrix<NX, NU, T> B_d;
    if (A_inv) {
        // Standard formula: B_d = A^{-1} * (exp(A*Ts) - I) * B
        const Matrix I = Matrix<NX, NX, T>::identity();
        B_d = A_inv.value() * (exp_A_Ts - I) * sys.B;
    } else {
        // Fallback: Use series expansion directly
        // B_d ≈ (I*Ts + A*Ts^2/2 + A^2*Ts^3/6 + ...) * B
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

    // C_d = C, D_d = D (output equation is the same)
    const Matrix C_d = sys.C;
    const Matrix D_d = sys.D;

    // G_d and H_d can be transformed similarly, but for simplicity use continuous versions
    // (noise models don't discretize as simply without differential equations)
    const Matrix G_d = sys.G * sampling_time; // Approximate
    const Matrix H_d = sys.H;                 // Direct

    return StateSpace{exp_A_Ts, B_d, C_d, D_d, G_d, H_d, sampling_time};
}

/**
 * @brief Discretize a continuous-time state-space system using Tustin method
 *
 * Tustin (Bilinear Transform) implementation
 *    Maps: `s -> (2/Ts) * (z - 1) / (z + 1)`
 *    For A, B, C, D matrices:
 *      `A_d = (I + A*Ts/2)^{-1} * (I - A*Ts/2)`
 *      `B_d = (I + A*Ts/2)^{-1} * B * Ts`
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

    // Compute (I + A*Ts/2) and its inverse
    const Matrix I_plus_A_ts2 = I + sys.A * ts_half;
    const auto   inv_I_plus = I_plus_A_ts2.inverse();

    if (!inv_I_plus) {
        // Fallback to ZOH if Tustin matrix is singular
        return discretize_zoh_impl(sys, sampling_time);
    }

    // A_d = (I + A*Ts/2)^{-1} * (I - A*Ts/2)
    const Matrix I_minus_A_ts2 = I - sys.A * ts_half;
    const Matrix A_d = inv_I_plus.value() * I_minus_A_ts2;

    // B_d = (I + A*Ts/2)^{-1} * B * Ts
    const Matrix B_d = inv_I_plus.value() * sys.B * sampling_time;

    // C_d ≈ C (output mapping typically unchanged)
    const Matrix C_d = sys.C;
    const Matrix D_d = sys.D;

    // Noise models (approximate)
    const Matrix G_d = sys.G * sampling_time;
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
        return sys; // No discretization needed
    }

    if (sys.Ts > T{0}) {
        return sys; // Already discrete
    }

    switch (method) {
        case DiscretizationMethod::ZOH:
            return detail::discretize_zoh_impl(sys, sampling_time);
        case DiscretizationMethod::Tustin:
            return detail::discretize_tustin_impl(sys, sampling_time);
        default:
            return detail::discretize_zoh_impl(sys, sampling_time);
    }
}
