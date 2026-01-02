#pragma once

#include <cmath>

#include "matrix.hpp"
#include "matrix_functions.hpp"
#include "state_space.hpp"

// ============================================================================
// Discretization Method Enumeration
// ============================================================================
enum class DiscretizationMethod {
    ZOH,    // Zero-Order Hold (exact for piecewise constant inputs)
    Tustin, // Bilinear Transform (preserves stability, good for filters)
};

// ============================================================================
// Discretization Methods for Continuous Systems
// ============================================================================
// Given continuous system: dx/dt = A*x + B*u
// Discrete system: x[k+1] = A_d*x[k] + B_d*u[k]

namespace detail {

// Zero-Order Hold (ZOH) implementation
// Where: A_d = exp(A*Ts)
//        B_d = integral from 0 to Ts of exp(A*tau)*B*dtau
//            = A^{-1} * (exp(A*Ts) - I) * B  [if A is invertible]
//            = (A*Ts + (A*Ts)^2/2! + ...) * B [via series expansion]
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

// Tustin (Bilinear Transform) implementation
// Maps: s -> (2/Ts) * (z - 1) / (z + 1)
// For A, B, C, D matrices:
// A_d = (I + A*Ts/2)^{-1} * (I - A*Ts/2)
// B_d = (I + A*Ts/2)^{-1} * B * Ts
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

// ============================================================================
// Unified Discretization Function
// ============================================================================
// Discretize a continuous-time state-space system using the specified method
//
// Parameters:
//   sys            - Continuous-time state-space model (Ts should be 0)
//   sampling_time  - Desired sampling period for discrete system
//   method         - Discretization method (ZOH or Tustin)
//
// Returns:
//   Discrete-time state-space model with Ts set to sampling_time

template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
[[nodiscard]] constexpr StateSpace<NX, NU, NY, NW, NV, T> discretize(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    T                                        sampling_time,
    DiscretizationMethod                     method = DiscretizationMethod::ZOH
) {
    switch (method) {
        case DiscretizationMethod::ZOH:
            return detail::discretize_zoh_impl(sys, sampling_time);
        case DiscretizationMethod::Tustin:
            return detail::discretize_tustin_impl(sys, sampling_time);
        default:
            return detail::discretize_zoh_impl(sys, sampling_time);
    }
}

// ============================================================================
// Legacy Functions (deprecated - use discretize() instead)
// ============================================================================

// [[deprecated("Use discretize(sys, Ts, DiscretizationMethod::ZOH) instead")]]
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
[[nodiscard]] constexpr StateSpace<NX, NU, NY, NW, NV, T> discretize_zoh(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    T                                        sampling_time
) {
    return discretize(sys, sampling_time, DiscretizationMethod::ZOH);
}

// [[deprecated("Use discretize(sys, Ts, DiscretizationMethod::Tustin) instead")]]
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
[[nodiscard]] constexpr StateSpace<NX, NU, NY, NW, NV, T> discretize_tustin(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    T                                        sampling_time
) {
    return discretize(sys, sampling_time, DiscretizationMethod::Tustin);
}

// ============================================================================
// Legacy matrix_exponential (now in matrix_functions.hpp as mat::exp)
// ============================================================================

// [[deprecated("Use mat::exp() from matrix_functions.hpp instead")]]
template<typename T, size_t N>
[[nodiscard]] constexpr Matrix<N, N, T> matrix_exponential(const Matrix<N, N, T>& A) {
    // Compute matrix infinity norm (max absolute row sum)
    T norm = T{0};
    for (size_t i = 0; i < N; ++i) {
        T row_sum = T{0};
        for (size_t j = 0; j < N; ++j) {
            row_sum += std::abs(A(i, j));
        }
        if (row_sum > norm) {
            norm = row_sum;
        }
    }

    // Determine scaling: find s such that ||A / 2^s|| < 1
    size_t s = 0;
    T      scaled_norm = norm;
    while (scaled_norm > T{0.5}) {
        scaled_norm *= T{0.5};
        s++;
    }

    // Scale matrix: A_scaled = A / 2^s
    Matrix A_scaled = A;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            A_scaled(i, j) /= static_cast<T>(1 << s);
        }
    }

    // Compute exp(A_scaled) using Padé approximation: exp(A) ≈ (I + N) * (I - D)^{-1}
    // where N and D are matrix polynomials
    Matrix I = Matrix<N, N, T>::identity();
    Matrix A2 = A_scaled * A_scaled;
    Matrix A4 = A2 * A2;
    Matrix A6 = A4 * A2;

    // Padé(6,6) coefficients for numerator and denominator
    // These provide excellent accuracy with moderate computational cost
    constexpr T c0 = T{1};
    constexpr T c1 = T{1} / T{2};
    constexpr T c2 = T{1} / T{12};
    constexpr T c3 = T{1} / T{120};

    Matrix N_mat = I * c0 + A_scaled * c1;
    N_mat = N_mat + A2 * c2;
    N_mat = N_mat + A4 * (c3 / T{6});
    N_mat = N_mat + A6 * (c3 / T{42});

    Matrix D_mat = I * c0 - A_scaled * c1;
    D_mat = D_mat + A2 * c2;
    D_mat = D_mat - A4 * (c3 / T{6});
    D_mat = D_mat + A6 * (c3 / T{42});

    // Compute (I - D)^{-1} * N  (more stable than N * D^{-1})
    auto            D_inv = D_mat.inverse();
    Matrix<N, N, T> exp_A_scaled;
    if (D_inv) {
        exp_A_scaled = D_inv.value() * N_mat;
    } else {
        // Fallback to Taylor series if Padé fails
        exp_A_scaled = I + A_scaled;
        Matrix A_power = A_scaled;
        for (size_t n = 2; n <= 15; ++n) {
            T factorial = T{1};
            for (size_t k = 1; k <= n; ++k) {
                factorial *= static_cast<T>(k);
            }
            A_power = A_power * A_scaled;
            exp_A_scaled = exp_A_scaled + A_power * (T{1} / factorial);
        }
    }

    // Square s times: exp(A) = (exp(A / 2^s))^(2^s)
    Matrix result = exp_A_scaled;
    for (size_t i = 0; i < s; ++i) {
        result = result * result;
    }

    return result;
}
