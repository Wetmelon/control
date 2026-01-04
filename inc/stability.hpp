#pragma once

/**
 * @defgroup stability_analysis Stability Analysis
 * @brief Functions to analyze closed-loop stability of control systems
 *
 * For continuous systems: stable if all eigenvalues have Re(λ) < 0 (left half plane)
 * For discrete systems: stable if all eigenvalues have |λ| < 1 (inside unit circle)
 */
#include <cstddef>

#include "constexpr_complex.hpp"
#include "matrix.hpp"

namespace wetmelon::control {
namespace stability {

/**
 * @brief Check if a discrete-time system matrix A is stable
 *
 * A discrete system is stable if all eigenvalues have magnitude less than 1 (inside unit circle).
 *
 * @tparam N   Number of states (must be ≤ 4)
 * @tparam T   Numeric type (default: double)
 * @param A    State matrix to check
 *
 * @return true if all eigenvalues satisfy |λ| < 1, false otherwise
 */
template<size_t N, typename T = double>
[[nodiscard]] constexpr bool is_stable_discrete(const Matrix<N, N, T>& A) {
    static_assert(N <= 4, "Stability analysis only supported for systems up to 4 states");
    auto eigen = compute_eigenvalues(A);
    if (!eigen.converged)
        return false;

    for (size_t i = 0; i < N; ++i) {
        T magnitude = wet::abs(eigen.values[i]);
        if (magnitude >= T{1}) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Check closed-loop stability for discrete system with state feedback
 *
 * Checks stability of the closed-loop system A_cl = A - B*K with feedback u = -K*x.
 *
 * @tparam NX  Number of states
 * @tparam NU  Number of inputs
 * @tparam T   Numeric type (default: double)
 * @param A    State matrix
 * @param B    Control input matrix
 * @param K    State feedback gain matrix (u = -K*x)
 *
 * @return true if closed-loop system is stable, false otherwise
 */
template<size_t NX, size_t NU, typename T = double>
[[nodiscard]] constexpr bool is_closed_loop_stable_discrete(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NU, NX, T>& K
) {
    Matrix<NX, NX, T> A_cl = A - B * K;
    return is_stable_discrete(A_cl);
}

/**
 * @brief Compute stability margin for continuous system
 *
 * Returns the distance to the stability boundary (imaginary axis).
 * Computed as the negative of the most positive real eigenvalue part.
 * Positive values indicate stability; larger values indicate more stability margin.
 *
 * @tparam N   Number of states (must be ≤ 4)
 * @tparam T   Numeric type (default: double)
 * @param A    State matrix
 *
 * @return Stability margin (positive = stable, negative = unstable)
 */
template<size_t N, typename T = double>
[[nodiscard]] constexpr T stability_margin_continuous(const Matrix<N, N, T>& A) {
    static_assert(N <= 4, "Stability margin only supported for systems up to 4 states");
    auto eigen = compute_eigenvalues(A);
    if (!eigen.converged)
        return T{1}; // Return unstable indicator

    T max_real = eigen.values[0].real();
    for (size_t i = 1; i < N; ++i) {
        if (eigen.values[i].real() > max_real) {
            max_real = eigen.values[i].real();
        }
    }
    return -max_real; // Positive means stable, larger is more stable
}

/**
 * @brief Compute stability margin for discrete system
 *
 * Returns the distance to the stability boundary (unit circle).
 * Computed as 1 - (maximum magnitude eigenvalue).
 * Positive values indicate stability; larger values indicate more stability margin.
 *
 * @tparam N   Number of states (must be ≤ 4)
 * @tparam T   Numeric type (default: double)
 * @param A    State matrix
 *
 * @return Stability margin (positive = stable, negative = unstable)
 */
template<size_t N, typename T = double>
[[nodiscard]] constexpr T stability_margin_discrete(const Matrix<N, N, T>& A) {
    static_assert(N <= 4, "Stability margin only supported for systems up to 4 states");
    auto eigen = compute_eigenvalues(A);
    if (!eigen.converged)
        return T{-1}; // Return unstable indicator

    T max_mag = T{0};
    for (size_t i = 0; i < N; ++i) {
        T magnitude = wet::sqrt(
            eigen.values[i].real() * eigen.values[i].real() + eigen.values[i].imag() * eigen.values[i].imag()
        );
        if (magnitude > max_mag) {
            max_mag = magnitude;
        }
    }
    return T{1} - max_mag; // Positive means stable, larger is more stable
}

/**
 * @brief Compute closed-loop poles (eigenvalues) with state feedback
 *
 * Computes the eigenvalues of the closed-loop state matrix A_cl = A - B*K.
 * These poles determine the closed-loop system dynamics.
 *
 * @tparam NX  Number of states (must be ≤ 4)
 * @tparam NU  Number of inputs
 * @tparam T   Numeric type (default: double)
 * @param A    State matrix
 * @param B    Control input matrix
 * @param K    State feedback gain matrix (u = -K*x)
 *
 * @return Vector of closed-loop pole locations (eigenvalues as complex numbers)
 */
template<size_t NX, size_t NU, typename T = double>
[[nodiscard]] constexpr ColVec<NX, wet::complex<T>> closed_loop_poles(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NU, NX, T>& K
) {
    static_assert(NX <= 4, "Pole computation only supported for systems up to 4 states");
    Matrix<NX, NX, T> A_cl = A - B * K;
    auto              eigen = compute_eigenvalues(A_cl);
    return eigen.values;
}

} // namespace stability
} // namespace wetmelon::control
