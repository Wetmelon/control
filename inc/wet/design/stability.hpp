#pragma once

/**
 * @defgroup stability_analysis Stability Analysis
 * @brief Functions to analyze closed-loop stability of control systems
 *
 * For continuous systems: stable if all eigenvalues have Re(λ) < 0 (left half plane)
 * For discrete systems: stable if all eigenvalues have |λ| < 1 (inside unit circle)
 *
 * @note The eigenvalue-based routines (is_stable_discrete, stability_margin_*,
 *       closed_loop_poles) are capped at N ≤ 4. They use the closed-form
 *       mat::compute_eigenvalues, which returns fully-resolved complex
 *       eigenvalues. The N > 4 QR path (mat::compute_eigenvalues, used by
 *       riccati.hpp) only yields the real Schur diagonal — it does not resolve
 *       complex conjugate pairs, so it cannot report pole locations or |λ|
 *       accurately. Lift the cap only once a full eigen-solver exists.
 */
#include <cstddef>

#include "wet/math/complex.hpp"
#include "wet/matrix/eigen.hpp"
#include "wet/matrix/matrix.hpp"

namespace wet {
namespace stability {

// ============================================================================
// Structural Analysis (Controllability / Observability)
// ============================================================================

/**
 * @brief Compute the controllability matrix [B, AB, A²B, ..., A^(N-1)B]
 *
 * The system (A, B) is controllable iff controllability_matrix(A, B) has full row rank.
 *
 * @tparam NX Number of states
 * @tparam NU Number of inputs
 * @tparam T  Scalar type
 * @param A   State matrix
 * @param B   Input matrix
 * @return Matrix<NX, NX*NU, T> controllability matrix
 */
template<size_t NX, size_t NU, typename T = double>
[[nodiscard]] constexpr Matrix<NX, NX * NU, T>
controllability_matrix(const Matrix<NX, NX, T>& A, const Matrix<NX, NU, T>& B) noexcept {
    Matrix<NX, NX * NU, T> Co{};
    Matrix<NX, NU, T>      AB = B;
    for (size_t k = 0; k < NX; ++k) {
        for (size_t r = 0; r < NX; ++r) {
            for (size_t c = 0; c < NU; ++c) {
                Co(r, (k * NU) + c) = AB(r, c);
            }
        }
        if (k + 1 < NX) {
            AB = A * AB;
        }
    }
    return Co;
}

/**
 * @brief Compute the observability matrix [C; CA; CA²; ...; CA^(N-1)]
 *
 * The system (A, C) is observable iff observability_matrix(A, C) has full column rank.
 *
 * @tparam NX Number of states
 * @tparam NY Number of outputs
 * @tparam T  Scalar type
 * @param A   State matrix
 * @param C   Output matrix
 * @return Matrix<NX*NY, NX, T> observability matrix
 */
template<size_t NX, size_t NY, typename T = double>
[[nodiscard]] constexpr Matrix<NX * NY, NX, T>
observability_matrix(const Matrix<NX, NX, T>& A, const Matrix<NY, NX, T>& C) noexcept {
    Matrix<NX * NY, NX, T> Ob{};
    Matrix<NY, NX, T>      CA = C;
    for (size_t k = 0; k < NX; ++k) {
        for (size_t r = 0; r < NY; ++r) {
            for (size_t c = 0; c < NX; ++c) {
                Ob((k * NY) + r, c) = CA(r, c);
            }
        }
        if (k + 1 < NX) {
            CA = CA * A;
        }
    }
    return Ob;
}

/**
 * @brief Compute rank of a matrix via Gaussian elimination with partial pivoting
 *
 * @param M  Input matrix
 * @param tol Tolerance for zero detection (default: 1e-10)
 * @return size_t rank of the matrix
 */
template<size_t R, size_t C, typename T>
[[nodiscard]] constexpr size_t rank(const Matrix<R, C, T>& M, T tol = T{1e-10}) noexcept {
    return mat::rank(M, tol);
}

/**
 * @brief Check if a system is controllable
 *
 * @param A State matrix
 * @param B Input matrix
 * @return true if the controllability matrix has full rank (NX)
 */
template<size_t NX, size_t NU, typename T = double>
[[nodiscard]] constexpr bool is_controllable(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    T                        tol = T{1e-10}
) noexcept {
    auto Co = controllability_matrix(A, B);
    return rank(Co, tol) == NX;
}

/**
 * @brief Check if a system is observable
 *
 * @param A State matrix
 * @param C Output matrix
 * @return true if the observability matrix has full rank (NX)
 */
template<size_t NX, size_t NY, typename T = double>
[[nodiscard]] constexpr bool is_observable(
    const Matrix<NX, NX, T>& A,
    const Matrix<NY, NX, T>& C,
    T                        tol = T{1e-10}
) noexcept {
    auto Ob = observability_matrix(A, C);
    return rank(Ob, tol) == NX;
}

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
    auto eigen = mat::compute_eigenvalues(A);
    if (!eigen.converged) {
        return false;
    }

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
    auto eigen = mat::compute_eigenvalues(A);
    if (!eigen.converged) {
        return T{-1}; // Negative margin signals "not provably stable"
    }

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
    auto eigen = mat::compute_eigenvalues(A);
    if (!eigen.converged) {
        return T{-1}; // Return unstable indicator
    }

    T max_mag = T{0};
    for (size_t i = 0; i < N; ++i) {
        const T magnitude = eigen.values[i].abs();
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
    auto              eigen = mat::compute_eigenvalues(A_cl);
    return eigen.values;
}

} // namespace stability
} // namespace wet
