#pragma once

/**
 * @file decomposition.hpp
 * @brief Matrix factorizations: Cholesky, LU (partial pivoting), and QR
 *        (modified Gram-Schmidt), plus the symmetry/Hermitian predicate used to
 *        choose between them. Linear-system solvers built on these live in
 *        solve.hpp; the eigensolver in eigen.hpp.
 */

#include <cstddef>
#include <tuple>
#include <type_traits>

#include "matrix.hpp"
#include "matrix_traits.hpp"
#include "wet/backend.hpp"
#include "wet/math/complex.hpp"

namespace wet {

namespace mat {

template<size_t N, typename T>
constexpr bool is_symmetric_or_hermitian(const Matrix<N, N, T>& A) {
    if constexpr (std::is_floating_point_v<T>) {
        constexpr auto tol = default_tol<T>();
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = i + 1; j < N; ++j) {
                if (wet::abs(A(i, j) - A(j, i)) > tol) {
                    return false;
                }
            }
        }
        return true;
    } else {
        constexpr auto tol = default_tol<T>();
        constexpr auto tol_sq = tol * tol;
        // Hermitian requires real diagonal elements
        for (size_t i = 0; i < N; ++i) {
            auto imag_diag = wet::imag(A(i, i));
            if (imag_diag * imag_diag > tol_sq) {
                return false;
            }
        }
        // Off-diagonal: A(i,j) == conj(A(j,i))
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = i + 1; j < N; ++j) {
                auto diff = A(i, j) - wet::conj(A(j, i));
                auto mag_sq = wet::real(diff * wet::conj(diff));
                if (mag_sq > tol_sq) {
                    return false;
                }
            }
        }
        return true;
    }
}

/**
 * @brief Cholesky decomposition for positive-definite matrices
 *
 * For real matrices, computes L such that A = LLᵀ (A must be symmetric).
 * For complex matrices, computes L such that A = LLᴴ (A must be Hermitian).
 *
 * @note Compare with MATLAB's chol(A, 'lower').
 * @see Golub & Van Loan, "Matrix Computations" (4th ed., 2013), §4.2
 *
 * @param A Symmetric positive-definite (real) or Hermitian positive-definite (complex) matrix
 * @return Lower-triangular L, or wet::nullopt if A is not PD or not symmetric/Hermitian
 */
template<size_t N, typename T>
constexpr wet::optional<Matrix<N, N, T>> cholesky(const Matrix<N, N, T>& A) {
    if (!is_symmetric_or_hermitian(A)) {
        return wet::nullopt;
    }

    Matrix<N, N, T> L = Matrix<N, N, T>::zeros();

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j <= i; ++j) {

            T sum = A(i, j);
            for (size_t k = 0; k < j; ++k) {
                sum -= L(i, k) * wet::conj(L(j, k));
            }

            if (i == j) {
                if constexpr (std::is_floating_point_v<T>) {
                    auto diag_val = sum;
                    if (diag_val <= 0) {
                        return wet::nullopt;
                    }
                    L(i, j) = wet::sqrt(diag_val);
                } else {
                    auto diag_val = wet::real(sum);
                    if (diag_val <= 0) {
                        return wet::nullopt;
                    }
                    L(i, j) = wet::sqrt(diag_val);
                }
            } else {
                L(i, j) = sum / L(j, j);
            }
        }
    }

    return L;
}

/**
 * @brief LU decomposition with partial pivoting
 *
 * Returns pair<L,U> and a pivot vector piv such that P*A = L*U
 * L: lower triangular with unit diagonal
 * U: upper triangular
 */
template<size_t N, typename T>
constexpr wet::optional<wet::tuple<Matrix<N, N, T>, Matrix<N, N, T>, wet::array<size_t, N>>>
lu_decomposition(const Matrix<N, N, T>& A) {
    Matrix<N, N, T>       L = Matrix<N, N, T>::identity();
    Matrix<N, N, T>       U = A;
    wet::array<size_t, N> piv;
    for (size_t i = 0; i < N; ++i) {
        piv[i] = i;
    }

    for (size_t i = 0; i < N; ++i) {
        // Partial pivot
        size_t max_row = i;
        auto   max_val = wet::abs(U(i, i));
        for (size_t r = i + 1; r < N; ++r) {
            auto val = wet::abs(U(r, i));
            if (val > max_val) {
                max_val = val;
                max_row = r;
            }
        }

        constexpr auto singularity_tol = std::is_same_v<decltype(max_val), float>
                                           ? decltype(max_val){1e-6}
                                           : decltype(max_val){1e-12};
        if (max_val < singularity_tol) {
            return wet::nullopt;
        }

        if (max_row != i) {
            // Swap rows in U
            for (size_t col = 0; col < N; ++col) {
                wet::swap(U(i, col), U(max_row, col));
            }
            // Swap previous columns in L
            for (size_t col = 0; col < i; ++col) {
                wet::swap(L(i, col), L(max_row, col));
            }
            // Track pivot
            wet::swap(piv[i], piv[max_row]);
        }

        // Elimination
        for (size_t j = i + 1; j < N; ++j) {
            auto factor = U(j, i) / U(i, i);
            L(j, i) = factor;
            for (size_t k = i; k < N; ++k) {
                U(j, k) -= factor * U(i, k);
            }
        }
    }

    return wet::make_tuple(L, U, piv);
}

/**
 * @brief QR decomposition via Gram-Schmidt orthogonalization
 *
 * Computes Q (orthogonal) and R (upper triangular) such that A = Q*R.
 * Uses modified Gram-Schmidt for better numerical stability.
 */
template<typename T, size_t N, size_t M>
struct QRDecomposition {
    Matrix<N, M, T> Q{};
    Matrix<M, M, T> R{};

    /**
     * @brief Check validity of QR decomposition
     * @return true if all diagonal elements of R are sufficiently non-zero
     */
    [[nodiscard]] constexpr bool is_valid() const {
        for (size_t i = 0; i < M; ++i) {
            if (wet::abs(R(i, i)) < T{1e-12}) {
                return false;
            }
        }
        return true;
    }
};

/**
 * @brief Perform QR decomposition on a matrix
 * @tparam T   Scalar type
 * @tparam N   Number of rows
 * @tparam M   Number of columns
 * @param A    Input matrix
 * @param eps  Tolerance for zero diagonal elements
 * @return QR decomposition with Q and R matrices
 */
template<typename T, size_t N, size_t M>
[[nodiscard]] constexpr QRDecomposition<T, N, M> qr_decompose(
    const Matrix<N, M, T>& A,
    T                      eps = T{1e-12}
) {
    QRDecomposition<T, N, M> result;
    result.Q = A; // Start with columns of A

    // Modified Gram-Schmidt
    for (size_t j = 0; j < M; ++j) {
        // Compute R(j,j) = norm of column j
        T col_norm_sq = T{0};
        for (size_t i = 0; i < N; ++i) {
            col_norm_sq += result.Q(i, j) * result.Q(i, j);
        }
        result.R(j, j) = wet::sqrt(col_norm_sq);

        if (result.R(j, j) > eps) {
            // Normalize column j
            for (size_t i = 0; i < N; ++i) {
                result.Q(i, j) /= result.R(j, j);
            }

            // Orthogonalize remaining columns against column j
            for (size_t k = j + 1; k < M; ++k) {
                // R(j,k) = dot(Q(:,j), Q(:,k))
                T dot_prod = T{0};
                for (size_t i = 0; i < N; ++i) {
                    dot_prod += result.Q(i, j) * result.Q(i, k);
                }
                result.R(j, k) = dot_prod;

                // Q(:,k) -= R(j,k) * Q(:,j)
                for (size_t i = 0; i < N; ++i) {
                    result.Q(i, k) -= result.R(j, k) * result.Q(i, j);
                }
            }
        }
    }

    return result;
}

/**
 * @brief Result of a full (complete) QR factorization.
 *
 * A = Q·R with Q an N×N orthogonal matrix and R N×M upper-triangular. Unlike
 * qr_decompose (thin modified-Gram-Schmidt, N×M Q), this retains the *full*
 * orthogonal factor: its leading rank columns span the range of A and its
 * trailing columns span the left null space (orthogonal complement). Those
 * complement columns are what robust pole placement (#16) needs from B.
 */
template<typename T, size_t N, size_t M>
struct FullQR {
    Matrix<N, N, T> Q{}; ///< Orthogonal factor (full N×N).
    Matrix<N, M, T> R{}; ///< Upper-triangular factor (N×M).
};

/**
 * @brief Full QR factorization via Householder reflections (real T).
 *
 * Numerically robust (orthogonal reflections, not Gram-Schmidt). For A with
 * full column rank M ≤ N, columns 0..M−1 of Q form an orthonormal basis of
 * range(A) and columns M..N−1 form an orthonormal basis of its complement
 * (Qᵀ_⊥·A = 0).
 *
 * @see Golub & Van Loan, "Matrix Computations" (4th ed., 2013), §5.2
 */
template<typename T, size_t N, size_t M>
[[nodiscard]] constexpr FullQR<T, N, M> full_qr(const Matrix<N, M, T>& A) {
    FullQR<T, N, M> out;
    out.Q = Matrix<N, N, T>::identity();
    out.R = A;

    constexpr size_t steps = (M < N) ? M : (N - 1); // columns to reflect
    for (size_t k = 0; k < steps; ++k) {
        // Householder vector v that zeroes R(k+1.., k).
        T norm_sq = T{0};
        for (size_t i = k; i < N; ++i) {
            norm_sq += out.R(i, k) * out.R(i, k);
        }
        T alpha = wet::sqrt(norm_sq);
        if (alpha == T{0}) {
            continue; // column already zero below the diagonal
        }
        if (out.R(k, k) > T{0}) {
            alpha = -alpha; // choose sign to avoid cancellation in v[k]
        }

        wet::array<T, N> v{};
        v[k] = out.R(k, k) - alpha;
        for (size_t i = k + 1; i < N; ++i) {
            v[i] = out.R(i, k);
        }
        T vtv = T{0};
        for (size_t i = k; i < N; ++i) {
            vtv += v[i] * v[i];
        }
        if (vtv == T{0}) {
            continue;
        }

        // Apply H = I − 2·v·vᵀ/(vᵀv) to R from the left: R −= (2/vᵀv)·v·(vᵀR).
        for (size_t j = 0; j < M; ++j) {
            T dot = T{0};
            for (size_t i = k; i < N; ++i) {
                dot += v[i] * out.R(i, j);
            }
            const T s = (T{2} * dot) / vtv;
            for (size_t i = k; i < N; ++i) {
                out.R(i, j) -= s * v[i];
            }
        }
        // Accumulate Q = Q·H from the right: Q −= (2/vᵀv)·(Q·v)·vᵀ.
        for (size_t i = 0; i < N; ++i) {
            T dot = T{0};
            for (size_t j = k; j < N; ++j) {
                dot += out.Q(i, j) * v[j];
            }
            const T s = (T{2} * dot) / vtv;
            for (size_t j = k; j < N; ++j) {
                out.Q(i, j) -= s * v[j];
            }
        }
    }
    return out;
}

} // namespace mat
} // namespace wet
