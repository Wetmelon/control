#pragma once

#include <cmath>

#include "constexpr_math.hpp"
#include "matrix.hpp"

// ============================================================================
// QR Decomposition via Gram-Schmidt
// ============================================================================
// Computes Q (orthogonal) and R (upper triangular) such that A = Q*R
// Uses modified Gram-Schmidt for better numerical stability

template<typename T, size_t N, size_t M>
struct QRDecomposition {
    Matrix<N, M, T> Q{};
    Matrix<M, M, T> R{};

    [[nodiscard]] constexpr bool is_valid() const {
        // Check if any diagonal element of R is near zero
        for (size_t i = 0; i < M; ++i) {
            if (std::abs(R(i, i)) < T{1e-12}) {
                return false;
            }
        }
        return true;
    }
};

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

// ============================================================================
// QR Algorithm for Eigenvalues
// ============================================================================
// Computes eigenvalues via iterative QR decomposition
// For symmetric matrices, converges to diagonal matrix with eigenvalues
// For general matrices, converges to quasi-upper-triangular Schur form

template<typename T, size_t N>
struct EigenResult {
    Matrix<N, N, T> eigenvalues_real{}; // Real parts (diagonal for symmetric)
    Matrix<N, N, T> eigenvalues_imag{}; // Imaginary parts (zero for symmetric)
    Matrix<N, N, T> eigenvectors{};     // Columns are eigenvectors
    bool            converged = false;
};

template<typename T, size_t N>
[[nodiscard]] constexpr EigenResult<T, N> compute_eigenvalues_qr(
    const Matrix<N, N, T>& A,
    size_t                 max_iter = 100,
    T                      tol = T{1e-8}
) {
    EigenResult<T, N> result;
    Matrix<N, N, T>   Ak = A;
    Matrix<N, N, T>   Q_total = Matrix<N, N, T>::identity();

    for (size_t iter = 0; iter < max_iter; ++iter) {
        // QR decomposition of current matrix
        auto qr = qr_decompose(Ak, tol);
        if (!qr.is_valid()) {
            break;
        }

        // Update: A_{k+1} = R*Q
        Ak = qr.R * qr.Q;

        // Accumulate eigenvector transformation
        Q_total = Q_total * qr.Q;

        // Check convergence: off-diagonal elements should approach zero
        T off_diag_norm = T{0};
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                if (i != j) {
                    off_diag_norm += std::abs(Ak(i, j));
                }
            }
        }

        if (off_diag_norm < tol) {
            result.converged = true;
            break;
        }
    }

    // Extract eigenvalues from diagonal (for symmetric matrices)
    for (size_t i = 0; i < N; ++i) {
        result.eigenvalues_real(i, i) = Ak(i, i);
    }

    result.eigenvectors = Q_total;
    return result;
}

// ============================================================================
// Ordered Schur Decomposition
// ============================================================================
// Separates eigenvalues/eigenvectors into stable (Re(λ) < 0 for continuous,
// |λ| < 1 for discrete) and unstable subspaces. This is critical for solving
// Ricatti equations via Hamiltonian/symplectic methods.

template<typename T, size_t N>
struct OrderedSchurResult {
    Matrix<N, N, T> U_stable{};   // Stable eigenvector subspace
    Matrix<N, N, T> U_unstable{}; // Unstable eigenvector subspace
    size_t          n_stable = 0; // Number of stable eigenvalues
    bool            converged = false;
};

// For continuous systems: stable if Re(λ) < 0
template<typename T, size_t N>
[[nodiscard]] constexpr OrderedSchurResult<T, N> ordered_schur_continuous(
    const Matrix<N, N, T>& A,
    size_t                 max_iter = 100,
    T                      tol = T{1e-8}
) {
    OrderedSchurResult<T, N> result;

    auto eigen = compute_eigenvalues_qr(A, max_iter, tol);
    result.converged = eigen.converged;

    // Sort eigenvalues/eigenvectors by stability
    size_t stable_count = 0;
    size_t unstable_count = 0;

    for (size_t i = 0; i < N; ++i) {
        T eigenval = eigen.eigenvalues_real(i, i);

        if (eigenval < T{0}) {
            // Stable: copy eigenvector to stable subspace
            for (size_t row = 0; row < N; ++row) {
                result.U_stable(row, stable_count) = eigen.eigenvectors(row, i);
            }
            stable_count++;
        } else {
            // Unstable: copy to unstable subspace
            for (size_t row = 0; row < N; ++row) {
                result.U_unstable(row, unstable_count) = eigen.eigenvectors(row, i);
            }
            unstable_count++;
        }
    }

    result.n_stable = stable_count;
    return result;
}

// For discrete systems: stable if |λ| < 1
template<typename T, size_t N>
[[nodiscard]] constexpr OrderedSchurResult<T, N> ordered_schur_discrete(
    const Matrix<N, N, T>& A,
    size_t                 max_iter = 100,
    T                      tol = T{1e-8}
) {
    OrderedSchurResult<T, N> result;

    auto eigen = compute_eigenvalues_qr(A, max_iter, tol);
    result.converged = eigen.converged;

    // Sort eigenvalues/eigenvectors by stability
    size_t stable_count = 0;
    size_t unstable_count = 0;

    for (size_t i = 0; i < N; ++i) {
        T eigenval = eigen.eigenvalues_real(i, i);
        T magnitude_sq = eigenval * eigenval;

        if (magnitude_sq < T{1}) {
            // Stable: |λ| < 1
            for (size_t row = 0; row < N; ++row) {
                result.U_stable(row, stable_count) = eigen.eigenvectors(row, i);
            }
            stable_count++;
        } else {
            // Unstable: |λ| >= 1
            for (size_t row = 0; row < N; ++row) {
                result.U_unstable(row, unstable_count) = eigen.eigenvectors(row, i);
            }
            unstable_count++;
        }
    }

    result.n_stable = stable_count;
    return result;
}
