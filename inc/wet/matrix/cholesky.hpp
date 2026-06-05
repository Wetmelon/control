#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <type_traits>

#include "matrix.hpp"
#include "matrix_traits.hpp"
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
 * @return Lower-triangular L, or std::nullopt if A is not PD or not symmetric/Hermitian
 */
template<size_t N, typename T>
constexpr std::optional<Matrix<N, N, T>> cholesky(const Matrix<N, N, T>& A) {
    if (!is_symmetric_or_hermitian(A)) {
        return std::nullopt;
    }

    Matrix<N, N, T> L = Matrix<N, N, T>::zeros();

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j <= i; ++j) {

            T sum = A(i, j);
            for (size_t k = 0; k < j; ++k)
                sum -= L(i, k) * wet::conj(L(j, k));

            if (i == j) {
                if constexpr (std::is_floating_point_v<T>) {
                    auto diag_val = sum;
                    if (diag_val <= 0)
                        return std::nullopt;
                    L(i, j) = wet::sqrt(diag_val);
                } else {
                    auto diag_val = wet::real(sum);
                    if (diag_val <= 0)
                        return std::nullopt;
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
 * @brief Forward substitution to solve L * x = b
 *
 * Works for both unit-diagonal LU and Cholesky L.
 */
template<size_t N, typename T>
constexpr ColVec<N, T> forward_substitute(const Matrix<N, N, T>& L, const ColVec<N, T>& b) {
    ColVec<N, T> x;

    for (size_t i = 0; i < N; ++i) {
        T sum = b(i);
        for (size_t j = 0; j < i; ++j)
            sum -= L(i, j) * x(j);

        x(i) = sum / L(i, i); // safe for both unit-diagonal LU (1/1) and Cholesky
    }

    return x;
}

/**
 * @brief Backward substitution to solve Lᵀ * x = b
 */
template<size_t N, typename T>
constexpr ColVec<N, T> backward_substitute_transpose(const Matrix<N, N, T>& L, const ColVec<N, T>& b) {
    ColVec<N, T> x;

    for (int i = int(N) - 1; i >= 0; --i) {
        T sum = b(i);
        for (size_t j = i + 1; j < N; ++j)
            sum -= L(j, i) * x(j);

        x(i) = sum / L(i, i);
    }

    return x;
}

/**
 * @brief Backward substitution to solve U * x = b where U is upper-triangular
 */
template<size_t N, typename T>
constexpr ColVec<N, T> backward_substitute_upper(const Matrix<N, N, T>& U, const ColVec<N, T>& b) {
    ColVec<N, T> x;

    for (int i = int(N) - 1; i >= 0; --i) {
        T sum = b(i);
        for (size_t j = i + 1; j < N; ++j)
            sum -= U(i, j) * x(j);
        x(i) = sum / U(i, i);
    }

    return x;
}

/**
 * @brief Solve lower-triangular system L * X = B via forward substitution
 *
 * Requires that L is a non-singular lower-triangular matrix (all diagonal
 * elements non-zero). Returns std::nullopt if any diagonal element is zero.
 */
template<size_t N, size_t M, typename T>
constexpr std::optional<Matrix<N, M, std::remove_const_t<T>>>
solve(const LowerTriangle<N, T>& L, const Matrix<N, M, std::remove_const_t<T>>& B) {
    using VT = std::remove_const_t<T>;
    constexpr auto tol = default_tol<VT>();

    Matrix<N, M, VT> X;
    for (size_t col = 0; col < M; ++col) {
        ColVec<N, VT> b;
        for (size_t i = 0; i < N; ++i) {
            b(i) = B(i, col);
        }

        ColVec<N, VT> x;
        for (size_t i = 0; i < N; ++i) {
            if (wet::abs(L(i, i)) < tol) {
                return std::nullopt;
            }
            VT sum = b(i);
            for (size_t j = 0; j < i; ++j) {
                sum -= L(i, j) * x(j);
            }
            x(i) = sum / L(i, i);
        }

        for (size_t i = 0; i < N; ++i) {
            X(i, col) = x(i);
        }
    }
    return X;
}

/**
 * @brief Solve upper-triangular system U * X = B via backward substitution
 *
 * Requires that U is a non-singular upper-triangular matrix (all diagonal
 * elements non-zero). Returns std::nullopt if any diagonal element is zero.
 */
template<size_t N, size_t M, typename T>
constexpr std::optional<Matrix<N, M, std::remove_const_t<T>>>
solve(const UpperTriangle<N, T>& U, const Matrix<N, M, std::remove_const_t<T>>& B) {
    using VT = std::remove_const_t<T>;
    constexpr auto tol = default_tol<VT>();

    Matrix<N, M, VT> X;
    for (size_t col = 0; col < M; ++col) {
        ColVec<N, VT> b;
        for (size_t i = 0; i < N; ++i) {
            b(i) = B(i, col);
        }

        ColVec<N, VT> x;
        for (int i = int(N) - 1; i >= 0; --i) {
            if (wet::abs(U(size_t(i), size_t(i))) < tol) {
                return std::nullopt;
            }
            VT sum = b(i);
            for (size_t j = size_t(i) + 1; j < N; ++j) {
                sum -= U(size_t(i), j) * x(j);
            }
            x(i) = sum / U(size_t(i), size_t(i));
        }

        for (size_t i = 0; i < N; ++i) {
            X(i, col) = x(i);
        }
    }
    return X;
}

/**
 * @brief Solve linear system A * X = B using Cholesky decomposition
 */
template<size_t N, size_t M, typename T>
constexpr std::optional<Matrix<N, M, T>> cholesky_solve(const Matrix<N, N, T>& A, const Matrix<N, M, T>& B) {
    auto Lopt = cholesky(A);
    if (!Lopt)
        return std::nullopt;

    const auto& L = Lopt.value();

    // Forward substitution: solve L * Y = B
    auto Y_opt = solve(L.lower_triangle(), B);
    if (!Y_opt)
        return std::nullopt;

    // Backward substitution: solve Lᴴ * X = Y
    const auto Lh = L.conjugate_transpose();
    return solve(Lh.upper_triangle(), Y_opt.value());
}

/**
 * @brief LU decomposition with partial pivoting
 *
 * Returns pair<L,U> and a pivot vector piv such that P*A = L*U
 * L: lower triangular with unit diagonal
 * U: upper triangular
 */
template<size_t N, typename T>
constexpr std::optional<std::tuple<Matrix<N, N, T>, Matrix<N, N, T>, std::array<size_t, N>>>
lu_decomposition(const Matrix<N, N, T>& A) {
    Matrix<N, N, T>       L = Matrix<N, N, T>::identity();
    Matrix<N, N, T>       U = A;
    std::array<size_t, N> piv;
    for (size_t i = 0; i < N; ++i)
        piv[i] = i;

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
            return std::nullopt;
        }

        if (max_row != i) {
            // Swap rows in U
            for (size_t col = 0; col < N; ++col)
                std::swap(U(i, col), U(max_row, col));
            // Swap previous columns in L
            for (size_t col = 0; col < i; ++col)
                std::swap(L(i, col), L(max_row, col));
            // Track pivot
            std::swap(piv[i], piv[max_row]);
        }

        // Elimination
        for (size_t j = i + 1; j < N; ++j) {
            auto factor = U(j, i) / U(i, i);
            L(j, i) = factor;
            for (size_t k = i; k < N; ++k)
                U(j, k) -= factor * U(i, k);
        }
    }

    return std::make_tuple(L, U, piv);
}

/**
 * @brief Solve linear system using LU decomposition (with pivot vector)
 */
template<size_t N, size_t M, typename T>
constexpr std::optional<Matrix<N, M, T>> lu_solve(const Matrix<N, N, T>& A, const Matrix<N, M, T>& B) {
    auto lu_opt = lu_decomposition(A);
    if (!lu_opt)
        return std::nullopt;

    const auto& [L, U, piv] = lu_opt.value();

    // Apply row permutation to B
    Matrix<N, M, T> B_perm;
    for (size_t r = 0; r < N; ++r)
        for (size_t c = 0; c < M; ++c)
            B_perm(r, c) = B(piv[r], c);

    // Forward substitution: solve L * Y = P*B
    auto Y_opt = solve(L.lower_triangle(), B_perm);
    if (!Y_opt)
        return std::nullopt;

    // Backward substitution: solve U * X = Y
    return solve(U.upper_triangle(), Y_opt.value());
}

/**
 * @brief Solve linear system using Cholesky if SPD, else LU
 */
template<size_t N, size_t M, typename T>
constexpr std::optional<Matrix<N, M, T>> solve(const Matrix<N, N, T>& A, const Matrix<N, M, T>& B) {
    // Try Cholesky for symmetric (or Hermitian) positive-definite matrices, fall back to LU
    if (is_symmetric_or_hermitian<N, T>(A)) {
        auto Xopt = cholesky_solve(A, B);
        if (Xopt) {
            return Xopt;
        }
    }

    return lu_solve(A, B);
}

} // namespace mat

/**
 * @brief Matrix inverse using mat::solve()
 */
template<size_t Rows, size_t Cols, typename T>
[[nodiscard]] constexpr std::optional<Matrix<Rows, Cols, T>> Matrix<Rows, Cols, T>::inverse() const
    requires(Rows == Cols)
{
    return mat::solve(*this, Matrix<Rows, Cols, T>::identity());
}

} // namespace wet
