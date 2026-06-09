#pragma once

/**
 * @file solve.hpp
 * @brief Linear-system solvers built on the factorizations in decomposition.hpp:
 *        triangular substitution, Cholesky/LU solves, the SPD-aware generic
 *        solve(), and the Matrix::inverse() definition.
 */

#include <cstddef>
#include <type_traits>

#include "decomposition.hpp"
#include "matrix.hpp"
#include "matrix_traits.hpp"
#include "wet/backend.hpp"
#include "wet/math/complex.hpp"

namespace wet {

namespace mat {

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
        for (size_t j = 0; j < i; ++j) {
            sum -= L(i, j) * x(j);
        }

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
        for (size_t j = i + 1; j < N; ++j) {
            sum -= L(j, i) * x(j);
        }

        x(i) = sum / L(i, i);
    }

    return x;
}

/**
 * @brief Solve lower-triangular system L * X = B via forward substitution
 *
 * Requires that L is a non-singular lower-triangular matrix (all diagonal
 * elements non-zero). Returns wet::nullopt if any diagonal element is zero.
 */
template<size_t N, size_t M, typename T>
constexpr wet::optional<Matrix<N, M, std::remove_const_t<T>>>
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
                return wet::nullopt;
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
 * elements non-zero). Returns wet::nullopt if any diagonal element is zero.
 */
template<size_t N, size_t M, typename T>
constexpr wet::optional<Matrix<N, M, std::remove_const_t<T>>>
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
                return wet::nullopt;
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
constexpr wet::optional<Matrix<N, M, T>> cholesky_solve(const Matrix<N, N, T>& A, const Matrix<N, M, T>& B) {
    auto Lopt = cholesky(A);
    if (!Lopt) {
        return wet::nullopt;
    }

    const auto& L = Lopt.value();

    // Forward substitution: solve L * Y = B
    auto Y_opt = solve(L.lower_triangle(), B);
    if (!Y_opt) {
        return wet::nullopt;
    }

    // Backward substitution: solve Lᴴ * X = Y
    const auto Lh = L.conjugate_transpose();
    return solve(Lh.upper_triangle(), Y_opt.value());
}

/**
 * @brief Solve linear system using LU decomposition (with pivot vector)
 */
template<size_t N, size_t M, typename T>
constexpr wet::optional<Matrix<N, M, T>> lu_solve(const Matrix<N, N, T>& A, const Matrix<N, M, T>& B) {
    auto lu_opt = lu_decomposition(A);
    if (!lu_opt) {
        return wet::nullopt;
    }

    const auto& [L, U, piv] = lu_opt.value();

    // Apply row permutation to B
    Matrix<N, M, T> B_perm;
    for (size_t r = 0; r < N; ++r) {
        for (size_t c = 0; c < M; ++c) {
            B_perm(r, c) = B(piv[r], c);
        }
    }

    // Forward substitution: solve L * Y = P*B
    auto Y_opt = solve(L.lower_triangle(), B_perm);
    if (!Y_opt) {
        return wet::nullopt;
    }

    // Backward substitution: solve U * X = Y
    return solve(U.upper_triangle(), Y_opt.value());
}

/**
 * @brief Solve linear system using Cholesky if SPD, else LU
 */
template<size_t N, size_t M, typename T>
constexpr wet::optional<Matrix<N, M, T>> solve(const Matrix<N, N, T>& A, const Matrix<N, M, T>& B) {
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
[[nodiscard]] constexpr wet::optional<Matrix<Rows, Cols, T>> Matrix<Rows, Cols, T>::inverse() const
    requires(Rows == Cols)
{
    return mat::solve(*this, Matrix<Rows, Cols, T>::identity());
}

} // namespace wet
