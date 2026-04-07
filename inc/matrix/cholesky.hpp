#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <type_traits>

#include "constexpr_complex.hpp"
#include "matrix.hpp"

namespace wetmelon::control {

namespace mat {

template<size_t N, typename T>
constexpr bool is_symmetric_or_hermitian(const Matrix<N, N, T>& A) {
    if constexpr (std::is_floating_point_v<T>) {
        T tol = std::is_same_v<T, float> ? T{1e-6} : T{1e-12};
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = i + 1; j < N; ++j) {
                if (wet::abs(A(i, j) - A(j, i)) > tol)
                    return false;
            }
        }
        return true;
    } else {
        // Complex: check A(i,j) == conj(A(j,i)) within tolerance using squared magnitude
        using real_t = typename T::value_type;
        real_t     tol = std::is_same_v<real_t, float> ? real_t{1e-6} : real_t{1e-12};
        const auto tol_sq = tol * tol;

        for (size_t i = 0; i < N; ++i) {
            for (size_t j = i + 1; j < N; ++j) {
                auto diff = A(i, j) - wet::conj(A(j, i));
                // diff * conj(diff) is real and equals |diff|^2
                auto mag_sq = wet::real(diff * wet::conj(diff));
                if (mag_sq > tol_sq)
                    return false;
            }
        }
        return true;
    }
}

/**
 * @brief Cholesky decomposition for symmetric positive-definite matrices
 *
 * @param A Input SPD matrix
 *
 * @return Lower-triangular matrix L such that A = L * Lᵀ, or std::nullopt if A is not SPD
 */
template<size_t N, typename T>
constexpr std::optional<Matrix<N, N, T>> cholesky(const Matrix<N, N, T>& A) {
    // Cholesky only works for real matrices
    if constexpr (!std::is_floating_point_v<T>)
        return std::nullopt;

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
 * @brief Solve linear system A * X = B using Cholesky decomposition
 */
template<size_t N, size_t M, typename T>
constexpr std::optional<Matrix<N, M, T>> cholesky_solve(const Matrix<N, N, T>& A, const Matrix<N, M, T>& B) {
    auto Lopt = cholesky(A);
    if (!Lopt)
        return std::nullopt;

    const auto&     L = Lopt.value();
    Matrix<N, M, T> X;

    for (size_t col = 0; col < M; ++col) {
        ColVec<N, T> b;
        for (size_t i = 0; i < N; ++i)
            b(i) = B(i, col);

        auto y = forward_substitute(L, b);
        auto x = backward_substitute_transpose(L, y);

        for (size_t i = 0; i < N; ++i)
            X(i, col) = x(i);
    }

    return X;
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
    Matrix<N, M, T> X;

    for (size_t col = 0; col < M; ++col) {
        ColVec<N, T> b_permuted;
        for (size_t r = 0; r < N; ++r)
            b_permuted(r) = B(piv[r], col);

        auto y = forward_substitute(L, b_permuted);
        auto x = backward_substitute_upper(U, y);

        for (size_t r = 0; r < N; ++r)
            X(r, col) = x(r);
    }

    return X;
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

} // namespace wetmelon::control
