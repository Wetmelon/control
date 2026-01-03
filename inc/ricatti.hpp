#pragma once

#include <cmath>
#include <cstddef>
#include <optional>

#include "eigen.hpp"
#include "matrix.hpp"

namespace wetmelon::control {

/**
 * @brief Solve 2x2 Continuous Algebraic Riccati Equation using Hamiltonian eigenvalue approach
 *
 * Solves A'P + PA - PBR⁻¹B'P + Q = 0 for 2x2 systems with scalar control.
 * Uses Hamiltonian matrix eigenvalue decomposition for numerical robustness.
 *
 * @tparam T Scalar type (default: double)
 * @param A 2x2 state matrix
 * @param B 2x1 control input matrix
 * @param Q 2x2 state cost matrix (positive semidefinite)
 * @param R 1x1 control cost matrix (positive definite)
 *
 * @return Solution matrix P (2x2, positive semidefinite)
 */
template<typename T = double>
constexpr Matrix<2, 2, T> care_2x2_hamiltonian(
    const Matrix<2, 2, T>& A,
    const Matrix<2, 1, T>& B,
    const Matrix<2, 2, T>& Q,
    const Matrix<1, 1, T>& R
) {
    T rinv = T{1} / R(0, 0);

    //! Build the 4x4 Hamiltonian matrix:
    //! H = [ A   -B*R⁻¹*B' ]
    //!     [ -Q      -A'  ]
    Matrix<4, 4, T> H = Matrix<4, 4, T>::zeros();

    //! Top-left: A
    H(0, 0) = A(0, 0);
    H(0, 1) = A(0, 1);
    H(1, 0) = A(1, 0);
    H(1, 1) = A(1, 1);

    //! Top-right: -B*R⁻¹*B'
    T b0 = B(0, 0), b1 = B(1, 0);
    H(0, 2) = -b0 * rinv * b0;
    H(0, 3) = -b0 * rinv * b1;
    H(1, 2) = -b1 * rinv * b0;
    H(1, 3) = -b1 * rinv * b1;

    //! Bottom-left: -Q
    H(2, 0) = -Q(0, 0);
    H(2, 1) = -Q(0, 1);
    H(3, 0) = -Q(1, 0);
    H(3, 1) = -Q(1, 1);

    //! Bottom-right: -A'
    H(2, 2) = -A(0, 0);
    H(2, 3) = -A(1, 0);
    H(3, 2) = -A(0, 1);
    H(3, 3) = -A(1, 1);

    //! Compute eigenvalues/eigenvectors of H
    auto eigen = compute_eigenvalues(H);

    if (!eigen.converged) {
        return Matrix<2, 2, T>::zeros(); //! Return zero on failure
    }

    //! Find the two eigenvectors corresponding to stable eigenvalues (Re < 0)
    //! and form the matrix [U1; U2] where each column is a stable eigenvector
    Matrix<2, 2, wet::complex<T>> U1 = Matrix<2, 2, wet::complex<T>>::zeros();
    Matrix<2, 2, wet::complex<T>> U2 = Matrix<2, 2, wet::complex<T>>::zeros();

    size_t stable_count = 0;
    for (size_t i = 0; i < 4 && stable_count < 2; ++i) {
        if (eigen.values[i].real() < T{0}) {
            // Extract this eigenvector
            for (size_t j = 0; j < 2; ++j) {
                U1(j, stable_count) = eigen.vectors(j, i);
                U2(j, stable_count) = eigen.vectors(j + 2, i);
            }
            stable_count++;
        }
    }

    if (stable_count < 2) {
        // Not enough stable eigenvalues - system may be unstabilizable
        return Matrix<2, 2, T>::zeros();
    }

    //! Solution: P = U2 * U1⁻¹
    //! Compute U1 inverse using 2x2 formula
    wet::complex<T> det_U1 = U1(0, 0) * U1(1, 1) - U1(0, 1) * U1(1, 0);
    T               det_mag = det_U1.norm();

    if (det_mag < T{1e-30}) {
        // U1 is singular
        return Matrix<2, 2, T>::zeros();
    }

    Matrix<2, 2, wet::complex<T>> U1_inv = Matrix<2, 2, wet::complex<T>>::zeros();
    U1_inv(0, 0) = U1(1, 1) / det_U1;
    U1_inv(0, 1) = -U1(0, 1) / det_U1;
    U1_inv(1, 0) = -U1(1, 0) / det_U1;
    U1_inv(1, 1) = U1(0, 0) / det_U1;

    //! P = U2 * U1_inv
    Matrix<2, 2, wet::complex<T>> P_complex = U2 * U1_inv;

    //! Extract real part (should be real for proper CARE)
    Matrix<2, 2, T> P = Matrix<2, 2, T>::zeros();
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            P(i, j) = P_complex(i, j).real();
        }
    }

    //! Make symmetric (average to reduce numerical errors)
    P(0, 1) = (P(0, 1) + P(1, 0)) / T{2};
    P(1, 0) = P(0, 1);

    return P;
}

/**
 * @brief Solve Discrete Algebraic Riccati Equation (DARE)
 *
 * Solves P = A'PA - A'PB(R + B'PB)⁻¹B'PA + Q for optimal discrete LQR.
 * Uses fixed-point iteration for numerical solution.
 *
 * @tparam NX Number of states
 * @tparam NU Number of control inputs
 * @tparam T  Scalar type (default: double)
 * @param A   State transition matrix (NX × NX)
 * @param B   Control input matrix (NX × NU)
 * @param Q   State cost matrix (NX × NX, positive semidefinite)
 * @param R   Control cost matrix (NU × NU, positive definite)
 *
 * @return Solution matrix P (NX × NX, positive semidefinite) or std::nullopt on failure
 */
template<size_t NX, size_t NU, typename T = double>
constexpr std::optional<Matrix<NX, NX, T>> dare(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NX, NX, T>& Q,
    const Matrix<NU, NU, T>& R,
    const Matrix<NX, NU, T>& N = Matrix<NX, NU, T>{}
) {
    //! Check positive definiteness of R via eigenvalues
    auto R_eigen = compute_eigenvalues_qr(R);
    if (R_eigen.converged) {
        for (size_t i = 0; i < NU; ++i) {
            if (R_eigen.eigenvalues_real(i, i) <= T{0}) {
                return std::nullopt; //! R not positive definite
            }
        }
    }

    //! Check positive semidefiniteness of Q via eigenvalues
    auto Q_eigen = compute_eigenvalues_qr(Q);
    if (Q_eigen.converged) {
        for (size_t i = 0; i < NX; ++i) {
            if (Q_eigen.eigenvalues_real(i, i) < T{-1e-12}) {
                return std::nullopt; //! Q not positive semidefinite
            }
        }
    }

    // Use improved Kleinman iteration for DARE with optional cross-term N:
    // Cost: x'Qx + u'Ru + 2 x' N u
    // Riccati update:
    //   K = S^{-1}(B^T X A + N^T),  S = R + B^T X B
    //   X_{k+1} = Q + A^T X A - (A^T X B + N) K
    // This Joseph-form update reduces matrix operations and improves numerical stability.

    //! Initial guess from discrete Lyapunov: solve X = A^T X A + Q
    //! This gives a much better starting point than just Q, especially for stable systems
    Matrix<NX, NX, T> X = dlyap(A, Q).value_or(Q);

    const T tol = T{1e-9};
    //! Use dimension-adaptive iteration limit
    //! Larger systems and/or systems with eigenvalues near the unit circle need more iterations
    //! Scale roughly with N^2 to account for slower convergence in larger state spaces
    constexpr int base_iter = 200;
    constexpr int max_iter = base_iter + static_cast<int>(NX * NX * 50);
    bool          converged = false;

    for (int iter = 0; iter < max_iter; ++iter) {
        Matrix X_prev = X;

        const Matrix At = A.transpose();
        const Matrix Bt = B.transpose();

        //! Compute S = R + B^T X B
        const Matrix S = R + Bt * X * B;
        const auto   S_inv_opt = S.inverse();

        if (!S_inv_opt)
            return std::nullopt; //! Singular S matrix
        const Matrix S_inv = S_inv_opt.value();

        //! Compute gain K = S^{-1}(B^T X A + N^T)
        const Matrix K = S_inv * (Bt * X * A + N.transpose());

        //! Joseph-like update: X_new = Q + A^T X A - (A^T X B + N)K
        Matrix X_new = (At * X * A) - (At * X * B + N) * K + Q;

        Matrix diff = X_new - X;
        X = X_new;

        if (diff.norm() < tol * std::max(1.0, X.norm())) {
            converged = true;
            break;
        };
    }

    if (!converged)
        return std::nullopt;

    return X;
}

/**
 * @brief Direct solve 2x2 discrete Lyapunov equation: A'XA - X + Q = 0
 *
 * Uses vectorization: (A^T ⊗ A^T - I)vec(X) = -vec(Q)
 * For 2x2, this is a 4x4 linear system solved directly.
 *
 * @tparam T Scalar type (default: double)
 * @param A 2x2 system matrix (must be stable: all eigenvalues inside unit circle)
 * @param Q 2x2 right-hand side matrix (typically positive semidefinite)
 *
 * @return Solution X, or nullopt if singular
 */
template<typename T = double>
constexpr auto dlyap_2x2(const Matrix<2, 2, T>& A, const Matrix<2, 2, T>& Q) -> std::optional<Matrix<2, 2, T>> {
    // Build the 4x4 coefficient matrix M = A^T ⊗ A^T - I
    // Using vec(A^T X A) = (A^T ⊗ A^T) vec(X)
    // Equation: (A^T ⊗ A^T - I) vec(X) = -vec(Q)

    // A^T elements
    T at00 = A(0, 0), at01 = A(1, 0), at10 = A(0, 1), at11 = A(1, 1);

    // Kronecker product A^T ⊗ A^T for 2x2:
    // [at00*A^T  at01*A^T]
    // [at10*A^T  at11*A^T]
    Matrix<4, 4, T> M;
    M(0, 0) = at00 * at00 - T{1};
    M(0, 1) = at00 * at01;
    M(0, 2) = at01 * at00;
    M(0, 3) = at01 * at01;
    M(1, 0) = at00 * at10;
    M(1, 1) = at00 * at11 - T{1};
    M(1, 2) = at01 * at10;
    M(1, 3) = at01 * at11;
    M(2, 0) = at10 * at00;
    M(2, 1) = at10 * at01;
    M(2, 2) = at11 * at00 - T{1};
    M(2, 3) = at11 * at01;
    M(3, 0) = at10 * at10;
    M(3, 1) = at10 * at11;
    M(3, 2) = at11 * at10;
    M(3, 3) = at11 * at11 - T{1};

    // RHS: -vec(Q) where vec stacks columns
    Matrix<4, 1, T> rhs;
    rhs(0, 0) = -Q(0, 0);
    rhs(1, 0) = -Q(1, 0);
    rhs(2, 0) = -Q(0, 1);
    rhs(3, 0) = -Q(1, 1);

    // Solve M * vec(X) = rhs using 4x4 inverse
    auto M_inv_opt = M.inverse();
    if (!M_inv_opt) {
        return std::nullopt;
    }

    auto vec_X = M_inv_opt.value() * rhs;

    // Unpack solution
    Matrix<2, 2, T> X;
    X(0, 0) = vec_X(0, 0);
    X(1, 0) = vec_X(1, 0);
    X(0, 1) = vec_X(2, 0);
    X(1, 1) = vec_X(3, 0);

    // Symmetrize (numerical cleanup)
    X = (X + X.transpose()) * T{0.5};

    return X;
}

/**
 * @brief Solve discrete Lyapunov equation
 *
 * Solves X = AXA' + Q by vectorizing and solving linear system.
 * Used in discrete Riccati equation solvers.
 *
 * @tparam Rows Matrix dimension (must equal Cols)
 * @tparam Cols Matrix dimension (must equal Rows)
 * @tparam T    Scalar type (default: double)
 * @param A     System matrix
 * @param Q     Right-hand side matrix
 *
 * @return Solution X, or zero matrix if singular
 */
template<size_t Rows, size_t Cols, typename T = double>
constexpr auto dlyap(const Matrix<Rows, Cols, T>& A, const Matrix<Rows, Cols, T>& Q) -> std::optional<Matrix<Rows, Cols, T>> {
    static_assert(Rows == Cols, "Lyapunov equation requires square matrices");

    // Use direct solver for 2x2 (O(1) operations vs O(n³) iterations)
    if constexpr (Rows == 2) {
        return dlyap_2x2(A, Q);
    }

    // Fixed-point iteration: X_{k+1} = A^T X_k A + Q
    Matrix<Rows, Cols, T> X = Q;
    const T               tol = T{1e-9};
    const int             max_iter = 500;
    const T               guard = T{1e150}; // prevents overflow in constexpr evaluation

    bool converged = false;

    for (int iter = 0; iter < max_iter; ++iter) {
        Matrix<Rows, Cols, T> X_prev = X;
        Matrix<Rows, Cols, T> X_next = A.transpose() * X * A + Q;

        bool diverged = false;
        for (size_t i = 0; i < Rows && !diverged; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                if (!std::isfinite(X_next(i, j)) || wet::abs(X_next(i, j)) > guard) {
                    diverged = true;
                    break;
                }
            }
        }
        if (diverged) {
            return std::nullopt;
        }

        X = X_next;

        // Frobenius norm convergence check with clamping
        T diff_norm_sq = T{0};
        T x_norm_sq = T{0};
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                T diff = X(i, j) - X_prev(i, j);
                if (diff > guard)
                    diff = guard;
                else if (diff < -guard)
                    diff = -guard;
                diff_norm_sq += diff * diff;

                T xval = X(i, j);
                if (xval > guard)
                    xval = guard;
                else if (xval < -guard)
                    xval = -guard;
                x_norm_sq += xval * xval;
            }
        }

        if (wet::sqrt(diff_norm_sq) < tol * std::max(T{1}, wet::sqrt(x_norm_sq))) {
            converged = true;
            break;
        }
    }

    if (!converged)
        return std::nullopt;

    return (X + X.transpose()) * T{0.5};
}
} // namespace wetmelon::control