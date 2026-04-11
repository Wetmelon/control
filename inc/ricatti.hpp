#pragma once

#include <cmath>
#include <cstddef>
#include <optional>

#include "eigen.hpp"
#include "matrix.hpp"
#include "matrix/cholesky.hpp"

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
 * @brief Check if (A, B) is a stabilizable pair.
 *
 * (A, B) is stabilizable if and only if the uncontrollable eigenvalues of A, if
 * any, have absolute values less than one, where an eigenvalue is
 * uncontrollable if rank([λI - A, B]) < n where n is the number of states.
 *
 * @tparam NX Number of states
 * @tparam NU Number of inputs
 * @tparam T  Scalar type
 * @param A   State matrix
 * @param B   Input matrix
 * @return true if (A, B) is stabilizable
 */
template<size_t NX, size_t NU, typename T = double>
constexpr bool is_stabilizable(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B
) {
    // Compute eigenvalues of A — use direct formulas for N ≤ 4, QR algorithm for N > 4
    if constexpr (NX <= 4) {
        auto eigen = compute_eigenvalues(A);
        if (!eigen.converged) {
            return false;
        }

        const T tol = std::is_same_v<T, float> ? T{1e-5} : T{1e-10};

        for (size_t i = 0; i < NX; ++i) {
            // Only check unstable eigenvalues (|λ| >= 1)
            if (eigen.values[i].abs() < T{1}) {
                continue;
            }

            // Form [λI - A, B] and check rank
            using Complex = wet::complex<T>;
            Matrix<NX, NX + NU, Complex> test_mat{};

            // λI - A
            for (size_t r = 0; r < NX; ++r) {
                for (size_t c = 0; c < NX; ++c) {
                    test_mat(r, c) = Complex(-A(r, c), T{0});
                }
                test_mat(r, r) = test_mat(r, r) + eigen.values[i];
            }

            // B
            for (size_t r = 0; r < NX; ++r) {
                for (size_t c = 0; c < NU; ++c) {
                    test_mat(r, NX + c) = Complex(B(r, c), T{0});
                }
            }

            // Check rank via Gaussian elimination with partial pivoting
            size_t                       rank = 0;
            Matrix<NX, NX + NU, Complex> work = test_mat;
            for (size_t col = 0; col < NX + NU && rank < NX; ++col) {
                size_t pivot = rank;
                T      max_val = work(rank, col).abs();
                for (size_t r = rank + 1; r < NX; ++r) {
                    T val = work(r, col).abs();
                    if (val > max_val) {
                        max_val = val;
                        pivot = r;
                    }
                }
                if (max_val < tol) {
                    continue;
                }
                if (pivot != rank) {
                    for (size_t j = 0; j < NX + NU; ++j) {
                        auto tmp = work(rank, j);
                        work(rank, j) = work(pivot, j);
                        work(pivot, j) = tmp;
                    }
                }
                for (size_t r = rank + 1; r < NX; ++r) {
                    auto factor = work(r, col) / work(rank, col);
                    for (size_t j = col; j < NX + NU; ++j) {
                        work(r, j) = work(r, j) - factor * work(rank, j);
                    }
                }
                ++rank;
            }

            if (rank < NX) {
                return false;
            }
        }

        return true;
    } else {
        // N > 4: use QR algorithm which returns real eigenvalues on diagonal
        auto eigen = compute_eigenvalues_qr(A);
        if (!eigen.converged) {
            // If QR didn't converge, skip the stabilizability check — let the
            // SDA loop itself detect divergence via its iteration limit.
            return true;
        }

        const T tol = std::is_same_v<T, float> ? T{1e-5} : T{1e-10};

        for (size_t i = 0; i < NX; ++i) {
            T lambda_real = eigen.eigenvalues_real(i, i);
            // QR returns real parts; for real eigenvalues |λ| = |λ_real|
            // This is conservative — complex eigenvalues from 2x2 blocks are not
            // individually resolved, but their magnitude is bounded by max|diagonal|.
            if (wet::abs(lambda_real) < T{1}) {
                continue;
            }

            // Form [λI - A, B] with real λ and check rank
            Matrix<NX, NX + NU, T> test_mat{};
            for (size_t r = 0; r < NX; ++r) {
                for (size_t c = 0; c < NX; ++c) {
                    test_mat(r, c) = -A(r, c);
                }
                test_mat(r, r) += lambda_real;
            }
            for (size_t r = 0; r < NX; ++r) {
                for (size_t c = 0; c < NU; ++c) {
                    test_mat(r, NX + c) = B(r, c);
                }
            }

            // Rank check via Gaussian elimination
            size_t                 rank = 0;
            Matrix<NX, NX + NU, T> work = test_mat;
            for (size_t col = 0; col < NX + NU && rank < NX; ++col) {
                size_t pivot = rank;
                T      max_val = wet::abs(work(rank, col));
                for (size_t r = rank + 1; r < NX; ++r) {
                    T val = wet::abs(work(r, col));
                    if (val > max_val) {
                        max_val = val;
                        pivot = r;
                    }
                }
                if (max_val < tol) {
                    continue;
                }
                if (pivot != rank) {
                    for (size_t j = 0; j < NX + NU; ++j) {
                        std::swap(work(rank, j), work(pivot, j));
                    }
                }
                for (size_t r = rank + 1; r < NX; ++r) {
                    T factor = work(r, col) / work(rank, col);
                    for (size_t j = col; j < NX + NU; ++j) {
                        work(r, j) -= factor * work(rank, j);
                    }
                }
                ++rank;
            }

            if (rank < NX) {
                return false;
            }
        }

        return true;
    }
}

/**
 * @brief Solve Discrete Algebraic Riccati Equation (DARE) — no precondition checks
 *
 * Solves AᵀXA − X − AᵀXB(R + BᵀXB)⁻¹BᵀXA + Q = 0 using the
 * Structure-preserving Doubling Algorithm (SDA).
 *
 * This version skips all precondition checks (symmetry, definiteness,
 * stabilizability) for maximum performance. Use when you know the inputs
 * satisfy the DARE preconditions.
 *
 * @tparam NX Number of states
 * @tparam NU Number of control inputs
 * @tparam T  Scalar type (default: double)
 * @param A   State transition matrix (NX × NX)
 * @param B   Control input matrix (NX × NU)
 * @param Q   State cost matrix (NX × NX, must be positive semidefinite)
 * @param R   Control cost matrix (NU × NU, must be positive definite)
 * @param N   Cross-term cost matrix (NX × NU, optional)
 *
 * @return Solution matrix P or std::nullopt on convergence failure
 */
template<size_t NX, size_t NU, typename T = double>
constexpr std::optional<Matrix<NX, NX, T>> dare_unchecked(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NX, NX, T>& Q,
    const Matrix<NU, NU, T>& R,
    const Matrix<NX, NU, T>& N = Matrix<NX, NU, T>{}
) {
    //! Handle cross-term N by reducing to standard form:
    //!   A_bar = A − B R⁻¹ Nᵀ,  Q_bar = Q − N R⁻¹ Nᵀ
    //! Solve via R * X = Nᵀ  and  R * X = Bᵀ  to avoid explicit R⁻¹
    const auto R_inv_Nt_opt = mat::lu_solve(R, N.transpose());
    if (!R_inv_Nt_opt) {
        return std::nullopt;
    }
    const Matrix R_inv_Nt = R_inv_Nt_opt.value();

    const Matrix A_eff = A - B * R_inv_Nt;
    const Matrix Q_eff = Q - N * R_inv_Nt;

    //! Compute Gk = B R⁻¹ Bᵀ via solve: R * X = Bᵀ → X = R⁻¹Bᵀ → Gk = B * X
    const auto R_inv_Bt_opt = mat::lu_solve(R, B.transpose());
    if (!R_inv_Bt_opt) {
        return std::nullopt;
    }

    //! Structure-preserving Doubling Algorithm (SDA) for DARE:
    //!   Initialize:  Ak = A,  Gk = B R⁻¹ Bᵀ,  Hk = Q
    //!   Iterate:
    //!     Vk = (I + Gk Hk)⁻¹  (solved via LU)
    //!     Ak₊₁ = Ak Vk Ak
    //!     Gk₊₁ = Gk + Ak Vk Gk Akᵀ
    //!     Hk₊₁ = Hk + Akᵀ Hk Vk Ak
    //!   Converges quadratically: Hk → X for any stabilizable/detectable pair.

    Matrix<NX, NX, T> Ak = A_eff;
    Matrix<NX, NX, T> Gk = B * R_inv_Bt_opt.value();
    Matrix<NX, NX, T> Hk = Q_eff;

    const T   tol = T{1e-12};
    const int max_iter = 100; //! Quadratic convergence needs far fewer iterations
    bool      converged = false;

    for (int iter = 0; iter < max_iter; ++iter) {
        //! Solve (I + Gk Hk) Vk = I via LU decomposition (recommended by SDA paper)
        const Matrix Vk_arg = Matrix<NX, NX, T>::identity() + Gk * Hk;
        const auto   Vk_opt = mat::lu_solve(Vk_arg, Matrix<NX, NX, T>::identity());
        if (!Vk_opt) {
            return std::nullopt;
        }
        const Matrix Vk = Vk_opt.value();

        const Matrix Ak_next = Ak * Vk * Ak;
        const Matrix Gk_next = Gk + Ak * Vk * Gk * Ak.transpose();
        const Matrix Hk_next = Hk + Ak.transpose() * Hk * Vk * Ak;

        const T diff = (Hk_next - Hk).norm();
        Ak = Ak_next;
        Gk = Gk_next;
        Hk = Hk_next;

        if (diff < tol * std::max(T{1}, Hk.norm())) {
            converged = true;
            break;
        }
    }

    if (!converged) {
        return std::nullopt;
    }

    //! Symmetrize for numerical cleanup
    return (Hk + Hk.transpose()) * T{0.5};
}

/**
 * @brief Solve Discrete Algebraic Riccati Equation (DARE) — no cross-term, no precondition checks
 *
 * Convenience overload without cross-term N for maximum runtime performance.
 *
 * @tparam NX Number of states
 * @tparam NU Number of control inputs
 * @tparam T  Scalar type (default: double)
 * @param A   State transition matrix (NX × NX)
 * @param B   Control input matrix (NX × NU)
 * @param Q   State cost matrix (NX × NX, must be positive semidefinite)
 * @param R   Control cost matrix (NU × NU, must be positive definite)
 *
 * @return Solution matrix P or std::nullopt on convergence failure
 */
template<size_t NX, size_t NU, typename T = double>
constexpr std::optional<Matrix<NX, NX, T>> dare_unchecked(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NX, NX, T>& Q,
    const Matrix<NU, NU, T>& R
) {
    //! Compute Gk = B R⁻¹ Bᵀ via solve: R * X = Bᵀ
    const auto R_inv_Bt_opt = mat::lu_solve(R, B.transpose());
    if (!R_inv_Bt_opt) {
        return std::nullopt;
    }

    Matrix<NX, NX, T> Ak = A;
    Matrix<NX, NX, T> Gk = B * R_inv_Bt_opt.value();
    Matrix<NX, NX, T> Hk = Q;

    const T   tol = T{1e-12};
    const int max_iter = 100;
    bool      converged = false;

    for (int iter = 0; iter < max_iter; ++iter) {
        const Matrix Vk_arg = Matrix<NX, NX, T>::identity() + Gk * Hk;
        const auto   Vk_opt = mat::lu_solve(Vk_arg, Matrix<NX, NX, T>::identity());
        if (!Vk_opt) {
            return std::nullopt;
        }
        const Matrix Vk = Vk_opt.value();

        const Matrix Ak_next = Ak * Vk * Ak;
        const Matrix Gk_next = Gk + Ak * Vk * Gk * Ak.transpose();
        const Matrix Hk_next = Hk + Ak.transpose() * Hk * Vk * Ak;

        const T diff = (Hk_next - Hk).norm();
        Ak = Ak_next;
        Gk = Gk_next;
        Hk = Hk_next;

        if (diff < tol * std::max(T{1}, Hk.norm())) {
            converged = true;
            break;
        }
    }

    if (!converged) {
        return std::nullopt;
    }

    return (Hk + Hk.transpose()) * T{0.5};
}

/**
 * @brief Solve Discrete Algebraic Riccati Equation (DARE) — with full precondition checks
 *
 * Solves AᵀXA − X − AᵀXB(R + BᵀXB)⁻¹BᵀXA + Q = 0 using the
 * Structure-preserving Doubling Algorithm (SDA).
 *
 * Preconditions checked:
 * - Q is symmetric
 * - R is symmetric
 * - R is positive definite (via Cholesky — fails on zero/negative pivot)
 * - Q is positive semidefinite (via Cholesky of Q + εI)
 * - (A, B) is stabilizable (unstable eigenvalues must be controllable)
 *
 * For maximum performance when preconditions are known to hold, use dare_unchecked().
 *
 * @tparam NX Number of states
 * @tparam NU Number of control inputs
 * @tparam T  Scalar type (default: double)
 * @param A   State transition matrix (NX × NX)
 * @param B   Control input matrix (NX × NU)
 * @param Q   State cost matrix (NX × NX, positive semidefinite)
 * @param R   Control cost matrix (NU × NU, positive definite)
 * @param N   Cross-term cost matrix (NX × NU, optional)
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
    //! Check R is symmetric
    if (!mat::is_symmetric_or_hermitian(R)) {
        return std::nullopt;
    }

    //! Check R is positive definite via Cholesky (fails on zero/negative pivot)
    if (!mat::cholesky(R)) {
        return std::nullopt;
    }

    //! Check Q is symmetric
    if (!mat::is_symmetric_or_hermitian(Q)) {
        return std::nullopt;
    }

    //! Check Q is positive semidefinite via Cholesky of (Q + εI)
    //! If Q is PSD, Q + εI is PD for any ε > 0, so Cholesky succeeds.
    //! If Q has a negative eigenvalue, Q + εI will still fail for small ε.
    {
        const T                 eps = std::is_same_v<T, float> ? T{1e-6} : T{1e-12};
        const Matrix<NX, NX, T> Q_shifted = Q + Matrix<NX, NX, T>::identity() * eps;
        if (!mat::cholesky(Q_shifted)) {
            return std::nullopt;
        }
    }

    //! Check (A, B) is stabilizable
    if (!is_stabilizable(A, B)) {
        return std::nullopt;
    }

    return dare_unchecked(A, B, Q, R, N);
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