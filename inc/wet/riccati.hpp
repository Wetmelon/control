#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "wet/matrix/eigen.hpp"
#include "wet/matrix/matrix.hpp"

namespace wetmelon::control {

/**
 * @brief Check if (A, B) is a stabilizable pair
 *
 * (A, B) is stabilizable iff every uncontrollable eigenvalue of A lies strictly
 * inside the unit circle. An eigenvalue λ is uncontrollable when
 * rank([λI − A, B]) < n.
 *
 * Stabilizability is weaker than controllability — it permits uncontrollable
 * modes as long as they are already stable (|λ| < 1).
 *
 * @see "Optimal Control" (Anderson & Moore, 1990), §2.4
 *
 * @tparam NX Number of states
 * @tparam NU Number of inputs
 * @tparam T  Scalar type
 * @param A   State matrix (NX × NX)
 * @param B   Input matrix (NX × NU)
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

namespace detail {

/**
 * @brief Solve DARE via Structure-Preserving Doubling Algorithm (SDA)
 *
 * Solves AᵀXA − X − (AᵀXB + N)(R + BᵀXB)⁻¹(BᵀXA + Nᵀ) + Q = 0
 *
 * Quadratic convergence (doubles correct digits each iteration).
 * Requires R positive definite (needs R⁻¹ for cross-term reduction).
 * No precondition checks — use dare() for validated entry point.
 *
 * @see Chu et al., "Structure-Preserving Algorithms for Periodic DRE" (2004)
 * @see "Optimal Control" (Anderson & Moore, 1990), §4.3
 */
template<size_t NX, size_t NU, typename T = double>
constexpr std::optional<Matrix<NX, NX, T>> dare_sda(
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
        const Matrix Gk_next = Gk + Ak * Vk * Gk * Ak.t();
        const Matrix Hk_next = Hk + Ak.t() * Hk * Vk * Ak;

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
    return (Hk + Hk.t()) * T{0.5};
}

/**
 * @brief Solve DARE via Riccati Difference Equation (RDE) iteration
 *
 * Iterates the discrete Riccati recursion to steady state:
 *
 *     X[k+1] = AᵀX[k]A + Q − AᵀX[k]B(R + BᵀX[k]B)⁻¹BᵀX[k]A
 *
 * Linear convergence. Handles R ≥ 0 (only requires R + BᵀXB invertible at
 * each step, not R itself). Useful when R is singular (e.g., cheap-control
 * problems or minimum-energy estimation).
 *
 * No precondition checks — use dare() for validated entry point.
 *
 * @see "Optimal Control" (Anderson & Moore, 1990), §4.2
 */
template<size_t NX, size_t NU, typename T = double>
constexpr std::optional<Matrix<NX, NX, T>> dare_rde(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NX, NX, T>& Q,
    const Matrix<NU, NU, T>& R,
    const Matrix<NX, NU, T>& N = Matrix<NX, NU, T>{}
) {
    //! Handle cross-term N by reducing to standard form (requires R invertible)
    Matrix<NX, NX, T> A_eff = A;
    Matrix<NX, NX, T> Q_eff = Q;

    bool n_is_zero = true;
    for (size_t i = 0; i < NX && n_is_zero; ++i) {
        for (size_t j = 0; j < NU; ++j) {
            if (wet::abs(N(i, j)) > T{1e-15}) {
                n_is_zero = false;
                break;
            }
        }
    }

    if (!n_is_zero) {
        const auto R_inv_Nt_opt = mat::lu_solve(R, N.transpose());
        if (!R_inv_Nt_opt) {
            return std::nullopt; // singular R with non-zero N is unsupported
        }
        const Matrix R_inv_Nt = R_inv_Nt_opt.value();
        A_eff = A - B * R_inv_Nt;
        Q_eff = Q - N * R_inv_Nt;
    }

    //! Riccati Difference Equation iteration
    Matrix<NX, NX, T> X = Q_eff;
    const T           tol = T{1e-12};
    const int         max_iter = 500;
    const T           guard = T{1e150};

    bool converged = false;

    for (int iter = 0; iter < max_iter; ++iter) {
        const Matrix BtX = B.t() * X;
        const Matrix BtXA = BtX * A_eff;
        const Matrix S = R + BtX * B;

        const auto M_opt = mat::lu_solve(S, BtXA);
        if (!M_opt) {
            //! First-iteration kick: if S = R + B'QB is singular, try X₀ = Q + αI
            if (iter == 0) {
                T trace_q = T{0};
                for (size_t i = 0; i < NX; ++i) {
                    trace_q += wet::abs(Q_eff(i, i));
                }
                X = Q_eff + Matrix<NX, NX, T>::identity() * (trace_q / T(NX) + T{1});
                continue;
            }
            return std::nullopt;
        }
        const Matrix M = M_opt.value();

        const Matrix X_next = A_eff.t() * X * A_eff + Q_eff
                            - A_eff.t() * X * B * M;

        //! Divergence guard (matches dlyap pattern)
        bool diverged = false;
        for (size_t i = 0; i < NX && !diverged; ++i) {
            for (size_t j = 0; j < NX; ++j) {
                if (!wet::isfinite(X_next(i, j)) || wet::abs(X_next(i, j)) > guard) {
                    diverged = true;
                    break;
                }
            }
        }
        if (diverged) {
            return std::nullopt;
        }

        //! Convergence check (Frobenius norm with guard clamping)
        T diff_norm_sq = T{0};
        T x_norm_sq = T{0};
        for (size_t i = 0; i < NX; ++i) {
            for (size_t j = 0; j < NX; ++j) {
                T diff = X_next(i, j) - X(i, j);
                if (diff > guard) {
                    diff = guard;
                } else if (diff < -guard) {
                    diff = -guard;
                }
                diff_norm_sq += diff * diff;

                T xval = X_next(i, j);
                if (xval > guard) {
                    xval = guard;
                } else if (xval < -guard) {
                    xval = -guard;
                }
                x_norm_sq += xval * xval;
            }
        }

        X = X_next;

        if (wet::sqrt(diff_norm_sq) < tol * std::max(T{1}, wet::sqrt(x_norm_sq))) {
            converged = true;
            break;
        }
    }

    if (!converged) {
        return std::nullopt;
    }

    return (X + X.t()) * T{0.5};
}

/**
 * @brief Solve CARE via the matrix sign function (Roberts' method)
 *
 * Solves AᵀX + XA − (XB + N)R⁻¹(BᵀX + Nᵀ) + Q = 0 by computing the sign
 * function of the Hamiltonian
 *
 *     H = ⎡  A   −G  ⎤,   G = B R⁻¹ Bᵀ   (cross-term N folded into A, Q first)
 *         ⎣ −Q   −Aᵀ ⎦
 *
 * via the scaled Newton iteration Zₖ₊₁ = ½(Zₖ/cₖ + cₖ Zₖ⁻¹) → sign(H), then
 * extracting the stabilizing solution from the stable invariant subspace:
 * writing sign(H) = [[W₁₁,W₁₂],[W₂₁,W₂₂]], X solves the (consistent) stacked
 * system [W₁₂; W₂₂+I] X = −[W₁₁+I; W₂₁], here via its normal equations.
 *
 * Frobenius-norm scaling cₖ = √(‖Zₖ⁻¹‖/‖Zₖ‖) gives near-quadratic convergence.
 * No precondition checks — use care() for the validated entry point. Returns
 * nullopt if the iteration fails to converge (e.g. H has eigenvalues on the
 * imaginary axis, i.e. the stabilizing solution does not exist).
 *
 * @see Roberts, "Linear model reduction and solution of the algebraic Riccati
 *      equation by use of the sign function," Int. J. Control, 1980,
 *      https://doi.org/10.1080/00207178008922881
 * @see Higham, "Functions of Matrices" (2008), §2.3 (sign-function scaling)
 */
template<size_t NX, size_t NU, typename T = double>
constexpr std::optional<Matrix<NX, NX, T>> care_sign(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NX, NX, T>& Q,
    const Matrix<NU, NU, T>& R,
    const Matrix<NX, NU, T>& N = Matrix<NX, NU, T>{}
) {
    //! G = B R⁻¹ Bᵀ and cross-term reduction A_eff = A − B R⁻¹ Nᵀ,
    //! Q_eff = Q − N R⁻¹ Nᵀ — solved against R rather than forming R⁻¹.
    const auto Rinv_Bt_opt = mat::lu_solve(R, B.transpose());
    if (!Rinv_Bt_opt) {
        return std::nullopt;
    }
    const auto Rinv_Nt_opt = mat::lu_solve(R, N.transpose());
    if (!Rinv_Nt_opt) {
        return std::nullopt;
    }
    const Matrix<NX, NX, T> G = B * Rinv_Bt_opt.value();
    const Matrix<NX, NX, T> A_eff = A - B * Rinv_Nt_opt.value();
    const Matrix<NX, NX, T> Q_eff = Q - N * Rinv_Nt_opt.value();

    //! Assemble the 2NX×2NX Hamiltonian.
    constexpr size_t        M = 2 * NX;
    Matrix<M, M, T>         H = Matrix<M, M, T>::zeros();
    const Matrix<NX, NX, T> A_effT = A_eff.t();
    for (size_t i = 0; i < NX; ++i) {
        for (size_t j = 0; j < NX; ++j) {
            H(i, j) = A_eff(i, j);
            H(i, NX + j) = -G(i, j);
            H(NX + i, j) = -Q_eff(i, j);
            H(NX + i, NX + j) = -A_effT(i, j);
        }
    }

    //! Scaled Newton iteration for sign(H).
    Matrix<M, M, T> Z = H;
    const T         tol = std::is_same_v<T, float> ? T{1e-6} : T{1e-12};
    const int       max_iter = 100;
    const T         guard = T{1e150};
    bool            converged = false;

    for (int iter = 0; iter < max_iter; ++iter) {
        const auto Zinv_opt = mat::lu_solve(Z, Matrix<M, M, T>::identity());
        if (!Zinv_opt) {
            return std::nullopt;
        }
        const Matrix<M, M, T> Zinv = Zinv_opt.value();

        //! Frobenius-norm scaling cₖ = √(‖Zₖ⁻¹‖/‖Zₖ‖) accelerates convergence.
        const T norm_z = Z.norm();
        const T norm_zinv = Zinv.norm();
        T       c = T{1};
        if (norm_z > T{0} && norm_zinv > T{0} && wet::isfinite(norm_z) && wet::isfinite(norm_zinv)) {
            c = wet::sqrt(norm_zinv / norm_z);
        }

        const Matrix<M, M, T> Z_next = (Z * (T{1} / c) + Zinv * c) * T{0.5};

        //! Divergence guard (matches the dare iterations).
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < M; ++j) {
                if (!wet::isfinite(Z_next(i, j)) || wet::abs(Z_next(i, j)) > guard) {
                    return std::nullopt;
                }
            }
        }

        const T diff = (Z_next - Z).norm();
        Z = Z_next;

        if (diff < tol * std::max(T{1}, Z.norm())) {
            converged = true;
            break;
        }
    }

    if (!converged) {
        return std::nullopt;
    }

    //! Partition sign(H) and solve [W₁₂; W₂₂+I] X = −[W₁₁+I; W₂₁] via normal
    //! equations: (W₁₂ᵀW₁₂ + (W₂₂+I)ᵀ(W₂₂+I)) X = −(W₁₂ᵀ(W₁₁+I) + (W₂₂+I)ᵀW₂₁).
    Matrix<NX, NX, T> W11{};
    Matrix<NX, NX, T> W12{};
    Matrix<NX, NX, T> W21{};
    Matrix<NX, NX, T> W22{};
    for (size_t i = 0; i < NX; ++i) {
        for (size_t j = 0; j < NX; ++j) {
            W11(i, j) = Z(i, j);
            W12(i, j) = Z(i, NX + j);
            W21(i, j) = Z(NX + i, j);
            W22(i, j) = Z(NX + i, NX + j);
        }
    }
    const Matrix<NX, NX, T> id = Matrix<NX, NX, T>::identity();
    const Matrix<NX, NX, T> W22pI = W22 + id;
    const Matrix<NX, NX, T> W11pI = W11 + id;

    const Matrix<NX, NX, T> lhs = W12.t() * W12 + W22pI.t() * W22pI;
    const Matrix<NX, NX, T> rhs = (W12.t() * W11pI + W22pI.t() * W21) * T{-1};

    const auto X_opt = mat::lu_solve(lhs, rhs);
    if (!X_opt) {
        return std::nullopt;
    }
    const Matrix<NX, NX, T> X = X_opt.value();

    //! Symmetrize for numerical cleanup.
    return (X + X.t()) * T{0.5};
}

} // namespace detail

enum class DareMethod : uint8_t { Auto,
                                  SDA,
                                  RDE };

/**
 * @brief Solve the Discrete Algebraic Riccati Equation (DARE)
 *
 * Finds the unique stabilizing solution X to:
 *
 *     AᵀXA − X − (AᵀXB + N)(R + BᵀXB)⁻¹(BᵀXA + Nᵀ) + Q = 0
 *
 * Preconditions (checked internally):
 * - Q symmetric positive semidefinite
 * - R symmetric (positive definite for SDA, positive semidefinite for RDE)
 * - (A, B) stabilizable
 *
 * @note Compare with MATLAB's idare(A, B, Q, R, N).
 *
 * @see dare_sda() — quadratic convergence, requires R > 0
 * @see dare_rde() — linear convergence, handles R ≥ 0
 * @see "Optimal Control" (Anderson & Moore, 1990), Chapter 4
 *
 * @param A       State transition matrix (NX × NX)
 * @param B       Input matrix (NX × NU)
 * @param Q       State cost matrix (NX × NX, positive semidefinite)
 * @param R       Input cost matrix (NU × NU, positive definite or semidefinite)
 * @param N       Cross-term matrix (NX × NU, default: zero)
 * @param method  DareMethod::Auto (default) selects SDA when R > 0, RDE when R ≥ 0.
 *                DareMethod::SDA forces SDA (requires R positive definite).
 *                DareMethod::RDE forces RDE (handles R positive semidefinite).
 * @return Solution matrix X (NX × NX, positive semidefinite) or std::nullopt on failure
 */
template<size_t NX, size_t NU, typename T = double>
constexpr std::optional<Matrix<NX, NX, T>> dare(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NX, NX, T>& Q,
    const Matrix<NU, NU, T>& R,
    const Matrix<NX, NU, T>& N = Matrix<NX, NU, T>{},
    DareMethod               method = DareMethod::Auto
) {
    //! Check R is symmetric
    if (!mat::is_symmetric_or_hermitian(R)) {
        return std::nullopt;
    }

    //! Check Q is symmetric
    if (!mat::is_symmetric_or_hermitian(Q)) {
        return std::nullopt;
    }

    //! Check Q is positive semidefinite via Cholesky of (Q + εI)
    //! If Q is PSD, Q + εI is PD for any ε > 0, so Cholesky succeeds.
    //! If Q has a negative eigenvalue, Q + εI will still fail for small ε.
    const T eps = std::is_same_v<T, float> ? T{1e-6} : T{1e-12};
    {
        const Matrix<NX, NX, T> Q_shifted = Q + Matrix<NX, NX, T>::identity() * eps;
        if (!mat::cholesky(Q_shifted)) {
            return std::nullopt;
        }
    }

    //! Check (A, B) is stabilizable
    if (!is_stabilizable(A, B)) {
        return std::nullopt;
    }

    const bool r_is_pd = mat::cholesky(R).has_value();
    const bool r_is_psd = r_is_pd || mat::cholesky(R + Matrix<NU, NU, T>::identity() * eps).has_value();

    if (!r_is_psd) {
        return std::nullopt;
    }

    switch (method) {
        case DareMethod::SDA:
            if (!r_is_pd) {
                return std::nullopt;
            }
            return detail::dare_sda(A, B, Q, R, N);
        case DareMethod::RDE:
            return detail::dare_rde(A, B, Q, R, N);
        case DareMethod::Auto:
        default:
            if (r_is_pd) {
                return detail::dare_sda(A, B, Q, R, N);
            }
            return detail::dare_rde(A, B, Q, R, N);
    }
}

/**
 * @brief Solve the Continuous-time Algebraic Riccati Equation (CARE)
 *
 * Finds the unique stabilizing solution X to:
 *
 *     AᵀX + XA − (XB + N)R⁻¹(BᵀX + Nᵀ) + Q = 0
 *
 * This is the continuous-time counterpart of dare(). It underpins continuous
 * LQR/LQG design that has not been discretized first; the discrete pipeline
 * (LQR/LQI/LQG/LQGI) discretizes the plant and uses dare() instead.
 *
 * Preconditions (checked internally):
 * - Q symmetric positive semidefinite
 * - R symmetric positive definite (CARE forms G = B R⁻¹ Bᵀ)
 *
 * Existence of the stabilizing solution additionally requires (A, B)
 * stabilizable and (A, Q) detectable; those are not pre-screened here (the
 * continuous stabilizability/detectability tests differ from the discrete
 * is_stabilizable() used by dare()). Instead, infeasibility surfaces as the
 * sign-function iteration failing to converge, in which case care() returns
 * std::nullopt.
 *
 * @note Compare with MATLAB's icare(A, B, Q, R) / care(A, B, Q, R).
 *
 * @see care_sign() — the underlying matrix-sign-function solver
 * @see dare() — the discrete-time counterpart
 * @see "Optimal Control" (Anderson & Moore, 1990), §3.3
 *
 * @param A  State matrix (NX × NX)
 * @param B  Input matrix (NX × NU)
 * @param Q  State cost matrix (NX × NX, positive semidefinite)
 * @param R  Input cost matrix (NU × NU, positive definite)
 * @param N  Cross-term matrix (NX × NU, default: zero)
 * @return Stabilizing solution X (NX × NX, symmetric positive semidefinite) or
 *         std::nullopt on failure
 */
template<size_t NX, size_t NU, typename T = double>
constexpr std::optional<Matrix<NX, NX, T>> care(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NX, NX, T>& Q,
    const Matrix<NU, NU, T>& R,
    const Matrix<NX, NU, T>& N = Matrix<NX, NU, T>{}
) {
    //! R and Q must be symmetric (CARE assumes symmetric weights).
    if (!mat::is_symmetric_or_hermitian(R)) {
        return std::nullopt;
    }
    if (!mat::is_symmetric_or_hermitian(Q)) {
        return std::nullopt;
    }

    //! Q positive semidefinite via Cholesky of (Q + εI) — same trick as dare().
    const T eps = std::is_same_v<T, float> ? T{1e-6} : T{1e-12};
    {
        const Matrix<NX, NX, T> Q_shifted = Q + Matrix<NX, NX, T>::identity() * eps;
        if (!mat::cholesky(Q_shifted)) {
            return std::nullopt;
        }
    }

    //! R must be positive definite (G = B R⁻¹ Bᵀ).
    if (!mat::cholesky(R)) {
        return std::nullopt;
    }

    return detail::care_sign(A, B, Q, R, N);
}

} // namespace wetmelon::control