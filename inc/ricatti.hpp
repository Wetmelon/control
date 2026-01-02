#pragma once

#include <cstddef>

#include "eigen.hpp"
#include "eigenvalues.hpp"
#include "matrix.hpp"

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
    Matrix<2, 2, std::complex<T>> U1 = Matrix<2, 2, std::complex<T>>::zeros();
    Matrix<2, 2, std::complex<T>> U2 = Matrix<2, 2, std::complex<T>>::zeros();

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
    std::complex<T> det_U1 = U1(0, 0) * U1(1, 1) - U1(0, 1) * U1(1, 0);
    T               det_mag = std::norm(det_U1);

    if (det_mag < T{1e-30}) {
        // U1 is singular
        return Matrix<2, 2, T>::zeros();
    }

    Matrix<2, 2, std::complex<T>> U1_inv = Matrix<2, 2, std::complex<T>>::zeros();
    U1_inv(0, 0) = U1(1, 1) / det_U1;
    U1_inv(0, 1) = -U1(0, 1) / det_U1;
    U1_inv(1, 0) = -U1(1, 0) / det_U1;
    U1_inv(1, 1) = U1(0, 0) / det_U1;

    //! P = U2 * U1_inv
    Matrix<2, 2, std::complex<T>> P_complex = U2 * U1_inv;

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
 * @brief Solve Continuous Algebraic Riccati Equation (CARE)
 *
 * Solves A'P + PA - PBR⁻¹B'P + Q = 0 for optimal LQR gain computation.
 * Uses specialized solvers for 1x1 and 2x2 cases, Newton iteration for larger systems.
 *
 * @tparam NX Number of states
 * @tparam NU Number of control inputs
 * @tparam T  Scalar type (default: double)
 * @param A   State matrix (NX × NX)
 * @param B   Control input matrix (NX × NU)
 * @param Q   State cost matrix (NX × NX, positive semidefinite)
 * @param R   Control cost matrix (NU × NU, positive definite)
 *
 * @return Solution matrix P (NX × NX, positive semidefinite), or zero matrix on failure
 */
template<size_t NX, size_t NU, typename T = double>
constexpr Matrix<NX, NX, T> care(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NX, NX, T>& Q,
    const Matrix<NU, NU, T>& R
) {
    const auto Rinv = R.inverse();
    if (!Rinv) {
        return Matrix<NX, NX, T>::zeros(); //! Return zero on failure
    }

    //! For 1x1 case, solve quadratic directly
    if constexpr (NX == 1 && NU == 1) {
        T a = A(0, 0), b = B(0, 0), q = Q(0, 0), r = R(0, 0);
        T rinv = T{1} / r;
        //! Equation: a*p + p*a - p*b*rinv*b*p + q = 0
        //! Rearranged: b²*rinv*p² + 2*a*p - q = 0
        T aa = b * b * rinv;
        T bb = T{-2} * a;
        T cc = -q;
        T disc = bb * bb - T{4} * aa * cc;
        T p1 = (-bb + wet::sqrt(disc)) / (T{2} * aa);
        T p2 = (-bb - wet::sqrt(disc)) / (T{2} * aa);
        //! Choose the positive definite one
        T p = p1 > T{0} ? p1 : p2;
        // values computed above are used to select positive root `p`
        Matrix<NX, NX, T> result = Matrix<NX, NX, T>::zeros();
        result(0, 0) = p;
        return result;
    }

    //! For 2x2 case with 1 input, use Hamiltonian eigenvalue approach
    //! (more numerically robust for ill-conditioned systems)
    if constexpr (NX == 2 && NU == 1) {
        return care_2x2_hamiltonian(A, B, Q, R);
    }

    //! Check positive definiteness of R
    auto R_eigen = compute_eigenvalues_qr(R);
    if (R_eigen.converged) {
        for (size_t i = 0; i < NU; ++i) {
            if (R_eigen.eigenvalues_real(i, i) <= T{0}) {
                return Matrix<NX, NX, T>::zeros(); //! R not positive definite
            }
        }
    }

    //! Check positive semidefiniteness of Q
    auto Q_eigen = compute_eigenvalues_qr(Q);
    if (Q_eigen.converged) {
        for (size_t i = 0; i < NX; ++i) {
            if (Q_eigen.eigenvalues_real(i, i) < T{-1e-12}) {
                return Matrix<NX, NX, T>::zeros(); //! Q not positive semidefinite
            }
        }
    }

    //! Fall back to a damped Newton iteration
    Matrix<NX, NX, T> X = Matrix<NX, NX, T>::identity() * T{1e-6};
    const T           tol = T{1e-10};
    const int         max_iter = 1000;
    const T           alpha = T{0.25};

    for (int iter = 0; iter < max_iter; ++iter) {
        Matrix<NX, NX, T> X_prev = X;
        Matrix<NX, NX, T> F = A.transpose() * X + X * A - X * B * Rinv.value() * B.transpose() * X + Q;
        Matrix<NX, NX, T> DX = lyap(A.transpose(), -F);

        //! Check if DX is finite and compute Frobenius norm
        bool is_finite = true;
        T    dx_norm_sq = T{0};
        for (size_t i = 0; i < NX; ++i) {
            for (size_t j = 0; j < NX; ++j) {
                T val = DX(i, j);
                if (!std::isfinite(val)) {
                    is_finite = false;
                    break;
                }
                //! Check if safe to square (avoid overflow during constexpr)
                T abs_val = val < T{0} ? -val : val;
                if (abs_val > T{1e150}) { //! sqrt(DBL_MAX) ≈ 1.34e154
                    is_finite = false;
                    break;
                }
                dx_norm_sq += val * val;
            }
            if (!is_finite)
                break;
        }
        if (!is_finite)
            break;

        X = X_prev + DX * alpha;
        if (wet::sqrt(dx_norm_sq) < tol)
            break;
    }

    return X;
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
 * @return Solution matrix P (NX × NX, positive semidefinite), or zero matrix on failure
 */
template<size_t NX, size_t NU, typename T = double>
constexpr Matrix<NX, NX, T> dare(const Matrix<NX, NX, T>& A, const Matrix<NX, NU, T>& B, const Matrix<NX, NX, T>& Q, const Matrix<NU, NU, T>& R) {
    //! Check positive definiteness of R
    auto R_eigen = compute_eigenvalues_qr(R);
    if (R_eigen.converged) {
        for (size_t i = 0; i < NU; ++i) {
            if (R_eigen.eigenvalues_real(i, i) <= T{0}) {
                return Matrix<NX, NX, T>::zeros(); //! R not positive definite
            }
        }
    }

    //! Check positive semidefiniteness of Q
    auto Q_eigen = compute_eigenvalues_qr(Q);
    if (Q_eigen.converged) {
        for (size_t i = 0; i < NX; ++i) {
            if (Q_eigen.eigenvalues_real(i, i) < T{-1e-12}) {
                return Matrix<NX, NX, T>::zeros(); //! Q not positive semidefinite
            }
        }
    }

    // Use simple fixed-point (Kleinman) iteration for DARE:
    // X_{k+1} = A^T X_k A - A^T X_k B (R + B^T X_k B)^{-1} B^T X_k A + Q
    Matrix<NX, NX, T> X = Q; // initial guess
    const T           tol = T{1e-9};
    const int         max_iter = 1000;

    for (int iter = 0; iter < max_iter; ++iter) {
        Matrix<NX, NX, T> X_prev = X;

        Matrix<NU, NU, T> BRB = B.transpose() * X * B;
        Matrix<NU, NU, T> temp = R + BRB;
        auto              temp_inv_opt = temp.inverse();

        if (!temp_inv_opt)
            break; // Singular matrix

        Matrix<NU, NU, T> temp_inv = temp_inv_opt.value();
        Matrix<NX, NX, T> X_new = A.transpose() * X * A - A.transpose() * X * B * temp_inv * B.transpose() * X * A + Q;

        Matrix<NX, NX, T> diff = X_new - X_prev;
        X = X_new;

        // Compute Frobenius norms
        T diff_norm_sq = T{0};
        T x_norm_sq = T{0};
        for (size_t i = 0; i < NX; ++i) {
            for (size_t j = 0; j < NX; ++j) {
                diff_norm_sq += diff(i, j) * diff(i, j);
                x_norm_sq += X(i, j) * X(i, j);
            }
        }

        if (wet::sqrt(diff_norm_sq) < tol * std::max(T{1}, wet::sqrt(x_norm_sq)))
            break;
    }

    return X;
}

/**
 * @brief Solve continuous Lyapunov equation
 *
 * Solves A'X + XA + Q = 0 by vectorizing and solving linear system.
 * Used in Riccati equation Newton iterations.
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
constexpr auto lyap(const Matrix<Rows, Cols, T>& A, const Matrix<Rows, Cols, T>& Q) -> Matrix<Rows, Cols, T> {
    static_assert(Rows == Cols, "Lyapunov equation requires square matrices");

    constexpr auto n = Rows;
    constexpr int  N = n * n;

    Matrix<N, N, T>       K = Matrix<N, N, T>::zeros();
    auto                  idx = [&](size_t i, size_t j) { return i + j * n; };
    Matrix<Rows, Cols, T> At = A.transpose();
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            size_t row = idx(i, j);
            for (size_t r = 0; r < n; ++r) {
                for (size_t s = 0; s < n; ++s) {
                    size_t col = idx(r, s);
                    T      val = T{0};
                    if (i == r)
                        val += At(j, s);
                    if (j == s)
                        val += At(i, r);
                    K(row, col) = val;
                }
            }
        }
    }

    ColVec<N, T> vecQ = ColVec<N, T>::zeros();
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < n; ++i)
            vecQ[idx(i, j)] = Q(i, j);

    auto K_inv = K.inverse();
    if (!K_inv)
        return Matrix<Rows, Cols, T>::zeros(); // Return zero matrix if singular

    ColVec<N, T>          vecX = K_inv.value() * (-vecQ);
    Matrix<Rows, Cols, T> X = Matrix<Rows, Cols, T>::zeros();
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < n; ++i)
            X(i, j) = vecX[idx(i, j)];

    return (X + X.transpose()) * 0.5;
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
constexpr auto dlyap(const Matrix<Rows, Cols, T>& A, const Matrix<Rows, Cols, T>& Q) -> Matrix<Rows, Cols, T> {
    static_assert(Rows == Cols, "Lyapunov equation requires square matrices");

    constexpr auto n = Rows;
    constexpr int  N = n * n;

    Matrix<N, N, T> K = Matrix<N, N, T>::zeros();
    auto            idx = [&](size_t i, size_t j) { return i + j * n; };
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            size_t row = idx(i, j);
            for (size_t r = 0; r < n; ++r) {
                for (size_t s = 0; s < n; ++s) {
                    size_t col = idx(r, s);
                    T      val = A(i, r) * A(s, j);
                    if (row == col)
                        val -= T{1};
                    K(row, col) = val;
                }
            }
        }
    }

    ColVec<N, T> vecQ = ColVec<N, T>::zeros();
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < n; ++i)
            vecQ[idx(i, j)] = Q(i, j);

    auto K_inv = K.inverse();
    if (!K_inv)
        return Matrix<Rows, Cols, T>::zeros(); // Return zero matrix if singular

    ColVec<N, T>          vecX = K_inv.value() * (-vecQ);
    Matrix<Rows, Cols, T> X = Matrix<Rows, Cols, T>::zeros();
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < n; ++i)
            X(i, j) = vecX[idx(i, j)];

    return (X + X.transpose()) * 0.5;
}
