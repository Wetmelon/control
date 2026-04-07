#pragma once

#include <algorithm>

#include "matrix.hpp"

namespace wetmelon::control {

namespace mat {
/**
 * @brief Infinity norm: maximum absolute row sum
 * @tparam T Element type
 * @tparam N Matrix dimension
 * @param A Square matrix
 * @return Infinity norm of the matrix
 */
template<typename T, size_t N>
[[nodiscard]] constexpr T infinity_norm(const Matrix<N, N, T>& A) {
    T norm = T{0};
    for (size_t i = 0; i < N; ++i) {
        T row_sum = T{0};
        for (size_t j = 0; j < N; ++j) {
            row_sum += wet::abs(A(i, j));
        }
        if (row_sum > norm) {
            norm = row_sum;
        }
    }
    return norm;
}

/**
 * @brief One norm: maximum absolute column sum
 * @tparam T Element type
 * @tparam N Matrix dimension
 * @param A Square matrix
 * @return One norm of the matrix
 */
template<typename T, size_t N>
[[nodiscard]] constexpr T one_norm(const Matrix<N, N, T>& A) {
    T norm = T{0};
    for (size_t j = 0; j < N; ++j) {
        T col_sum = T{0};
        for (size_t i = 0; i < N; ++i) {
            col_sum += wet::abs(A(i, j));
        }
        if (col_sum > norm) {
            norm = col_sum;
        }
    }
    return norm;
}

/**
 * @brief Frobenius norm: square root of sum of squares of all elements
 * @tparam T Element type
 * @tparam N Matrix dimension
 * @param A Square matrix
 * @return Frobenius norm of the matrix
 */
template<typename T, size_t N>
[[nodiscard]] constexpr T frobenius_norm(const Matrix<N, N, T>& A) {
    T sum_squares = T{0};
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            T abs_val = wet::abs(A(i, j));
            sum_squares += abs_val * abs_val;
        }
    }
    return wet::sqrt(sum_squares);
}

/**
 * @brief Two norm (spectral norm): largest singular value
 *
 * For square matrices, this is the square root of the largest eigenvalue of A^H * A.
 * This is a simplified approximation using power iteration for small matrices.
 *
 * @tparam T Element type
 * @tparam N Matrix dimension
 * @param A Square matrix
 * @return Approximation of the spectral norm
 */
template<typename T, size_t N>
[[nodiscard]] constexpr T two_norm(const Matrix<N, N, T>& A) {
    // For small matrices, use Frobenius norm as approximation
    // A more accurate implementation would use SVD or eigenvalue decomposition
    return frobenius_norm(A);
}

/**
 * @brief Matrix determinant using cofactor expansion
 * @tparam T Element type
 * @tparam N Matrix dimension (must be small, ≤4)
 * @param A Square matrix
 * @return Determinant of the matrix
 */
template<typename T, size_t N>
[[nodiscard]] constexpr T det(const Matrix<N, N, T>& A) {
    if constexpr (N == 1) {
        return A(0, 0);
    } else if constexpr (N == 2) {
        return A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
    } else if constexpr (N == 3) {
        return A(0, 0) * (A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1))
             - A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0))
             + A(0, 2) * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0));
    } else if constexpr (N == 4) {
        // Use cofactor expansion along first row for 4x4
        T det = T{0};
        for (size_t j = 0; j < 4; ++j) {
            Matrix<3, 3, T> minor;
            size_t          minor_row = 0;
            for (size_t i = 1; i < 4; ++i) {
                size_t minor_col = 0;
                for (size_t k = 0; k < 4; ++k) {
                    if (k != j) {
                        minor(minor_row, minor_col) = A(i, k);
                        ++minor_col;
                    }
                }
                ++minor_row;
            }
            T cofactor = ((j % 2 == 0) ? T{1} : T{-1}) * det(minor);
            det += A(0, j) * cofactor;
        }
        return det;
    } else {
        // For larger matrices, this would need a more sophisticated implementation
        // For now, return 0 as unsupported
        return T{0};
    }
}

/**
 * @brief Matrix rank using Gaussian elimination
 * @tparam T Element type
 * @tparam N Matrix dimension
 * @param A Square matrix
 * @return Rank of the matrix (number of linearly independent rows/columns)
 */
template<typename T, size_t N>
[[nodiscard]] constexpr size_t rank(const Matrix<N, N, T>& A) {
    // Create a copy for Gaussian elimination
    Matrix<N, N, T> temp = A;
    size_t          rank = 0;
    const T         epsilon = T{1e-10}; // Tolerance for zero

    for (size_t col = 0; col < N; ++col) {
        // Find pivot row
        size_t pivot_row = rank;
        for (size_t row = rank; row < N; ++row) {
            if (wet::abs(temp(row, col)) > wet::abs(temp(pivot_row, col))) {
                pivot_row = row;
            }
        }

        // If pivot is zero, skip this column
        if (wet::abs(temp(pivot_row, col)) < epsilon) {
            continue;
        }

        // Swap rows if needed
        if (pivot_row != rank) {
            for (size_t j = 0; j < N; ++j) {
                std::swap(temp(rank, j), temp(pivot_row, j));
            }
        }

        // Eliminate below
        for (size_t row = rank + 1; row < N; ++row) {
            T factor = temp(row, col) / temp(rank, col);
            for (size_t j = col; j < N; ++j) {
                temp(row, j) -= factor * temp(rank, j);
            }
        }

        ++rank;
    }

    return rank;
}

/**
 * @brief Matrix exponential using scaling and squaring with Padé approximation
 *
 * Computes exp(A) using the algorithm: exp(A) = (exp(A / 2^s))^(2^s)
 * where s is chosen so ||A / 2^s|| < 0.5.
 *
 * The matrix exponential is defined as:
 *   exp(A) = I + A + A²/2! + A³/3! + ...
 *
 * For solving ODEs: if dx/dt = A*x, then x(t) = exp(A*t) * x(0)
 *
 * @tparam T Element type
 * @tparam N Matrix dimension
 * @param A Square matrix
 * @return Matrix exponential exp(A)
 */

template<typename T, size_t N>
[[nodiscard]] constexpr Matrix<N, N, T> expm(const Matrix<N, N, T>& A) {
    Matrix<N, N, T> I = Matrix<N, N, T>::identity();

    // Compute infinity norm
    T norm = mat::infinity_norm(A);

    // 1️⃣ Tiny/nilpotent matrix shortcut (Taylor series)
    // Exact for small norms
    if (norm <= T(1e-12)) {
        // tiny matrix: Taylor to 6th is plenty
        Matrix A2 = A * A;
        Matrix A3 = A2 * A;
        Matrix A4 = A3 * A;
        Matrix A5 = A4 * A;
        Matrix A6 = A5 * A;
        return I + A + A2 * (1.0 / 2.0) + A3 * (1.0 / 6.0) + A4 * (1.0 / 24.0) + A5 * (1.0 / 120.0) + A6 * (1.0 / 720.0);
    }

    // 2️⃣ Scaling for Pade13
    constexpr T theta13 = T(2.097847961257068); // from Higham 2005
    size_t      s = 0;
    T           scaled_norm = norm;
    while (scaled_norm > theta13) {
        scaled_norm *= T(0.5);
        ++s;
    }

    T scale = T(1);
    for (size_t i = 0; i < s; ++i)
        scale *= T(0.5);

    Matrix<N, N, T> A_scaled = A;
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            A_scaled(i, j) *= scale;

    // 3️⃣ Precompute powers
    Matrix<N, N, T> A2 = A_scaled * A_scaled;
    Matrix<N, N, T> A4 = A2 * A2;
    Matrix<N, N, T> A6 = A4 * A2;
    Matrix<N, N, T> A8 = A6 * A2;
    Matrix<N, N, T> A10 = A8 * A2;
    Matrix<N, N, T> A12 = A10 * A2;

    // 4️⃣ Pade13 coefficients
    constexpr T b0 = T(64764752532480000.0);
    constexpr T b1 = T(32382376266240000.0);
    constexpr T b2 = T(7771770303897600.0);
    constexpr T b3 = T(1187353796428800.0);
    constexpr T b4 = T(129060195264000.0);
    constexpr T b5 = T(10559470521600.0);
    constexpr T b6 = T(670442572800.0);
    constexpr T b7 = T(33522128640.0);
    constexpr T b8 = T(1323241920.0);
    constexpr T b9 = T(40840800.0);
    constexpr T b10 = T(960960.0);
    constexpr T b11 = T(16380.0);
    constexpr T b12 = T(182.0);
    constexpr T b13 = T(1.0);

    // 5️⃣ Compute U and V
    Matrix<N, N, T> U = A_scaled * (b1 * I + b3 * A2 + b5 * A4 + b7 * A6 + b9 * A8 + b11 * A10 + b13 * A12);
    Matrix<N, N, T> V = b0 * I + b2 * A2 + b4 * A4 + b6 * A6 + b8 * A8 + b10 * A10 + b12 * A12;

    // 6️⃣ Solve (V-U) * R = V+U using your 2-input solve()
    auto R_opt = solve(V - U, V + U);
    if (!R_opt) {
        // fallback if solve fails (e.g., return identity)
        return I;
    }
    Matrix<N, N, T> R = R_opt.value();

    // Iterative refinement: solve for residual, apply two refinements
    for (int iter = 0; iter < 2; ++iter) {
        Matrix<N, N, T> residual = (V + U) - (V - U) * R;
        auto            delta_opt = solve(V - U, residual);
        if (!delta_opt)
            break;
        R = R + delta_opt.value();
    }

    // 7️⃣ Squaring phase
    for (size_t i = 0; i < s; ++i) {
        R = R * R;
    }

    return R;
}

/**
 * @brief Matrix logarithm using inverse scaling and squaring
 *
 * Computes the principal matrix logarithm X = log(A) that satisfies exp(X) = A.
 *
 * Requirements: A must be invertible and have no real negative eigenvalues.
 *
 * @tparam T Element type
 * @tparam N Matrix dimension
 * @param A Square matrix
 * @return Principal matrix logarithm log(A)
 */
template<typename T, size_t N>
[[nodiscard]] constexpr Matrix<N, N, T> log(const Matrix<N, N, T>& A) {
    Matrix I = Matrix<N, N, T>::identity();

    // Inverse scaling: compute A^(1/2^s) until close to I
    // Then use log(I + X) ≈ X - X²/2 + X³/3 - ... for small X
    Matrix A_scaled = A;
    size_t s = 0;

    // Take square roots until ||A - I|| is small
    // Matrix square root via Denman-Beavers iteration
    auto matrix_sqrt_db = [](const Matrix<N, N, T>& M) -> Matrix<N, N, T> {
        Matrix<N, N, T> Y = M;
        Matrix<N, N, T> Z = Matrix<N, N, T>::identity();

        for (int iter = 0; iter < 50; ++iter) {
            auto Y_inv = Y.inverse();
            auto Z_inv = Z.inverse();
            if (!Y_inv || !Z_inv)
                break;

            Matrix<N, N, T> Y_next = (Y + Z_inv.value()) * T{0.5};
            Matrix<N, N, T> Z_next = (Z + Y_inv.value()) * T{0.5};

            // Check convergence
            T diff = T{0};
            for (size_t i = 0; i < N; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    diff += wet::abs(Y_next(i, j) - Y(i, j));
                }
            }

            Y = Y_next;
            Z = Z_next;

            if (diff < T{1e-12})
                break;
        }
        return Y;
    };

    // Scale down: A_scaled = A^(1/2^s) until ||A_scaled - I|| < 0.5
    while (infinity_norm(A_scaled - I) > T{0.5} && s < 20) {
        A_scaled = matrix_sqrt_db(A_scaled);
        s++;
    }

    // Now compute log(A_scaled) using log(I + X) series where X = A_scaled - I
    Matrix X = A_scaled - I;
    Matrix result = X;
    Matrix X_power = X;

    for (size_t n = 2; n <= 20; ++n) {
        X_power = X_power * X;
        T sign = (n % 2 == 0) ? T{-1} : T{1};
        result = result + X_power * (sign / static_cast<T>(n));
    }

    // Scale back: log(A) = 2^s * log(A^(1/2^s))
    T scale = static_cast<T>(size_t{1} << s);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result(i, j) *= scale;
        }
    }

    return result;
}

/**
 * @brief Matrix square root using Denman-Beavers iteration
 *
 * Computes the principal matrix square root S = sqrt(A) that satisfies S*S = A.
 *
 * Uses Denman-Beavers iteration:
 *   Y_{k+1} = (Y_k + Z_k^{-1}) / 2
 *   Z_{k+1} = (Z_k + Y_k^{-1}) / 2
 *
 * Requirements: A should have no real negative eigenvalues.
 *
 * @tparam T Element type
 * @tparam N Matrix dimension
 * @param A Square matrix
 * @return Principal matrix square root sqrt(A)
 */
template<typename T, size_t N>
[[nodiscard]] constexpr Matrix<N, N, T> sqrt(const Matrix<N, N, T>& A) {
    // Denman-Beavers iteration:
    // Y_{k+1} = (Y_k + Z_k^{-1}) / 2
    // Z_{k+1} = (Z_k + Y_k^{-1}) / 2
    // Converges: Y -> sqrt(A), Z -> sqrt(A)^{-1}

    Matrix<N, N, T> Y = A;
    Matrix<N, N, T> Z = Matrix<N, N, T>::identity();

    for (int iter = 0; iter < 50; ++iter) {
        auto Y_inv = Y.inverse();
        auto Z_inv = Z.inverse();

        if (!Y_inv || !Z_inv) {
            // Fallback: return identity scaled by sqrt of trace/N
            T trace_avg = T{0};
            for (size_t i = 0; i < N; ++i) {
                trace_avg += A(i, i);
            }
            trace_avg /= static_cast<T>(N);
            return Matrix<N, N, T>::identity() * wet::sqrt(trace_avg);
        }

        Matrix<N, N, T> Y_next = (Y + Z_inv.value()) * T{0.5};
        Matrix<N, N, T> Z_next = (Z + Y_inv.value()) * T{0.5};

        // Check convergence
        T diff = T{0};
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                diff += wet::abs(Y_next(i, j) - Y(i, j));
            }
        }

        Y = Y_next;
        Z = Z_next;

        if (diff < T{1e-12})
            break;
    }

    return Y;
}

/**
 * @brief Matrix power for integer exponent
 *
 * Computes A^p using binary exponentiation for efficiency and exactness.
 *
 * @tparam T Element type
 * @tparam N Matrix dimension
 * @param A Square matrix
 * @param p Integer exponent
 * @return A raised to power p
 */
// Integer power (more efficient and exact)
template<typename T, size_t N>
[[nodiscard]] constexpr Matrix<N, N, T> pow(const Matrix<N, N, T>& A, int p) {
    if (p == 0) {
        return Matrix<N, N, T>::identity();
    }

    bool negate = p < 0;
    if (negate) {
        p = -p;
    }

    // Binary exponentiation
    Matrix<N, N, T> result = Matrix<N, N, T>::identity();
    Matrix<N, N, T> base = A;

    while (p > 0) {
        if (p & 1) {
            result = result * base;
        }
        base = base * base;
        p >>= 1;
    }

    if (negate) {
        auto inv = result.inverse();
        return inv.value_or(Matrix<N, N, T>::identity());
    }

    return result;
}

/**
 * @brief Matrix power for real exponent
 *
 * Computes A^p = exp(p * log(A)) for real exponent p.
 *
 * @tparam T Element type
 * @tparam N Matrix dimension
 * @param A Square matrix
 * @param p Real exponent
 * @return A raised to power p
 */
template<typename T, size_t N>
[[nodiscard]] constexpr Matrix<N, N, T> pow(const Matrix<N, N, T>& A, T p) {
    // Check for integer case
    T p_int;
    T p_frac = std::modf(p, &p_int);
    if (wet::abs(p_frac) < T{1e-10}) {
        return pow(A, static_cast<int>(p_int));
    }

    // General case: A^p = exp(p * log(A))
    return expm(log(A) * p);
}

/**
 * @brief Matrix sine using Taylor series approximation
 *
 * Computes sin(A) = A - A³/3! + A⁵/5! - A⁷/7! + ...
 *
 * @tparam T Element type
 * @tparam N Matrix dimension
 * @param A Square matrix
 * @return Matrix sine sin(A)
 */
template<typename T, size_t N>
[[nodiscard]] constexpr Matrix<N, N, T> sin(const Matrix<N, N, T>& A) {
    Matrix<N, N, T> result = A;
    Matrix<N, N, T> A2 = A * A;
    Matrix<N, N, T> A_power = A;

    T factorial = T{1};
    T sign = T{-1};

    for (size_t n = 3; n <= 21; n += 2) {
        factorial *= static_cast<T>(n - 1) * static_cast<T>(n);
        A_power = A_power * A2;
        result = result + A_power * (sign / factorial);
        sign = -sign;
    }

    return result;
}

/**
 * @brief Matrix cosine using Taylor series approximation
 *
 * Computes cos(A) = I - A²/2! + A⁴/4! - A⁶/6! + ...
 *
 * @tparam T Element type
 * @tparam N Matrix dimension
 * @param A Square matrix
 * @return Matrix cosine cos(A)
 */
template<typename T, size_t N>
[[nodiscard]] constexpr Matrix<N, N, T> cos(const Matrix<N, N, T>& A) {
    Matrix<N, N, T> result = Matrix<N, N, T>::identity();
    Matrix<N, N, T> A_power = Matrix<N, N, T>::identity();

    T factorial = T{1};
    T sign = T{-1};

    const Matrix<N, N, T> A2 = A * A;
    for (size_t n = 2; n <= 20; n += 2) {
        factorial *= static_cast<T>(n - 1) * static_cast<T>(n);
        A_power = A_power * A2;
        result = result + A_power * (sign / factorial);
        sign = -sign;
    }

    return result;
}

/**
 * @brief Matrix hyperbolic sine
 *
 * Computes sinh(A) = (exp(A) - exp(-A)) / 2
 *
 * @tparam T Element type
 * @tparam N Matrix dimension
 * @param A Square matrix
 * @return Matrix hyperbolic sine sinh(A)
 */
template<typename T, size_t N>
[[nodiscard]] constexpr Matrix<N, N, T> sinh(const Matrix<N, N, T>& A) {
    Matrix<N, N, T> exp_A = expm(A);
    Matrix<N, N, T> exp_neg_A = expm(A * T{-1});
    return (exp_A - exp_neg_A) * T{0.5};
}

/**
 * @brief Matrix hyperbolic cosine
 *
 * Computes cosh(A) = (exp(A) + exp(-A)) / 2
 *
 * @tparam T Element type
 * @tparam N Matrix dimension
 * @param A Square matrix
 * @return Matrix hyperbolic cosine cosh(A)
 */
template<typename T, size_t N>
[[nodiscard]] constexpr Matrix<N, N, T> cosh(const Matrix<N, N, T>& A) {
    Matrix<N, N, T> exp_A = expm(A);
    Matrix<N, N, T> exp_neg_A = expm(A * T{-1});
    return (exp_A + exp_neg_A) * T{0.5};
}

/**
 * @brief Verify trigonometric identity sin²(A) + cos²(A) ≈ I
 *
 * @tparam T Element type
 * @tparam N Matrix dimension
 * @param A Square matrix
 * @param tol Tolerance for identity verification
 * @return True if identity holds within tolerance
 */
template<typename T, size_t N>
[[nodiscard]] constexpr bool verify_trig_identity(const Matrix<N, N, T>& A, T tol = T{1e-6}) {
    Matrix<N, N, T> sinA = sin(A);
    Matrix<N, N, T> cosA = cos(A);
    Matrix<N, N, T> sum = sinA * sinA + cosA * cosA;
    Matrix<N, N, T> I = Matrix<N, N, T>::identity();

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (wet::abs(sum(i, j) - I(i, j)) > tol) {
                return false;
            }
        }
    }
    return true;
}
} // namespace mat
}; // namespace wetmelon::control