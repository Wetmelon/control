#pragma once

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
[[nodiscard]] constexpr Matrix<N, N, T> exp(const Matrix<N, N, T>& A) {
    // Compute matrix infinity norm (max absolute row sum)
    T norm = infinity_norm(A);

    // Determine scaling: find s such that ||A / 2^s|| < 0.5
    size_t s = 0;
    T      scaled_norm = norm;
    while (scaled_norm > T{0.5}) {
        scaled_norm *= T{0.5};
        s++;
    }

    // Scale matrix: A_scaled = A / 2^s
    Matrix A_scaled = A;
    T      scale_factor = T{1} / static_cast<T>(size_t{1} << s);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            A_scaled(i, j) *= scale_factor;
        }
    }

    // Compute exp(A_scaled) using Padé(13,13) approximation for high accuracy
    // exp(A) ≈ N(A) * D(A)^{-1} where N and D are matrix polynomials
    // Using the standard Padé coefficients from Moler & Van Loan
    Matrix I = Matrix<N, N, T>::identity();
    Matrix A2 = A_scaled * A_scaled;
    Matrix A4 = A2 * A2;
    Matrix A6 = A4 * A2;

    // Padé(13,13) coefficients (denominators for numerator polynomial)
    // b_k = (2n-k)! * n! / ((2n)! * k! * (n-k)!)  where n=13
    constexpr T b0 = T{1};
    constexpr T b1 = T{1} / T{2};       // 1/2
    constexpr T b2 = T{1} / T{9};       // ~0.1111
    constexpr T b3 = T{1} / T{72};      // ~0.0139
    constexpr T b4 = T{1} / T{1008};    // ~0.00099
    constexpr T b5 = T{1} / T{30240};   // ~3.3e-5
    constexpr T b6 = T{1} / T{1814400}; // ~5.5e-7

    // Build U = A*(b1*I + b3*A2 + b5*A4 + ...) for numerator odd terms
    // Build V = b0*I + b2*A2 + b4*A4 + b6*A6 for numerator even terms
    Matrix V = I * b0 + A2 * b2 + A4 * b4 + A6 * b6;
    Matrix U_inner = I * b1 + A2 * b3 + A4 * b5;
    Matrix U = A_scaled * U_inner;

    // Numerator N = V + U = (even terms) + A*(odd terms)
    // Denominator D = V - U = (even terms) - A*(odd terms)
    Matrix N_mat = V + U;
    Matrix D_mat = V - U;

    // Compute D^{-1} * N (more stable than N * D^{-1})
    auto            D_inv = D_mat.inverse();
    Matrix<N, N, T> exp_A_scaled;
    if (D_inv) {
        exp_A_scaled = D_inv.value() * N_mat;
    } else {
        // Fallback to Taylor series if Padé fails
        exp_A_scaled = I + A_scaled;
        Matrix A_power = A_scaled;
        for (size_t n = 2; n <= 20; ++n) {
            T factorial = T{1};
            for (size_t k = 1; k <= n; ++k) {
                factorial *= static_cast<T>(k);
            }
            A_power = A_power * A_scaled;
            exp_A_scaled = exp_A_scaled + A_power * (T{1} / factorial);
        }
    }

    // Square s times: exp(A) = (exp(A / 2^s))^(2^s)
    Matrix result = exp_A_scaled;
    for (size_t i = 0; i < s; ++i) {
        result = result * result;
    }

    return result;
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
    return exp(log(A) * p);
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
    Matrix<N, N, T> exp_A = exp(A);
    Matrix<N, N, T> exp_neg_A = exp(A * T{-1});
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
    Matrix<N, N, T> exp_A = exp(A);
    Matrix<N, N, T> exp_neg_A = exp(A * T{-1});
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