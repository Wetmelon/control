#pragma once

#include <cmath>
#include <complex>
#include <type_traits>

#include "constexpr_math.hpp"
#include "matrix.hpp"

// ============================================================================
// Matrix Functions Module
// ============================================================================
// Eigen-inspired API for computing matrix functions. These operate on the
// matrix as a whole (e.g., matrix exponential) rather than element-wise.
//
// For element-wise operations, use standard loops or ArrayBase-style access.
//
// Available functions (in mat:: namespace):
//   - mat::exp(A)   : Matrix exponential
//   - mat::log(A)   : Matrix logarithm (principal branch)
//   - mat::sqrt(A)  : Matrix square root (principal branch)
//   - mat::pow(A,p) : Matrix power (integer or real exponent)
//   - mat::sin(A)   : Matrix sine
//   - mat::cos(A)   : Matrix cosine
//   - mat::sinh(A)  : Matrix hyperbolic sine
//   - mat::cosh(A)  : Matrix hyperbolic cosine
//
// Implementation uses scaling-and-squaring with Padé approximation for exp(),
// and Taylor series for trigonometric functions. Designed for small fixed-size
// matrices typical in control systems (up to 4x4).

namespace mat {

// ============================================================================
// Matrix Infinity Norm (helper for scaling decisions)
// ============================================================================
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

// ============================================================================
// Matrix Exponential: exp(A)
// ============================================================================
// Computes the matrix exponential using scaling and squaring with Padé approximation.
// Algorithm: exp(A) = (exp(A / 2^s))^(2^s) where s chosen so ||A / 2^s|| < 0.5
//
// The matrix exponential is defined as:
//   exp(A) = I + A + A²/2! + A³/3! + ...
//
// For solving ODEs: if dx/dt = A*x, then x(t) = exp(A*t) * x(0)

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

    // Compute exp(A_scaled) using Padé(6,6) approximation
    // exp(A) ≈ N(A) * D(A)^{-1} where N and D are matrix polynomials
    Matrix I = Matrix<N, N, T>::identity();
    Matrix A2 = A_scaled * A_scaled;
    Matrix A4 = A2 * A2;
    Matrix A6 = A4 * A2;

    // Padé(6,6) coefficients
    constexpr T c0 = T{1};
    constexpr T c1 = T{1} / T{2};
    constexpr T c2 = T{1} / T{12};
    constexpr T c3 = T{1} / T{120};

    Matrix N_mat = I * c0 + A_scaled * c1;
    N_mat = N_mat + A2 * c2;
    N_mat = N_mat + A4 * (c3 / T{6});
    N_mat = N_mat + A6 * (c3 / T{42});

    Matrix D_mat = I * c0 - A_scaled * c1;
    D_mat = D_mat + A2 * c2;
    D_mat = D_mat - A4 * (c3 / T{6});
    D_mat = D_mat + A6 * (c3 / T{42});

    // Compute D^{-1} * N (more stable than N * D^{-1})
    auto            D_inv = D_mat.inverse();
    Matrix<N, N, T> exp_A_scaled;
    if (D_inv) {
        exp_A_scaled = D_inv.value() * N_mat;
    } else {
        // Fallback to Taylor series if Padé fails
        exp_A_scaled = I + A_scaled;
        Matrix A_power = A_scaled;
        for (size_t n = 2; n <= 15; ++n) {
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

// ============================================================================
// Matrix Logarithm: log(A)
// ============================================================================
// Computes the principal matrix logarithm using inverse scaling and squaring.
// The matrix logarithm X = log(A) satisfies exp(X) = A.
//
// Requirements: A must be invertible and have no real negative eigenvalues.

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

// ============================================================================
// Matrix Square Root: sqrt(A)
// ============================================================================
// Computes the principal matrix square root using Denman-Beavers iteration.
// The matrix square root S = sqrt(A) satisfies S*S = A.
//
// Requirements: A should have no real negative eigenvalues.

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

// ============================================================================
// Matrix Power: pow(A, p)
// ============================================================================
// Computes A^p for integer or real exponent p.
// For integer p: uses binary exponentiation
// For real p: uses pow(A, p) = exp(p * log(A))

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

// Real power (uses exp/log)
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

// ============================================================================
// Matrix Sine: sin(A)
// ============================================================================
// Computes the matrix sine using Taylor series:
//   sin(A) = A - A³/3! + A⁵/5! - A⁷/7! + ...

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

// ============================================================================
// Matrix Cosine: cos(A)
// ============================================================================
// Computes the matrix cosine using Taylor series:
//   cos(A) = I - A²/2! + A⁴/4! - A⁶/6! + ...

template<typename T, size_t N>
[[nodiscard]] constexpr Matrix<N, N, T> cos(const Matrix<N, N, T>& A) {
    Matrix<N, N, T> result = Matrix<N, N, T>::identity();
    Matrix<N, N, T> A2 = A * A;
    Matrix<N, N, T> A_power = Matrix<N, N, T>::identity();

    T factorial = T{1};
    T sign = T{-1};

    for (size_t n = 2; n <= 20; n += 2) {
        factorial *= static_cast<T>(n - 1) * static_cast<T>(n);
        A_power = A_power * A2;
        result = result + A_power * (sign / factorial);
        sign = -sign;
    }

    return result;
}

// ============================================================================
// Matrix Hyperbolic Sine: sinh(A)
// ============================================================================
// Computes the matrix hyperbolic sine:
//   sinh(A) = (exp(A) - exp(-A)) / 2

template<typename T, size_t N>
[[nodiscard]] constexpr Matrix<N, N, T> sinh(const Matrix<N, N, T>& A) {
    Matrix<N, N, T> exp_A = exp(A);
    Matrix<N, N, T> exp_neg_A = exp(A * T{-1});
    return (exp_A - exp_neg_A) * T{0.5};
}

// ============================================================================
// Matrix Hyperbolic Cosine: cosh(A)
// ============================================================================
// Computes the matrix hyperbolic cosine:
//   cosh(A) = (exp(A) + exp(-A)) / 2

template<typename T, size_t N>
[[nodiscard]] constexpr Matrix<N, N, T> cosh(const Matrix<N, N, T>& A) {
    Matrix<N, N, T> exp_A = exp(A);
    Matrix<N, N, T> exp_neg_A = exp(A * T{-1});
    return (exp_A + exp_neg_A) * T{0.5};
}

// ============================================================================
// Identity check helper (useful for verifying trig identities)
// ============================================================================
// Verifies sin²(A) + cos²(A) ≈ I within tolerance

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
