#pragma once

#include "matrix.hpp"
#include "wet/backend.hpp" // wet::optional, wet::pair, wet::nullopt, wet::swap

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
 * @brief Spectral norm (2-norm): largest singular value of A
 *
 * ||A||_2 = sigma_max(A) = sqrt( lambda_max(A^H A) )
 *
 * Computed via power iteration on A^H A, which converges geometrically
 * at a rate proportional to sigma_1 / sigma_2. Reliable for the small,
 * fixed-size matrices typical in control systems (state dimensions ≤~20).
 *
 * @tparam T Element type
 * @tparam N Matrix dimension
 * @param A Square matrix
 * @return Spectral norm
 */
template<typename T, size_t N>
[[nodiscard]] constexpr T two_norm(const Matrix<N, N, T>& A) {
    constexpr T      tol = default_tol<T>();
    constexpr size_t max_iter = 100;

    Matrix<N, N, T> AtA = A.t() * A;

    // Initial vector: all ones, then normalize
    ColVec<N, T> v;
    for (size_t i = 0; i < N; ++i) {
        v[i] = T{1};
    }

    {
        auto n = v.norm();
        if (n > tol) {
            v = v * (T{1} / n);
        }
    }

    T lambda = T{0};
    for (size_t iter = 0; iter < max_iter; ++iter) {
        ColVec<N, T> w = AtA * v;
        T            new_lambda = w.norm();
        if (new_lambda < tol)
            return T{0};

        v = w * (T{1} / new_lambda);

        if (wet::abs(new_lambda - lambda) < tol * wet::abs(new_lambda)) {
            return wet::sqrt(new_lambda);
        }
        lambda = new_lambda;
    }
    return wet::sqrt(lambda);
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
        T d = A(0, 0) * det(Matrix<3, 3, T>{
                  {A(1, 1), A(1, 2), A(1, 3)},
                  {A(2, 1), A(2, 2), A(2, 3)},
                  {A(3, 1), A(3, 2), A(3, 3)},
              });

        d -= A(0, 1) * det(Matrix<3, 3, T>{
                 {A(1, 0), A(1, 2), A(1, 3)},
                 {A(2, 0), A(2, 2), A(2, 3)},
                 {A(3, 0), A(3, 2), A(3, 3)},
             });

        d += A(0, 2) * det(Matrix<3, 3, T>{
                 {A(1, 0), A(1, 1), A(1, 3)},
                 {A(2, 0), A(2, 1), A(2, 3)},
                 {A(3, 0), A(3, 1), A(3, 3)},
             });

        d -= A(0, 3) * det(Matrix<3, 3, T>{
                 {A(1, 0), A(1, 1), A(1, 2)},
                 {A(2, 0), A(2, 1), A(2, 2)},
                 {A(3, 0), A(3, 1), A(3, 2)},
             });

        return d;
    } else {
        // General case: det(A) = det(P) * det(L) * det(U) = sign * product(U_ii)
        auto lu = lu_decomposition(A);
        if (!lu) {
            return T{0}; // singular
        }

        const auto& [L, U, piv] = lu.value();

        // Compute sign from permutation parity
        auto   p = piv;
        size_t swaps = 0;
        for (size_t i = 0; i < N; ++i) {
            while (p[i] != i) {
                wet::swap(p[i], p[p[i]]);
                ++swaps;
            }
        }

        T d = (swaps % 2 == 0) ? T{1} : T{-1};
        for (size_t i = 0; i < N; ++i) {
            d *= U(i, i);
        }
        return d;
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
    constexpr T     epsilon = default_tol<T>();

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
                wet::swap(temp(rank, j), temp(pivot_row, j));
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

    // Tiny/nilpotent matrix shortcut (Taylor series)
    if (norm <= default_tol<T>()) {
        Matrix A2 = A * A;
        Matrix A3 = A2 * A;
        Matrix A4 = A3 * A;
        Matrix A5 = A4 * A;
        Matrix A6 = A5 * A;
        return I + A
             + A2 * (T{1} / T{2})
             + A3 * (T{1} / T{6})
             + A4 * (T{1} / T{24})
             + A5 * (T{1} / T{120})
             + A6 * (T{1} / T{720});
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

            if (diff < default_tol<T>())
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
 * @brief Matrix square root via Denman–Beavers iteration
 *
 * Computes the principal matrix square root S = √A satisfying S·S = A.
 *
 * Uses Denman–Beavers iteration:
 *     Yₖ₊₁ = (Yₖ + Zₖ⁻¹) / 2
 *     Zₖ₊₁ = (Zₖ + Yₖ⁻¹) / 2
 *
 * Converges: Y → √A, Z → (√A)⁻¹.
 *
 * Returns wet::nullopt if the iteration fails to converge or encounters
 * a singular iterate (e.g., A has a real negative eigenvalue).
 *
 * @note Compare with MATLAB's sqrtm(A).
 * @see Higham, "Functions of Matrices" (2008), §6.3
 *
 * @tparam T Element type
 * @tparam N Matrix dimension
 * @param A Square matrix (must have no real negative eigenvalues)
 * @return Principal matrix square root √A, or wet::nullopt on failure
 */
template<typename T, size_t N>
[[nodiscard]] constexpr wet::optional<Matrix<N, N, T>> sqrt(const Matrix<N, N, T>& A) {
    Matrix<N, N, T> Y = A;
    Matrix<N, N, T> Z = Matrix<N, N, T>::identity();

    for (int iter = 0; iter < 50; ++iter) {
        auto Y_inv = Y.inverse();
        auto Z_inv = Z.inverse();

        if (!Y_inv || !Z_inv) {
            return wet::nullopt;
        }

        Matrix<N, N, T> Y_next = (Y + Z_inv.value()) * T{0.5};
        Matrix<N, N, T> Z_next = (Z + Y_inv.value()) * T{0.5};

        T diff = T{0};
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                diff += wet::abs(Y_next(i, j) - Y(i, j));
            }
        }

        Y = Y_next;
        Z = Z_next;

        if (diff < default_tol<T>()) {
            return Y;
        }
    }

    return wet::nullopt;
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
    // Check for integer case (constexpr-safe, no std::modf)
    int p_int = static_cast<int>(p);
    T   p_frac = p - static_cast<T>(p_int);
    if (wet::abs(p_frac) < default_tol<T>()) {
        return pow(A, p_int);
    }

    // General case: A^p = exp(p * log(A))
    return expm(log(A) * p);
}

/**
 * @brief Compute sin(A) and cos(A) together via scaling and double-angle reconstruction
 *
 * More efficient than calling sin() and cos() separately — computes both
 * with one scaling pass and shared Taylor series evaluation.
 *
 * Scales A down so ||A/2ˢ|| < 0.5 (where the Taylor series converges
 * accurately), then recovers via repeated double-angle formulas:
 *
 *     sin(2A) = 2·sin(A)·cos(A)
 *     cos(2A) = 2·cos²(A) − I
 *
 * Since sin(A) and cos(A) are polynomials in A, they commute with each other,
 * so the scalar double-angle formulas apply directly.
 *
 * @see Higham, "Functions of Matrices" (2008), §12.3
 *
 * @param A Square matrix
 * @return {sin(A), cos(A)}
 */
template<typename T, size_t N>
[[nodiscard]] constexpr wet::pair<Matrix<N, N, T>, Matrix<N, N, T>> sincos(const Matrix<N, N, T>& A) {
    Matrix<N, N, T> I = Matrix<N, N, T>::identity();

    // Scale down until ||A/2^s|| < 0.5
    T      norm = infinity_norm(A);
    size_t s = 0;
    T      scaled_norm = norm;
    while (scaled_norm > T{0.5}) {
        scaled_norm *= T{0.5};
        ++s;
    }

    T scale = T{1};
    for (size_t i = 0; i < s; ++i) {
        scale *= T{0.5};
    }

    Matrix<N, N, T> As = A * scale;
    Matrix<N, N, T> As2 = As * As;

    // Taylor series for sin(As) = As - As³/3! + As⁵/5! - ...
    Matrix<N, N, T> sinA = As;
    {
        Matrix<N, N, T> A_power = As;
        T               factorial = T{1};
        T               sign = T{-1};
        for (size_t n = 3; n <= 21; n += 2) {
            factorial *= static_cast<T>(n - 1) * static_cast<T>(n);
            A_power = A_power * As2;
            sinA = sinA + A_power * (sign / factorial);
            sign = -sign;
        }
    }

    // Taylor series for cos(As) = I - As²/2! + As⁴/4! - ...
    Matrix<N, N, T> cosA = I;
    {
        Matrix<N, N, T> A_power = I;
        T               factorial = T{1};
        T               sign = T{-1};
        for (size_t n = 2; n <= 20; n += 2) {
            factorial *= static_cast<T>(n - 1) * static_cast<T>(n);
            A_power = A_power * As2;
            cosA = cosA + A_power * (sign / factorial);
            sign = -sign;
        }
    }

    // Double-angle reconstruction: s iterations
    for (size_t i = 0; i < s; ++i) {
        Matrix<N, N, T> new_sin = sinA * cosA * T{2};
        Matrix<N, N, T> new_cos = cosA * cosA * T{2} - I;
        sinA = new_sin;
        cosA = new_cos;
    }

    return {sinA, cosA};
}

/**
 * @brief Matrix sine via scaling and double-angle reconstruction
 *
 * @note Compare with MATLAB's funm(A, @sin).
 * @see sincos() to compute both sin(A) and cos(A) in one call
 * @see Higham, "Functions of Matrices" (2008), §12.3
 *
 * @param A Square matrix
 * @return Matrix sine sin(A)
 */
template<typename T, size_t N>
[[nodiscard]] constexpr Matrix<N, N, T> sin(const Matrix<N, N, T>& A) {
    return sincos(A).first;
}

/**
 * @brief Matrix cosine via scaling and double-angle reconstruction
 *
 * @note Compare with MATLAB's funm(A, @cos).
 * @see sincos() to compute both sin(A) and cos(A) in one call
 * @see Higham, "Functions of Matrices" (2008), §12.3
 *
 * @param A Square matrix
 * @return Matrix cosine cos(A)
 */
template<typename T, size_t N>
[[nodiscard]] constexpr Matrix<N, N, T> cos(const Matrix<N, N, T>& A) {
    return sincos(A).second;
}

/**
 * @brief Matrix hyperbolic sine
 *
 * Computes sinh(A) = (exp(A) − exp(−A)) / 2
 *
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
 * Computes cosh(A) = (exp(A) + exp(−A)) / 2
 *
 * @param A Square matrix
 * @return Matrix hyperbolic cosine cosh(A)
 */
template<typename T, size_t N>
[[nodiscard]] constexpr Matrix<N, N, T> cosh(const Matrix<N, N, T>& A) {
    Matrix<N, N, T> exp_A = expm(A);
    Matrix<N, N, T> exp_neg_A = expm(A * T{-1});
    return (exp_A + exp_neg_A) * T{0.5};
}
} // namespace mat
} // namespace wetmelon::control