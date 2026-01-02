#pragma once

#include <complex>
#include <cstddef>

#include "constexpr_math.hpp"
#include "matrix.hpp"

// ============================================================================
// Eigenvalue Result Structure
// ============================================================================
template<size_t N, typename T = double>
struct DirectEigenResult {
    ColVec<N, std::complex<T>>    values;  // Eigenvalues
    Matrix<N, N, std::complex<T>> vectors; // Column eigenvectors
    bool                          converged{true};

    constexpr DirectEigenResult() : values(ColVec<N, std::complex<T>>::zeros()),
                                    vectors(Matrix<N, N, std::complex<T>>::identity()) {}
};

// ============================================================================
// 2x2 Eigenvalue Computation (Direct Quadratic Formula)
// ============================================================================
// For [a b; c d], eigenvalues are roots of λ² - (a+d)λ + (ad-bc) = 0
// λ = (trace ± sqrt(trace² - 4*det)) / 2

template<typename T>
constexpr DirectEigenResult<2, T> eigenvalues_2x2(const Matrix<2, 2, T>& A) {
    using Complex = std::complex<T>;
    DirectEigenResult<2, T> result;

    T a = A(0, 0), b = A(0, 1);
    T c = A(1, 0), d = A(1, 1);

    T trace = a + d;
    T det = a * d - b * c;
    T discriminant = trace * trace - T{4} * det;

    if (discriminant >= T{0}) {
        // Real eigenvalues
        T sqrt_disc = wet::sqrt(discriminant);
        result.values[0] = Complex((trace + sqrt_disc) / T{2}, T{0});
        result.values[1] = Complex((trace - sqrt_disc) / T{2}, T{0});
    } else {
        // Complex conjugate eigenvalues
        T sqrt_disc = wet::sqrt(-discriminant);
        result.values[0] = Complex(trace / T{2}, sqrt_disc / T{2});
        result.values[1] = Complex(trace / T{2}, -sqrt_disc / T{2});
    }

    // Compute eigenvectors
    for (size_t i = 0; i < 2; ++i) {
        Complex lambda = result.values[i];

        // Solve (A - λI)v = 0
        Complex a_l = Complex(a, T{0}) - lambda;
        Complex b_c = Complex(b, T{0});
        Complex c_c = Complex(c, T{0});
        Complex d_l = Complex(d, T{0}) - lambda;

        Complex v1, v2;

        // Choose the most numerically stable way to compute eigenvector
        T mag_al = std::norm(a_l);
        T mag_b = std::norm(b_c);
        T mag_c = std::norm(c_c);
        T mag_dl = std::norm(d_l);

        // Find which element we can safely divide by
        if (mag_b > T{1e-20} && mag_al > T{1e-20}) {
            // Row 0: (a-λ)*v1 + b*v2 = 0 => v1 = -b*v2/(a-λ)
            v2 = Complex{T{1}, T{0}};
            v1 = -b_c * v2 / a_l;
        } else if (mag_c > T{1e-20} && mag_dl > T{1e-20}) {
            // Row 1: c*v1 + (d-λ)*v2 = 0 => v2 = -c*v1/(d-λ)
            v1 = Complex{T{1}, T{0}};
            v2 = -c_c * v1 / d_l;
        } else if (mag_b > T{1e-20}) {
            // b is nonzero, a-λ ≈ 0, so from row 0: b*v2 ≈ 0 => v2 ≈ 0
            v1 = Complex{T{1}, T{0}};
            v2 = Complex{T{0}, T{0}};
        } else if (mag_c > T{1e-20}) {
            // c is nonzero, d-λ ≈ 0, so from row 1: c*v1 ≈ 0 => v1 ≈ 0
            v1 = Complex{T{0}, T{0}};
            v2 = Complex{T{1}, T{0}};
        } else {
            // Matrix is essentially diagonal or zero
            v1 = Complex{T{1}, T{0}};
            v2 = Complex{T{0}, T{0}};
        }

        // Normalize (with safe check for zero vector)
        T norm_sq = std::norm(v1) + std::norm(v2);
        if (norm_sq > T{1e-30}) {
            T norm = wet::sqrt(norm_sq);
            v1 /= norm;
            v2 /= norm;
        } else {
            // Fallback to unit vector
            v1 = Complex{T{1}, T{0}};
            v2 = Complex{T{0}, T{0}};
        }

        result.vectors(0, i) = v1;
        result.vectors(1, i) = v2;
    }

    result.converged = true;
    return result;
}

// ============================================================================
// 3x3 Eigenvalue Computation (Cardano's Formula)
// ============================================================================
// Characteristic polynomial: λ³ - tr(A)λ² + (sum of 2x2 minors)λ - det(A) = 0

template<typename T>
constexpr DirectEigenResult<3, T> eigenvalues_3x3(const Matrix<3, 3, T>& A) {
    using Complex = std::complex<T>;
    DirectEigenResult<3, T> result;
    constexpr T             pi = T{3.14159265358979323846};

    // Check if matrix is diagonal (optimization and accuracy)
    T off_diag_sum = T{0};
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            if (i != j) {
                off_diag_sum += wet::abs(A(i, j));
            }
        }
    }
    if (off_diag_sum < T{1e-14}) {
        // Diagonal matrix: eigenvalues are diagonal elements
        result.values[0] = Complex(A(0, 0), T{0});
        result.values[1] = Complex(A(1, 1), T{0});
        result.values[2] = Complex(A(2, 2), T{0});
        result.vectors = Matrix<3, 3, Complex>::identity();
        result.converged = true;
        return result;
    }

    // Characteristic polynomial coefficients: λ³ - c₂λ² + c₁λ - c₀ = 0
    T c2 = A(0, 0) + A(1, 1) + A(2, 2); // trace

    // c1 = sum of 2x2 principal minors
    T m00 = A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1);
    T m11 = A(0, 0) * A(2, 2) - A(0, 2) * A(2, 0);
    T m22 = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
    T c1 = m00 + m11 + m22;

    // c0 = determinant
    T c0 = A(0, 0) * (A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1))
         - A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0))
         + A(0, 2) * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0));

    // Substitute λ = t + c2/3 to get depressed cubic: t³ + pt + q = 0
    // where p = c1 - c2²/3 and q = 2c2³/27 - c2·c1/3 + c0
    T c2_3 = c2 / T{3};
    T c2_sq = c2 * c2;
    T p = c1 - c2_sq / T{3};
    T q = (T{2} * c2_sq * c2 / T{27}) - (c2 * c1 / T{3}) + c0;

    // Cardano's discriminant: Δ = (q/2)² + (p/3)³
    T p_div_3 = p / T{3};
    T q_div_2 = q / T{2};
    T disc = q_div_2 * q_div_2 + p_div_3 * p_div_3 * p_div_3;

    if (disc > T{1e-15}) {
        // One real root, two complex conjugate roots
        T sqrt_disc = wet::sqrt(disc);
        T u = wet::cbrt(-q_div_2 + sqrt_disc);
        T v = wet::cbrt(-q_div_2 - sqrt_disc);

        T t1 = u + v;
        result.values[0] = Complex(t1 + c2_3, T{0});

        // Complex roots: t = -(u+v)/2 ± i·√3·(u-v)/2
        T real_part = -(u + v) / T{2} + c2_3;
        T imag_part = wet::sqrt(T{3}) * (u - v) / T{2};
        result.values[1] = Complex(real_part, imag_part);
        result.values[2] = Complex(real_part, -imag_part);
    } else if (disc < -T{1e-15}) {
        // Three distinct real roots (trigonometric solution)
        // t_k = 2·√(-p/3)·cos(φ/3 + 2πk/3) for k = 0, 1, 2
        // where cos(φ) = (3q)/(2p)·√(-3/p)
        T m = T{2} * wet::sqrt(-p_div_3);
        T theta = wet::atan2(wet::sqrt(-disc), -q_div_2) / T{3};

        result.values[0] = Complex(m * wet::cos(theta) + c2_3, T{0});
        result.values[1] = Complex(m * wet::cos(theta + T{2} * pi / T{3}) + c2_3, T{0});
        result.values[2] = Complex(m * wet::cos(theta + T{4} * pi / T{3}) + c2_3, T{0});
    } else {
        // Repeated roots (discriminant ≈ 0)
        T u = wet::cbrt(-q_div_2);
        result.values[0] = Complex(T{2} * u + c2_3, T{0});
        result.values[1] = Complex(-u + c2_3, T{0});
        result.values[2] = Complex(-u + c2_3, T{0});
    }

    // Compute eigenvectors for each eigenvalue
    for (size_t i = 0; i < 3; ++i) {
        Complex lambda = result.values[i];

        // Form (A - λI)
        Matrix<3, 3, Complex> M = Matrix<3, 3, Complex>::zeros();
        for (size_t r = 0; r < 3; ++r) {
            for (size_t c = 0; c < 3; ++c) {
                M(r, c) = Complex(A(r, c), T{0});
            }
            M(r, r) -= lambda;
        }

        // Find eigenvector in null space using cross product of two rows
        ColVec<3, Complex> row0, row1, row2;
        for (size_t j = 0; j < 3; ++j) {
            row0[j] = M(0, j);
            row1[j] = M(1, j);
            row2[j] = M(2, j);
        }

        // Try cross products of different row pairs, pick the one with largest norm
        ColVec<3, Complex> v;
        T                  best_norm = T{0};

        // Cross product of rows 0 and 1
        ColVec<3, Complex> cross01;
        cross01[0] = row0[1] * row1[2] - row0[2] * row1[1];
        cross01[1] = row0[2] * row1[0] - row0[0] * row1[2];
        cross01[2] = row0[0] * row1[1] - row0[1] * row1[0];
        T norm01 = std::norm(cross01[0]) + std::norm(cross01[1]) + std::norm(cross01[2]);
        if (norm01 > best_norm) {
            best_norm = norm01;
            v = cross01;
        }

        // Cross product of rows 0 and 2
        ColVec<3, Complex> cross02;
        cross02[0] = row0[1] * row2[2] - row0[2] * row2[1];
        cross02[1] = row0[2] * row2[0] - row0[0] * row2[2];
        cross02[2] = row0[0] * row2[1] - row0[1] * row2[0];
        T norm02 = std::norm(cross02[0]) + std::norm(cross02[1]) + std::norm(cross02[2]);
        if (norm02 > best_norm) {
            best_norm = norm02;
            v = cross02;
        }

        // Cross product of rows 1 and 2
        ColVec<3, Complex> cross12;
        cross12[0] = row1[1] * row2[2] - row1[2] * row2[1];
        cross12[1] = row1[2] * row2[0] - row1[0] * row2[2];
        cross12[2] = row1[0] * row2[1] - row1[1] * row2[0];
        T norm12 = std::norm(cross12[0]) + std::norm(cross12[1]) + std::norm(cross12[2]);
        if (norm12 > best_norm) {
            best_norm = norm12;
            v = cross12;
        }

        // Normalize
        if (best_norm > T{1e-20}) {
            T norm = wet::sqrt(best_norm);
            for (size_t j = 0; j < 3; ++j) {
                v[j] /= norm;
            }
        } else {
            // Fallback: use standard basis
            v = ColVec<3, Complex>::zeros();
            v[i] = Complex{T{1}, T{0}};
        }

        for (size_t j = 0; j < 3; ++j) {
            result.vectors(j, i) = v[j];
        }
    }

    result.converged = true;
    return result;
}

// ============================================================================
// 4x4 Eigenvalue Computation (Ferrari's Method)
// ============================================================================

template<typename T>
constexpr DirectEigenResult<4, T> eigenvalues_4x4(const Matrix<4, 4, T>& A) {
    using Complex = std::complex<T>;
    DirectEigenResult<4, T> result;

    // Compute characteristic polynomial coefficients
    // p(λ) = λ⁴ - c₃λ³ + c₂λ² - c₁λ + c₀
    T c3 = A.trace();

    // c2 = sum of 2x2 principal minors
    T c2 = T{0};
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = i + 1; j < 4; ++j) {
            c2 += A(i, i) * A(j, j) - A(i, j) * A(j, i);
        }
    }

    // c1 = sum of 3x3 principal minors
    T c1 = T{0};
    for (size_t skip = 0; skip < 4; ++skip) {
        T      m[3][3];
        size_t ri = 0;
        for (size_t i = 0; i < 4; ++i) {
            if (i == skip)
                continue;
            size_t ci = 0;
            for (size_t j = 0; j < 4; ++j) {
                if (j == skip)
                    continue;
                m[ri][ci] = A(i, j);
                ++ci;
            }
            ++ri;
        }
        T det3 = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
               - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
               + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        c1 += det3;
    }

    // c0 = determinant of A
    T c0 = T{0};
    for (size_t j = 0; j < 4; ++j) {
        T m[3][3];
        for (size_t ri = 0; ri < 3; ++ri) {
            size_t ci = 0;
            for (size_t k = 0; k < 4; ++k) {
                if (k == j)
                    continue;
                m[ri][ci] = A(ri + 1, k);
                ++ci;
            }
        }
        T det3 = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
               - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
               + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        T sign = (j % 2 == 0) ? T{1} : T{-1};
        c0 += sign * A(0, j) * det3;
    }

    // Substitute λ = t + c₃/4 to get depressed quartic: t⁴ + pt² + qt + r = 0
    T c3_4 = c3 / T{4};
    T c3_sq = c3 * c3;
    T p = c2 - T{3} * c3_sq / T{8};
    T q = c3_sq * c3 / T{8} - c3 * c2 / T{2} + c1;
    T r = -T{3} * c3_sq * c3_sq / T{256} + c3_sq * c2 / T{16} - c3 * c1 / T{4} + c0;

    // Solve resolvent cubic: y³ + (p/2)y² + ((p²-4r)/16)y - q²/64 = 0
    T a2_res = p / T{2};
    T a1_res = (p * p - T{4} * r) / T{16};
    T a0_res = -q * q / T{64};

    // Depress resolvent: y = s - a2_res/3
    T shift = a2_res / T{3};
    T p_res = a1_res - a2_res * shift;
    T q_res = a0_res - a1_res * shift + T{2} * shift * shift * shift;

    T disc_res = q_res * q_res / T{4} + p_res * p_res * p_res / T{27};
    T y_real;

    if (disc_res >= T{0}) {
        T sqrt_disc = wet::sqrt(disc_res);
        T u = wet::cbrt(-q_res / T{2} + sqrt_disc);
        T v = wet::cbrt(-q_res / T{2} - sqrt_disc);
        y_real = u + v - shift;
    } else {
        T r_mag = wet::sqrt(-p_res * p_res * p_res / T{27});
        T phi = wet::atan2(wet::sqrt(-disc_res), -q_res / T{2});
        y_real = T{2} * wet::cbrt(r_mag) * wet::cos(phi / T{3}) - shift;
    }

    // Factor quartic using y_real: solve two quadratics
    T two_y = T{2} * y_real;
    T sqrt_2y = (two_y > T{0}) ? wet::sqrt(two_y) : T{0};

    T half_y_plus_p2 = y_real + p / T{2};
    T q_term = (sqrt_2y > T{1e-15}) ? q / (T{2} * sqrt_2y) : T{0};

    // Quadratic 1: t² + sqrt_2y*t + (half_y_plus_p2 + q_term) = 0
    T b1 = sqrt_2y;
    T c1_q = half_y_plus_p2 + q_term;
    T disc1 = b1 * b1 - T{4} * c1_q;

    if (disc1 >= T{0}) {
        T sd1 = wet::sqrt(disc1);
        result.values[0] = Complex((-b1 + sd1) / T{2} + c3_4, T{0});
        result.values[1] = Complex((-b1 - sd1) / T{2} + c3_4, T{0});
    } else {
        T sd1 = wet::sqrt(-disc1);
        result.values[0] = Complex(-b1 / T{2} + c3_4, sd1 / T{2});
        result.values[1] = Complex(-b1 / T{2} + c3_4, -sd1 / T{2});
    }

    // Quadratic 2: t² - sqrt_2y*t + (half_y_plus_p2 - q_term) = 0
    T b2 = -sqrt_2y;
    T c2_q = half_y_plus_p2 - q_term;
    T disc2 = b2 * b2 - T{4} * c2_q;

    if (disc2 >= T{0}) {
        T sd2 = wet::sqrt(disc2);
        result.values[2] = Complex((-b2 + sd2) / T{2} + c3_4, T{0});
        result.values[3] = Complex((-b2 - sd2) / T{2} + c3_4, T{0});
    } else {
        T sd2 = wet::sqrt(-disc2);
        result.values[2] = Complex(-b2 / T{2} + c3_4, sd2 / T{2});
        result.values[3] = Complex(-b2 / T{2} + c3_4, -sd2 / T{2});
    }

    // Compute eigenvectors via cofactor method
    for (size_t i = 0; i < 4; ++i) {
        Complex lambda = result.values[i];

        // Form (A - λI)
        Matrix<4, 4, Complex> M = Matrix<4, 4, Complex>::zeros();
        for (size_t rr = 0; rr < 4; ++rr) {
            for (size_t cc = 0; cc < 4; ++cc) {
                M(rr, cc) = Complex(A(rr, cc), T{0});
            }
            M(rr, rr) -= lambda;
        }

        // Find column with largest 3x3 cofactor
        size_t best_col = 0;
        T      best_cof_mag = T{0};

        for (size_t col = 0; col < 4; ++col) {
            Complex m3[3][3];
            for (size_t ri = 0; ri < 3; ++ri) {
                size_t ci = 0;
                for (size_t k = 0; k < 4; ++k) {
                    if (k == col)
                        continue;
                    m3[ri][ci] = M(ri + 1, k);
                    ++ci;
                }
            }
            Complex det3 = m3[0][0] * (m3[1][1] * m3[2][2] - m3[1][2] * m3[2][1])
                         - m3[0][1] * (m3[1][0] * m3[2][2] - m3[1][2] * m3[2][0])
                         + m3[0][2] * (m3[1][0] * m3[2][1] - m3[1][1] * m3[2][0]);
            T mag = std::norm(det3);
            if (mag > best_cof_mag) {
                best_cof_mag = mag;
                best_col = col;
            }
        }

        // Build eigenvector from cofactors
        ColVec<4, Complex> v;
        for (size_t row = 0; row < 4; ++row) {
            Complex m3[3][3];
            size_t  ri = 0;
            for (size_t rr = 0; rr < 4; ++rr) {
                if (rr == row)
                    continue;
                size_t ci = 0;
                for (size_t cc = 0; cc < 4; ++cc) {
                    if (cc == best_col)
                        continue;
                    m3[ri][ci] = M(rr, cc);
                    ++ci;
                }
                ++ri;
            }
            Complex det3 = m3[0][0] * (m3[1][1] * m3[2][2] - m3[1][2] * m3[2][1])
                         - m3[0][1] * (m3[1][0] * m3[2][2] - m3[1][2] * m3[2][0])
                         + m3[0][2] * (m3[1][0] * m3[2][1] - m3[1][1] * m3[2][0]);
            T sign = ((row + best_col) % 2 == 0) ? T{1} : T{-1};
            v[row] = sign * det3;
        }

        // Normalize
        T norm_sq = T{0};
        for (size_t j = 0; j < 4; ++j) {
            norm_sq += std::norm(v[j]);
        }
        if (norm_sq > T{1e-20}) {
            T norm = wet::sqrt(norm_sq);
            for (size_t j = 0; j < 4; ++j) {
                v[j] /= norm;
            }
        } else {
            v = ColVec<4, Complex>::zeros();
            v[i] = Complex{T{1}, T{0}};
        }

        for (size_t j = 0; j < 4; ++j) {
            result.vectors(j, i) = v[j];
        }
    }

    result.converged = true;
    return result;
}

// ============================================================================
// General Eigenvalue Computation (dispatch by size)
// ============================================================================
template<size_t N, typename T = double>
constexpr DirectEigenResult<N, T> compute_eigenvalues(const Matrix<N, N, T>& A) {
    if constexpr (N == 1) {
        DirectEigenResult<1, T> result;
        result.values[0] = std::complex<T>(A(0, 0), T{0});
        result.vectors(0, 0) = std::complex<T>{T{1}, T{0}};
        result.converged = true;
        return result;
    } else if constexpr (N == 2) {
        return eigenvalues_2x2(A);
    } else if constexpr (N == 3) {
        return eigenvalues_3x3(A);
    } else if constexpr (N == 4) {
        return eigenvalues_4x4(A);
    } else {
        // For N > 4, return unconverged result
        DirectEigenResult<N, T> result;
        result.converged = false;
        return result;
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

// Extract real parts from complex eigenvalues
template<size_t N, typename T = double>
constexpr ColVec<N, T> real_parts(const ColVec<N, std::complex<T>>& v) {
    ColVec<N, T> result = ColVec<N, T>::zeros();
    for (size_t i = 0; i < N; ++i) {
        result[i] = v[i].real();
    }
    return result;
}

// Extract imaginary parts from complex eigenvalues
template<size_t N, typename T = double>
constexpr ColVec<N, T> imag_parts(const ColVec<N, std::complex<T>>& v) {
    ColVec<N, T> result = ColVec<N, T>::zeros();
    for (size_t i = 0; i < N; ++i) {
        result[i] = v[i].imag();
    }
    return result;
}

// Compute closed-loop poles: real parts of eigenvalues of (A - B*K)
template<size_t NX, size_t NU, typename T = double>
constexpr ColVec<NX, T> compute_closed_loop_poles(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NU, NX, T>& K
) {
    static_assert(NX <= 4, "Closed-loop pole computation only supported for systems up to 4 states");

    // Form closed-loop system matrix: A_cl = A - B*K
    Matrix<NX, NX, T> A_cl = A - B * K;

    // Compute eigenvalues using direct formulas
    auto eigen = compute_eigenvalues(A_cl);

    if (!eigen.converged) {
        return ColVec<NX, T>::zeros();
    }

    return real_parts(eigen.values);
}

// ============================================================================
// Matrix Determinant (direct formulas for small matrices)
// ============================================================================
template<size_t N, typename T>
constexpr T determinant(const Matrix<N, N, T>& A) {
    if constexpr (N == 1) {
        return A(0, 0);
    } else if constexpr (N == 2) {
        return A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
    } else if constexpr (N == 3) {
        return A(0, 0) * (A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1))
             - A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0))
             + A(0, 2) * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0));
    } else if constexpr (N == 4) {
        T det = T{0};
        for (size_t j = 0; j < 4; ++j) {
            T m[3][3];
            for (size_t ri = 0; ri < 3; ++ri) {
                size_t ci = 0;
                for (size_t k = 0; k < 4; ++k) {
                    if (k == j)
                        continue;
                    m[ri][ci] = A(ri + 1, k);
                    ++ci;
                }
            }
            T det3 = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                   - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                   + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
            T sign = (j % 2 == 0) ? T{1} : T{-1};
            det += sign * A(0, j) * det3;
        }
        return det;
    } else {
        // LU decomposition for larger matrices
        Matrix<N, N, T> U = A;
        T               det = T{1};

        for (size_t i = 0; i < N; ++i) {
            // Find pivot
            size_t pivot = i;
            T      max_val = wet::abs(U(i, i));
            for (size_t k = i + 1; k < N; ++k) {
                T val = wet::abs(U(k, i));
                if (val > max_val) {
                    max_val = val;
                    pivot = k;
                }
            }

            if (max_val < T{1e-15}) {
                return T{0}; // Singular
            }

            if (pivot != i) {
                for (size_t j = 0; j < N; ++j) {
                    T tmp = U(i, j);
                    U(i, j) = U(pivot, j);
                    U(pivot, j) = tmp;
                }
                det = -det;
            }

            det *= U(i, i);

            for (size_t k = i + 1; k < N; ++k) {
                T factor = U(k, i) / U(i, i);
                for (size_t j = i; j < N; ++j) {
                    U(k, j) -= factor * U(i, j);
                }
            }
        }

        return det;
    }
}

// ============================================================================
// Legacy compatibility aliases
// ============================================================================
template<size_t N, typename T = double>
using ComplexEigenResult = DirectEigenResult<N, T>;

template<size_t N, typename T = double>
constexpr DirectEigenResult<N, T> compute_eigen(const Matrix<N, N, T>& A, int = 1000, T = T{1e-10}) {
    return compute_eigenvalues(A);
}

template<size_t N, typename T = double>
constexpr ColVec<N, std::complex<T>> get_eigenvalues(const DirectEigenResult<N, T>& result) {
    return result.values;
}

template<size_t N, typename T = double>
constexpr ColVec<N, T> get_real_parts(const ColVec<N, std::complex<T>>& v) {
    return real_parts(v);
}
