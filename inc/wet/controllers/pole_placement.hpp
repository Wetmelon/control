#pragma once

/**
 * @defgroup pole_placement Robust pole placement (place)
 * @brief Multi-input eigenvalue assignment with eigenvector-conditioning
 *        minimization (the numerical routine MATLAB's `place` implements).
 *
 * For a controllable pair (A, B) and a desired closed-loop spectrum, find a
 * state-feedback gain K such that the eigenvalues of (A − B·K) equal the desired
 * poles. A multi-input system has extra freedom beyond merely assigning the
 * spectrum; the Kautsky–Nichols–Van Dooren (KNV) method spends that freedom to
 * make the eigenvector matrix X as well-conditioned as possible, which minimizes
 * the sensitivity of the placed poles to perturbations in A, B, and K — the
 * robustness that distinguishes `place` from the single-input Ackermann formula
 * (`matlab::acker`).
 *
 * Method 0 (orthogonal) is implemented: each eigenvector is repeatedly re-chosen
 * within its admissible subspace to be maximally orthogonal to the span of the
 * others, driving κ(X) down. The spectrum is assigned *exactly* regardless of
 * how far the conditioning sweep runs (placement correctness depends only on X
 * being invertible); the sweep only improves robustness.
 *
 * @see J. Kautsky, N. K. Nichols, P. Van Dooren, "Robust pole assignment in
 *      linear state feedback," Int. J. Control 41(5), 1985,
 *      https://doi.org/10.1080/00207178508933420
 * @see matlab::acker for the single-input Ackermann path
 */

#include <cstddef>
#include <optional>

#include "wet/backend.hpp"
#include "wet/math/math.hpp"
#include "wet/matrix/matrix.hpp"

namespace wet {

namespace design {

/**
 * @brief Robust multi-input pole placement (Kautsky–Nichols–Van Dooren, real poles).
 *
 * Computes the state-feedback gain K placing the eigenvalues of (A − B·K) at the
 * requested real poles, using KNV Method 0 to minimize the conditioning of the
 * eigenvector basis.
 *
 * @tparam NX Number of states
 * @tparam NU Number of inputs (NU ≤ NX)
 * @param A      State matrix (NX×NX)
 * @param B      Input matrix (NX×NU), assumed full column rank (controllable)
 * @param poles  Desired closed-loop eigenvalues (real)
 * @return Gain K (NU×NX), or std::nullopt if B is rank-deficient or the assigned
 *         eigenvectors are linearly dependent (e.g. a pole repeated more than NU
 *         times — not assignable with independent eigenvectors).
 *
 * @note Compare with MATLAB's K = place(A, B, poles).
 */
template<size_t NX, size_t NU, typename T = double>
[[nodiscard]] constexpr std::optional<Matrix<NU, NX, T>> place(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const wet::array<T, NX>& poles
) {
    static_assert(NU <= NX, "place requires NU <= NX (no more inputs than states)");

    // Build the closed-loop matrix A − B·K = X·Λ·X⁻¹ from a chosen eigenvector
    // basis X, then recover K. The eigenvector matrix is assembled below.
    Matrix<NX, NX, T> X;

    if constexpr (NU == NX) {
        // Square, full-rank B: any spectrum is assignable with X = I. Each
        // standard basis vector is a valid eigenvector, so A − B·K = diag(poles).
        X = Matrix<NX, NX, T>::identity();
    } else {
        // QR of B: U0 = range(B), U1 = its orthogonal complement (U1ᵀ·B = 0).
        const auto qrB = mat::full_qr(B);

        // U1 = last (NX − NU) columns of Q.
        constexpr size_t  NC = NX - NU; // complement dimension
        Matrix<NX, NC, T> U1;
        for (size_t i = 0; i < NX; ++i) {
            for (size_t c = 0; c < NC; ++c) {
                U1(i, c) = qrB.Q(i, NU + c);
            }
        }

        // For each eigenvalue λ_j, the admissible eigenvectors are the null space
        // of U1ᵀ(A − λ_j I): vectors x with (A − λ_j I)x ∈ range(B). A basis S_j
        // (NX×NU) is the trailing NU columns of the full QR of (A − λ_j I)ᵀ·U1.
        wet::array<Matrix<NX, NU, T>, NX> S{};
        for (size_t j = 0; j < NX; ++j) {
            // Mtʲ = (A − λ_j I)ᵀ · U1   (NX×NC), whose left null space (the
            // trailing NU columns of its full-Q) is the admissible space S_j.
            Matrix<NX, NC, T> Mt;
            for (size_t r = 0; r < NX; ++r) {
                for (size_t c = 0; c < NC; ++c) {
                    T acc = T{0};
                    for (size_t k = 0; k < NX; ++k) {
                        const T aki = (A(k, r) - (k == r ? poles[j] : T{0})); // (A − λI)ᵀ at (r,k)
                        acc += aki * U1(k, c);
                    }
                    Mt(r, c) = acc;
                }
            }
            const auto qrM = mat::full_qr(Mt);
            for (size_t i = 0; i < NX; ++i) {
                for (size_t s = 0; s < NU; ++s) {
                    S[j](i, s) = qrM.Q(i, NC + s);
                }
            }
        }

        // Initialize each eigenvector as the first admissible basis vector.
        for (size_t j = 0; j < NX; ++j) {
            for (size_t i = 0; i < NX; ++i) {
                X(i, j) = S[j](i, 0);
            }
        }

        // KNV Method 0: sweep, re-choosing each x_j ∈ S_j to be maximally
        // orthogonal to the span of the other columns. q⊥ = the 1-D orthogonal
        // complement of the others (trailing column of their full-Q); the new
        // x_j is q⊥ projected onto S_j and normalized.
        constexpr size_t max_sweeps = 40;
        constexpr T      tol = T{1e-14};
        for (size_t sweep = 0; sweep < max_sweeps; ++sweep) {
            T max_change = T{0};
            for (size_t j = 0; j < NX; ++j) {
                // Others = X with column j removed (NX × (NX−1)).
                Matrix<NX, NX - 1, T> others;
                for (size_t i = 0; i < NX; ++i) {
                    size_t cc = 0;
                    for (size_t c = 0; c < NX; ++c) {
                        if (c == j) {
                            continue;
                        }
                        others(i, cc) = X(i, c);
                        ++cc;
                    }
                }
                const auto qrO = mat::full_qr(others);
                // q⊥ = trailing column of Q (orthogonal to span(others)).
                wet::array<T, NX> qperp{};
                for (size_t i = 0; i < NX; ++i) {
                    qperp[i] = qrO.Q(i, NX - 1);
                }
                // Project q⊥ onto S_j: x_new = S_j (S_jᵀ q⊥).
                wet::array<T, NU> coeffs{};
                for (size_t s = 0; s < NU; ++s) {
                    T acc = T{0};
                    for (size_t i = 0; i < NX; ++i) {
                        acc += S[j](i, s) * qperp[i];
                    }
                    coeffs[s] = acc;
                }
                wet::array<T, NX> xnew{};
                T                 nrm_sq = T{0};
                for (size_t i = 0; i < NX; ++i) {
                    T acc = T{0};
                    for (size_t s = 0; s < NU; ++s) {
                        acc += S[j](i, s) * coeffs[s];
                    }
                    xnew[i] = acc;
                    nrm_sq += acc * acc;
                }
                const T nrm = wet::sqrt(nrm_sq);
                if (nrm < tol) {
                    continue; // degenerate; keep the current vector
                }
                // Track change (sign-agnostic) and commit the normalized vector.
                T diff_pos = T{0};
                T diff_neg = T{0};
                for (size_t i = 0; i < NX; ++i) {
                    const T u = xnew[i] / nrm;
                    diff_pos += (u - X(i, j)) * (u - X(i, j));
                    diff_neg += (u + X(i, j)) * (u + X(i, j));
                    xnew[i] = u;
                }
                const T change = wet::sqrt(diff_pos < diff_neg ? diff_pos : diff_neg);
                if (change > max_change) {
                    max_change = change;
                }
                for (size_t i = 0; i < NX; ++i) {
                    X(i, j) = xnew[i];
                }
            }
            if (max_change < tol) {
                break;
            }
        }
    }

    // Recover the gain. A − B·K = X·Λ·X⁻¹, so B·K = A − X·Λ·X⁻¹.
    const auto Xinv_opt = X.inverse();
    if (!Xinv_opt) {
        return std::nullopt; // eigenvectors dependent (e.g. multiplicity > NU)
    }
    const auto& Xinv = Xinv_opt.value();

    // M = X·Λ·X⁻¹ (Λ = diag(poles): X·Λ scales column j by poles[j]).
    Matrix<NX, NX, T> XL;
    for (size_t i = 0; i < NX; ++i) {
        for (size_t j = 0; j < NX; ++j) {
            XL(i, j) = X(i, j) * poles[j];
        }
    }
    const Matrix<NX, NX, T> M = XL * Xinv;
    const Matrix<NX, NX, T> A_minus_M = A - M;

    if constexpr (NU == NX) {
        // K = B⁻¹·(A − M).
        const auto Binv_opt = B.inverse();
        if (!Binv_opt) {
            return std::nullopt; // B not invertible
        }
        return Binv_opt.value() * A_minus_M;
    } else {
        // B = U0·Z with Z the top NU×NU block of R, so U0ᵀ·B = Z and
        // K = Z⁻¹·U0ᵀ·(A − M).
        const auto        qrB = mat::full_qr(B);
        Matrix<NX, NU, T> U0;
        for (size_t i = 0; i < NX; ++i) {
            for (size_t c = 0; c < NU; ++c) {
                U0(i, c) = qrB.Q(i, c);
            }
        }
        Matrix<NU, NU, T> Z;
        for (size_t r = 0; r < NU; ++r) {
            for (size_t c = 0; c < NU; ++c) {
                Z(r, c) = qrB.R(r, c);
            }
        }
        const auto Zinv_opt = Z.inverse();
        if (!Zinv_opt) {
            return std::nullopt; // B rank-deficient
        }
        return Zinv_opt.value() * (U0.transpose() * A_minus_M);
    }
}

/**
 * @brief Robust pole placement with complex-conjugate poles (KNV).
 *
 * Generalizes place() to complex spectra. Each conjugate pair (σ ± jω) is
 * assigned in real arithmetic via a real eigenvector pair (Re v, Im v) and a
 * 2×2 real block [[σ, ω], [−ω, σ]] in the closed-loop matrix, so K stays real.
 * Complex poles must be supplied as adjacent conjugate pairs.
 *
 * An all-real spectrum forwards to the real-pole overload (which additionally
 * runs the Method-0 conditioning sweep). The complex path assigns each
 * eigenvector from the orthonormal admissible basis without the extra sweep —
 * the spectrum is still placed exactly; the conditioning is good but not sweep-
 * optimized.
 *
 * @param poles Desired closed-loop eigenvalues as adjacent conjugate pairs
 *              (real poles may appear anywhere).
 * @return Gain K (NU×NX), or std::nullopt if B is rank-deficient, the poles are
 *         not in valid conjugate pairs, or the eigenvectors are dependent.
 *
 * @note Compare with MATLAB's K = place(A, B, poles) for complex spectra.
 */
template<size_t NX, size_t NU, typename T = double>
[[nodiscard]] constexpr std::optional<Matrix<NU, NX, T>> place(
    const Matrix<NX, NX, T>&               A,
    const Matrix<NX, NU, T>&               B,
    const wet::array<wet::complex<T>, NX>& poles
) {
    static_assert(NU <= NX, "place requires NU <= NX (no more inputs than states)");
    constexpr T tol_im = T{1e-9};

    // All-real spectrum → reuse the conditioned real-pole routine.
    bool all_real = true;
    for (size_t i = 0; i < NX; ++i) {
        if (wet::abs(poles[i].imag()) >= tol_im) {
            all_real = false;
            break;
        }
    }
    if (all_real) {
        wet::array<T, NX> rp{};
        for (size_t i = 0; i < NX; ++i) {
            rp[i] = poles[i].real();
        }
        return place(A, B, rp);
    }

    // Build the real block-diagonal Λ from the (paired) spectrum: a real pole is
    // a 1×1 entry, a conjugate pair (σ ± jω) a 2×2 block [[σ, ω], [−ω, σ]].
    // Returns false if the poles are not validly paired.
    Matrix<NX, NX, T> Lambda = Matrix<NX, NX, T>::zeros();
    {
        size_t j = 0;
        while (j < NX) {
            if (wet::abs(poles[j].imag()) < tol_im) {
                Lambda(j, j) = poles[j].real();
                ++j;
            } else {
                if (j + 1 >= NX) {
                    return std::nullopt; // dangling complex pole
                }
                const T sg = poles[j].real();
                const T w = poles[j].imag();
                if (wet::abs(poles[j + 1].real() - sg) > T{1e-7} || wet::abs(poles[j + 1].imag() + w) > T{1e-7}) {
                    return std::nullopt; // not a conjugate pair
                }
                Lambda(j, j) = sg;
                Lambda(j, j + 1) = w;
                Lambda(j + 1, j) = -w;
                Lambda(j + 1, j + 1) = sg;
                j += 2;
            }
        }
    }

    if constexpr (NU == NX) {
        // X = I ⇒ A − B·K = Λ, so K = B⁻¹·(A − Λ).
        const auto Binv_opt = B.inverse();
        if (!Binv_opt) {
            return std::nullopt;
        }
        return Binv_opt.value() * (A - Lambda);
    } else {
        const auto        qrB = mat::full_qr(B);
        constexpr size_t  NC = NX - NU;
        Matrix<NX, NC, T> U1;
        Matrix<NX, NU, T> U0;
        for (size_t i = 0; i < NX; ++i) {
            for (size_t c = 0; c < NU; ++c) {
                U0(i, c) = qrB.Q(i, c);
            }
            for (size_t c = 0; c < NC; ++c) {
                U1(i, c) = qrB.Q(i, NU + c);
            }
        }
        Matrix<NU, NU, T> Z;
        for (size_t r = 0; r < NU; ++r) {
            for (size_t c = 0; c < NU; ++c) {
                Z(r, c) = qrB.R(r, c);
            }
        }

        // U1ᵀ(A − σI) as an NC×NX block (function of σ).
        const auto P_block = [&](size_t r, size_t c, T sigma) {
            T acc = T{0};
            for (size_t k = 0; k < NX; ++k) {
                acc += U1(k, r) * (A(k, c) - (k == c ? sigma : T{0}));
            }
            return acc;
        };

        Matrix<NX, NX, T> X;
        size_t            j = 0;
        while (j < NX) {
            if (wet::abs(poles[j].imag()) < tol_im) {
                // Real pole: x_j = first admissible vector (null space of
                // U1ᵀ(A − σI) = trailing NU columns of full_qr((A − σI)ᵀU1)).
                const T           sigma = poles[j].real();
                Matrix<NX, NC, T> Mt;
                for (size_t r = 0; r < NX; ++r) {
                    for (size_t c = 0; c < NC; ++c) {
                        Mt(r, c) = P_block(c, r, sigma);
                    }
                }
                const auto qrM = mat::full_qr(Mt);
                for (size_t i = 0; i < NX; ++i) {
                    X(i, j) = qrM.Q(i, NC);
                }
                ++j;
            } else {
                // Complex pair: admissible (Re v, Im v) from the real-stacked
                // null space of [[U1ᵀ(A−σI), ω U1ᵀ], [−ω U1ᵀ, U1ᵀ(A−σI)]].
                const T sg = poles[j].real();
                const T w = poles[j].imag();

                Matrix<2 * NX, 2 * NC, T> Ht; // (2NX × 2NC) = transpose of the stacked H
                for (size_t a = 0; a < 2 * NX; ++a) {
                    const size_t ar = a % NX;
                    const bool   a_imag = a >= NX;
                    for (size_t b = 0; b < 2 * NC; ++b) {
                        const size_t br = b % NC;
                        const bool   b_bot = b >= NC;
                        T            h;
                        if (!b_bot) {
                            h = a_imag ? (w * U1(ar, br)) : P_block(br, ar, sg);
                        } else {
                            h = a_imag ? P_block(br, ar, sg) : (-w * U1(ar, br));
                        }
                        Ht(a, b) = h;
                    }
                }
                const auto   qrH = mat::full_qr(Ht);
                const size_t col0 = (2 * NX) - (2 * NU); // first trailing (null-space) column
                for (size_t i = 0; i < NX; ++i) {
                    X(i, j) = qrH.Q(i, col0);          // Re v
                    X(i, j + 1) = qrH.Q(NX + i, col0); // Im v
                }
                j += 2;
            }
        }

        const auto Xinv_opt = X.inverse();
        if (!Xinv_opt) {
            return std::nullopt; // eigenvectors dependent
        }
        const Matrix<NX, NX, T> M = (X * Lambda) * Xinv_opt.value();
        const auto              Zinv_opt = Z.inverse();
        if (!Zinv_opt) {
            return std::nullopt; // B rank-deficient
        }
        return Zinv_opt.value() * (U0.transpose() * (A - M));
    }
}

} // namespace design

} // namespace wet
