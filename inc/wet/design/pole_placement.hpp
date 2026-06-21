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
#include <cstdint>
#include <limits>

#include "wet/backend.hpp"
#include "wet/design/stability.hpp"
#include "wet/math/complex.hpp"
#include "wet/math/math.hpp"
#include "wet/matrix/matrix.hpp"
#include "wet/matrix/svd.hpp"

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
 * @return Gain K (NU×NX), or wet::nullopt if B is rank-deficient or the assigned
 *         eigenvectors are linearly dependent (e.g. a pole repeated more than NU
 *         times — not assignable with independent eigenvectors).
 *
 * @note Compare with MATLAB's K = place(A, B, poles).
 */
template<size_t NX, size_t NU, typename T = double>
[[nodiscard]] constexpr wet::optional<Matrix<NU, NX, T>> place(
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
        const T          tol = T{64} * std::numeric_limits<T>::epsilon();
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
        return wet::nullopt; // eigenvectors dependent (e.g. multiplicity > NU)
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
        // K = B⁻¹·(A − M) by solving B·K = (A − M).
        const auto K = mat::solve(B, A_minus_M);
        if (!K) {
            return wet::nullopt; // B not invertible
        }
        return *K;
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
        // K = Z⁻¹·U0ᵀ·(A − M) by solving Z·K = U0ᵀ·(A − M).
        const auto K = mat::solve(Z, Matrix<NU, NX, T>(U0.transpose() * A_minus_M));
        if (!K) {
            return wet::nullopt; // B rank-deficient
        }
        return *K;
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
 * @return Gain K (NU×NX), or wet::nullopt if B is rank-deficient, the poles are
 *         not in valid conjugate pairs, or the eigenvectors are dependent.
 *
 * @note Compare with MATLAB's K = place(A, B, poles) for complex spectra.
 */
template<size_t NX, size_t NU, typename T = double>
[[nodiscard]] constexpr wet::optional<Matrix<NU, NX, T>> place(
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
                    return wet::nullopt; // dangling complex pole
                }
                const T sg = poles[j].real();
                const T w = poles[j].imag();
                if (wet::abs(poles[j + 1].real() - sg) > T{1e-7} || wet::abs(poles[j + 1].imag() + w) > T{1e-7}) {
                    return wet::nullopt; // not a conjugate pair
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
        // X = I ⇒ A − B·K = Λ, so K = B⁻¹·(A − Λ) by solving B·K = (A − Λ).
        const auto K = mat::solve(B, Matrix<NX, NX, T>(A - Lambda));
        if (!K) {
            return wet::nullopt;
        }
        return *K;
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
            return wet::nullopt; // eigenvectors dependent
        }
        const Matrix<NX, NX, T> M = (X * Lambda) * Xinv_opt.value();
        // K = Z⁻¹·U0ᵀ·(A − M) by solving Z·K = U0ᵀ·(A − M).
        const auto K = mat::solve(Z, Matrix<NU, NX, T>(U0.transpose() * (A - M)));
        if (!K) {
            return wet::nullopt; // B rank-deficient
        }
        return *K;
    }
}

/**
 * @brief One Jordan mini-block of a desired closed-loop spectrum.
 *
 * A block contributes `size` consecutive eigenvalues equal to `eigenvalue`,
 * coupled into a single Jordan chain of that order. `size == 1` is an ordinary
 * (semisimple) eigenvalue; `size > 1` is a defective block requiring generalized
 * eigenvectors. Complex eigenvalues must be supplied as conjugate-pair blocks of
 * equal size (e.g. one block at σ+jω and one at σ−jω).
 */
template<typename T = double>
struct JordanBlock {
    wet::complex<T> eigenvalue; ///< The eigenvalue λ this block places.
    size_t          size;       ///< Mini-block order pᵢₖ (chain length).
};

namespace detail {

/**
 * @brief Precomputed, K-independent data for the Klein–Moore construction.
 *
 * For each distinct processed eigenvalue (reals and the +imaginary member of each
 * conjugate pair) it stores the kernel basis N and pseudoinverse M of the pencil
 * [A − λI | B] — neither depends on the free parameter K — plus the mini-block
 * layout. assemble_vw() then turns any K into the eigenvector/input matrices,
 * which lets an optimizer sweep K without recomputing any SVDs.
 */
template<size_t NX, size_t NU, size_t NB, typename T>
struct JordanPlan {
    using C = wet::complex<T>;
    static constexpr size_t NS = NX + NU;

    size_t                                 ndistinct = 0; ///< Processed eigenvalues.
    size_t                                 total_pos = 0; ///< Chain positions (used K columns).
    wet::array<bool, NB>                   is_real{};
    wet::array<size_t, NB>                 g{};          ///< Mini-blocks per eigenvalue.
    wet::array<wet::array<size_t, NU>, NB> orders{};     ///< Block orders.
    wet::array<size_t, NB>                 pos_offset{}; ///< First K column for eigenvalue.
    wet::array<size_t, NB>                 npos{};       ///< Positions (= multiplicity).
    wet::array<Matrix<NS, NU, C>, NB>      N{};          ///< Kernel bases.
    wet::array<Matrix<NS, NX, C>, NB>      M{};          ///< Pseudoinverses.
};

/// Build the K-independent plan, or nullopt if the requested structure is inadmissible.
template<size_t NX, size_t NU, size_t NB, typename T>
[[nodiscard]] constexpr wet::optional<JordanPlan<NX, NU, NB, T>> prepare_jordan_plan(
    const Matrix<NX, NX, T>&              A,
    const Matrix<NX, NU, T>&              B,
    const wet::array<JordanBlock<T>, NB>& blocks
) {
    using C = wet::complex<T>;
    constexpr size_t NS = NX + NU;
    const T          tol_eq = T{1e-7};

    size_t total = 0;
    for (size_t b = 0; b < NB; ++b) {
        total += blocks[b].size;
    }
    if (total != NX) {
        return wet::nullopt; // block sizes must place exactly NX eigenvalues
    }

    const Matrix<NX, NX, C> Ac = A.template as<C>();
    const Matrix<NX, NU, C> Bc = B.template as<C>();

    JordanPlan<NX, NU, NB, T> plan;
    size_t                    col = 0;
    wet::array<bool, NB>      consumed{};
    for (size_t bi = 0; bi < NB; ++bi) {
        if (consumed[bi]) {
            continue;
        }
        const C    lam = blocks[bi].eigenvalue;
        const bool is_real = (wet::abs(lam.imag()) <= tol_eq);
        if (!is_real && lam.imag() < T{0}) {
            continue; // negative-imaginary member is realified with its partner
        }

        // Count then gather this eigenvalue's mini-blocks.
        size_t cnt = 0;
        for (size_t bj = 0; bj < NB; ++bj) {
            if (!consumed[bj] && wet::abs(blocks[bj].eigenvalue - lam) <= tol_eq) {
                ++cnt;
            }
        }
        if (cnt > NU) {
            return wet::nullopt; // more chains than the kernel dimension allows
        }
        const size_t           e = plan.ndistinct;
        wet::array<size_t, NU> ords{};
        size_t                 g = 0;
        size_t                 mult = 0;
        for (size_t bj = 0; bj < NB; ++bj) {
            if (!consumed[bj] && wet::abs(blocks[bj].eigenvalue - lam) <= tol_eq) {
                ords[g] = blocks[bj].size;
                mult += blocks[bj].size;
                ++g;
                consumed[bj] = true;
            }
        }
        // A complex eigenvalue needs a matching conjugate of equal multiplicity.
        if (!is_real) {
            size_t conj_mult = 0;
            for (size_t bj = 0; bj < NB; ++bj) {
                if (!consumed[bj] && wet::abs(blocks[bj].eigenvalue - wet::conj(lam)) <= tol_eq) {
                    conj_mult += blocks[bj].size;
                    consumed[bj] = true;
                }
            }
            if (conj_mult != mult) {
                return wet::nullopt; // unmatched conjugate pair
            }
        }

        // Pencil S = [A − λI | B]: kernel basis N (last NU columns of the null
        // space) and pseudoinverse M; both independent of K.
        Matrix<NX, NS, C> S;
        for (size_t i = 0; i < NX; ++i) {
            for (size_t c = 0; c < NX; ++c) {
                S(i, c) = Ac(i, c) - (i == c ? lam : C{0});
            }
            for (size_t c = 0; c < NU; ++c) {
                S(i, NX + c) = Bc(i, c);
            }
        }
        const auto ns = mat::null_space(S);
        if (ns.dim != NU) {
            return wet::nullopt; // (A,B) not reachable at λ
        }
        Matrix<NS, NU, C> N;
        for (size_t i = 0; i < NS; ++i) {
            for (size_t k = 0; k < NU; ++k) {
                N(i, k) = ns.vectors(i, NX + k);
            }
        }

        plan.is_real[e] = is_real;
        plan.g[e] = g;
        plan.orders[e] = ords;
        plan.N[e] = N;
        plan.M[e] = mat::pseudo_inverse(S);
        plan.pos_offset[e] = col;
        plan.npos[e] = mult;
        col += mult;
        ++plan.ndistinct;
    }
    plan.total_pos = col;
    if (col == 0) {
        return wet::nullopt;
    }
    return plan;
}

/// Eigenvector matrix V (state parts, real) and input matrix W (input parts) for a given K.
template<size_t NX, size_t NU, size_t NB, typename T>
constexpr void assemble_vw(
    const JordanPlan<NX, NU, NB, T>&       plan,
    const Matrix<NU, NX, wet::complex<T>>& K,
    Matrix<NX, NX, T>&                     V,
    Matrix<NU, NX, T>&                     W
) {
    using C = wet::complex<T>;
    constexpr size_t NS = NX + NU;
    V = Matrix<NX, NX, T>::zeros();
    W = Matrix<NU, NX, T>::zeros();
    size_t col = 0;

    for (size_t e = 0; e < plan.ndistinct; ++e) {
        const bool        is_real = plan.is_real[e];
        const auto&       N = plan.N[e];
        const auto&       M = plan.M[e];
        size_t            pcol = plan.pos_offset[e]; // K-column cursor
        Matrix<NS, NX, C> chains{};
        size_t            nchain = 0;
        for (size_t k = 0; k < plan.g[e]; ++k) {
            wet::array<C, NS> prev{};
            for (size_t l = 0; l < plan.orders[e][k]; ++l) {
                wet::array<C, NS> h{};
                for (size_t i = 0; i < NS; ++i) {
                    C acc{0};
                    for (size_t r = 0; r < NU; ++r) {
                        acc += N(i, r) * K(r, pcol); // N · K(l)
                    }
                    h[i] = acc;
                }
                if (l > 0) {
                    for (size_t i = 0; i < NS; ++i) {
                        C acc{0};
                        for (size_t r = 0; r < NX; ++r) {
                            acc += M(i, r) * prev[r]; // + M · overp(prev)
                        }
                        h[i] += acc;
                    }
                }
                ++pcol;
                if (is_real) {
                    for (size_t i = 0; i < NX; ++i) {
                        V(i, col) = h[i].real();
                    }
                    for (size_t i = 0; i < NU; ++i) {
                        W(i, col) = h[NX + i].real();
                    }
                    ++col;
                } else {
                    for (size_t i = 0; i < NS; ++i) {
                        chains(i, nchain) = h[i];
                    }
                    ++nchain;
                }
                prev = h;
            }
        }
        if (!is_real) {
            // Realify: real parts then imaginary parts of the chain columns.
            for (size_t t = 0; t < nchain; ++t) {
                for (size_t i = 0; i < NX; ++i) {
                    V(i, col) = chains(i, t).real();
                }
                for (size_t i = 0; i < NU; ++i) {
                    W(i, col) = chains(NX + i, t).real();
                }
                ++col;
            }
            for (size_t t = 0; t < nchain; ++t) {
                for (size_t i = 0; i < NX; ++i) {
                    V(i, col) = chains(i, t).imag();
                }
                for (size_t i = 0; i < NU; ++i) {
                    W(i, col) = chains(NX + i, t).imag();
                }
                ++col;
            }
        }
    }
}

/// The canonical (minimum-norm) parameter: chain k starts from kernel column k.
template<size_t NX, size_t NU, size_t NB, typename T>
[[nodiscard]] constexpr Matrix<NU, NX, wet::complex<T>> canonical_kparams(
    const JordanPlan<NX, NU, NB, T>& plan
) {
    using C = wet::complex<T>;
    Matrix<NU, NX, C> K = Matrix<NU, NX, C>::zeros();
    for (size_t e = 0; e < plan.ndistinct; ++e) {
        size_t pcol = plan.pos_offset[e];
        for (size_t k = 0; k < plan.g[e]; ++k) {
            K(k, pcol) = C{1}; // h(1) = N·e_k = kernel column k
            pcol += plan.orders[e][k];
        }
    }
    return K;
}

} // namespace detail

/**
 * @ingroup pole_placement
 * @brief Exact pole placement with an arbitrary Jordan structure
 *        (Schmid–Ntogramatzidis–Nguyen–Pandey / Klein–Moore parametric form).
 *
 * Unlike place(), which assigns a non-defective spectrum (each eigenvalue with
 * independent eigenvectors), this assigns *any* admissible eigenstructure: any
 * eigenvalues with any algebraic multiplicities and any Jordan mini-block
 * orders — including fully defective blocks and closed-loop poles that coincide
 * with open-loop ones. It computes a real gain K such that A − B·K has exactly
 * the requested Jordan structure (same A − B·K convention as place()).
 *
 * The method builds each Jordan chain from the kernel and Moore–Penrose
 * pseudoinverse of the matrix pencil S(λ) = [A − λI | B]: the chain head is a
 * kernel vector (a closed-loop eigenvector) and each successor solves
 * S(λ)·h = v_prev for the generalized eigenvector, which the pseudoinverse does
 * directly since S(λ) has full row rank for a reachable (A, B). Stacking the
 * state parts gives the eigenvector matrix V and the input parts W; the gain is
 * K = −W·V⁻¹. Conjugate eigenpairs are realified (Re/Im columns) so K is real.
 *
 * This uses the canonical minimum-norm chain parameter, which places the
 * structure exactly but does not optimize robustness or gain. For the paper's
 * robust/minimum-gain selection of the free parameter, use place_jordan_optimal.
 *
 * @tparam NX Number of states
 * @tparam NU Number of inputs (NU ≤ NX)
 * @tparam NB Number of Jordan blocks supplied
 * @param A      State matrix (NX×NX)
 * @param B      Input matrix (NX×NU), assumed full column rank and (A,B) reachable
 * @param blocks Desired Jordan structure; block sizes must sum to NX, complex
 *               eigenvalues given as equal-size conjugate pairs
 * @return Gain K (NU×NX) with A − B·K in the requested Jordan form, or
 *         wet::nullopt if the structure is inadmissible (block sizes do not sum
 *         to NX, an eigenvalue is asked for more mini-blocks than NU, (A,B) is
 *         not reachable at some λ, or the canonical parameter yields a singular
 *         eigenvector matrix).
 *
 * @see R. Schmid, L. Ntogramatzidis, T. Nguyen, A. Pandey, "A unified method for
 *      optimal arbitrary pole placement," Automatica 50(8), 2014,
 *      https://doi.org/10.1016/j.automatica.2014.05.020
 * @see G. Klein, B. C. Moore, "Eigenvalue-generalized eigenvector assignment
 *      with state feedback," IEEE TAC 22(1), 1977.
 * @see place for the non-defective (distinct/semisimple) robust path.
 */
template<size_t NX, size_t NU, size_t NB, typename T = double>
[[nodiscard]] constexpr wet::optional<Matrix<NU, NX, T>> place_jordan(
    const Matrix<NX, NX, T>&              A,
    const Matrix<NX, NU, T>&              B,
    const wet::array<JordanBlock<T>, NB>& blocks
) {
    static_assert(NU <= NX, "place_jordan requires NU <= NX (no more inputs than states)");

    const auto plan_opt = detail::prepare_jordan_plan(A, B, blocks);
    if (!plan_opt) {
        return wet::nullopt;
    }
    const auto&       plan = plan_opt.value();
    const auto        K = detail::canonical_kparams(plan);
    Matrix<NX, NX, T> V;
    Matrix<NU, NX, T> W;
    detail::assemble_vw(plan, K, V, W);

    const auto Vinv = V.inverse();
    if (!Vinv) {
        return wet::nullopt; // canonical parameter gave dependent eigenvectors
    }
    // A − B·K = V·Λ·V⁻¹ ⇒ closed loop has the Jordan form, with K = −W·V⁻¹.
    return Matrix<NU, NX, T>(T{-1} * (W * Vinv.value()));
}

/// Robustness objective for place_jordan_optimal (the paper's two methods).
enum class JordanObjective : std::uint8_t {
    ConditionNumber,        ///< Method 1: minimize the Frobenius condition number of V.
    DepartureFromNormality, ///< Method 2: minimize the departure from normality of A − B·K.
};

/// Result of optimized arbitrary pole placement (place_jordan_optimal).
template<size_t NU, size_t NX, typename T = double>
struct OptimalJordanPlacement {
    Matrix<NU, NX, T> gain;          ///< K, with the A − B·K convention.
    T                 cond_fro;      ///< Achieved κ_F(V) = ‖V‖_F·‖V⁻¹‖_F (eigenvalue robustness).
    T                 gain_fro;      ///< Achieved ‖K‖_F (control effort).
    T                 departure_fro; ///< Achieved δ_F(A − B·K) (departure from normality).
    size_t            iterations;    ///< Gradient-descent iterations taken.
    bool              converged;     ///< Search reached a stationary point.
};

/**
 * @ingroup pole_placement
 * @brief Robust / minimum-gain arbitrary pole placement (Schmid et al., Methods 1–2).
 *
 * Places the same arbitrary Jordan structure as place_jordan, but spends the free
 * parameter K of the Klein–Moore parameterization to optimize a weighted blend of
 * eigenvalue robustness and control effort. Every K in the family places the
 * structure *exactly* (Theorem 2.1), so the search only trades robustness against
 * gain — the assigned poles never move.
 *
 * The objective is f(K) = α·R(K) + (1 − α)·‖K‖²_F, with the robustness term R
 * selected by @p objective:
 *  - ConditionNumber (Method 1): R = ‖V‖²_F + ‖V⁻¹‖²_F, the Byers–Nash proxy for
 *    the Frobenius condition number κ_F(V) (the eigenvalue sensitivity).
 *  - DepartureFromNormality (Method 2): R = δ²_F(A − B·K) = ‖A − B·K‖²_F − Σ|λᵢ|²,
 *    a closed form since the closed-loop eigenvalues are exactly the targets.
 *
 * α = 1 is the pure robust problem (REPP), α = 0 the pure minimum-gain problem
 * (MGEPP). The unconstrained nonconvex objective is minimized by gradient descent
 * (central finite differences, Armijo backtracking) from the canonical parameter;
 * as the paper notes, the result is a local minimum dependent on that start. The
 * kernel/pseudoinverse factors are precomputed once, so each step is cheap.
 *
 * @tparam NX Number of states
 * @tparam NU Number of inputs (NU ≤ NX)
 * @tparam NB Number of Jordan blocks supplied
 * @param A         State matrix (NX×NX)
 * @param B         Input matrix (NX×NU), full column rank and (A,B) reachable
 * @param blocks    Desired Jordan structure (see place_jordan)
 * @param alpha     Robustness/gain weight in [0,1]; 1 = robust, 0 = minimum gain
 * @param objective Which robustness measure to minimize (Method 1 or 2)
 * @param max_iter  Maximum gradient-descent iterations
 * @return OptimalJordanPlacement with the gain and achieved metrics, or
 *         wet::nullopt if the structure is inadmissible (as in place_jordan).
 *
 * @see place_jordan for the unoptimized (canonical-parameter) placement.
 * @see R. Schmid et al., "A unified method for optimal arbitrary pole placement,"
 *      Automatica 50(8), 2014.
 */
template<size_t NX, size_t NU, size_t NB, typename T = double>
[[nodiscard]] wet::optional<OptimalJordanPlacement<NU, NX, T>> place_jordan_optimal(
    const Matrix<NX, NX, T>&              A,
    const Matrix<NX, NU, T>&              B,
    const wet::array<JordanBlock<T>, NB>& blocks,
    T                                     alpha = T{1},
    JordanObjective                       objective = JordanObjective::ConditionNumber,
    size_t                                max_iter = 200
) {
    static_assert(NU <= NX, "place_jordan_optimal requires NU <= NX");
    using C = wet::complex<T>;

    const auto plan_opt = detail::prepare_jordan_plan(A, B, blocks);
    if (!plan_opt) {
        return wet::nullopt;
    }
    const auto& plan = plan_opt.value();

    // Σ|λ|² over all eigenvalues, for the departure-from-normality measure.
    T sum_lambda_sq = T{0};
    for (size_t b = 0; b < NB; ++b) {
        const C lam = blocks[b].eigenvalue;
        sum_lambda_sq += static_cast<T>(blocks[b].size) * ((lam.real() * lam.real()) + (lam.imag() * lam.imag()));
    }

    // Real degrees of freedom: NU per real-eigenvalue chain position, 2·NU per
    // complex one (the conjugate is determined). Laid out into a flat θ vector.
    constexpr size_t MaxDof = 2 * NU * NX;
    size_t           dof = 0;
    for (size_t e = 0; e < plan.ndistinct; ++e) {
        dof += plan.npos[e] * (plan.is_real[e] ? NU : (2 * NU));
    }

    const auto theta_to_k = [&](const wet::array<T, MaxDof>& th) {
        Matrix<NU, NX, C> K = Matrix<NU, NX, C>::zeros();
        size_t            idx = 0;
        for (size_t e = 0; e < plan.ndistinct; ++e) {
            size_t pcol = plan.pos_offset[e];
            for (size_t p = 0; p < plan.npos[e]; ++p) {
                for (size_t r = 0; r < NU; ++r) {
                    if (plan.is_real[e]) {
                        K(r, pcol) = C{th[idx], T{0}};
                        idx += 1;
                    } else {
                        K(r, pcol) = C{th[idx], th[idx + 1]};
                        idx += 2;
                    }
                }
                ++pcol;
            }
        }
        return K;
    };

    // Objective value for a θ; returns a large penalty when V is singular.
    const T    penalty = T{1e30};
    const auto eval = [&](const wet::array<T, MaxDof>& th) -> T {
        Matrix<NX, NX, T> V;
        Matrix<NU, NX, T> W;
        detail::assemble_vw(plan, theta_to_k(th), V, W);
        const auto Vinv = V.inverse();
        if (!Vinv) {
            return penalty;
        }
        const Matrix<NU, NX, T> F = Matrix<NU, NX, T>(W * Vinv.value()); // paper's F = −K
        const T                 gain2 = F.norm() * F.norm();
        if (objective == JordanObjective::ConditionNumber) {
            const T vn = V.norm();
            const T vin = Vinv.value().norm();
            return (alpha * ((vn * vn) + (vin * vin))) + ((T{1} - alpha) * gain2);
        }
        const Matrix<NX, NX, T> M = Matrix<NX, NX, T>(A + (B * F)); // closed loop A + B·F
        T                       dep2 = (M.norm() * M.norm()) - sum_lambda_sq;
        if (dep2 < T{0}) {
            dep2 = T{0};
        }
        return (alpha * dep2) + ((T{1} - alpha) * gain2);
    };

    // Start from the canonical parameter.
    wet::array<T, MaxDof> theta{};
    {
        const auto Kc = detail::canonical_kparams(plan);
        size_t     idx = 0;
        for (size_t e = 0; e < plan.ndistinct; ++e) {
            size_t pcol = plan.pos_offset[e];
            for (size_t p = 0; p < plan.npos[e]; ++p) {
                for (size_t r = 0; r < NU; ++r) {
                    theta[idx] = Kc(r, pcol).real();
                    idx += plan.is_real[e] ? 1 : 2;
                    if (!plan.is_real[e]) {
                        theta[idx - 1] = Kc(r, pcol).imag();
                    }
                }
                ++pcol;
            }
        }
    }

    T f0 = eval(theta);
    if (f0 >= penalty) {
        return wet::nullopt; // canonical parameter already gives a singular V
    }

    // Gradient descent: central finite-difference gradient + Armijo backtracking.
    const T h = T{1e-6};
    const T c1 = T{1e-4};
    const T ftol = T{1e-10}; // relative function-decrease stop
    bool    converged = false;
    size_t  iter = 0;
    for (; iter < max_iter; ++iter) {
        wet::array<T, MaxDof> grad{};
        T                     gnorm2 = T{0};
        for (size_t i = 0; i < dof; ++i) {
            wet::array<T, MaxDof> tp = theta;
            wet::array<T, MaxDof> tm = theta;
            tp[i] += h;
            tm[i] -= h;
            grad[i] = (eval(tp) - eval(tm)) / (T{2} * h);
            gnorm2 += grad[i] * grad[i];
        }
        if (gnorm2 == T{0}) {
            converged = true;
            break;
        }
        const T f_before = f0;
        T       step = T{1};
        bool    improved = false;
        for (size_t ls = 0; ls < 40; ++ls) {
            wet::array<T, MaxDof> tn = theta;
            for (size_t i = 0; i < dof; ++i) {
                tn[i] -= step * grad[i];
            }
            const T fn = eval(tn);
            if (fn < (f0 - (c1 * step * gnorm2))) {
                theta = tn;
                f0 = fn;
                improved = true;
                break;
            }
            step *= T{0.5};
        }
        // Stop when the line search stalls or the relative decrease is negligible.
        if (!improved || ((f_before - f0) <= ftol * (wet::abs(f_before) + T{1}))) {
            converged = true;
            break;
        }
    }

    // Assemble the final gain and metrics.
    Matrix<NX, NX, T> V;
    Matrix<NU, NX, T> W;
    detail::assemble_vw(plan, theta_to_k(theta), V, W);
    const auto Vinv = V.inverse();
    if (!Vinv) {
        return wet::nullopt;
    }
    const Matrix<NU, NX, T> F = Matrix<NU, NX, T>(W * Vinv.value());
    const Matrix<NX, NX, T> M = Matrix<NX, NX, T>(A + (B * F));
    T                       dep2 = (M.norm() * M.norm()) - sum_lambda_sq;
    if (dep2 < T{0}) {
        dep2 = T{0};
    }
    OptimalJordanPlacement<NU, NX, T> result;
    result.gain = Matrix<NU, NX, T>(T{-1} * F);
    result.cond_fro = V.norm() * Vinv.value().norm();
    result.gain_fro = F.norm();
    result.departure_fro = wet::sqrt(dep2);
    result.iterations = iter;
    result.converged = converged;
    return result;
}

/**
 * @brief Single-input pole placement via Ackermann's formula.
 *
 * Computes the state-feedback gain K placing the eigenvalues of (A − B·K) at the
 * requested poles. Unlike place(), Ackermann works through the characteristic
 * polynomial rather than eigenvector assignment, so it assigns repeated/defective
 * spectra fine — but it is single-input only and numerically weaker for large NX.
 * Use place() for multi-input systems or when robustness matters.
 *
 * @tparam NX Number of states
 * @param A      State matrix (NX×NX)
 * @param B      Input vector (NX×1)
 * @param poles  Desired closed-loop eigenvalues (complex; conjugate pairs cancel
 *               to a real characteristic polynomial)
 * @return Gain K (1×NX), or wet::nullopt if (A, B) is uncontrollable.
 *
 * @note Compare with MATLAB's K = acker(A, B, p).
 * @see place for the robust multi-input path.
 */
template<size_t NX, typename T = double>
[[nodiscard]] constexpr wet::optional<Matrix<1, NX, T>> ackermann(
    const Matrix<NX, NX, T>&               A,
    const Matrix<NX, 1, T>&                B,
    const wet::array<wet::complex<T>, NX>& poles
) {
    // Controllability matrix Co = [B, AB, …, A^{NX-1}B] (NU = 1 ⇒ NX×NX).
    const Matrix<NX, NX, T> Co = stability::controllability_matrix(A, B);

    // Desired characteristic polynomial φ(s) = Π(s − pᵢ), built in complex so
    // conjugate pairs cancel to real coefficients.
    wet::array<wet::complex<T>, NX + 1> cc{};
    cc[0] = wet::complex<T>{T{1}, T{0}};
    for (size_t i = 0; i < NX; ++i) {
        const wet::complex<T> root = poles[i];
        wet::complex<T>       carry = cc[0];
        cc[0] = wet::complex<T>{T{0}, T{0}} - (root * cc[0]);
        for (size_t j = 1; j <= NX; ++j) {
            const wet::complex<T> next = cc[j];
            cc[j] = carry - (root * cc[j]);
            carry = next;
        }
    }

    // φ(A) = Σ Re(coeffs[k]) · Aᵏ
    Matrix<NX, NX, T> phi_A = Matrix<NX, NX, T>::zeros();
    Matrix<NX, NX, T> A_power = Matrix<NX, NX, T>::identity();
    for (size_t k = 0; k <= NX; ++k) {
        phi_A = phi_A + (cc[k].real() * A_power);
        A_power = A_power * A;
    }

    // K = e_Nᵀ · Co⁻¹ · φ(A). Solve Co·X = φ(A) instead of forming Co⁻¹;
    // a singular Co (uncontrollable) makes solve() fail.
    Matrix<1, NX, T> e_N{};
    e_N(0, NX - 1) = T{1};
    const auto CoInv_phi = mat::solve(Co, phi_A);
    if (!CoInv_phi) {
        return wet::nullopt;
    }
    return Matrix<1, NX, T>(e_N * (*CoInv_phi));
}

} // namespace design

} // namespace wet
