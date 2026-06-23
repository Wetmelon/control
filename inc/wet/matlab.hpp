#pragma once

#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

#include "design/stability.hpp"
#include "wet/analysis/analysis.hpp"
#include "wet/backend.hpp"
#include "wet/controllers/lqg.hpp"
#include "wet/controllers/lqgi.hpp"
#include "wet/controllers/lqi.hpp"
#include "wet/controllers/lqr.hpp"
#include "wet/controllers/pid.hpp"
#include "wet/design/linearization.hpp"
#include "wet/design/pole_placement.hpp"
#include "wet/estimation/kalman.hpp"
#include "wet/math/complex.hpp"
#include "wet/matrix/eigen.hpp"
#include "wet/matrix/matrix.hpp"
#include "wet/matrix/svd.hpp"
#include "wet/systems/discretization.hpp"
#include "wet/systems/state_space.hpp"
#include "wet/systems/transfer_function.hpp"

namespace wet {
// MATLAB®-style matrix functions
namespace matlab {

/**
 * @brief MATLAB-style transfer function constructor
 *
 * Coefficients are in ascending powers of s or z.
 */
template<size_t Nnum, size_t Nden, typename T = double>
[[nodiscard]] constexpr TransferFunction<Nnum, Nden, T>
tf(const wet::array<T, Nnum>& num, const wet::array<T, Nden>& den) noexcept {
    return TransferFunction<Nnum, Nden, T>{.num = num, .den = den};
}

/**
 * @brief MATLAB-style transfer function constructor from braced coefficient lists
 *
 * Enables calls like: tf({1.0, 2.0}, {3.0, 4.0, 5.0})
 * Coefficients are in ascending powers of s or z.
 */
template<typename TNum, size_t Nnum, typename TDen, size_t Nden>
[[nodiscard]] constexpr TransferFunction<Nnum, Nden, std::common_type_t<TNum, TDen>>
tf(const TNum (&num)[Nnum], const TDen (&den)[Nden]) noexcept {
    using T = std::common_type_t<TNum, TDen>;
    TransferFunction<Nnum, Nden, T> result{};

    for (size_t i = 0; i < Nnum; ++i) {
        result.num[i] = static_cast<T>(num[i]);
    }
    for (size_t i = 0; i < Nden; ++i) {
        result.den[i] = static_cast<T>(den[i]);
    }

    return result;
}

/**
 * @brief MATLAB short alias for controllability_matrix
 */
template<size_t NX, size_t NU, typename T = double>
[[nodiscard]] constexpr Matrix<NX, NX * NU, T> ctrb(const Matrix<NX, NX, T>& A, const Matrix<NX, NU, T>& B) noexcept {
    return stability::controllability_matrix(A, B);
}

/**
 * @brief MATLAB short alias for controllability_matrix taking StateSpace
 */
template<size_t NX, size_t NU, size_t NY, size_t NW, size_t NV, typename T = double>
[[nodiscard]] constexpr Matrix<NX, NX * NU, T> ctrb(const StateSpace<NX, NU, NY, NW, NV, T>& sys) noexcept {
    return stability::controllability_matrix(sys.A, sys.B);
}

/**
 * @brief MATLAB short alias for observability_matrix
 */
template<size_t NX, size_t NY, typename T = double>
[[nodiscard]] constexpr Matrix<NX * NY, NX, T> obsv(const Matrix<NX, NX, T>& A, const Matrix<NY, NX, T>& C) noexcept {
    return stability::observability_matrix(A, C);
}

/**
 * @brief MATLAB short alias for observability_matrix
 */
template<size_t NX, size_t NU, size_t NY, size_t NW, size_t NV, typename T = double>
[[nodiscard]] constexpr Matrix<NX * NY, NX, T> obsv(const StateSpace<NX, NU, NY, NW, NV, T>& sys) noexcept {
    return stability::observability_matrix(sys.A, sys.C);
}

/**
 * @brief Matlab interface function c2d to discretize a continuous-time state-space system
 *
 * @param sys           Continuous-time state-space model (Ts should be 0)
 * @param sampling_time Desired sampling period for discrete system
 * @param method        Discretization method (ZOH or Tustin)
 *
 * @return constexpr StateSpace<NX, NU, NY, NW, NV, T>
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
[[nodiscard]] constexpr StateSpace<NX, NU, NY, NW, NV, T> c2d(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    T                                        sampling_time,
    DiscretizationMethod                     method = DiscretizationMethod::ZOH
) {
    return discretize(sys, sampling_time, method);
}

/**
 * @brief MATLAB-style c2d for SISO transfer functions
 *
 * Returns a discrete-time state-space model for now.
 */
template<size_t Nnum, size_t Nden, typename T = double>
[[nodiscard]] constexpr StateSpace<Nden - 1, 1, 1, 0, 0, T> c2d(
    const TransferFunction<Nnum, Nden, T>& tf_sys,
    T                                      sampling_time,
    DiscretizationMethod                   method = DiscretizationMethod::ZOH
) {
    return discretize(tf_sys.to_state_space(), sampling_time, method);
}

/**
 * @brief MATLAB-style nonlinear linearization about an operating point
 *
 * Similar in spirit to MATLAB's linmod workflow: linearize nonlinear
 * dynamics and output maps, then return continuous-time state-space.
 */
template<size_t NX, size_t NU, size_t NY, typename T = double, typename Dynamics, typename Output>
[[nodiscard]] constexpr StateSpace<NX, NU, NY, 0, 0, T> linmod(
    const Dynamics&      dynamics,
    const Output&        output,
    const ColVec<NX, T>& x_op,
    const ColVec<NU, T>& u_op,
    T                    epsilon = T{0}
) {
    if (epsilon > T{0}) {
        return linearize<NX, NU, NY, T>(dynamics, output, x_op, u_op, epsilon).to_state_space();
    }
    return linearize<NX, NU, NY, T>(dynamics, output, x_op, u_op).to_state_space();
}

/**
 * @brief Block diagonal matrix construction
 * @ingroup linear_algebra
 * @tparam Args Variadic list of matrix types
 * @return Block diagonal matrix composed of the input matrices
 */
template<typename... Args>
[[nodiscard]] constexpr auto blkdiag(Args... args) noexcept
    requires(sizeof...(Args) > 0 && (is_matrix_type<Args>::value && ...))
{
    constexpr size_t total_rows = (args.rows() + ...);
    constexpr size_t total_cols = (args.cols() + ...);
    using T = std::common_type_t<typename std::decay_t<Args>::value_type...>;
    Matrix<total_rows, total_cols, T> result = Matrix<total_rows, total_cols, T>::zeros();

    size_t row_offset = 0;
    size_t col_offset = 0;
    ((result.template block<args.rows(), args.cols()>(row_offset, col_offset) = args, row_offset += args.rows(), col_offset += args.cols()), ...);

    return result;
}

/**
 * @brief Returns a square diagonal matrix from the given array
 *
 * @param diag Array of diagonal elements
 * @param k    Diagonal offset. k = 0 (main diagonal), k > 0 (superdiagonal), k < 0 (subdiagonal)
 *
 * @return Matrix<N, N, T>
 */
template<size_t N, typename T, int k = 0>
[[nodiscard]] constexpr auto diag(const wet::array<T, N>& diag) noexcept {
    Matrix<N, N, T> result = Matrix<N, N, T>::zeros();
    for (size_t i = 0; i < N; ++i) {
        size_t row = i;
        size_t col = i + k;
        if (col < N) {
            result(row, col) = diag[i];
        }
    }
    return result;
}

/**
 * @brief Returns the diagonal elements of a square matrix as a column vector
 *
 * @param A Input square matrix
 *
 * @return ColVec<N, T>
 */
template<size_t N, typename T, int k = 0>
[[nodiscard]] constexpr ColVec<N, T> diag(const Matrix<N, N, T>& A) noexcept {
    ColVec<N, T> result;
    for (size_t i = 0; i < N; ++i) {
        size_t row = i;
        size_t col = i + k;
        if (col < N) {
            result(i, 0) = A(row, col);
        } else {
            result(i, 0) = T{0};
        }
    }
    return result;
}

/**
 * @brief Create an identity matrix of size n x n
 *
 * @tparam N Size of the identity matrix
 * @return constexpr Matrix<N, N, T>
 */
template<size_t N, typename T = double>
[[nodiscard]] constexpr auto eye() noexcept {
    return Matrix<N, N, T>::identity();
}

template<size_t N, typename T = double>
[[nodiscard]]
constexpr auto ones() noexcept {
    Matrix<N, N, T> result;
    for (auto& row : result.data_) {
        row.fill(T{1});
    }
    return result;
}

/**
 * @brief MATLAB short alias for the singular value decomposition.
 *
 * Returns a mat::SVDResult{singular_U, singular_values, singular_V} rather than
 * the MATLAB [U, S, V] tuple; the singular values are a descending array, not a
 * diagonal matrix.
 *
 * @note Compare with MATLAB's [U, S, V] = svd(A).
 */
template<size_t M, size_t N, typename T>
[[nodiscard]] constexpr auto svd(const Matrix<M, N, T>& A) {
    return mat::svd(A);
}

/**
 * @brief MATLAB short alias for the Moore–Penrose pseudoinverse.
 * @note Compare with MATLAB's pinv(A).
 */
template<size_t M, size_t N, typename T>
[[nodiscard]] constexpr Matrix<N, M, T> pinv(const Matrix<M, N, T>& A) {
    return mat::pseudo_inverse(A);
}

/**
 * @brief MATLAB short alias for an orthonormal null-space basis.
 *
 * Returns a mat::NullSpace whose trailing `dim` columns are the kernel basis
 * (MATLAB's null returns those columns directly as an N×dim matrix).
 *
 * @note Compare with MATLAB's null(A).
 */
template<size_t M, size_t N, typename T>
[[nodiscard]] constexpr mat::NullSpace<N, T> null(const Matrix<M, N, T>& A) {
    return mat::null_space(A);
}

/**
 * @brief MATLAB short alias for the eigenvalues of a square matrix.
 *
 * Returns the eigenvalues as a complex column vector. Closed-form for N ≤ 4,
 * Francis double-shift QR for larger systems. For eigenvectors and the
 * convergence flag, call mat::compute_eigenvalues() directly.
 *
 * @note Compare with MATLAB's e = eig(A).
 */
template<size_t N, typename T>
[[nodiscard]] constexpr ColVec<N, wet::complex<T>> eig(const Matrix<N, N, T>& A) {
    return mat::compute_eigenvalues(A).values;
}

/**
 * @brief Form state estimator from system and estimator gain
 *
 * est = estim(sys,L) produces a state/output estimator est given the plant state-space model sys
 *       and the estimator gain L. All inputs w of sys are assumed stochastic (process and/or measurement noise),
 *       and all outputs y are measured. The estimator est is returned in state-space form.
 *
 * @param sys State-space system (discrete or continuous)
 * @param L   Estimator gain matrix
 *
 * @return StateSpace<NX, NY, NX, NW, NV, T>
 */
template<size_t NX, size_t NU, size_t NY, size_t NW, size_t NV, typename T = double>
constexpr auto estim(const StateSpace<NX, NU, NY, NW, NV, T>& sys, const Matrix<NX, NY, T>& L) noexcept {
    Matrix<NX, NX, T> A_est = sys.A - L * sys.C;
    Matrix<NX, NY, T> B_est = L;
    Matrix<NX, NX, T> C_est = Matrix<NX, NX, T>::identity();
    Matrix<NX, NY, T> D_est = Matrix<NX, NY, T>::zeros();
    Matrix<NX, NW, T> G_est = sys.G - L * sys.H;
    Matrix<NX, NV, T> H_est = L;

    return StateSpace<NX, NY, NX, NW, NV, T>{A_est, B_est, C_est, D_est, G_est, H_est};
}

/**
 * @brief Form dynamic regulator from system, state-feedback gain, and estimator gain
 *
 * rsys = reg(sys,K,L) forms a dynamic regulator or compensator rsys given a state-space model sys of the plant,
 *         a state-feedback gain matrix K, and an estimator gain matrix L.
 *         The gains K and L are typically designed using pole placement or LQG techniques.
 *         The function reg handles both continuous- and discrete-time cases.
 *
 *    This syntax assumes that all inputs of sys are controls, and all outputs are measured.
 *    The regulator rsys is obtained by connecting the state-feedback law u = –Kx and the state estimator with gain matrix L (see estim).
 *
 * @param sys State-space system (discrete or continuous)
 * @param K   State-feedback gain matrix
 * @param L   Estimator gain matrix
 *
 * @return StateSpace<2*NX, NW+NV, NY, 0, 0, T>
 */
template<size_t NX, size_t NU, size_t NY, size_t NW, size_t NV, typename T = double>
constexpr auto reg(const StateSpace<NX, NU, NY, NW, NV, T>& sys, const Matrix<NU, NX, T>& K, const Matrix<NX, NY, T>& L) noexcept {
    constexpr size_t N_reg = 2 * NX;
    constexpr size_t M_reg = NW + NV;
    constexpr size_t P_reg = NY;

    Matrix<N_reg, N_reg, T> A_reg = Matrix<N_reg, N_reg, T>::zeros();
    A_reg.template block<NX, NX>(0, 0) = sys.A;
    A_reg.template block<NX, NX>(0, NX) = -sys.B * K;
    A_reg.template block<NX, NX>(NX, 0) = L * sys.C;
    A_reg.template block<NX, NX>(NX, NX) = sys.A - L * sys.C - L * sys.D * K;

    Matrix<N_reg, M_reg, T> B_reg = Matrix<N_reg, M_reg, T>::zeros();
    B_reg.template block<NX, NW>(0, 0) = sys.G;
    B_reg.template block<NX, NV>(NX, NW) = L * sys.H;

    Matrix<P_reg, N_reg, T> C_reg = Matrix<P_reg, N_reg, T>::zeros();
    C_reg.template block<NY, NX>(0, 0) = sys.C;
    C_reg.template block<NY, NX>(0, NX) = -sys.D * K;

    Matrix<P_reg, M_reg, T> D_reg = Matrix<P_reg, M_reg, T>::zeros();
    D_reg.template block<NY, NV>(0, NW) = sys.H;

    return StateSpace<N_reg, M_reg, P_reg, 0, 0, T>{A_reg, B_reg, C_reg, D_reg};
}

/**
 * @brief Pole placement for state-feedback control
 *
 * @param A State matrix
 * @param B Input matrix
 * @param p Desired poles (accepts wet::array<std::complex<T>, NX> or wet::array<_Complex T, NX>)
 *
 * @return wet::optional<Matrix<NU, NX, T>> State-feedback gain K, or nullopt if not implementable
 */
template<size_t NX, size_t NU, typename T = double>
constexpr wet::optional<Matrix<NU, NX, T>> acker(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const auto&              p
) {
    static_assert(NU == 1, "acker is single-input only; use place for multi-input systems");
    const auto poles = std::apply(
        [](const auto&... elems) { return wet::array<wet::complex<T>, sizeof...(elems)>{elems...}; }, p
    );
    return design::ackermann(A, B, poles);
}

/**
 * @brief Robust multi-input pole placement (MATLAB's place)
 *
 * Thin alias for design::place — Kautsky–Nichols–Van Dooren robust eigenvalue
 * assignment, spending the multi-input freedom to minimize eigenvector
 * conditioning. Use this (not acker) for multi-input systems; acker remains the
 * single-input Ackermann path.
 *
 * @param A State matrix
 * @param B Input matrix
 * @param p Desired poles (wet::array<std::complex<T>, NX> or wet::array<_Complex T, NX>)
 * @return State-feedback gain K (NU×NX), or nullopt if not assignable.
 *
 * @note Compare with MATLAB's K = place(A, B, p).
 */
template<size_t NX, size_t NU, typename T = double>
[[nodiscard]] constexpr wet::optional<Matrix<NU, NX, T>> place(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const auto&              p
) {
    const auto poles = std::apply(
        [](const auto&... elems) { return wet::array<wet::complex<T>, sizeof...(elems)>{elems...}; }, p
    );
    return design::place(A, B, poles);
}

/**
 * @brief Continuous-time LQR design (MATLAB's lqr)
 *
 * Thin alias for design::continuous_lqr — the optimal gain K for u = −Kx
 * minimizing ∫(xᵀQx + uᵀRu + 2xᵀNu) dt via the continuous ARE (care()).
 *
 * @note Compare with MATLAB's K = lqr(A, B, Q, R, N). For the Riccati solution
 *       S and the closed-loop poles, call design::continuous_lqr directly.
 */
template<size_t NX, size_t NU, typename T = double>
[[nodiscard]] constexpr Matrix<NU, NX, T> lqr(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NX, NX, T>& Q,
    const Matrix<NU, NU, T>& R,
    const Matrix<NX, NU, T>& N = Matrix<NX, NU, T>{}
) {
    return design::continuous_lqr(A, B, Q, R, N).K; // Only return K
}

/**
 * @brief Discrete-time Linear-Quadratic Regulator design
 * @note Alias for design::discrete_lqr. Compare with MATLAB's dlqr(A, B, Q, R, N).
 */
template<size_t NX, size_t NU, typename T = double>
[[nodiscard]] constexpr auto dlqr(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NX, NX, T>& Q,
    const Matrix<NU, NU, T>& R,
    const Matrix<NX, NU, T>& N = Matrix<NX, NU, T>{}
) {
    return design::discrete_lqr(A, B, Q, R, N);
}

/**
 * @brief Design discrete LQR from continuous-time system via discretization
 * @note Alias for design::discrete_lqr_from_continuous. Compare with MATLAB's lqrd(A, B, Q, R, Ts).
 */
template<size_t NX, size_t NU, typename T = double>
[[nodiscard]] constexpr auto lqrd(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const Matrix<NX, NX, T>& Q,
    const Matrix<NU, NU, T>& R,
    T                        Ts,
    const Matrix<NX, NU, T>& N = Matrix<NX, NU, T>{}
) {
    return design::discrete_lqr_from_continuous(A, B, Q, R, Ts, N);
}

/**
 * @brief Design discrete LQR from continuous state-space system via discretization
 * @note Alias for design::discrete_lqr_from_continuous. Compare with MATLAB's lqrd(sys, Q, R, Ts).
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = 0, size_t NV = 0, typename T = double>
[[nodiscard]] constexpr auto lqrd(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NX, NX, T>&                 Q,
    const Matrix<NU, NU, T>&                 R,
    T                                        Ts,
    const Matrix<NX, NU, T>&                 N = Matrix<NX, NU, T>{}
) {
    return design::discrete_lqr_from_continuous(sys, Q, R, Ts, N);
}

/**
 * @brief Linear-Quadratic Integral design for tracking
 * @note Alias for design::discrete_lqi. Compare with MATLAB's lqi(sys, Q, R).
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = 0, size_t NV = 0, typename T = double>
[[nodiscard]] constexpr auto lqi(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NX + NY, NX + NY, T>&       Q,
    const Matrix<NU, NU, T>&                 R
) {
    return design::discrete_lqi(sys, Q, R);
}

/**
 * @brief Linear-Quadratic-Gaussian regulator design
 * @note Alias for design::discrete_lqg. Compare with MATLAB's lqg(sys, ...).
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = 0, size_t NV = 0, typename T = double>
[[nodiscard]] constexpr auto lqg(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NX, NX, T>&                 Q_lqr,
    const Matrix<NU, NU, T>&                 R_lqr,
    const Matrix<NW, NW, T>&                 Q_kf,
    const Matrix<NV, NV, T>&                 R_kf,
    const Matrix<NX, NU, T>&                 N = Matrix<NX, NU, T>{}
) {
    return design::discrete_lqg(sys, Q_lqr, R_lqr, Q_kf, R_kf, N);
}

/**
 * @brief Combine separate Kalman filter and LQR designs into an LQG controller
 * @note Alias for design::lqg_from_parts. Compare with MATLAB's lqgreg(kest, k).
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
[[nodiscard]] constexpr auto lqgreg(
    const design::KalmanResult<NX, NU, NY, NW, NV, T>& kest,
    const design::LQRResult<NX, NU, T>&                lqr_result
) {
    return design::lqg_from_parts(kest, lqr_result);
}

/**
 * @brief Linear-Quadratic-Gaussian design with integral action for tracking
 * @note Alias for design::discrete_lqgi. Compare with MATLAB's lqgtrack(...).
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = 0, size_t NV = 0, typename T = double>
[[nodiscard]] constexpr auto lqgtrack(
    const StateSpace<NX, NU, NY, NW, NV, T>& sys,
    const Matrix<NX + NY, NX + NY, T>&       Q_aug,
    const Matrix<NU, NU, T>&                 R,
    const Matrix<NW, NW, T>&                 Q_kf,
    const Matrix<NV, NV, T>&                 R_kf
) {
    return design::discrete_lqgi(sys, Q_aug, R, Q_kf, R_kf);
}

/**
 * @brief PID controller tuning using frequency domain method
 *
 * Tunes a PID controller for a given plant to achieve a specified crossover frequency wc.
 * Uses the method similar to MATLAB's pidtune, aiming for 60 degrees phase margin.
 *
 * @param sys Plant state-space system (SISO, continuous-time)
 * @param wc Desired crossover frequency (rad/s)
 * @return PIDResult with tuned gains
 */
template<size_t NX, typename T = double>
[[nodiscard]] constexpr auto pidtune(const StateSpace<NX, 1, 1, 0, 0, T>& sys, T wc) noexcept {
    using Cplx = wet::complex<T>;
    constexpr T pi = wet::numbers::pi_v<T>;
    Cplx        jwc{0, wc};
    auto        G_frf = eval_frf(sys, jwc);
    // Assume SISO
    Cplx G = G_frf(0, 0);
    T    mag_G = wet::abs(G);
    T    arg_G = wet::arg(G);
    // Desired phase margin: 60 degrees = pi/3 radians
    T desired_phase = -pi + pi / 3 - arg_G;
    T mag_C = 1 / mag_G;
    T real_C = mag_C * wet::cos(desired_phase);
    T imag_C = mag_C * wet::sin(desired_phase);
    T Kp = real_C;
    T Ki = -imag_C * wc; // imag_C = -Ki/wc for PI
    // For PID, set Td = Ti/4
    T Ti = Kp / Ki;
    T Td = Ti / 4;
    T Kd = Kp * Td;
    T Kbc = Ki; // Back-calculation gain

    return wet::design::PIDResult<T>{
        Kp, Ki, Kd, T{0},
        -std::numeric_limits<T>::max(), std::numeric_limits<T>::max(),
        -std::numeric_limits<T>::max(), std::numeric_limits<T>::max(),
        Kbc
    };
}

// ===========================================================================
// Frequency-domain analysis — MATLAB spellings over wet::analysis
// ===========================================================================

// These already carry their MATLAB names in analysis::; surface them under
// matlab:: too so a MATLAB-style call site finds them in one namespace.
using analysis::bode;
using analysis::damp;
using analysis::dcgain;
using analysis::impulse;
using analysis::initial;
using analysis::linspace;
using analysis::logspace;
using analysis::nyquist;
using analysis::step;

/**
 * @brief MATLAB short alias for the open-loop poles of a system.
 * @note Compare with MATLAB's p = pole(sys). (analysis names it `poles`.)
 */
template<size_t NX, typename T = double>
[[nodiscard]] constexpr ColVec<NX, wet::complex<T>> pole(const Matrix<NX, NX, T>& A) {
    return analysis::poles(A);
}

/**
 * @brief Gain/phase margins and their crossover frequencies.
 *
 * @note Mirrors MATLAB's [Gm, Pm, Wcg, Wcp] = margin(...). Gm is a linear
 *       ratio (not dB), matching MATLAB; +inf marks a missing crossover.
 */
template<typename T = double>
struct MarginResult {
    T Gm{};  //!< Gain margin (linear ratio); +inf if phase never crosses -180°
    T Pm{};  //!< Phase margin (degrees); +inf if gain never crosses 0 dB
    T Wcg{}; //!< Gain-crossover frequency (rad/s, where phase = -180°)
    T Wcp{}; //!< Phase-crossover frequency (rad/s, where |G| = 1)
};

/**
 * @brief Gain and phase margins of a SISO loop over a frequency grid.
 *
 * Thin composition of analysis::bode and BodeResult::gain_margin /
 * phase_margin. Unlike MATLAB's margin(sys), the grid is explicit — pass
 * e.g. matlab::logspace(...) — since wet does no automatic frequency gridding.
 *
 * @param sys   SISO state-space loop (continuous or discrete)
 * @param omega Frequency grid (rad/s)
 * @return MarginResult{Gm, Pm, Wcg, Wcp}
 *
 * @note Compare with MATLAB's [Gm, Pm, Wcg, Wcp] = margin(sys, w).
 */
template<size_t NX, size_t NW, size_t NV, typename T>
[[nodiscard]] MarginResult<T> margin(const StateSpace<NX, 1, 1, NW, NV, T>& sys, const std::vector<T>& omega) {
    const auto      bode_data = analysis::bode(sys, omega);
    constexpr T     inf = std::numeric_limits<T>::max();
    MarginResult<T> m{inf, inf, T{0}, T{0}};
    if (const auto gm = bode_data.gain_margin()) {
        m.Gm = wet::pow(T{10}, gm->first / T{20}); // dB -> linear ratio
        m.Wcg = gm->second;
    }
    if (const auto pm = bode_data.phase_margin()) {
        m.Pm = pm->first;
        m.Wcp = pm->second;
    }
    return m;
}

/**
 * @brief -3 dB bandwidth of a SISO system over a frequency grid.
 *
 * @param sys   SISO state-space system
 * @param omega Frequency grid (rad/s), ascending from ~DC
 * @return Bandwidth (rad/s), or nullopt if the response never drops 3 dB.
 *
 * @note Compare with MATLAB's fb = bandwidth(sys). Grid is explicit here.
 */
template<size_t NX, size_t NW, size_t NV, typename T>
[[nodiscard]] wet::optional<T> bandwidth(const StateSpace<NX, 1, 1, NW, NV, T>& sys, const std::vector<T>& omega) {
    return analysis::bode(sys, omega).bandwidth();
}

} // namespace matlab

} // namespace wet