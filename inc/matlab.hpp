#pragma once

#include <array>
#include <cstddef>
#include <optional>

#include "constexpr_complex.hpp"
#include "discretization.hpp"
#include "matrix.hpp"
#include "pid.hpp"
#include "state_space.hpp"

namespace wetmelon::control {
// MATLAB®-style matrix functions
namespace matlab {

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
[[nodiscard]] constexpr auto diag(const std::array<T, N>& diag) noexcept {
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
 * @param p Desired poles (accepts std::array<std::complex<T>, NX> or std::array<_Complex T, NX>)
 *
 * @return std::optional<Matrix<NU, NX, T>> State-feedback gain K, or nullopt if not implementable
 */
template<size_t NX, size_t NU, typename T = double>
constexpr std::optional<Matrix<NU, NX, T>> place(
    const Matrix<NX, NX, T>& A,
    const Matrix<NX, NU, T>& B,
    const auto&              p
) noexcept {
    // Convert poles to wet::complex<T> array
    const auto poles = std::apply([](const auto&... elems) { return std::array<wet::complex<T>, sizeof...(elems)>{elems...}; }, p);

    if constexpr (NU != 1) {
        // Multi-input pole placement not implemented yet
        return std::nullopt;
    } else {
        // Compute controllability matrix
        Matrix<NX, NX, T> Co;
        Matrix<NX, 1, T>  AB = B;
        for (size_t r = 0; r < NX; ++r) {
            Co(r, 0) = AB(r, 0);
        }
        for (size_t i = 1; i < NX; ++i) {
            AB = A * AB;
            for (size_t r = 0; r < NX; ++r) {
                Co(r, i) = AB(r, 0);
            }
        }

        // Check controllability by inverting Co
        auto Co_inv_opt = Co.inverse();
        if (!Co_inv_opt)
            return std::nullopt;
        auto Co_inv = *Co_inv_opt;

        // Compute desired characteristic polynomial coefficients (assuming real poles)
        std::array<T, NX + 1> coeffs{};
        coeffs[0] = 1.0;
        for (size_t i = 0; i < NX; ++i) {
            T root = poles[i].real(); // Use real part
            T temp = coeffs[0];
            coeffs[0] = -root * coeffs[0];
            for (size_t j = 1; j <= NX; ++j) {
                T next_temp = coeffs[j];
                coeffs[j] = temp - root * coeffs[j];
                temp = next_temp;
            }
        }

        // Compute phi_d(A) = sum_{k=0}^NX coeffs[k] * A^k
        Matrix<NX, NX, T> phi_A = Matrix<NX, NX, T>::zeros();
        Matrix<NX, NX, T> A_power = Matrix<NX, NX, T>::identity();
        for (size_t k = 0; k <= NX; ++k) {
            phi_A = phi_A + coeffs[k] * A_power;
            A_power = A_power * A;
        }

        // Compute K = e_N^T * Co^{-1} * phi_d(A)
        // e_N is [0, 0, ..., 1]^T
        Matrix<1, NX, T> e_N{};
        e_N(0, NX - 1) = 1.0;
        auto temp = e_N * Co_inv * phi_A;

        return temp;
    }
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
    using C = wet::complex<T>;
    constexpr T pi = std::numbers::pi_v<T>;
    C           jwc{0, wc};
    auto        G_frf = eval_frf(sys, jwc);
    // Assume SISO
    C G = G_frf(0, 0);
    T mag_G = wet::abs(G);
    T arg_G = wet::arg(G);
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
    return ::wetmelon::control::design::PIDResult<T>{
        Kp, Ki, Kd, T{0},
        -std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity(),
        -std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity(),
        Kbc
    };
}

} // namespace matlab

} // namespace wetmelon::control