#pragma once

#include <array>
#include <cstddef>
#include <type_traits>

#include "state_space.hpp"

namespace wetmelon::control {

namespace tf_detail {

template<size_t Nres, size_t Nsrc, typename T>
constexpr void accumulate_poly(std::array<T, Nres>& dst, const std::array<T, Nsrc>& src, T scale = T{1}) {
    constexpr size_t N = (Nsrc < Nres) ? Nsrc : Nres;
    for (size_t i = 0; i < N; ++i) {
        dst[i] += scale * src[i];
    }
}

template<size_t Na, size_t Nb, typename T>
[[nodiscard]] constexpr std::array<T, Na + Nb - 1> convolve_poly(const std::array<T, Na>& a, const std::array<T, Nb>& b) {
    std::array<T, Na + Nb - 1> result{};
    for (size_t i = 0; i < Na; ++i) {
        for (size_t j = 0; j < Nb; ++j) {
            result[i + j] += a[i] * b[j];
        }
    }
    return result;
}

} // namespace tf_detail

template<typename TNum, typename TDen>
using transfer_function_scalar_t = std::conditional_t<
    std::is_floating_point_v<std::common_type_t<TNum, TDen>>,
    std::common_type_t<TNum, TDen>,
    double>;

template<size_t Nnum, size_t Nden, typename T = double>
    requires std::is_floating_point_v<T>
struct TransferFunction {
    std::array<T, Nnum> num{}; //!< Numerator coefficients
    std::array<T, Nden> den{}; //!< Denominator coefficients

    template<typename U>
    [[nodiscard]] constexpr TransferFunction<Nnum, Nden, U> as() const {
        TransferFunction<Nnum, Nden, U> result{};
        for (size_t i = 0; i < Nnum; ++i) {
            result.num[i] = static_cast<U>(num[i]);
        }
        for (size_t i = 0; i < Nden; ++i) {
            result.den[i] = static_cast<U>(den[i]);
        }
        return result;
    }

    /**
     * @brief Convert a SISO transfer function to state-space representation
     *
     * @return State-space representation of the transfer function
     */
    [[nodiscard]] constexpr StateSpace<Nden - 1, 1, 1, 0, 0, T> to_state_space() const {
        StateSpace<Nden - 1, 1, 1, 0, 0, T> sys{};

        // Construct A matrix (companion form)
        for (size_t i = 0; i < Nden - 2; ++i) {
            sys.A(i, i + 1) = T{1};
        }
        for (size_t j = 0; j < Nden - 1; ++j) {
            sys.A(Nden - 2, j) = -den[j] / den[Nden - 1];
        }

        // Construct B matrix
        sys.B(Nden - 2, 0) = T{1};

        // Construct C matrix
        for (size_t k = 0; k < Nden - 1; ++k) {
            T num_coeff = (k < Nnum) ? num[k] : T{0};
            sys.C(0, k) = num_coeff / den[Nden - 1];
        }

        // Construct D matrix
        T d_coeff = ((Nden - 1) < Nnum) ? num[Nden - 1] : T{0};
        sys.D(0, 0) = d_coeff / den[Nden - 1];

        return sys;
    }

    /**
     * @brief Implicit conversion to state-space representation
     */
    [[nodiscard]] constexpr operator StateSpace<Nden - 1, 1, 1, 0, 0, T>() const {
        return to_state_space();
    }
};

template<typename TNum, size_t Nnum, typename TDen, size_t Nden>
TransferFunction(const std::array<TNum, Nnum>&, const std::array<TDen, Nden>&)
    -> TransferFunction<Nnum, Nden, transfer_function_scalar_t<TNum, TDen>>;

template<typename TNum, size_t Nnum, typename TDen, size_t Nden>
TransferFunction(const TNum (&)[Nnum], const TDen (&)[Nden])
    -> TransferFunction<Nnum, Nden, transfer_function_scalar_t<TNum, TDen>>;

template<size_t Nnum1, size_t Nden1, size_t Nnum2, size_t Nden2, typename T>
[[nodiscard]] constexpr auto operator*(
    const TransferFunction<Nnum1, Nden1, T>& tf1,
    const TransferFunction<Nnum2, Nden2, T>& tf2
) {
    constexpr size_t Nnum_res = Nnum1 + Nnum2 - 1;
    constexpr size_t Nden_res = Nden1 + Nden2 - 1;

    TransferFunction<Nnum_res, Nden_res, T> result{};
    result.num = tf_detail::convolve_poly(tf1.num, tf2.num);
    result.den = tf_detail::convolve_poly(tf1.den, tf2.den);

    return result;
}

template<size_t Nnum1, size_t Nden1, size_t Nnum2, size_t Nden2, typename T>
[[nodiscard]] constexpr auto operator+(
    const TransferFunction<Nnum1, Nden1, T>& tf1,
    const TransferFunction<Nnum2, Nden2, T>& tf2
) {
    constexpr size_t Nnum_l = Nnum1 + Nden2 - 1;
    constexpr size_t Nnum_r = Nnum2 + Nden1 - 1;
    constexpr size_t Nnum_res = (Nnum_l > Nnum_r) ? Nnum_l : Nnum_r;
    constexpr size_t Nden_res = Nden1 + Nden2 - 1;

    TransferFunction<Nnum_res, Nden_res, T> result{};

    const auto left_num = tf_detail::convolve_poly(tf1.num, tf2.den);
    const auto right_num = tf_detail::convolve_poly(tf2.num, tf1.den);
    const auto den = tf_detail::convolve_poly(tf1.den, tf2.den);

    tf_detail::accumulate_poly(result.num, left_num);
    tf_detail::accumulate_poly(result.num, right_num);
    result.den = den;

    return result;
}

template<size_t Nnum1, size_t Nden1, size_t Nnum2, size_t Nden2, typename T>
[[nodiscard]] constexpr auto operator-(
    const TransferFunction<Nnum1, Nden1, T>& tf1,
    const TransferFunction<Nnum2, Nden2, T>& tf2
) {
    constexpr size_t Nnum_l = Nnum1 + Nden2 - 1;
    constexpr size_t Nnum_r = Nnum2 + Nden1 - 1;
    constexpr size_t Nnum_res = (Nnum_l > Nnum_r) ? Nnum_l : Nnum_r;
    constexpr size_t Nden_res = Nden1 + Nden2 - 1;

    TransferFunction<Nnum_res, Nden_res, T> result{};

    const auto left_num = tf_detail::convolve_poly(tf1.num, tf2.den);
    const auto right_num = tf_detail::convolve_poly(tf2.num, tf1.den);
    const auto den = tf_detail::convolve_poly(tf1.den, tf2.den);

    tf_detail::accumulate_poly(result.num, left_num);
    tf_detail::accumulate_poly(result.num, right_num, T{-1});
    result.den = den;

    return result;
}

template<size_t Nnum1, size_t Nden1, size_t Nnum2, size_t Nden2, typename T>
[[nodiscard]] constexpr auto series(
    const TransferFunction<Nnum1, Nden1, T>& tf1,
    const TransferFunction<Nnum2, Nden2, T>& tf2
) {
    return tf1 * tf2;
}

template<size_t Nnum1, size_t Nden1, size_t Nnum2, size_t Nden2, typename T>
[[nodiscard]] constexpr auto parallel(
    const TransferFunction<Nnum1, Nden1, T>& tf1,
    const TransferFunction<Nnum2, Nden2, T>& tf2
) {
    return tf1 + tf2;
}

template<size_t Nnum1, size_t Nden1, size_t Nnum2, size_t Nden2, typename T>
[[nodiscard]] constexpr auto subtract(
    const TransferFunction<Nnum1, Nden1, T>& tf1,
    const TransferFunction<Nnum2, Nden2, T>& tf2
) {
    return tf1 - tf2;
}

template<size_t Nnum1, size_t Nden1, size_t Nnum2, size_t Nden2, typename T>
[[nodiscard]] constexpr auto feedback(
    const TransferFunction<Nnum1, Nden1, T>& sys1,
    const TransferFunction<Nnum2, Nden2, T>& sys2
) {
    // Negative feedback: sys1 / (1 + sys1*sys2)
    constexpr size_t Nnum_res = Nnum1 + Nden2 - 1;
    constexpr size_t Nden_l = Nden1 + Nden2 - 1;
    constexpr size_t Nden_r = Nnum1 + Nnum2 - 1;
    constexpr size_t Nden_res = (Nden_l > Nden_r) ? Nden_l : Nden_r;

    TransferFunction<Nnum_res, Nden_res, T> result{};

    const auto num = tf_detail::convolve_poly(sys1.num, sys2.den);
    const auto den_l = tf_detail::convolve_poly(sys1.den, sys2.den);
    const auto den_r = tf_detail::convolve_poly(sys1.num, sys2.num);

    result.num = num;
    tf_detail::accumulate_poly(result.den, den_l);
    tf_detail::accumulate_poly(result.den, den_r);

    return result;
}

template<size_t Nnum1, size_t Nden1, size_t Nnum2, size_t Nden2, typename T>
[[nodiscard]] constexpr auto operator/(
    const TransferFunction<Nnum1, Nden1, T>& sys1,
    const TransferFunction<Nnum2, Nden2, T>& sys2
) {
    return feedback(sys1, sys2);
}

template<size_t Nz, size_t Np, typename T = double>
    requires std::is_floating_point_v<T>
struct ZPK {
    std::array<wet::complex<T>, Nz> zeros{}; //!< Zeros of the transfer function
    std::array<wet::complex<T>, Np> poles{}; //!< Poles of the transfer function
    T                               gain{1}; //!< Gain of the transfer function

    template<typename U>
    [[nodiscard]] constexpr ZPK<Nz, Np, U> as() const {
        ZPK<Nz, Np, U> result{};
        for (size_t i = 0; i < Nz; ++i) {
            result.zeros[i] = wet::complex<U>{static_cast<U>(zeros[i].real()), static_cast<U>(zeros[i].imag())};
        }
        for (size_t j = 0; j < Np; ++j) {
            result.poles[j] = wet::complex<U>{static_cast<U>(poles[j].real()), static_cast<U>(poles[j].imag())};
        }
        result.gain = static_cast<U>(gain);
        return result;
    }

    /**
     * @brief Convert ZPK representation to transfer function form
     *
     * @return  Transfer function representation of the ZPK system
     */
    [[nodiscard]] constexpr TransferFunction<Nz + 1, Np + 1, T> to_transfer_function() const {
        TransferFunction<Nz + 1, Np + 1, T> tf{};

        // Compute numerator coefficients from zeros
        tf.num[0] = gain;
        for (size_t i = 0; i < Nz; ++i) {
            for (size_t j = i + 1; j > 0; --j) {
                tf.num[j] -= zeros[i].real() * tf.num[j - 1];
            }
        }

        // Compute denominator coefficients from poles
        tf.den[0] = T{1};
        for (size_t i = 0; i < Np; ++i) {
            for (size_t j = i + 1; j > 0; --j) {
                tf.den[j] -= poles[i].real() * tf.den[j - 1];
            }
        }

        return tf;
    }

    /**
     * @brief Convert ZPK representation to state-space form
     *
     * @return State-space representation of the ZPK system
     */
    [[nodiscard]] constexpr StateSpace<Np, 1, 1, 0, 0, T> to_state_space() const {
        return to_transfer_function().to_state_space();
    }
};

} // namespace wetmelon::control
