#pragma once

#include <array>
#include <cstddef>
#include <type_traits>

#include "state_space.hpp"

namespace wetmelon::control {
template<size_t Nnum, size_t Nden, typename T = double>
    requires std::is_floating_point_v<T>
struct TransferFunction {
    std::array<T, Nnum> num{}; //!< Numerator coefficients
    std::array<T, Nden> den{}; //!< Denominator coefficients

    template<typename U>
    [[nodiscard]] constexpr TransferFunction<Nnum, Nden, U> as() const {
        return TransferFunction<Nnum, Nden, U>{
            num.template as<U>(),
            den.template as<U>()
        };
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
            sys.C(0, k) = num[k] / den[Nden - 1];
        }

        // Construct D matrix
        sys.D(0, 0) = num[Nden - 1] / den[Nden - 1];

        return sys;
    }
};

template<size_t Nnum1, size_t Nden1, size_t Nnum2, size_t Nden2, typename T>
[[nodiscard]] constexpr auto operator*(
    const TransferFunction<Nnum1, Nden1, T>& tf1,
    const TransferFunction<Nnum2, Nden2, T>& tf2
) {
    constexpr size_t Nnum_res = Nnum1 + Nnum2 - 1;
    constexpr size_t Nden_res = Nden1 + Nden2 - 1;

    TransferFunction<Nnum_res, Nden_res, T> result{};

    // Convolve numerators
    for (size_t i = 0; i < Nnum1; ++i) {
        for (size_t j = 0; j < Nnum2; ++j) {
            result.num[i + j] += tf1.num[i] * tf2.num[j];
        }
    }

    // Convolve denominators
    for (size_t i = 0; i < Nden1; ++i) {
        for (size_t j = 0; j < Nden2; ++j) {
            result.den[i + j] += tf1.den[i] * tf2.den[j];
        }
    }

    return result;
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
