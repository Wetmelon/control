#pragma once

#include <cstddef>
#include <type_traits>

#include "state_space.hpp"
#include "transfer_function.hpp"
#include "wet/backend.hpp"

namespace wet {

/**
 * @brief Zero-pole-gain (ZPK) representation of a SISO LTI system
 *
 * Represents the factored transfer function
 * @f[
 *   H(s) = k \frac{\prod_{i} (s - z_i)}{\prod_{j} (s - p_j)}
 * @f]
 * with zeros @f$z_i@f$, poles @f$p_j@f$, and scalar gain @f$k@f$.
 *
 * MATLAB equivalent: `zpk(z, p, k)`.
 *
 * @tparam Nz   Number of zeros
 * @tparam Np   Number of poles
 * @tparam T    Floating-point scalar type
 */
template<size_t Nz, size_t Np, typename T = double>
    requires std::is_floating_point_v<T>
struct ZPK {
    wet::array<wet::complex<T>, Nz> zeros{}; //!< Zeros of the transfer function
    wet::array<wet::complex<T>, Np> poles{}; //!< Poles of the transfer function
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

} // namespace wet
