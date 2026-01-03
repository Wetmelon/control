#pragma once

#include <type_traits>

#include "matrix.hpp"

namespace wetmelon::control {
/**
 * @brief State-space representation for linear time-invariant systems (discrete or continuous)
 *
 * Fixed-size, stack-allocated state-space container supporting both continuous and discrete systems.
 * No heap allocation, suitable for embedded systems.
 *
 * Discrete-time (Ts > 0):   x_{k+1} = A x_k + B u_k + G w_k;  y_k = C x_k + D u_k + H v_k
 * Continuous-time (Ts = 0): dx/dt = A x + B u + G w;           y = C x + D u + H v
 *
 * @tparam NX Number of states
 * @tparam NU Number of control inputs
 * @tparam NY Number of outputs
 * @tparam NW Number of process noise inputs (default: NX)
 * @tparam NV Number of measurement noise inputs (default: NY)
 * @tparam T  Scalar type (default: double)
 */
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
    requires std::is_floating_point_v<T>
struct StateSpace {
    Matrix<NX, NX, T> A{};       //!< State dynamics matrix
    Matrix<NX, NU, T> B{};       //!< Control input matrix
    Matrix<NY, NX, T> C{};       //!< Output matrix
    Matrix<NY, NU, T> D{};       //!< Direct feedthrough matrix
    Matrix<NX, NW, T> G{};       //!< Process noise input matrix
    Matrix<NY, NV, T> H{};       //!< Measurement noise input matrix
    T                 Ts = T{0}; //!< Sampling period (0 for continuous, > 0 for discrete)

    template<typename U>
    [[nodiscard]] constexpr StateSpace<NX, NU, NY, NW, NV, U> as() const {
        return StateSpace<NX, NU, NY, NW, NV, U>{
            A.template as<U>(),
            B.template as<U>(),
            C.template as<U>(),
            D.template as<U>(),
            G.template as<U>(),
            H.template as<U>(),
            static_cast<U>(Ts)
        };
    }
};
} // namespace wetmelon::control
