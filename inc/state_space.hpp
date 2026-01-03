#pragma once

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

    /**
     * @brief Predict next state (discrete) or state derivative (continuous)
     *
     * For discrete systems (Ts > 0): returns x[k+1] = A*x + B*u + G*w
     * For continuous systems (Ts = 0): returns dx/dt = A*x + B*u + G*w
     *
     * @param x Current state vector
     * @param u Control input vector (default: zero)
     * @param w Process noise vector (default: zero)
     *
     * @return Next state or state derivative
     */
    [[nodiscard]] constexpr ColVec<NX, T> predict_x(
        const ColVec<NX, T>& x,
        const ColVec<NU, T>& u = ColVec<NU, T>{},
        const ColVec<NW, T>& w = ColVec<NW, T>{}
    ) const {
        const auto ax = A * x;
        const auto bu = B * u;
        const auto gw = G * w;
        return ColVec<NX, T>(ax + bu + gw);
    }

    /**
     * @brief Compute system output
     *
     * Computes y = C*x + D*u + H*v (same for both continuous and discrete systems)
     *
     * @param x State vector
     * @param u Control input vector (default: zero)
     * @param v Measurement noise vector (default: zero)
     *
     * @return Output vector y
     */
    [[nodiscard]] constexpr ColVec<NY, T> predict_y(
        const ColVec<NX, T>& x,
        const ColVec<NU, T>& u = ColVec<NU, T>{},
        const ColVec<NV, T>& v = ColVec<NV, T>{}
    ) const {
        const auto cx = C * x;
        const auto du = D * u;
        const auto hv = H * v;
        return ColVec<NY, T>(cx + du + hv);
    }

    /**
     * @brief Propagate state covariance matrix
     *
     * Computes P⁺ = A*P*Aᵀ + G*Q*Gᵀ for Kalman filter prediction step
     *
     * @param P Current state covariance matrix
     * @param Q Process noise covariance matrix
     *
     * @return Predicted state covariance matrix
     */
    [[nodiscard]] constexpr Matrix<NX, NX, T> propagate_P(
        const Matrix<NX, NX, T>& P,
        const Matrix<NW, NW, T>& Q
    ) const {
        return A * P * A.transpose() + G * Q * G.transpose();
    }

    /**
     * @brief Compute innovation covariance matrix
     *
     * Computes S = C*P*Cᵀ + H*R*Hᵀ for Kalman filter update step
     *
     * @tparam M Output dimension (default: NY)
     * @param P  State covariance matrix
     * @param R  Measurement noise covariance matrix
     *
     * @return Innovation covariance matrix S
     */
    template<size_t M = NY>
    [[nodiscard]] constexpr Matrix<M, M, T> innovation(
        const Matrix<NX, NX, T>& P,
        const Matrix<NV, NV, T>& R
    ) const {
        return C * P * C.transpose() + H * R * H.transpose();
    }
};

/**
 * @brief Propagate discrete-time state with control input
 *
 * Computes x[k+1] = A*x + B*u + w
 *
 * @tparam N Number of states
 * @tparam U Number of control inputs
 * @tparam T Scalar type (default: double)
 * @param A State transition matrix
 * @param B Control input matrix
 * @param u Control input vector
 * @param x Current state vector
 * @param w Process noise vector (default: zero)
 *
 * @return Next state vector
 */
template<size_t N, size_t U, typename T = double>
constexpr ColVec<N, T> propagate_discrete(
    const Matrix<N, N, T>& A,
    const Matrix<N, U, T>& B,
    const ColVec<U, T>&    u,
    const ColVec<N, T>&    x,
    const ColVec<N, T>&    w = ColVec<N, T>{}
) {
    const auto ax = A * x;
    const auto bu = B * u;
    return ColVec<N, T>(ax + bu + w);
}

/**
 * @brief Propagate discrete-time state without control input
 *
 * Computes x[k+1] = A*x + w
 *
 * @tparam N Number of states
 * @tparam T Scalar type (default: double)
 * @param A State transition matrix
 * @param x Current state vector
 * @param w Process noise vector (default: zero)
 *
 * @return Next state vector
 */
template<size_t N, typename T = double>
constexpr ColVec<N, T> propagate_discrete(
    const Matrix<N, N, T>& A,
    const ColVec<N, T>&    x,
    const ColVec<N, T>&    w = ColVec<N, T>{}
) {
    return ColVec<N, T>(A * x + w);
}

/**
 * @brief Propagate covariance matrix for Kalman filter
 *
 * Computes P⁺ = F*P*Fᵀ + Q
 *
 * @tparam N State dimension
 * @tparam T Scalar type (default: double)
 * @param F State transition matrix (Jacobian)
 * @param P Current covariance matrix
 * @param Q Process noise covariance matrix
 *
 * @return Propagated covariance matrix
 */
template<size_t N, typename T = double>
constexpr Matrix<N, N, T> propagate_covariance(
    const Matrix<N, N, T>& F,
    const Matrix<N, N, T>& P,
    const Matrix<N, N, T>& Q
) {
    StateSpace<N, 0, 0, 0, 0, T> sys{};
    sys.A = F;
    sys.G = Matrix<N, N, T>::identity();
    return sys.propagate_P(P, Q);
}

/**
 * @brief Project state to measurement space
 *
 * Computes z = H*x + v
 *
 * @tparam M Measurement dimension
 * @tparam N State dimension
 * @tparam T Scalar type (default: double)
 * @param H Measurement matrix
 * @param x State vector
 * @param v Measurement noise vector (default: zero)
 *
 * @return Projected measurement vector
 */
template<size_t M, size_t N, typename T = double>
constexpr ColVec<M, T> project_output(
    const Matrix<M, N, T>& H,
    const ColVec<N, T>&    x,
    const ColVec<M, T>&    v = ColVec<M, T>{}
) {
    return ColVec<M, T>(H * x + v);
}
} // namespace wetmelon::control
