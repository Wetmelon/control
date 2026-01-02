#pragma once

#include "matrix.hpp"

// State-space container (discrete or continuous, no heap, fixed sizes)
// Discrete-time (Ts > 0):    x_{k+1} = A x_k + B u_k + G w_k;  y_k = C x_k + D u_k + H v_k
// Continuous-time (Ts = 0): dx/dt = A x + B u + G w;           y = C x + D u + H v
template<size_t NX, size_t NU, size_t NY, size_t NW = NX, size_t NV = NY, typename T = double>
struct StateSpace {
    Matrix<NX, NX, T> A{};       //< state dynamics matrix
    Matrix<NX, NU, T> B{};       //< control input matrix
    Matrix<NY, NX, T> C{};       //< output matrix
    Matrix<NY, NU, T> D{};       //< direct feedthrough
    Matrix<NX, NW, T> G{};       //< process noise matrix
    Matrix<NY, NV, T> H{};       //< measurement noise matrix
    T                 Ts = T{0}; //< sampling period (0 for continuous, > 0 for discrete)

    constexpr StateSpace() = default;

    constexpr StateSpace(
        const Matrix<NX, NX, T>& A_,
        const Matrix<NX, NU, T>& B_,
        const Matrix<NY, NX, T>& C_,
        const Matrix<NY, NU, T>& D_ = Matrix<NY, NU, T>{},
        const Matrix<NX, NW, T>& G_ = Matrix<NX, NW, T>{},
        const Matrix<NY, NV, T>& H_ = Matrix<NY, NV, T>{},
        T                        Ts_ = T{0}
    ) : A(A_), B(B_), C(C_), D(D_), G(G_), H(H_), Ts(Ts_) {}

    // Type conversion constructor (e.g., double -> float)
    template<typename U>
    constexpr StateSpace(const StateSpace<NX, NU, NY, NW, NV, U>& other)
        : A(other.A), B(other.B), C(other.C), D(other.D), G(other.G), H(other.H), Ts(static_cast<T>(other.Ts)) {}

    // State update: for discrete (Ts > 0) returns x[k+1]; for continuous (Ts = 0) returns dx/dt
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

    // Output: y = C x + D u + H v (same for both continuous and discrete)
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

    // Covariance propagation: P+ = A P A^T + G Q G^T
    [[nodiscard]] constexpr Matrix<NX, NX, T> propagate_P(
        const Matrix<NX, NX, T>& P,
        const Matrix<NW, NW, T>& Q
    ) const {
        return A * P * A.transpose() + G * Q * G.transpose();
    }

    // Innovation covariance: S = C P C^T + H R H^T
    template<size_t M = NY>
    [[nodiscard]] constexpr Matrix<M, M, T> innovation(
        const Matrix<NX, NX, T>& P,
        const Matrix<NV, NV, T>& R
    ) const {
        return C * P * C.transpose() + H * R * H.transpose();
    }
};

// Convenience helpers that mirror the previous free functions
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

template<size_t N, typename T = double>
constexpr ColVec<N, T> propagate_discrete(
    const Matrix<N, N, T>& A,
    const ColVec<N, T>&    x,
    const ColVec<N, T>&    w = ColVec<N, T>{}
) {
    return ColVec<N, T>(A * x + w);
}

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

template<size_t M, size_t N, typename T = double>
constexpr ColVec<M, T> project_output(
    const Matrix<M, N, T>& H,
    const ColVec<N, T>&    x,
    const ColVec<M, T>&    v = ColVec<M, T>{}
) {
    return ColVec<M, T>(H * x + v);
}
