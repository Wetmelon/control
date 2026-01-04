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

    [[nodiscard]] constexpr bool is_discrete() const { return Ts > T{0}; }
    [[nodiscard]] constexpr bool is_continuous() const { return Ts == T{0}; }
};

/**
 * @brief Concept for StateSpace-like systems
 */
template<typename S>
concept StateSpaceSystem = requires(S s) {
    s.A;
    s.B;
    s.C;
    s.D;
    s.G;
    s.H;
    s.Ts;
    requires std::is_floating_point_v<decltype(s.Ts)>;
};

/**
 * @brief Series connection of two state-space systems
 *
 * Series connection: sys2 follows sys1 (sys1 -> sys2)
 *      State space representation:
 *      x = [x1; x2]
 *      A = [A1,    0  ]    B = [B1]
 *          [B2*C1, A2 ]        [B2*D1]
 *      C = [D2*C1, C2]    D = [D2*D1]
 *
 * @param sys1 First system
 * @param sys2 Second system
 *
 * @return Resulting state-space system from series connection
 */
constexpr auto series(const auto& sys1, const auto& sys2)
    requires StateSpaceSystem<decltype(sys1)>
          && StateSpaceSystem<decltype(sys2)>
          && std::same_as<decltype(sys1.Ts), decltype(sys2.Ts)>
          && (sys1.B.cols() == sys2.B.cols())
          && (sys1.C.rows() == sys2.C.rows())
{
    using T = decltype(sys1.Ts);

    constexpr size_t n1 = sys1.A.rows();
    constexpr size_t n2 = sys2.A.rows();
    constexpr size_t m = sys1.B.cols();
    constexpr size_t p = sys2.C.rows();
    constexpr size_t nw1 = sys1.G.cols();
    constexpr size_t nw2 = sys2.G.cols();
    constexpr size_t nv1 = sys1.H.cols();
    constexpr size_t nv2 = sys2.H.cols();

    Matrix<n1 + n2, n1 + n2, T>   A{};
    Matrix<n1 + n2, m, T>         B{};
    Matrix<p, n1 + n2, T>         C{};
    Matrix<p, m, T>               D{};
    Matrix<n1 + n2, nw1 + nw2, T> G{};
    Matrix<p, nv1 + nv2, T>       H{};

    // Fill A matrix
    A.template block<n1, n1>(0, 0) = sys1.A;
    A.template block<n2, n1>(n1, 0) = sys2.B * sys1.C;
    A.template block<n2, n2>(n1, n1) = sys2.A;

    // Fill B matrix
    B.template block<n1, m>(0, 0) = sys1.B;
    B.template block<n2, m>(n1, 0) = sys2.B * sys1.D;

    // Fill C matrix
    C.template block<p, n1>(0, 0) = sys2.D * sys1.C;
    C.template block<p, n2>(0, n1) = sys2.C;

    // Fill D matrix
    D = sys2.D * sys1.D;

    // Fill G matrix (process noise): [G1, 0; B2*H1, G2]
    G.template block<n1, nw1>(0, 0) = sys1.G;
    G.template block<n2, nw1>(n1, 0) = sys2.B * sys1.H;
    G.template block<n2, nw2>(n1, nw1) = sys2.G;

    // Fill H matrix (measurement noise): [D2*H1, H2]
    H.template block<p, nv1>(0, 0) = sys2.D * sys1.H;
    H.template block<p, nv2>(0, nv1) = sys2.H;

    return StateSpace<n1 + n2, m, p, nw1 + nw2, nv1 + nv2, T>{A, B, C, D, G, H, sys1.Ts};
}

/**
 * @brief Parallel connection of two state-space systems
 *
 * Parallel connection: outputs are summed (sys1 + sys2)
 *      State space representation:
 *      x = [x1; x2]
 *      A = [A1,  0 ]    B = [B1]
 *          [0,  A2 ]        [B2]
 *      C = [C1, C2]    D = [D1 + D2]
 *
 * @param sys1 First system
 * @param sys2 Second system
 *
 * @return Resulting state-space system from parallel connection
 */
constexpr auto parallel(const auto& sys1, const auto& sys2)
    requires StateSpaceSystem<decltype(sys1)>
          && StateSpaceSystem<decltype(sys2)>
          && std::same_as<decltype(sys1.Ts), decltype(sys2.Ts)>
          && (sys1.B.cols() == sys2.B.cols())
          && (sys1.C.rows() == sys2.C.rows())
{
    using T = decltype(sys1.Ts);

    constexpr size_t n1 = sys1.A.rows();
    constexpr size_t n2 = sys2.A.rows();
    constexpr size_t m = sys1.B.cols();
    constexpr size_t p = sys1.C.rows();
    constexpr size_t nw1 = sys1.G.cols();
    constexpr size_t nw2 = sys2.G.cols();
    constexpr size_t nv1 = sys1.H.cols();
    constexpr size_t nv2 = sys2.H.cols();

    Matrix<n1 + n2, n1 + n2, T>   A{};
    Matrix<n1 + n2, m, T>         B{};
    Matrix<p, n1 + n2, T>         C{};
    Matrix<p, m, T>               D{};
    Matrix<n1 + n2, nw1 + nw2, T> G{};
    Matrix<p, nv1 + nv2, T>       H{};

    // Fill A matrix (block diagonal)
    A.template block<n1, n1>(0, 0) = sys1.A;
    A.template block<n2, n2>(n1, n1) = sys2.A;

    // Fill B matrix (stacked)
    B.template block<n1, m>(0, 0) = sys1.B;
    B.template block<n2, m>(n1, 0) = sys2.B;

    // Fill C matrix (side-by-side)
    C.template block<p, n1>(0, 0) = sys1.C;
    C.template block<p, n2>(0, n1) = sys2.C;

    // Fill D matrix (summed)
    D = sys1.D + sys2.D;

    // Fill G matrix (block diagonal process noise): [G1, 0; 0, G2]
    G.template block<n1, nw1>(0, 0) = sys1.G;
    G.template block<n2, nw2>(n1, nw1) = sys2.G;

    // Fill H matrix (side-by-side measurement noise): [H1, H2]
    H.template block<p, nv1>(0, 0) = sys1.H;
    H.template block<p, nv2>(0, nv1) = sys2.H;

    return StateSpace<n1 + n2, m, p, nw1 + nw2, nv1 + nv2, T>{A, B, C, D, G, H, sys1.Ts};
}

/**
 * @brief Negative feedback connection of two state-space systems
 *
 * Negative feedback: y = sys1(u - sys2(y))
 *      State space representation:
 *      x = [x1; x2]
 *      A = [A1,            -B1*C2       ]    B = [B1*B2]
 *          [B2*C1, A2 - B2*D1*C2 ]        [0 ]
 *      C = [C1 - D1*C2,  -D1*C2]    D = [D1*D2]
 *
 * @param sys1 Forward system
 * @param sys2 Feedback system
 *
 * @return Resulting state-space system from negative feedback connection
 */
constexpr auto feedback(const auto& sys1, const auto& sys2)
    requires StateSpaceSystem<decltype(sys1)>
          && StateSpaceSystem<decltype(sys2)>
          && std::same_as<decltype(sys1.Ts), decltype(sys2.Ts)>
          && (sys1.B.cols() == sys2.B.cols())
          && (sys1.C.rows() == sys2.C.rows())
{
    using T = decltype(sys1.Ts);

    constexpr size_t n1 = sys1.A.rows();
    constexpr size_t n2 = sys2.A.rows();
    constexpr size_t m = sys1.B.cols();
    constexpr size_t p = sys1.C.rows();
    constexpr size_t nw1 = sys1.G.cols();
    constexpr size_t nw2 = sys2.G.cols();
    constexpr size_t nv1 = sys1.H.cols();
    constexpr size_t nv2 = sys2.H.cols();

    Matrix<n1 + n2, n1 + n2, T>   A{};
    Matrix<n1 + n2, m, T>         B{};
    Matrix<p, n1 + n2, T>         C{};
    Matrix<p, m, T>               D{};
    Matrix<n1 + n2, nw1 + nw2, T> G{};
    Matrix<p, nv1 + nv2, T>       H{};

    // Fill A matrix
    A.template block<n1, n1>(0, 0) = sys1.A;
    A.template block<n1, n2>(0, n1) = -sys1.B * sys2.C;
    A.template block<n2, n1>(n1, 0) = sys2.B * sys1.C;
    A.template block<n2, n2>(n1, n1) = sys2.A - sys2.B * sys1.D * sys2.C;

    // Fill B matrix
    B.template block<n1, m>(0, 0) = sys1.B * sys2.B;
    // B.template block<n2, m>(n1, 0) stays zero

    // Fill C matrix
    C.template block<p, n1>(0, 0) = sys1.C - sys1.D * sys2.C;
    C.template block<p, n2>(0, n1) = -sys1.D * sys2.C;

    // Fill D matrix
    D = sys1.D * sys2.B;

    // Fill G matrix (process noise): [G1, 0; B2*H1, G2]
    G.template block<n1, nw1>(0, 0) = sys1.G;
    G.template block<n2, nw1>(n1, 0) = sys2.B * sys1.H;
    G.template block<n2, nw2>(n1, nw1) = sys2.G;

    // Fill H matrix (measurement noise): [H1, H2]
    H.template block<p, nv1>(0, 0) = sys1.H;
    H.template block<p, nv2>(0, nv1) = sys2.H;

    return StateSpace<n1 + n2, m, p, nw1 + nw2, nv1 + nv2, T>{A, B, C, D, G, H, sys1.Ts};
}

/**
 * @brief Subtraction/differencing connection of two state-space systems
 *
 * Differencing connection: outputs are subtracted (sys1 - sys2)
 *      State space representation:
 *      x = [x1; x2]
 *      A = [A1,  0 ]    B = [B1]
 *          [0,  A2 ]        [B2]
 *      C = [C1, -C2]    D = [D1 - D2]
 *
 * @param sys1 First system
 * @param sys2 Second system (subtracted from sys1)
 *
 * @return Resulting state-space system from subtraction connection
 */
constexpr auto subtract(const auto& sys1, const auto& sys2)
    requires StateSpaceSystem<decltype(sys1)>
          && StateSpaceSystem<decltype(sys2)>
          && std::same_as<decltype(sys1.Ts), decltype(sys2.Ts)>
          && (sys1.B.cols() == sys2.B.cols())
          && (sys1.C.rows() == sys2.C.rows())
{
    using T = decltype(sys1.Ts);

    constexpr size_t n1 = sys1.A.rows();
    constexpr size_t n2 = sys2.A.rows();
    constexpr size_t m = sys1.B.cols();
    constexpr size_t p = sys1.C.rows();
    constexpr size_t nw1 = sys1.G.cols();
    constexpr size_t nw2 = sys2.G.cols();
    constexpr size_t nv1 = sys1.H.cols();
    constexpr size_t nv2 = sys2.H.cols();

    Matrix<n1 + n2, n1 + n2, T>   A{};
    Matrix<n1 + n2, m, T>         B{};
    Matrix<p, n1 + n2, T>         C{};
    Matrix<p, m, T>               D{};
    Matrix<n1 + n2, nw1 + nw2, T> G{};
    Matrix<p, nv1 + nv2, T>       H{};

    // Fill A matrix (block diagonal)
    A.template block<n1, n1>(0, 0) = sys1.A;
    A.template block<n2, n2>(n1, n1) = sys2.A;

    // Fill B matrix (stacked)
    B.template block<n1, m>(0, 0) = sys1.B;
    B.template block<n2, m>(n1, 0) = sys2.B;

    // Fill C matrix (sys1 output minus sys2 output)
    C.template block<p, n1>(0, 0) = sys1.C;
    C.template block<p, n2>(0, n1) = -sys2.C;

    // Fill D matrix (subtracted)
    D = sys1.D - sys2.D;

    // Fill G matrix (block diagonal process noise): [G1, 0; 0, G2]
    G.template block<n1, nw1>(0, 0) = sys1.G;
    G.template block<n2, nw2>(n1, nw1) = sys2.G;

    // Fill H matrix (side-by-side measurement noise): [H1, H2]
    H.template block<p, nv1>(0, 0) = sys1.H;
    H.template block<p, nv2>(0, nv1) = sys2.H;

    return StateSpace<n1 + n2, m, p, nw1 + nw2, nv1 + nv2, T>{A, B, C, D, G, H, sys1.Ts};
}

// Operator overloads for convenience
constexpr auto operator+(const auto& sys1, const auto& sys2)
    requires StateSpaceSystem<decltype(sys1)> && StateSpaceSystem<decltype(sys2)>
{
    return parallel(sys1, sys2);
}

constexpr auto operator-(const auto& sys1, const auto& sys2)
    requires StateSpaceSystem<decltype(sys1)> && StateSpaceSystem<decltype(sys2)>
{
    return subtract(sys1, sys2);
}

constexpr auto operator*(const auto& sys1, const auto& sys2)
    requires StateSpaceSystem<decltype(sys1)> && StateSpaceSystem<decltype(sys2)>
{
    return series(sys2, sys1); // sys1 * sys2 means sys1 feeds into sys2
}

constexpr auto operator/(const auto& sys1, const auto& sys2)
    requires StateSpaceSystem<decltype(sys1)> && StateSpaceSystem<decltype(sys2)>
{
    return feedback(sys1, sys2);
}
} // namespace wetmelon::control
