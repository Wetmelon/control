#pragma once

#include <format>

#include "solver.hpp"
#include "types.hpp"

namespace control {

class DiscreteStateSpace;    // Forward declaration
class ContinuousStateSpace;  // Forward declaration
using StateSpace = ContinuousStateSpace;
template <typename Derived>
class StateSpaceBase {
   protected:
    StateSpaceBase(const Matrix& A, const Matrix& B, const Matrix& C, const Matrix& D)
        : A(A), B(B), C(C), D(D) {}

    StateSpaceBase(Matrix&& A, Matrix&& B, Matrix&& C, Matrix&& D)
        : A(std::move(A)), B(std::move(B)), C(std::move(C)), D(std::move(D)) {}

   public:
    Matrix output(const Matrix& x, const Matrix& u) const { return C * x + D * u; }

    StepResponse step(double tStart = 0.0, double tEnd = 10.0, Matrix uStep = Matrix()) const {
        if (uStep.size() == 0) {
            uStep = Matrix::Ones(B.cols(), 1);
        }

        return static_cast<const Derived*>(this)->stepImpl(tStart, tEnd, uStep);
    };

    FrequencyResponse bode(double fStart = 0.1, double fEnd = 1.0e4, size_t numFreq = 1000) const {
        return static_cast<const Derived*>(this)->bodeImpl(fStart, fEnd, numFreq);
    }

    const Matrix A = {}, B = {}, C = {}, D = {};
};

class ContinuousStateSpace : public StateSpaceBase<ContinuousStateSpace> {
   public:
    ContinuousStateSpace(const Matrix& A, const Matrix& B, const Matrix& C, const Matrix& D)
        : StateSpaceBase(A, B, C, D) {}

    ContinuousStateSpace(Matrix&& A, Matrix&& B, Matrix&& C, Matrix&& D)
        : StateSpaceBase(std::move(A), std::move(B), std::move(C), std::move(D)) {}

    // Convert to discrete-time state space using specified method
    DiscreteStateSpace discretize(double Ts, DiscretizationMethod method = DiscretizationMethod::ZOH, std::optional<double> prewarp = std::nullopt) const;

   private:
    friend class StateSpaceBase<ContinuousStateSpace>;

    StepResponse      stepImpl(double tStart, double tEnd, Matrix uStep) const;
    FrequencyResponse bodeImpl(double fStart, double fEnd, size_t numFreq) const;
};

class DiscreteStateSpace : public StateSpaceBase<DiscreteStateSpace> {
   public:
    DiscreteStateSpace(const Matrix& A, const Matrix& B, const Matrix& C, const Matrix& D, double Ts)
        : StateSpaceBase(A, B, C, D), Ts(Ts) {}

    DiscreteStateSpace(Matrix&& A, Matrix&& B, Matrix&& C, Matrix&& D, double Ts)
        : StateSpaceBase(std::move(A), std::move(B), std::move(C), std::move(D)), Ts(Ts) {}

   private:
    friend class StateSpaceBase<DiscreteStateSpace>;

    StepResponse      stepImpl(double tStart, double tEnd, Matrix uStep) const;
    FrequencyResponse bodeImpl(double fStart, double fEnd, size_t numFreq) const;

    const double Ts;
};

template <class Derived>
Derived ss(const Matrix& A, const Matrix& B, const Matrix& C, const Matrix& D, std::optional<double> Ts = std::nullopt) {
    if (Ts.has_value()) {
        return Derived{A, B, C, D, Ts.value()};
    } else {
        return Derived{A, B, C, D};
    }
}

template <class Derived>
Derived ss(const TransferFunction& tf) {
    // Convert transfer function to state space using controllable canonical form
    const int n = tf.den.cols() - 1;  // Order of the system
    const int m = tf.num.cols() - 1;  // Order of the numerator

    Matrix A = Matrix::Zero(n, n);
    Matrix B = Matrix::Zero(n, 1);
    Matrix C = Matrix::Zero(1, n);
    Matrix D = Matrix::Zero(1, 1);

    // Fill A matrix
    for (int i = 0; i < n - 1; ++i) {
        A(i, i + 1) = 1.0;
    }
    for (int i = 0; i < n; ++i) {
        A(n - 1, i) = -tf.den(0, i) / tf.den(0, n);
    }

    // Fill B matrix
    B(n - 1, 0) = 1.0;

    // Fill C and D matrices
    for (int i = 0; i <= m; ++i) {
        C(0, i) = tf.num(0, i) / tf.den(0, n);
    }
    if (m < n) {
        D(0, 0) = 0.0;
    } else {
        D(0, 0) = tf.num(0, m) / tf.den(0, n);
    }

    return ss<Derived>(A, B, C, D);
}

template <typename SystemType>
auto c2d(const SystemType& sys, double Ts, DiscretizationMethod method = DiscretizationMethod::ZOH, std::optional<double> prewarp = std::nullopt) {
    return sys.discretize(Ts, method, prewarp);
}

template <typename SystemType>
std::string formatStateSpaceMatrices(const SystemType& sys) {
    std::string result = "A = \n";
    for (int i = 0; i < sys.A.rows(); ++i) {
        for (int j = 0; j < sys.A.cols(); ++j) {
            result += std::format("{:>10.4f}", sys.A(i, j));
        }
        result += "\n";
    }
    result += "\nB = \n";
    for (int i = 0; i < sys.B.rows(); ++i) {
        for (int j = 0; j < sys.B.cols(); ++j) {
            result += std::format("{:>10.4f}", sys.B(i, j));
        }
        result += "\n";
    }
    result += "\nC = \n";
    for (int i = 0; i < sys.C.rows(); ++i) {
        for (int j = 0; j < sys.C.cols(); ++j) {
            result += std::format("{:>10.4f}", sys.C(i, j));
        }
        result += "\n";
    }
    result += "\nD = \n";
    for (int i = 0; i < sys.D.rows(); ++i) {
        for (int j = 0; j < sys.D.cols(); ++j) {
            result += std::format("{:>10.4f}", sys.D(i, j));
        }
        result += "\n";
    }
    return result;
}

// ============================================================================
// LTI System Arithmetic Operations
// ============================================================================
// These operators enable combining LTI systems to create complex control
// systems from simple building blocks (controllers, plants, sensors).
//
// Usage Examples:
//   auto open_loop = controller * plant;           // Series connection
//   auto parallel_sys = sys1 + sys2;               // Parallel (sum)
//   auto error_sys = reference - measurement;      // Parallel (difference)
//   auto closed_loop = feedback(fwd_path, fb_path); // Negative feedback
//
// Control System Construction:
//   1. Create individual components (Controller C, Plant G, Sensor H)
//   2. Combine them: T = feedback(C * G, H)
//   3. This creates closed-loop: T(s) = C*G / (1 + C*G*H)
// ============================================================================

// Series connection operator: sys1 * sys2
// Connects output of sys1 to input of sys2
template <typename Derived1, typename Derived2>
Derived1 operator*(const StateSpaceBase<Derived1>& sys1, const StateSpaceBase<Derived2>& sys2) {
    static_assert(std::is_same_v<Derived1, Derived2>,
                  "Cannot combine continuous and discrete systems. Use discretize() or c2d() first.");

    // Series connection: sys2 follows sys1 (sys1 -> sys2)
    // State space representation:
    // x = [x1; x2]
    // A = [A1,    0  ]    B = [B1]
    //     [B2*C1, A2 ]        [B2*D1]
    // C = [D2*C1, C2]    D = [D2*D1]

    const int n1 = sys1.A.rows();
    const int n2 = sys2.A.rows();
    const int m  = sys1.B.cols();
    const int p  = sys2.C.rows();

    Matrix A = Matrix::Zero(n1 + n2, n1 + n2);
    Matrix B = Matrix::Zero(n1 + n2, m);
    Matrix C = Matrix::Zero(p, n1 + n2);
    Matrix D = Matrix::Zero(p, m);

    // Fill A matrix
    A.block(0, 0, n1, n1)   = sys1.A;
    A.block(n1, 0, n2, n1)  = sys2.B * sys1.C;
    A.block(n1, n1, n2, n2) = sys2.A;

    // Fill B matrix
    B.block(0, 0, n1, m)  = sys1.B;
    B.block(n1, 0, n2, m) = sys2.B * sys1.D;

    // Fill C matrix
    C.block(0, 0, p, n1)  = sys2.D * sys1.C;
    C.block(0, n1, p, n2) = sys2.C;

    // Fill D matrix
    D = sys2.D * sys1.D;

    return Derived1{A, B, C, D};
}

// Parallel connection operator: sys1 + sys2
// Outputs are summed
template <typename Derived1, typename Derived2>
Derived1 operator+(const StateSpaceBase<Derived1>& sys1, const StateSpaceBase<Derived2>& sys2) {
    static_assert(std::is_same_v<Derived1, Derived2>,
                  "Cannot combine continuous and discrete systems. Use discretize() or c2d() first.");

    // Parallel connection: outputs are added
    // State space representation:
    // x = [x1; x2]
    // A = [A1, 0 ]    B = [B1]
    //     [0,  A2]        [B2]
    // C = [C1, C2]    D = [D1 + D2]

    const int n1 = sys1.A.rows();
    const int n2 = sys2.A.rows();
    const int m  = sys1.B.cols();
    const int p  = sys1.C.rows();

    Matrix A = Matrix::Zero(n1 + n2, n1 + n2);
    Matrix B = Matrix::Zero(n1 + n2, m);
    Matrix C = Matrix::Zero(p, n1 + n2);
    Matrix D = Matrix::Zero(p, m);

    // Fill A matrix (block diagonal)
    A.block(0, 0, n1, n1)   = sys1.A;
    A.block(n1, n1, n2, n2) = sys2.A;

    // Fill B matrix (stacked)
    B.block(0, 0, n1, m)  = sys1.B;
    B.block(n1, 0, n2, m) = sys2.B;

    // Fill C matrix (concatenated)
    C.block(0, 0, p, n1)  = sys1.C;
    C.block(0, n1, p, n2) = sys2.C;

    // Fill D matrix (sum)
    D = sys1.D + sys2.D;

    return Derived1{A, B, C, D};
}

// Parallel connection operator: sys1 - sys2
// Output of sys2 is subtracted from sys1
template <typename Derived1, typename Derived2>
Derived1 operator-(const StateSpaceBase<Derived1>& sys1, const StateSpaceBase<Derived2>& sys2) {
    static_assert(std::is_same_v<Derived1, Derived2>,
                  "Cannot combine continuous and discrete systems. Use discretize() or c2d() first.");

    // Parallel connection with subtraction: outputs are subtracted
    // State space representation:
    // x = [x1; x2]
    // A = [A1, 0 ]    B = [B1]
    //     [0,  A2]        [B2]
    // C = [C1, -C2]   D = [D1 - D2]

    const int n1 = sys1.A.rows();
    const int n2 = sys2.A.rows();
    const int m  = sys1.B.cols();
    const int p  = sys1.C.rows();

    Matrix A = Matrix::Zero(n1 + n2, n1 + n2);
    Matrix B = Matrix::Zero(n1 + n2, m);
    Matrix C = Matrix::Zero(p, n1 + n2);
    Matrix D = Matrix::Zero(p, m);

    // Fill A matrix (block diagonal)
    A.block(0, 0, n1, n1)   = sys1.A;
    A.block(n1, n1, n2, n2) = sys2.A;

    // Fill B matrix (stacked)
    B.block(0, 0, n1, m)  = sys1.B;
    B.block(n1, 0, n2, m) = sys2.B;

    // Fill C matrix (concatenated with negation)
    C.block(0, 0, p, n1)  = sys1.C;
    C.block(0, n1, p, n2) = -sys2.C;

    // Fill D matrix (difference)
    D = sys1.D - sys2.D;

    return Derived1{A, B, C, D};
}

// Feedback connection: feedback(sys_forward, sys_feedback, sign)
// Creates closed-loop system with feedback
// If sign = +1: positive feedback, if sign = -1: negative feedback (default)
template <typename Derived1, typename Derived2>
Derived1 feedback(const StateSpaceBase<Derived1>& sys_forward, const StateSpaceBase<Derived2>& sys_feedback, int sign = -1) {
    static_assert(std::is_same_v<Derived1, Derived2>,
                  "Cannot combine continuous and discrete systems. Use discretize() or c2d() first.");

    // Feedback connection:
    // Closed-loop transfer function: G_cl = G / (1 - sign*G*H)
    // where G is forward path, H is feedback path
    // sign = -1 for negative feedback (default), +1 for positive feedback
    //
    // State space representation:
    // A_cl = [A_G + sign*B_G*(I - sign*D_H*D_G)^-1*D_H*C_G,  sign*B_G*(I - sign*D_H*D_G)^-1*C_H]
    //        [B_H*(I - sign*D_G*D_H)^-1*C_G,                   A_H + sign*B_H*(I - sign*D_G*D_H)^-1*D_G*C_H]
    // B_cl = [B_G*(I - sign*D_H*D_G)^-1]
    //        [sign*B_H*(I - sign*D_G*D_H)^-1*D_G]
    // C_cl = [(I - sign*D_H*D_G)^-1*C_G,  sign*(I - sign*D_H*D_G)^-1*D_H*C_H]
    // D_cl = (I - sign*D_H*D_G)^-1*D_G

    const auto& G = sys_forward;
    const auto& H = sys_feedback;

    const int nG = G.A.rows();
    const int nH = H.A.rows();
    const int m  = G.B.cols();
    const int p  = G.C.rows();

    // Calculate the inverses we need
    const auto   I_p = Matrix::Identity(p, p);
    const auto   I_m = Matrix::Identity(m, m);
    const double s   = static_cast<double>(sign);

    // (I - sign*D_H*D_G)^-1
    const auto inv1 = (I_p - s * H.D * G.D).inverse();
    // (I - sign*D_G*D_H)^-1
    const auto inv2 = (I_m - s * G.D * H.D).inverse();

    Matrix A_cl = Matrix::Zero(nG + nH, nG + nH);
    Matrix B_cl = Matrix::Zero(nG + nH, m);
    Matrix C_cl = Matrix::Zero(p, nG + nH);
    Matrix D_cl = Matrix::Zero(p, m);

    // Fill A_cl matrix
    A_cl.block(0, 0, nG, nG)   = G.A + s * G.B * inv1 * H.D * G.C;
    A_cl.block(0, nG, nG, nH)  = s * G.B * inv1 * H.C;
    A_cl.block(nG, 0, nH, nG)  = H.B * inv2 * G.C;
    A_cl.block(nG, nG, nH, nH) = H.A + s * H.B * inv2 * G.D * H.C;

    // Fill B_cl matrix
    B_cl.block(0, 0, nG, m)  = G.B * inv1;
    B_cl.block(nG, 0, nH, m) = s * H.B * inv2 * G.D;

    // Fill C_cl matrix
    C_cl.block(0, 0, p, nG)  = inv1 * G.C;
    C_cl.block(0, nG, p, nH) = s * inv1 * H.D * H.C;

    // Fill D_cl matrix
    D_cl = inv1 * G.D;

    return Derived1{A_cl, B_cl, C_cl, D_cl};
}
};  // namespace control

template <>
struct std::formatter<control::ContinuousStateSpace> {
    constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }

    auto format(const control::ContinuousStateSpace& sys, std::format_context& ctx) const {
        return std::format_to(ctx.out(), "{}", control::formatStateSpaceMatrices(sys));
    }
};

template <>
struct std::formatter<control::DiscreteStateSpace> {
    constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }

    auto format(const control::DiscreteStateSpace& sys, std::format_context& ctx) const {
        return std::format_to(ctx.out(), "{}", control::formatStateSpaceMatrices(sys));
    }
};
