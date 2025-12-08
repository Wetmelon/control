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

    const Eigen::MatrixXd A = {}, B = {}, C = {}, D = {};
};

class ContinuousStateSpace : public StateSpaceBase<ContinuousStateSpace> {
   public:
    ContinuousStateSpace(const Matrix& A, const Matrix& B, const Matrix& C, const Matrix& D)
        : StateSpaceBase(A, B, C, D) {}

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
