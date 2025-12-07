#pragma once

#include <cmath>
#include <format>
#include <optional>

#include "Eigen/Dense"

namespace control {
using Matrix = Eigen::MatrixXd;

enum class Method {
    ZOH,
    FOH,
    Bilinear,
    Tustin,
};

enum class IntegrationMethod {
    ForwardEuler,
    BackwardEuler,
    Trapezoidal,
    RK4,
};

enum class SystemType {
    Continuous,
    Discrete,
};

struct TransferFunction {
    Eigen::MatrixXd num, den;
};

struct FrequencyResponse {
    std::vector<double> freq;       // Frequency in Hz
    std::vector<double> magnitude;  // Magnitude in dB
    std::vector<double> phase;      // Phase in degrees
};

struct StepResponse {
    std::vector<double> time;
    std::vector<double> output;
};

class DiscreteStateSpace;  // Forward declaration

template <typename Derived>
class StateSpaceBase {
   protected:
    StateSpaceBase(const Matrix& A, const Matrix& B, const Matrix& C, const Matrix& D)
        : A(A), B(B), C(C), D(D) {}

   public:
    Matrix output(const Matrix& x, const Matrix& u) const { return C * x + D * u; }

    auto generateFrequencyResponse(double fStart = 0.1, double fEnd = 100.0, int numFreq = 1000) const -> FrequencyResponse;
    auto generateStepResponse(double tStart = 0.0, double tEnd = 10.0, int numPoints = 1000,
                              Matrix x0 = Matrix(), Matrix uStep = Matrix(),
                              IntegrationMethod method = IntegrationMethod::RK4) const -> StepResponse;

    const Eigen::MatrixXd A = {}, B = {}, C = {}, D = {};
};

class ContinuousStateSpace : public StateSpaceBase<ContinuousStateSpace> {
   public:
    ContinuousStateSpace(const Matrix& A, const Matrix& B, const Matrix& C, const Matrix& D)
        : StateSpaceBase(A, B, C, D) {}

    // Integration methods for evolving the state
    Matrix evolve(const Matrix& x, const Matrix& u, double h, IntegrationMethod method) const;

    DiscreteStateSpace c2d(double Ts, Method method = Method::ZOH, std::optional<double> prewarp = std::nullopt) const;

   private:
    Matrix evolveForwardEuler(const Matrix& x, const Matrix& u, double h) const;
    Matrix evolveBackwardEuler(const Matrix& x, const Matrix& u, double h) const;
    Matrix evolveTrapezoidal(const Matrix& x, const Matrix& u, double h) const;
    Matrix evolveRK4(const Matrix& x, const Matrix& u, double h) const;
};

class DiscreteStateSpace : public StateSpaceBase<DiscreteStateSpace> {
   public:
    DiscreteStateSpace(const Matrix& A, const Matrix& B, const Matrix& C, const Matrix& D, double Ts)
        : StateSpaceBase(A, B, C, D), Ts(Ts) {}

    Matrix step(const Matrix& x, const Matrix& u) const { return A * x + B * u; }

    const double Ts;
};

// Backward compatibility typedef
using StateSpace = ContinuousStateSpace;

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
