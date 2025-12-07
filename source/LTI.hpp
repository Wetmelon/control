#pragma once

#include <cmath>
#include <format>
#include <optional>

#include "Eigen/Dense"

namespace control {
using Matrix = Eigen::MatrixXd;

enum class DiscretizationMethod {
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
    RK45
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

struct IntegrationResult {
    Matrix x;
    double error;
};

class DiscreteStateSpace;    // Forward declaration
class ContinuousStateSpace;  // Forward declaration

template <typename Derived>
class StateSpaceBase {
   protected:
    StateSpaceBase(const Matrix& A, const Matrix& B, const Matrix& C, const Matrix& D)
        : A(A), B(B), C(C), D(D) {}

   public:
    Matrix output(const Matrix& x, const Matrix& u) const { return C * x + D * u; }

    FrequencyResponse bode(double fStart = 0.1, double fEnd = 100.0, int numFreq = 1000) const;
    StepResponse      step(double tStart = 0.0, double tEnd = 10.0, Matrix uStep = Matrix()) const;

    IntegrationResult evolve(const Matrix& x, const Matrix& u, double h) const {
        return static_cast<const Derived*>(this)->evolveImpl(x, u, h);
    }

    const Eigen::MatrixXd A = {}, B = {}, C = {}, D = {};
};

class ContinuousStateSpace : public StateSpaceBase<ContinuousStateSpace> {
   public:
    ContinuousStateSpace(const Matrix& A, const Matrix& B, const Matrix& C, const Matrix& D,
                         IntegrationMethod method = IntegrationMethod::RK45, std::optional<double> timestep = std::nullopt)
        : StateSpaceBase(A, B, C, D), integrationMethod(method), timestep(timestep) {}

    void setIntegrationMethod(IntegrationMethod method) { integrationMethod = method; }
    void setTimestep(std::optional<double> timestep) { this->timestep = timestep; }

    // Convert to discrete-time state space using specified method
    DiscreteStateSpace discretize(double Ts, DiscretizationMethod method = DiscretizationMethod::ZOH, std::optional<double> prewarp = std::nullopt) const;

   private:
    friend class StateSpaceBase<ContinuousStateSpace>;

    StepResponse      stepImpl(double tStart, double tEnd, Matrix uStep) const;
    IntegrationResult evolveImpl(const Matrix& x, const Matrix& u, double h) const;

    IntegrationResult evolveForwardEuler(const Matrix& x, const Matrix& u, double h) const;
    IntegrationResult evolveBackwardEuler(const Matrix& x, const Matrix& u, double h) const;
    IntegrationResult evolveTrapezoidal(const Matrix& x, const Matrix& u, double h) const;
    IntegrationResult evolveRK4(const Matrix& x, const Matrix& u, double h) const;
    IntegrationResult evolveRK45(const Matrix& x, const Matrix& u, double h) const;

    IntegrationMethod     integrationMethod;
    std::optional<double> timestep;
};

class DiscreteStateSpace : public StateSpaceBase<DiscreteStateSpace> {
   public:
    DiscreteStateSpace(const Matrix& A, const Matrix& B, const Matrix& C, const Matrix& D, double Ts)
        : StateSpaceBase(A, B, C, D), Ts(Ts) {}

   private:
    friend class StateSpaceBase<DiscreteStateSpace>;

    StepResponse      stepImpl(double tStart, double tEnd, Matrix uStep) const;
    IntegrationResult evolveImpl(const Matrix& x, const Matrix& u, double h) const;

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
