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
};

enum class SystemType {
    Continuous,
    Discrete,
};

class DiscreteStateSpace; // Forward declaration

struct TransferFunction {
    Eigen::MatrixXd num, den;
};

struct FrequencyResponse {
    std::vector<double> freq;      // Frequency in Hz
    std::vector<double> magnitude; // Magnitude in dB
    std::vector<double> phase;     // Phase in degrees
};

template <typename Derived>
class StateSpaceBase {
  protected:
    StateSpaceBase(const Matrix &A, const Matrix &B, const Matrix &C, const Matrix &D)
        : A(A), B(B), C(C), D(D) {}

  public:
    Matrix output(const Matrix &x, const Matrix &u) const { return C * x + D * u; }

    FrequencyResponse generateFrequencyResponse(double fStart = 0.1, double fEnd = 100.0, int numFreq = 1000) const {
        auto response = FrequencyResponse{
            .freq      = std::vector<double>(numFreq),
            .magnitude = std::vector<double>(numFreq),
            .phase     = std::vector<double>(numFreq)};

        // Generate logarithmically spaced frequencies
        const auto [logStart, logEnd] = std::tuple{std::log10(fStart), std::log10(fEnd)};
        const auto logStep            = (logEnd - logStart) / (numFreq - 1);

        for (int i = 0; i < numFreq; ++i) {
            response.freq[i] = std::pow(10.0, logStart + i * logStep);    // Hz
            const auto w     = 2.0 * std::numbers::pi * response.freq[i]; // Convert to rad/s for calculations

            const auto s = std::complex<double>(0, w); // s = jω

            // Calculate transfer function H(s) = C(sI - A)^(-1)B + D
            const auto I = control::Matrix::Identity(A.rows(), A.cols());
            const auto H = (C * (((s * I) - A).inverse()) * B) + D;

            // Store magnitude in dB and phase in degrees
            response.magnitude[i] = 20.0 * std::log10(std::abs(H(0, 0)));
            response.phase[i]     = std::arg(H(0, 0)) * 180.0 / std::numbers::pi;
        }

        return response;
    }

    const Eigen::MatrixXd A = {}, B = {}, C = {}, D = {};
};

class ContinuousStateSpace : public StateSpaceBase<ContinuousStateSpace> {
  public:
    ContinuousStateSpace(const Matrix &A, const Matrix &B, const Matrix &C, const Matrix &D)
        : StateSpaceBase(A, B, C, D) {}

    // Integration methods for evolving the state
    Matrix evolve(const Matrix &x, const Matrix &u, double h, IntegrationMethod method) const;

    DiscreteStateSpace c2d(double Ts, Method method = Method::ZOH, std::optional<double> prewarp = std::nullopt) const;

  private:
    Matrix evolveForwardEuler(const Matrix &x, const Matrix &u, double h) const;
    Matrix evolveBackwardEuler(const Matrix &x, const Matrix &u, double h) const;
    Matrix evolveTrapezoidal(const Matrix &x, const Matrix &u, double h) const;
};

class DiscreteStateSpace : public StateSpaceBase<DiscreteStateSpace> {
  public:
    DiscreteStateSpace(const Matrix &A, const Matrix &B, const Matrix &C, const Matrix &D, double Ts)
        : StateSpaceBase(A, B, C, D), Ts(Ts) {}

    Matrix step(const Matrix &x, const Matrix &u) const { return A * x + B * u; }

    const double Ts;
};

// Backward compatibility typedef
using StateSpace = ContinuousStateSpace;

template <typename SystemType>
std::string formatStateSpaceMatrices(const SystemType &sys) {
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
}; // namespace control

template <>
struct std::formatter<control::ContinuousStateSpace> {
    constexpr auto parse(std::format_parse_context &ctx) {
        return ctx.begin();
    }

    auto format(const control::ContinuousStateSpace &sys, std::format_context &ctx) const {
        return std::format_to(ctx.out(), "{}", control::formatStateSpaceMatrices(sys));
    }
};

template <>
struct std::formatter<control::DiscreteStateSpace> {
    constexpr auto parse(std::format_parse_context &ctx) {
        return ctx.begin();
    }

    auto format(const control::DiscreteStateSpace &sys, std::format_context &ctx) const {
        return std::format_to(ctx.out(), "{}", control::formatStateSpaceMatrices(sys));
    }
};
