#pragma once

#include <cmath>
#include <format>
#include <optional>
#include <sstream>

#include "Eigen/Dense"

namespace control {
using Matrix = Eigen::MatrixXd;

enum class Method {
    ZOH,
    FOH,
    Bilinear,
    Tustin,
};

struct TransferFunction {
    Eigen::MatrixXd num, den;
};

struct FrequencyResponse {
    std::vector<double> freq;      // Frequency in Hz
    std::vector<double> magnitude; // Magnitude in dB
    std::vector<double> phase;     // Phase in degrees
};

struct StateSpace {
    StateSpace(const Matrix &A, const Matrix &B, const Matrix &C, const Matrix &D,
               const std::optional<double> &Ts      = std::nullopt,
               const std::optional<Method> &method  = std::nullopt,
               const std::optional<double> &prewarp = std::nullopt)
        : A(A), B(B), C(C), D(D), Ts(Ts), method(method), prewarp(prewarp) {};

    Matrix step(const Matrix &x, const Matrix &u) const { return A * x + B * u; }
    Matrix output(const Matrix &x, const Matrix &u) const { return C * x + D * u; }

    StateSpace c2d(const double Ts, const Method method = Method::ZOH,
                   std::optional<double> prewarp = std::nullopt) const;

    FrequencyResponse generateFrequencyResponse(double fStart = 0.1, double fEnd = 100.0, int numFreq = 1000) const {
        auto response = FrequencyResponse{
            .freq      = std::vector<double>(numFreq),
            .magnitude = std::vector<double>(numFreq),
            .phase     = std::vector<double>(numFreq)};

        // Generate logarithmically spaced frequencies
        const auto [logStart, logEnd] = std::tuple{std::log10(fStart), std::log10(fEnd)};
        const auto logStep            = (logEnd - logStart) / (numFreq - 1);

        for (int i = 0; i < numFreq; ++i) {
            response.freq[i] = std::pow(10.0, logStart + i * logStep);      // Hz
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

    const std::optional<double> Ts      = std::nullopt;
    const std::optional<Method> method  = std::nullopt;
    const std::optional<double> prewarp = std::nullopt;

  private:
    friend std::ostream &operator<<(std::ostream &os, const StateSpace &sys) {
        os << "A = \n"
           << sys.A << '\n'
           << '\n';
        os << "B = \n"
           << sys.B << '\n'
           << '\n';
        os << "C = \n"
           << sys.C << '\n'
           << '\n';
        os << "D = \n"
           << sys.D << '\n'
           << '\n';

        return os;
    }
};
}; // namespace control

template <>
struct std::formatter<control::StateSpace> {
    constexpr auto parse(std::format_parse_context &ctx) {
        return ctx.begin();
    }

    auto format(const control::StateSpace &sys, std::format_context &ctx) const {
        std::stringstream ss;
        ss << sys;
        return std::format_to(ctx.out(), "{}", ss.str());
    }
};
