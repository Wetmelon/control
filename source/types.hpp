#pragma once

#include <cmath>
#include <string>
#include <vector>

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
    RK45,
    Exact,
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

struct AdaptiveStepResult {
    Matrix x;
    double step_size;
};

struct SolveResult {
    std::vector<double> t;  // time points
    std::vector<Matrix> x;  // states (x) at each time point
    bool                success = true;
    std::string         message;
    size_t              nfev = 0;  // number of function evaluations
};

}  // namespace control
