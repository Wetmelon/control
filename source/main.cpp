
#include <print>

#include "LTI.hpp"

auto generateFrequencyResponse(const control::Matrix &A, const control::Matrix &B, const control::Matrix &C, const control::Matrix &D,
                               double fStart = 0.1, double fEnd = 100.0, int numFreq = 1000) {
    struct FrequencyResponse {
        std::vector<double> freq;      // Frequency in Hz
        std::vector<double> magnitude; // Magnitude in dB
        std::vector<double> phase;     // Phase in degrees
    };

    auto response = FrequencyResponse{
        .freq      = std::vector<double>(numFreq),
        .magnitude = std::vector<double>(numFreq),
        .phase     = std::vector<double>(numFreq)};

    // Generate logarithmically spaced frequencies
    const auto [logStart, logEnd] = std::tuple{std::log10(fStart), std::log10(fEnd)};
    const auto logStep            = (logEnd - logStart) / (numFreq - 1);

    for (int i = 0; i < numFreq; ++i) {
        response.freq[i] = std::pow(10, logStart + i * logStep);      // Hz
        const auto w     = 2.0 * std::numbers::pi * response.freq[i]; // Convert to rad/s for calculations

        // For each frequency, compute response using state space matrices
        const auto s = std::complex<double>(0, w); // s = jω, imaginary frequency point

        // Calculate transfer function H(s) = C(sI - A)^(-1)B + D
        const auto I = control::Matrix::Identity(A.rows(), A.cols());
        const auto H = (C * (((s * I) - A).inverse()) * B) + D;

        // Store magnitude in dB and phase in degrees
        response.magnitude[i] = 20 * std::log10(std::abs(H(0, 0)));
        response.phase[i]     = std::arg(H(0, 0)) * 180.0 / std::numbers::pi;
    }

    return response;
}

int main() {
    using namespace control;
    using namespace std::literals;

    constexpr auto m = 250.0; // system mass
    constexpr auto k = 40.0;  // spring constant
    constexpr auto b = 60.0;  // damping constant

    // Format styled after scipy.signal.StateSpace examples
    const auto A = Matrix{{0, 1.0}, {-k / m, -b / m}};
    const auto B = Matrix{{0}, {1.0 / m}};
    const auto C = Matrix{{1.0, 0}};
    const auto D = Matrix{{0.0}};

    const auto sys = StateSpace{A, B, C, D, 0.01, Method::Tustin};
    std::println("Continuous time matrix sys: \n{}", sys);

    const auto sysd = sys.c2d(0.01, Method::Tustin);
    std::println("Discrete time matrix sysd: \n{}", sysd);

    // Simulate step response
    const auto simTime   = 10.0; // 10 seconds simulation
    const auto numPoints = static_cast<int>(simTime / sys.Ts.value()) + 1;
    auto       time      = std::vector<double>(numPoints);
    auto       response  = std::vector<double>(numPoints);

    // Initialize step input and state
    Matrix input = Matrix::Ones(1, 1); // unit step
    Matrix state = Matrix::Zero(2, 1); // initial conditions [0; 0]

    // Simulate system
    for (int i = 0; i < numPoints; ++i) {
        time[i]     = i * sys.Ts.value();
        response[i] = sys.output(state, input)(0, 0); // get scalar output
        state       = sys.step(state, input);
    }

    // Add function call after simulation
    auto freqResp = generateFrequencyResponse(A, B, C, D);

    return 0;
}