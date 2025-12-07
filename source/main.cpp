
#include <print>

#include "LTI.hpp"

namespace control {

} // namespace control

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

    auto time     = std::vector<double>(numPoints);
    auto response = std::vector<double>(numPoints);

    // Initialize step input and state
    Matrix input = Matrix::Ones(1, 1); // unit step
    Matrix state = Matrix::Zero(2, 1); // initial conditions [0; 0]

    // Simulate system
    for (int i = 0; i < numPoints; ++i) {
        time[i]     = i * sys.Ts.value();
        response[i] = sys.output(state, input)(0, 0); // get scalar output
        state       = sys.step(state, input);
    }

    const auto freqResp = sys.generateFrequencyResponse();
    
    // Plot the frequency response
    plotFrequencyResponse(freqResp);

    return 0;
}