#include <format>
#include <iostream>

#include "../source/control.hpp"

int main() {
    using namespace control;

    std::cout << "=== LTI System Arithmetic Operations Demo ===\n\n";

    // Create a simple plant: G(s) = 1/(s+1)
    const StateSpace plant{
        Matrix::Constant(1, 1, -1.0),  // A
        Matrix::Constant(1, 1, 1.0),   // B
        Matrix::Constant(1, 1, 1.0),   // C
        Matrix::Constant(1, 1, 0.0)    // D
    };

    std::cout << "Plant G(s) = 1/(s+1):\n"
              << std::format("{}\n", plant);

    // Create a controller: C(s) = 2 (proportional gain)
    const StateSpace controller{
        Matrix::Zero(0, 0),          // A - No states (pure gain)
        Matrix::Zero(0, 1),          // B
        Matrix::Zero(1, 0),          // C
        Matrix::Constant(1, 1, 2.0)  // D
    };

    std::cout << "Controller C(s) = 2:\n"
              << std::format("{}\n", controller);

    // Create a sensor: H(s) = 1 (unity feedback)
    const StateSpace sensor{
        Matrix::Zero(0, 0),          // A
        Matrix::Zero(0, 1),          // B
        Matrix::Zero(1, 0),          // C
        Matrix::Constant(1, 1, 1.0)  // D
    };

    std::cout << "Sensor H(s) = 1:\n"
              << std::format("{}\n", sensor);

    // Demonstrate series connection
    std::cout << "\n=== Series Connection Example ===\n";
    std::cout << "Open-loop: L(s) = C(s) * G(s) = 2/(s+1)\n";
    auto open_loop = controller * plant;
    std::cout << std::format("{}\n", open_loop);

    // Demonstrate feedback connection
    std::cout << "\n=== Feedback Connection Example ===\n";
    std::cout << "Closed-loop with unity feedback: T(s) = L(s) / (1 + L(s))\n";
    auto closed_loop = feedback(open_loop, sensor, -1);
    std::cout << std::format("{}\n", closed_loop);

    // Generate step response of closed-loop system
    std::cout << "\n=== Step Response of Closed-Loop System ===\n";
    auto step_resp = closed_loop.step(0.0, 5.0);
    std::cout << "Time points: " << step_resp.time.size() << "\n";
    std::cout << "First few values:\n";
    for (size_t i = 0; i < std::min(size_t(10), step_resp.time.size()); ++i) {
        std::cout << std::format("  t={:.3f}, y={:.4f}\n", step_resp.time[i], step_resp.output[i]);
    }

    std::cout << "\n=== Demo Complete ===\n";
    std::cout << "Note: Run test_lti_operations for comprehensive unit tests\n";
    return 0;
}
