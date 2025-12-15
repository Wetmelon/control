#include <format>
#include <iostream>

#include "../source/control.hpp"

int main() {
    using namespace control;

    std::cout << "=== LTI System Arithmetic Operations Demo ===\n\n";

    // Create a simple plant: G(s) = 1/(s+1)
    StateSpace plant{
        Matrix::Constant(1, 1, -1.0),  // A
        Matrix::Constant(1, 1, 1.0),   // B
        Matrix::Constant(1, 1, 1.0),   // C
        Matrix::Constant(1, 1, 0.0)    // D
    };

    std::cout << "Plant G(s) = 1/(s+1):\n"
              << std::format("{}\n", plant);

    // Create a controller: C(s) = 2 (proportional gain)
    StateSpace controller{
        Matrix::Zero(0, 0),          // A - No states (pure gain)
        Matrix::Zero(0, 1),          // B
        Matrix::Zero(1, 0),          // C
        Matrix::Constant(1, 1, 2.0)  // D
    };

    std::cout << "Controller C(s) = 2:\n"
              << std::format("{}\n", controller);

    // Create a sensor: H(s) = 1 (unity feedback)
    StateSpace sensor{
        Matrix::Zero(0, 0),          // A
        Matrix::Zero(0, 1),          // B
        Matrix::Zero(1, 0),          // C
        Matrix::Constant(1, 1, 1.0)  // D
    };

    std::cout << "Sensor H(s) = 1:\n"
              << std::format("{}\n", sensor);

    // Test 1: Series connection (Controller * Plant)
    std::cout << "\n=== Test 1: Series Connection ===\n";
    std::cout << "Open-loop: L(s) = C(s) * G(s) = 2/(s+1)\n";
    auto open_loop = controller * plant;
    std::cout << std::format("{}\n", open_loop);

    // Test 2: Parallel connection (Add two systems)
    std::cout << "\n=== Test 2: Parallel Connection ===\n";
    std::cout << "Sum: G1(s) + G2(s)\n";
    auto parallel_sum = plant + plant;  // Should give 2/(s+1)
    std::cout << std::format("{}\n", parallel_sum);

    // Test 3: Parallel connection with subtraction
    std::cout << "\n=== Test 3: Parallel Subtraction ===\n";
    std::cout << "Difference: G1(s) - G2(s)\n";
    auto parallel_diff = plant - plant;  // Should give 0
    std::cout << std::format("{}\n", parallel_diff);

    // Test 4: Feedback connection (Unity negative feedback)
    std::cout << "\n=== Test 4: Negative Feedback ===\n";
    std::cout << "Closed-loop with unity feedback: T(s) = L(s) / (1 + L(s))\n";
    auto closed_loop = feedback(open_loop, sensor, -1);
    std::cout << std::format("{}\n", closed_loop);

    // Test 5: Complete control system
    std::cout << "\n=== Test 5: Complete Control System ===\n";
    std::cout << "Controller -> Plant with Unity Feedback\n";
    std::cout << "T(s) = C(s)*G(s) / (1 + C(s)*G(s)*H(s))\n";
    auto control_system = feedback(controller * plant, sensor, -1);
    std::cout << std::format("{}\n", control_system);

    // Generate step response of closed-loop system
    std::cout << "\n=== Step Response of Closed-Loop System ===\n";
    auto step_resp = control_system.step(0.0, 5.0);
    std::cout << "Time points: " << step_resp.time.size() << "\n";
    std::cout << "First few values:\n";
    for (size_t i = 0; i < std::min(size_t(10), step_resp.time.size()); ++i) {
        std::cout << std::format("  t={:.3f}, y={:.4f}\n", step_resp.time[i], step_resp.output[i]);
    }

    std::cout << "\n=== Tests Complete ===\n";
    return 0;
}
