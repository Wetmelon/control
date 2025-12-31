#include "../source/control.hpp"
#include "fmt/core.h"
#include "fmt/ranges.h"

int main() {
    using namespace control;

    fmt::print("=== LTI System Arithmetic Operations Demo ===\n\n");

    // Create a simple plant: G(s) = 1/(s+1)
    const StateSpace plant{
        Matrix::Constant(1, 1, -1.0),  // A
        Matrix::Constant(1, 1, 1.0),   // B
        Matrix::Constant(1, 1, 1.0),   // C
        Matrix::Constant(1, 1, 0.0)    // D
    };

    fmt::print("Plant G(s) = 1/(s+1):\n{}\n", plant);

    // Create a controller: C(s) = 2 (proportional gain)
    const StateSpace controller{
        Matrix::Zero(0, 0),          // A - No states (pure gain)
        Matrix::Zero(0, 1),          // B
        Matrix::Zero(1, 0),          // C
        Matrix::Constant(1, 1, 2.0)  // D
    };

    fmt::print("Controller C(s) = 2:\n{}\n", controller);

    // Create a sensor: H(s) = 1 (unity feedback)
    const StateSpace sensor{
        Matrix::Zero(0, 0),          // A
        Matrix::Zero(0, 1),          // B
        Matrix::Zero(1, 0),          // C
        Matrix::Constant(1, 1, 1.0)  // D
    };

    fmt::print("Sensor H(s) = 1:\n{}\n", sensor);

    // Demonstrate series connection
    fmt::print("\n=== Series Connection Example ===\n");
    fmt::print("Open-loop: L(s) = C(s) * G(s) = 2/(s+1)\n");
    auto open_loop = controller * plant;
    fmt::print("{}\n", open_loop);

    // Demonstrate feedback connection
    fmt::print("\n=== Feedback Connection Example ===\n");
    fmt::print("Closed-loop with unity feedback: T(s) = L(s) / (1 + L(s))\n");
    auto closed_loop = feedback(open_loop, sensor, -1);
    fmt::print("{}\n", closed_loop);

    // Demonstrate StateSpace to TransferFunction conversion
    fmt::print("\n=== StateSpace to TransferFunction Conversion ===\n");
    auto tf_plant = tf(plant);
    fmt::print("Plant as TF: num=[{}], den=[{}]\n",
               fmt::join(tf_plant.num, ", "),
               fmt::join(tf_plant.den, ", "));
    fmt::print("Expected: num=[1], den=[1, 1] representing 1/(s+1)\n");

    // Demonstrate MIMO extraction
    fmt::print("\n=== MIMO Transfer Function Extraction ===\n");
    StateSpace mimo_sys{
        Matrix{{-1.0, 0.0}, {0.0, -2.0}},  // A (diagonal)
        Matrix{{1.0, 0.0}, {0.0, 1.0}},    // B
        Matrix{{1.0, 0.0}, {0.0, 2.0}},    // C
        Matrix::Zero(2, 2)                 // D
    };

    fmt::print("MIMO system (2 inputs, 2 outputs) with diagonal structure\n");

    // Try SISO conversion on MIMO (should throw)
    try {
        auto tf_mimo = tf(mimo_sys);
        fmt::print("ERROR: Should have thrown exception!\n");
    } catch (const std::invalid_argument& e) {
        fmt::print("✓ SISO tf() correctly throws for MIMO: {}\n", e.what());
    }

    // Extract individual transfer functions
    auto tf_00 = tf(mimo_sys, 0, 0);
    auto tf_11 = tf(mimo_sys, 1, 1);
    fmt::print("G_00: num=[{}], den=[{}]\n",
               fmt::join(tf_00.num, ", "),
               fmt::join(tf_00.den, ", "));
    fmt::print("G_11: num=[{}], den=[{}]\n",
               fmt::join(tf_11.num, ", "),
               fmt::join(tf_11.den, ", "));

    // Generate step response of closed-loop system
    fmt::print("\n=== Step Response of Closed-Loop System ===\n");
    auto step_resp = closed_loop.step(0.0, 5.0);
    fmt::print("Time points: {}\n", step_resp.time.size());
    fmt::print("First few values:\n");
    for (size_t i = 0; i < std::min(size_t(10), step_resp.time.size()); ++i) {
        fmt::print("  t={:.3f}, y={:.4f}\n", step_resp.time[i], step_resp.output[i](0, 0));
    }

    fmt::print("\n=== Demo Complete ===\n");
    fmt::print("Note: Run test_lti_operations for comprehensive unit tests\n");

    fmt::print("is_stable checks:\n");
    fmt::print("  Plant is stable: {}\n", plant.is_stable() ? "Yes" : "No");
    fmt::print("  Closed-loop is stable: {}\n", closed_loop.is_stable() ? "Yes" : "No");
    return 0;
}
