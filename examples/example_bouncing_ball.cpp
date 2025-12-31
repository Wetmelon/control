#include <vector>

#include "control.hpp"
#include "matplot/matplot.h"
#include "solver.hpp"

int main() {
    using namespace control;
    namespace plt = matplot;

    // Bouncing ball simulation
    // State: [position, velocity]
    // ODE: d²x/dt² = -g (gravity)
    // Event: when x <= 0 and v < 0, bounce with energy loss

    const double g       = 9.81;   // gravity (m/s²)
    const double e       = 0.8;    // coefficient of restitution (energy loss)
    const double t_final = 100.0;  // simulation time (s) - increased to allow ball to come to rest

    // Initial conditions: dropped from 10m height
    ColVec x0 = {10.0, 0.0};  // [position, velocity]

    // ODE: dx/dt = [v, -g]
    auto ode = [g](double /*t*/, const ColVec& x) -> ColVec {
        return ColVec{{x(1), -g}};  // [velocity, acceleration]
    };

    // Create solver with zero-crossing detection
    AdaptiveStepSolver<RK45> solver(0.001, 1e-6, 1e-8, 0.001);  // Very small max_step to ensure we catch all zero crossings

    // Define zero-crossing function: position crosses zero (ground level)
    solver.set_zero_crossings({[](double /*t*/, const ColVec& x) -> double {
        return x(0);  // Position - crosses zero when hitting ground
    }});

    // Event detector for ground impact (simplified since zero-crossing ensures position = 0)
    int bounce_count = 0;
    solver.set_event_detector([e, &bounce_count](double /*t*/, const ColVec& x) -> EventResult {
        double vel = x(1);  // velocity

        if (vel < 0.0) {  // Only bounce if moving downward (though zero-crossing should ensure this)
            bounce_count++;
            double new_vel = -e * vel;  // Reverse and dampen velocity
            std::cout << "Bounce #" << bounce_count << " at position " << x(0) << "m, velocity " << vel << "m/s -> " << new_vel << "m/s" << std::endl;

            // Check if bounce is too weak to continue (ball comes to rest)
            if (std::abs(new_vel) < 0.01) {
                std::cout << "Ball has come to rest after bounce #" << bounce_count << std::endl;
                return {true, ColVec{{0.0, 0.0}}, false, true, "Ball at rest"};  // Stop simulation
            }

            return {true, ColVec{{0.0, new_vel}}, false, false, "Ground impact - bounce!"};
        }

        return {false, x, false, false, ""};  // No event
    });

    // Stop condition removed - stopping is handled in event detector when ball comes to rest

    // Solve the system
    auto result = solver.solve(ode, x0, {0.0, t_final});

    // Extract data for plotting
    std::vector<double> time, position, velocity;
    for (const auto& [t, state] : result) {
        time.push_back(t);
        position.push_back(state(0));
        velocity.push_back(state(1));
    }

    // Create plots
    auto fig = plt::figure(true);
    fig->size(1200, 1200);

    // Position vs time
    plt::subplot(3, 1, 0);
    plt::plot(time, position)->line_width(2);
    plt::xlabel("Time (s)");
    plt::ylabel("Position (m)");
    plt::title("Bouncing Ball - Position");
    plt::grid(plt::on);
    plt::ylim({-0.5, 11.0});

    // Velocity vs time
    plt::subplot(3, 1, 1);
    plt::plot(time, velocity)->line_width(2).color("red");
    plt::xlabel("Time (s)");
    plt::ylabel("Velocity (m/s)");
    plt::title("Bouncing Ball - Velocity");
    plt::grid(plt::on);

    // Phase portrait (velocity vs position)
    plt::subplot(3, 1, 2);
    plt::plot(position, velocity)->line_width(1.5);
    plt::xlabel("Position (m)");
    plt::ylabel("Velocity (m/s)");
    plt::title("Phase Portrait: Velocity vs Position");
    plt::grid(plt::on);

    std::cout << "Bouncing ball simulation completed!\n";
    std::cout << "Simulation time: " << result.t.back() << " seconds\n";
    std::cout << "Time steps: " << result.t.size() << "\n";
    std::cout << "Final position: " << position.back() << " m\n";
    std::cout << "Final velocity: " << velocity.back() << " m/s\n";

    // plt::show();
    return 0;
}