#include <fmt/core.h>

#include "control.hpp"
#include "matplot/matplot.h"

int main() {
    using namespace control;
    namespace plt = matplot;

    fmt::print("=== Motor Control with Rotating Load Example ===\n");
    fmt::print("DC motor with inertia load and viscous friction\n\n");

    // Motor parameters
    const double Ra = 1.0;    // Armature resistance [Ohm]
    const double La = 0.01;   // Armature inductance [H]
    const double Kb = 0.1;    // Back-EMF constant [V/(rad/s)]
    const double Kt = 0.1;    // Torque constant [Nm/A]
    const double Jm = 0.01;   // Motor inertia [kg*m^2]
    const double Jl = 0.1;    // Load inertia [kg*m^2]
    const double Bm = 0.001;  // Motor viscous friction [Nm/(rad/s)]
    const double Bl = 0.01;   // Load viscous friction [Nm/(rad/s)]
    const double K  = 10.0;   // Shaft stiffness [Nm/rad]

    // Combined inertia and friction
    const double J_total = Jm + Jl;
    const double B_total = Bm + Bl;

    fmt::print("Motor parameters:\n");
    fmt::print("  Ra = {:.3f} Ohm, La = {:.3f} H\n", Ra, La);
    fmt::print("  Kb = {:.3f} V/(rad/s), Kt = {:.3f} Nm/A\n", Kb, Kt);
    fmt::print("  J = {:.3f} kg*m², B = {:.3f} Nm/(rad/s)\n", J_total, B_total);
    fmt::print("  K = {:.1f} Nm/rad\n\n", K);

    // State-space model: [ia, ω_m, θ_m, ω_l, θ_l]^T
    // x1 = ia (armature current)
    // x2 = ω_m (motor speed)
    // x3 = θ_m (motor position)
    // x4 = ω_l (load speed)
    // x5 = θ_l (load position)
    Matrix A_sys = Matrix{{-Ra / La, -Kb / La, 0, 0, 0},
                          {Kt / J_total, -B_total / J_total, -K / J_total, B_total / J_total, K / J_total},
                          {0, 1, 0, 0, 0},
                          {0, B_total / Jl, K / Jl, -B_total / Jl, -K / Jl},
                          {0, 0, 0, 1, 0}};

    Matrix B_sys = Matrix{{1.0 / La}, {0}, {0}, {0}, {0}};  // Input voltage Va

    Matrix C_sys = Matrix{{0, 0, 1, 0, 0},   // Motor position θ_m
                          {0, 0, 0, 0, 1}};  // Load position θ_l

    Matrix D_sys = Matrix::Zero(2, 1);

    StateSpace motor_system(A_sys, B_sys, C_sys, D_sys);

    // Check stability
    bool stable = is_stable(motor_system);
    fmt::print("System is {}stable\n\n", stable ? "" : "un");

    // Compute transfer functions
    TransferFunction motor_tf = tf(motor_system, 0, 0);  // θ_m / Va
    TransferFunction load_tf  = tf(motor_system, 1, 0);  // θ_l / Va

    // Frequency response analysis
    auto bode_data = bode(motor_system, 0.01, 100.0, 100);

    // Plot Bode magnitude
    auto fig = plt::figure(true);
    fig->size(1200, 800);

    plt::sgtitle("Motor Control - Frequency Response");
    plt::subplot(2, 1, 0);
    plt::semilogx(bode_data.freq, bode_data.magnitude);
    plt::ylabel("Magnitude [dB]");
    plt::grid(true);

    plt::subplot(2, 1, 1);
    plt::semilogx(bode_data.freq, bode_data.phase);
    plt::xlabel("Frequency [rad/s]");
    plt::ylabel("Phase [deg]");
    plt::grid(true);

    plt::show();

    // Step response
    auto step_resp = step(motor_system, 0.0, 5.0, Matrix::Constant(1, 1, 12.0));  // 12V step

    fmt::print("Step response analysis (motor position):\n");
    // Create SISO system for motor position only
    StateSpace motor_pos_system(A_sys, B_sys, C_sys.row(0), D_sys.row(0));
    auto       step_info = stepinfo(motor_pos_system);
    fmt::print("Rise time: {:.3f} s\n", step_info.riseTime[0]);
    fmt::print("Settling time: {:.3f} s\n", step_info.settlingTime[0]);
    fmt::print("Overshoot: {:.1f}%\n", step_info.overshoot[0] * 100);
    fmt::print("Steady-state value: {:.3f}\n", step_info.peak[0]);

    return 0;
}