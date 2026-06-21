#include <array>
#include <cmath>
#include <string_view>

#include "fmt/core.h"
#include "wet/trajectory/trajectory.hpp"
#include "wet/utility/actuator.hpp"

using namespace wet;

// ===== Trajectory → drive command bridge =====
//
// A 3-axis coordinated S-curve move is planned in SI joint units (rad), then mapped
// to drive-native {input_pos [turns], vel_ff [turns/s], torque_ff [Nm]} triples ready
// to stream to three independent servo drives.
//
//   plan (rad) ── TrajectoryBank ──▶ {pos,vel,acc} ── ServoBank ──▶ {turns, turns/s, Nm}
//                                       per axis    + inertia FF        per drive
//
// Each axis is one number — its gear ratio — built with rotary_gearbox(). The
// constant-inertia feedforward (J·a + b·v + τc·sign(v) + g) is exact for these
// decoupled single-actuator joints; The torque is computed joint-side and reflected to the
// motor through the same gear ratio automatically.

int main() {
    fmt::print("===== Trajectory → servo drive command stream (3 axes) =====\n\n");

    // Three joints, each its own reduction and kinematic limits {v, a, d, j} (rad).
    struct Axis {
        std::string_view         name;
        double                   start, target;
        double                   gear;    // n:1 reduction
        double                   inertia; // joint-side inertia [kg·m²]
        TrajectoryLimits<double> lim;
    };

    const std::array<Axis, 3> axes{{
        {"J1", 0.0, 1.5, 40.0, 0.80, {2.0, 6.0, 6.0, 40.0}},
        {"J2", 0.0, -0.8, 25.0, 0.30, {3.0, 10.0, 10.0, 60.0}},
        {"J3", 0.0, 2.2, 10.0, 0.05, {5.0, 20.0, 20.0, 120.0}},
    }};

    // --- Plan: per-axis minimum-time S-curves, coordinated into one timeline. ---
    std::array<ScurveTrajectory<double>, 3> trajs{};
    wet::array<ServoAxis<double>, 3>        servo_axes{};
    ConstantInertiaFeedforward<3, double>   ff{};
    for (size_t i = 0; i < axes.size(); ++i) {
        trajs[i] = ScurveTrajectory<double>(design::synthesize_scurve(axes[i].start, axes[i].target, axes[i].lim));
        servo_axes[i] = rotary_gearbox<double>(axes[i].gear);
        ff.inertia[i] = axes[i].inertia;
        ff.viscous[i] = 0.05; // a little drivetrain damping
        ff.coulomb[i] = 0.10; // Coulomb friction [Nm]
    }
    TrajectoryBank<3, ScurveTrajectory<double>> bank(trajs);
    ServoBank<3, double>                        servos(servo_axes);

    fmt::print("Coordinated move duration T_sync = {:.3f} s\n", bank.duration());
    fmt::print("Axis setup:\n");
    for (size_t i = 0; i < axes.size(); ++i) {
        fmt::print("  {}: {:.0f}:1 gearbox -> {:.4f} turns/rad, J={:.2f} kg·m²\n", axes[i].name, axes[i].gear, servo_axes[i].turns_per_unit, axes[i].inertia);
    }

    // --- Stream: the whole per-tick loop body is two calls. ---
    const double dt = 0.05;
    fmt::print("\n  t[s]   | axis |  input_pos[turns]  vel_ff[turns/s]  torque_ff[Nm]\n");
    fmt::print("  -------+------+-------------------------------------------------\n");
    bank.reset();
    for (double t = 0.0; t <= bank.duration() + 1e-9; t += dt) {
        const auto states = bank.eval(t);             // SI joint units
        const auto cmds = servos.command(states, ff); // drive-native, with torque FF
        if (std::abs(t - 0.0) < 1e-9 || std::abs(t - (bank.duration() / 2)) < dt / 2 || t + dt > bank.duration()) {
            for (size_t i = 0; i < 3; ++i) {
                fmt::print("  {:5.2f}  |  {}  | {:14.4f}   {:13.4f}   {:11.4f}\n", t, axes[i].name, cmds[i].position, cmds[i].velocity, cmds[i].torque);
            }
            fmt::print("\n");
        }
    }

    fmt::print("Each row is ready to write straight to a drive:\n");
    fmt::print("  drive[i].controller.input_pos = cmds[i].position;   // turns\n");
    fmt::print("  drive[i].controller.input_vel = cmds[i].velocity;   // turns/s (vel_ff)\n");
    fmt::print("  drive[i].controller.input_torque = cmds[i].torque;  // Nm (torque_ff)\n");
    return 0;
}
