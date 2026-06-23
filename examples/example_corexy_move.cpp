// ===== CoreXY straight-line move -> time-optimal timing -> servo commands =====
//
// A complete, copy-pasteable motion pipeline for a CoreXY machine (the belt
// layout used by many 3D printers and pen plotters) driven by two servo drives.
// What it does, in plain English:
//
//   1. You ask for a straight line from point A to point B on the bed.
//   2. CoreXY math turns each (X, Y) point into the two belt-motor positions.
//   3. TOPP ("time-optimal path parameterization") finds the FASTEST way to run
//      along that line without ever exceeding your speed/acceleration limits.
//   4. Every control tick we convert that into ready-to-send motor commands:
//      position [turns], velocity [turns/s], and torque [Nm].
//
// You do NOT need to understand the math. To use it on your machine, change the
// numbers in the CONFIG block and read the commands out of the loop at the end.
//
// Units note: everything here is in metres and seconds (SI), because that makes
// the torque feedforward come out in real Nm. 0.200 m is just 200 mm.

#include "fmt/base.h"
#include "fmt/core.h"
#include "wet/backend.hpp"
#include "wet/control.hpp"

using namespace wet;

int main() {

    // ============================ CONFIG ====================================
    // --- The move you want (bed coordinates, in metres) ---
    const double start_x = 0.000, start_y = 0.000; // from here (0, 0) mm
    const double end_x = 0.200, end_y = 0.150;     // to here  (200, 150) mm

    // --- Your machine's belt drive ---
    // Belt travel per ONE motor revolution. GT2 belt (2 mm pitch) on a 20-tooth
    // pulley = 2 mm x 20 = 40 mm = 0.040 m per turn.
    const double belt_travel_per_rev = 0.040; // [m/rev]

    // --- Your speed and acceleration limits (per belt motor) ---
    const double max_speed = 0.5; // [m/s]   how fast a belt may run
    const double max_accel = 5.0; // [m/s^2] how hard a belt may accelerate

    // --- Optional: moving mass, for torque feedforward ---
    // The mass the belts have to push around (carriage + hotend, etc.). Set to 0
    // to skip torque feedforward entirely (position + velocity still work fine).
    // CoreXY keeps this mass constant no matter where the head is, so one number
    // is a good model here.
    const double moving_mass = 0.6; // [kg]

    const double control_dt = 0.01; // [s] how often you push a command (100 Hz)
    // ========================================================================

    fmt::print("===== CoreXY move: ({:.0f}, {:.0f}) mm  ->  ({:.0f}, {:.0f}) mm =====\n\n", start_x * 1e3, start_y * 1e3, end_x * 1e3, end_y * 1e3);

    // --- Step 1: the straight-line path on the bed (start -> end). -----------
    const LinearPath<double> path{
        {start_x, start_y, 0.0},
        {end_x, end_y, 0.0},
    };

    // --- Step 2: CoreXY inverse kinematics: a bed point -> two belt-motor -----
    //     positions {A, B}. TOPP calls this for us; it must return the motor
    //     positions plus a "reachable?" flag (always true for CoreXY).
    auto corexy_ik = [](const Vec3<double>& p) {
        const auto motors = CoreXY<double>::inverse(p); // {A, B} belt travel
        return wet::pair<wet::array<double, 2>, bool>{{motors.a, motors.b}, true};
    };

    // --- Step 3: ask TOPP for the time-optimal run along the path. -----------
    //     Same speed/accel cap on both belt motors.
    const JointLimits<2, double> limits{{max_speed, max_speed}, {max_accel, max_accel}};
    auto                         move = make_topp_move<256>(path, corexy_ik, path.length(), limits);

    // --- Step 4: describe each belt motor as one servo drive. ----------------
    //     "linear_screw" = a rotary motor that produces linear travel; the belt
    //     travel-per-rev is its lead. One ratio handles position, velocity, AND
    //     torque (the library reflects force -> motor torque automatically).
    ServoBank<2, double> servos({linear_screw<double>(belt_travel_per_rev), linear_screw<double>(belt_travel_per_rev)});

    // Torque feedforward model: each belt pushes the same moving mass.
    ConstantInertiaFeedforward<2, double> torque_ff{};
    torque_ff.inertia = {moving_mass, moving_mass};

    fmt::print("Path length: {:.1f} mm    Move time (time-optimal): {:.3f} s\n", path.length() * 1e3, move.duration());
    fmt::print("Streaming commands at {:.0f} Hz:\n\n", 1.0 / control_dt);
    fmt::print("   t[s]  | motor |  input_pos[turns]   vel_ff[turns/s]   torque_ff[Nm]\n");
    fmt::print("  -------+-------+----------------------------------------------------\n");

    // --- Step 5: the run loop. This is the whole thing you put in your code. -
    move.reset();
    for (double t = 0.0; t <= move.duration() + 1e-9; t += control_dt) {
        // One line gets you the time-optimal motor motion at this instant...
        const auto states = move.eval(t);
        // ...and one line turns it into drive-ready commands (with torque FF).
        const auto cmds = servos.command(states, torque_ff);

        // Send cmds[0] to motor A and cmds[1] to motor B. On a typical position
        // drive that is:
        //   driveA.input_pos = cmds[0].position;     // turns
        //   driveA.input_vel = cmds[0].velocity;     // turns/s  (velocity FF)
        //   driveA.input_torque = cmds[0].torque;    // Nm       (torque FF)

        // Print a few rows so you can see it work (every ~8th tick).
        const bool show = (t < 1e-9) || (t + control_dt > move.duration()) || (static_cast<int>(t / control_dt) % 8 == 0);
        if (show) {
            for (int m = 0; m < 2; ++m) {
                fmt::print("  {:5.2f}  |   {}   | {:14.4f}    {:13.4f}    {:11.4f}\n", t, (m == 0 ? 'A' : 'B'), cmds[m].position, cmds[m].velocity, cmds[m].torque);
            }
            fmt::print("\n");
        }
    }

    fmt::print("Done. Copy the loop body into your firmware and you have coordinated,\n");
    fmt::print("time-optimal CoreXY motion with two servo drives.\n");
    return 0;
}
