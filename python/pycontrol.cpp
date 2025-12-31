#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../source/control.hpp"
#include "../source/solver.hpp"
#include "types.hpp"

namespace py = pybind11;
using namespace control;

// ODrive cascade controller dynamics with torque saturation
// State: [theta_motor, omega_motor, theta_load, omega_load, integrator]
// Input: position_command
struct ODriveParams {
    // Physical parameters
    double J_motor;
    double J_load;
    double k_coupling;
    double c_coupling;
    double b_motor;
    double b_load;
    double max_torque;

    // Controller gains
    double pos_gain;
    double vel_gain;
    double vel_integrator_gain;
};

// Dynamics function for ODrive system with torque saturation
Matrix odrive_dynamics(double t, const Matrix& x, const ODriveParams& params, double pos_command) {
    double theta_m    = x(0, 0);  // motor angle
    double omega_m    = x(1, 0);  // motor velocity
    double theta_l    = x(2, 0);  // load angle
    double omega_l    = x(3, 0);  // load velocity
    double integrator = x(4, 0);  // velocity integrator state

    // Position controller: generates velocity setpoint
    double pos_error    = pos_command - theta_m;
    double vel_setpoint = params.pos_gain * pos_error;

    // Velocity controller: generates torque
    double vel_error = vel_setpoint - omega_m;
    double tau_motor = params.vel_gain * vel_error + params.vel_integrator_gain * integrator;

    // Apply torque saturation
    tau_motor = std::clamp(tau_motor, -params.max_torque, params.max_torque);

    // Coupling torque
    double tau_coupling = params.k_coupling * (theta_m - theta_l) +
                          params.c_coupling * (omega_m - omega_l);

    // Motor dynamics
    double d_theta_m = omega_m;
    double d_omega_m = (tau_motor - tau_coupling - params.b_motor * omega_m) / params.J_motor;

    // Load dynamics
    double d_theta_l = omega_l;
    double d_omega_l = (tau_coupling - params.b_load * omega_l) / params.J_load;

    // Integrator dynamics
    double d_integrator = vel_error;

    const auto dx = ColVec{d_theta_m,
                           d_omega_m,
                           d_theta_l,
                           d_omega_l,
                           d_integrator};
    return dx;
}

// Solve ODrive system with step input
SolveResult solve_odrive_step(
    const ODriveParams& params,
    double              step_size,
    double t0, double tf,
    const std::vector<double>& t_eval,
    double                     initial_step = 0.01,
    double                     tol          = 1e-3,
    double                     max_step     = 0.1) {
    // Initial condition: at rest at origin
    Matrix x0 = Matrix::Zero(5, 1);

    // Create dynamics function with constant step input
    auto dynamics = [&params, step_size](double t, const Matrix& x) -> Matrix {
        return odrive_dynamics(t, x, params, step_size);
    };

    try {
        // Use adaptive stepping with RK45 for speed
        // Limit max function evaluations to prevent hangs with stiff systems
        constexpr size_t         max_nfev = 100000;
        AdaptiveStepSolver<RK45> solver(initial_step, tol, 1e-8, max_step, max_nfev);
        auto                     result = solver.solve(dynamics, x0, {t0, tf}, t_eval);

        // Add helpful context to error messages
        if (!result.success && result.message.find("stiff") != std::string::npos) {
            result.message += " Try increasing motor/load inertia or reducing controller gains.";
        }
        return result;
    } catch (const std::exception& e) {
        SolveResult error_result;
        error_result.success = false;
        error_result.message = std::string("C++ exception: ") + e.what();
        return error_result;
    }
}

// Trapezoidal trajectory generator (pure C++)
inline double trapezoidal_position(double t, double distance, double max_vel, double max_accel) {
    double t_accel        = max_vel / max_accel;
    double actual_max_vel = max_vel;
    double t_cruise       = 0.0;

    // Check if we reach max velocity
    if (t_accel * max_vel > distance) {
        // Triangular profile - can't reach max_vel
        t_accel        = std::sqrt(distance / max_accel);
        actual_max_vel = max_accel * t_accel;
    } else {
        // Trapezoidal profile - we do reach max_vel
        t_cruise = (distance - max_accel * t_accel * t_accel) / max_vel;
    }

    double t_decel = t_accel;

    if (t < 0) {
        return 0.0;
    } else if (t < t_accel) {
        // Acceleration phase
        return 0.5 * max_accel * t * t;
    } else if (t < t_accel + t_cruise) {
        // Cruise phase
        return 0.5 * max_accel * t_accel * t_accel + actual_max_vel * (t - t_accel);
    } else if (t < t_accel + t_cruise + t_decel) {
        // Deceleration phase
        double t_in_decel = t - t_accel - t_cruise;
        double cruise_pos = 0.5 * max_accel * t_accel * t_accel + actual_max_vel * t_cruise;
        return cruise_pos + actual_max_vel * t_in_decel - 0.5 * max_accel * t_in_decel * t_in_decel;
    } else {
        // Hold final position
        return distance;
    }
}

// Solve ODrive system with time-varying trajectory (Python callback)
SolveResult solve_odrive_trajectory(
    const ODriveParams& params,
    py::function        pos_traj_callback,  // Python function: pos_command = pos_traj(t)
    double t0, double tf,
    const std::vector<double>& t_eval,
    double                     initial_step = 0.01,
    double                     tol          = 1e-3,
    double                     max_step     = 0.1) {
    // Initial condition
    Matrix x0 = Matrix::Zero(5, 1);

    // Create dynamics function with time-varying input
    auto dynamics = [&params, &pos_traj_callback](double t, const Matrix& x) -> Matrix {
        // Call Python function to get position command at time t
        double pos_command = pos_traj_callback(t).cast<double>();
        return odrive_dynamics(t, x, params, pos_command);
    };

    try {
        // Use adaptive stepping with RK45 for speed
        // Limit max function evaluations to prevent hangs with stiff systems
        constexpr size_t         max_nfev = 100000;
        AdaptiveStepSolver<RK45> solver(initial_step, tol, 1e-8, max_step, max_nfev);
        auto                     result = solver.solve(dynamics, x0, {t0, tf}, t_eval);

        // Add helpful context to error messages
        if (!result.success && result.message.find("stiff") != std::string::npos) {
            result.message += " Try increasing motor/load inertia or reducing controller gains.";
        }
        return result;
    } catch (const std::exception& e) {
        SolveResult error_result;
        error_result.success = false;
        error_result.message = std::string("C++ exception: ") + e.what();
        return error_result;
    }
}

// Solve ODrive system with trapezoidal trajectory (pure C++, no Python callbacks)
SolveResult solve_odrive_trajectory_trapezoidal(
    const ODriveParams& params,
    double              distance,
    double              max_vel,
    double              max_accel,
    double t0, double tf,
    const std::vector<double>& t_eval,
    double                     initial_step = 0.01,
    double                     tol          = 1e-3,
    double                     max_step     = 0.1) {
    // Initial condition
    Matrix x0 = Matrix::Zero(5, 1);

    // Create dynamics function with trapezoidal trajectory
    auto dynamics = [&params, distance, max_vel, max_accel](double t, const Matrix& x) -> Matrix {
        double pos_command = trapezoidal_position(t, distance, max_vel, max_accel);
        return odrive_dynamics(t, x, params, pos_command);
    };

    try {
        // Use adaptive stepping with RK45 for speed
        // Limit max function evaluations to prevent hangs with stiff systems
        constexpr size_t         max_nfev = 100000;
        AdaptiveStepSolver<RK45> solver(initial_step, tol, 1e-8, max_step, max_nfev);
        auto                     result = solver.solve(dynamics, x0, {t0, tf}, t_eval);

        // Add helpful context to error messages
        if (!result.success && result.message.find("stiff") != std::string::npos) {
            result.message += " Try increasing motor/load inertia or reducing controller gains.";
        }
        return result;
    } catch (const std::exception& e) {
        SolveResult error_result;
        error_result.success = false;
        error_result.message = std::string("C++ exception: ") + e.what();
        return error_result;
    }
}

PYBIND11_MODULE(pycontrol, m) {
    m.doc() = "Python bindings for ODrive motor control simulation with nonlinear dynamics";

    // SolveResult
    py::class_<SolveResult>(m, "SolveResult")
        .def_readonly("t", &SolveResult::t)
        .def_readonly("x", &SolveResult::x)
        .def_readonly("success", &SolveResult::success)
        .def_readonly("message", &SolveResult::message)
        .def_readonly("nfev", &SolveResult::nfev);

    // ODriveParams
    py::class_<ODriveParams>(m, "ODriveParams")
        .def(py::init<double, double, double, double, double, double, double, double, double, double>(),
             py::arg("J_motor"), py::arg("J_load"), py::arg("k_coupling"), py::arg("c_coupling"),
             py::arg("b_motor"), py::arg("b_load"), py::arg("max_torque"),
             py::arg("pos_gain"), py::arg("vel_gain"), py::arg("vel_integrator_gain"))
        .def_readwrite("J_motor", &ODriveParams::J_motor)
        .def_readwrite("J_load", &ODriveParams::J_load)
        .def_readwrite("k_coupling", &ODriveParams::k_coupling)
        .def_readwrite("c_coupling", &ODriveParams::c_coupling)
        .def_readwrite("b_motor", &ODriveParams::b_motor)
        .def_readwrite("b_load", &ODriveParams::b_load)
        .def_readwrite("max_torque", &ODriveParams::max_torque)
        .def_readwrite("pos_gain", &ODriveParams::pos_gain)
        .def_readwrite("vel_gain", &ODriveParams::vel_gain)
        .def_readwrite("vel_integrator_gain", &ODriveParams::vel_integrator_gain);

    // ODrive-specific solvers with nonlinear dynamics and torque saturation
    m.def("solve_odrive_step",
          &solve_odrive_step,
          py::arg("params"),
          py::arg("step_size"),
          py::arg("t0"),
          py::arg("tf"),
          py::arg("t_eval")       = std::vector<double>{},
          py::arg("initial_step") = 0.01,
          py::arg("tol")          = 1e-3,
          py::arg("max_step")     = 0.1,
          "Solve ODrive step response with adaptive RK45 integration. "
          "params: ODriveParams struct with physical and controller parameters. "
          "initial_step: starting step size, tol: error tolerance (lower=more accurate), "
          "max_step: maximum allowed step size for speed");

    m.def("solve_odrive_trajectory",
          &solve_odrive_trajectory,
          py::arg("params"),
          py::arg("pos_traj_callback"),
          py::arg("t0"),
          py::arg("tf"),
          py::arg("t_eval")       = std::vector<double>{},
          py::arg("initial_step") = 0.01,
          py::arg("tol")          = 1e-3,
          py::arg("max_step")     = 0.1,
          "Solve ODrive trajectory tracking with adaptive RK45 integration (Python callback). "
          "params: ODriveParams struct with physical and controller parameters. "
          "initial_step: starting step size, tol: error tolerance (lower=more accurate), "
          "max_step: maximum allowed step size for speed");

    m.def("solve_odrive_trajectory_trapezoidal",
          &solve_odrive_trajectory_trapezoidal,
          py::arg("params"),
          py::arg("distance"),
          py::arg("max_vel"),
          py::arg("max_accel"),
          py::arg("t0"),
          py::arg("tf"),
          py::arg("t_eval")       = std::vector<double>{},
          py::arg("initial_step") = 0.01,
          py::arg("tol")          = 1e-3,
          py::arg("max_step")     = 0.1,
          "Solve ODrive trapezoidal trajectory tracking with adaptive RK45 integration (pure C++). "
          "params: ODriveParams struct with physical and controller parameters. "
          "distance: total distance to travel, max_vel: maximum velocity, max_accel: maximum acceleration. "
          "initial_step: starting step size, tol: error tolerance (lower=more accurate), "
          "max_step: maximum allowed step size for speed");
}
