"""
ODrive Motor Controller Simulator with Flexible Coupling

This module simulates a 2-mass system (motor + load) with a flexible coupling,
controlled by an ODrive-like cascade controller.

The system dynamics are:
- Motor inertia J_m driven by torque tau
- Load inertia J_l
- Flexible coupling with stiffness k and damping c
- Cascade control: Position -> Velocity -> Current

State vector: [theta_m, omega_m, theta_l, omega_l, vel_integrator]
where:
  theta_m: motor position
  omega_m: motor velocity
  theta_l: load position
  omega_l: load velocity
  vel_integrator: velocity error integral
"""

import numpy as np
from typing import Tuple, Callable
from dataclasses import dataclass
from scipy.integrate import solve_ivp

import pycontrol


@dataclass
class PhysicalParameters:
    """Physical parameters of the motor-load system"""

    J_motor: float = 0.001  # Motor inertia (kg·m²)
    J_load: float = 0.005  # Load inertia (kg·m²)
    k_coupling: float = 100.0  # Coupling stiffness (N·m/rad)
    c_coupling: float = 0.5  # Coupling damping (N·m·s/rad)
    b_motor: float = 0.0001  # Motor friction (N·m·s/rad)
    b_load: float = 0.0001  # Load friction (N·m·s/rad)
    max_torque: float = 5.0  # Maximum motor torque (N·m)


@dataclass
class ControllerGains:
    """ODrive cascade controller gains"""

    pos_gain: float = 20.0  # Position controller gain (1/s)
    vel_gain: float = 0.15  # Velocity controller gain (N·m·s/rad)
    vel_integrator_gain: float = 0.3  # Velocity integrator gain (N·m/rad)


class ODriveFlexibleSystem:
    """
    Simulates an ODrive controlling a motor with flexible coupling to a load.

    Uses cascade control architecture:
    pos_error -> pos_controller -> vel_setpoint -> vel_controller -> torque
    """

    def __init__(
        self,
        phys_params: PhysicalParameters = None,
        ctrl_gains: ControllerGains = None,
        use_scipy: bool = False,
    ):
        self.phys = phys_params or PhysicalParameters()
        self.gains = ctrl_gains or ControllerGains()
        self.use_scipy = use_scipy

    def _odrive_dynamics(
        self, t: float, x: np.ndarray, pos_command: float
    ) -> np.ndarray:
        """
        Compute ODrive system dynamics for scipy.integrate.solve_ivp.

        State: [theta_m, omega_m, theta_l, omega_l, vel_integrator]
        """
        theta_m, omega_m, theta_l, omega_l, vel_integrator = x

        # Cascade controller
        pos_error = pos_command - theta_m
        vel_setpoint = self.gains.pos_gain * pos_error
        vel_error = vel_setpoint - omega_m

        # Velocity controller with integrator
        torque_command = (
            self.gains.vel_gain * vel_error
            + self.gains.vel_integrator_gain * vel_integrator
        )

        # Torque saturation
        torque = np.clip(torque_command, -self.phys.max_torque, self.phys.max_torque)

        # Coupling torque
        torque_coupling = self.phys.k_coupling * (
            theta_l - theta_m
        ) + self.phys.c_coupling * (omega_l - omega_m)

        # Dynamics
        theta_m_dot = omega_m
        omega_m_dot = (
            torque - self.phys.b_motor * omega_m + torque_coupling
        ) / self.phys.J_motor
        theta_l_dot = omega_l
        omega_l_dot = (-torque_coupling - self.phys.b_load * omega_l) / self.phys.J_load
        vel_integrator_dot = vel_error

        return np.array(
            [theta_m_dot, omega_m_dot, theta_l_dot, omega_l_dot, vel_integrator_dot]
        )

    def simulate_step(
        self, step_size: float = 1.0, duration: float = 2.0, dt: float = 0.001
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate step response.

        Args:
            step_size: Step input magnitude
            duration: Simulation duration
            dt: Time step

        Returns:
            (time_array, state_array) where state_array has shape (n_steps, 5)
        """
        if self.use_scipy:
            return self._simulate_step_scipy(step_size, duration, dt)
        else:
            return self._simulate_step_cpp(step_size, duration, dt)

    def _simulate_step_scipy(
        self, step_size: float, duration: float, dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate using scipy.integrate.solve_ivp."""
        x0 = np.zeros(5)
        n_steps = int(duration / dt)
        t_eval = np.linspace(0, duration, n_steps)

        # Create dynamics function with constant step input
        def dynamics(t, x):
            return self._odrive_dynamics(t, x, step_size)

        sol = solve_ivp(
            dynamics,
            (0, duration),
            x0,
            method="RK45",
            t_eval=t_eval,
            rtol=1e-3,
            atol=1e-6,
        )

        return sol.t, sol.y.T

    def _simulate_step_cpp(
        self, step_size: float, duration: float, dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate using C++ nonlinear solver with torque saturation.

        Dynamics computed entirely in C++, including torque saturation.
        """
        # Pack parameters for C++
        params = pycontrol.ODriveParams(
            J_motor=self.phys.J_motor,
            J_load=self.phys.J_load,
            k_coupling=self.phys.k_coupling,
            c_coupling=self.phys.c_coupling,
            b_motor=self.phys.b_motor,
            b_load=self.phys.b_load,
            max_torque=self.phys.max_torque,
            pos_gain=self.gains.pos_gain,
            vel_gain=self.gains.vel_gain,
            vel_integrator_gain=self.gains.vel_integrator_gain,
        )

        # Create time evaluation points
        n_steps = int(duration / dt)
        t_eval = np.linspace(0, duration, n_steps).tolist()

        # Use C++ nonlinear solver with adaptive stepping for speed
        # Relaxed tolerance (1e-3) and larger max_step (0.1) for faster GUI updates
        try:
            result = pycontrol.solve_odrive_step(
                params,
                step_size,
                0.0,
                duration,
                t_eval,
                initial_step=0.01,  # Initial adaptive step size
                tol=1e-3,  # Relaxed tolerance for speed
                max_step=0.1,  # Allow large steps for faster computation
            )

            if result.success:
                t = np.array(result.t)
                x = np.array([np.array(xi).flatten() for xi in result.x])
                return t, x
            else:
                raise RuntimeError(f"C++ ODrive solver failed: {result.message}")
        except Exception as e:
            print(f"Error using C++ ODrive solver: {e}")
            raise

    def _simulate_trajectory_cpp_trapezoidal(
        self, duration: float, dt: float, distance: float, max_vel: float, max_accel: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate trajectory using C++ trapezoidal profile (pure C++, no Python callbacks)."""
        # Pack parameters for C++
        params = pycontrol.ODriveParams(
            J_motor=self.phys.J_motor,
            J_load=self.phys.J_load,
            k_coupling=self.phys.k_coupling,
            c_coupling=self.phys.c_coupling,
            b_motor=self.phys.b_motor,
            b_load=self.phys.b_load,
            max_torque=self.phys.max_torque,
            pos_gain=self.gains.pos_gain,
            vel_gain=self.gains.vel_gain,
            vel_integrator_gain=self.gains.vel_integrator_gain,
        )

        # Generate time vector
        n_steps = int(duration / dt)
        t_eval = np.linspace(0, duration, n_steps).tolist()

        try:
            result = pycontrol.solve_odrive_trajectory_trapezoidal(
                params,
                distance,
                max_vel,
                max_accel,
                0.0,
                duration,
                t_eval,
                initial_step=0.01,
                tol=1e-3,
                max_step=0.1,
            )

            if result.success:
                t = np.array(result.t)
                x = np.array([np.array(xi).flatten() for xi in result.x])
                # Compute command trajectory for plotting
                from odrive_system import generate_trapezoidal_profile
                pos_func, _ = generate_trapezoidal_profile(duration, max_vel, max_accel, distance)
                commands = np.array([pos_func(ti) for ti in t])
                return t, x, commands
            else:
                raise RuntimeError(f"C++ ODrive solver failed: {result.message}")
        except Exception as e:
            print(f"Error using C++ ODrive solver: {e}")
            raise

    def simulate_trajectory(
        self,
        pos_traj: Callable[[float], float],
        duration: float = 2.0,
        dt: float = 0.001,
        traj_params: dict = None,  # Optional: {distance, max_vel, max_accel} for C++ trapezoidal
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate tracking of a position trajectory.

        Args:
            pos_traj: Function that returns position command at time t
            duration: Simulation duration
            dt: Time step
            traj_params: If provided and using C++, use pure C++ trapezoidal trajectory
                        dict with keys: distance, max_vel, max_accel

        Returns:
            (time_array, state_array, command_array)
        """
        if self.use_scipy:
            return self._simulate_trajectory_scipy(pos_traj, duration, dt)
        elif traj_params is not None and not self.use_scipy:
            return self._simulate_trajectory_cpp_trapezoidal(duration, dt, **traj_params)
        else:
            return self._simulate_trajectory_cpp(pos_traj, duration, dt)

    def _simulate_trajectory_scipy(
        self, pos_traj: Callable[[float], float], duration: float, dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate trajectory using scipy.integrate.solve_ivp."""
        x0 = np.zeros(5)
        n_steps = int(duration / dt)
        t_eval = np.linspace(0, duration, n_steps)

        # Create dynamics function with time-varying input
        def dynamics(t, x):
            return self._odrive_dynamics(t, x, pos_traj(t))

        sol = solve_ivp(
            dynamics,
            (0, duration),
            x0,
            method="RK45",
            t_eval=t_eval,
            rtol=1e-3,
            atol=1e-6,
        )

        commands = np.array([pos_traj(t) for t in sol.t])
        return sol.t, sol.y.T, commands

    def _simulate_trajectory_cpp(
        self, pos_traj: Callable[[float], float], duration: float, dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate trajectory using C++ nonlinear solver with torque saturation.

        Dynamics computed entirely in C++, only calls Python for trajectory function pos_traj(t).
        """
        # Pack parameters for C++
        params = pycontrol.ODriveParams(
            J_motor=self.phys.J_motor,
            J_load=self.phys.J_load,
            k_coupling=self.phys.k_coupling,
            c_coupling=self.phys.c_coupling,
            b_motor=self.phys.b_motor,
            b_load=self.phys.b_load,
            max_torque=self.phys.max_torque,
            pos_gain=self.gains.pos_gain,
            vel_gain=self.gains.vel_gain,
            vel_integrator_gain=self.gains.vel_integrator_gain,
        )

        # Generate time vector
        n_steps = int(duration / dt)
        t_eval = np.linspace(0, duration, n_steps).tolist()

        # Use C++ nonlinear solver with adaptive stepping for speed
        # Relaxed tolerance (1e-3) and larger max_step (0.1) for faster GUI updates
        try:
            result = pycontrol.solve_odrive_trajectory(
                params,
                pos_traj,  # Python callback for trajectory
                0.0,
                duration,
                t_eval,
                initial_step=0.01,  # Initial adaptive step size
                tol=1e-3,  # Relaxed tolerance for speed
                max_step=0.1,  # Allow large steps for faster computation
            )

            if result.success:
                t = np.array(result.t)
                x = np.array([np.array(xi).flatten() for xi in result.x])
                commands = np.array([pos_traj(ti) for ti in t])
                return t, x, commands
            else:
                print(
                    f"C++ ODrive solver failed: {result.message}, falling back to Python"
                )
                return self._simulate_trajectory_python(pos_traj, duration, dt)
        except Exception as e:
            print(f"Error using C++ ODrive solver: {e}")
            traceback.print_exc()
            raise


def generate_trapezoidal_profile(
    duration: float, max_vel: float, max_accel: float, distance: float
) -> Tuple[Callable[[float], float], Callable[[float], float]]:
    """
    Generate a trapezoidal velocity profile for position.

    Args:
        duration: Total move duration
        max_vel: Maximum velocity
        max_accel: Maximum acceleration
        distance: Total distance to travel

    Returns:
        Tuple of (position_func, velocity_func) that return position/velocity at time t
    """
    # Compute profile times
    t_accel = max_vel / max_accel

    # Check if we reach max velocity
    # Distance for accel+decel (symmetric triangular profile) = t_accel * max_vel
    if t_accel * max_vel > distance:
        # Triangular profile - can't reach max_vel
        # For symmetric profile: distance = 0.5 * actual_max_vel * t_accel + 0.5 * actual_max_vel * t_accel
        #                                  = actual_max_vel * t_accel
        # Also: actual_max_vel = max_accel * t_accel
        # Therefore: distance = max_accel * t_accel^2
        # Solving: t_accel = sqrt(distance / max_accel)
        t_accel = np.sqrt(distance / max_accel)
        actual_max_vel = max_accel * t_accel  # This is sqrt(max_accel * distance)
        t_cruise = 0
    else:
        # Trapezoidal profile - we do reach max_vel
        actual_max_vel = max_vel
        # Distance = d_accel + d_cruise + d_decel
        #          = 0.5 * max_accel * t_accel^2 + max_vel * t_cruise + 0.5 * max_accel * t_accel^2
        #          = max_accel * t_accel^2 + max_vel * t_cruise
        t_cruise = (distance - max_accel * t_accel**2) / max_vel

    t_decel = t_accel

    def position(t: float) -> float:
        if t < 0:
            return 0.0
        elif t < t_accel:
            # Acceleration phase
            return 0.5 * max_accel * t**2
        elif t < t_accel + t_cruise:
            # Cruise phase
            return 0.5 * max_accel * t_accel**2 + actual_max_vel * (t - t_accel)
        elif t < t_accel + t_cruise + t_decel:
            # Deceleration phase
            t_in_decel = t - t_accel - t_cruise
            cruise_pos = 0.5 * max_accel * t_accel**2 + actual_max_vel * t_cruise
            return (
                cruise_pos
                + actual_max_vel * t_in_decel
                - 0.5 * max_accel * t_in_decel**2
            )
        else:
            # Hold final position
            return distance

    def velocity(t: float) -> float:
        if t < 0:
            return 0.0
        elif t < t_accel:
            # Acceleration phase: linear ramp up
            return max_accel * t
        elif t < t_accel + t_cruise:
            # Cruise phase: constant velocity
            return actual_max_vel
        elif t < t_accel + t_cruise + t_decel:
            # Deceleration phase: linear ramp down
            t_in_decel = t - t_accel - t_cruise
            return actual_max_vel - max_accel * t_in_decel
        else:
            # Hold: zero velocity
            return 0.0

    return position, velocity


if __name__ == "__main__":
    # Quick test
    system = ODriveFlexibleSystem()

    print("Testing step response...")
    t, x = system.simulate_step(step_size=1.0, duration=1.0)
    print(f"Simulated {len(t)} points")
    print(f"Final motor position: {x[-1, 0]:.4f} rad")
    print(f"Final load position: {x[-1, 2]:.4f} rad")

    print("\nTesting trapezoidal trajectory...")
    traj = generate_trapezoidal_profile(
        duration=2.0, max_vel=10.0, max_accel=50.0, distance=10.0
    )
    t, x, cmd = system.simulate_trajectory(traj, duration=2.0)
    print(f"Simulated {len(t)} points")
    print(f"Final motor position: {x[-1, 0]:.4f} rad (commanded: {cmd[-1]:.4f})")
    print(f"Final load position: {x[-1, 2]:.4f} rad")
