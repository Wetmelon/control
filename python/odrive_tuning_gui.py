"""
ODrive Motor Tuning GUI

Interactive GUI for tuning ODrive controller parameters and visualizing system response.
Uses the libcontrol C++ library via Python bindings for accurate simulation.

Features:
- Real-time parameter adjustment via sliders
- Step response visualization
- Trapezoidal profile tracking visualization
- Stability analysis
- Motor and load position/velocity tracking
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import time
from odrive_system import (
    ODriveFlexibleSystem,
    PhysicalParameters,
    ControllerGains,
    generate_trapezoidal_profile,
)


class ODriveTuningGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ODrive Motor Controller Tuning")
        self.root.geometry("1400x900")

        # Initialize system
        self.system = ODriveFlexibleSystem()

        # Trapezoidal profile parameters (in turns/s units for display)
        self.trap_distance = 1.0  # turns
        self.trap_max_vel = 1.0  # turns/s
        self.trap_max_accel = 5.0  # turns/s²

        # Real-time update control
        self.realtime_mode = tk.BooleanVar(value=True)
        self.update_pending = False
        self.last_update_time = 0
        self.min_update_interval = 0.0167  # ~60 fps max (1/60 seconds)
        self.high_quality_mode = tk.BooleanVar(value=True)
        
        # Cache axis limits to avoid expensive autoscale
        self.axis_limits_cache = None

        # Create UI
        self._create_widgets()

        # Initial plot
        self.update_plots()

    def _create_widgets(self):
        """Create all GUI widgets"""

        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel: Controls
        control_frame = ttk.Frame(main_container, width=350)
        control_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        control_frame.pack_propagate(False)

        # Right panel: Plots
        plot_frame = ttk.Frame(main_container)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Create control sections
        self._create_controller_controls(control_frame)
        self._create_physical_controls(control_frame)
        self._create_trajectory_controls(control_frame)
        self._create_action_buttons(control_frame)

        # Create plots
        self._create_plots(plot_frame)

    def _create_controller_controls(self, parent):
        """Create controller gain sliders"""
        frame = ttk.LabelFrame(parent, text="Controller Gains", padding=10)
        frame.pack(fill=tk.X, pady=(0, 10))

        # Position gain
        self._create_slider(
            frame,
            "Position Gain (1/s)",
            0.1,
            100.0,
            self.system.gains.pos_gain,
            lambda v: setattr(self.system.gains, "pos_gain", v),
        )

        # Velocity gain
        self._create_slider(
            frame,
            "Velocity Gain (N·m·s/rad)",
            0.01,
            1.0,
            self.system.gains.vel_gain,
            lambda v: setattr(self.system.gains, "vel_gain", v),
        )

        # Velocity integrator gain
        self._create_slider(
            frame,
            "Vel Integrator Gain (N·m/rad)",
            0.0,
            2.0,
            self.system.gains.vel_integrator_gain,
            lambda v: setattr(self.system.gains, "vel_integrator_gain", v),
        )

    def _create_physical_controls(self, parent):
        """Create physical parameter sliders"""
        frame = ttk.LabelFrame(parent, text="Physical Parameters", padding=10)
        frame.pack(fill=tk.X, pady=(0, 10))

        # Motor inertia
        self._create_slider(
            frame,
            "Motor Inertia (kg·m²)",
            0.0001,
            0.01,
            self.system.phys.J_motor,
            lambda v: setattr(self.system.phys, "J_motor", v),
            resolution=0.0001,
        )

        # Load inertia
        self._create_slider(
            frame,
            "Load Inertia (kg·m²)",
            0.001,
            0.02,
            self.system.phys.J_load,
            lambda v: setattr(self.system.phys, "J_load", v),
            resolution=0.0001,
        )

        # Coupling stiffness
        self._create_slider(
            frame,
            "Coupling Stiffness (N·m/rad)",
            10.0,
            500.0,
            self.system.phys.k_coupling,
            lambda v: setattr(self.system.phys, "k_coupling", v),
        )

        # Coupling damping
        self._create_slider(
            frame,
            "Coupling Damping (N·m·s/rad)",
            0.01,
            2.0,
            self.system.phys.c_coupling,
            lambda v: setattr(self.system.phys, "c_coupling", v),
            resolution=0.01,
        )

        # Maximum motor torque
        self._create_slider(
            frame,
            "Max Motor Torque (N·m)",
            0.1,
            10.0,
            self.system.phys.max_torque,
            lambda v: setattr(self.system.phys, "max_torque", v),
            resolution=0.1,
        )

    def _create_trajectory_controls(self, parent):
        """Create trapezoidal trajectory parameter sliders"""
        frame = ttk.LabelFrame(parent, text="Trapezoidal Profile", padding=10)
        frame.pack(fill=tk.X, pady=(0, 10))

        # Distance
        self._create_slider(
            frame,
            "Distance (turns)",
            0.1,
            5.0,
            self.trap_distance,
            lambda v: setattr(self, "trap_distance", v),
        )

        # Max velocity
        self._create_slider(
            frame,
            "Max Velocity (turns/s)",
            0.1,
            10.0,
            self.trap_max_vel,
            lambda v: setattr(self, "trap_max_vel", v),
        )

        # Max acceleration
        self._create_slider(
            frame,
            "Max Acceleration (turns/s²)",
            1.0,
            50.0,
            self.trap_max_accel,
            lambda v: setattr(self, "trap_max_accel", v),
        )

    def _create_slider(
        self, parent, label, min_val, max_val, initial_val, callback, resolution=None
    ):
        """Create a labeled slider with value display"""
        container = ttk.Frame(parent)
        container.pack(fill=tk.X, pady=5)

        # Label
        label_widget = ttk.Label(container, text=label)
        label_widget.pack(anchor=tk.W)

        # Slider and value frame
        slider_frame = ttk.Frame(container)
        slider_frame.pack(fill=tk.X)

        # Value display
        value_var = tk.StringVar(value=f"{initial_val:.4g}")
        value_label = ttk.Label(
            slider_frame, textvariable=value_var, width=10, anchor=tk.E
        )
        value_label.pack(side=tk.RIGHT)

        # Slider
        if resolution is None:
            resolution = (max_val - min_val) / 1000

        slider = ttk.Scale(
            slider_frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL
        )
        slider.set(initial_val)
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        def on_change(event):
            val = slider.get()
            value_var.set(f"{val:.4g}")
            callback(val)

            # Trigger real-time update if enabled
            if self.realtime_mode.get():
                self._schedule_update()

        slider.bind("<ButtonRelease-1>", on_change)
        slider.bind("<B1-Motion>", on_change)

    def _create_action_buttons(self, parent):
        """Create action buttons"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=10)

        # Real-time mode toggle
        realtime_check = ttk.Checkbutton(
            frame,
            text="Real-time Updates",
            variable=self.realtime_mode,
            command=self._on_realtime_toggle,
        )
        realtime_check.pack(fill=tk.X, pady=2)

        # Quality mode toggle
        quality_check = ttk.Checkbutton(
            frame, text="High Quality Mode (slower)", variable=self.high_quality_mode
        )
        quality_check.pack(fill=tk.X, pady=2)

        # Solver selection
        solver_frame = ttk.Frame(frame)
        solver_frame.pack(fill=tk.X, pady=5)
        ttk.Label(solver_frame, text="Solver:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.solver_var = tk.StringVar(value="C++")
        solver_combo = ttk.Combobox(
            solver_frame,
            textvariable=self.solver_var,
            values=["C++", "SciPy"],
            state="readonly",
            width=10
        )
        solver_combo.pack(side=tk.LEFT)
        solver_combo.bind("<<ComboboxSelected>>", lambda e: self._on_solver_change())

        # Manual update button
        update_btn = ttk.Button(
            frame, text="Update Plots (Manual)", command=self.update_plots
        )
        update_btn.pack(fill=tk.X, pady=5)

        # FPS display
        self.fps_var = tk.StringVar(value="FPS: --")
        fps_label = ttk.Label(frame, textvariable=self.fps_var, font=("Arial", 9))
        fps_label.pack(pady=2)

        # Performance display
        self.perf_var = tk.StringVar(value="")
        perf_label = ttk.Label(frame, textvariable=self.perf_var, font=("Arial", 8), foreground="gray")
        perf_label.pack(pady=2)

        # Info label
        info_label = ttk.Label(
            frame,
            text="Real-time mode updates as you drag.\n"
            "Disable for manual control.\n"
            "High quality uses more points (slower).",
            wraplength=300,
            justify=tk.LEFT,
            foreground="gray",
        )
        info_label.pack(pady=10)

    def _on_solver_change(self):
        """Handle solver selection change"""
        self.system.use_scipy = (self.solver_var.get() == "SciPy")
        if self.realtime_mode.get():
            self._schedule_update()

    def _on_realtime_toggle(self):
        """Handle real-time mode toggle"""
        if self.realtime_mode.get():
            self._schedule_update()

    def _schedule_update(self):
        """Schedule a plot update with rate limiting"""
        # If an update is already pending, skip this request to prevent queue buildup
        # This ensures smooth dragging without lag accumulation
        if self.update_pending:
            return
            
        # Check if enough time has passed since last update
        current_time = time.time()
        time_since_last = current_time - self.last_update_time

        if time_since_last >= self.min_update_interval:
            # Update immediately
            self.update_pending = True
            self.root.after(1, self._do_update)
        else:
            # Schedule for later
            delay_ms = int((self.min_update_interval - time_since_last) * 1000)
            self.update_pending = True
            self.root.after(delay_ms, self._do_update)

    def _do_update(self):
        """Actually perform the update"""
        start_time = time.time()
        self.update_plots()
        self.update_pending = False
        
        # Calculate and display actual FPS (frames per second)
        time_since_last_frame = start_time - self.last_update_time
        actual_fps = 1.0 / time_since_last_frame if time_since_last_frame > 0 else 0
        
        self.last_update_time = time.time()
        elapsed = self.last_update_time - start_time
        
        self.fps_var.set(f"FPS: {actual_fps:.1f} (frame: {time_since_last_frame*1000:.0f}ms, render: {elapsed*1000:.0f}ms)")

    def _create_plots(self, parent):
        """Create matplotlib plots"""

        # Create figure with subplots
        self.fig = Figure(figsize=(10, 8), dpi=100)

        # Step response plots
        self.ax_step_pos = self.fig.add_subplot(3, 2, 1)
        self.ax_step_vel = self.fig.add_subplot(3, 2, 3)

        # Trajectory tracking plots
        self.ax_traj_pos = self.fig.add_subplot(3, 2, 2)
        self.ax_traj_vel = self.fig.add_subplot(3, 2, 4)

        # Torque plots
        self.ax_step_torque = self.fig.add_subplot(3, 2, 5)
        self.ax_traj_torque = self.fig.add_subplot(3, 2, 6)

        # Create line objects for step response
        (self.line_step_motor_pos,) = self.ax_step_pos.plot(
            [], [], label="Motor Position", linewidth=2
        )
        (self.line_step_load_pos,) = self.ax_step_pos.plot(
            [], [], label="Load Position", linewidth=2, linestyle="--"
        )
        (self.line_step_cmd,) = self.ax_step_pos.plot(
            [], [], color="k", linestyle=":", alpha=0.5, label="Command"
        )
        self.ax_step_pos.set_xlabel("Time (s)")
        self.ax_step_pos.set_ylabel("Position (turns)")
        self.ax_step_pos.set_title("Step Response - Position")
        self.ax_step_pos.legend()
        self.ax_step_pos.grid(True, alpha=0.3)

        (self.line_step_motor_vel,) = self.ax_step_vel.plot(
            [], [], label="Motor Velocity", linewidth=2
        )
        (self.line_step_load_vel,) = self.ax_step_vel.plot(
            [], [], label="Load Velocity", linewidth=2, linestyle="--"
        )
        self.ax_step_vel.set_xlabel("Time (s)")
        self.ax_step_vel.set_ylabel("Velocity (turns/s)")
        self.ax_step_vel.set_title("Step Response - Velocity")
        self.ax_step_vel.legend()
        self.ax_step_vel.grid(True, alpha=0.3)

        (self.line_step_torque,) = self.ax_step_torque.plot(
            [], [], label="Motor Torque", linewidth=2, color="C2"
        )
        (self.line_step_torque_limit,) = self.ax_step_torque.plot(
            [], [], color="r", linestyle="--", alpha=0.5, label="Torque Limit"
        )
        self.ax_step_torque.set_xlabel("Time (s)")
        self.ax_step_torque.set_ylabel("Torque (N·m)")
        self.ax_step_torque.set_title("Step Response - Motor Torque")
        self.ax_step_torque.legend()
        self.ax_step_torque.grid(True, alpha=0.3)

        # Create line objects for trajectory
        (self.line_traj_cmd_pos,) = self.ax_traj_pos.plot(
            [], [], label="Command", linewidth=2, color="k", linestyle=":"
        )
        (self.line_traj_motor_pos,) = self.ax_traj_pos.plot(
            [], [], label="Motor Position", linewidth=2
        )
        (self.line_traj_load_pos,) = self.ax_traj_pos.plot(
            [], [], label="Load Position", linewidth=2, linestyle="--"
        )
        self.ax_traj_pos.set_xlabel("Time (s)")
        self.ax_traj_pos.set_ylabel("Position (turns)")
        self.ax_traj_pos.set_title("Trapezoidal Profile - Position")
        self.ax_traj_pos.legend()
        self.ax_traj_pos.grid(True, alpha=0.3)

        (self.line_traj_cmd_vel,) = self.ax_traj_vel.plot(
            [], [], label="Command Velocity", linewidth=2, color="k", linestyle=":"
        )
        (self.line_traj_motor_vel,) = self.ax_traj_vel.plot(
            [], [], label="Motor Velocity", linewidth=2
        )
        (self.line_traj_load_vel,) = self.ax_traj_vel.plot(
            [], [], label="Load Velocity", linewidth=2, linestyle="--"
        )
        self.ax_traj_vel.set_xlabel("Time (s)")
        self.ax_traj_vel.set_ylabel("Velocity (turns/s)")
        self.ax_traj_vel.set_title("Trapezoidal Profile - Velocity")
        self.ax_traj_vel.legend()
        self.ax_traj_vel.grid(True, alpha=0.3)

        (self.line_traj_torque,) = self.ax_traj_torque.plot(
            [], [], label="Motor Torque", linewidth=2, color="C2"
        )
        (self.line_traj_torque_limit,) = self.ax_traj_torque.plot(
            [], [], color="r", linestyle="--", alpha=0.5, label="Torque Limit"
        )
        self.ax_traj_torque.set_xlabel("Time (s)")
        self.ax_traj_torque.set_ylabel("Torque (N·m)")
        self.ax_traj_torque.set_title("Trapezoidal Profile - Motor Torque")
        self.ax_traj_torque.legend()
        self.ax_traj_torque.grid(True, alpha=0.3)

        self.fig.tight_layout(pad=3.0)

        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_plots(self):
        """Update all plots with current parameters"""

        import time

        t_start_total = time.perf_counter()

        # Determine simulation resolution based on quality mode
        if self.high_quality_mode.get():
            dt = 0.001  # High quality: 1ms steps
            step_duration = 2.0
        else:
            dt = 0.004  # Fast mode: 4ms steps (5x faster)
            step_duration = 1.5

        # Calculate trajectory duration based on profile parameters (in radians)
        distance_rad = self.trap_distance * 2 * np.pi
        max_vel_rad = self.trap_max_vel * 2 * np.pi
        max_accel_rad = self.trap_max_accel * 2 * np.pi

        # Time to reach max velocity
        t_accel = max_vel_rad / max_accel_rad

        # Check if we can reach max velocity
        if 2 * t_accel * max_vel_rad > distance_rad:
            # Triangular profile - can't reach max vel
            traj_duration = 2 * np.sqrt(distance_rad / max_accel_rad)
        else:
            # Trapezoidal profile
            t_cruise = (distance_rad - max_accel_rad * t_accel**2) / max_vel_rad
            traj_duration = 2 * t_accel + t_cruise

        # Add 50% extra time for settling
        traj_duration = traj_duration * 1.5
        traj_duration = max(traj_duration, 0.5)  # Minimum 0.5s

        # Store dt for metrics calculation
        self.current_dt = dt
        self.current_duration = step_duration

        # Simulate step response (1 turn = 2π rad)
        t_sim_start = time.perf_counter()
        t_step, x_step = self.system.simulate_step(
            step_size=2 * np.pi, duration=step_duration, dt=dt
        )
        t_sim_step = time.perf_counter() - t_sim_start

        # Simulate trajectory tracking (convert turns to radians)
        t_sim_start = time.perf_counter()
        traj_pos, traj_vel = generate_trapezoidal_profile(
            duration=traj_duration,
            max_vel=self.trap_max_vel * 2 * np.pi,
            max_accel=self.trap_max_accel * 2 * np.pi,
            distance=self.trap_distance * 2 * np.pi,
        )
        
        # Pass trajectory parameters for pure C++ implementation when not using scipy
        traj_params = None if self.system.use_scipy else {
            'distance': self.trap_distance * 2 * np.pi,
            'max_vel': self.trap_max_vel * 2 * np.pi,
            'max_accel': self.trap_max_accel * 2 * np.pi,
        }
        
        t_traj, x_traj, cmd_traj = self.system.simulate_trajectory(
            traj_pos, duration=traj_duration, dt=dt, traj_params=traj_params
        )
        t_sim_traj = time.perf_counter() - t_sim_start

        # Update line data instead of clearing and recreating (much faster!)
        t_plot_start = time.perf_counter()

        # Update step response - Position
        self.line_step_motor_pos.set_data(t_step, x_step[:, 0] / (2 * np.pi))
        self.line_step_load_pos.set_data(t_step, x_step[:, 2] / (2 * np.pi))
        self.line_step_cmd.set_data([t_step[0], t_step[-1]], [1.0, 1.0])

        # Update step response - Velocity
        self.line_step_motor_vel.set_data(t_step, x_step[:, 1] / (2 * np.pi))
        self.line_step_load_vel.set_data(t_step, x_step[:, 3] / (2 * np.pi))

        # Update step response - Torque
        # Compute motor torque: tau = Kv*(vel_setpoint - omega_m) + Ki*integrator
        # where vel_setpoint = Kp*(pos_cmd - theta_m)
        pos_cmd_step = 2 * np.pi  # 1 turn command
        vel_setpoint_step = self.system.gains.pos_gain * (pos_cmd_step - x_step[:, 0])
        vel_error_step = vel_setpoint_step - x_step[:, 1]
        torque_step = self.system.gains.vel_gain * vel_error_step + self.system.gains.vel_integrator_gain * x_step[:, 4]
        torque_step = np.clip(torque_step, -self.system.phys.max_torque, self.system.phys.max_torque)
        
        self.line_step_torque.set_data(t_step, torque_step)
        self.line_step_torque_limit.set_data([t_step[0], t_step[-1]], [self.system.phys.max_torque, self.system.phys.max_torque])

        # Update trajectory tracking - Position
        self.line_traj_cmd_pos.set_data(t_traj, cmd_traj / (2 * np.pi))
        self.line_traj_motor_pos.set_data(t_traj, x_traj[:, 0] / (2 * np.pi))
        self.line_traj_load_pos.set_data(t_traj, x_traj[:, 2] / (2 * np.pi))

        # Update trajectory tracking - Velocity
        cmd_vel = np.array([traj_vel(t) for t in t_traj]) / (2 * np.pi)
        self.line_traj_cmd_vel.set_data(t_traj, cmd_vel)
        self.line_traj_motor_vel.set_data(t_traj, x_traj[:, 1] / (2 * np.pi))
        self.line_traj_load_vel.set_data(t_traj, x_traj[:, 3] / (2 * np.pi))

        # Update trajectory tracking - Torque
        # Compute motor torque: tau = Kv*(vel_setpoint - omega_m) + Ki*integrator
        # where vel_setpoint = Kp*(pos_cmd - theta_m)
        vel_setpoint_traj = self.system.gains.pos_gain * (cmd_traj - x_traj[:, 0])
        vel_error_traj = vel_setpoint_traj - x_traj[:, 1]
        torque_traj = self.system.gains.vel_gain * vel_error_traj + self.system.gains.vel_integrator_gain * x_traj[:, 4]
        torque_traj = np.clip(torque_traj, -self.system.phys.max_torque, self.system.phys.max_torque)
        
        self.line_traj_torque.set_data(t_traj, torque_traj)
        self.line_traj_torque_limit.set_data([t_traj[0], t_traj[-1]], [self.system.phys.max_torque, self.system.phys.max_torque])
        
        # Smart axis limit updates - only rescale when needed
        self._update_axis_limits(t_step, x_step, t_traj, x_traj, cmd_traj, torque_step, torque_traj, cmd_vel)

        # Compute and display stability metrics
        self._compute_stability_metrics(x_step, x_traj, cmd_traj)

        # Redraw canvas immediately (don't defer with draw_idle for realtime responsiveness)
        self.canvas.draw()
        t_render = time.perf_counter() - t_plot_start

        # Performance logging
        t_total = time.perf_counter() - t_start_total
        t_sim_total = t_sim_step + t_sim_traj
        t_plot_total = t_render

        solver_name = self.solver_var.get()
        traj_real_time_ratio = traj_duration / t_sim_traj if t_sim_traj > 0 else float('inf')
        print(
            f"Performance: Total={t_total*1000:.1f}ms | "
            f"Sim={t_sim_total*1000:.1f}ms (step={t_sim_step*1000:.1f}ms, traj={t_sim_traj*1000:.1f}ms ({traj_real_time_ratio:.2f}x)) [{solver_name}] | "
            f"Plot={t_plot_total*1000:.1f}ms"
        )
        
        # Update performance display
        self.perf_var.set(f"Sim: {t_sim_total*1000:.0f}ms | Plot: {t_plot_total*1000:.0f}ms")

    def _compute_stability_metrics(self, x_step, x_traj, cmd_traj):
        """Compute and display stability metrics"""

        # Step response metrics (convert to turns for comparison)
        motor_pos_step = x_step[:, 0] / (2 * np.pi)  # Convert to turns

        # Settling time (2% criterion)
        final_value = motor_pos_step[-1]
        tolerance = 0.02 * abs(1.0)  # 2% of step size (1 turn)
        settled_mask = np.abs(motor_pos_step - 1.0) <= tolerance

        if np.any(settled_mask):
            settling_idx = np.where(settled_mask)[0][0]
            # Make sure it stays settled
            if np.all(settled_mask[settling_idx:]):
                settling_time = settling_idx * self.current_dt  # Use actual dt
            else:
                settling_time = None
        else:
            settling_time = None

        # Overshoot
        max_val = np.max(motor_pos_step)
        overshoot = (max_val - 1.0) / 1.0 * 100  # Percentage

        # Steady-state error
        ss_error = abs(1.0 - motor_pos_step[-1])

        # Trajectory tracking error (RMS) - convert to turns
        motor_error = (cmd_traj - x_traj[:, 0]) / (2 * np.pi)
        rms_error = np.sqrt(np.mean(motor_error**2))
        max_error = np.max(np.abs(motor_error))

        # Display in title
        title_text = (
            f"Step: Overshoot={overshoot:.1f}%, "
            f"Settling={'N/A' if settling_time is None else f'{settling_time:.3f}s'}, "
            f"SS Error={ss_error:.4f} | "
            f"Traj: RMS={rms_error:.4f}, Max={max_error:.4f}"
        )

        self.fig.suptitle(title_text, fontsize=10)

    def _update_axis_limits(self, t_step, x_step, t_traj, x_traj, cmd_traj, torque_step, torque_traj, cmd_vel):
        """Smart axis limit updates - only rescale when data significantly changes"""
        
        # Compute current data ranges
        pos_min_step = min(np.min(x_step[:, 0] / (2 * np.pi)), np.min(x_step[:, 2] / (2 * np.pi)))
        pos_max_step = max(np.max(x_step[:, 0] / (2 * np.pi)), np.max(x_step[:, 2] / (2 * np.pi)), 1.0)
        vel_min_step = min(np.min(x_step[:, 1] / (2 * np.pi)), np.min(x_step[:, 3] / (2 * np.pi)))
        vel_max_step = max(np.max(x_step[:, 1] / (2 * np.pi)), np.max(x_step[:, 3] / (2 * np.pi)))
        torque_min = np.min(torque_step)
        torque_max = max(np.max(torque_step), self.system.phys.max_torque)
        
        pos_min_traj = min(np.min(x_traj[:, 0] / (2 * np.pi)), np.min(x_traj[:, 2] / (2 * np.pi)), np.min(cmd_traj / (2 * np.pi)))
        pos_max_traj = max(np.max(x_traj[:, 0] / (2 * np.pi)), np.max(x_traj[:, 2] / (2 * np.pi)), np.max(cmd_traj / (2 * np.pi)))
        vel_min_traj = min(np.min(x_traj[:, 1] / (2 * np.pi)), np.min(x_traj[:, 3] / (2 * np.pi)), np.min(cmd_vel))
        vel_max_traj = max(np.max(x_traj[:, 1] / (2 * np.pi)), np.max(x_traj[:, 3] / (2 * np.pi)), np.max(cmd_vel))
        torque_min_traj = np.min(torque_traj)
        torque_max_traj = max(np.max(torque_traj), self.system.phys.max_torque)
        
        current_limits = {
            'step_pos': (pos_min_step, pos_max_step),
            'step_vel': (vel_min_step, vel_max_step),
            'step_torque': (torque_min, torque_max),
            'traj_pos': (pos_min_traj, pos_max_traj),
            'traj_vel': (vel_min_traj, vel_max_traj),
            'traj_torque': (torque_min_traj, torque_max_traj),
            'time_step': (t_step[0], t_step[-1]),
            'time_traj': (t_traj[0], t_traj[-1]),
            'max_torque': (self.system.phys.max_torque, self.system.phys.max_torque)  # Track parameter changes
        }
        
        # Check if limits changed significantly (>10%) or if first time
        needs_update = self.axis_limits_cache is None
        if not needs_update and self.axis_limits_cache:
            for key, (new_min, new_max) in current_limits.items():
                old_min, old_max = self.axis_limits_cache[key]
                range_old = old_max - old_min
                range_new = new_max - new_min
                if range_old == 0:
                    range_old = 1e-6
                change = abs(range_new - range_old) / range_old
                if change > 0.1 or new_min < old_min or new_max > old_max:
                    needs_update = True
                    break
        
        if needs_update:
            # Add 5% margin
            margin = 0.05
            self.ax_step_pos.set_xlim(t_step[0], t_step[-1])
            self.ax_step_pos.set_ylim(pos_min_step - margin * (pos_max_step - pos_min_step), 
                                      pos_max_step + margin * (pos_max_step - pos_min_step))
            
            self.ax_step_vel.set_xlim(t_step[0], t_step[-1])
            vel_range = vel_max_step - vel_min_step
            self.ax_step_vel.set_ylim(vel_min_step - margin * vel_range, 
                                      vel_max_step + margin * vel_range)
            
            self.ax_step_torque.set_xlim(t_step[0], t_step[-1])
            torque_range = torque_max - torque_min
            self.ax_step_torque.set_ylim(torque_min - margin * torque_range, 
                                         torque_max + margin * torque_range)
            
            self.ax_traj_pos.set_xlim(t_traj[0], t_traj[-1])
            self.ax_traj_pos.set_ylim(pos_min_traj - margin * (pos_max_traj - pos_min_traj), 
                                      pos_max_traj + margin * (pos_max_traj - pos_min_traj))
            
            self.ax_traj_vel.set_xlim(t_traj[0], t_traj[-1])
            vel_range_traj = vel_max_traj - vel_min_traj
            self.ax_traj_vel.set_ylim(vel_min_traj - margin * vel_range_traj, 
                                      vel_max_traj + margin * vel_range_traj)
            
            self.ax_traj_torque.set_xlim(t_traj[0], t_traj[-1])
            torque_range_traj = torque_max_traj - torque_min_traj
            self.ax_traj_torque.set_ylim(torque_min_traj - margin * torque_range_traj, 
                                         torque_max_traj + margin * torque_range_traj)
            
            # Cache the new limits
            self.axis_limits_cache = current_limits


def main():
    root = tk.Tk()
    app = ODriveTuningGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
