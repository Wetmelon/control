"""
PMSM Servo Drive Real-Time Simulation GUI

Wraps the C++ servo_sim.dll via ctypes and provides a real-time interactive
GUI with parameter sliders, live plots, and start/stop/reset controls.

Requirements:
    pip install dearpygui

Usage:
    1. Build servo_sim.dll (see servo_sim.h for build instructions)
    2. python servo_gui.py
       or: python servo_gui.py --dll path/to/servo_sim.dll
"""

import ctypes
import os
import sys
import time
import argparse
import collections
from pathlib import Path

import dearpygui.dearpygui as dpg

# ===== C struct mirrors =====

class ServoMotorParams(ctypes.Structure):
    _fields_ = [
        ("Rs", ctypes.c_double),
        ("Ls", ctypes.c_double),
        ("lambda_pm", ctypes.c_double),
        ("pole_pairs", ctypes.c_int),
        ("Jm", ctypes.c_double),
        ("Bm", ctypes.c_double),
        ("Ks", ctypes.c_double),
        ("Bs", ctypes.c_double),
        ("JL", ctypes.c_double),
        ("BL", ctypes.c_double),
        ("Tc", ctypes.c_double),
        ("Vdc", ctypes.c_double),
    ]

class ServoControlParams(ctypes.Structure):
    _fields_ = [
        ("bw_current", ctypes.c_double),
        ("Kp_speed", ctypes.c_double),
        ("Ki_speed", ctypes.c_double),
        ("Kp_position", ctypes.c_double),
        ("i_max", ctypes.c_double),
        ("speed_max", ctypes.c_double),
        ("speed_ratio", ctypes.c_int),
        ("position_ratio", ctypes.c_int),
        ("position_mode", ctypes.c_int),
        ("ref_filter_bw", ctypes.c_double),
    ]

class ServoState(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_double),
        ("iq", ctypes.c_double),
        ("omega_m", ctypes.c_double),
        ("theta_m", ctypes.c_double),
        ("omega_L", ctypes.c_double),
        ("theta_L", ctypes.c_double),
        ("vd", ctypes.c_double),
        ("vq", ctypes.c_double),
        ("iq_ref", ctypes.c_double),
        ("omega_ref", ctypes.c_double),
        ("theta_ref", ctypes.c_double),
        ("Te", ctypes.c_double),
        ("Tshaft", ctypes.c_double),
        ("twist", ctypes.c_double),
        ("P_elec", ctypes.c_double),
        ("P_mech", ctypes.c_double),
        ("speed_err", ctypes.c_double),
        ("pos_err", ctypes.c_double),
        ("omega_e", ctypes.c_double),
        ("int_id", ctypes.c_double),
        ("int_iq", ctypes.c_double),
        ("int_spd", ctypes.c_double),
        ("t", ctypes.c_double),
        ("step_count", ctypes.c_long),
        ("max_dxdt", ctypes.c_double),
        ("tau_elec", ctypes.c_double),
        ("tau_mech", ctypes.c_double),
        ("omega_shaft", ctypes.c_double),
        ("lambda_max", ctypes.c_double),
        ("rk4_margin", ctypes.c_double),
        ("courant_elec", ctypes.c_double),
        ("courant_mech", ctypes.c_double),
        ("dt_used", ctypes.c_double),
        ("plant_substeps", ctypes.c_int),
    ]

# ===== DLL wrapper =====

class ServoSimDLL:
    def __init__(self, dll_path: str):
        self.lib = ctypes.CDLL(dll_path)

        # servo_create
        self.lib.servo_create.restype = ctypes.c_void_p
        self.lib.servo_create.argtypes = []

        # servo_destroy
        self.lib.servo_destroy.restype = None
        self.lib.servo_destroy.argtypes = [ctypes.c_void_p]

        # servo_set_motor_params
        self.lib.servo_set_motor_params.restype = None
        self.lib.servo_set_motor_params.argtypes = [ctypes.c_void_p, ctypes.POINTER(ServoMotorParams)]

        # servo_set_control_params
        self.lib.servo_set_control_params.restype = None
        self.lib.servo_set_control_params.argtypes = [ctypes.c_void_p, ctypes.POINTER(ServoControlParams)]

        # servo_set_speed_ref
        self.lib.servo_set_speed_ref.restype = None
        self.lib.servo_set_speed_ref.argtypes = [ctypes.c_void_p, ctypes.c_double]

        # servo_set_position_ref
        self.lib.servo_set_position_ref.restype = None
        self.lib.servo_set_position_ref.argtypes = [ctypes.c_void_p, ctypes.c_double]

        # servo_set_load_torque
        self.lib.servo_set_load_torque.restype = None
        self.lib.servo_set_load_torque.argtypes = [ctypes.c_void_p, ctypes.c_double]

        # servo_step
        self.lib.servo_step.restype = None
        self.lib.servo_step.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_int]

        # servo_get_state
        self.lib.servo_get_state.restype = ServoState
        self.lib.servo_get_state.argtypes = [ctypes.c_void_p]

        # servo_reset
        self.lib.servo_reset.restype = None
        self.lib.servo_reset.argtypes = [ctypes.c_void_p]

        self.handle = self.lib.servo_create()

    def destroy(self):
        if self.handle:
            self.lib.servo_destroy(self.handle)
            self.handle = None

    def set_motor_params(self, params: ServoMotorParams):
        self.lib.servo_set_motor_params(self.handle, ctypes.byref(params))

    def set_control_params(self, params: ServoControlParams):
        self.lib.servo_set_control_params(self.handle, ctypes.byref(params))

    def set_speed_ref(self, omega: float):
        self.lib.servo_set_speed_ref(self.handle, omega)

    def set_position_ref(self, theta: float):
        self.lib.servo_set_position_ref(self.handle, theta)

    def set_load_torque(self, torque: float):
        self.lib.servo_set_load_torque(self.handle, torque)

    def step(self, dt: float, n_steps: int):
        self.lib.servo_step(self.handle, dt, n_steps)

    def get_state(self) -> ServoState:
        return self.lib.servo_get_state(self.handle)

    def reset(self):
        self.lib.servo_reset(self.handle)


# ===== Application =====

# Simulation rate: 8 kHz control loop
_TWO_PI = 2.0 * 3.14159265358979323846
DT = 125e-6          # 125 µs control period
STEPS_PER_FRAME = 10  # steps per frame (used when RT lock is off)
HISTORY_SECONDS = 1.0
MAX_SAMPLE_RATE = 300 # max expected GUI frame rate (one sample per frame)
HISTORY_LEN = int(HISTORY_SECONDS * MAX_SAMPLE_RATE)

# All ring buffers in a list for easy resizing
t_hist = collections.deque(maxlen=HISTORY_LEN)
omega_m_hist = collections.deque(maxlen=HISTORY_LEN)
omega_L_hist = collections.deque(maxlen=HISTORY_LEN)
omega_ref_hist = collections.deque(maxlen=HISTORY_LEN)
id_hist = collections.deque(maxlen=HISTORY_LEN)
iq_hist = collections.deque(maxlen=HISTORY_LEN)
vd_hist = collections.deque(maxlen=HISTORY_LEN)
vq_hist = collections.deque(maxlen=HISTORY_LEN)
Te_hist = collections.deque(maxlen=HISTORY_LEN)
Tshaft_hist = collections.deque(maxlen=HISTORY_LEN)
twist_hist = collections.deque(maxlen=HISTORY_LEN)
theta_L_hist = collections.deque(maxlen=HISTORY_LEN)
theta_ref_hist = collections.deque(maxlen=HISTORY_LEN)
ALL_HISTS = [t_hist, omega_m_hist, omega_L_hist, omega_ref_hist,
             id_hist, iq_hist, vd_hist, vq_hist, Te_hist, Tshaft_hist, twist_hist,
             theta_L_hist, theta_ref_hist]

running = False
sim: ServoSimDLL = None

# FPS tracking
_fps_last_time = 0.0
_fps_frame_count = 0
_fps_value = 0.0

# Real-time ratio tracking
_rt_last_wall = 0.0
_rt_last_sim = 0.0
_rt_ratio = 0.0

# Real-time lock (on by default)
_realtime_lock = True
_rt_lock_wall = 0.0
_rt_lock_residual = 0.0


def clear_history():
    for h in ALL_HISTS:
        h.clear()


def resize_history(new_seconds: float):
    """Resize all ring buffers to hold new_seconds of data."""
    global HISTORY_SECONDS, HISTORY_LEN, ALL_HISTS
    HISTORY_SECONDS = new_seconds
    HISTORY_LEN = max(2, int(HISTORY_SECONDS * MAX_SAMPLE_RATE))
    new_hists = []
    for old in ALL_HISTS:
        h = collections.deque(old, maxlen=HISTORY_LEN)
        new_hists.append(h)
    ALL_HISTS[:] = new_hists
    # Rebind module-level names
    g = globals()
    names = ['t_hist', 'omega_m_hist', 'omega_L_hist', 'omega_ref_hist',
             'id_hist', 'iq_hist', 'vd_hist', 'vq_hist', 'Te_hist', 'Tshaft_hist', 'twist_hist',
             'theta_L_hist', 'theta_ref_hist']
    for i, name in enumerate(names):
        g[name] = ALL_HISTS[i]


def default_motor_params() -> ServoMotorParams:
    p = ServoMotorParams()
    p.Rs = 1.2
    p.Ls = 4.7e-3
    p.lambda_pm = 0.1
    p.pole_pairs = 4
    p.Jm = 5e-5
    p.Bm = 1e-4
    p.Ks = 5000.0
    p.Bs = 0.05
    p.JL = 5e-4
    p.BL = 1e-3
    p.Tc = 0.02
    p.Vdc = 48.0
    return p


def default_control_params() -> ServoControlParams:
    c = ServoControlParams()
    c.bw_current = 1000.0
    c.Kp_speed = 0.3
    c.Ki_speed = 18.0
    c.Kp_position = 60.0
    c.i_max = 10.0
    c.speed_max = 500.0
    c.speed_ratio = 1
    c.position_ratio = 1
    c.position_mode = 1
    c.ref_filter_bw = 5.0
    return c


def push_motor_params():
    """Read GUI sliders and push motor params to the DLL."""
    p = ServoMotorParams()
    p.Rs = dpg.get_value("sl_Rs")
    p.Ls = dpg.get_value("sl_Ls") * 1e-3
    p.lambda_pm = dpg.get_value("sl_lambda")
    p.pole_pairs = int(dpg.get_value("sl_poles"))
    p.Jm = dpg.get_value("sl_Jm")
    p.Bm = dpg.get_value("sl_Bm")
    p.Ks = dpg.get_value("sl_Ks")
    p.Bs = dpg.get_value("sl_Bs")
    p.JL = dpg.get_value("sl_JL")
    p.BL = dpg.get_value("sl_BL")
    p.Tc = dpg.get_value("sl_Tc")
    p.Vdc = dpg.get_value("sl_Vdc")
    sim.set_motor_params(p)


def push_control_params():
    """Read GUI sliders and push control params to the DLL."""
    c = ServoControlParams()
    c.bw_current = dpg.get_value("sl_bw_current")
    c.Kp_speed = dpg.get_value("sl_Kp_speed")
    c.Ki_speed = dpg.get_value("sl_Ki_speed")
    c.Kp_position = dpg.get_value("sl_Kp_position")
    c.i_max = dpg.get_value("sl_i_max")
    c.speed_max = dpg.get_value("sl_speed_max") * _TWO_PI
    c.speed_ratio = int(dpg.get_value("sl_speed_ratio"))
    c.position_ratio = int(dpg.get_value("sl_pos_ratio"))
    c.position_mode = 1 if dpg.get_value("cb_pos_mode") else 0
    c.ref_filter_bw = dpg.get_value("sl_ref_filter_bw")
    sim.set_control_params(c)


def on_toggle(sender, app_data):
    global running, _rt_lock_wall, _rt_lock_residual
    running = not running
    dpg.configure_item("btn_toggle", label="Stop" if running else "Start")
    if running:
        _rt_lock_wall = time.perf_counter()
        _rt_lock_residual = 0.0

def on_reset(sender, app_data):
    global running
    running = False
    dpg.configure_item("btn_toggle", label="Start")
    sim.reset()
    clear_history()

def on_motor_change(sender, app_data):
    push_motor_params()

def on_control_change(sender, app_data):
    push_control_params()

def on_speed_ref_change(sender, app_data):
    sim.set_speed_ref(dpg.get_value("sl_omega_ref") * _TWO_PI)

def on_position_ref_change(sender, app_data):
    sim.set_position_ref(dpg.get_value("sl_theta_ref") * _TWO_PI)

def on_load_torque_change(sender, app_data):
    sim.set_load_torque(dpg.get_value("sl_T_load"))

def on_reset_pos_ref(sender, app_data):
    dpg.set_value("sl_theta_ref", 0.0)
    sim.set_position_ref(0.0)

def on_reset_speed_ref(sender, app_data):
    dpg.set_value("sl_omega_ref", 0.0)
    sim.set_speed_ref(0.0)

def on_reset_load_torque(sender, app_data):
    dpg.set_value("sl_T_load", 0.0)
    sim.set_load_torque(0.0)

def on_reset_load_torque(sender, app_data):
    dpg.set_value("sl_T_load", 0.0)
    sim.set_load_torque(0.0)

def on_sim_speed_change(sender, app_data):
    global STEPS_PER_FRAME
    STEPS_PER_FRAME = int(dpg.get_value("sl_sim_speed"))

def on_realtime_lock_change(sender, app_data):
    global _realtime_lock, _rt_lock_wall, _rt_lock_residual
    _realtime_lock = dpg.get_value("cb_rt_lock")
    _rt_lock_wall = time.perf_counter()
    _rt_lock_residual = 0.0
    dpg.configure_item("sl_sim_speed", enabled=not _realtime_lock)

def on_dt_change(sender, app_data):
    global DT
    DT = int(dpg.get_value("sl_dt")) * 1e-6
    resize_history(HISTORY_SECONDS)

def on_window_change(sender, app_data):
    resize_history(dpg.get_value("sl_window"))


def _fit_y_padded(axis_tag: str, *series, margin: float = 0.1):
    """Set y-axis limits with margin so data doesn't touch the edges."""
    import math
    lo = float('inf')
    hi = float('-inf')
    for s in series:
        for v in s:
            if math.isfinite(v):
                lo = min(lo, v)
                hi = max(hi, v)
    if lo > hi:
        # No finite data yet
        dpg.set_axis_limits_auto(axis_tag)
        return
    span = hi - lo
    if span < 1e-6:
        # Flat signal — show ±5% of the value, or ±1 if near zero
        half = max(abs(hi) * 0.05, 1.0)
        lo, hi = lo - half, hi + half
        span = hi - lo
    pad = span * margin
    dpg.set_axis_limits(axis_tag, lo - pad, hi + pad)


def update_frame():
    """Called each GUI frame — advance simulation and update plots."""
    global _fps_last_time, _fps_frame_count, _fps_value
    global _rt_last_wall, _rt_last_sim, _rt_ratio

    # Resize plots to fill available vertical space
    panel_size = dpg.get_item_rect_size("plot_panel")
    if panel_size and panel_size[1] > 0:
        plot_h = max(50, int(panel_size[1] // 5 - 10))
        for tag in ("plot_position", "plot_speed", "plot_current", "plot_torque", "plot_voltage"):
            dpg.configure_item(tag, height=plot_h)

    # FPS counter (update once per second)
    now = time.perf_counter()
    _fps_frame_count += 1
    elapsed = now - _fps_last_time
    if elapsed >= 1.0:
        _fps_value = _fps_frame_count / elapsed
        _fps_frame_count = 0
        _fps_last_time = now
        dpg.set_value("txt_fps", f"FPS: {_fps_value:3.0f}")
        if _rt_ratio >= 1.0:
            rt_color = (100, 230, 100)   # green
        elif _rt_ratio >= 0.5:
            rt_color = (230, 200, 60)    # yellow
        else:
            rt_color = (230, 80, 80)     # red
        rt_label = f"RT: {_rt_ratio:5.1f}x"
        if _rt_ratio < 1.0:
            rt_label += " \u26a0 SLOW"
        dpg.set_value("txt_rt", rt_label)
        dpg.configure_item("txt_rt", color=rt_color)

    if not running:
        return

    # Advance simulation
    global _rt_lock_wall, _rt_lock_residual
    if _realtime_lock:
        wall_now = time.perf_counter()
        wall_elapsed = wall_now - _rt_lock_wall + _rt_lock_residual
        _rt_lock_wall = wall_now
        steps = int(wall_elapsed / DT)
        _rt_lock_residual = wall_elapsed - steps * DT
        steps = max(steps, 0)
        if steps == 0:
            return
    else:
        steps = STEPS_PER_FRAME

    wall_before = time.perf_counter()
    sim.step(DT, steps)
    wall_after = time.perf_counter()
    s = sim.get_state()

    # Real-time ratio: sim time advanced / wall time elapsed
    sim_dt_advanced = DT * steps
    wall_dt = wall_after - wall_before
    if wall_dt > 1e-9:
        _rt_ratio = sim_dt_advanced / wall_dt

    # Record (mechanical quantities converted to rev / rev/s)
    t_hist.append(s.t)
    omega_m_hist.append(s.omega_m / _TWO_PI)
    omega_L_hist.append(s.omega_L / _TWO_PI)
    omega_ref_hist.append(s.omega_ref / _TWO_PI)
    id_hist.append(s.id)
    iq_hist.append(s.iq)
    vd_hist.append(s.vd)
    vq_hist.append(s.vq)
    Te_hist.append(s.Te)
    Tshaft_hist.append(s.Tshaft)
    twist_hist.append((s.theta_m - s.theta_L) / _TWO_PI)
    theta_L_hist.append(s.theta_L / _TWO_PI)
    theta_ref_hist.append(s.theta_ref / _TWO_PI)

    # Convert to lists for plotting
    t = list(t_hist)
    if len(t) < 2:
        return

    # Update speed plot
    dpg.set_value("ser_omega_m", [t, list(omega_m_hist)])
    dpg.set_value("ser_omega_L", [t, list(omega_L_hist)])
    dpg.set_value("ser_omega_ref", [t, list(omega_ref_hist)])
    dpg.fit_axis_data("ax_speed_x")
    _fit_y_padded("ax_speed_y", omega_m_hist, omega_L_hist, omega_ref_hist)

    # Update position plot
    dpg.set_value("ser_theta_L", [t, list(theta_L_hist)])
    dpg.set_value("ser_theta_ref", [t, list(theta_ref_hist)])
    dpg.fit_axis_data("ax_position_x")
    _fit_y_padded("ax_position_y", theta_L_hist, theta_ref_hist)

    # Update current plot
    dpg.set_value("ser_id", [t, list(id_hist)])
    dpg.set_value("ser_iq", [t, list(iq_hist)])
    dpg.fit_axis_data("ax_current_x")
    _fit_y_padded("ax_current_y", id_hist, iq_hist)

    # Update torque plot
    dpg.set_value("ser_Te", [t, list(Te_hist)])
    dpg.set_value("ser_Tshaft", [t, list(Tshaft_hist)])
    dpg.fit_axis_data("ax_torque_x")
    _fit_y_padded("ax_torque_y", Te_hist, Tshaft_hist)

    # Update voltage plot
    dpg.set_value("ser_vd", [t, list(vd_hist)])
    dpg.set_value("ser_vq", [t, list(vq_hist)])
    dpg.fit_axis_data("ax_voltage_x")
    _fit_y_padded("ax_voltage_y", vd_hist, vq_hist)

    # Status text
    revs_m = s.omega_m / _TWO_PI
    revs_L = s.omega_L / _TWO_PI
    dpg.set_value("txt_status",
        f"t = {s.t:7.3f} s  |  Motor: {revs_m:+8.2f} rev/s  |  "
        f"Load: {revs_L:+8.2f} rev/s  |  "
        f"i_q = {s.iq:+7.2f} A  |  Te = {s.Te:+7.3f} Nm")

    # Solver stats
    dpg.set_value("txt_dt", f"dt: {s.dt_used*1e6:.0f} µs")
    dpg.set_value("txt_steps", f"Steps: {s.step_count}  (×{s.plant_substeps} substeps)")
    margin_str = f"RK4 margin: {s.rk4_margin:.2f}x"
    if s.rk4_margin < 1.0:
        margin_str += "  ⚠ UNSTABLE"
    elif s.rk4_margin < 2.0:
        margin_str += "  ⚠ marginal"
    dpg.set_value("txt_rk4_margin", margin_str)
    dpg.set_value("txt_courant_e", f"Courant (elec): {s.courant_elec:.4f} / 2.78")
    dpg.set_value("txt_courant_m", f"Courant (mech): {s.courant_mech:.4f} / 2.78")
    dpg.set_value("txt_tau_elec", f"τ_elec (L/R): {s.tau_elec*1e3:.2f} ms  ({1.0/s.tau_elec:.0f} Hz)" if s.tau_elec < 1e5 else "τ_elec: ∞")
    dpg.set_value("txt_tau_mech", f"τ_mech (J/B): {s.tau_mech:.2f} s" if s.tau_mech < 1e5 else "τ_mech: ∞ (no friction)")
    dpg.set_value("txt_omega_shaft", f"ω_shaft: {s.omega_shaft/_TWO_PI:.1f} rev/s ({s.omega_shaft/_TWO_PI:.0f} Hz)")
    dpg.set_value("txt_lambda_max", f"λ_max: {s.lambda_max:.0f} /s")
    dpg.set_value("txt_max_dxdt", f"max|dx/dt|: {s.max_dxdt:.1f}")
    dpg.set_value("txt_omega_e", f"ω_e: {s.omega_e/_TWO_PI:.1f} rev/s ({abs(s.omega_e)/_TWO_PI:.0f} Hz)")


def build_gui():
    dpg.create_context()
    dpg.create_viewport(title="PMSM Servo Drive Simulator", width=1600, height=950)

    # Bump default font size (default ~13px → 18px)
    dpg.set_global_font_scale(1.0)

    # Load a system font with Greek/math symbol support
    import os
    import platform

    def _find_system_font():
        """Find a suitable system font with Greek/math glyph coverage."""
        system = platform.system()
        if system == "Windows":
            windir = os.environ.get("WINDIR", r"C:\Windows")
            candidates = [
                os.path.join(windir, "Fonts", "segoeui.ttf"),
                os.path.join(windir, "Fonts", "arial.ttf"),
            ]
        elif system == "Darwin":
            candidates = [
                "/System/Library/Fonts/SFPro.ttf",
                "/System/Library/Fonts/SFNS.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
                "/Library/Fonts/Arial.ttf",
                "/System/Library/Fonts/Supplemental/Arial.ttf",
            ]
        else:  # Linux / other
            candidates = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/TTF/DejaVuSans.ttf",
                "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
                "/usr/share/fonts/noto/NotoSans-Regular.ttf",
            ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    font_path = _find_system_font()
    if font_path:
        with dpg.font_registry():
            with dpg.font(font_path, 16) as default_font:
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)
                dpg.add_font_range(0x0370, 0x03FF)  # Greek and Coptic
                dpg.add_font_range(0x00B0, 0x00FF)  # Latin-1 Supplement (µ, ·, etc.)
                dpg.add_font_range(0x2200, 0x22FF)  # Mathematical Operators (∞, √, etc.)
        dpg.bind_font(default_font)

    mp = default_motor_params()
    cp = default_control_params()

    with dpg.window(label="PMSM Servo Drive Simulator", tag="primary", no_close=True):

        # ===== Status bar =====
        with dpg.group(horizontal=True):
            dpg.add_text("Stopped", tag="txt_status")
            dpg.add_spacer(width=20)
            dpg.add_text("FPS: --", tag="txt_fps")
            dpg.add_spacer(width=10)
            dpg.add_text("RT: --", tag="txt_rt", color=(180, 180, 180))
        dpg.add_separator()

        with dpg.group(horizontal=True):

            # ===== Left panel: controls =====
            with dpg.child_window(width=380, height=-1):

                # ===== Simulation Params =====
                dpg.add_text("Simulation")
                dpg.add_separator()

                with dpg.group(horizontal=True):
                    dpg.add_button(label="Start", tag="btn_toggle", callback=on_toggle, width=80)
                    dpg.add_button(label="Reset", callback=on_reset, width=80)

                dpg.add_slider_int(label="Steps/frame", tag="sl_sim_speed",
                    default_value=STEPS_PER_FRAME, min_value=1, max_value=200,
                    callback=on_sim_speed_change, width=200, enabled=False)
                dpg.add_checkbox(label="Real-time lock", tag="cb_rt_lock",
                    default_value=True, callback=on_realtime_lock_change)
                dpg.add_slider_int(label="dt (µs)", tag="sl_dt",
                    default_value=int(DT * 1e6), min_value=10, max_value=500,
                    callback=on_dt_change, width=200)
                dpg.add_slider_float(label="Window (s)", tag="sl_window",
                    default_value=HISTORY_SECONDS, min_value=0.1, max_value=10.0,
                    callback=on_window_change, width=200)

                # ---- Solver Stats ----
                with dpg.collapsing_header(label="Solver Stats", default_open=False):
                    dpg.add_text("dt: --", tag="txt_dt")
                    dpg.add_text("Steps: --", tag="txt_steps")
                    dpg.add_text("", tag="txt_sep1")
                    dpg.add_text("RK4 margin: --", tag="txt_rk4_margin")
                    dpg.add_text("Courant (elec): --", tag="txt_courant_e")
                    dpg.add_text("Courant (mech): --", tag="txt_courant_m")
                    dpg.add_text("", tag="txt_sep2")
                    dpg.add_text("τ_elec (L/R): --", tag="txt_tau_elec")
                    dpg.add_text("τ_mech (J/B): --", tag="txt_tau_mech")
                    dpg.add_text("ω_shaft: --", tag="txt_omega_shaft")
                    dpg.add_text("λ_max: --", tag="txt_lambda_max")
                    dpg.add_text("max|dx/dt|: --", tag="txt_max_dxdt")
                    dpg.add_text("ω_e: --", tag="txt_omega_e")

                dpg.add_spacer(height=10)

                # ===== System Parameters =====
                dpg.add_text("System")
                dpg.add_separator()

                # ---- Setpoints ----
                with dpg.collapsing_header(label="Setpoints", default_open=True):
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="x", callback=on_reset_pos_ref, width=20)
                        dpg.add_slider_float(label="Position ref (rev)", tag="sl_theta_ref",
                            default_value=2.0, min_value=-10.0, max_value=10.0,
                            callback=on_position_ref_change, width=180)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="x", callback=on_reset_speed_ref, width=20)
                        dpg.add_slider_float(label="Speed ref (rev/s)", tag="sl_omega_ref",
                            default_value=0.0, min_value=-80.0, max_value=80.0,
                            callback=on_speed_ref_change, width=180)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="x", callback=on_reset_load_torque, width=20)
                        dpg.add_slider_float(label="Load torque (Nm)", tag="sl_T_load",
                            default_value=0.0, min_value=-2.0, max_value=2.0,
                            callback=on_load_torque_change, width=180)

                # ---- Motor Electrical ----
                with dpg.collapsing_header(label="Motor Electrical", default_open=True):
                    dpg.add_slider_float(label="Rs (Ohm)", tag="sl_Rs",
                        default_value=mp.Rs, min_value=0.1, max_value=10.0,
                        callback=on_motor_change, width=200)
                    dpg.add_slider_float(label="Ls (mH)", tag="sl_Ls",
                        default_value=mp.Ls*1e3, min_value=0.1, max_value=50.0,
                        callback=on_motor_change, width=200)
                    dpg.add_slider_float(label="λ_pm (Wb)", tag="sl_lambda",
                        default_value=mp.lambda_pm, min_value=0.01, max_value=1.0,
                        callback=on_motor_change, width=200)
                    dpg.add_slider_int(label="Pole pairs", tag="sl_poles",
                        default_value=mp.pole_pairs, min_value=1, max_value=12,
                        callback=on_motor_change, width=200)
                    dpg.add_slider_float(label="Vdc (V)", tag="sl_Vdc",
                        default_value=mp.Vdc, min_value=12.0, max_value=400.0,
                        callback=on_motor_change, width=200)

                # ---- Motor Mechanical ----
                with dpg.collapsing_header(label="Motor Mechanical", default_open=True):
                    dpg.add_input_float(label="Jm (kg·m²)", tag="sl_Jm",
                        default_value=mp.Jm, step=1e-5, format="%.2e",
                        callback=on_motor_change, width=200)
                    dpg.add_input_float(label="Bm (Nm·s/rad)", tag="sl_Bm",
                        default_value=mp.Bm, step=1e-4, format="%.2e",
                        callback=on_motor_change, width=200)

                # ---- Coupling ----
                with dpg.collapsing_header(label="Flexible Coupling", default_open=True):
                    dpg.add_slider_float(label="Ks (Nm/rad)", tag="sl_Ks",
                        default_value=mp.Ks, min_value=100.0, max_value=50000.0,
                        callback=on_motor_change, width=200)
                    dpg.add_slider_float(label="Bs (Nm·s/rad)", tag="sl_Bs",
                        default_value=mp.Bs, min_value=0.0, max_value=1.0,
                        callback=on_motor_change, width=200)

                # ---- Load ----
                with dpg.collapsing_header(label="Load", default_open=True):
                    dpg.add_input_float(label="JL (kg·m²)", tag="sl_JL",
                        default_value=mp.JL, step=1e-4, format="%.2e",
                        callback=on_motor_change, width=200)
                    dpg.add_input_float(label="BL (Nm·s/rad)", tag="sl_BL",
                        default_value=mp.BL, step=1e-3, format="%.2e",
                        callback=on_motor_change, width=200)
                    dpg.add_slider_float(label="Tc Coulomb (Nm)", tag="sl_Tc",
                        default_value=mp.Tc, min_value=0.0, max_value=1.0,
                        callback=on_motor_change, width=200)

                dpg.add_spacer(height=10)

                # ===== Controller Params =====
                dpg.add_text("Controller")
                dpg.add_separator()

                with dpg.collapsing_header(label="Position Loop", default_open=True):
                    dpg.add_checkbox(label="Position mode", tag="cb_pos_mode",
                        default_value=True, callback=on_control_change)
                    dpg.add_slider_float(label="Kp", tag="sl_Kp_position",
                        default_value=cp.Kp_position, min_value=1.0, max_value=500.0,
                        callback=on_control_change, width=200)
                    dpg.add_slider_float(label="Ref filter BW (Hz)", tag="sl_ref_filter_bw",
                        default_value=cp.ref_filter_bw, min_value=0.0, max_value=100.0,
                        callback=on_control_change, width=200)
                    dpg.add_slider_float(label="Speed limit (rev/s)", tag="sl_speed_max",
                        default_value=cp.speed_max / _TWO_PI, min_value=1.0, max_value=300.0,
                        callback=on_control_change, width=200)
                    dpg.add_slider_int(label="Position decimation", tag="sl_pos_ratio",
                        default_value=cp.position_ratio, min_value=1, max_value=100,
                        callback=on_control_change, width=200)

                with dpg.collapsing_header(label="Speed Loop", default_open=True):
                    dpg.add_slider_float(label="Kp", tag="sl_Kp_speed",
                        default_value=cp.Kp_speed, min_value=0.01, max_value=10.0,
                        callback=on_control_change, width=200)
                    dpg.add_slider_float(label="Ki", tag="sl_Ki_speed",
                        default_value=cp.Ki_speed, min_value=0.1, max_value=500.0,
                        callback=on_control_change, width=200)
                    dpg.add_slider_float(label="I_max (A)", tag="sl_i_max",
                        default_value=cp.i_max, min_value=1.0, max_value=50.0,
                        callback=on_control_change, width=200)
                    dpg.add_slider_int(label="Speed decimation", tag="sl_speed_ratio",
                        default_value=cp.speed_ratio, min_value=1, max_value=100,
                        callback=on_control_change, width=200)

                with dpg.collapsing_header(label="Current Loop", default_open=True):
                    dpg.add_slider_float(label="BW (Hz)", tag="sl_bw_current",
                        default_value=cp.bw_current, min_value=100.0, max_value=5000.0,
                        callback=on_control_change, width=200)

            # ===== Right panel: plots =====
            with dpg.child_window(width=-1, height=-1, tag="plot_panel"):
                # Position plot
                with dpg.plot(label="Position", height=200, width=-1, tag="plot_position"):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag="ax_position_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="rev", tag="ax_position_y"):
                        dpg.add_line_series([], [], label="θ_load", tag="ser_theta_L")
                        dpg.add_line_series([], [], label="θ_ref", tag="ser_theta_ref")

                # Speed plot
                with dpg.plot(label="Speed", height=200, width=-1, tag="plot_speed"):
                    dpg.add_plot_legend()
                    ax_x = dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag="ax_speed_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="rev/s", tag="ax_speed_y"):
                        dpg.add_line_series([], [], label="ω_motor", tag="ser_omega_m")
                        dpg.add_line_series([], [], label="ω_load", tag="ser_omega_L")
                        dpg.add_line_series([], [], label="ω_ref", tag="ser_omega_ref")

                # Torque plot
                with dpg.plot(label="Torques", height=200, width=-1, tag="plot_torque"):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag="ax_torque_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="Nm", tag="ax_torque_y"):
                        dpg.add_line_series([], [], label="Te (electromagnetic)", tag="ser_Te")
                        dpg.add_line_series([], [], label="T_shaft (coupling)", tag="ser_Tshaft")

                # Current plot
                with dpg.plot(label="Currents", height=200, width=-1, tag="plot_current"):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag="ax_current_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="A", tag="ax_current_y"):
                        dpg.add_line_series([], [], label="i_d", tag="ser_id")
                        dpg.add_line_series([], [], label="i_q", tag="ser_iq")

                # Voltage plot
                with dpg.plot(label="Voltages", height=200, width=-1, tag="plot_voltage"):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag="ax_voltage_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="V", tag="ax_voltage_y"):
                        dpg.add_line_series([], [], label="v_d", tag="ser_vd")
                        dpg.add_line_series([], [], label="v_q", tag="ser_vq")

    dpg.set_primary_window("primary", True)


def main():
    global sim

    parser = argparse.ArgumentParser(description="PMSM Servo Drive Simulator GUI")
    parser.add_argument("--dll", default=None, help="Path to servo_sim.dll")
    args = parser.parse_args()

    # Find DLL
    if args.dll:
        dll_path = args.dll
    else:
        # Look next to this script
        script_dir = Path(__file__).parent
        candidates = [
            script_dir / "servo_sim.dll",
            script_dir / "libservo_sim.so",
            script_dir / "servo_sim.so",
        ]
        dll_path = None
        for c in candidates:
            if c.exists():
                dll_path = str(c)
                break
        if dll_path is None:
            print("ERROR: Could not find servo_sim.dll. Build it first or pass --dll path.")
            print("  Build command (Windows/GCC):")
            print("    g++ -std=c++20 -O2 -shared -DBUILD_DLL -I../../inc -I../../inc/matrix servo_sim.cpp -o servo_sim.dll")
            sys.exit(1)

    print(f"Loading DLL: {dll_path}")
    sim = ServoSimDLL(dll_path)

    # Apply defaults
    sim.set_speed_ref(0.0)
    sim.set_position_ref(2.0 * _TWO_PI)
    sim.set_load_torque(0.0)

    # Push default params to DLL before building GUI
    sim.set_motor_params(default_motor_params())
    sim.set_control_params(default_control_params())

    build_gui()
    dpg.setup_dearpygui()
    dpg.show_viewport()

    # Main loop — advance simulation each frame
    while dpg.is_dearpygui_running():
        update_frame()
        dpg.render_dearpygui_frame()

    sim.destroy()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
