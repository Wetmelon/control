#ifndef BUILD_DLL
#define BUILD_DLL
#endif
#include "servo_sim.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numbers>
#include <span>

#include "wet/backend.hpp" // wet::max
#include "wet/matrix/colvec.hpp"
#include "wet/motor/servo.hpp"
#include "wet/transforms.hpp"

using namespace wet;
using namespace wet::motor;

namespace {
constexpr double two_pi = 2.0 * std::numbers::pi;
constexpr double sim_step_max = 5e-6; // plant integrates at <= 25 us; control runs at the coarser dt
} // namespace

// ===== Internal simulation state =====
struct ServoSim {
    // Motor and mechanical parameters
    ServoMotorParams mp{};

    // Control parameters
    ServoControlParams cp{};

    // Plant state: [i_d, i_q, omega_m, theta_m, omega_L, theta_L]
    double x[6]{};

    // Simulation time
    double t = 0.0;

    // External inputs (SI: rad, rad/s)
    double omega_ref = 0.0;
    double theta_ref = 0.0;
    double T_load_ext = 0.0;

    // Controller: the library servo object, run as a single unified-rate cascade.
    PmacServo<double> servo{};

    double iq_ref = 0.0;
    double omega_ref_cmd = 0.0; // speed command readback [rad/s] (position-loop output or setpoint)

    // Last computed outputs (for state readback)
    double vd_out = 0.0;
    double vq_out = 0.0;
    double Te_out = 0.0;
    double Tshaft_out = 0.0;
    long   step_count = 0;
    double last_dt = 100e-6;
    double last_max_dxdt = 0.0;
    int    last_substeps = 1;

    // 2nd-order critically damped reference pre-filter on the position command.
    // Plant-independent smoothing kept in the wrapper (not part of PmacServo).
    // H(s) = wn^2 / (s^2 + 2*wn*s + wn^2), zeta=1, Tustin-discretized each step.
    double ref_filt_x1 = 0.0;  // filter state 1 (position)
    double ref_filt_x2 = 0.0;  // filter state 2 (velocity)
    double ref_filt_out = 0.0; // filter output [rad]
    double ref_filt_wn = 0.0;  // natural frequency [rad/s]

    // Compute dx/dt given state and voltage inputs
    void dynamics(const std::span<double> state, double vd, double vq, std::span<double> dxdt) const {
        const double id = state[0];
        const double iq = state[1];
        const double omega_m = state[2];
        const double theta_m = state[3];
        const double omega_L = state[4];
        const double theta_L = state[5];

        const double omega_e = mp.pole_pairs * omega_m;

        // Electrical dynamics (dq frame)
        dxdt[0] = (vd - (mp.Rs * id) + (omega_e * mp.Ls * iq)) / mp.Ls;
        dxdt[1] = (vq - (mp.Rs * iq) - (omega_e * mp.Ls * id) - (omega_e * mp.lambda_pm)) / mp.Ls;

        // Electromagnetic torque
        const double Te = 1.5 * mp.pole_pairs * mp.lambda_pm * iq;

        // Shaft coupling
        const double twist = theta_m - theta_L;
        const double Tshaft = (mp.Ks * twist) + (mp.Bs * (omega_m - omega_L));

        // Motor mechanical
        dxdt[2] = (Te - (mp.Bm * omega_m) - Tshaft) / mp.Jm;
        dxdt[3] = omega_m;

        // Load mechanical (smooth Coulomb friction)
        const double T_coulomb = mp.Tc * std::tanh(omega_L * 100.0);
        dxdt[4] = (Tshaft - (mp.BL * omega_L) - T_coulomb - T_load_ext) / mp.JL;
        dxdt[5] = omega_L;
    }

    // Single RK4 step
    void rk4_step(double dt) {
        double k1[6]{};
        double k2[6]{};
        double k3[6]{};
        double k4[6]{};
        double tmp[6]{};

        dynamics(x, vd_out, vq_out, k1);

        for (int i = 0; i < 6; ++i) {
            tmp[i] = x[i] + (0.5 * dt * k1[i]);
        }
        dynamics(tmp, vd_out, vq_out, k2);

        for (int i = 0; i < 6; ++i) {
            tmp[i] = x[i] + (0.5 * dt * k2[i]);
        }
        dynamics(tmp, vd_out, vq_out, k3);

        for (int i = 0; i < 6; ++i) {
            tmp[i] = x[i] + (dt * k3[i]);
        }
        dynamics(tmp, vd_out, vq_out, k4);

        for (int i = 0; i < 6; ++i) {
            x[i] += dt / 6.0 * (k1[i] + (2.0 * k2[i]) + (2.0 * k3[i]) + k4[i]);
        }

        t += dt;
    }

    // 2nd-order critically damped position reference pre-filter (Tustin, runs once per step).
    void update_ref_filter(double dt) {
        if (ref_filt_wn <= 0.0) {
            // Filter off: pass the command straight through, no velocity feedforward.
            ref_filt_x1 = theta_ref; // track input so re-enabling is bumpless
            ref_filt_x2 = 0.0;
            ref_filt_out = theta_ref;
            return;
        }
        const double wn = ref_filt_wn;
        const double wn2 = wn * wn;
        const double a = dt * 0.5;
        const double d = 1.0 + (2.0 * a * wn) + (a * a * wn2); // determinant of (I - a*A)
        const double inv_d = 1.0 / d;
        const double x1 = ref_filt_x1;
        const double x2 = ref_filt_x2;
        const double u = theta_ref;
        // rhs = (I + a*A)*x + dt*B*u
        const double r1 = x1 + (a * x2);
        const double r2 = (-a * wn2 * x1) + ((1.0 - (2.0 * a * wn)) * x2) + (dt * wn2 * u);
        // x_new = (I - a*A)^{-1} * rhs
        ref_filt_x1 = inv_d * (((1.0 + (2.0 * a * wn)) * r1) + (a * r2));
        ref_filt_x2 = inv_d * ((-a * wn2 * r1) + r2);
        ref_filt_out = ref_filt_x1;
    }

    // Run the servo cascade once and integrate the plant one dt (unified rate).
    void step_once(double dt) {
        update_ref_filter(dt);

        const double theta_m = x[3]; // motor angle [rad]
        const double theta_e = mp.pole_pairs * theta_m;
        const double enc_turns = theta_m / two_pi; // motor encoder [turns]

        // dq plant currents -> abc feedback for the servo
        const auto Iabc = inverse_park_clarke_transform(DirectQuadrature<double>{x[0], x[1]}, theta_e);

        // Commands in SI -> servo units (turns, turns/s). In position mode the reference
        // filter supplies a matched (position, velocity) pair: feed ref_filt_x2 as the
        // smooth velocity feedforward so the position-loop P term only trims error. In
        // velocity mode omega_ref is the command itself.
        // Position mode: feedforward is the reference-filter velocity only. The external
        // speed-ref is a velocity *command* (velocity mode only); applying it here as a
        // constant feedforward would fight the position hold -> surge + steady offset.
        const bool   position_mode = (cp.position_mode >= 2);
        const double vel_ref = position_mode ? ref_filt_x2 : omega_ref; // [rad/s]
        const double pos_target = ref_filt_out / two_pi;
        const double vel_ff = vel_ref / two_pi;

        const auto res = servo.update(pos_target, vel_ff, 0.0, Iabc, mp.Vdc, enc_turns, dt);

        // SVPWM duties -> phase voltages -> dq voltage held across the substeps (ZOH)
        const ColVec<3, double> Vabc{
            (res.duties[0] - 0.5) * mp.Vdc,
            (res.duties[1] - 0.5) * mp.Vdc,
            (res.duties[2] - 0.5) * mp.Vdc,
        };
        const auto Vdq = clarke_park_transform(Vabc, theta_e);
        vd_out = Vdq.d;
        vq_out = Vdq.q;

        // Readbacks: report the smooth velocity reference (filter feedforward), not the
        // position-loop P output, so the omega_ref trace mirrors the planned trajectory.
        iq_ref = servo.Idq_ref_.q;
        omega_ref_cmd = vel_ref;

        // Observable torques (from the plant) before integrating
        Te_out = 1.5 * mp.pole_pairs * mp.lambda_pm * x[1];
        Tshaft_out = (mp.Ks * (x[3] - x[5])) + (mp.Bs * (x[2] - x[4]));

        // Auto-compute plant sub-steps to keep |λ|*dt_sub ≤ target_courant
        const double     sigma_e = mp.Rs / mp.Ls;
        const double     omega_e_abs = std::abs(mp.pole_pairs * x[2]);
        const double     lambda_elec = std::sqrt((sigma_e * sigma_e) + (omega_e_abs * omega_e_abs));
        const double     J_red = (mp.Jm * mp.JL) / wet::max(mp.Jm + mp.JL, 1e-12);
        const double     omega_shaft = std::sqrt(mp.Ks / J_red);
        const double     lambda_max_now = wet::max(lambda_elec, omega_shaft);
        constexpr double target_courant = 1.0;
        int              substeps = static_cast<int>(std::ceil(lambda_max_now * dt / target_courant));
        // Floor the plant step at sim_step_max so the control period (dt) is decoupled
        // from the simulation step; finer auto-substeps still win if stability needs them.
        const int fixed_substeps = static_cast<int>(std::ceil(dt / sim_step_max));
        substeps = std::max({substeps, fixed_substeps, 1});
        last_substeps = substeps;

        // Integrate plant with sub-stepping (ZOH on voltages)
        const double dt_sub = dt / substeps;
        for (int s = 0; s < substeps; ++s) {
            rk4_step(dt_sub);
        }
        ++step_count;
        last_dt = dt;

        // Track max derivative for diagnostics
        double dxdt[6]{};
        dynamics(x, vd_out, vq_out, dxdt);
        double max_d = 0.0;
        for (int i = 0; i < 6; ++i) {
            max_d = std::max(max_d, std::abs(dxdt[i]));
        }
        last_max_dxdt = max_d;
    }

    // Build a servo config from the current parameter structs.
    PmacServoConfig<double> make_config() const {
        PmacServoConfig<double> cfg{};
        cfg.Ldq = {mp.Ls, mp.Ls};
        cfg.R = mp.Rs;
        cfg.lambda = mp.lambda_pm;
        cfg.pole_pairs = mp.pole_pairs;
        cfg.J = mp.Jm + mp.JL; // lumped reflected inertia (stiff-shaft model)
        cfg.b = mp.Bm + mp.BL; // lumped viscous friction
        cfg.iq_max = cp.i_max;
        cfg.vel_max = cp.speed_max / two_pi; // rad/s -> turns/s
        cfg.zeta = 1.0;
        cfg.bandwidths = CascadeBandwidths<double>{
            .omega_position = two_pi * cp.bw_position,
            .omega_velocity = two_pi * cp.bw_velocity,
            .omega_current = two_pi * cp.bw_current,
        };
        cfg.observer.bandwidth = two_pi * cp.bw_current; // commutation tracker ~ current-loop rate
        return cfg;
    }

    // Apply parameter changes mid-run, bumpless: reconfigure in place so the rotor
    // estimator (multi-turn position, speed) and loop integrators are preserved.
    void apply_params() {
        servo.reconfigure(make_config());
        servo.set_mode(static_cast<ControlMode>(cp.position_mode));
        ref_filt_wn = (cp.ref_filter_bw > 0.0) ? two_pi * cp.ref_filter_bw : 0.0;
    }

    // Build a fresh servo (create / reset): zeroes estimator and integrators.
    void build_servo() {
        servo = PmacServo<double>{make_config()};
        servo.set_mode(static_cast<ControlMode>(cp.position_mode));
        ref_filt_wn = (cp.ref_filter_bw > 0.0) ? two_pi * cp.ref_filter_bw : 0.0;
    }
};

// ===== Default parameters: ODrive D5065 270KV outrunner =====
// Electrical values are from the motor datasheet; the mechanical inertia, flexible
// coupling, and load are a representative test rig (not part of the datasheet).
static ServoMotorParams default_motor_params() {
    ServoMotorParams p{};
    p.Rs = 0.039;                    // 39 mOhm phase-to-neutral
    p.Ls = 16e-6;                    // 16 uH phase-to-neutral (Ld = Lq)
    p.lambda_pm = 0.031 / (1.5 * 7); // lambda = Kt/(1.5*pp), Kt = 0.031 Nm/A -> 2.95 mWb (KV 270)
    p.pole_pairs = 7;
    p.Jm = 5e-5;
    p.Bm = 1e-4;
    p.Ks = 5000.0;
    p.Bs = 0.05;
    p.JL = 5e-4;
    p.BL = 1e-3;
    p.Tc = 0.02;
    p.Vdc = 48.0;
    return p;
}

static ServoControlParams default_control_params() {
    ServoControlParams c{};
    c.bw_position = 5.0;   // 5 Hz outer position loop
    c.bw_velocity = 50.0;  // 50 Hz velocity loop
    c.bw_current = 1000.0; // 1 kHz current loop
    c.i_max = 65.0;        // [A] continuous (peak 85 A); no thermal model wired
    c.speed_max = 1300.0;  // [rad/s] velocity-command ceiling (~207 rev/s, near base speed)
    c.position_mode = 2;   // ControlMode::Position
    c.ref_filter_bw = 5.0; // 5 Hz ref filter
    return c;
}

// ===== C API implementation =====

extern "C" {

SERVO_API void* servo_create(void) {
    auto* sim = new ServoSim();
    sim->mp = default_motor_params();
    sim->cp = default_control_params();
    sim->build_servo();
    return sim;
}

SERVO_API void servo_destroy(void* handle) {
    delete static_cast<ServoSim*>(handle);
}

SERVO_API void servo_set_motor_params(void* handle, const ServoMotorParams* params) {
    auto* sim = static_cast<ServoSim*>(handle);
    sim->mp = *params;
    sim->apply_params();
}

SERVO_API void servo_set_control_params(void* handle, const ServoControlParams* params) {
    auto* sim = static_cast<ServoSim*>(handle);
    sim->cp = *params;
    sim->apply_params();
}

SERVO_API void servo_set_speed_ref(void* handle, double omega_ref) {
    static_cast<ServoSim*>(handle)->omega_ref = omega_ref;
}

SERVO_API void servo_set_position_ref(void* handle, double theta_ref) {
    static_cast<ServoSim*>(handle)->theta_ref = theta_ref;
}

SERVO_API void servo_set_load_torque(void* handle, double T_load) {
    static_cast<ServoSim*>(handle)->T_load_ext = T_load;
}

SERVO_API void servo_step(void* handle, double dt, int n_steps) {
    auto* sim = static_cast<ServoSim*>(handle);
    for (int i = 0; i < n_steps; ++i) {
        sim->step_once(dt);
    }
}

SERVO_API ServoState servo_get_state(void* handle) {
    auto*      sim = static_cast<ServoSim*>(handle);
    ServoState s{};

    // Plant states
    s.id = sim->x[0];
    s.iq = sim->x[1];
    s.omega_m = sim->x[2];
    s.theta_m = sim->x[3];
    s.omega_L = sim->x[4];
    s.theta_L = sim->x[5];

    // Controller outputs
    s.vd = sim->vd_out;
    s.vq = sim->vq_out;
    s.iq_ref = sim->iq_ref;
    s.omega_ref = sim->omega_ref_cmd;
    s.theta_ref = sim->ref_filt_out;

    // Torques
    s.Te = sim->Te_out;
    s.Tshaft = sim->Tshaft_out;

    // Derived quantities
    s.twist = sim->x[3] - sim->x[5];
    s.P_elec = 1.5 * ((sim->vd_out * sim->x[0]) + (sim->vq_out * sim->x[1]));
    s.P_mech = sim->Te_out * sim->x[2];
    s.speed_err = sim->omega_ref_cmd - sim->x[2];
    s.pos_err = sim->ref_filt_out - sim->x[5];
    s.omega_e = sim->mp.pole_pairs * sim->x[2];

    // Controller integrator states (internal to PmacServo — not exposed)
    s.int_id = 0.0;
    s.int_iq = 0.0;
    s.int_spd = 0.0;

    // Timing
    s.t = sim->t;
    s.step_count = sim->step_count;

    // Solver / stiffness diagnostics
    s.max_dxdt = sim->last_max_dxdt;
    s.dt_used = sim->last_dt;

    // Electrical time constant
    s.tau_elec = (sim->mp.Rs > 1e-12) ? sim->mp.Ls / sim->mp.Rs : 1e6;

    // Mechanical time constant (motor side)
    s.tau_mech = (sim->mp.Bm > 1e-12) ? sim->mp.Jm / sim->mp.Bm : 1e6;

    // Shaft natural frequency: sqrt(Ks * (1/Jm + 1/JL))
    double J_red = (sim->mp.Jm * sim->mp.JL) / wet::max(sim->mp.Jm + sim->mp.JL, 1e-12);
    s.omega_shaft = std::sqrt(sim->mp.Ks / J_red);

    // Largest eigenvalue magnitude of the electrical subsystem
    // dq dynamics: eigenvalues are -R/L ± j*omega_e
    const double sigma_e = sim->mp.Rs / sim->mp.Ls;
    const double omega_e_abs = std::abs(s.omega_e);
    const double lambda_elec = std::sqrt((sigma_e * sigma_e) + (omega_e_abs * omega_e_abs));

    // Overall max eigenvalue (electrical vs mechanical shaft)
    s.lambda_max = wet::max(lambda_elec, s.omega_shaft);

    // RK4 stability diagnostics use the actual sub-step dt
    const double dt = sim->last_dt;
    const double dt_sub = (sim->last_substeps > 0) ? dt / sim->last_substeps : dt;
    s.courant_elec = lambda_elec * dt_sub;
    s.courant_mech = s.omega_shaft * dt_sub;
    s.rk4_margin = (s.lambda_max > 1e-12) ? 2.78 / (s.lambda_max * dt_sub) : 999.0;
    s.plant_substeps = sim->last_substeps;

    return s;
}

SERVO_API void servo_reset(void* handle) {
    auto* sim = static_cast<ServoSim*>(handle);
    std::memset(sim->x, 0, sizeof(sim->x));
    sim->t = 0.0;
    sim->vd_out = 0.0;
    sim->vq_out = 0.0;
    sim->Te_out = 0.0;
    sim->Tshaft_out = 0.0;
    sim->iq_ref = 0.0;
    sim->omega_ref_cmd = 0.0;
    sim->ref_filt_x1 = 0.0;
    sim->ref_filt_x2 = 0.0;
    sim->ref_filt_out = 0.0;
    sim->step_count = 0;
    sim->last_max_dxdt = 0.0;
    sim->build_servo();
}

} // extern "C"
