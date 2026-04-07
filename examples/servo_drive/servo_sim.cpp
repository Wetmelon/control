#ifndef BUILD_DLL
#define BUILD_DLL
#endif
#include "servo_sim.h"

#include <cmath>
#include <cstring>

#include "pid.hpp"

using namespace wetmelon::control;

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

    // External inputs
    double omega_ref = 0.0;
    double theta_ref = 0.0;
    double T_load_ext = 0.0;

    // Controller internals
    PIDController<double> pi_d{};
    PIDController<double> pi_q{};
    PIDController<double> pi_speed{};
    double Kp_pos = 0.0;         // Position P gain
    int    speed_decim = 0;
    int    pos_decim = 0;
    double iq_ref = 0.0;
    double omega_ref_cmd = 0.0;  // Speed command (from position loop or direct)

    // Last computed outputs (for state readback)
    double vd_out = 0.0;
    double vq_out = 0.0;
    double Te_out = 0.0;
    double Tshaft_out = 0.0;
    long   step_count = 0;
    double last_dt = 100e-6;
    double last_max_dxdt = 0.0;
    int    last_substeps = 1;

    // 2nd-order critically damped reference filter state
    // H(s) = wn^2 / (s^2 + 2*wn*s + wn^2) with zeta=1
    // State-space: x1' = x2, x2' = -wn^2*x1 - 2*wn*x2 + wn^2*u, y = x1 + u_offset
    // Discretized with Tustin (bilinear) each time step.
    double ref_filt_x1 = 0.0;  // filter state 1 (position)
    double ref_filt_x2 = 0.0;  // filter state 2 (velocity)
    double ref_filt_out = 0.0; // filter output
    double ref_filt_wn = 0.0;  // natural frequency [rad/s]

    // Compute dx/dt given state and voltage inputs
    void dynamics(const double* state, double vd, double vq, double* dxdt) const {
        const double id      = state[0];
        const double iq      = state[1];
        const double omega_m = state[2];
        const double theta_m = state[3];
        const double omega_L = state[4];
        const double theta_L = state[5];

        const double omega_e = mp.pole_pairs * omega_m;

        // Electrical dynamics (dq frame)
        dxdt[0] = (vd - mp.Rs * id + omega_e * mp.Ls * iq) / mp.Ls;
        dxdt[1] = (vq - mp.Rs * iq - omega_e * mp.Ls * id - omega_e * mp.lambda_pm) / mp.Ls;

        // Electromagnetic torque
        const double Te = 1.5 * mp.pole_pairs * mp.lambda_pm * iq;

        // Shaft coupling
        const double twist   = theta_m - theta_L;
        const double Tshaft  = mp.Ks * twist + mp.Bs * (omega_m - omega_L);

        // Motor mechanical
        dxdt[2] = (Te - mp.Bm * omega_m - Tshaft) / mp.Jm;
        dxdt[3] = omega_m;

        // Load mechanical (smooth Coulomb friction)
        const double T_coulomb = mp.Tc * std::tanh(omega_L * 100.0);
        dxdt[4] = (Tshaft - mp.BL * omega_L - T_coulomb - T_load_ext) / mp.JL;
        dxdt[5] = omega_L;
    }

    // Single RK4 step
    void rk4_step(double dt) {
        double k1[6], k2[6], k3[6], k4[6], tmp[6];

        dynamics(x, vd_out, vq_out, k1);

        for (int i = 0; i < 6; ++i) { tmp[i] = x[i] + 0.5 * dt * k1[i]; }
        dynamics(tmp, vd_out, vq_out, k2);

        for (int i = 0; i < 6; ++i) { tmp[i] = x[i] + 0.5 * dt * k2[i]; }
        dynamics(tmp, vd_out, vq_out, k3);

        for (int i = 0; i < 6; ++i) { tmp[i] = x[i] + dt * k3[i]; }
        dynamics(tmp, vd_out, vq_out, k4);

        for (int i = 0; i < 6; ++i) {
            x[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }

        t += dt;
    }

    // Run controller and integrate one dt
    void step_once(double dt) {
        // --- 2nd-order critically damped position reference filter ---
        // Continuous: x1' = x2, x2' = wn^2*(u - x1) - 2*wn*x2
        // Tustin discretization with pre-warp
        if (ref_filt_wn > 0.0) {
            const double wn = ref_filt_wn;
            const double wn2 = wn * wn;
            const double h = dt;  // full controller dt (filter runs once per step)
            // Tustin matrix: A_c = [[0, 1],[-wn2, -2*wn]], B_c = [[0],[wn2]]
            // x_{k+1} = (I - h/2*A)^{-1} * ((I + h/2*A)*x_k + h*B*u_k)
            const double a = h * 0.5;
            // (I - a*A) = [[1, -a], [a*wn2, 1+2*a*wn]]
            const double d = 1.0 + 2.0 * a * wn + a * a * wn2;  // determinant
            const double inv_d = 1.0 / d;
            // (I + a*A) = [[1, a], [-a*wn2, 1-2*a*wn]]
            const double x1 = ref_filt_x1;
            const double x2 = ref_filt_x2;
            const double u = theta_ref;
            // rhs = (I + a*A)*x + h*B*u
            const double r1 = x1 + a * x2 + 0.0;
            const double r2 = -a * wn2 * x1 + (1.0 - 2.0 * a * wn) * x2 + h * wn2 * u;
            // x_new = (I - a*A)^{-1} * rhs
            ref_filt_x1 = inv_d * ((1.0 + 2.0 * a * wn) * r1 + a * r2);
            ref_filt_x2 = inv_d * (-a * wn2 * r1 + 1.0 * r2);
            ref_filt_out = ref_filt_x1;
        } else {
            ref_filt_out = theta_ref;
        }

        const double id      = x[0];
        const double iq      = x[1];
        const double omega_m = x[2];
        const double theta_L = x[5];

        // Position loop (decimated relative to speed loop)
        if (cp.position_mode) {
            if (++pos_decim >= cp.position_ratio) {
                pos_decim = 0;
                double pos_err = ref_filt_out - theta_L;
                double cmd = Kp_pos * pos_err + omega_ref;
                // Clamp speed command
                if (cmd > cp.speed_max) { cmd = cp.speed_max; }
                if (cmd < -cp.speed_max) { cmd = -cp.speed_max; }
                omega_ref_cmd = cmd;
            }
        } else {
            omega_ref_cmd = omega_ref;
        }

        // Speed loop (decimated)
        if (++speed_decim >= cp.speed_ratio) {
            speed_decim = 0;
            iq_ref = pi_speed.control(omega_ref_cmd - omega_m);
        }

        // Current loops
        double vd = pi_d.control(0.0 - id);
        double vq = pi_q.control(iq_ref - iq);

        // Decoupling feedforward
        const double omega_e = mp.pole_pairs * omega_m;
        vd -= omega_e * mp.Ls * iq;
        vq += omega_e * mp.Ls * id + omega_e * mp.lambda_pm;

        vd_out = vd;
        vq_out = vq;

        // Compute observable torques before integrating
        Te_out = 1.5 * mp.pole_pairs * mp.lambda_pm * iq;
        Tshaft_out = mp.Ks * (x[3] - x[5]) + mp.Bs * (x[2] - x[4]);

        // Auto-compute plant sub-steps to keep |λ|*dt_sub ≤ target_courant
        // Electrical eigenvalue: |λ_elec| = sqrt((R/L)^2 + ω_e^2)
        const double sigma_e = mp.Rs / mp.Ls;
        const double omega_e_abs = std::abs(mp.pole_pairs * omega_m);
        const double lambda_elec = std::sqrt(sigma_e * sigma_e + omega_e_abs * omega_e_abs);
        // Shaft resonance eigenvalue: ω_shaft = sqrt(Ks / J_reduced)
        const double J_red = (mp.Jm * mp.JL) / std::max(mp.Jm + mp.JL, 1e-12);
        const double omega_shaft = std::sqrt(mp.Ks / J_red);
        const double lambda_max_now = std::max(lambda_elec, omega_shaft);
        // Target: |λ|*dt_sub ≤ 1.0 (well within RK4 stability limit of 2.78)
        constexpr double target_courant = 1.0;
        int substeps = static_cast<int>(std::ceil(lambda_max_now * dt / target_courant));
        if (substeps < 1) { substeps = 1; }
        last_substeps = substeps;

        // Integrate plant with sub-stepping (ZOH on voltages)
        const double dt_sub = dt / substeps;
        for (int s = 0; s < substeps; ++s) {
            rk4_step(dt_sub);
        }
        ++step_count;
        last_dt = dt;

        // Track max derivative for diagnostics
        double dxdt[6];
        dynamics(x, vd_out, vq_out, dxdt);
        double max_d = 0.0;
        for (int i = 0; i < 6; ++i) {
            double a = std::abs(dxdt[i]);
            if (a > max_d) { max_d = a; }
        }
        last_max_dxdt = max_d;
    }

    void rebuild_controllers(double dt) {
        constexpr double pi2 = 2.0 * 3.14159265358979323846;
        double v_lim = mp.Vdc / 2.0;
        double i_lim = cp.i_max;

        // Current loop: PI from bandwidth
        double wc_i  = pi2 * cp.bw_current;
        double Kp_i  = wc_i * mp.Ls;
        double Ki_i  = wc_i * mp.Rs;

        // Speed loop: explicit gains
        double Kp_s = cp.Kp_speed;
        double Ki_s = cp.Ki_speed;

        // Position loop: explicit P gain
        Kp_pos = cp.Kp_position;

        // Reference filter
        ref_filt_wn = (cp.ref_filter_bw > 0.0) ? pi2 * cp.ref_filter_bw : 0.0;

        pi_d = PIDController<double>{
            online::pid(Kp_i, Ki_i, 0.0, dt, -v_lim, v_lim, -v_lim / std::max(Ki_i, 1e-6), v_lim / std::max(Ki_i, 1e-6), Ki_i)};
        pi_q = PIDController<double>{
            online::pid(Kp_i, Ki_i, 0.0, dt, -v_lim, v_lim, -v_lim / std::max(Ki_i, 1e-6), v_lim / std::max(Ki_i, 1e-6), Ki_i)};
        pi_speed = PIDController<double>{
            online::pid(Kp_s, Ki_s, 0.0, dt * cp.speed_ratio, -i_lim, i_lim, -i_lim / std::max(Ki_s, 1e-6), i_lim / std::max(Ki_s, 1e-6), Ki_s)};

        speed_decim = 0;
        pos_decim = 0;
        iq_ref = 0.0;
        omega_ref_cmd = 0.0;
    }
};

// ===== Default parameters (same as the standalone C++ example) =====
static ServoMotorParams default_motor_params() {
    ServoMotorParams p{};
    p.Rs         = 1.2;
    p.Ls         = 4.7e-3;
    p.lambda_pm  = 0.1;
    p.pole_pairs = 4;
    p.Jm         = 5e-5;
    p.Bm         = 1e-4;
    p.Ks         = 5000.0;
    p.Bs         = 0.05;
    p.JL         = 5e-4;
    p.BL         = 1e-3;
    p.Tc         = 0.02;
    p.Vdc        = 48.0;
    return p;
}

static ServoControlParams default_control_params() {
    ServoControlParams c{};
    c.bw_current    = 1000.0;   // 1 kHz current loop
    c.Kp_speed      = 0.3;      // Speed P gain
    c.Ki_speed      = 18.0;     // Speed I gain
    c.Kp_position   = 60.0;     // Position P gain
    c.i_max         = 10.0;
    c.speed_max     = 500.0;
    c.speed_ratio = 1;
    c.position_ratio = 1;
    c.position_mode  = 0;
    c.ref_filter_bw  = 5.0;   // 5 Hz ref filter
    return c;
}

// ===== C API implementation =====

extern "C" {

SERVO_API void* servo_create(void) {
    auto* sim = new ServoSim();
    sim->mp = default_motor_params();
    sim->cp = default_control_params();
    sim->rebuild_controllers(100e-6);
    return sim;
}

SERVO_API void servo_destroy(void* handle) {
    delete static_cast<ServoSim*>(handle);
}

SERVO_API void servo_set_motor_params(void* handle, const ServoMotorParams* params) {
    auto* sim = static_cast<ServoSim*>(handle);
    sim->mp = *params;
}

SERVO_API void servo_set_control_params(void* handle, const ServoControlParams* params) {
    auto* sim = static_cast<ServoSim*>(handle);
    sim->cp = *params;
    sim->rebuild_controllers(100e-6);
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
    auto* sim = static_cast<ServoSim*>(handle);
    ServoState s{};

    // Plant states
    s.id      = sim->x[0];
    s.iq      = sim->x[1];
    s.omega_m = sim->x[2];
    s.theta_m = sim->x[3];
    s.omega_L = sim->x[4];
    s.theta_L = sim->x[5];

    // Controller outputs
    s.vd      = sim->vd_out;
    s.vq      = sim->vq_out;
    s.iq_ref  = sim->iq_ref;
    s.omega_ref = sim->omega_ref_cmd;
    s.theta_ref = sim->ref_filt_out;

    // Torques
    s.Te      = sim->Te_out;
    s.Tshaft  = sim->Tshaft_out;

    // Derived quantities
    s.twist     = sim->x[3] - sim->x[5];
    s.P_elec    = 1.5 * (sim->vd_out * sim->x[0] + sim->vq_out * sim->x[1]);
    s.P_mech    = sim->Te_out * sim->x[2];
    s.speed_err = sim->omega_ref_cmd - sim->x[2];
    s.pos_err   = sim->ref_filt_out - sim->x[5];
    s.omega_e   = sim->mp.pole_pairs * sim->x[2];

    // Controller integrator states
    s.int_id  = sim->pi_d.integral;
    s.int_iq  = sim->pi_q.integral;
    s.int_spd = sim->pi_speed.integral;

    // Timing
    s.t          = sim->t;
    s.step_count = sim->step_count;

    // Solver / stiffness diagnostics
    s.max_dxdt = sim->last_max_dxdt;
    s.dt_used  = sim->last_dt;

    // Electrical time constant
    s.tau_elec = (sim->mp.Rs > 1e-12) ? sim->mp.Ls / sim->mp.Rs : 1e6;

    // Mechanical time constant (motor side)
    s.tau_mech = (sim->mp.Bm > 1e-12) ? sim->mp.Jm / sim->mp.Bm : 1e6;

    // Shaft natural frequency: sqrt(Ks * (1/Jm + 1/JL))
    double J_red = (sim->mp.Jm * sim->mp.JL) / std::max(sim->mp.Jm + sim->mp.JL, 1e-12);
    s.omega_shaft = std::sqrt(sim->mp.Ks / J_red);

    // Largest eigenvalue magnitude of the electrical subsystem
    // dq dynamics: eigenvalues are -R/L ± j*omega_e
    double sigma_e = sim->mp.Rs / sim->mp.Ls;
    double omega_e_abs = std::abs(s.omega_e);
    double lambda_elec = std::sqrt(sigma_e * sigma_e + omega_e_abs * omega_e_abs);

    // Overall max eigenvalue (electrical vs mechanical shaft)
    s.lambda_max = std::max(lambda_elec, s.omega_shaft);

    // RK4 stability diagnostics use the actual sub-step dt
    double dt = sim->last_dt;
    double dt_sub = (sim->last_substeps > 0) ? dt / sim->last_substeps : dt;
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
    sim->speed_decim = 0;
    sim->pos_decim = 0;
    sim->step_count = 0;
    sim->last_max_dxdt = 0.0;
    sim->rebuild_controllers(100e-6);
}

} // extern "C"
