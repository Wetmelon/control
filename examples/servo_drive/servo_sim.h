#ifndef SERVO_SIM_H
#define SERVO_SIM_H

/**
 * @file servo_sim.h
 * @brief C API for the PMSM servo drive simulation DLL
 *
 * Build (Windows, GCC):
 *   g++ -std=c++20 -O2 -shared -DBUILD_DLL -I../../inc -I../../inc/matrix servo_sim.cpp -o servo_sim.dll
 *
 * Build (Linux, GCC):
 *   g++ -std=c++20 -O2 -shared -fPIC -DBUILD_DLL -I../../inc -I../../inc/matrix servo_sim.cpp -o libservo_sim.so
 *
 * Build (macOS, clang):
 *   clang++ -std=c++20 -O2 -shared -fPIC -DBUILD_DLL -I../../inc -I../../inc/matrix servo_sim.cpp -o libservo_sim.dylib
 */

#ifdef _WIN32
    #ifdef BUILD_DLL
        #define SERVO_API __declspec(dllexport)
    #else
        #define SERVO_API __declspec(dllimport)
    #endif
#else
    #ifdef BUILD_DLL
        #define SERVO_API __attribute__((visibility("default")))
    #else
        #define SERVO_API
    #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Motor electrical and mechanical parameters.
 * Coupling and load mechanical parameters.
 */
typedef struct {
    /* Electrical */
    double Rs;         /* Stator resistance [Ohm]          */
    double Ls;         /* Stator inductance [H] (Ld = Lq)  */
    double lambda_pm;  /* PM flux linkage [Wb]             */
    int    pole_pairs; /* Number of pole pairs             */

    /* Motor mechanical */
    double Jm;         /* Motor rotor inertia [kg*m^2]     */
    double Bm;         /* Motor viscous friction [Nm*s/rad]*/

    /* Flexible coupling */
    double Ks;         /* Shaft torsional stiffness [Nm/rad]*/
    double Bs;         /* Shaft torsional damping [Nm*s/rad]*/

    /* Load mechanical */
    double JL;         /* Load inertia [kg*m^2]            */
    double BL;         /* Load viscous friction [Nm*s/rad] */
    double Tc;         /* Coulomb friction torque [Nm]     */

    /* System */
    double Vdc;        /* DC bus voltage [V]               */
} ServoMotorParams;

/**
 * Controller gains and limits.
 */
typedef struct {
    /* Current loop bandwidth [Hz] — PI gains computed as Kp = wc*Ls, Ki = wc*Rs */
    double bw_current;

    /* Speed loop PI gains */
    double Kp_speed;
    double Ki_speed;

    /* Position loop P gain */
    double Kp_position;

    /* Current limit [A]              */
    double i_max;

    /* Speed limit [rad/s] for position mode */
    double speed_max;

    /* Speed loop decimation ratio    */
    int speed_ratio;

    /* Position loop decimation ratio (relative to speed loop) */
    int position_ratio;

    /* 1 = position mode (P-PI-PI), 0 = speed mode (PI-PI) */
    int position_mode;

    /* Position reference filter bandwidth [Hz] (2nd order, critically damped) */
    double ref_filter_bw;
} ServoControlParams;

/**
 * Observable simulation state (returned to caller each step).
 */
typedef struct {
    /* Plant states */
    double id;       /* d-axis current [A]         */
    double iq;       /* q-axis current [A]         */
    double omega_m;  /* Motor speed [rad/s]        */
    double theta_m;  /* Motor angle [rad]          */
    double omega_L;  /* Load speed [rad/s]         */
    double theta_L;  /* Load angle [rad]           */

    /* Controller outputs */
    double vd;       /* Applied d-axis voltage [V] */
    double vq;       /* Applied q-axis voltage [V] */
    double iq_ref;   /* q-axis current reference [A] */
    double omega_ref;/* Speed reference [rad/s] (from position loop or setpoint) */
    double theta_ref;/* Position reference [rad] (in position mode) */

    /* Torques */
    double Te;       /* Electromagnetic torque [Nm]*/
    double Tshaft;   /* Shaft coupling torque [Nm] */

    /* Derived quantities */
    double twist;    /* Shaft twist angle [rad]    */
    double P_elec;   /* Electrical power [W]       */
    double P_mech;   /* Mechanical power [W]       */
    double speed_err;/* Speed error [rad/s]        */
    double pos_err;  /* Position error [rad]        */
    double omega_e;  /* Electrical frequency [rad/s]*/

    /* Controller integrator states */
    double int_id;   /* d-axis PI integrator       */
    double int_iq;   /* q-axis PI integrator       */
    double int_spd;  /* Speed PI integrator        */

    /* Timing */
    double t;        /* Current simulation time [s]*/
    long   step_count; /* Total RK4 steps taken    */

    /* Solver / stiffness diagnostics */
    double max_dxdt;     /* Max |dx/dt| across all states    */
    double tau_elec;     /* Electrical time constant L/R [s] */
    double tau_mech;     /* Mechanical time constant J/B [s] */
    double omega_shaft;  /* Shaft natural freq sqrt(Ks/Jred) [rad/s] */
    double lambda_max;   /* Largest eigenvalue magnitude [1/s] */
    double rk4_margin;   /* RK4 stability margin: 2.78/(|λ|*dt), >1 = stable */
    double courant_elec; /* |λ_elec|*dt — should be < 2.78 for RK4 */
    double courant_mech; /* ω_shaft*dt — should be < 2.78 for RK4 */
    double dt_used;      /* Timestep used [s]                */
    int    plant_substeps; /* Auto-computed plant integration sub-steps */
} ServoState;

/**
 * Create a new simulation instance with default parameters.
 * @return Opaque handle (never NULL).
 */
SERVO_API void* servo_create(void);

/**
 * Destroy a simulation instance and free memory.
 */
SERVO_API void servo_destroy(void* handle);

/**
 * Update motor and mechanical parameters (takes effect immediately).
 */
SERVO_API void servo_set_motor_params(void* handle, const ServoMotorParams* params);

/**
 * Update controller gains (resets integrators).
 */
SERVO_API void servo_set_control_params(void* handle, const ServoControlParams* params);

/**
 * Set speed reference [rad/s].
 */
SERVO_API void servo_set_speed_ref(void* handle, double omega_ref);

/**
 * Set position reference [rad] (used when position mode is enabled).
 */
SERVO_API void servo_set_position_ref(void* handle, double theta_ref);

/**
 * Set external load torque [Nm] applied to the load mass.
 */
SERVO_API void servo_set_load_torque(void* handle, double T_load);

/**
 * Advance the simulation by n_steps of dt seconds each.
 * Uses RK4 integration internally.
 */
SERVO_API void servo_step(void* handle, double dt, int n_steps);

/**
 * Get the current simulation state.
 */
SERVO_API ServoState servo_get_state(void* handle);

/**
 * Reset all states and integrators to zero.
 */
SERVO_API void servo_reset(void* handle);

#ifdef __cplusplus
}
#endif

#endif /* SERVO_SIM_H */
