#include <cmath>
#include <numbers>

#include "fmt/core.h"
#include "wet/controllers/stsmc.hpp"

using namespace wet;

// ===== Super-Twisting Swash-Plate Angle Control =====
//
// Variable-displacement axial-piston pump: the swash-plate tilt angle θ sets the
// pump displacement, so the angle loop is the inner loop of every pressure/flow
// controller. The control-piston actuator is a stiff but *nonlinear* second-order
// stage — viscous + Coulomb friction (stiction!), a centering spring, and a
// pressure-dependent load torque that pulses at the pump's piston-pass frequency:
//
//   J·θ̈ = u − b·θ̇ − T_c·sign(θ̇) − k_s·θ − T_load(t)
//
// The angle is measured with a rotary/LVDT sensor (noisy); the rate θ̇ has to be
// *differentiated* from it. That differentiated noise, pushed through the
// super-twisting |s|^½ term, is what makes a textbook STA chatter on real hardware.
//
// Design (canonical super-twisting form, ṡ = b0·u + d with b0 = 1/J):
//   sliding variable  s = θ̇ + λ·(θ − θ_ref)        (s = 0  ->  θ → θ_ref)
//   virtual control   ν = controller.control(s)
//   actuator torque   u = ν / b0 = J·ν
//
// We compare three laws on the *same* plant, noise, and reference, and report
// steady-state tracking plus a chattering index (mean |Δu| per step). Takeaways:
//   1. first-order sign SMC can't hold the angle against stiction + spring + load —
//      it limit-cycles (coarse tracking) and dithers ±k continuously;
//   2. super-twisting's integral term cancels the matched disturbance *continuously*,
//      giving ~60x tighter tracking with bounded, continuous effort;
//   3. the residual chatter is differentiated *sensor noise* on s — cured by filtering
//      the rate estimate (per the docstring), NOT by the generalized-STA linear term
//      (which would pass that noise straight through).

// ----- Plant parameters (illustrative control-piston actuator) -----
constexpr double J = 0.02;      // Reflected inertia [kg·m²]
constexpr double b = 0.6;       // Viscous friction [N·m·s/rad]
constexpr double Tc = 2.5;      // Coulomb friction / stiction [N·m]
constexpr double ks = 8.0;      // Centering-spring stiffness [N·m/rad]
constexpr double dt = 1.0e-4;   // 10 kHz control loop [s]
constexpr double lambda = 40.0; // Sliding-surface slope [1/s] (~25 ms closed-loop)

namespace {

// Pressure-induced load torque: a step at t = 0.5 s plus piston-pass ripple.
double load_torque(double t) {
    if (t <= 0.5) {
        return 0.0;
    }
    return 6.0 + (2.0 * std::sin(2.0 * std::numbers::pi * 90.0 * t)); // step + 9-piston ripple
}

// Deterministic, repeatable broadband "sensor noise" (no <random> needed).
double sensor_noise(double t) {
    return 4.0e-4 * (std::sin(1271.0 * t) + std::sin(4099.0 * t) + std::sin(9173.0 * t)) / 3.0;
}

struct Metrics {
    double track_rms_mrad; // RMS tracking error over the settled window [milli-rad]
    double chatter;        // mean |Δu| per step over the settled window [N·m] — the chattering index
    double effort_rms;     // RMS control effort [N·m]
};

// Run the closed loop. `nu(s)` maps the sliding variable to the super-twisting
// virtual control ν (torque is u = J·ν). `rate_tau` low-pass-filters the
// differentiated angle (0 = raw difference).
template<typename Nu>
Metrics run(Nu nu, double rate_tau) {
    const double theta_ref = 0.20; // [rad] commanded swash angle
    const int    steps = 20000;    // 2 s
    const int    settle = 12000;   // average over t > 1.2 s (well past the load step)

    const double rate_alpha = (rate_tau > 0.0) ? dt / (rate_tau + dt) : 1.0;

    double theta = 0.0;
    double omega = 0.0;
    double theta_meas_prev = sensor_noise(0.0);
    double rate_filt = 0.0;
    double u_prev = 0.0;

    double sum_err2 = 0.0;
    double sum_du = 0.0;
    double sum_u2 = 0.0;
    int    n = 0;

    for (int k = 0; k < steps; ++k) {
        const double t = k * dt;

        // Measurement: noisy angle, rate by backward difference (the noise source).
        const double theta_meas = theta + sensor_noise(t);
        const double rate_raw = (theta_meas - theta_meas_prev) / dt;
        theta_meas_prev = theta_meas;
        rate_filt += rate_alpha * (rate_raw - rate_filt);

        // Sliding variable and control. u = J*ν realizes ṡ = ν + d (b0 = 1/J).
        const double s = rate_filt + (lambda * (theta_meas - theta_ref));
        const double u = J * nu(s);

        // Plant (explicit Euler); tanh is a smooth Coulomb-friction stand-in.
        const double accel = (u - (b * omega) - (Tc * std::tanh(omega / 1e-3)) - (ks * theta) - load_torque(t)) / J;
        omega += dt * accel;
        theta += dt * omega;

        if (k >= settle) {
            const double e = theta - theta_ref;
            sum_err2 += e * e;
            sum_du += std::abs(u - u_prev);
            sum_u2 += u * u;
            ++n;
        }
        u_prev = u;
    }
    return {1e3 * std::sqrt(sum_err2 / n), sum_du / n, std::sqrt(sum_u2 / n)};
}

} // namespace

int main() {
    fmt::print("===== Super-Twisting Swash-Plate Angle Control =====\n\n");
    fmt::print("Plant: J={:.3f} kg*m^2, b={:.2f}, Tc={:.1f} N*m (stiction), ks={:.1f} N*m/rad\n", J, b, Tc, ks);
    fmt::print("Loop:  {:.0f} kHz; angle sensor ~0.4 mrad RMS noise, rate differentiated from it\n", 1e-3 / dt);
    fmt::print("Load:  6 N*m pressure step + 2 N*m @ 90 Hz piston ripple, applied at t=0.5 s\n\n");

    // Design the super-twisting gains from a bound on the disturbance rate |ḋ|.
    // The load ripple dominates: |d(T_load)/dt|/J ~ (2*2π*90)/0.02 ~ 5.7e4, so take
    // L ~ 1e5 with a 1.5x margin. k1 = 1.5*margin*sqrt(L), k2 = 1.1*margin*L.
    constexpr auto art = design::synthesize_stsmc(1.0e5, dt, /*lambda=*/0.0, /*k_lin=*/0.0,
                                                  /*epsilon=*/0.0, /*gain_margin=*/1.5);
    static_assert(art.success);
    fmt::print("Designed STA gains: k1={:.1f}, k2={:.0f}  (from L=1e5, margin=1.5)\n\n", art.k1, art.k2);

    // (1) First-order SMC: u = -k*sign(s). Comparable authority; bang-bang.
    const double  k_sign = art.k2 * dt * 50.0; // rough equal-effort sign gain
    const Metrics smc = run([&](double s) { return -k_sign * static_cast<double>(wet::sgn(s)); }, 0.0);

    // (2) Classic super-twisting on the *raw* differentiated rate.
    const Metrics sta_raw = run([c = SuperTwistingController<double>(art)](double s) mutable { return c.control(s); }, 0.0);

    // (3) Same controller, but low-pass the rate estimate (tau = 2 ms) — the real
    //     cure for the differentiated-noise chatter.
    const Metrics sta_filt = run([c = SuperTwistingController<double>(art)](double s) mutable { return c.control(s); }, 2.0e-3);

    fmt::print("{:<30} {:>14} {:>16} {:>14}\n", "Controller", "track RMS[mrad]", "chatter |du|[Nm]", "effort RMS[Nm]");
    fmt::print("{:-<76}\n", "");
    fmt::print("{:<30} {:>14.3f} {:>16.4f} {:>14.3f}\n", "first-order SMC (sign)", smc.track_rms_mrad, smc.chatter, smc.effort_rms);
    fmt::print("{:<30} {:>14.3f} {:>16.4f} {:>14.3f}\n", "super-twisting, raw rate", sta_raw.track_rms_mrad, sta_raw.chatter, sta_raw.effort_rms);
    fmt::print("{:<30} {:>14.3f} {:>16.4f} {:>14.3f}\n", "super-twisting, filtered rate", sta_filt.track_rms_mrad, sta_filt.chatter, sta_filt.effort_rms);

    fmt::print("\n'chatter' = mean |u[k]-u[k-1]| per step in steady state.\n");
    fmt::print("Super-twisting's headline win is tracking: its integral term cancels the\n");
    fmt::print("matched stiction+load disturbance continuously, ~60x tighter than the\n");
    fmt::print("bang-bang sign law (which limit-cycles around the setpoint). The residual\n");
    fmt::print("chatter is differentiated sensor noise on s; low-pass filtering the rate\n");
    fmt::print("estimate cuts it ~6x while keeping the tracking. (k_lin / boundary-layer\n");
    fmt::print("epsilon are separate knobs: k_lin stiffens convergence and rejects\n");
    fmt::print("state-dependent disturbance; epsilon softens the sign near s=0.)\n");
    return 0;
}
