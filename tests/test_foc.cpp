
#include <cmath>
#include <limits>
#include <numbers>

#include "wet/controllers/pid.hpp"
#include "wet/design/pid_design.hpp"
#include "wet/matrix/colvec.hpp"
#include "wet/motor/foc.hpp"
#include "wet/transforms.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

TEST_SUITE("FOC Controller") {
    constexpr float Ts = 1.0f / 8000.0f;

    TEST_CASE("Current Controller") {
        wet::FOController<float> foc;

        wet::DirectQuadrature<float> Idq_ref = {.d = 0.0f, .q = 1.0f};
        wet::DirectQuadrature<float> Idq = {.d = 0.0f, .q = 0.5f};

        (void)foc.current_controller(Idq_ref, Idq, Ts);
    }

    TEST_CASE("Step") {
        wet::FOController<float> foc;
        (void)foc.step(
            {0.0f, 1.0f},
            {0.0f, 0.0f, 0.0f},
            0.0f,
            48.0f,
            Ts
        );
    }

    TEST_CASE("Pole-placement tuning") {
        const float Ld = 200e-6f;
        const float Lq = 350e-6f; // salient: Lq != Ld -> distinct gains
        const float R = 0.5f;
        const float omega_bw = 2.0f * std::numbers::pi_v<float> * 1000.0f; // 1 kHz

        wet::FOController<float> foc({.d = Ld, .q = Lq}, R, 0.0f, 0.0f);
        foc.tune(omega_bw); // default zeta = 1 (critically damped)

        // Kp = 2*zeta*wn*L - R, Ki = L*wn^2, per axis
        CHECK(foc.dctrl.Kp == doctest::Approx((2.0f * omega_bw * Ld) - R));
        CHECK(foc.dctrl.Ki == doctest::Approx(Ld * omega_bw * omega_bw));
        CHECK(foc.dctrl.Kbc == doctest::Approx(foc.dctrl.Kp));

        CHECK(foc.qctrl.Kp == doctest::Approx((2.0f * omega_bw * Lq) - R));
        CHECK(foc.qctrl.Ki == doctest::Approx(Lq * omega_bw * omega_bw));
        CHECK(foc.qctrl.Kbc == doctest::Approx(foc.qctrl.Kp));

        // q-axis has larger inductance -> larger gains
        CHECK(foc.qctrl.Kp > foc.dctrl.Kp);
    }

    TEST_CASE("current_loop_pi delegates to pi_pole_placement_first_order") {
        // current_loop_pi(L,R,...) is exactly the SISO PI pole-placement kernel
        // with (a1,a0)=(L,R) — verify the delegation rather than a re-derivation.
        constexpr float L = 200e-6f, R = 0.5f, omega = 6000.0f, zeta = 1.0f, b = 0.0f;
        constexpr auto  via_current = wet::design::current_loop_pi(L, R, omega, zeta, b);
        constexpr auto  via_kernel = wet::design::pi_pole_placement_first_order(L, R, omega, zeta, b);
        static_assert(via_current.Kp == via_kernel.Kp && via_current.Ki == via_kernel.Ki && via_current.b == via_kernel.b);
        CHECK(via_current.Kp == doctest::Approx(via_kernel.Kp));
    }

    TEST_CASE("Voltage-circle limit and anti-windup") {
        const float              Vmax = 10.0f;
        wet::FOController<float> foc({.d = 200e-6f, .q = 200e-6f}, 0.5f, 0.05f, 0.0f);
        foc.tune(2.0f * std::numbers::pi_v<float> * 800.0f);

        wet::DirectQuadrature<float> Idq_ref = {.d = 0.0f, .q = 50.0f}; // unreachable -> saturates
        wet::DirectQuadrature<float> Idq = {.d = 0.0f, .q = 0.0f};

        // Output magnitude must stay on/inside the voltage circle throughout.
        for (int k = 0; k < 200; ++k) {
            const auto  cmd = foc.current_controller(Idq_ref, Idq, Ts, Vmax);
            const float Vmag = std::sqrt((cmd.Vdq.d * cmd.Vdq.d) + (cmd.Vdq.q * cmd.Vdq.q));
            CHECK(Vmag <= doctest::Approx(Vmax).epsilon(1e-4f));
            CHECK(cmd.is_saturated); // unreachable ref -> always saturated here
        }

        // Anti-windup: the integrator must stop growing once saturated. Without
        // back-calculation it would ramp by Ki*e*Ts every step; with it, the
        // value over a further 200 saturated steps is essentially unchanged.
        const float i_settled = foc.qctrl.integral;
        for (int k = 0; k < 200; ++k) {
            (void)foc.current_controller(Idq_ref, Idq, Ts, Vmax);
        }
        CHECK(std::abs(foc.qctrl.integral - i_settled) * foc.qctrl.Ki < 1.0f);
    }

    TEST_CASE("step() reports saturation status for cascade anti-windup") {
        wet::FOController<float> foc({.d = 200e-6f, .q = 200e-6f}, 0.5f, 0.05f, 0.0f);
        foc.tune(2.0f * std::numbers::pi_v<float> * 800.0f);

        // Modest reference, ample bus -> no saturation, valid duties.
        const auto ok = foc.step({.d = 0.0f, .q = 1.0f}, {0.1f, -0.05f, -0.05f}, 0.0f, 48.0f, Ts);
        CHECK_FALSE(ok.v_saturated);
        CHECK_FALSE(ok.svm_clipped);
        CHECK(ok.v_excess < 1.0f);
        for (int i = 0; i < 3; ++i) {
            CHECK(ok.duties[i] >= 0.0f);
            CHECK(ok.duties[i] <= 1.0f);
        }

        // Huge reference on a tiny bus -> voltage circle saturates; status flags it
        // and exposes the realized current the outer loop uses as u_sat.
        foc.reset();
        wet::DirectQuadrature<float> big_ref = {.d = 0.0f, .q = 80.0f};
        wet::FocResult<float>        st;
        for (int k = 0; k < 50; ++k) {
            st = foc.step(big_ref, {0.0f, 0.0f, 0.0f}, 0.0f, 2.0f, Ts);
        }
        CHECK(st.v_saturated);
        CHECK(st.v_excess > 1.0f);
        CHECK(st.Idq.q == doctest::Approx(0.0f)); // realized current exposed (fixed plant -> 0)
    }

    TEST_CASE("Cross-axis decoupling feedforward is always applied") {
        // omega != 0, Id != 0 -> the q-axis sees an omega*Ld*Id decoupling term
        // even with PI gains and plant-inversion FF off.
        const float              omega = 3000.0f;
        wet::FOController<float> foc({.d = 200e-6f, .q = 350e-6f}, 0.5f, 0.1f, omega);

        wet::DirectQuadrature<float> Idq = {.d = 5.0f, .q = 0.0f};
        const auto                   cmd = foc.current_controller(Idq, Idq, Ts); // zero error -> PI contributes 0

        // Vq = omega*Ld*Id + omega*lambda ; Vd = -omega*Lq*Iq = 0
        CHECK(cmd.Vdq.q == doctest::Approx((omega * 200e-6f * 5.0f) + (omega * 0.1f)));
        CHECK(cmd.Vdq.d == doctest::Approx(-(omega * 350e-6f * 0.0f)));
    }

    TEST_CASE("plant_inversion_ff toggles the R*I + L*dI/dt term") {
        wet::FOController<float> foc({.d = 200e-6f, .q = 200e-6f}, 0.5f, 0.0f, 0.0f);

        // Isolate the feedforward term: zero the auto-tuned PI gains so only the
        // model-inversion FF contributes to the command.
        foc.dctrl = {};
        foc.qctrl = {};

        wet::DirectQuadrature<float> Idq_ref = {.d = 0.0f, .q = 2.0f};
        wet::DirectQuadrature<float> Idq = {.d = 0.0f, .q = 0.0f};

        // Off (default): no FF and PI gains zeroed above -> zero command.
        CHECK(foc.current_controller(Idq_ref, Idq, Ts).Vdq.q == doctest::Approx(0.0f));

        // On: deadbeat L*dI/dt + R*I appears (R*Iq = 0 since Iq = 0).
        foc.plant_inversion_ff = true;
        const float expected_q = 200e-6f * ((2.0f - 0.0f) / Ts); // L * dIq/dt
        CHECK(foc.current_controller(Idq_ref, Idq, Ts).Vdq.q == doctest::Approx(expected_q));
    }

    TEST_CASE("current_loop_pi returns a clean 2-DOF PIDResult") {
        const double L = 250e-6;
        const double R = 0.35;
        const double wbw = 1500.0;
        const auto   res = wet::design::current_loop_pi(L, R, wbw); // zeta=1, b=1 defaults

        CHECK(res.Kp == doctest::Approx((2.0 * wbw * L) - R));
        CHECK(res.Ki == doctest::Approx(L * wbw * wbw));
        CHECK(res.Kd == doctest::Approx(0.0));
        CHECK(res.Kbc == doctest::Approx(res.Kp)); // T_t = T_i anti-windup seed
        CHECK(res.b == doctest::Approx(1.0));

        // Regression: output/integrator limits must stay unbounded. A positional
        // brace-init slip {Kp, Ki, Kd, Ts} once landed Ts in the u_min slot, which
        // tune() discards (so gains looked fine) but which would silently clamp any
        // PIDController built straight from the result.
        constexpr double inf = std::numeric_limits<double>::max();
        CHECK(res.u_min == -inf);
        CHECK(res.u_max == inf);
        CHECK(res.i_min == -inf);
        CHECK(res.i_max == inf);

        // The setpoint weight propagates for the I-P structure.
        CHECK(wet::design::current_loop_pi(L, R, wbw, 1.0, 0.0).b == doctest::Approx(0.0));
    }

    TEST_CASE("tune() setpoint weight selects PI vs I-P") {
        const float              wbw = 2.0f * std::numbers::pi_v<float> * 800.0f;
        wet::FOController<float> foc({.d = 200e-6f, .q = 200e-6f}, 0.5f, 0.1f, 0.0f);

        foc.tune(wbw); // default b = 1 -> standard PI on both axes
        CHECK(foc.dctrl.b == doctest::Approx(1.0f));
        CHECK(foc.qctrl.b == doctest::Approx(1.0f));

        foc.tune(wbw, 1.0f, 0.0f); // b = 0 -> I-P
        CHECK(foc.dctrl.b == doctest::Approx(0.0f));
        CHECK(foc.qctrl.b == doctest::Approx(0.0f));
    }

    TEST_CASE("I-P removes the proportional step-kick") {
        // omega = 0, lambda = 0 -> all feedforward terms vanish, isolating the PI/I-P.
        const float wbw = 2.0f * std::numbers::pi_v<float> * 800.0f;

        auto make_foc = [&](float b) {
            wet::FOController<float> foc({.d = 200e-6f, .q = 200e-6f}, 0.5f, 0.0f, 0.0f);
            foc.tune(wbw, 1.0f, b);
            return foc;
        };

        wet::DirectQuadrature<float> ref = {.d = 0.0f, .q = 1.0f};
        wet::DirectQuadrature<float> meas = {.d = 0.0f, .q = 0.0f};

        auto pi = make_foc(1.0f);
        auto ip = make_foc(0.0f);

        const float vq_pi = pi.current_controller(ref, meas, Ts).Vdq.q;
        const float vq_ip = ip.current_controller(ref, meas, Ts).Vdq.q;

        // Backward-Euler: on the first tick the integrator is seeded by e*Ts, so both
        // structures share the same integral term (Ki*e*Ts). The structural difference
        // is purely the proportional path: PI feeds Kp*(r - y), I-P feeds Kp*(0 - y),
        // so the step-kick PI carries over I-P is exactly Kp*r.
        CHECK((vq_pi - vq_ip) == doctest::Approx(pi.qctrl.Kp * 1.0f)); // proportional step-kick
        CHECK(vq_ip == doctest::Approx(ip.qctrl.Ki * Ts));             // I-P: only the integral acts
    }

    TEST_CASE("PMSM constant conversions") {
        using namespace wet::design;
        const double p = 7.0;
        const double lambda = 0.00295;

        // Kt <-> flux round trip (amplitude convention Kt = 1.5 p lambda).
        const double Kt = torque_constant_from_flux(p, lambda);
        CHECK(Kt == doctest::Approx(1.5 * p * lambda));
        CHECK(flux_from_torque_constant(p, Kt) == doctest::Approx(lambda));

        // iq for a requested torque inverts Te = Kt iq.
        CHECK(iq_from_torque(2.0, p, lambda) == doctest::Approx(2.0 / Kt));

        // Kv -> Kt uses the peak line-to-line constant 60*sqrt(3)/(4*pi) ~ 8.27.
        const double Kv = 270.0;
        CHECK(torque_constant_from_Kv(Kv) * Kv == doctest::Approx(8.2699).epsilon(1e-3));
        CHECK(flux_from_Kv(p, Kv) == doctest::Approx(flux_from_torque_constant(p, torque_constant_from_Kv(Kv))));

        // Motor constant figure of merit Km = Kt / sqrt(R).
        CHECK(motor_constant(Kt, 0.039) == doctest::Approx(Kt / std::sqrt(0.039)));

        // Voltage-circle radius = m * Vdc / sqrt(3).
        CHECK(voltage_circle_radius(24.0) == doctest::Approx(24.0 / std::sqrt(3.0)));
        CHECK(voltage_circle_radius(24.0, 0.9) == doctest::Approx(0.9 * 24.0 / std::sqrt(3.0)));

        // Base speed: unloaded SPMSM (Id = Iq = 0) reduces to Vmax / lambda.
        const double Vmax = voltage_circle_radius(24.0);
        CHECK(base_speed(Vmax, wet::DirectQuadrature<double>{200e-6, 200e-6}, lambda) == doctest::Approx(Vmax / lambda));
    }
}
