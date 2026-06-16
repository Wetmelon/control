
#include <cmath>
#include <numbers>

#include "wet/utility/foc.hpp"
#include "wet/utility/modulation.hpp"

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
}
