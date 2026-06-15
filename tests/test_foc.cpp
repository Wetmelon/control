
#include <cmath>
#include <numbers>

#include "wet/utility/foc.hpp"
#include "wet/utility/motor_control.hpp"

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
            const auto  Vdq = foc.current_controller(Idq_ref, Idq, Ts, Vmax);
            const float Vmag = std::sqrt((Vdq.d * Vdq.d) + (Vdq.q * Vdq.q));
            CHECK(Vmag <= doctest::Approx(Vmax).epsilon(1e-4f));
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

    TEST_CASE("Cross-axis decoupling feedforward is always applied") {
        // omega != 0, Id != 0 -> the q-axis sees an omega*Ld*Id decoupling term
        // even with PI gains and plant-inversion FF off.
        const float              omega = 3000.0f;
        wet::FOController<float> foc({.d = 200e-6f, .q = 350e-6f}, 0.5f, 0.1f, omega);

        wet::DirectQuadrature<float> Idq = {.d = 5.0f, .q = 0.0f};
        const auto                   Vdq = foc.current_controller(Idq, Idq, Ts); // zero error -> PI contributes 0

        // Vq = omega*Ld*Id + omega*lambda ; Vd = -omega*Lq*Iq = 0
        CHECK(Vdq.q == doctest::Approx((omega * 200e-6f * 5.0f) + (omega * 0.1f)));
        CHECK(Vdq.d == doctest::Approx(-(omega * 350e-6f * 0.0f)));
    }

    TEST_CASE("plant_inversion_ff toggles the R*I + L*dI/dt term") {
        wet::FOController<float> foc({.d = 200e-6f, .q = 200e-6f}, 0.5f, 0.0f, 0.0f);

        wet::DirectQuadrature<float> Idq_ref = {.d = 0.0f, .q = 2.0f};
        wet::DirectQuadrature<float> Idq = {.d = 0.0f, .q = 0.0f};

        // Off (default): no FF and PI is untuned (Kp=Ki=0) -> zero command.
        CHECK(foc.current_controller(Idq_ref, Idq, Ts).q == doctest::Approx(0.0f));

        // On: deadbeat L*dI/dt + R*I appears (R*Iq = 0 since Iq = 0).
        foc.plant_inversion_ff = true;
        const float expected_q = 200e-6f * ((2.0f - 0.0f) / Ts); // L * dIq/dt
        CHECK(foc.current_controller(Idq_ref, Idq, Ts).q == doctest::Approx(expected_q));
    }
}
