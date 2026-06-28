
#include "doctest.h"
#include "wet/matrix/colvec.hpp"
#include "wet/motor/servo.hpp"
#include "wet/simulation/integrator.hpp"
#include "wet/transforms.hpp"

using namespace wet;
using namespace wet::motor;

namespace {

// Average dq PMSM + 1-DOF mechanical plant, driven by an applied dq voltage. The dq
// cross-coupling makes this nonlinear, so it is integrated with the library's generic
// f(t,x) ODE solver (one RK4 step per call) rather than hand-rolled Euler substepping.
struct Plant {
    ColVec<4, double>   x{}; // [id, iq, w, theta]: currents [A], speed [rad/s], angle [rad]
    double              Ld, Lq, R, lambda, p, J, b;
    double              tau_load{0};
    sim::RK4<4, double> solver{};

    [[nodiscard]] double id() const { return x[0]; }
    [[nodiscard]] double iq() const { return x[1]; }
    [[nodiscard]] double w() const { return x[2]; }
    [[nodiscard]] double th() const { return x[3]; }

    void step(double vd, double vq, double Ts) {
        auto dynamics = [&](double, const ColVec<4, double>& s) {
            const double we = p * s[2];
            return ColVec<4, double>{
                (vd - (R * s[0]) + (we * Lq * s[1])) / Ld,                 // d(id)/dt
                (vq - (R * s[1]) - (we * Ld * s[0]) - (we * lambda)) / Lq, // d(iq)/dt
                ((1.5 * p * lambda * s[1]) - tau_load - (b * s[2])) / J,   // d(w)/dt
                s[2],                                                      // d(theta)/dt = w
            };
        };
        x = solver.evolve(dynamics, x, 0.0, Ts).x;
    }
};

constexpr float Vdc = 48.0f;
constexpr float Ts = 1.0f / 20000.0f;

PmacServoConfig<float> motor_config() {
    return PmacServoConfig<float>{
        .Ldq = {200e-6f, 200e-6f},
        .R = 0.5f,
        .lambda = 0.01f,
        .pole_pairs = 4.0f,
        .J = 2e-4f,
        .b = 1e-3f,
        .iq_max = 20.0f,
        .bandwidths = {
            .omega_position = 31.0f,
            .omega_velocity = 314.0f,
            .omega_current = 6283.0f,
        },
        .Ts = Ts,
    };
}

Plant motor_plant() {
    return Plant{.Ld = 200e-6, .Lq = 200e-6, .R = 0.5, .lambda = 0.01, .p = 4.0, .J = 2e-4, .b = 1e-3};
}

// Close the loop: currents -> abc feedback -> servo -> duties -> applied dq voltage.
void run(PmacServo<float>& servo, Plant& plant, int steps) {
    for (int k = 0; k < steps; ++k) {
        const float theta_e = static_cast<float>(plant.p * plant.th());
        const auto  Iabc = inverse_park_clarke_transform(
            DirectQuadrature<float>{static_cast<float>(plant.id()), static_cast<float>(plant.iq())}, theta_e
        );

        const auto res = servo.update(ServoFeedback<float>{.Iabc = Iabc, .Vdc = Vdc, .theta_mech = static_cast<float>(plant.th())});

        const ColVec<3, float> Vabc{
            (res.duties[0] - 0.5f) * Vdc, (res.duties[1] - 0.5f) * Vdc, (res.duties[2] - 0.5f) * Vdc
        };
        const auto Vdq = clarke_park_transform(Vabc, theta_e);
        plant.step(Vdq.d, Vdq.q, Ts);
    }
}

} // namespace

TEST_SUITE("PmacServo") {

    TEST_CASE("bandwidths validate the cascade separation") {
        CHECK(CascadeBandwidths<float>{31.0f, 314.0f, 6283.0f}.valid());
        CHECK_FALSE(CascadeBandwidths<float>{6283.0f, 314.0f, 31.0f}.valid()); // inverted
    }

    TEST_CASE("torque mode drives iq to the commanded torque") {
        PmacServo<float> servo{motor_config()};
        Plant            plant = motor_plant();
        servo.set_mode(ControlMode::Torque);
        servo.set_target(0.1f); // Nm; Kt = 1.5*4*0.01 = 0.06 -> iq ~ 1.667 A

        run(servo, plant, 4000); // 0.2 s
        CHECK(plant.iq() == doctest::Approx(0.1 / 0.06).epsilon(0.05));
    }

    TEST_CASE("velocity mode tracks the speed command and rejects a load") {
        PmacServo<float> servo{motor_config()};
        Plant            plant = motor_plant();
        servo.set_mode(ControlMode::Velocity);
        servo.set_target(100.0f); // rad/s

        run(servo, plant, 10000); // 0.5 s
        CHECK(plant.w() == doctest::Approx(100.0).epsilon(0.02));

        plant.tau_load = 0.05; // Nm load step
        run(servo, plant, 10000);
        CHECK(plant.w() == doctest::Approx(100.0).epsilon(0.02)); // integral action rejects it
    }

    TEST_CASE("position mode settles at the commanded angle") {
        PmacServo<float> servo{motor_config()};
        Plant            plant = motor_plant();
        servo.set_mode(ControlMode::Position);
        servo.set_target(5.0f); // rad

        run(servo, plant, 40000); // 2 s
        CHECK(plant.th() == doctest::Approx(5.0).epsilon(0.02));
        CHECK(plant.w() == doctest::Approx(0.0).epsilon(0.05)); // at rest
    }

    TEST_CASE("the q-current ceiling is respected") {
        auto cfg = motor_config();
        cfg.iq_max = 3.0f;
        PmacServo<float> servo{cfg};
        Plant            plant = motor_plant();
        servo.set_mode(ControlMode::Torque);
        servo.set_target(5.0f); // 5 Nm -> iq ~ 83 A demanded, must clamp to 3 A

        run(servo, plant, 2000);
        CHECK(plant.iq() <= doctest::Approx(3.0).epsilon(0.05));
    }

    TEST_CASE("thermal derate shrinks the current ceiling") {
        auto cfg = motor_config();
        cfg.iq_max = 10.0f;
        PmacServo<float> servo{cfg};
        Plant            plant = motor_plant();
        servo.set_mode(ControlMode::Torque);
        servo.set_target(5.0f);        // demands far more than the ceiling
        servo.set_thermal_scale(0.2f); // 20% derate -> ceiling 2 A

        run(servo, plant, 2000);
        CHECK(plant.iq() <= doctest::Approx(2.0).epsilon(0.1));
    }
}
