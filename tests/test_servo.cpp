#include <cstddef>
#include <vector>

#include "doctest.h"
#include "plot_check.hpp"
#include "wet/matrix/colvec.hpp"
#include "wet/motor/servo.hpp"
#include "wet/simulation/integrator.hpp"
#include "wet/simulation/simulate.hpp"
#include "wet/simulation/solver.hpp"
#include "wet/transforms.hpp"

using namespace wet;
using namespace wet::motor;

namespace {

constexpr float  Vdc = 48.0f;
constexpr double Ts = 1.0 / 20000.0;
constexpr double P = 4.0; // pole pairs (matches motor_config)

PmacServoConfig<float> motor_config() {
    return PmacServoConfig<float>{
        .Ldq = {200e-6f, 200e-6f},
        .R = 0.5f,
        .lambda = 0.01f,
        .pole_pairs = static_cast<float>(P),
        .J = 2e-4f,
        .b = 1e-3f,
        .iq_max = 20.0f,
        .bandwidths = {
            .omega_position = 31.0f,
            .omega_velocity = 314.0f,
            .omega_current = 6283.0f,
        },
        .Ts = static_cast<float>(Ts),
    };
}

// Average dq PMSM + 1-DOF mechanical plant. State x = [id, iq, w, theta]; input u = Vdq.
struct PlantParams {
    double Ld, Lq, R, lambda, J, b;
};
constexpr PlantParams plant_params{.Ld = 200e-6, .Lq = 200e-6, .R = 0.5, .lambda = 0.01, .J = 2e-4, .b = 1e-3};

// Continuous dynamics dx/dt = f(x, Vdq); load torque is read by reference so a test can
// step it mid-simulation.
auto servo_dynamics(const PlantParams& pp, const double& tau_load) {
    return [pp, &tau_load](double /*t*/, const ColVec<4, double>& x, const ColVec<2, double>& u) {
        const double we = P * x[2];
        return ColVec<4, double>{
            (u[0] - (pp.R * x[0]) + (we * pp.Lq * x[1])) / pp.Ld,
            (u[1] - (pp.R * x[1]) - (we * pp.Ld * x[0]) - (we * pp.lambda)) / pp.Lq,
            ((1.5 * P * pp.lambda * x[1]) - tau_load - (pp.b * x[2])) / pp.J,
            x[2],
        };
    };
}

constexpr auto full_state = [](const ColVec<4, double>& x) { return x; };

// Wrap a PmacServo as a sampled controller (t,x) -> Vdq: rebuild the phase-current
// feedback, run the servo, convert its duties back to an applied dq voltage.
template<class Servo>
auto servo_controller(Servo& servo) {
    return [&servo](double /*t*/, const ColVec<4, double>& y) {
        const float theta = static_cast<float>(y[3]);
        const float theta_e = static_cast<float>(P) * theta;
        const auto  Iabc = inverse_park_clarke_transform(
            DirectQuadrature<float>{static_cast<float>(y[0]), static_cast<float>(y[1])}, theta_e
        );
        const auto             res = servo.update(ServoFeedback<float>{.Iabc = Iabc, .Vdc = Vdc, .theta_mech = theta});
        const ColVec<3, float> Vabc{(res.duties[0] - 0.5f) * Vdc, (res.duties[1] - 0.5f) * Vdc, (res.duties[2] - 0.5f) * Vdc};
        const auto             Vdq = clarke_park_transform(Vabc, theta_e);
        return ColVec<2, double>{static_cast<double>(Vdq.d), static_cast<double>(Vdq.q)};
    };
}

// Closed-loop run: servo at the control rate Ts, plant integrated with RK4 (one step per
// period — the servo's cascade is designed at Ts). Returns the full state history.
template<class Servo, class Dyn>
sim::SimulationResult<4, 2, 4, double> drive(Servo& servo, Dyn&& dyn, const ColVec<4, double>& x0, int steps) {
    const sim::FixedStepSolver fine{sim::RK4<4, double>{}, Ts};
    return sim::simulate_sampled<4, 2, 4, double>(
        dyn, full_state, servo_controller(servo), fine, Ts, x0, {0.0, static_cast<double>(steps) * Ts}
    );
}

} // namespace

TEST_SUITE("PmacServo") {

    TEST_CASE("bandwidths validate the cascade separation") {
        CHECK(CascadeBandwidths<float>{31.0f, 314.0f, 6283.0f}.valid());
        CHECK_FALSE(CascadeBandwidths<float>{6283.0f, 314.0f, 31.0f}.valid()); // inverted
    }

    TEST_CASE("torque mode drives iq to the commanded torque") {
        PmacServo<float> servo{motor_config()};
        servo.set_mode(ControlMode::Torque);
        servo.set_target(0.1f); // Nm; Kt = 1.5*4*0.01 = 0.06 -> iq ~ 1.667 A

        double     tau = 0.0;
        const auto res = drive(servo, servo_dynamics(plant_params, tau), ColVec<4, double>{}, 4000); // 0.2 s
        CHECK(res.x.back()[1] == doctest::Approx(0.1 / 0.06).epsilon(0.05));
    }

    TEST_CASE("velocity mode tracks the speed command and rejects a load") {
        PmacServo<float> servo{motor_config()};
        servo.set_mode(ControlMode::Velocity);
        servo.set_target(100.0f); // rad/s

        double     tau = 0.0;
        const auto dyn = servo_dynamics(plant_params, tau);
        const auto r1 = drive(servo, dyn, ColVec<4, double>{}, 10000); // 0.5 s
        CHECK(r1.x.back()[2] == doctest::Approx(100.0).epsilon(0.02));

        tau = 0.05; // Nm load step
        const auto r2 = drive(servo, dyn, r1.x.back(), 10000);
        CHECK(r2.x.back()[2] == doctest::Approx(100.0).epsilon(0.02)); // integral action rejects it
    }

    TEST_CASE("position mode settles at the commanded angle") {
        PmacServo<float> servo{motor_config()};
        servo.set_mode(ControlMode::Position);
        servo.set_target(5.0f); // rad

        double     tau = 0.0;
        const auto res = drive(servo, servo_dynamics(plant_params, tau), ColVec<4, double>{}, 40000); // 2 s
        CHECK(res.x.back()[3] == doctest::Approx(5.0).epsilon(0.02));
        CHECK(res.x.back()[2] == doctest::Approx(0.0).epsilon(0.05)); // at rest
    }

    TEST_CASE("the q-current ceiling is respected") {
        auto cfg = motor_config();
        cfg.iq_max = 3.0f;
        PmacServo<float> servo{cfg};
        servo.set_mode(ControlMode::Torque);
        servo.set_target(5.0f); // 5 Nm -> iq ~ 83 A demanded, must clamp to 3 A

        double     tau = 0.0;
        const auto res = drive(servo, servo_dynamics(plant_params, tau), ColVec<4, double>{}, 2000);
        CHECK(res.x.back()[1] <= doctest::Approx(3.0).epsilon(0.05));
    }

    TEST_CASE("thermal derate shrinks the current ceiling") {
        auto cfg = motor_config();
        cfg.iq_max = 10.0f;
        PmacServo<float> servo{cfg};
        servo.set_mode(ControlMode::Torque);
        servo.set_target(5.0f);        // demands far more than the ceiling
        servo.set_thermal_scale(0.2f); // 20% derate -> ceiling 2 A

        double     tau = 0.0;
        const auto res = drive(servo, servo_dynamics(plant_params, tau), ColVec<4, double>{}, 2000);
        CHECK(res.x.back()[1] <= doctest::Approx(2.0).epsilon(0.1));
    }

    // // Under load the voltage circle caps this motor's id=0 speed near ~560 mech (the
    // // back-EMF ω·λ leaves no room for the torque current). The default servo plateaus
    // // there; a FieldWeakening policy drives id<0 to climb past it. (The weak magnet vs
    // // the 20 A limit caps even the weakened ceiling near ~640 mech, so 600 is a target
    // // the plain servo cannot reach but the weakened one can.)
    // TEST_CASE("Plot: field-weakening policy lets PmacServo exceed base speed") {
    //     const auto  cfg = motor_config();
    //     const float target = 600.0f; // mech rad/s, above the plain servo's loaded ceiling
    //     const int   steps = 60000;   // 3 s
    //     double      tau = 0.0;

    //     PmacServo<float> plain{cfg};
    //     plain.set_mode(ControlMode::Velocity);
    //     plain.set_target(target);
    //     const auto rp = drive(plain, servo_dynamics(plant_params, tau), ColVec<4, double>{}, steps);

    //     const FieldWeakeningConfig<float> fwc{
    //         .Ldq = cfg.Ldq,
    //         .lambda = cfg.lambda,
    //         .i_max = cfg.iq_max,
    //         .v_margin = 0.95f,
    //         .ki = 50.0f,
    //         .method = FwMethod::VoltageFeedback
    //     };
    //     PmacServo<float, FieldWeakening<float>> weak{cfg, FieldWeakening<float>{fwc}};
    //     weak.set_mode(ControlMode::Velocity);
    //     weak.set_target(target);
    //     const auto rw = drive(weak, servo_dynamics(plant_params, tau), ColVec<4, double>{}, steps);

    //     const double tgt = static_cast<double>(target);
    //     CHECK(rp.x.back()[2] < tgt - 15.0);                          // plain can't reach the command
    //     CHECK(rw.x.back()[2] == doctest::Approx(tgt).epsilon(0.05)); // weakening reaches it
    //     CHECK(rw.x.back()[2] > rp.x.back()[2] + 20.0);
    //     CHECK(rw.x.back()[0] < -0.5); // weakening engaged (id driven negative)

    //     std::vector<double> tt, sp_plain, sp_weak, id_plain, id_weak;
    //     for (size_t k = 0; k < rp.t.size(); k += 50) {
    //         tt.push_back(rp.t[k]);
    //         sp_plain.push_back(rp.x[k][2]);
    //         sp_weak.push_back(rw.x[k][2]);
    //         id_plain.push_back(rp.x[k][0]);
    //         id_weak.push_back(rw.x[k][0]);
    //     }

    //     plotcheck::xy("servo_field_weakening_speed.html", "PmacServo to 600 rad/s (above base): field weakening vs none", "time (s)", "mechanical speed (rad/s)", {{.name = "with field weakening", .x = tt, .y = sp_weak}, {.name = "no field weakening", .x = tt, .y = sp_plain}});
    //     plotcheck::xy("servo_field_weakening_id.html", "PmacServo d-axis current: field weakening drives id negative", "time (s)", "id (A)", {{.name = "with field weakening", .x = tt, .y = id_weak}, {.name = "no field weakening", .x = tt, .y = id_plain}});
    // }
}
