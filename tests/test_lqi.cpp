#include <fmt/format.h>
#include <fmt/ostream.h>

#include "doctest.h"
#include "integrator.hpp"
#include "lqi.hpp"
#include "state_space.hpp"

using namespace wetmelon::control;

TEST_CASE("design::lqi matches scipy/control golden data") {
    // Simple system with integral action

    constexpr StateSpace sys = {
        .A = Matrix<2, 2>{{1.0, 0.1}, {0.0, 0.9}},
        .B = Matrix<2, 1>{{0.0}, {0.1}},
        .C = Matrix<1, 2>{{1.0, 0.0}},
        .D = Matrix<1, 1>{{0.0}},
    };

    constexpr Matrix<3, 3> Q_aug{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    constexpr Matrix<1, 1> R{{1.0}};

    constexpr auto result = design::lqi(sys, Q_aug, R);

    static_assert(result.success);

    // Check that gains are reasonable
    CHECK(result.K(0, 0) != 0.0);
    CHECK(result.K(0, 1) != 0.0);
    CHECK(result.K(0, 2) != 0.0);

    // Check Riccati solution
    CHECK(result.S(0, 0) > 0.0);
    CHECK(result.S(1, 1) > 0.0);
    CHECK(result.S(2, 2) > 0.0);
}

TEST_CASE("Capacitor LQI control simulation") {
    // Capacitor model: dx/dt = (1/C) * i, y = x (voltage)
    constexpr double C = 1e-3; // Capacitance
    constexpr size_t NX = 1, NU = 1, NY = 1, NX_aug = NX + NY;

    constexpr StateSpace sys = {
        .A = Matrix<NX, NX>{},          // zero dynamics
        .B = Matrix<NX, NU>{{1.0 / C}}, // input is current
        .C = Matrix<NY, NX>{{1.0}},     // output is voltage
        .D = Matrix<NY, NU>{},          // no direct feedthrough
    };

    // Design double-precision LQI at compile-time (augments state with integral error)
    constexpr Matrix Q = Matrix<NX_aug, NX_aug>::identity() * 1.0;
    constexpr Matrix R = Matrix<NU, NU>{{1.0}};

    // Get LQI design result
    constexpr auto lqi_result = design::lqi(sys, Q, R);
    REQUIRE(lqi_result.success);

    // Instantiate single-precision runtime controller
    LQI controller = lqi_result.as<float>();

    // Simulation parameters
    constexpr double h = 10e-6;   // Time step [s]`
    constexpr double t_end = 0.5; // Simulation time [s]

    float i_term = 0.0f;  // Integral term
    float v_ref = 800.0f; // Reference target

    // Integrator for augmented system
    RK45<NX, double> integrator;

    // Open CSV file with fmt
    std::ofstream csv_file("capacitor_lqi_simulation.csv");
    fmt::print(csv_file, "time,x,y,u\n");

    // Simulation loop
    ColVec u = {0.0}; // Initial input current
    ColVec x = {0.0}; // Initial capacitor voltage
    for (double t = 0.0; t <= t_end; t += h) {
        // Get output from plant state (first NX elements of augmented state)
        ColVec<NY, double> y = sys.C * x + sys.D * u;

        // Compute control using augmented state
        i_term += v_ref - (float)(y(0));               // Update integral error
        u = controller.control({(float)x(0), i_term}); // Compute control input

        // Evolve augmented system
        auto result = integrator.evolve(sys.A, sys.B, x, u.as<double>(), h);
        x(0) = result.x(0); // Update plant state

        // Write to CSV
        fmt::print(csv_file, "{:04.6f},{:04.6f},{:04.6f},{:04.6f}\n", t, x(0), y(0), u(0));
    }

    // Check final voltage is close to reference
    CHECK(x(0) == doctest::Approx((double)v_ref).epsilon(0.01));

    csv_file.close();
    fmt::print("Simulation complete. Results saved to capacitor_lqi_simulation.csv\n");
}

TEST_CASE("LQIResult::as<U>() conversion") {
    constexpr auto lqi_d = design::lqi(
        StateSpace<1, 1, 1, 1, 1>{Matrix<1, 1>{{1.0}}, Matrix<1, 1>{{1.0}}, Matrix<1, 1>{{1.0}}, Matrix<1, 1>::zeros()},
        Matrix<2, 2>{{1.0, 0.0}, {0.0, 1.0}}, Matrix<1, 1>{{1.0}}
    );

    constexpr auto lqi_f = lqi_d.as<float>();

    static_assert(lqi_f.success);
    CHECK(lqi_f.success);
    CHECK(lqi_f.K(0, 0) != 0.0f);
    CHECK(lqi_f.K(0, 1) != 0.0f);
}
