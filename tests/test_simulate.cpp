#include "wet/controllers/lqr.hpp"
#include "wet/math/wetmelon_math.hpp"
#include "wet/simulation/simulate.hpp"
#include "wet/simulation/solver.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wetmelon::control;

// Pendulum parameters for simulation tests
constexpr double g_pend = 9.81;
constexpr double L_pend = 1.0;
constexpr double m_pend = 1.0;
constexpr double b_pend = 0.1;

TEST_CASE("simulate_state_feedback - pendulum converges under LQR") {
    // Linearize at upright equilibrium
    Matrix<2, 2> A{{0.0, 1.0}, {g_pend / L_pend, -b_pend / (m_pend * L_pend * L_pend)}};
    Matrix<2, 1> B{{0.0}, {1.0 / (m_pend * L_pend * L_pend)}};

    auto Q = Matrix<2, 2>::identity() * 10.0;
    auto R = Matrix<1, 1>{{1.0}};
    auto Ts = 0.01;

    auto lqr_result = design::discrete_lqr_from_continuous(A, B, Q, R, Ts);
    CHECK(lqr_result.success);
    LQR controller{lqr_result};

    // Nonlinear plant
    auto plant = [](double /*t*/, const ColVec<2>& x, const ColVec<1>& u) -> ColVec<2> {
        double theta = x(0, 0);
        double theta_dot = x(1, 0);
        double torque = u(0, 0);

        double theta_ddot = ((g_pend / L_pend) * wet::sin(theta))
                          - ((b_pend / (m_pend * L_pend * L_pend)) * theta_dot)
                          + ((1.0 / (m_pend * L_pend * L_pend)) * torque);

        return ColVec<2>{theta_dot, theta_ddot};
    };

    auto output = [](const ColVec<2>& x) -> ColVec<2> { return x; };
    auto ctrl = [&](const ColVec<2>& x) -> ColVec<1> { return controller.control(x); };

    RK4<2>          rk4;
    FixedStepSolver solver(rk4, 0.001);

    // Start at 15 degrees
    ColVec<2> x0{0.2618, 0.0};
    auto      sim = simulate_state_feedback<2, 1, 2>(plant, output, ctrl, solver, x0, {0.0, 5.0});

    // Should converge to near zero by 5 seconds
    CHECK(sim.x.back()(0, 0) == doctest::Approx(0.0).epsilon(0.01));
    CHECK(sim.x.back()(1, 0) == doctest::Approx(0.0).epsilon(0.01));

    // Check that we recorded everything
    CHECK(sim.t.size() == sim.x.size());
    CHECK(sim.t.size() == sim.y.size());
    CHECK(sim.t.size() == sim.u.size());
}

TEST_CASE("simulate_lti - matches step_response for zero-input") {
    // Simple 1st-order system: dx/dt = -x, y = x
    Matrix<1, 1> A{{-1.0}};
    Matrix<1, 1> B{{1.0}};
    Matrix<1, 1> C{{1.0}};
    StateSpace   sys{.A = A, .B = B, .C = C, .Ts = 0.0};

    // Zero control (free response)
    auto controller = [](const ColVec<1>& /*y*/) -> ColVec<1> {
        return ColVec<1>{0.0};
    };

    RK4<1>          rk4;
    FixedStepSolver solver(rk4, 0.01);

    ColVec<1> x0{1.0};
    auto      sim = simulate_lti(sys, controller, solver, x0, {0.0, 3.0});

    // x(t) = exp(-t), so x(3) ≈ 0.0498
    CHECK(sim.x.back()(0, 0) == doctest::Approx(std::exp(-3.0)).epsilon(1e-6));
}

TEST_CASE("simulate_discrete - basic feedback") {
    // Discrete integrator: x_{k+1} = x_k + Ts * u_k
    double       Ts = 0.1;
    Matrix<1, 1> A_d{{1.0}};
    Matrix<1, 1> B_d{{Ts}};
    Matrix<1, 1> C_d{{1.0}};
    StateSpace   sys{.A = A_d, .B = B_d, .C = C_d, .Ts = Ts};

    // Simple proportional controller: u = -10 * y (drives x to 0)
    auto controller = [](const ColVec<1>& y) -> ColVec<1> {
        return ColVec<1>{-10.0 * y(0, 0)};
    };

    ColVec<1> x0{1.0};
    auto      sim = simulate_discrete(sys, controller, x0, 100);

    CHECK(sim.t.size() == 101);
    CHECK(sim.t.front() == 0.0);
    CHECK(sim.t.back() == doctest::Approx(10.0).epsilon(1e-10));

    // State should converge to zero
    CHECK(sim.x.back()(0, 0) == doctest::Approx(0.0).epsilon(1e-6));
}

TEST_CASE("simulate - output feedback with D matrix") {
    // Controller: u = 0 (open loop, step input via plant)
    auto plant = [](double /*t*/, const ColVec<1>& x, const ColVec<1>& u) -> ColVec<1> {
        return ColVec<1>{(-2.0 * x(0, 0)) + u(0, 0)};
    };
    auto output = [](const ColVec<1>& x) -> ColVec<1> { return x; };
    auto ctrl = [](const ColVec<1>& /*y*/) -> ColVec<1> { return ColVec<1>{1.0}; }; // Constant input

    RK4<1>          rk4;
    FixedStepSolver solver(rk4, 0.01);

    ColVec<1> x0{0.0};
    auto      sim = simulate<1, 1, 1>(plant, output, ctrl, solver, x0, {0.0, 5.0});

    // Steady state: dx/dt=0 => x = u/2 = 0.5
    CHECK(sim.x.back()(0, 0) == doctest::Approx(0.5).epsilon(0.01));
}

TEST_CASE("simulate_discrete_nonlinear - logistic-style nonlinear map") {
    // Natively-discrete nonlinear plant with no continuous-time analogue:
    //   x[k+1] = x[k] + Ts*(-x[k]^3 + u[k])
    // A cubic restoring term — not expressible as Ax+Bu, so this must use the
    // discrete nonlinear path, not simulate_discrete (linear) or the ODE solver.
    const double Ts = 0.01;

    auto plant = [&](std::size_t /*k*/, const ColVec<1>& x, const ColVec<1>& u) -> ColVec<1> {
        const double xv = x(0, 0);
        return ColVec<1>{xv + (Ts * (-(xv * xv * xv) + u(0, 0)))};
    };
    auto output = [](const ColVec<1>& x) -> ColVec<1> { return x; };
    // Constant input u = 8 → equilibrium where x^3 = 8 → x = 2.
    auto ctrl = [](const ColVec<1>& /*y*/) -> ColVec<1> { return ColVec<1>{8.0}; };

    auto sim = simulate_discrete_nonlinear<1, 1, 1>(plant, output, ctrl, ColVec<1>{0.0}, Ts, 5000);

    CHECK(sim.t.size() == 5001);
    CHECK(sim.t.front() == 0.0);
    CHECK(sim.t.back() == doctest::Approx(50.0).epsilon(1e-9));
    CHECK(sim.x.back()(0, 0) == doctest::Approx(2.0).epsilon(1e-3)); // settles at cube root of 8
}

TEST_CASE("simulate_discrete_nonlinear - matches simulate_discrete on a linear plant") {
    // When f(k,x,u) = Ax + Bu, the nonlinear path must reproduce the linear one.
    const double Ts = 0.1;
    Matrix<1, 1> A{{1.0}};
    Matrix<1, 1> B{{Ts}};
    Matrix<1, 1> C{{1.0}};
    StateSpace   sys{.A = A, .B = B, .C = C, .Ts = Ts};

    auto ctrl = [](const ColVec<1>& y) -> ColVec<1> { return ColVec<1>{-10.0 * y(0, 0)}; };

    auto lin = simulate_discrete(sys, ctrl, ColVec<1>{1.0}, 100);

    auto plant = [&](std::size_t, const ColVec<1>& x, const ColVec<1>& u) -> ColVec<1> {
        return A * x + B * u;
    };
    auto output = [&](const ColVec<1>& x) -> ColVec<1> { return C * x; };
    auto nl = simulate_discrete_nonlinear<1, 1, 1>(plant, output, ctrl, ColVec<1>{1.0}, Ts, 100);

    REQUIRE(lin.x.size() == nl.x.size());
    for (std::size_t k = 0; k < lin.x.size(); ++k) {
        CHECK(nl.x[k](0, 0) == doctest::Approx(lin.x[k](0, 0)));
        CHECK(nl.t[k] == doctest::Approx(lin.t[k]));
    }
}

TEST_CASE("SimulationResult has consistent sizes") {
    auto plant = [](double, const ColVec<1>& x, const ColVec<1>& u) -> ColVec<1> { return -x + u; };
    auto output = [](const ColVec<1>& x) -> ColVec<1> { return x; };
    auto ctrl = [](const ColVec<1>&) -> ColVec<1> { return ColVec<1>{0.0}; };

    RK4<1>          rk4;
    FixedStepSolver solver(rk4, 0.1);

    auto sim = simulate<1, 1, 1>(plant, output, ctrl, solver, ColVec<1>{1.0}, {0.0, 1.0});

    CHECK(sim.t.size() == sim.x.size());
    CHECK(sim.t.size() == sim.y.size());
    CHECK(sim.t.size() == sim.u.size());
    CHECK(sim.t.front() == 0.0);
    CHECK(sim.t.back() == doctest::Approx(1.0).epsilon(0.01));
}
