#include <numbers>

#include "fmt/core.h"
#include "wet/analysis/analysis.hpp"
#include "wet/analysis/linearization.hpp"
#include "wet/controllers/pid_design.hpp"
#include "wet/controllers/pr.hpp"
#include "wet/controllers/synthesis.hpp"
#include "wet/simulation/simulate.hpp"
#include "wet/simulation/solver.hpp"
#include "wet/systems/discretization.hpp"

using namespace wetmelon::control;

int main() {
    constexpr double Ts = 0.001; // 1 kHz

    // Nonlinear plant: x = [position, velocity], u = force/torque
    auto plant_nonlinear = [](double /*t*/, const ColVec<2>& x, const ColVec<1>& u) -> ColVec<2> {
        const double x1 = x(0, 0);
        const double x2 = x(1, 0);
        const double uu = u(0, 0);

        const double x1_dot = x2;
        const double x2_dot = (-0.8 * x2) - (2.0 * std::sin(x1)) + (1.5 * uu);
        return ColVec<2>{x1_dot, x2_dot};
    };

    auto output = [](const ColVec<2>& x) -> ColVec<1> {
        return ColVec<1>{x(0, 0)};
    };

    // 1) Linearize around operating point
    const ColVec<2> x_op{0.0, 0.0};
    const ColVec<1> u_op{0.0};
    const auto      lin = analysis::linearize<2, 1, 1>(plant_nonlinear, output, x_op, u_op);

    StateSpace<2, 1, 1, 2, 1> sys_c{
        .A = lin.A,
        .B = lin.B,
        .C = lin.C,
        .D = lin.D,
        .G = Matrix<2, 2>::identity(),
        .H = Matrix<1, 1>::identity(),
        .Ts = 0.0
    };

    const auto sys_d = discretize(sys_c, Ts, DiscretizationMethod::ZOH);

    // 2) Loop-shaping style controller seed from settling/overshoot specs
    const design::PIDPerformanceSpec<double> pid_spec{
        .settling_time = 0.20,
        .overshoot_percent = 10.0,
        .Ts = Ts,
        .type = design::PIDType::PI,
        .bandwidth_scale = 1.0
    };

    const auto pid_seed = design::pid_from_performance_spec(pid_spec);

    // 3) Observer + servo controller synthesis (LQGI) + optional PR internal model
    const auto Q_aug = Matrix<3, 3>::diagonal({20.0, 2.0, 80.0});

    const Matrix<1, 1> R{{0.25}};
    const Matrix<2, 2> Q_kf{{1e-3, 0.0}, {0.0, 1e-2}};
    const Matrix<1, 1> R_kf{{5e-3}};

    const auto artifacts = design::synthesize_lqgi(sys_d, Q_aug, R, Q_kf, R_kf);

    const auto pr_design = design::pr(0.0, 10.0, 2.0 * std::numbers::pi, 6.0, Ts);

    auto                runtime = artifacts.runtime; // default float runtime bundle
    PRController<float> pr_runtime(pr_design.as<float>());

    // 4) Frequency-domain analysis data from linearized model.
    //    logspace takes actual frequency endpoints (rad/s), not exponents:
    //    sweep 1 rad/s to 1000 rad/s. bode() auto-dispatches on the discrete Ts.
    const auto omega = analysis::logspace(1.0, 1000.0, 150);
    const auto bode_open = analysis::bode(sys_d, omega);
    const auto nyq_open = analysis::nyquist(sys_d, omega);

    // 5) Nonlinear closed-loop simulation with observer + internal-model compensation
    RK4<2>          rk4;
    FixedStepSolver solver(rk4, Ts);

    const float reference = 0.5f;
    auto        controller = [&](const ColVec<1>& y) -> ColVec<1> {
        const float y_f = static_cast<float>(y(0, 0));
        const float u_lqgi = runtime.step(y_f, reference)(0, 0);
        const float u_pr = pr_runtime.control(reference - y_f);
        return ColVec<1>{static_cast<double>(u_lqgi + u_pr)};
    };

    const ColVec<2> x0{0.25, 0.0};
    const auto      sim = simulate<2, 1, 1>(plant_nonlinear, output, controller, solver, x0, {0.0, 2.0});

    fmt::print("Workflow end-to-end summary\n");
    fmt::print("  Linearized A(1,0): {:.6f}\n", lin.A(1, 0));
    fmt::print("  PID seed from specs (PI): Kp={:.6f}, Ki={:.6f}\n", pid_seed.Kp, pid_seed.Ki);
    fmt::print("  LQGI synthesis success: {}\n", artifacts.success ? "true" : "false");
    fmt::print("  Bode points: {}, Nyquist points: {}\n", bode_open.points.size(), nyq_open.points.size());
    fmt::print("  Final output: {:.6f}\n", sim.y.back()(0, 0));
    fmt::print("  Final control: {:.6f}\n", sim.u.back()(0, 0));

    return 0;
}
