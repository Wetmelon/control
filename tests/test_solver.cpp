#include "plot_plotly.hpp"
#include "solver.hpp"

using namespace wetmelon::control;

#include "doctest.h"

// Simple exponential decay: dx/dt = -x, x(0) = 1 => x(t) = exp(-t)
static auto exp_decay = [](double /*t*/, const ColVec<1>& x) -> ColVec<1> {
    return ColVec<1>{{-x(0, 0)}};
};

TEST_CASE("SolveResult iterator") {
    SolveResult<1> result;
    result.t = {0.0, 0.5, 1.0};
    result.x = {ColVec<1>{{1.0}}, ColVec<1>{{0.6}}, ColVec<1>{{0.3}}};

    size_t count = 0;
    for (const auto& [t, x] : result) {
        CHECK(t == result.t[count]);
        CHECK(x(0, 0) == result.x[count](0, 0));
        ++count;
    }
    CHECK(count == 3);
    CHECK(result.size() == 3);
}

TEST_CASE("FixedStepSolver - RK4 exponential decay") {
    RK4<1>          rk4;
    FixedStepSolver solver(rk4, 0.01);

    ColVec<1> x0{{1.0}};
    auto      result = solver.solve(exp_decay, x0, {0.0, 1.0});

    CHECK(result.success);
    CHECK(result.t.size() > 1);

    // x(1) = exp(-1) ≈ 0.3678794
    double x_final = result.x.back()(0, 0);
    CHECK(x_final == doctest::Approx(0.36787944117).epsilon(1e-8));
}

TEST_CASE("FixedStepSolver - matches manual RK4 loop") {
    // Manually integrate with RK4 and compare
    RK4<1>    rk4;
    double    h = 0.1;
    double    t = 0.0;
    ColVec<1> x{{1.0}};

    for (int i = 0; i < 10; ++i) {
        auto r = rk4.evolve(exp_decay, x, t, h);
        x = r.x;
        t += h;
    }
    double manual_result = x(0, 0);

    FixedStepSolver solver(rk4, 0.1);
    auto            result = solver.solve(exp_decay, ColVec<1>{{1.0}}, {0.0, 1.0});

    CHECK(result.x.back()(0, 0) == doctest::Approx(manual_result).epsilon(1e-14));
}

TEST_CASE("FixedStepSolver - stop condition") {
    RK4<1>          rk4;
    FixedStepSolver solver(rk4, 0.01);

    // Stop when x < 0.5
    solver.set_stop_condition([](double /*t*/, const ColVec<1>& x) {
        return x(0, 0) < 0.5;
    });

    ColVec<1> x0{{1.0}};
    auto      result = solver.solve(exp_decay, x0, {0.0, 5.0});

    // Should have stopped well before t=5
    CHECK(result.t.back() < 1.0);
    CHECK(result.x.back()(0, 0) < 0.5);
}

TEST_CASE("FixedStepSolver - 2D harmonic oscillator") {
    // dx/dt = [x1, -x0]  ==> x0(t)=cos(t), x1(t)=-sin(t)
    auto harmonic = [](double /*t*/, const ColVec<2>& x) -> ColVec<2> {
        return ColVec<2>{{x(1, 0)}, {-x(0, 0)}};
    };

    RK4<2>          rk4;
    FixedStepSolver solver(rk4, 0.001);

    ColVec<2> x0{{1.0}, {0.0}};
    auto      result = solver.solve(harmonic, x0, {0.0, 6.283185307});

    // After one full period, should return near initial condition
    CHECK(result.x.back()(0, 0) == doctest::Approx(1.0).epsilon(1e-6));
    CHECK(result.x.back()(1, 0) == doctest::Approx(0.0).epsilon(1e-6));

    // Plot x0=cos(t) and x1=-sin(t)
    auto t = plot::to_double_vector(result.t);
    auto x0_hist = plot::extract_channel(result.x, 0);
    auto x1_hist = plot::extract_channel(result.x, 1);

    using namespace plotlypp;
    auto fig = Figure()
                   .addTrace(Scatter().x(t).y(x0_hist).mode({Scatter::Mode::Lines}).name("x0 = cos(t)"))
                   .addTrace(Scatter().x(t).y(x1_hist).mode({Scatter::Mode::Lines}).name("x1 = -sin(t)"))
                   .setLayout(Layout().title([](auto& t) { t.text("Harmonic Oscillator"); }).xaxis(Layout::Xaxis().title([](auto& t) { t.text("Time (s)"); })).yaxis(Layout::Yaxis().title([](auto& t) { t.text("State"); })));
    fig.writeHtml("tests/build/harmonic_oscillator.html");
}

TEST_CASE("AdaptiveStepSolver - RK45 exponential decay") {
    RK45<1>            rk45;
    AdaptiveStepSolver solver(rk45, 0.1, 1e-8);

    ColVec<1> x0{{1.0}};
    auto      result = solver.solve(exp_decay, x0, {0.0, 1.0});

    CHECK(result.success);
    CHECK(result.x.back()(0, 0) == doctest::Approx(0.36787944117).epsilon(1e-6));
}

TEST_CASE("AdaptiveStepSolver - fewer steps than fixed for smooth problem") {
    RK45<1>            rk45;
    AdaptiveStepSolver adaptive(rk45, 0.1, 1e-6);

    RK4<1>          rk4;
    FixedStepSolver fixed_solver(rk4, 0.001);

    ColVec<1> x0{{1.0}};
    auto      result_adaptive = adaptive.solve(exp_decay, x0, {0.0, 2.0});
    auto      result_fixed = fixed_solver.solve(exp_decay, x0, {0.0, 2.0});

    // Adaptive should use fewer function evaluations for same accuracy
    CHECK(result_adaptive.nfev < result_fixed.nfev);

    // Both should converge to the same answer
    CHECK(result_adaptive.x.back()(0, 0) == doctest::Approx(result_fixed.x.back()(0, 0)).epsilon(1e-5));
}

TEST_CASE("AdaptiveStepSolver - zero-crossing detection") {
    // dx/dt = 1 (linear ramp: x(t) = t - 0.5, crosses zero at t=0.5)
    auto ramp = [](double /*t*/, const ColVec<1>& /*x*/) -> ColVec<1> {
        return ColVec<1>{{1.0}};
    };

    RK45<1>            rk45;
    AdaptiveStepSolver solver(rk45, 0.1, 1e-8);

    // Detect when x crosses zero
    solver.add_zero_crossing([](double /*t*/, const ColVec<1>& x) -> double {
        return x(0, 0);
    });

    ColVec<1> x0{{-0.5}};
    auto      result = solver.solve(ramp, x0, {0.0, 1.0});

    // Should have a point very close to t=0.5, x=0
    bool found_crossing = false;
    for (const auto& [t, x] : result) {
        if (std::abs(x(0, 0)) < 1e-6 && std::abs(t - 0.5) < 0.01) {
            found_crossing = true;
            break;
        }
    }
    CHECK(found_crossing);
}

TEST_CASE("FixedStepSolver - Van der Pol oscillator") {
    // Van der Pol oscillator: x'' - mu*(1-x^2)*x' + x = 0
    // State form: x0' = x1, x1' = mu*(1 - x0^2)*x1 - x0
    constexpr double mu = 1.0;
    auto             vdp = [](double /*t*/, const ColVec<2>& x) -> ColVec<2> {
        return ColVec<2>{{x(1, 0)}, {mu * (1.0 - x(0, 0) * x(0, 0)) * x(1, 0) - x(0, 0)}};
    };

    RK4<2>          rk4;
    FixedStepSolver solver(rk4, 0.001);

    ColVec<2> x0{{2.0}, {0.0}};
    auto      result = solver.solve(vdp, x0, {0.0, 30.0});

    CHECK(result.success);

    // The Van der Pol oscillator converges to a limit cycle with amplitude ~2
    // After 30s the trajectory should be on the limit cycle
    double x_max = 0.0;
    // Check last ~10 seconds worth of data for peak amplitude
    size_t start_idx = result.t.size() * 2 / 3;
    for (size_t i = start_idx; i < result.t.size(); ++i) {
        double val = std::abs(result.x[i](0, 0));
        if (val > x_max) {
            x_max = val;
        }
    }
    // Limit cycle peak amplitude is slightly above 2 for mu=1
    CHECK(x_max == doctest::Approx(2.009).epsilon(0.01));

    // Plot time history and phase portrait as stacked subplots
    auto t = plot::to_double_vector(result.t);
    auto x0_hist = plot::extract_channel(result.x, 0);
    auto x1_hist = plot::extract_channel(result.x, 1);

    using namespace plotlypp;

    auto fig = Figure()
                   // Time history (top)
                   .addTrace(Scatter().x(t).y(x0_hist).mode({Scatter::Mode::Lines}).name("x (position)"))
                   .addTrace(Scatter().x(t).y(x1_hist).mode({Scatter::Mode::Lines}).name("x' (velocity)"))
                   // Phase portrait (bottom)
                   .addTrace(Scatter().x(x0_hist).y(x1_hist).mode({Scatter::Mode::Lines}).name("Trajectory").xaxis("x2").yaxis("y2"))
                   .setLayout(Layout().title([](auto& t) { t.text("Van der Pol Oscillator (mu=1)"); }).grid(Layout::Grid().rows(2).columns(1).pattern(Layout::Grid::Pattern::Independent).roworder(Layout::Grid::Roworder::TopToBottom)).xaxis(Layout::Xaxis().title([](auto& t) { t.text("Time (s)"); })).yaxis(Layout::Yaxis().title([](auto& t) { t.text("State"); })).xaxis(2, Layout::Xaxis().title([](auto& t) { t.text("x"); })).yaxis(2, Layout::Yaxis().title([](auto& t) { t.text("x'"); })));
    fig.writeHtml("tests/build/vanderpol.html");
}
