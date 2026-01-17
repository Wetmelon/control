#include <plotlypp/figure.hpp>
#include <plotlypp/trace.hpp>
#include <plotlypp/traces/scatter.hpp>
#include <vector>

#include "control.hpp"
#include "integrator.hpp"

using namespace plotlypp;

int main() {
    using namespace control;

    // Second-order system (damped oscillator / mass-spring-damper):
    // x1' = x2
    // x2' = -k/m * x1 - c/m * x2 + (1/m) * u
    // Put in state-space form: x' = A x + B u
    const double k_over_m = 4.0;  // stiffness/mass
    const double c_over_m = 0.5;  // damping/mass
    const Matrix A        = Matrix{{0.0, 1.0}, {-k_over_m, -c_over_m}};
    const Matrix B        = Matrix{{0.0}, {1.0}};  // input affects acceleration
    const ColVec x0       = ColVec({1.0, 0.0});    // initial: x1=1, x2=0
    const ColVec u_const  = ColVec({0.5});         // constant forcing

    // Using new solve API

    // Time span and evaluation points
    const double        t0 = 0.0;
    const double        tf = 5.0;
    std::vector<double> t_eval;
    const size_t        N = 200;
    for (size_t i = 0; i <= N; ++i) {
        t_eval.push_back(t0 + (tf - t0) * (double(i) / double(N)));
    }

    // Exact solution using matrix exponential
    auto res_exact = ExactSolver{}.solve(A, B, x0, u_const, {t0, tf}, t_eval);

    // Time-varying input example
    auto dynamics_varying = [&](double t, const ColVec& x) -> ColVec {
        ColVec u = ColVec({std::sin(t)});  // time-varying input
        return A * x + B * u;
    };
    auto res_time_varying = AdaptiveStepSolver<RK45>{}.solve(dynamics_varying, x0, {t0, tf}, t_eval);

    // Convert results to vectors for plotting (two states)
    std::vector<double> t_plot_const, x1_plot_const, x2_plot_const;
    std::vector<double> t_plot_varying, x1_plot_varying, x2_plot_varying;

    for (size_t i = 0; i < res_exact.t.size(); ++i) {
        t_plot_const.push_back(res_exact.t[i]);
        x1_plot_const.push_back(res_exact.x[i](0, 0));
        x2_plot_const.push_back(res_exact.x[i](1, 0));
    }
    for (size_t i = 0; i < res_time_varying.t.size(); ++i) {
        t_plot_varying.push_back(res_time_varying.t[i]);
        x1_plot_varying.push_back(res_time_varying.x[i](0, 0));
        x2_plot_varying.push_back(res_time_varying.x[i](1, 0));
    }

    // Plot two subplots: x1 and x2 (Constant input vs Time-varying)
    Figure fig;

    auto trace_x1_const = Scatter()
                              .x(t_plot_const)
                              .y(x1_plot_const)
                              .name("Constant input")
                              .mode({Scatter::Mode::Lines})
                              .line(Scatter::Line().width(2))
                              .xaxis("x")
                              .yaxis("y");

    auto trace_x1_vary = Scatter()
                             .x(t_plot_varying)
                             .y(x1_plot_varying)
                             .name("Time-varying input")
                             .mode({Scatter::Mode::Lines})
                             .line(Scatter::Line().width(1).dash("dash"))
                             .xaxis("x")
                             .yaxis("y");

    auto trace_x2_const = Scatter()
                              .x(t_plot_const)
                              .y(x2_plot_const)
                              .name("Constant input")
                              .mode({Scatter::Mode::Lines})
                              .line(Scatter::Line().width(2))
                              .xaxis("x2")
                              .yaxis("y2")
                              .showlegend(false);

    auto trace_x2_vary = Scatter()
                             .x(t_plot_varying)
                             .y(x2_plot_varying)
                             .name("Time-varying input")
                             .mode({Scatter::Mode::Lines})
                             .line(Scatter::Line().width(1).dash("dash"))
                             .xaxis("x2")
                             .yaxis("y2")
                             .showlegend(false);

    fig.addTrace(trace_x1_const);
    fig.addTrace(trace_x1_vary);
    fig.addTrace(trace_x2_const);
    fig.addTrace(trace_x2_vary);

    fig.setLayout(Layout()
                      .title([](auto& t) { t.text("Second-order LTI: State Responses"); })
                      .grid(Layout::Grid{}
                                .rows(2)
                                .columns(1)
                                .subplots(std::vector<std::vector<std::string>>{{"xy"}, {"x2y2"}})
                                .roworder(Layout::Grid::Roworder::BottomToTop))
                      .xaxis(1, Layout::Xaxis().title([](auto& t) { t.text("Time (s)"); }).showgrid(true))
                      .yaxis(1, Layout::Yaxis().title([](auto& t) { t.text("x1"); }).showgrid(true))
                      .xaxis(2, Layout::Xaxis().title([](auto& t) { t.text("Time (s)"); }).showgrid(true))
                      .yaxis(2, Layout::Yaxis().title([](auto& t) { t.text("x2"); }).showgrid(true))
                      .width(1200)
                      .height(800)
                      .showlegend(true));

    fig.writeHtml("solve_lti_responses.html");

    return 0;
}
