#include <vector>

#include "control.hpp"
#include "integrator.hpp"
#include "matplot/matplot.h"

namespace plt = matplot;

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
    auto fig = plt::figure(true);
    fig->size(1200, 800);

    // Figure-level title
    plt::sgtitle("Second-order LTI: State Responses");
    plt::subplot(2, 1, 0);
    auto le1 = plt::plot(t_plot_const, x1_plot_const);
    le1->display_name("Constant input");
    le1->line_width(2);
    plt::hold(plt::on);
    auto ln1 = plt::plot(t_plot_varying, x1_plot_varying);
    ln1->display_name("Time-varying input");
    ln1->line_style("--");
    ln1->line_width(1);
    plt::hold(plt::off);
    plt::xlabel("Time (s)");
    plt::ylabel("x1");
    plt::title("Second-order LTI: x1 (position)");
    plt::legend()->location(plt::legend::general_alignment::topright);
    plt::grid(plt::on);

    plt::subplot(2, 1, 1);
    auto le2 = plt::plot(t_plot_const, x2_plot_const);
    le2->display_name("Constant input");
    le2->line_width(2);
    plt::hold(plt::on);
    auto ln2 = plt::plot(t_plot_varying, x2_plot_varying);
    ln2->display_name("Time-varying input");
    ln2->line_style("--");
    ln2->line_width(1);
    plt::hold(plt::off);
    plt::xlabel("Time (s)");
    plt::ylabel("x2");
    plt::title("Second-order LTI: x2 (velocity)");
    plt::legend()->location(plt::legend::general_alignment::topright);
    plt::grid(plt::on);

    plt::show();

    return 0;
}
