#include <matplot/matplot.h>

#include <iostream>
#include <vector>

#include "LTI.hpp"

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
    const Matrix B        = Matrix{{0.0}, {1.0}};                   // input affects acceleration
    const Matrix x0       = (Matrix(2, 1) << 1.0, 0.0).finished();  // initial: x1=1, x2=0
    const Matrix u_const  = Matrix{{0.5}};                          // constant forcing

    Solver solver;  // default RK45

    // Time span and evaluation points
    const double        t0 = 0.0;
    const double        tf = 5.0;
    std::vector<double> t_eval;
    const size_t        N = 200;
    for (size_t i = 0; i <= N; ++i) {
        t_eval.push_back(t0 + (tf - t0) * (double(i) / double(N)));
    }

    // Exact solution using matrix exponential (should be available for this system)
    auto res_exact = solver.solveLTI(A, B, x0, u_const, {t0, tf}, t_eval, IntegrationMethod::Exact);

    // Numeric solution (RK45)
    auto res_num = solver.solveLTI(A, B, x0, u_const, {t0, tf}, t_eval, IntegrationMethod::RK45);

    // Convert results to vectors for plotting (two states)
    std::vector<double> t_plot_exact, x1_plot_exact, x2_plot_exact;
    std::vector<double> t_plot_num, x1_plot_num, x2_plot_num;

    for (size_t i = 0; i < res_exact.t.size(); ++i) {
        t_plot_exact.push_back(res_exact.t[i]);
        x1_plot_exact.push_back(res_exact.x[i](0, 0));
        x2_plot_exact.push_back(res_exact.x[i](1, 0));
    }
    for (size_t i = 0; i < res_num.t.size(); ++i) {
        t_plot_num.push_back(res_num.t[i]);
        x1_plot_num.push_back(res_num.x[i](0, 0));
        x2_plot_num.push_back(res_num.x[i](1, 0));
    }

    // Plot two subplots: x1 and x2 (Exact vs RK45)
    auto fig = plt::figure(true);
    fig->size(1200, 800);

    plt::subplot(2, 1, 1);
    auto le1 = plt::plot(t_plot_exact, x1_plot_exact);
    le1->display_name("Exact x1");
    le1->line_width(2);
    plt::hold(plt::on);
    auto ln1 = plt::plot(t_plot_num, x1_plot_num);
    ln1->display_name("RK45 x1");
    ln1->line_style("--");
    ln1->line_width(1);
    plt::hold(plt::off);
    plt::xlabel("Time (s)");
    plt::ylabel("x1");
    plt::title("Second-order LTI: x1 (position)");
    plt::legend()->location(plt::legend::general_alignment::topright);
    plt::grid(plt::on);

    plt::subplot(2, 1, 2);
    auto le2 = plt::plot(t_plot_exact, x2_plot_exact);
    le2->display_name("Exact x2");
    le2->line_width(2);
    plt::hold(plt::on);
    auto ln2 = plt::plot(t_plot_num, x2_plot_num);
    ln2->display_name("RK45 x2");
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
