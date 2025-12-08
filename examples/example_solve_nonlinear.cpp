#include <vector>

#include "control.hpp"
#include "matplot/matplot.h"

namespace plt = matplot;

int main() {
    using namespace control;

    Solver solver;  // default RK45

    // Van der Pol oscillator (nonlinear ODE)
    // x1' = x2
    // x2' = mu * (1 - x1^2) * x2 - x1
    const double mu = 1.0;

    Matrix x0 = Matrix::Zero(2, 1);
    x0(0, 0)  = 2.0;  // initial x1
    x0(1, 0)  = 0.0;  // initial x2

    // Define the ODE as a callable: f(t, x) -> dx/dt
    auto fun = [mu](double /*t*/, const Matrix& x) -> Matrix {
        Matrix dx = Matrix::Zero(2, 1);
        dx(0, 0)  = x(1, 0);
        dx(1, 0)  = mu * (1.0 - x(0, 0) * x(0, 0)) * x(1, 0) - x(0, 0);
        return dx;
    };

    // Integrate from 0 to 20 seconds
    const double        t0 = 0.0;
    const double        tf = 20.0;
    std::vector<double> t_eval;
    const size_t        N = 2000;
    for (size_t i = 0; i <= N; ++i) {
        t_eval.push_back(t0 + (tf - t0) * (double(i) / double(N)));
    }

    auto res = solver.solve(fun, x0, {t0, tf}, t_eval, IntegrationMethod::RK45, 1e-6, 1e-3, 0.0, 0.0);

    std::vector<double> t_plot, x1_plot, x2_plot;
    t_plot.reserve(res.t.size());
    x1_plot.reserve(res.t.size());
    x2_plot.reserve(res.t.size());
    for (size_t i = 0; i < res.t.size(); ++i) {
        t_plot.push_back(res.t[i]);
        x1_plot.push_back(res.x[i](0, 0));
        x2_plot.push_back(res.x[i](1, 0));
    }

    // Plot states over time and phase portrait
    auto fig = plt::figure(true);
    fig->size(1200, 800);
    // Compose a short info string for the figure
    std::string info = "Van der Pol oscillator (mu=" + std::to_string(mu) +
                       ") — "
                       "Solver: RK45, abs tol=1e-6, rel tol=1e-3, points=" +
                       std::to_string(N);

    // Use a suptitle (supported by matplot++) and individual subplot titles/labels
    plt::sgtitle(info);

    plt::subplot(3, 1, 1);
    auto l1 = plt::plot(t_plot, x1_plot);
    l1->display_name("x1");
    l1->line_width(1.5);
    plt::title("x1 over time");
    plt::xlabel("Time (s)");
    plt::ylabel("x1");
    plt::legend()->location(plt::legend::general_alignment::topright);
    plt::grid(plt::on);

    plt::subplot(3, 1, 2);
    auto l2 = plt::plot(t_plot, x2_plot);
    l2->display_name("x2");
    l2->line_width(1.5);
    l2->line_style("--");
    plt::title("x2 over time");
    plt::xlabel("Time (s)");
    plt::ylabel("x2");
    plt::legend()->location(plt::legend::general_alignment::topright);
    plt::grid(plt::on);

    // Phase portrait (x1 vs x2)
    plt::subplot(3, 1, 3);
    auto lp = plt::plot(x1_plot, x2_plot);
    lp->display_name("phase");
    lp->line_width(1.0);
    plt::title("Phase portrait (x1 vs x2)");
    plt::xlabel("x1");
    plt::ylabel("x2");
    plt::grid(plt::on);

    plt::show();

    return 0;
}
