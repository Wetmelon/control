#include <plotlypp/figure.hpp>
#include <plotlypp/trace.hpp>
#include <plotlypp/traces/scatter.hpp>
#include <vector>

#include "control.hpp"
#include "integrator.hpp"
#include "solver.hpp"

int main() {
    using namespace control;
    using namespace plotlypp;

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

    auto res = AdaptiveStepSolver<RK45>{}.solve(fun, x0, {t0, tf}, t_eval);

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
    Figure fig;

    // Compose a short info string for the figure
    std::string info = "Van der Pol oscillator (mu=" + std::to_string(mu) +
                       ") — "
                       "Solver: RK45, abs tol=1e-6, rel tol=1e-3, points=" +
                       std::to_string(N);

    // x1 over time
    auto trace_x1 = Scatter()
                        .x(t_plot)
                        .y(x1_plot)
                        .mode({Scatter::Mode::Lines})
                        .name("x1")
                        .line(Scatter::Line().width(1.5))
                        .xaxis("x")
                        .yaxis("y");

    // x2 over time
    auto trace_x2 = Scatter()
                        .x(t_plot)
                        .y(x2_plot)
                        .mode({Scatter::Mode::Lines})
                        .name("x2")
                        .line(Scatter::Line().width(1.5).dash("dash"))
                        .xaxis("x2")
                        .yaxis("y2");

    // Phase portrait (x1 vs x2)
    auto trace_phase = Scatter()
                           .x(x1_plot)
                           .y(x2_plot)
                           .mode({Scatter::Mode::Lines})
                           .name("phase")
                           .line(Scatter::Line().width(1.0))
                           .xaxis("x3")
                           .yaxis("y3");

    auto layout = Layout()
                      .title([&](auto& t) { t.text(info); })
                      .height(800)
                      .width(1200)
                      .xaxis(1, Layout::Xaxis().title([](auto& t) { t.text("Time (s)"); }).showgrid(true))
                      .yaxis(1, Layout::Yaxis().title([](auto& t) { t.text("x1"); }).showgrid(true))
                      .xaxis(2, Layout::Xaxis().title([](auto& t) { t.text("Time (s)"); }).showgrid(true))
                      .yaxis(2, Layout::Yaxis().title([](auto& t) { t.text("x2"); }).showgrid(true))
                      .xaxis(3, Layout::Xaxis().title([](auto& t) { t.text("x1"); }).showgrid(true))
                      .yaxis(3, Layout::Yaxis().title([](auto& t) { t.text("x2"); }).showgrid(true))
                      .grid(Layout::Grid{}
                                .rows(3)
                                .columns(1)
                                .subplots(std::vector<std::vector<std::string>>{{"xy"}, {"x2y2"}, {"x3y3"}})
                                .roworder(Layout::Grid::Roworder::BottomToTop));

    fig.addTraces(std::vector<Trace>{trace_x1, trace_x2, trace_phase});
    fig.setLayout(layout);

    fig.writeHtml("van_der_pol_oscillator.html");

    return 0;
}
