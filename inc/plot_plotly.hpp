#pragma once

/**
 * @defgroup plot_plotly Plotly++ Visualization
 * @brief Interactive HTML plotting for control system data via plotlypp
 *
 * Separate header to avoid mandatory plotlypp dependency in the core library.
 * Requires plotlypp and nlohmann-json to be available on the include path.
 *
 * Usage:
 * @code
 *   auto sim = simulate(...);
 *   auto fig = plot::plot_simulation(sim, "Pendulum Response");
 *   fig.show();  // Opens in browser
 * @endcode
 */

#include <plotlypp/figure.hpp>
#include <plotlypp/traces/scatter.hpp>
#include <string>
#include <vector>

#include "analysis.hpp"
#include "simulate.hpp"

namespace wetmelon::control {
namespace plot {

/**
 * @brief Convert a ColVec<N,T> to std::vector<double> for plotlypp
 */
template<size_t N, typename T>
std::vector<double> to_std_vector(const ColVec<N, T>& v) {
    std::vector<double> out(N);
    for (size_t i = 0; i < N; ++i) {
        out[i] = static_cast<double>(v(i, 0));
    }
    return out;
}

/**
 * @brief Extract the i-th element from each vector entry into a std::vector<double>
 */
template<size_t N, typename T>
std::vector<double> extract_channel(const std::vector<ColVec<N, T>>& history, size_t channel) {
    std::vector<double> out;
    out.reserve(history.size());
    for (const auto& v : history) {
        out.push_back(static_cast<double>(v(channel, 0)));
    }
    return out;
}

/**
 * @brief Convert std::vector<T> to std::vector<double>
 */
template<typename T>
std::vector<double> to_double_vector(const std::vector<T>& v) {
    std::vector<double> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        out[i] = static_cast<double>(v[i]);
    }
    return out;
}

/**
 * @brief Plot simulation results with subplots for states, outputs, and inputs
 *
 * Creates a 3-row subplot: states on top, outputs in middle, control inputs on bottom.
 *
 * @param sim   SimulationResult from simulate() or simulate_state_feedback()
 * @param title Plot title
 * @return plotlypp::Figure ready for .show() or .writeHtml()
 */
template<size_t NX, size_t NU, size_t NY, typename T>
plotlypp::Figure plot_simulation(
    const SimulationResult<NX, NU, NY, T>& sim,
    const std::string&                     title = "Simulation"
) {
    using namespace plotlypp;

    auto   t = to_double_vector(sim.t);
    Figure fig;

    // States subplot (row 1)
    for (size_t i = 0; i < NX; ++i) {
        auto trace = Scatter()
                         .x(t)
                         .y(extract_channel(sim.x, i))
                         .mode({Scatter::Mode::Lines})
                         .name("x" + std::to_string(i));
        fig.addTrace(std::move(trace));
    }

    // Outputs subplot (row 2)
    for (size_t i = 0; i < NY; ++i) {
        auto trace = Scatter()
                         .x(t)
                         .y(extract_channel(sim.y, i))
                         .mode({Scatter::Mode::Lines})
                         .name("y" + std::to_string(i))
                         .xaxis("x2")
                         .yaxis("y2");
        fig.addTrace(std::move(trace));
    }

    // Inputs subplot (row 3)
    for (size_t i = 0; i < NU; ++i) {
        auto trace = Scatter()
                         .x(t)
                         .y(extract_channel(sim.u, i))
                         .mode({Scatter::Mode::Lines})
                         .name("u" + std::to_string(i))
                         .xaxis("x3")
                         .yaxis("y3");
        fig.addTrace(std::move(trace));
    }

    auto layout = Layout()
                      .title([&](auto& t) { t.text(title); })
                      .grid(Layout::Grid().rows(3).columns(1).pattern(Layout::Grid::Pattern::Independent).roworder(Layout::Grid::Roworder::TopToBottom))
                      .xaxis(Layout::Xaxis().title([](auto& t) { t.text("Time (s)"); }))
                      .yaxis(Layout::Yaxis().title([](auto& t) { t.text("States"); }))
                      .xaxis(2, Layout::Xaxis().title([](auto& t) { t.text("Time (s)"); }))
                      .yaxis(2, Layout::Yaxis().title([](auto& t) { t.text("Outputs"); }))
                      .xaxis(3, Layout::Xaxis().title([](auto& t) { t.text("Time (s)"); }))
                      .yaxis(3, Layout::Yaxis().title([](auto& t) { t.text("Inputs"); }));

    fig.setLayout(std::move(layout));
    return fig;
}

/**
 * @brief Plot Bode magnitude and phase as subplots
 *
 * @param bode  BodeResult from analysis::bode()
 * @param title Plot title
 * @return plotlypp::Figure
 */
template<typename T>
plotlypp::Figure plot_bode(
    const analysis::BodeResult<T>& bode,
    const std::string&             title = "Bode Plot"
) {
    using namespace plotlypp;

    std::vector<double> omega, mag_db, phase_deg;
    omega.reserve(bode.points.size());
    mag_db.reserve(bode.points.size());
    phase_deg.reserve(bode.points.size());

    for (const auto& pt : bode.points) {
        omega.push_back(static_cast<double>(pt.omega));
        mag_db.push_back(static_cast<double>(pt.magnitude_db));
        phase_deg.push_back(static_cast<double>(pt.phase_deg));
    }

    auto mag_trace = Scatter()
                         .x(omega)
                         .y(mag_db)
                         .mode({Scatter::Mode::Lines})
                         .name("Magnitude");

    auto phase_trace = Scatter()
                           .x(omega)
                           .y(phase_deg)
                           .mode({Scatter::Mode::Lines})
                           .name("Phase")
                           .xaxis("x2")
                           .yaxis("y2");

    auto layout = Layout()
                      .title([&](auto& t) { t.text(title); })
                      .grid(Layout::Grid().rows(2).columns(1).pattern(Layout::Grid::Pattern::Independent).roworder(Layout::Grid::Roworder::TopToBottom))
                      .xaxis(Layout::Xaxis().title([](auto& t) { t.text("Frequency (rad/s)"); }).type(Layout::Xaxis::Type::Log))
                      .yaxis(Layout::Yaxis().title([](auto& t) { t.text("Magnitude (dB)"); }))
                      .xaxis(2, Layout::Xaxis().title([](auto& t) { t.text("Frequency (rad/s)"); }).type(Layout::Xaxis::Type::Log))
                      .yaxis(2, Layout::Yaxis().title([](auto& t) { t.text("Phase (deg)"); }));

    return Figure()
        .addTrace(std::move(mag_trace))
        .addTrace(std::move(phase_trace))
        .setLayout(std::move(layout));
}

/**
 * @brief Simple line plot of time vs value
 *
 * @param time   Time vector
 * @param values Value vector
 * @param title  Plot title
 * @return plotlypp::Figure
 */
template<typename T>
plotlypp::Figure plot_line(
    const std::vector<T>& time,
    const std::vector<T>& values,
    const std::string&    title = "Plot"
) {
    using namespace plotlypp;

    auto trace = Scatter()
                     .x(to_double_vector(time))
                     .y(to_double_vector(values))
                     .mode({Scatter::Mode::Lines});

    auto layout = Layout()
                      .title([&](auto& t) { t.text(title); })
                      .xaxis(Layout::Xaxis().title([](auto& t) { t.text("Time (s)"); }))
                      .yaxis(Layout::Yaxis().title([](auto& t) { t.text("Value"); }));

    return Figure().addTrace(std::move(trace)).setLayout(std::move(layout));
}

/**
 * @brief Plot step response data
 *
 * @param time_values Pair of {time_vector, response_vector} from step_response()
 * @param title Plot title
 */
template<typename T>
plotlypp::Figure plot_step(
    const std::pair<std::vector<T>, std::vector<T>>& time_values,
    const std::string&                               title = "Step Response"
) {
    return plot_line(time_values.first, time_values.second, title);
}

} // namespace plot
} // namespace wetmelon::control
