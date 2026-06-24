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

#include <cstddef>
#include <plotlypp/figure.hpp>
#include <plotlypp/layout/layout.hpp>
#include <plotlypp/traces/scatter.hpp>
#include <string>
#include <vector>

#include "wet/analysis/analysis.hpp"
#include "wet/backend.hpp"
#include "wet/math/complex.hpp"
#include "wet/matrix/colvec.hpp"
#include "wet/simulation/simulate.hpp"

namespace wet {
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
    const sim::SimulationResult<NX, NU, NY, T>& sim,
    const std::string&                          title = "Simulation"
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
        fig.addTrace(wet::move(trace));
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
        fig.addTrace(wet::move(trace));
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
        fig.addTrace(wet::move(trace));
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

    fig.setLayout(wet::move(layout));
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
        .addTrace(wet::move(mag_trace))
        .addTrace(wet::move(phase_trace))
        .setLayout(wet::move(layout));
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

    return Figure().addTrace(wet::move(trace)).setLayout(wet::move(layout));
}

/**
 * @brief Plot step response data
 *
 * @param time_values Pair of {time_vector, response_vector} from step_response()
 * @param title Plot title
 */
template<typename T>
plotlypp::Figure plot_step(
    const wet::pair<std::vector<T>, std::vector<T>>& time_values,
    const std::string&                               title = "Step Response"
) {
    return plot_line(time_values.first, time_values.second, title);
}

// ============================================================================
// MATLAB-style plot helpers (thin plotlypp wrappers over analysis:: results)
// ============================================================================

namespace detail {

/**
 * @brief Build a time-response figure with one line per (output, input) pair
 *
 * Shared implementation for stepplot() and impulseplot(): a `TimeResponse`
 * stores `y[k](i, j)` (output i from a canonical input on channel j), so this
 * emits NY·NU traces. Single-input systems are labelled `y<i>`; multi-input
 * systems `y<i> <- u<j>`.
 *
 * @tparam NY Number of outputs
 * @tparam NU Number of inputs
 * @param resp  Per-channel time response
 * @param title Figure title
 * @return plotlypp::Figure ready for .show() or .writeHtml()
 */
template<size_t NY, size_t NU, typename T>
plotlypp::Figure time_response_figure(const analysis::TimeResponse<NY, NU, T>& resp, const std::string& title) {
    using namespace plotlypp;

    const auto t = to_double_vector(resp.t);
    Figure     fig;
    for (size_t j = 0; j < NU; ++j) {
        for (size_t i = 0; i < NY; ++i) {
            std::vector<double> y;
            y.reserve(resp.y.size());
            for (const auto& yk : resp.y) {
                y.push_back(static_cast<double>(yk(i, j)));
            }
            const std::string name = (NU > 1) ? ("y" + std::to_string(i) + " <- u" + std::to_string(j)) : ("y" + std::to_string(i));
            fig.addTrace(Scatter().x(t).y(y).mode({Scatter::Mode::Lines}).name(name));
        }
    }
    fig.setLayout(
        Layout()
            .title([&](auto& tt) { tt.text(title); })
            .xaxis(Layout::Xaxis().title([](auto& tt) { tt.text("Time (s)"); }))
            .yaxis(Layout::Yaxis().title([](auto& tt) { tt.text("Amplitude"); }))
    );
    return fig;
}

/**
 * @brief Build a markers-only scatter of complex points (real vs imaginary)
 *
 * Used by pzplot() to draw poles and zeros on the complex plane.
 *
 * @param pts  Complex values to plot
 * @param name Trace name (legend label)
 * @param sym  Marker symbol (e.g. X for poles, CircleOpen for zeros)
 * @return plotlypp::Scatter trace
 */
template<typename T>
plotlypp::Scatter complex_scatter(const std::vector<wet::complex<T>>& pts, const std::string& name, plotlypp::Scatter::Marker::Symbol sym) {
    using namespace plotlypp;

    std::vector<double> re, im;
    re.reserve(pts.size());
    im.reserve(pts.size());
    for (const auto& p : pts) {
        re.push_back(static_cast<double>(p.real()));
        im.push_back(static_cast<double>(p.imag()));
    }
    return Scatter()
        .x(re)
        .y(im)
        .mode({Scatter::Mode::Markers})
        .name(name)
        .marker([sym](auto& m) { m.symbol(sym).size(10.0); });
}

} // namespace detail

/**
 * @brief Plot a step response, one trace per input/output pair
 *
 * MATLAB equivalent: `stepplot(sys)`. Pair with `analysis::step`.
 *
 * @tparam NY Number of outputs
 * @tparam NU Number of inputs
 * @param resp  Step response from analysis::step()
 * @param title Figure title
 * @return plotlypp::Figure
 */
template<size_t NY, size_t NU, typename T>
plotlypp::Figure stepplot(const analysis::TimeResponse<NY, NU, T>& resp, const std::string& title = "Step Response") {
    return detail::time_response_figure(resp, title);
}

/**
 * @brief Plot an impulse response, one trace per input/output pair
 *
 * MATLAB equivalent: `impulseplot(sys)`. Pair with `analysis::impulse`.
 *
 * @tparam NY Number of outputs
 * @tparam NU Number of inputs
 * @param resp  Impulse response from analysis::impulse()
 * @param title Figure title
 * @return plotlypp::Figure
 */
template<size_t NY, size_t NU, typename T>
plotlypp::Figure impulseplot(const analysis::TimeResponse<NY, NU, T>& resp, const std::string& title = "Impulse Response") {
    return detail::time_response_figure(resp, title);
}

/**
 * @brief Plot a forced (lsim) simulation, one trace per output
 *
 * MATLAB equivalent: `lsimplot(sys, u, t)`. Pair with `analysis::lsim`.
 *
 * @tparam NX Number of states
 * @tparam NY Number of outputs
 * @param resp  Forced response from analysis::lsim()
 * @param title Figure title
 * @return plotlypp::Figure
 */
template<size_t NX, size_t NY, typename T>
plotlypp::Figure lsimplot(const analysis::LsimResult<NX, NY, T>& resp, const std::string& title = "Linear Simulation") {
    using namespace plotlypp;

    const auto t = to_double_vector(resp.t);
    Figure     fig;
    for (size_t i = 0; i < NY; ++i) {
        fig.addTrace(
            Scatter()
                .x(t)
                .y(extract_channel(resp.y, i))
                .mode({Scatter::Mode::Lines})
                .name("y" + std::to_string(i))
        );
    }
    fig.setLayout(
        Layout()
            .title([&](auto& tt) { tt.text(title); })
            .xaxis(Layout::Xaxis().title([](auto& tt) { tt.text("Time (s)"); }))
            .yaxis(Layout::Yaxis().title([](auto& tt) { tt.text("Amplitude"); }))
    );
    return fig;
}

/**
 * @brief Plot magnitude and phase Bode subplots
 *
 * MATLAB equivalent: `bodeplot(sys)`. Thin alias of plot_bode().
 *
 * @param bode  Frequency response from analysis::bode()
 * @param title Figure title
 * @return plotlypp::Figure
 */
template<typename T>
plotlypp::Figure bodeplot(const analysis::BodeResult<T>& bode, const std::string& title = "Bode Plot") {
    return plot_bode(bode, title);
}

/**
 * @brief Plot a magnitude-only Bode diagram (log frequency, dB magnitude)
 *
 * MATLAB equivalent: `bodemag(sys)`.
 *
 * @param bode  Frequency response from analysis::bode()
 * @param title Figure title
 * @return plotlypp::Figure
 */
template<typename T>
plotlypp::Figure bodemag(const analysis::BodeResult<T>& bode, const std::string& title = "Bode Magnitude") {
    using namespace plotlypp;

    std::vector<double> omega, mag_db;
    omega.reserve(bode.points.size());
    mag_db.reserve(bode.points.size());
    for (const auto& pt : bode.points) {
        omega.push_back(static_cast<double>(pt.omega));
        mag_db.push_back(static_cast<double>(pt.magnitude_db));
    }
    return Figure()
        .addTrace(Scatter().x(omega).y(mag_db).mode({Scatter::Mode::Lines}).name("Magnitude"))
        .setLayout(Layout().title([&](auto& tt) { tt.text(title); }).xaxis(Layout::Xaxis().title([](auto& tt) { tt.text("Frequency (rad/s)"); }).type(Layout::Xaxis::Type::Log)).yaxis(Layout::Yaxis().title([](auto& tt) { tt.text("Magnitude (dB)"); })));
}

/**
 * @brief Plot a Nyquist locus with the -1 critical point marked
 *
 * MATLAB equivalent: `nyquistplot(sys)`. Pair with `analysis::nyquist`.
 *
 * @param nyq   Nyquist response from analysis::nyquist()
 * @param title Figure title
 * @return plotlypp::Figure
 */
template<typename T>
plotlypp::Figure nyquistplot(const analysis::NyquistResult<T>& nyq, const std::string& title = "Nyquist Plot") {
    using namespace plotlypp;

    std::vector<double> re, im;
    re.reserve(nyq.points.size());
    im.reserve(nyq.points.size());
    for (const auto& p : nyq.points) {
        re.push_back(static_cast<double>(p.real));
        im.push_back(static_cast<double>(p.imag));
    }
    Figure fig;
    fig.addTrace(Scatter().x(re).y(im).mode({Scatter::Mode::Lines}).name("L(jw)"));
    fig.addTrace(Scatter().x(std::vector<double>{-1.0}).y(std::vector<double>{0.0}).mode({Scatter::Mode::Markers}).name("-1").marker([](auto& m) { m.symbol(Scatter::Marker::Symbol::X).size(10.0); }));
    fig.setLayout(Layout().title([&](auto& tt) { tt.text(title); }).xaxis(Layout::Xaxis().title([](auto& tt) { tt.text("Real"); })).yaxis(Layout::Yaxis().title([](auto& tt) { tt.text("Imag"); })));
    return fig;
}

/**
 * @brief Plot a pole-zero map on the complex plane (poles as ×, zeros as ○)
 *
 * MATLAB equivalent: `pzplot(sys)`. Pair with `analysis::pzmap`.
 *
 * @param pz    Pole-zero data from analysis::pzmap()
 * @param title Figure title
 * @return plotlypp::Figure
 */
template<typename T>
plotlypp::Figure pzplot(const analysis::PoleZeroMap<T>& pz, const std::string& title = "Pole-Zero Map") {
    using namespace plotlypp;

    Figure fig;
    fig.addTrace(detail::complex_scatter(pz.poles, "poles", Scatter::Marker::Symbol::X));
    fig.addTrace(detail::complex_scatter(pz.zeros, "zeros", Scatter::Marker::Symbol::CircleOpen));
    fig.setLayout(Layout().title([&](auto& tt) { tt.text(title); }).xaxis(Layout::Xaxis().title([](auto& tt) { tt.text("Real"); })).yaxis(Layout::Yaxis().title([](auto& tt) { tt.text("Imag"); })));
    return fig;
}

} // namespace plot
} // namespace wet
