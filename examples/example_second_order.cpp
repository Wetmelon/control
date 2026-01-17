#include <numbers>
#include <plotlypp/figure.hpp>
#include <plotlypp/trace.hpp>
#include <plotlypp/traces/scatter.hpp>
#include <vector>

#include "control.hpp"

using namespace control;
using namespace plotlypp;

void plotStepResponse(const std::vector<std::vector<double>>& times, const std::vector<std::vector<double>>& responses,
                      const std::vector<std::string>& labels, const std::string& title = "Step Response") {
    using namespace plotlypp;

    Figure fig;

    for (size_t i = 0; i < responses.size(); ++i) {
        auto trace = Scatter()
                         .x(times[i])
                         .y(responses[i])
                         .name(labels[i])
                         .mode({Scatter::Mode::Lines});
        fig.addTrace(trace);
    }

    fig.setLayout(Layout()
                      .title([&](auto& t) { t.text(title); })
                      .xaxis(1, Layout::Xaxis().title([](auto& t) { t.text("Time (s)"); }).showgrid(true))
                      .yaxis(1, Layout::Yaxis().title([](auto& t) { t.text("Output"); }).showgrid(true))
                      .width(1200)
                      .height(800)
                      .showlegend(true));

    fig.writeHtml("second_order_step_response.html");
}

void plotBodePlot(const std::vector<BodeResponse>& responses, const std::vector<std::string>& labels, const std::string& title = "Bode Plot") {
    using namespace plotlypp;

    Figure fig;

    for (size_t i = 0; i < responses.size(); ++i) {
        // Magnitude plot
        auto mag_trace = Scatter()
                             .x(responses[i].freq)
                             .y(responses[i].magnitude)
                             .name(labels[i])
                             .mode({Scatter::Mode::Lines})
                             .xaxis("x")
                             .yaxis("y");
        fig.addTrace(mag_trace);

        // Phase plot
        auto phase_trace = Scatter()
                               .x(responses[i].freq)
                               .y(responses[i].phase)
                               .name(labels[i])
                               .mode({Scatter::Mode::Lines})
                               .xaxis("x2")
                               .yaxis("y2")
                               .showlegend(false);  // Only show legend on first subplot
        fig.addTrace(phase_trace);
    }

    fig.setLayout(Layout()
                      .title([&](auto& t) { t.text(title); })
                      .grid(Layout::Grid{}
                                .rows(2)
                                .columns(1)
                                .subplots(std::vector<std::vector<std::string>>{{"xy"}, {"x2y2"}})
                                .roworder(Layout::Grid::Roworder::BottomToTop))
                      .xaxis(1, Layout::Xaxis().title([](auto& t) { t.text("Frequency (Hz)"); }).type(Layout::Xaxis::Type::Log).showgrid(true))
                      .yaxis(1, Layout::Yaxis().title([](auto& t) { t.text("Magnitude (dB)"); }).showgrid(true))
                      .xaxis(2, Layout::Xaxis().type(Layout::Xaxis::Type::Log).showgrid(true))
                      .yaxis(2, Layout::Yaxis().title([](auto& t) { t.text("Phase (deg)"); }).showgrid(true))
                      .width(1200)
                      .height(800)
                      .showlegend(true));

    fig.writeHtml("second_order_bode_plot.html");
}

int main() {
    using namespace control;
    using namespace std::literals;

    // Parameters for 2nd order system
    const double              omega_n = 6.0 * 2.0 * std::numbers::pi;  // Natural frequency (rad/s)
    const std::vector<double> zetas   = {0.3, 0.5, std::numbers::sqrt2 / 2.0, 1.0, 2.0};

    // Find minimum zeta for settling time calculation
    double min_zeta = *std::min_element(zetas.begin(), zetas.end());

    // Simulation parameters
    const double t_start = 0.0;
    const double t_end   = 1.5 * 5.0 / (min_zeta * omega_n);  // Settling time for smallest zeta

    // Initialize time vector (will be set from generateStepResponse)
    std::vector<double> time;

    // Simulate for each damping ratio
    std::vector<std::vector<double>> times;
    std::vector<std::vector<double>> stepResponses;
    std::vector<BodeResponse>        freqResponses;

    std::vector<std::string> labels;
    for (double zeta : zetas) {
        // State space matrices for 2nd order system: ẋ = Ax + Bu, y = Cx + Du
        // A = [0, 1; -ω_n², -2ζω_n]
        // B = [0; ω_n²]
        // C = [1, 0]
        // D = [0]
        const auto A = Matrix{{0.0, 1.0}, {-omega_n * omega_n, -2.0 * zeta * omega_n}};
        const auto B = Matrix{{0.0}, {omega_n * omega_n}};
        const auto C = Matrix{{1.0, 0.0}};
        const auto D = Matrix{{0.0}};

        // Create continuous-time system
        auto sys = StateSpace{A, B, C, D};

        // Generate step response
        auto stepResp = sys.step(t_start, t_end);

        // Store time vector from the first response
        times.push_back(stepResp.time);
        // Extract scalar output from Matrix (SISO system)
        std::vector<double> scalarOutput;
        scalarOutput.reserve(stepResp.output.size());
        for (const auto& y : stepResp.output) {
            scalarOutput.push_back(y(0, 0));
        }
        stepResponses.push_back(scalarOutput);
        labels.push_back(fmt::format("ζ = {:.1f}", zeta));
        freqResponses.push_back(sys.bode());
    }

    // Plot all step responses and Bode plots
    plotStepResponse(times, stepResponses, labels, "Step Response for Different Damping Ratios");
    plotBodePlot(freqResponses, labels, "Bode Plot for Different Damping Ratios");

    return 0;
}