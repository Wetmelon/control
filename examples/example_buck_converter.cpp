#include <fmt/core.h>

#include <plotlypp/figure.hpp>
#include <plotlypp/trace.hpp>
#include <plotlypp/traces/scatter.hpp>
#include <string>
#include <vector>

#include "control.hpp"

int main() {
    using namespace control;
    using namespace plotlypp;

    fmt::print("=== DC-DC Buck Converter Example ===\n");
    fmt::print("Average model with PI voltage control\n\n");

    // Converter parameters
    const double Vin  = 24.0;    // Input voltage [V]
    const double Vref = 12.0;    // Reference output voltage [V]
    const double L    = 100e-6;  // Inductor [H]
    const double C    = 470e-6;  // Capacitor [F]
    // Use a light load (larger R) to make the LC dynamics more resonant/under-damped
    const double R  = 500.0;  // Load resistance [Ohm]
    const double D0 = 0.5;    // Nominal duty cycle

    fmt::print("Converter parameters:\n");
    fmt::print("  Input voltage Vin = {:.1f} V\n", Vin);
    fmt::print("  Reference voltage Vref = {:.1f} V\n", Vref);
    fmt::print("  Inductor L = {:.1f} μH\n", L * 1e6);
    fmt::print("  Capacitor C = {:.1f} μF\n", C * 1e6);
    fmt::print("  Load resistance R = {:.1f} Ohm\n", R);
    fmt::print("  Nominal duty cycle D₀ = {:.2f}\n\n", D0);

    // Small-signal averaged model around operating point
    // States: [i_L, v_C]^T (inductor current, capacitor voltage)
    // Input: d (duty cycle perturbation)
    // Output: v_C (capacitor voltage)

    const double I0  = Vref / R;      // Nominal inductor current
    const double A11 = -R / L;        // di_L/dt coefficient
    const double A12 = -1 / L;        // di_L/dv_C coefficient
    const double A21 = 1 / C;         // dv_C/di_L coefficient
    const double A22 = -1 / (R * C);  // dv_C/dt coefficient
    const double B1  = Vin / L;       // di_L/dd coefficient
    const double B2  = I0 / C;        // dv_C/dd coefficient

    Matrix A_sys = Matrix{{A11, A12}, {A21, A22}};
    Matrix B_sys = Matrix{{B1}, {B2}};
    Matrix C_sys = Matrix{{0, 1}};  // Output capacitor voltage
    Matrix D_sys = Matrix::Zero(1, 1);

    StateSpace buck_model(A_sys, B_sys, C_sys, D_sys);

    // Check stability
    bool stable = is_stable(buck_model);
    fmt::print("Small-signal model is {}stable\n\n", stable ? "" : "un");

    // Transfer function from duty cycle to output voltage
    TransferFunction Gvd = tf(buck_model);

    // Design PI controller for voltage regulation
    // C(s) = Kp + Ki/s

    // Controller parameters (designed for crossover frequency ~1kHz, phase margin ~60°)
    const double Kp = 0.1;
    const double Ki = 50.0;

    // PI controller in state-space form
    StateSpace PI_controller{
        Matrix::Zero(1, 1),           // A
        Matrix::Constant(1, 1, 1.0),  // B
        Matrix::Constant(1, 1, Ki),   // C
        Matrix::Constant(1, 1, Kp)    // D
    };

    fmt::print("PI controller design:\n");
    fmt::print("  Kp = {:.3f}\n", Kp);
    fmt::print("  Ki = {:.1f}\n", Ki);
    fmt::print("  C(s) = {:.3f} + {:.1f}/s\n\n", Kp, Ki);

    // Closed-loop system: voltage feedback with PI controller in forward path
    StateSpace cl_system = feedback(PI_controller * buck_model);

    // Check closed-loop stability
    bool cl_stable = is_stable(cl_system);
    fmt::print("Closed-loop system is {}stable\n\n", cl_stable ? "" : "un");

    // Frequency response analysis
    auto bode_ol = bode(buck_model, 1.0, 1e5, 200);  // Open-loop
    auto bode_cl = bode(cl_system, 1.0, 1e5, 200);   // Closed-loop

    // Plot Bode plots: overlay open-loop and closed-loop on the same axes
    auto trace_ol_mag = Scatter()
                            .x(bode_ol.freq)
                            .y(bode_ol.magnitude)
                            .mode({Scatter::Mode::Lines})
                            .name("Open-Loop")
                            .line(Scatter::Line().width(2).color("blue"));

    auto trace_cl_mag = Scatter()
                            .x(bode_cl.freq)
                            .y(bode_cl.magnitude)
                            .mode({Scatter::Mode::Lines})
                            .name("Closed-Loop")
                            .line(Scatter::Line().width(2).color("red").dash("dash"));

    auto trace_ol_ph = Scatter()
                           .x(bode_ol.freq)
                           .y(bode_ol.phase)
                           .mode({Scatter::Mode::Lines})
                           .name("Open-Loop")
                           .line(Scatter::Line().width(2).color("blue"))
                           .yaxis("y2");

    auto trace_cl_ph = Scatter()
                           .x(bode_cl.freq)
                           .y(bode_cl.phase)
                           .mode({Scatter::Mode::Lines})
                           .name("Closed-Loop")
                           .line(Scatter::Line().width(2).color("red").dash("dash"))
                           .yaxis("y2");

    // Step response analysis will be performed below (longer simulation)
    fmt::print("Step response analysis (reference change from 12V to 15V):\n");
    const double tEnd         = 1.0;                                                       // simulate longer to observe convergence
    auto         step_resp_ol = step(buck_model, 0.0, tEnd, Matrix::Constant(1, 1, 3.0));  // Open-loop response to same step
    auto         step_resp_cl = step(cl_system, 0.0, tEnd, Matrix::Constant(1, 1, 3.0));   // Closed-loop response

    auto step_info = stepinfo(cl_system);
    fmt::print("Rise time: {:.6f} s\n", step_info.riseTime[0]);
    fmt::print("Settling time: {:.6f} s\n", step_info.settlingTime[0]);
    fmt::print("Overshoot: {:.3f}%\n", step_info.overshoot[0] * 100);
    fmt::print("Steady-state value: {:.3f} V\n", step_info.peak[0]);

    // Gain and phase margins
    auto margins = margin(buck_model);
    fmt::print("\nOpen-loop stability margins:\n");
    fmt::print("Gain margin: {:.2f} dB at {:.1f} rad/s\n", margins.gainMargin, margins.gainCrossover);
    fmt::print("Phase margin: {:.1f} deg at {:.1f} rad/s\n", margins.phaseMargin, margins.phaseCrossover);

    // Compute DC gain of the closed-loop system and expected steady-state
    // (SISO assumption)
    double step_mag = 3.0;
    double dc_gain  = 0.0;
    try {
        auto A_cl = cl_system.A;
        auto B_cl = cl_system.B;
        auto C_cl = cl_system.C;
        auto D_cl = cl_system.D;
        // DC gain = D + C * (-A)^{-1} * B
        Matrix inv_negA = (-A_cl).inverse();
        dc_gain         = (D_cl)(0, 0) + (C_cl * inv_negA * B_cl)(0, 0);
    } catch (const std::exception& e) {
        fmt::print("Failed to compute DC gain: {}\n", e.what());
    }
    double expected_ss = dc_gain * step_mag;
    fmt::print("Computed DC gain = {:.6e}, expected steady-state for step {:.3f} V = {:.6f} V\n", dc_gain, step_mag, expected_ss);

    // Convert Matrix outputs to scalar vectors for plotting
    std::vector<double> t_ol = step_resp_ol.time;
    std::vector<double> y_ol;
    y_ol.reserve(step_resp_ol.output.size());
    for (const auto& m : step_resp_ol.output) y_ol.push_back(m(0, 0));

    std::vector<double> t_cl = step_resp_cl.time;
    std::vector<double> y_cl;
    y_cl.reserve(step_resp_cl.output.size());
    for (const auto& m : step_resp_cl.output) y_cl.push_back(m(0, 0));

    // Plot step responses
    auto trace_ol_step = Scatter()
                             .x(t_ol)
                             .y(y_ol)
                             .mode({Scatter::Mode::Lines})
                             .name("Open-Loop")
                             .line(Scatter::Line().width(2).color("blue"))
                             .xaxis("x2")
                             .yaxis("y3");

    auto trace_cl_step = Scatter()
                             .x(t_cl)
                             .y(y_cl)
                             .mode({Scatter::Mode::Lines})
                             .name("Closed-Loop (ref step)")
                             .line(Scatter::Line().width(2).color("red"))
                             .xaxis("x2")
                             .yaxis("y3");

    // Create figure with subplots using grid layout
    auto layout = Layout()
                      .title([](auto& t) { t.text("Buck Converter - Bode Plots and Step Response"); })
                      .height(1200)
                      .width(1000)
                      .xaxis(1, Layout::Xaxis().type(Layout::Xaxis::Type::Log).title([](auto& t) { t.text("Frequency [rad/s]"); }))
                      .xaxis(2, Layout::Xaxis().title([](auto& t) { t.text("Time [s]"); }))
                      .yaxis(1, Layout::Yaxis().title([](auto& t) { t.text("Magnitude [dB]"); }))
                      .yaxis(2, Layout::Yaxis().title([](auto& t) { t.text("Phase [deg]"); }))
                      .yaxis(3, Layout::Yaxis().title([](auto& t) { t.text("Output Voltage [V]"); }))
                      .grid(Layout::Grid{}
                                .rows(3)
                                .columns(1)
                                .subplots(std::vector<std::vector<std::string>>{{"xy"}, {"xy2"}, {"x2y3"}})
                                .roworder(Layout::Grid::Roworder::BottomToTop));

    auto figure = Figure()
                      .addTraces(std::vector<Trace>{trace_ol_mag, trace_cl_mag, trace_ol_ph, trace_cl_ph, trace_ol_step, trace_cl_step})
                      .setLayout(layout);

    // Save the plot as HTML
    figure.show();
    // figure.writeHtml("buck_converter_plots.html");
    fmt::print("Plots saved to buck_converter_plots.html\n");

    return 0;
}