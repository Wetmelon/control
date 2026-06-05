#include <cmath>
#include <cstdlib>
#include <numbers>
#include <vector>

#define FMT_HEADER_ONLY
#include <fmt/format.h>

#include "wet/systems/discretization.hpp"
#include "wet/matlab.hpp"
#include "pll_report_support.hpp"
#include "wet/utility.hpp"

using namespace wet;
using namespace wet::pll_report;

int main(int argc, char** argv) {
    constexpr double f_nom = 50.0;
    constexpr double w_nom = 2.0 * std::numbers::pi * f_nom;
    constexpr double k_sogi = std::numbers::sqrt2;

    // Baseline analysis gains for linearized loop model.
    double Kp = 10.0;
    double Ki = 100.0;

    if (argc >= 2) {
        Kp = std::atof(argv[1]);
    }
    if (argc >= 3) {
        Ki = std::atof(argv[2]);
    }

    // SOGI quadrature channel transfer function:
    // Hq(s) = (k*w0^2) / (s^2 + k*w0*s + w0^2)
    const auto hq_tf = matlab::tf({k_sogi * w_nom * w_nom}, {w_nom * w_nom, k_sogi * w_nom, 1.0});

    // PI loop filter transfer function: Cpi(s) = (Kp*s + Ki)/s
    const auto pi_tf = matlab::tf({Ki, Kp}, {0.0, 1.0});

    // NCO-style phase-angle generator transfer function: Gtheta(s) = (2*pi)/s
    const auto phase_integrator_tf = matlab::tf({2.0 * std::numbers::pi}, {0.0, 1.0});

    const auto build_open_loop_tf = [&](double kp, double ki) {
        const auto pi_local = matlab::tf({ki, kp}, {0.0, 1.0});
        return hq_tf * pi_local * phase_integrator_tf;
    };

    const auto loop_tf = build_open_loop_tf(Kp, Ki);

    const auto omega = analysis::logspace(0.1, 2.0e4, 1200);
    const auto cont = compute_continuous(loop_tf, omega);

    // Discrete-time margin trends with Tustin-discretized blocks.
    const std::vector<double> fs_hz{5000.0, 10000.0, 20000.0, 40000.0};
    std::vector<LoopMetrics>  discrete_metrics;
    discrete_metrics.reserve(fs_hz.size());

    for (double fs : fs_hz) {
        const double Ts = 1.0 / fs;
        const auto   hq_d = matlab::c2d(hq_tf, Ts, DiscretizationMethod::Tustin);
        const auto   pi_d = matlab::c2d(pi_tf, Ts, DiscretizationMethod::Tustin);
        const auto   phase_integrator_d = matlab::c2d(phase_integrator_tf, Ts, DiscretizationMethod::Tustin);
        const auto   loop_d = series(series(hq_d, pi_d), phase_integrator_d);

        const double w_max = 0.95 * std::numbers::pi / Ts;
        const auto   omega_d = analysis::logspace(0.1, w_max, 1000);
        discrete_metrics.push_back(analyze_loop(loop_d, omega_d, true));
    }

    // Delay sensitivity around gain crossover.
    const double omega_c = cont.metrics.phase_margin.has_value() ? cont.metrics.phase_margin->second : 0.0;
    const double pm_deg = cont.metrics.phase_margin.has_value() ? cont.metrics.phase_margin->first : 0.0;

    const std::vector<double>              delay_s{25e-6, 50e-6, 100e-6, 200e-6, 400e-6};
    std::vector<std::pair<double, double>> delay_rows;
    delay_rows.reserve(delay_s.size());

    for (double td : delay_s) {
        const double added_lag_deg = omega_c * td * 180.0 / std::numbers::pi;
        const double pm_est = pm_deg - added_lag_deg;
        delay_rows.push_back({added_lag_deg, pm_est});
    }

    // PI sweep around baseline gains (continuous-time model).
    const std::vector<double> gain_scale{0.25, 0.5, 1.0, 2.0, 4.0};
    std::vector<PISweepEntry> sweep_entries;
    sweep_entries.reserve(gain_scale.size() * gain_scale.size());

    for (double kp_scale : gain_scale) {
        for (double ki_scale : gain_scale) {
            const double kp_try = Kp * kp_scale;
            const double ki_try = Ki * ki_scale;
            const auto   loop_try = build_open_loop_tf(kp_try, ki_try);
            sweep_entries.push_back({
                .kp = kp_try,
                .ki = ki_try,
                .metrics = analyze_loop(loop_try, omega, false),
            });
        }
    }

    const auto report_md = make_stdout_report(
        f_nom,
        w_nom,
        k_sogi,
        Kp,
        Ki,
        cont,
        fs_hz,
        discrete_metrics,
        delay_s,
        delay_rows,
        sweep_entries
    );

    write_plot_artifacts(cont, omega);

    fmt::print("Generated plots in analysis/pll/plots\n\n");
    fmt::print("{}", report_md);

    return 0;
}
