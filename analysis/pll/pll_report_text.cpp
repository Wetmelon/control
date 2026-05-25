#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

#define FMT_HEADER_ONLY
#include <fmt/format.h>

#include "pll_report_support.hpp"

namespace wetmelon::control::pll_report {

namespace {

std::string nyquist_distance_or_na(const std::optional<std::pair<double, double>>& value, int precision = 4) {
    if (!value.has_value()) {
        return "N/A";
    }
    return fmt::format("{:.{}f}", value->first, precision);
}

std::string pair_or_na_hz(const std::optional<std::pair<double, double>>& value, int p1 = 3, int p2 = 3) {
    if (!value.has_value()) {
        return "N/A";
    }
    const double f_hz = value->second / TWO_PI;
    return fmt::format("{:.{}f} @ {:.{}f}", value->first, p1, f_hz, p2);
}

std::string value_or_na_hz(const std::optional<double>& value, int precision = 4) {
    if (!value.has_value()) {
        return "N/A";
    }
    return fmt::format("{:.{}f}", value.value() / TWO_PI, precision);
}

std::string make_discrete_table_md(
    const std::vector<double>&      sample_rates_hz,
    const std::vector<LoopMetrics>& metrics
) {
    fmt::memory_buffer buffer;
    fmt::format_to(std::back_inserter(buffer), "| Fs [Hz] | PM [deg] @ fc [Hz] | GM [dB] @ f180 [Hz] | BW [Hz] | min\\|1+L\\| | Ms [dB] |\n");
    fmt::format_to(std::back_inserter(buffer), "|---:|:---|:---|---:|---:|---:|\n");

    for (size_t i = 0; i < sample_rates_hz.size(); ++i) {
        fmt::format_to(
            std::back_inserter(buffer),
            "| {:.0f} | {} | {} | {} | {} | {:.2f} |\n",
            sample_rates_hz[i],
            pair_or_na_hz(metrics[i].phase_margin, 2, 2),
            pair_or_na_hz(metrics[i].gain_margin, 2, 2),
            value_or_na_hz(metrics[i].bandwidth, 2),
            nyquist_distance_or_na(metrics[i].min_nyquist_distance, 4),
            metrics[i].peak_sensitivity_db
        );
    }

    return fmt::to_string(buffer);
}

std::string make_delay_table_md(
    const std::vector<double>&                    delays_s,
    const std::vector<std::pair<double, double>>& rows
) {
    fmt::memory_buffer buffer;
    fmt::format_to(std::back_inserter(buffer), "| Delay Td [us] | Added phase lag [deg] | Estimated PM [deg] | Fraction of PM consumed |\n");
    fmt::format_to(std::back_inserter(buffer), "|---:|---:|---:|---:|\n");

    double pm_ref_deg = std::numeric_limits<double>::quiet_NaN();
    if (!rows.empty()) {
        pm_ref_deg = rows.front().first + rows.front().second;
    }

    for (size_t i = 0; i < delays_s.size(); ++i) {
        const double td_us = delays_s[i] * 1e6;
        const double lag = rows[i].first;
        const double pm_est = rows[i].second;
        const double frac = std::isfinite(pm_ref_deg) && (pm_ref_deg > 0.0) ? (lag / pm_ref_deg) : 0.0;

        fmt::format_to(
            std::back_inserter(buffer),
            "| {:.1f} | {:.2f} | {:.2f} | {:.3f} |\n",
            td_us,
            lag,
            pm_est,
            frac
        );
    }

    return fmt::to_string(buffer);
}

std::string make_pi_sweep_table_md(const std::vector<PISweepEntry>& entries) {
    fmt::memory_buffer buffer;
    fmt::format_to(
        std::back_inserter(buffer),
        "| Kp | Ki | PM [deg] @ fc [Hz] | GM [dB] @ f180 [Hz] | BW [Hz] | Ms [dB] | min\\|1+L\\| |\n"
    );
    fmt::format_to(std::back_inserter(buffer), "|---:|---:|:---|:---|---:|---:|---:|\n");

    for (const auto& e : entries) {
        fmt::format_to(
            std::back_inserter(buffer),
            "| {:.3f} | {:.3f} | {} | {} | {} | {:.2f} | {} |\n",
            e.kp,
            e.ki,
            pair_or_na_hz(e.metrics.phase_margin, 2, 2),
            pair_or_na_hz(e.metrics.gain_margin, 2, 2),
            value_or_na_hz(e.metrics.bandwidth, 2),
            e.metrics.peak_sensitivity_db,
            nyquist_distance_or_na(e.metrics.min_nyquist_distance, 4)
        );
    }

    return fmt::to_string(buffer);
}

} // namespace

std::string make_stdout_report(
    double                                        f_nom,
    double                                        w_nom,
    double                                        k_sogi,
    double                                        k_p,
    double                                        k_i,
    const ContinuousResults&                      cont,
    const std::vector<double>&                    discrete_fs,
    const std::vector<LoopMetrics>&               discrete_metrics,
    const std::vector<double>&                    delay_s,
    const std::vector<std::pair<double, double>>& delay_rows,
    const std::vector<PISweepEntry>&              sweep_entries
) {
    const auto& pm = cont.metrics.phase_margin;
    const auto& gm = cont.metrics.gain_margin;

    const double omega_c = pm.has_value() ? pm->second : std::numeric_limits<double>::quiet_NaN();
    const double f_c_hz = std::isfinite(omega_c) ? omega_c / TWO_PI : std::numeric_limits<double>::quiet_NaN();
    const double pm_deg = pm.has_value() ? pm->first : std::numeric_limits<double>::quiet_NaN();

    std::ostringstream md;
    md << std::fixed << std::setprecision(6);

    md << "## PLL Analysis Metrics (stdout)\n\n";
    md << "Nominal parameters: f0 = " << std::setprecision(2) << f_nom << " Hz, w0 = " << std::setprecision(3) << w_nom
       << " rad/s, k = " << k_sogi << ", Kp = " << k_p << ", Ki = " << k_i << "\n\n";

    md << "### Continuous-Time Metrics\n\n";
    md << "| Metric | Value |\n";
    md << "|:---|:---|\n";
    md << "| Phase margin [deg] @ gain crossover [Hz] | " << pair_or_na_hz(pm, 3, 3) << " |\n";
    md << "| Gain margin [dB] @ phase crossover [Hz] | " << pair_or_na_hz(gm, 3, 3) << " |\n";
    md << "| Closed-loop -3 dB bandwidth [Hz] | " << value_or_na_hz(cont.metrics.bandwidth, 3) << " |\n";
    md << "| Min Nyquist distance min\\|1 + L(jw)\\| | " << nyquist_distance_or_na(cont.metrics.min_nyquist_distance, 6) << " |\n";
    md << "| Peak sensitivity Ms [dB] | " << std::setprecision(3) << cont.metrics.peak_sensitivity_db << " |\n\n";

    md << "### Interactive Plots\n\n";
    md << "![Open-Loop Bode (Magnitude + Phase, Shared X Axis)](plots/bode_shared.svg)\n\n";
    md << "![Closed-Loop T and Sensitivity S](plots/closed_loop_ts.svg)\n\n";
    md << "![Nyquist L(jw)](plots/nyquist.svg)\n\n";

    md << "### Sampled-Data Trends Versus Sampling Rate\n\n";
    md << "Discrete models are obtained with Tustin discretization per block and evaluated on the unit circle z = exp(j*w*Ts). Frequency entries are reported in Hz.\n\n";
    md << make_discrete_table_md(discrete_fs, discrete_metrics) << "\n";

    md << "### Phase-Delay Sensitivity\n\n";
    md << "Delay-induced phase loss at gain crossover is approximated by Delta_phi = -360 * f_c * Td.\n\n";
    if (std::isfinite(omega_c) && std::isfinite(pm_deg)) {
        md << "Continuous-time crossover used for this estimate: f_c = " << std::setprecision(4) << f_c_hz
           << " Hz, PM = " << std::setprecision(3) << pm_deg << " deg.\n\n";
    } else {
        md << "Gain crossover not detected in sampled data; delay sensitivity table is shown with computed values only where available.\n\n";
    }
    md << make_delay_table_md(delay_s, delay_rows) << "\n";

    md << "### PI Gain Sweep (Continuous-Time)\n\n";
    md << "Sweep grid: Kp in {0.25, 0.5, 1.0, 2.0, 4.0} x baseline, Ki in {0.25, 0.5, 1.0, 2.0, 4.0} x baseline.\n\n";
    md << make_pi_sweep_table_md(sweep_entries) << "\n";

    md << "Copy/paste relevant values into your hand-written markdown paper.\n";

    return md.str();
}

} // namespace wetmelon::control::pll_report
