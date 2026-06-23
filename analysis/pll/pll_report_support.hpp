#pragma once

#include <cmath>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

#include "wet/analysis/analysis.hpp"
#include "wet/systems/state_space.hpp"
#include "wet/systems/transfer_function.hpp"

namespace wet::pll_report {

constexpr double TWO_PI = 2.0 * std::numbers::pi;
using LoopMetrics = analysis::LoopSummary<double>;

struct ContinuousResults {
    analysis::BodeResult<double> open_loop;
    std::vector<double>          t_mag_db;
    std::vector<double>          s_mag_db;
    std::vector<double>          nyq_re;
    std::vector<double>          nyq_im;
    LoopMetrics                  metrics;
};

struct PISweepEntry {
    double      kp{0.0};
    double      ki{0.0};
    LoopMetrics metrics{};
};

template<size_t NX, size_t NW, size_t NV>
LoopMetrics analyze_loop(
    const StateSpace<NX, 1, 1, NW, NV, double>& loop,
    const std::vector<double>&                  omega,
    bool                                        discrete
) {
    (void)discrete;
    return analysis::loop_metrics(loop, omega);
}

template<size_t Nnum, size_t Nden>
LoopMetrics analyze_loop(
    const TransferFunction<Nnum, Nden, double>& loop,
    const std::vector<double>&                  omega,
    bool                                        discrete
) {
    (void)discrete;
    return analysis::loop_metrics(loop, omega);
}

template<size_t NX, size_t NW, size_t NV>
ContinuousResults compute_continuous(
    const StateSpace<NX, 1, 1, NW, NV, double>& loop,
    const std::vector<double>&                  omega
) {
    ContinuousResults result;
    const auto        response = analysis::loop_response(loop, omega);
    result.open_loop = response.open_loop;
    result.metrics = analysis::summarize_loop_response(response);

    result.t_mag_db.reserve(omega.size());
    result.s_mag_db.reserve(omega.size());
    result.nyq_re.reserve(omega.size());
    result.nyq_im.reserve(omega.size());

    for (size_t i = 0; i < response.open_loop.points.size(); ++i) {
        result.t_mag_db.push_back(response.complementary_sensitivity.points[i].magnitude_db);
        result.s_mag_db.push_back(response.sensitivity.points[i].magnitude_db);
        result.nyq_re.push_back(response.nyquist.points[i].real);
        result.nyq_im.push_back(response.nyquist.points[i].imag);
    }

    return result;
}

template<size_t Nnum, size_t Nden>
ContinuousResults compute_continuous(
    const TransferFunction<Nnum, Nden, double>& loop,
    const std::vector<double>&                  omega
) {
    ContinuousResults                    result;
    const TransferFunction<1, 1, double> unity_tf{{1.0}, {1.0}};
    const auto                           t_tf = loop / unity_tf;
    const auto                           s_tf = unity_tf - t_tf;

    result.open_loop = analysis::bode(loop, omega);
    const auto t_bode = analysis::bode(t_tf, omega);
    const auto s_bode = analysis::bode(s_tf, omega);
    const auto nyq = analysis::nyquist(loop, omega);
    result.metrics = analysis::loop_metrics(loop, omega);

    result.t_mag_db.reserve(omega.size());
    result.s_mag_db.reserve(omega.size());
    result.nyq_re.reserve(omega.size());
    result.nyq_im.reserve(omega.size());

    for (size_t i = 0; i < result.open_loop.points.size(); ++i) {
        result.t_mag_db.push_back(t_bode.points[i].magnitude_db);
        result.s_mag_db.push_back(s_bode.points[i].magnitude_db);
        result.nyq_re.push_back(nyq.points[i].real);
        result.nyq_im.push_back(nyq.points[i].imag);
    }

    return result;
}

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
);

void write_plot_artifacts(const ContinuousResults& cont, const std::vector<double>& omega);

} // namespace wet::pll_report
