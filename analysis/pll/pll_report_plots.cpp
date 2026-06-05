#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <plotlypp/figure.hpp>
#include <plotlypp/layout/layout.hpp>
#include <plotlypp/traces/scatter.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "pll_report_support.hpp"

namespace wet::pll_report {

namespace {

std::vector<double> get_phase_deg(const analysis::BodeResult<double>& result) {
    std::vector<double> out;
    out.reserve(result.points.size());
    for (const auto& p : result.points) {
        out.push_back(p.phase_deg);
    }
    return out;
}

std::vector<double> get_mag_db(const analysis::BodeResult<double>& result) {
    std::vector<double> out;
    out.reserve(result.points.size());
    for (const auto& p : result.points) {
        out.push_back(p.magnitude_db);
    }
    return out;
}

double vec_min(const std::vector<double>& v) {
    if (v.empty()) {
        return 0.0;
    }
    return *std::ranges::min_element(v);
}

double vec_max(const std::vector<double>& v) {
    if (v.empty()) {
        return 1.0;
    }
    return *std::ranges::max_element(v);
}

double with_margin_min(double vmin, double vmax, double ratio = 0.08) {
    const double span = std::max(1e-9, vmax - vmin);
    return vmin - (ratio * span);
}

double with_margin_max(double vmin, double vmax, double ratio = 0.08) {
    const double span = std::max(1e-9, vmax - vmin);
    return vmax + (ratio * span);
}

std::string polyline_points(
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    double                     x_min,
    double                     x_max,
    double                     y_min,
    double                     y_max,
    double                     left,
    double                     top,
    double                     width,
    double                     height,
    bool                       x_log
) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);

    const double xl_min = x_log ? std::log10(std::max(1e-12, x_min)) : x_min;
    const double xl_max = x_log ? std::log10(std::max(1e-12, x_max)) : x_max;
    const double x_span = std::max(1e-12, xl_max - xl_min);
    const double y_span = std::max(1e-12, y_max - y_min);

    bool first = true;
    for (size_t i = 0; i < xs.size() && i < ys.size(); ++i) {
        const double x_raw = xs[i];
        const double y_raw = ys[i];
        if (!std::isfinite(x_raw) || !std::isfinite(y_raw)) {
            continue;
        }
        if (x_log && x_raw <= 0.0) {
            continue;
        }

        const double x_val = x_log ? std::log10(x_raw) : x_raw;
        const double x_norm = (x_val - xl_min) / x_span;
        const double y_norm = (y_raw - y_min) / y_span;

        const double sx = left + (x_norm * width);
        const double sy = top + height - (y_norm * height);
        if (!first) {
            oss << ' ';
        }
        first = false;
        oss << sx << ',' << sy;
    }

    return oss.str();
}

void write_bode_svg(
    const std::string&         path,
    const std::vector<double>& freq_hz,
    const std::vector<double>& mag_db,
    const std::vector<double>& phase_deg
) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return;
    }

    const double width = 980.0;
    const double left = 82.0;
    const double right = 24.0;
    const double plot_w = width - left - right;

    const double top1 = 56.0;
    const double h1 = 226.0;
    const double top2 = 344.0;
    const double h2 = 226.0;

    const double x_min = std::max(1e-6, vec_min(freq_hz));
    const double x_max = std::max(x_min * 1.01, vec_max(freq_hz));

    const double mag_min = with_margin_min(vec_min(mag_db), vec_max(mag_db));
    const double mag_max = with_margin_max(vec_min(mag_db), vec_max(mag_db));
    const double ph_min = with_margin_min(vec_min(phase_deg), vec_max(phase_deg));
    const double ph_max = with_margin_max(vec_min(phase_deg), vec_max(phase_deg));

    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 980 640\" width=\"100%\" height=\"640\">\n";
    out << "  <rect x=\"0\" y=\"0\" width=\"980\" height=\"640\" fill=\"#ffffff\"/>\n";
    out << "  <text x=\"490\" y=\"30\" text-anchor=\"middle\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"18\">Open-Loop Bode (Shared Frequency Axis)</text>\n";
    out << "  <rect x=\"" << left << "\" y=\"" << top1 << "\" width=\"" << plot_w << "\" height=\"" << h1 << "\" fill=\"none\" stroke=\"#7f8c8d\" stroke-width=\"1\"/>\n";
    out << "  <rect x=\"" << left << "\" y=\"" << top2 << "\" width=\"" << plot_w << "\" height=\"" << h2 << "\" fill=\"none\" stroke=\"#7f8c8d\" stroke-width=\"1\"/>\n";
    out << "  <polyline fill=\"none\" stroke=\"#1f77b4\" stroke-width=\"1.8\" points=\""
        << polyline_points(freq_hz, mag_db, x_min, x_max, mag_min, mag_max, left, top1, plot_w, h1, true) << "\"/>\n";
    out << "  <polyline fill=\"none\" stroke=\"#d62728\" stroke-width=\"1.8\" points=\""
        << polyline_points(freq_hz, phase_deg, x_min, x_max, ph_min, ph_max, left, top2, plot_w, h2, true) << "\"/>\n";
    out << "  <text x=\"24\" y=\"170\" transform=\"rotate(-90 24 170)\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"12\">Magnitude (dB)</text>\n";
    out << "  <text x=\"24\" y=\"456\" transform=\"rotate(-90 24 456)\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"12\">Phase (deg)</text>\n";
    out << "  <text x=\"490\" y=\"615\" text-anchor=\"middle\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"12\">Frequency (Hz, log scale)</text>\n";
    out << "  <text x=\"" << left + 8.0 << "\" y=\"" << top1 + 18.0 << "\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"12\" fill=\"#1f77b4\">|L| [dB]</text>\n";
    out << "  <text x=\"" << left + 8.0 << "\" y=\"" << top2 + 18.0 << "\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"12\" fill=\"#d62728\">arg(L) [deg]</text>\n";
    out << "</svg>\n";
}

void write_single_logx_svg(
    const std::string&         path,
    const std::string&         title,
    const std::string&         y_label,
    const std::vector<double>& freq_hz,
    const std::vector<double>& y1,
    const std::vector<double>& y2,
    const std::string&         y1_name,
    const std::string&         y2_name
) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return;
    }

    const double left = 82.0;
    const double top = 52.0;
    const double plot_w = 980.0 - left - 24.0;
    const double plot_h = 360.0 - top - 58.0;

    const double x_min = std::max(1e-6, vec_min(freq_hz));
    const double x_max = std::max(x_min * 1.01, vec_max(freq_hz));
    const double y_min = with_margin_min(std::min(vec_min(y1), vec_min(y2)), std::max(vec_max(y1), vec_max(y2)));
    const double y_max = with_margin_max(std::min(vec_min(y1), vec_min(y2)), std::max(vec_max(y1), vec_max(y2)));

    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 980 360\" width=\"100%\" height=\"360\">\n";
    out << "  <rect x=\"0\" y=\"0\" width=\"980\" height=\"360\" fill=\"#ffffff\"/>\n";
    out << "  <text x=\"490\" y=\"28\" text-anchor=\"middle\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"18\">" << title << "</text>\n";
    out << "  <rect x=\"" << left << "\" y=\"" << top << "\" width=\"" << plot_w << "\" height=\"" << plot_h << "\" fill=\"none\" stroke=\"#7f8c8d\" stroke-width=\"1\"/>\n";
    out << "  <polyline fill=\"none\" stroke=\"#1f77b4\" stroke-width=\"1.8\" points=\""
        << polyline_points(freq_hz, y1, x_min, x_max, y_min, y_max, left, top, plot_w, plot_h, true) << "\"/>\n";
    out << "  <polyline fill=\"none\" stroke=\"#d62728\" stroke-width=\"1.8\" points=\""
        << polyline_points(freq_hz, y2, x_min, x_max, y_min, y_max, left, top, plot_w, plot_h, true) << "\"/>\n";
    out << "  <text x=\"24\" y=\"200\" transform=\"rotate(-90 24 200)\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"12\">" << y_label << "</text>\n";
    out << "  <text x=\"490\" y=\"344\" text-anchor=\"middle\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"12\">Frequency (Hz, log scale)</text>\n";
    out << "  <text x=\"" << left + 8.0 << "\" y=\"" << top + 18.0 << "\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"12\" fill=\"#1f77b4\">" << y1_name << "</text>\n";
    out << "  <text x=\"" << left + 8.0 << "\" y=\"" << top + 34.0 << "\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"12\" fill=\"#d62728\">" << y2_name << "</text>\n";
    out << "</svg>\n";
}

void write_nyquist_svg(const std::string& path, const std::vector<double>& re, const std::vector<double>& im) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return;
    }

    std::vector<double> x_all = re;
    std::vector<double> y_all = im;
    x_all.push_back(-1.0);
    y_all.push_back(0.0);

    const double x_min = with_margin_min(vec_min(x_all), vec_max(x_all));
    const double x_max = with_margin_max(vec_min(x_all), vec_max(x_all));
    const double y_min = with_margin_min(vec_min(y_all), vec_max(y_all));
    const double y_max = with_margin_max(vec_min(y_all), vec_max(y_all));

    const double left = 82.0;
    const double top = 52.0;
    const double plot_w = 980.0 - left - 24.0;
    const double plot_h = 360.0 - top - 58.0;

    auto to_screen = [&](double x, double y) {
        const double x_norm = (x - x_min) / std::max(1e-12, x_max - x_min);
        const double y_norm = (y - y_min) / std::max(1e-12, y_max - y_min);
        const double sx = left + (x_norm * plot_w);
        const double sy = top + plot_h - (y_norm * plot_h);
        return std::pair<double, double>{sx, sy};
    };

    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 980 360\" width=\"100%\" height=\"360\">\n";
    out << "  <rect x=\"0\" y=\"0\" width=\"980\" height=\"360\" fill=\"#ffffff\"/>\n";
    out << "  <text x=\"490\" y=\"28\" text-anchor=\"middle\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"18\">Nyquist L(jw)</text>\n";
    out << "  <rect x=\"" << left << "\" y=\"" << top << "\" width=\"" << plot_w << "\" height=\"" << plot_h << "\" fill=\"none\" stroke=\"#7f8c8d\" stroke-width=\"1\"/>\n";
    out << "  <polyline fill=\"none\" stroke=\"#1f77b4\" stroke-width=\"1.8\" points=\""
        << polyline_points(re, im, x_min, x_max, y_min, y_max, left, top, plot_w, plot_h, false) << "\"/>\n";

    const auto [mx, my] = to_screen(-1.0, 0.0);
    out << "  <circle cx=\"" << mx << "\" cy=\"" << my << "\" r=\"4\" fill=\"#d62728\"/>\n";
    out << "  <text x=\"" << (mx + 8.0) << "\" y=\"" << (my - 6.0) << "\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"11\" fill=\"#d62728\">-1 + j0</text>\n";
    out << "  <text x=\"24\" y=\"200\" transform=\"rotate(-90 24 200)\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"12\">Im{L(jw)}</text>\n";
    out << "  <text x=\"490\" y=\"344\" text-anchor=\"middle\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"12\">Re{L(jw)}</text>\n";
    out << "</svg>\n";
}

void replace_all_inplace(std::string& text, const std::string& from, const std::string& to) {
    if (from.empty()) {
        return;
    }

    size_t pos = 0;
    while ((pos = text.find(from, pos)) != std::string::npos) {
        text.replace(pos, from.size(), to);
        pos += to.size();
    }
}

void sanitize_plot_html(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return;
    }

    std::ostringstream buffer;
    buffer << in.rdbuf();
    std::string html = buffer.str();

    replace_all_inplace(html, "#plot { width: 100vw; height: 100vh; }", "#plot { width: 100%; height: 100%; }");
    replace_all_inplace(html, "document.write('<script src=\"https://cdn.plot.ly/plotly-3.0.1.min.js\"><\\/script>');", "");

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return;
    }
    out << html;
}

void write_plotly_figures(const ContinuousResults& cont, const std::vector<double>& omega) {
    using namespace plotlypp;

    std::vector<double> freq_hz;
    freq_hz.reserve(omega.size());
    for (double w : omega) {
        freq_hz.push_back(w / TWO_PI);
    }

    const auto ol_mag = get_mag_db(cont.open_loop);
    const auto ol_phase = get_phase_deg(cont.open_loop);

    Figure fig_bode;
    fig_bode
        .addTrace(
            Scatter()
                .x(freq_hz)
                .y(ol_mag)
                .mode({Scatter::Mode::Lines})
                .name("Magnitude")
                .line(Scatter::Line().color("rgb(0,114,189)").width(2.0))
        )
        .addTrace(
            Scatter()
                .x(freq_hz)
                .y(ol_phase)
                .mode({Scatter::Mode::Lines})
                .name("Phase")
                .line(Scatter::Line().color("rgb(0,114,189)").width(2.0))
                .xaxis("x2")
                .yaxis("y2")
        )
        .setLayout(
            Layout()
                .title([](auto& t) { t.text("Bode Diagram"); })
                .paper_bgcolor("rgb(245,245,245)")
                .plot_bgcolor("rgb(245,245,245)")
                .showlegend(false)
                .xaxis(
                    Layout::Xaxis()
                        .type(Layout::Xaxis::Type::Log)
                        .domain(std::vector<double>{0.0, 1.0})
                        .showticklabels(false)
                        .showgrid(true)
                        .gridcolor("rgb(217,217,217)")
                        .showline(true)
                        .linecolor("rgb(100,100,100)")
                )
                .yaxis(
                    Layout::Yaxis()
                        .title([](auto& t) { t.text("Magnitude (dB)"); })
                        .domain(std::vector<double>{0.55, 1.0})
                        .showgrid(true)
                        .gridcolor("rgb(217,217,217)")
                        .showline(true)
                        .linecolor("rgb(100,100,100)")
                )
                .xaxis(
                    2,
                    Layout::Xaxis()
                        .type(Layout::Xaxis::Type::Log)
                        .matches("x")
                        .title([](auto& t) { t.text("Frequency (Hz)"); })
                        .domain(std::vector<double>{0.0, 1.0})
                        .showgrid(true)
                        .gridcolor("rgb(217,217,217)")
                        .showline(true)
                        .linecolor("rgb(100,100,100)")
                )
                .yaxis(
                    2,
                    Layout::Yaxis()
                        .title([](auto& t) { t.text("Phase (deg)"); })
                        .domain(std::vector<double>{0.0, 0.45})
                        .showgrid(true)
                        .gridcolor("rgb(217,217,217)")
                        .showline(true)
                        .linecolor("rgb(100,100,100)")
                )
                .height(620)
        );
    fig_bode.writeHtml("plots/bode_shared.html");
    sanitize_plot_html("plots/bode_shared.html");

    Figure fig_ts;
    fig_ts
        .addTrace(
            Scatter()
                .x(freq_hz)
                .y(cont.t_mag_db)
                .mode({Scatter::Mode::Lines})
                .name("|T|")
                .line(Scatter::Line().color("rgb(0,114,189)").width(2.0))
        )
        .addTrace(
            Scatter()
                .x(freq_hz)
                .y(cont.s_mag_db)
                .mode({Scatter::Mode::Lines})
                .name("|S|")
                .line(Scatter::Line().color("rgb(217,83,25)").width(2.0))
        )
        .setLayout(
            Layout()
                .title([](auto& t) { t.text("Bode Diagram"); })
                .paper_bgcolor("rgb(245,245,245)")
                .plot_bgcolor("rgb(245,245,245)")
                .xaxis(
                    Layout::Xaxis()
                        .title([](auto& t) { t.text("Frequency (Hz)"); })
                        .type(Layout::Xaxis::Type::Log)
                        .showgrid(true)
                        .gridcolor("rgb(217,217,217)")
                        .showline(true)
                        .linecolor("rgb(100,100,100)")
                )
                .yaxis(
                    Layout::Yaxis()
                        .title([](auto& t) { t.text("Magnitude (dB)"); })
                        .showgrid(true)
                        .gridcolor("rgb(217,217,217)")
                        .showline(true)
                        .linecolor("rgb(100,100,100)")
                )
                .legend(
                    Layout::Legend()
                        .x(0.52)
                        .y(0.20)
                        .xanchor(Layout::Legend::Xanchor::Center)
                        .yanchor(Layout::Legend::Yanchor::Bottom)
                )
                .height(330)
        );
    fig_ts.writeHtml("plots/closed_loop_ts.html");
    sanitize_plot_html("plots/closed_loop_ts.html");

    Figure fig_nyquist;
    fig_nyquist
        .addTrace(
            Scatter()
                .x(cont.nyq_re)
                .y(cont.nyq_im)
                .mode({Scatter::Mode::Lines})
                .name("L(jw)")
                .line(Scatter::Line().color("rgb(0,114,189)").width(2.0))
        )
        .addTrace(
            Scatter()
                .x(std::vector<double>{-1.0})
                .y(std::vector<double>{0.0})
                .mode({Scatter::Mode::Markers})
                .name("-1 + j0")
                .marker(Scatter::Marker().color("rgb(217,83,25)").size(10))
        )
        .setLayout(
            Layout()
                .title([](auto& t) { t.text("Nyquist Diagram"); })
                .paper_bgcolor("rgb(245,245,245)")
                .plot_bgcolor("rgb(245,245,245)")
                .xaxis(
                    Layout::Xaxis()
                        .title([](auto& t) { t.text("Real Axis"); })
                        .showgrid(true)
                        .gridcolor("rgb(217,217,217)")
                        .showline(true)
                        .linecolor("rgb(100,100,100)")
                        .zeroline(true)
                        .zerolinecolor("rgb(170,170,170)")
                )
                .yaxis(
                    Layout::Yaxis()
                        .title([](auto& t) { t.text("Imaginary Axis"); })
                        .showgrid(true)
                        .gridcolor("rgb(217,217,217)")
                        .showline(true)
                        .linecolor("rgb(100,100,100)")
                        .zeroline(true)
                        .zerolinecolor("rgb(170,170,170)")
                )
                .height(330)
        );
    fig_nyquist.writeHtml("plots/nyquist.html");
    sanitize_plot_html("plots/nyquist.html");
}

void write_svg_plots(const ContinuousResults& cont, const std::vector<double>& omega) {
    std::vector<double> freq_hz;
    freq_hz.reserve(omega.size());
    for (double w : omega) {
        freq_hz.push_back(w / TWO_PI);
    }

    write_bode_svg("plots/bode_shared.svg", freq_hz, get_mag_db(cont.open_loop), get_phase_deg(cont.open_loop));
    write_single_logx_svg(
        "plots/closed_loop_ts.svg",
        "Closed-Loop T and Sensitivity S",
        "Magnitude (dB)",
        freq_hz,
        cont.t_mag_db,
        cont.s_mag_db,
        "|T| [dB]",
        "|S| [dB]"
    );
    write_nyquist_svg("plots/nyquist.svg", cont.nyq_re, cont.nyq_im);
}

} // namespace

void write_plot_artifacts(const ContinuousResults& cont, const std::vector<double>& omega) {
    std::filesystem::create_directories("plots");
    write_plotly_figures(cont, omega);
    write_svg_plots(cont, omega);
}

} // namespace wet::pll_report
