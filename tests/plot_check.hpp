#pragma once

// Test-only helper: dump a labelled xy plot to tests/plots/<file>.html so a human
// can eyeball-verify what a unit test asserts numerically. Pure side-output — the
// CHECK()s remain the pass/fail authority; the plot is for inspection only.

#include <filesystem>
#include <string>
#include <vector>

#include "plotlypp/figure.hpp"
#include "plotlypp/layout/layout.hpp"
#include "plotlypp/traces/scatter.hpp"

namespace plotcheck {

struct Series {
    std::string         name;
    std::vector<double> x;
    std::vector<double> y;
    bool                markers = false; //!< markers (scatter) vs lines (default)
};

/// Write a multi-series xy plot to tests/plots/<file>.
inline void xy(const std::string& file, const std::string& title, const std::string& xlabel, const std::string& ylabel, const std::vector<Series>& series) {
    using namespace plotlypp;
    Figure fig;
    for (const auto& s : series) {
        fig.addTrace(Scatter().x(s.x).y(s.y).mode({s.markers ? Scatter::Mode::Markers : Scatter::Mode::Lines}).name(s.name).legend("legend"));
    }
    fig.setLayout(Layout().title([&](auto& tt) { tt.text(title); }).xaxis(Layout::Xaxis().title([&](auto& tt) { tt.text(xlabel); })).yaxis(Layout::Yaxis().title([&](auto& tt) { tt.text(ylabel); })));

    std::filesystem::create_directories("tests/plots");
    fig.writeHtml("tests/plots/" + file);
}

} // namespace plotcheck
