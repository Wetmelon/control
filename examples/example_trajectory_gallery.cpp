#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "fmt/core.h"
#include "wet/plotting/plot_plotly.hpp"
#include "wet/trajectory/trajectory.hpp"

using namespace wet;

// ===== Motion-Profile Gallery =====
//
// Shows the three trajectory generators in trajectory.hpp side by side across a
// set of boundary conditions, as a grid of plots:
//
//   rows    = generator  (trapezoidal / jerk-limited S-curve / min-jerk quintic)
//   columns = boundary-condition scenario (long, short, nonzero end velocity, both
//             ends moving with asymmetric accel/decel, and a handbrake start where
//             the initial speed exceeds Vmax)
//
// Each cell overlays position, velocity and acceleration (each normalized to its
// own peak so the *shapes* line up on a common axis). The story the grid tells:
//
//   - Trapezoidal: acceleration is a square wave (jerk is impulsive).
//   - S-curve:     acceleration is trapezoidal — it ramps at ±Jmax, so the curve
//                  is C² (no torque step). Takes a little longer than trapezoidal.
//   - Min-jerk:    a fixed-time quintic; acceleration is a smooth bell, zero at the
//                  ends. Matches arbitrary boundary velocities but ignores limits.

namespace {

constexpr int kRows = 3; // generators
constexpr int kCols = 5; // scenarios

struct Curve {
    std::vector<double> t, p, v, a;
};

// Sample any profile's eval(t) over [0, Tf] (plus a short settle tail).
template<typename Eval>
Curve sample(Eval&& eval, double Tf, int N = 300) {
    Curve        c;
    const double span = (Tf > 0.0 ? Tf : 1.0) * 1.06;
    for (int i = 0; i < N; ++i) {
        const double tt = (span * i) / (N - 1);
        const auto   s = eval(tt);
        c.t.push_back(tt);
        c.p.push_back(s.position);
        c.v.push_back(s.velocity);
        c.a.push_back(s.acceleration);
    }
    return c;
}

// Scale a signal to unit peak magnitude so the three signals share one y-axis.
void normalize(std::vector<double>& y) {
    double m = 0.0;
    for (const double v : y) {
        m = std::max(m, std::abs(v));
    }
    if (m < 1e-12) {
        return;
    }
    for (double& v : y) {
        v /= m;
    }
}

struct Scenario {
    std::string              name;
    double                   Xi, Vi, Xf, Vf;
    TrajectoryLimits<double> lim;
};

} // namespace

int main() {
    fmt::print("===== Motion-Profile Gallery =====\n\n");

    const auto scenarios = std::to_array<Scenario, kCols>({
        {.name = "rest → rest (long)", .Xi = 0.0, .Vi = 0.0, .Xf = 10.0, .Vf = 0.0, .lim = {2.0, 3.0, 3.0, 12.0}},
        {.name = "rest → rest (short)", .Xi = 0.0, .Vi = 0.0, .Xf = 0.6, .Vf = 0.0, .lim = {2.0, 3.0, 3.0, 12.0}},
        {.name = "nonzero Vf", .Xi = 0.0, .Vi = 0.0, .Xf = 8.0, .Vf = 1.5, .lim = {2.0, 3.0, 3.0, 12.0}},
        {.name = "moving → moving (asym a/d)", .Xi = 0.0, .Vi = 1.0, .Xf = 9.0, .Vf = -0.8, .lim = {2.5, 2.0, 6.0, 12.0}},
        {.name = "handbrake (Vi > Vmax)", .Xi = 0.0, .Vi = 4.0, .Xf = 10.0, .Vf = 0.0, .lim = {2.0, 3.0, 3.0, 12.0}},
    });

    const std::array<std::string, kRows> gen_names{
        {"Trapezoidal", "S-curve (jerk-limited)", "Min-jerk quintic"}
    };

    // Build a kRows x kCols grid of curves: [generator][scenario].
    std::array<std::array<Curve, kCols>, kRows> grid{};
    std::array<double, kCols>                   column_tf{};
    for (int c = 0; c < kCols; ++c) {
        const Scenario& s = scenarios[c];

        const auto trap = design::synthesize_trapezoidal(s.Xi, s.Vi, s.Xf, s.Vf, s.lim);
        const auto scrv = design::synthesize_scurve(s.Xi, s.Vi, s.Xf, s.Vf, s.lim);
        // Fixed-time quintic over the trapezoidal's minimum time, matching the same
        // boundary velocities (accelerations zero at both ends).
        const auto poly = design::synthesize_poly_trajectory<5>(
            TrajectoryBoundary<double>{s.Xi, s.Vi, 0.0, 0.0}, TrajectoryBoundary<double>{s.Xf, s.Vf, 0.0, 0.0}, trap.Tf
        );

        fmt::print("{:<26}  trap Tf={:5.2f}s   scurve Tf={:5.2f}s   quintic T={:5.2f}s\n", s.name, trap.Tf, scrv.duration, poly.duration);

        column_tf[c] = std::max({trap.Tf, scrv.duration, poly.duration});

        grid[0][c] = sample([&](double t) { return trap.eval(t); }, trap.Tf);
        grid[1][c] = sample([&](double t) { return scrv.eval(t); }, scrv.duration);
        grid[2][c] = sample([&](double t) { return poly.eval(t); }, poly.duration);

        for (int r = 0; r < kRows; ++r) {
            normalize(grid[r][c].p);
            normalize(grid[r][c].v);
            normalize(grid[r][c].a);
        }
    }

    // --- Assemble the grid figure ---
    using namespace plotlypp;
    Figure fig;

    struct Signal {
        const char*         name;
        std::vector<double> Curve::* field;
        const char*                  color;
    };

    const auto signals = std::to_array<Signal>({
        {"position", &Curve::p, "#1f77b4"},
        {"velocity", &Curve::v, "#ff7f0e"},
        {"acceleration", &Curve::a, "#2ca02c"},
    });

    auto axis_id = [](int n, char ax) { return std::string(1, ax) + (n == 1 ? "" : std::to_string(n)); };

    for (int r = 0; r < kRows; ++r) {
        for (int c = 0; c < kCols; ++c) {
            const int    n = (r * kCols) + c + 1;
            const Curve& cur = grid[r][c];
            const bool   first = (n == 1); // only this cell contributes legend entries
            for (const auto& sig : signals) {
                fig.addTrace(
                    Scatter()
                        .x(cur.t)
                        .y(cur.*(sig.field))
                        .mode({Scatter::Mode::Lines})
                        .name(sig.name)
                        .legendgroup(sig.name)
                        .showlegend(first)
                        .line(Scatter::Line().color(sig.color).width(2.0))
                        .xaxis(axis_id(n, 'x'))
                        .yaxis(axis_id(n, 'y'))
                );
            }
        }
    }

    // Subplot domains (leave a left gutter for row labels, a top band for the
    // title / legend / column headers).
    constexpr double left = 0.06, right = 1.0, hgap = 0.035;
    constexpr double top = 0.88, bottom = 0.05, vgap = 0.075;
    constexpr double colw = ((right - left) - ((kCols - 1) * hgap)) / kCols;
    constexpr double rowh = ((top - bottom) - ((kRows - 1) * vgap)) / kRows;

    auto layout = Layout();

    std::vector<Layout::Annotation> notes;
    for (int r = 0; r < kRows; ++r) {
        for (int c = 0; c < kCols; ++c) {
            const int    n = (r * kCols) + c + 1;
            const double x0 = left + (c * (colw + hgap));
            const double x1 = x0 + colw;
            const double y1 = top - (r * (rowh + vgap));
            const double y0 = y1 - rowh;
            const bool   bottom_row = (r == kRows - 1);
            const bool   left_col = (c == 0);

            layout.xaxis(n, Layout::Xaxis().domain({x0, x1}).anchor(axis_id(n, 'y')).range({0.0, column_tf[c] * 1.1}).showticklabels(bottom_row).title([&](auto& t) { t.text(bottom_row ? "t [s]" : ""); }));
            layout.yaxis(n, Layout::Yaxis().domain({y0, y1}).anchor(axis_id(n, 'x')).range({-1.18, 1.18}).showticklabels(left_col).zeroline(true));

            if (r == 0) { // column header = scenario
                notes.push_back(Layout::Annotation().text(scenarios[c].name).xref("paper").yref("paper").x((x0 + x1) / 2.0).y(top + 0.035).xanchor(Layout::Annotation::Xanchor::Center).showarrow(false).font([](auto& ft) { ft.size(12); }));
            }
            if (c == 0) { // row label = generator (centered on the row's vertical midpoint)
                notes.push_back(Layout::Annotation().text(gen_names[r]).xref("paper").yref("paper").x(0.012).y((y0 + y1) / 2.0).xanchor(Layout::Annotation::Xanchor::Center).yanchor(Layout::Annotation::Yanchor::Middle).textangle(-90.0).showarrow(false).font([](auto& ft) { ft.size(13); }));
            }
        }
    }

    using Lg = Layout::Legend;
    layout.title([](auto& t) { t.text("Motion-profile gallery — position / velocity / acceleration (each normalized)"); })
        .annotations(notes)
        .legend(Lg().orientation(Lg::Orientation::H).x(0.5).xanchor(Lg::Xanchor::Center).y(0.95).yanchor(Lg::Yanchor::Bottom))
        .showlegend(true)
        .height(880)
        .width(1560);

    fig.setLayout(wet::move(layout));
    fig.writeHtml("trajectory_gallery.html");
    fmt::print("\nPlot written to trajectory_gallery.html\n");
    return 0;
}
