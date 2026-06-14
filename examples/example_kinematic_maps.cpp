#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "fmt/core.h"
#include "wet/controllers/trajectory.hpp"
#include "wet/kinematics/motion_maps.hpp"
#include "wet/plotting/plot_plotly.hpp"

using namespace wet;

// ===== One toolhead move, four machines: per-actuator kinematics =====
//
// A single straight-line tool move in the XY plane, time-profiled by a jerk-
// limited S-curve along its arc length (the trajectory generator), is pushed
// through each motion architecture's INVERSE kinematics. The grid shows what each
// machine's actuators must do to produce the *same* Cartesian motion:
//
//   row 0  Toolhead (x, y)        — the commanded Cartesian move (the input)
//   row 1  CoreXY (A, B)          — linear belt combination: just scaled S-curves
//   row 2  Polar (r, θ)           — nonlinear; θ whips around as the path passes
//                                   near the origin (a workspace singularity)
//   row 3  Rotary delta (θ₁,θ₂,θ₃)— parallel coupling: three dissimilar profiles
//
//   columns: position | velocity | acceleration   (each trace normalized to its
//                                                   own peak, so shapes compare)
//
// The punchline: identical tool path, wildly different actuator demands — which is
// exactly why the kinematic map matters before the per-actuator trajectory limits.

namespace {

// Central finite difference (endpoints copied from their neighbour).
std::vector<double> deriv(const std::vector<double>& y, double dt) {
    const size_t        n = y.size();
    std::vector<double> d(n, 0.0);
    for (size_t i = 1; i + 1 < n; ++i) {
        d[i] = (y[i + 1] - y[i - 1]) / (2.0 * dt);
    }
    if (n >= 2) {
        d[0] = d[1];
        d[n - 1] = d[n - 2];
    }
    return d;
}

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

// Keep an angle series continuous (no ±2π atan2 jumps).
void unwrap(std::vector<double>& th) {
    constexpr double two_pi = 2.0 * 3.14159265358979323846;
    for (size_t i = 1; i < th.size(); ++i) {
        while (th[i] - th[i - 1] > two_pi / 2) {
            th[i] -= two_pi;
        }
        while (th[i] - th[i - 1] < -two_pi / 2) {
            th[i] += two_pi;
        }
    }
}

struct Row {
    std::string                      label;
    std::vector<std::vector<double>> axes; // one position series per actuator
};

} // namespace

int main() {
    fmt::print("===== One toolhead move through four kinematic maps =====\n\n");

    // Straight-line tool move through (near) the origin, in a horizontal plane.
    const double ax = -60.0, ay = 35.0; // start (x, y)
    const double bx = 60.0, by = -35.0; // end
    const double z0 = -260.0;           // working height (for the delta)
    const double dx = bx - ax, dy = by - ay;
    const double L = std::sqrt((dx * dx) + (dy * dy));
    const double ux = dx / L, uy = dy / L; // unit direction

    // Jerk-limited speed profile along the path arc length s ∈ [0, L] (mm units).
    const TrajectoryLimits<double> lim{180.0, 600.0, 600.0, 6000.0};
    const auto                     prof = design::synthesize_scurve(0.0, L, lim);
    const double                   Tf = prof.duration;

    // Machines.
    const RotaryDelta<double> delta(RotaryDeltaGeometry<double>{300.0, 100.0, 140.0, 360.0});

    const int           N = 600;
    const double        dt = Tf / (N - 1);
    std::vector<double> th_x(N), th_y(N);          // toolhead
    std::vector<double> cx_a(N), cx_b(N);          // CoreXY motors
    std::vector<double> pr_r(N), pr_t(N);          // polar
    std::vector<double> dl_1(N), dl_2(N), dl_3(N); // delta angles [deg]
    const double        rad2deg = 180.0 / 3.14159265358979323846;
    int                 unreachable = 0;

    for (int i = 0; i < N; ++i) {
        const double s = prof.eval(i * dt).position;
        const double x = ax + (s * ux);
        const double y = ay + (s * uy);
        th_x[i] = x;
        th_y[i] = y;

        const auto cxy = CoreXY<double>::inverse(x, y);
        cx_a[i] = cxy.a;
        cx_b[i] = cxy.b;

        const auto pol = PolarMap<double>::inverse(x, y);
        pr_r[i] = pol.r;
        pr_t[i] = pol.theta;

        const auto dik = delta.inverse(x, y, z0);
        if (!dik.reachable) {
            ++unreachable;
        }
        dl_1[i] = dik.actuators[0] * rad2deg;
        dl_2[i] = dik.actuators[1] * rad2deg;
        dl_3[i] = dik.actuators[2] * rad2deg;
    }
    unwrap(pr_t);

    fmt::print("Tool move: ({:.0f},{:.0f}) -> ({:.0f},{:.0f}) mm, length {:.1f} mm, S-curve Tf = {:.2f}s\n", ax, ay, bx, by, L, Tf);
    fmt::print("Delta reachability: {} / {} samples in range\n\n", N - unreachable, N);

    std::array<Row, 4> rows{{
        {"Toolhead (x, y)", {th_x, th_y}},
        {"CoreXY (A, B)", {cx_a, cx_b}},
        {"Polar (r, θ)", {pr_r, pr_t}},
        {"Rotary Δ (θ₁,θ₂,θ₃)", {dl_1, dl_2, dl_3}},
    }};

    // Time axis.
    std::vector<double> t(N);
    for (int i = 0; i < N; ++i) {
        t[i] = i * dt;
    }

    // --- Plot: 4 rows (machine) x 3 cols (position / velocity / acceleration) ---
    using namespace plotlypp;
    Figure                           fig;
    const std::array<const char*, 3> axis_colors{"#1f77b4", "#ff7f0e", "#2ca02c"};
    std::array<bool, 3>              legend_shown{false, false, false};

    auto axis_id = [](int n, char ax) { return std::string(1, ax) + (n == 1 ? "" : std::to_string(n)); };

    for (int r = 0; r < 4; ++r) {
        for (size_t j = 0; j < rows[r].axes.size(); ++j) {
            std::vector<double> pos = rows[r].axes[j];
            std::vector<double> vel = deriv(pos, dt);
            std::vector<double> acc = deriv(vel, dt);
            normalize(pos);
            normalize(vel);
            normalize(acc);
            const std::array<std::vector<double>*, 3> col_data{&pos, &vel, &acc};
            for (int c = 0; c < 3; ++c) {
                const int  n = (r * 3) + c + 1;
                const bool show = (!legend_shown[j]) && (c == 0); // one legend entry per component colour
                fig.addTrace(Scatter().x(t).y(*col_data[c]).mode({Scatter::Mode::Lines}).name(fmt::format("component {}", j + 1)).legendgroup(fmt::format("c{}", j)).showlegend(show).line(Scatter::Line().color(axis_colors[j]).width(2.0)).xaxis(axis_id(n, 'x')).yaxis(axis_id(n, 'y')));
            }
            if (rows[r].axes.size() == 3 || j + 1 == rows[r].axes.size()) {
                legend_shown[j] = true;
            }
        }
    }

    constexpr double left = 0.08, right = 1.0, hgap = 0.05;
    constexpr double top = 0.88, bottom = 0.06, vgap = 0.055;
    constexpr double colw = ((right - left) - (2 * hgap)) / 3.0;
    constexpr double rowh = ((top - bottom) - (3 * vgap)) / 4.0;

    auto                             layout = Layout();
    std::vector<Layout::Annotation>  notes;
    const std::array<const char*, 3> col_titles{"position", "velocity", "acceleration"};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 3; ++c) {
            const int    n = (r * 3) + c + 1;
            const double x0 = left + (c * (colw + hgap));
            const double x1 = x0 + colw;
            const double y1 = top - (r * (rowh + vgap));
            const double y0 = y1 - rowh;
            const bool   bottom_row = (r == 3);
            layout.xaxis(n, Layout::Xaxis().domain({x0, x1}).anchor(axis_id(n, 'y')).showticklabels(bottom_row).title([&](auto& tt) { tt.text(bottom_row ? "t [s]" : ""); }));
            layout.yaxis(n, Layout::Yaxis().domain({y0, y1}).anchor(axis_id(n, 'x')).range({-1.18, 1.18}).zeroline(true));
            if (r == 0) {
                notes.push_back(Layout::Annotation().text(col_titles[c]).xref("paper").yref("paper").x((x0 + x1) / 2.0).y(top + 0.035).xanchor(Layout::Annotation::Xanchor::Center).showarrow(false).font([](auto& ft) { ft.size(13); }));
            }
            if (c == 0) {
                notes.push_back(Layout::Annotation().text(rows[r].label).xref("paper").yref("paper").x(0.01).y((y0 + y1) / 2.0).xanchor(Layout::Annotation::Xanchor::Center).yanchor(Layout::Annotation::Yanchor::Middle).textangle(-90.0).showarrow(false).font([](auto& ft) { ft.size(12); }));
            }
        }
    }

    using Lg = Layout::Legend;
    layout
        .title([](auto& tt) {
        tt.text("Same toolhead move, four machines — per-actuator pos/vel/accel (each normalized)");
    })
        .annotations(notes)
        .legend(Lg().orientation(Lg::Orientation::H).x(0.5).xanchor(Lg::Xanchor::Center).y(0.95).yanchor(Lg::Yanchor::Bottom))
        .height(940)
        .width(1240);

    fig.setLayout(wet::move(layout));
    fig.writeHtml("kinematic_maps.html");
    fmt::print("Plot written to kinematic_maps.html\n");
    return 0;
}
