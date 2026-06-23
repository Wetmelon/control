#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "fmt/base.h"
#include "fmt/core.h"
#include "plotlypp/figure.hpp"
#include "plotlypp/layout/layout.hpp"
#include "plotlypp/traces/scatter.hpp"
#include "wet/backend.hpp"
#include "wet/trajectory/trajectory.hpp"
#include "wet/trajectory/trajectory_types.hpp"

using namespace wet;

// ===== Multi-Axis Coordination: a hydraulic excavator joint move =====
//
// An excavator swings boom / stick / bucket from a "dig" pose to a "dump" pose.
// Each joint has its own kinematic limits (the heavy boom is slow; the light
// bucket is quick) and a different distance to travel, so left to themselves they
// finish at *different* times — the machine lurches, each joint snapping to its
// target and then sitting idle while the others catch up.
//
// `TrajectoryBank` fixes this: it takes the synchronized duration T_sync = the
// slowest joint's minimum time, and time-scales every other joint to it, so all
// three start and finish together (a coordinated / "MoveJ" move). The bottleneck
// joint runs at its full limit; the rest are gentled down to match.
//
// The plot contrasts the two: left column = each joint run independently at its
// own min-time (staggered finishes), right column = coordinated (one finish).

namespace {

struct Joint {
    std::string              name;
    double                   start_deg, target_deg;
    TrajectoryLimits<double> lim; // deg/s, deg/s^2, deg/s^2, deg/s^3
    const char*              color;
};

} // namespace

int main() {
    fmt::print("===== Coordinated Joint Move (excavator boom / stick / bucket) =====\n\n");

    // Limits in degrees: {v_max, a_max, d_max, j_max}. Heavy boom is slow, light
    // bucket is fast; the three moves are different sizes.
    const std::array<Joint, 3> joints{{
        {"boom", 0.0, 55.0, {18.0, 35.0, 35.0, 150.0}, "#1f77b4"},
        {"stick", 10.0, -40.0, {35.0, 90.0, 90.0, 400.0}, "#ff7f0e"},
        {"bucket", -20.0, 80.0, {70.0, 200.0, 200.0, 900.0}, "#2ca02c"},
    }};

    // Plan each joint's own minimum-time S-curve, wrap in a runtime, collect.
    std::array<ScurveTrajectory<double>, 3> trajs{};
    for (size_t j = 0; j < joints.size(); ++j) {
        trajs[j] = ScurveTrajectory<double>(
            design::synthesize_scurve(joints[j].start_deg, joints[j].target_deg, joints[j].lim)
        );
    }
    TrajectoryBank<3, ScurveTrajectory<double>> bank(trajs);

    const double Tsync = bank.duration();
    fmt::print("Per-joint minimum time (independent):\n");
    for (size_t j = 0; j < joints.size(); ++j) {
        fmt::print("  {:<7} {:+6.1f}° move -> Tf = {:4.2f}s  (coordinated scale {:.2f})\n", joints[j].name, joints[j].target_deg - joints[j].start_deg, trajs[j].duration(), bank.scale(j));
    }
    fmt::print("\nT_sync = {:.2f}s — coordinated move; all joints arrive together.\n", Tsync);

    // Sample both schemes over a common timeline (a little tail past T_sync).
    const int           N = 400;
    const double        span = Tsync * 1.06;
    std::vector<double> t(N);

    std::array<std::vector<double>, 3> ang_indep, vel_indep, ang_coord, vel_coord;
    for (auto* v : {&ang_indep, &vel_indep, &ang_coord, &vel_coord}) {
        for (auto& s : *v) {
            s.resize(N);
        }
    }

    double ang_lo = 1e9, ang_hi = -1e9, vel_lo = 1e9, vel_hi = -1e9;
    for (int i = 0; i < N; ++i) {
        const double tt = (span * i) / (N - 1);
        t[i] = tt;
        const auto coord = bank.eval(tt);
        for (size_t j = 0; j < 3; ++j) {
            const auto ind = trajs[j].eval(tt); // independent: clamps to its own endpoint past Tf
            ang_indep[j][i] = ind.position;
            vel_indep[j][i] = ind.velocity;
            ang_coord[j][i] = coord[j].position;
            vel_coord[j][i] = coord[j].velocity;
            ang_lo = std::min({ang_lo, ind.position, coord[j].position});
            ang_hi = std::max({ang_hi, ind.position, coord[j].position});
            vel_lo = std::min({vel_lo, ind.velocity, coord[j].velocity});
            vel_hi = std::max({vel_hi, ind.velocity, coord[j].velocity});
        }
    }

    // --- Plot: 2 rows (angle, velocity) x 2 cols (independent, coordinated) ---
    using namespace plotlypp;
    Figure fig;

    auto axis_id = [](int n, char ax) { return std::string(1, ax) + (n == 1 ? "" : std::to_string(n)); };

    // n: 1=ang/indep 2=ang/coord 3=vel/indep 4=vel/coord
    auto add_joint_traces = [&](int n, const std::array<std::vector<double>, 3>& data) {
        for (size_t j = 0; j < 3; ++j) {
            fig.addTrace(Scatter().x(t).y(data[j]).mode({Scatter::Mode::Lines}).name(joints[j].name).legendgroup(joints[j].name).showlegend(n == 1).line(Scatter::Line().color(joints[j].color).width(2.0)).xaxis(axis_id(n, 'x')).yaxis(axis_id(n, 'y')));
        }
        // T_sync reference line.
        const bool   is_angle = (n == 1 || n == 2);
        const double y0 = is_angle ? ang_lo : vel_lo;
        const double y1 = is_angle ? ang_hi : vel_hi;
        fig.addTrace(Scatter().x(std::vector<double>{Tsync, Tsync}).y(std::vector<double>{y0, y1}).mode({Scatter::Mode::Lines}).name("T_sync").legendgroup("T_sync").showlegend(n == 1).line(Scatter::Line().color("#999999").dash("dash").width(1.0)).xaxis(axis_id(n, 'x')).yaxis(axis_id(n, 'y')));
    };
    add_joint_traces(1, ang_indep);
    add_joint_traces(2, ang_coord);
    add_joint_traces(3, vel_indep);
    add_joint_traces(4, vel_coord);

    constexpr double left = 0.07, right = 1.0, hgap = 0.07;
    constexpr double top = 0.86, bottom = 0.08, vgap = 0.11;
    constexpr double colw = ((right - left) - hgap) / 2.0;
    constexpr double rowh = ((top - bottom) - vgap) / 2.0;

    auto                             layout = Layout();
    std::vector<Layout::Annotation>  notes;
    const std::array<std::string, 2> col_titles{{"Independent (each joint min-time)", "Coordinated (TrajectoryBank)"}};
    const std::array<std::string, 2> row_titles{{"joint angle [deg]", "joint velocity [deg/s]"}};
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            const int    n = (r * 2) + c + 1;
            const double x0 = left + (c * (colw + hgap));
            const double x1 = x0 + colw;
            const double y1 = top - (r * (rowh + vgap));
            const double y0 = y1 - rowh;
            const bool   bottom_row = (r == 1);
            layout.xaxis(n, Layout::Xaxis().domain({x0, x1}).anchor(axis_id(n, 'y')).title([&](auto& tt) { tt.text(bottom_row ? "t [s]" : ""); }));
            layout.yaxis(n, Layout::Yaxis().domain({y0, y1}).anchor(axis_id(n, 'x')).title([&](auto& tt) { tt.text(row_titles[r]); }).zeroline(true));
            if (r == 0) {
                notes.push_back(Layout::Annotation().text(col_titles[c]).xref("paper").yref("paper").x((x0 + x1) / 2.0).y(top + 0.045).xanchor(Layout::Annotation::Xanchor::Center).showarrow(false).font([](auto& ft) { ft.size(13); }));
            }
        }
    }

    using Lg = Layout::Legend;
    layout
        .title([](auto& tt) {
        tt.text("Multi-axis coordination — staggered finishes (left) vs synchronized arrival (right)");
    })
        .annotations(notes)
        .legend(Lg().orientation(Lg::Orientation::H).x(0.5).xanchor(Lg::Xanchor::Center).y(0.95).yanchor(Lg::Yanchor::Bottom))
        .height(760)
        .width(1180);

    fig.setLayout(wet::move(layout));
    fig.writeHtml("coordinated_joints.html");
    fmt::print("\nPlot written to coordinated_joints.html\n");
    return 0;
}
