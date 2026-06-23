/**
 * @file example_inverted_pendulum.cpp
 * @brief Continuous-time state-space control of the cart-pendulum, replicating the
 *        classic University of Michigan CTMS example:
 *        "Inverted Pendulum: State-Space Methods for Controller Design"
 *        https://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=ControlStateSpace
 *
 * Every step is continuous-time, mirroring the CTMS walkthrough:
 *   1. Linearized state-space model        (states: cart x, ẋ, pole φ, φ̇)
 *   2. Open-loop poles                      → a real RHP pole: inverted = unstable
 *   3. Controllability                      → full rank, so LQR can shape it freely
 *   4. LQR design  K = lqr(A,B,Q,R)         → via the continuous ARE (care)
 *   5. Reference scaling  Nbar              → so the cart tracks a position command
 *   6. Step response (full-state feedback)  → RK4 of the closed loop to a 0.2 m step
 *   7. Observer design  L = place(A',C',p)' → estimate the state from x and φ only
 *   8. Step response (observer in the loop) → u = Nbar·r − K·x̂
 *
 * Two interactive plots are written (plotly HTML), matching the CTMS figures:
 *   inverted_pendulum_state_feedback.html   inverted_pendulum_observer.html
 */

#include <cassert>
#include <cstddef>
#include <numbers>
#include <plotlypp/figure.hpp>
#include <plotlypp/layout/layout.hpp>
#include <plotlypp/traces/scatter.hpp>
#include <string>
#include <vector>

#include "fmt/base.h"
#include "fmt/core.h"
#include "wet/backend.hpp"
#include "wet/control.hpp"
#include "wet/math/math.hpp"

using namespace wet;

namespace {

// Classic RK4 step for an N-state autonomous closed loop ẋ = f(x).
template<size_t N, typename F>
ColVec<N> rk4_step(const ColVec<N>& x, F&& f, double dt) {
    const ColVec<N> k1 = f(x);
    const ColVec<N> k2 = f(ColVec<N>(x + (k1 * (dt / 2.0))));
    const ColVec<N> k3 = f(ColVec<N>(x + (k2 * (dt / 2.0))));
    const ColVec<N> k4 = f(ColVec<N>(x + (k3 * dt)));
    return ColVec<N>(x + ((k1 + (k2 * 2.0) + (k3 * 2.0) + k4) * (dt / 6.0)));
}

constexpr double kR2D = 180.0 / std::numbers::pi_v<double>;

// One line on a panel: a named (t, y) scatter bound to the given plotly axes.
plotlypp::Scatter line(const std::vector<double>& t, const std::vector<double>& y, const std::string& name, const char* xaxis, const char* yaxis, bool dashed = false) {
    auto s = plotlypp::Scatter().x(t).y(y).mode({plotlypp::Scatter::Mode::Lines}).name(name).xaxis(xaxis).yaxis(yaxis);
    if (dashed) {
        s = s.line([](auto& l) { l.dash("dash"); });
    }
    return s;
}

} // namespace

int main() {
    // ===== 1. Plant parameters (CTMS values) ================================
    constexpr double M = 0.5;   // cart mass                       [kg]
    constexpr double m = 0.2;   // pendulum mass                   [kg]
    constexpr double b = 0.1;   // cart friction                   [N/(m/s)]
    constexpr double I = 0.006; // pendulum inertia about its CoM  [kg·m²]
    constexpr double g = 9.8;   // gravity                         [m/s²]
    constexpr double l = 0.3;   // CoM distance from pivot         [m]

    constexpr double p = (I * (M + m)) + (M * m * l * l); // common denominator

    // Linearized about the *upright* equilibrium (φ ≈ 0). States [x, ẋ, φ, φ̇].
    const Matrix<4, 4> A{
        {0.0, 1.0, 0.0, 0.0},
        {0.0, -(I + (m * l * l)) * b / p, (m * m * g * l * l) / p, 0.0},
        {0.0, 0.0, 0.0, 1.0},
        {0.0, -(m * l * b) / p, m * g * l * (M + m) / p, 0.0},
    };
    const ColVec<4> B{
        0.0,
        (I + (m * l * l)) / p,
        0.0,
        (m * l) / p,
    };
    // Outputs: cart position x and pendulum angle φ (both measured).
    const Matrix<2, 4> C{
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0},
    };

    fmt::print("===== Inverted Pendulum — Continuous State-Space (CTMS replica) =====\n\n");
    fmt::print("Plant: M={} kg, m={} kg, l={} m, I={} kg·m²  (p = {:.4f})\n\n", M, m, l, I, p);

    // ===== 2. Open-loop poles ==============================================
    const auto ol = mat::compute_eigenvalues(A);
    fmt::print("Open-loop poles (eigenvalues of A):\n");
    for (size_t i = 0; i < 4; ++i) {
        fmt::print("   {:+.4f} {:+.4f}j\n", ol.values[i].real(), ol.values[i].imag());
    }
    fmt::print("  → one pole in the right half-plane: the upright pendulum is unstable.\n\n");

    // ===== 3. Controllability ==============================================
    const bool ctrl = stability::is_controllable(A, B);
    fmt::print("Controllable: {}  (full rank → LQR may place the dynamics freely)\n\n", ctrl ? "yes" : "no");
    assert(ctrl);

    // ===== 4. LQR design ===================================================
    // CTMS weighting: penalize cart position and pole angle; R = 1.
    const Matrix<4, 4> Q{
        {5000.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 100.0, 0.0},
        {0.0, 0.0, 0.0, 0.0},
    };
    const Matrix<1, 1> R{{1.0}};

    const auto lqr = design::continuous_lqr(A, B, Q, R);
    assert(lqr.success);
    fmt::print("LQR gain  K = lqr(A, B, Q, R):\n");
    fmt::print("   [{:.4f}  {:.4f}  {:.4f}  {:.4f}]   (x  ẋ  φ  φ̇)\n\n", lqr.K(0, 0), lqr.K(0, 1), lqr.K(0, 2), lqr.K(0, 3));

    fmt::print("Closed-loop poles (eigenvalues of A − BK):\n");
    for (size_t i = 0; i < 4; ++i) {
        fmt::print("   {:+.4f} {:+.4f}j\n", lqr.e[i].real(), lqr.e[i].imag());
        assert(lqr.e[i].real() < 0.0); // continuous-time stability: Re < 0
    }
    fmt::print("\n");

    // ===== 5. Reference precompensation Nbar ===============================
    // u = Nbar·r − K·x. Closed-loop DC gain to the outputs: y_ss = −C(A−BK)⁻¹B·Nbar·r.
    // Pick Nbar so the cart-position output (row 0 of C) tracks r.
    const Matrix<4, 4> Acl = A - (B * lqr.K);
    const auto         Acl_inv = Acl.inverse();
    assert(Acl_inv.has_value());
    const auto   dc = C * Acl_inv.value() * B; // Matrix<2,1>
    const double Nbar = -1.0 / dc(0, 0);
    fmt::print("Reference precompensator  Nbar = {:.4f}\n\n", Nbar);

    // ===== 6. Step response, full-state feedback (RK4) =====================
    constexpr double r = 0.2;    // commanded cart position [m]
    constexpr double dt = 0.005; // integration step        [s]
    constexpr double Tend = 5.0; // horizon                  [s]
    const int        steps = static_cast<int>(Tend / dt);

    // ẋ = (A−BK) x + B·Nbar·r.
    const auto sf_deriv = [&](const ColVec<4>& x) -> ColVec<4> {
        return ColVec<4>((Acl * x) + (B * (Nbar * r)));
    };

    std::vector<double> ts, sf_cart, sf_angle;
    ColVec<4>           x{0.0, 0.0, 0.0, 0.0}; // start upright, cart at origin
    fmt::print("Step response to r = {:.2f} m (full-state feedback):\n", r);
    fmt::print("    t[s]   cart x[m]   pole φ[deg]\n");
    for (int k = 0; k <= steps; ++k) {
        const double t = k * dt;
        ts.push_back(t);
        sf_cart.push_back(x[0]);
        sf_angle.push_back(x[2] * kR2D);
        if (k % 50 == 0) {
            fmt::print("   {:5.2f}  {:9.4f}   {:9.3f}\n", t, x[0], x[2] * kR2D);
        }
        x = rk4_step<4>(x, sf_deriv, dt);
    }
    assert(wet::abs(x[0] - r) < 0.01); // cart within 1 cm of the command
    assert(wet::abs(x[2]) < 1e-3);     // pole back upright
    fmt::print("  → cart settled at the command with the pole upright. ✓\n\n");

    // ===== 7. Observer design (continuous Luenberger) =====================
    // Place eig(A − LC) well to the left of the controller poles (≈ 8×) via the
    // dual:  L = place(Aᵀ, Cᵀ, p)ᵀ.  Both outputs measured → MIMO placement.
    const wet::array<double, 4> obs_poles{-40.0, -41.0, -42.0, -43.0};
    const auto                  Ld = design::place(A.transpose(), C.transpose(), obs_poles);
    assert(Ld.has_value());
    const Matrix<4, 2> L = Ld.value().transpose();

    fmt::print("Observer gain L = place(Aᵀ, Cᵀ, {{-40,-41,-42,-43}})ᵀ:\n");
    for (size_t i = 0; i < 4; ++i) {
        fmt::print("   [{:+10.4f}  {:+10.4f}]\n", L(i, 0), L(i, 1));
    }
    const auto obs_e = mat::compute_eigenvalues(Matrix<4, 4>(A - (L * C)));
    fmt::print("Observer error poles (eigenvalues of A − LC):\n");
    for (size_t i = 0; i < 4; ++i) {
        fmt::print("   {:+.4f} {:+.4f}j\n", obs_e.values[i].real(), obs_e.values[i].imag());
        assert(obs_e.values[i].real() < 0.0);
    }
    fmt::print("\n");

    // ===== 8. Step response, observer in the loop =========================
    // Combined 8-state z = [x_plant; x̂].  The controller sees only the estimate:
    //   u   = Nbar·r − K·x̂
    //   ẋ   = A x + B u
    //   x̂̇  = A x̂ + B u + L (y − C x̂),   y = C x
    // Seed the estimate with an angle error so the observer's convergence shows.
    const auto obs_deriv = [&](const ColVec<8>& z) -> ColVec<8> {
        ColVec<4> xp{}, xh{};
        for (size_t i = 0; i < 4; ++i) {
            xp[i] = z[i];
            xh[i] = z[i + 4];
        }
        const ColVec<2> y = ColVec<2>(C * xp);
        const double    u = (Nbar * r) - (lqr.K * xh)(0, 0);
        const ColVec<4> xpd = ColVec<4>((A * xp) + (B * u));
        const ColVec<4> xhd = ColVec<4>((A * xh) + (B * u) + (L * ColVec<2>(y - (C * xh))));
        ColVec<8>       dz{};
        for (size_t i = 0; i < 4; ++i) {
            dz[i] = xpd[i];
            dz[i + 4] = xhd[i];
        }
        return dz;
    };

    std::vector<double> ob_cart, ob_angle, ob_cart_hat, ob_angle_hat;
    ColVec<8>           z{};
    z[6] = 0.05; // x̂ initial angle estimate = 0.05 rad while the true angle is 0
    for (int k = 0; k <= steps; ++k) {
        ob_cart.push_back(z[0]);
        ob_angle.push_back(z[2] * kR2D);
        ob_cart_hat.push_back(z[4]);
        ob_angle_hat.push_back(z[6] * kR2D);
        z = rk4_step<8>(z, obs_deriv, dt);
    }
    const double err_x = wet::abs(z[0] - z[4]);
    const double err_phi = wet::abs(z[2] - z[6]);
    fmt::print("Observer in the loop: final estimate error |x−x̂|={:.2e}, |φ−φ̂|={:.2e}\n", err_x, err_phi);
    assert(wet::abs(z[0] - r) < 0.01); // cart still tracks the command
    assert(err_x < 1e-4 && err_phi < 1e-4);
    fmt::print("  → estimate converged and the cart tracked the command. ✓\n\n");

    // ===== Plots (plotly HTML, two stacked panels each, CTMS-style) =======
    using namespace plotlypp;

    auto two_panel_layout = [](const std::string& title, const std::string& top, const std::string& bot) {
        return Layout()
            .title([&](auto& t) { t.text(title); })
            .grid(Layout::Grid().rows(2).columns(1).pattern(Layout::Grid::Pattern::Independent).roworder(Layout::Grid::Roworder::TopToBottom))
            .xaxis(Layout::Xaxis().title([](auto& t) { t.text("time (s)"); }))
            .yaxis(Layout::Yaxis().title([&](auto& t) { t.text(top); }))
            .xaxis(2, Layout::Xaxis().title([](auto& t) { t.text("time (s)"); }))
            .yaxis(2, Layout::Yaxis().title([&](auto& t) { t.text(bot); }));
    };

    Figure fig_sf;
    fig_sf.addTrace(line(ts, sf_cart, "cart position", "x", "y"));
    fig_sf.addTrace(line(ts, sf_angle, "pendulum angle", "x2", "y2"));
    fig_sf.setLayout(two_panel_layout("Inverted Pendulum — LQR Step Response (full-state feedback)", "cart position (m)", "pendulum angle (deg)"));
    fig_sf.writeHtml("inverted_pendulum_state_feedback.html");

    Figure fig_ob;
    fig_ob.addTrace(line(ts, ob_cart, "cart x (true)", "x", "y"));
    fig_ob.addTrace(line(ts, ob_cart_hat, "cart x̂ (estimate)", "x", "y", /*dashed=*/true));
    fig_ob.addTrace(line(ts, ob_angle, "angle φ (true)", "x2", "y2"));
    fig_ob.addTrace(line(ts, ob_angle_hat, "angle φ̂ (estimate)", "x2", "y2", /*dashed=*/true));
    fig_ob.setLayout(two_panel_layout("Inverted Pendulum — Observer-Based Control (true vs estimate)", "cart position (m)", "pendulum angle (deg)"));
    fig_ob.writeHtml("inverted_pendulum_observer.html");

    fmt::print("Plots written:\n");
    fmt::print("   inverted_pendulum_state_feedback.html\n");
    fmt::print("   inverted_pendulum_observer.html\n");
    return 0;
}
