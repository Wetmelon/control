#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "doctest.h"
#include "plot_check.hpp"
#include "wet/matrix/colvec.hpp"
#include "wet/matrix/matrix.hpp"
#include "wet/motor/predictive_current.hpp"
#include "wet/simulation/integrator.hpp"
#include "wet/simulation/simulate.hpp"
#include "wet/simulation/solver.hpp"
#include "wet/transforms.hpp"

using namespace wet;

namespace {

// Continuous dq plant dx/dt = f(x,u): x = [id, iq], u = [vd, vq], at fixed speed.
auto dq_dynamics(const motor::PmsmModel<double>& p, double omega) {
    return [p, omega](double /*t*/, const ColVec<2, double>& x, const ColVec<2, double>& u) {
        return ColVec<2, double>{
            (u[0] - (p.R * x[0]) + (omega * p.Ldq.q * x[1])) / p.Ldq.d,
            (u[1] - (p.R * x[1]) - (omega * p.Ldq.d * x[0]) - (omega * p.lambda)) / p.Ldq.q,
        };
    };
}

// Discrete forward-Euler map x[k+1] = f(x[k], u[k]) — matches the deadbeat's own
// discretization, so a correct model deadbeats exactly in one step.
auto dq_map(const motor::PmsmModel<double>& p, double omega, double dt) {
    return [p, omega, dt](size_t /*k*/, const ColVec<2, double>& x, const ColVec<2, double>& u) {
        return ColVec<2, double>{
            x[0] + ((dt / p.Ldq.d) * (u[0] - (p.R * x[0]) + (omega * p.Ldq.q * x[1]))),
            x[1] + ((dt / p.Ldq.q) * (u[1] - (p.R * x[1]) - (omega * p.Ldq.d * x[0]) - (omega * p.lambda))),
        };
    };
}

constexpr auto measure_currents = [](const ColVec<2, double>& x) { return x; };

} // namespace

TEST_SUITE("Predictive (deadbeat) current control") {
    constexpr double               dt = 1e-4;
    constexpr double               omega = 200.0; // electrical rad/s
    constexpr double               Vdc = 200.0;   // ample headroom: isolate deadbeat behavior from saturation
    const motor::PmsmModel<double> motor_true{.Ldq = {.d = 1e-3, .q = 1e-3}, .R = 0.5, .lambda = 0.02};

    TEST_CASE("Matched model reaches the reference in one step") {
        const double                               iq_ref = 5.0;
        motor::PredictiveCurrentController<double> ctrl{motor_true};
        const DirectQuadrature<double>             ref{.d = 0.0, .q = iq_ref};
        auto                                       controller = [&](const ColVec<2, double>& y) {
            const auto c = ctrl.control(ref, {y[0], y[1]}, omega, Vdc, dt);
            return ColVec<2, double>{c.Vdq.d, c.Vdq.q};
        };

        const auto res = sim::simulate_discrete_nonlinear<2, 2, 2, double>(
            dq_map(motor_true, omega, dt), measure_currents, controller, ColVec<2, double>{}, dt, 40
        );

        CHECK(res.x[1][1] == doctest::Approx(iq_ref).epsilon(1e-6)); // deadbeats in one step
        CHECK(res.x.back()[1] == doctest::Approx(iq_ref).epsilon(1e-6));
        for (size_t k = 1; k < res.t.size(); ++k) {
            CHECK(res.x[k][0] == doctest::Approx(0.0).epsilon(1e-6)); // d-axis stays parked
        }
    }

    TEST_CASE("Voltage saturation slews as fast as the bus allows") {
        const double                               Vdc_low = 12.0;
        const double                               iq_ref = 200.0; // unreachable in one step at this Vdc
        motor::PredictiveCurrentController<double> ctrl{motor_true};
        const DirectQuadrature<double>             ref{.d = 0.0, .q = iq_ref};
        auto                                       controller = [&](const ColVec<2, double>& y) {
            const auto c = ctrl.control(ref, {y[0], y[1]}, omega, Vdc_low, dt);
            return ColVec<2, double>{c.Vdq.d, c.Vdq.q};
        };

        const auto res = sim::simulate_discrete_nonlinear<2, 2, 2, double>(
            dq_map(motor_true, omega, dt), measure_currents, controller, ColVec<2, double>{}, dt, 40
        );

        CHECK(res.x[1][1] < iq_ref); // couldn't deadbeat there in one step
        for (size_t k = 2; k < res.t.size(); ++k) {
            CHECK(res.x[k][1] >= res.x[k - 1][1] - 1e-9); // monotone ramp toward the target
        }
        CHECK(res.x.back()[1] > res.x[1][1]); // making progress
    }

    TEST_CASE("Plot: deadbeat tracking degrades under model mismatch (motivates the observer)") {
        // Continuous plant integrated with RK4 at 12 sub-steps/period, so the ringing is a
        // real waveform, not a control-rate-aliased triangle.
        const sim::FixedStepSolver fine{sim::RK4<2, double>{}, dt / 12.0};
        const auto                 step_run = [&](const motor::PmsmModel<double>& ctrl_model) {
            motor::PredictiveCurrentController<double> ctrl{ctrl_model};
            const DirectQuadrature<double>             ref{.d = 0.0, .q = 5.0};
            auto                                       controller = [&](double /*t*/, const ColVec<2, double>& y) {
                const auto c = ctrl.control(ref, {y[0], y[1]}, omega, Vdc, dt);
                return ColVec<2, double>{c.Vdq.d, c.Vdq.q};
            };
            const auto res = sim::simulate_sampled<2, 2, 2, double>(
                dq_dynamics(motor_true, omega), measure_currents, controller, fine, dt, ColVec<2, double>{}, {0.0, 30 * dt}
            );
            std::vector<double> ts, iqs;
            for (size_t k = 0; k < res.t.size(); ++k) {
                ts.push_back(res.t[k]);
                iqs.push_back(res.x[k][1]);
            }
            return std::pair{ts, iqs};
        };

        const auto [t_match, iq_match] = step_run(motor_true);
        motor::PmsmModel<double> wrong = motor_true;
        wrong.Ldq = {.d = 1.8e-3, .q = 1.8e-3}; // +80% L: error multiplier (1-1.8) = -0.8, rings
        const auto [t_wrong, iq_wrong] = step_run(wrong);

        CHECK(*std::ranges::max_element(iq_wrong) > 5.0 * 1.05); // visible overshoot

        plotcheck::xy("predictive_model_mismatch.html", "Deadbeat 5 A step (continuous RK4 plant): matched vs +80% inductance error", "time (s)", "iq (A)", {{.name = "iq matched", .x = t_match, .y = iq_match}, {.name = "iq (L over-est)", .x = t_wrong, .y = iq_wrong}});
    }

    TEST_CASE("Online KF recovers the model and stabilizes the deadbeat loop") {
        const motor::PmsmModel<double> guess{.Ldq = {.d = 1.8e-3, .q = 1.8e-3}, .R = 0.3, .lambda = 0.015};
        const auto                     make_cfg = [&] {
            return motor::PmsmEstimatorConfig<double>{
                                    .model0 = guess,
                                    .Q = Matrix<4, 4, double>::diagonal({1e-6, 1e-13, 1e-13, 1e-11}), // R drifts faster than L/λ
                                    .P0 = Matrix<4, 4, double>::diagonal({1e-1, 1e-6, 1e-6, 1e-4}),   // scaled per parameter
                                    .r = 0.1,
            };
        };

        const double omega_e = 300.0; // nonzero so λ is observable

        // Fast square-wave excitation on both axes (different periods) keeps Ld/Lq (need
        // di/dt) and R/λ observable.
        const auto iq_ref_at = [](int k) { return ((k / 80) % 2) ? 6.0 : 2.0; };
        const auto id_ref_at = [](int k) { return ((k / 120) % 2) ? -3.0 : 0.0; };

        // --- Convergence run on the coarse (deadbeat-matched) plant so the estimator
        // converges to the true parameters. Record the model each step from the closure. ---
        motor::AdaptivePredictiveCurrentController<double> adaptive{make_cfg()};

        std::vector<double> tp, rR, rLd, rlam;

        int           k = 0;
        constexpr int plot_until = 900; // ~0.09 s of transient
        auto          record_controller = [&](const ColVec<2, double>& y) {
            const DirectQuadrature<double> ref{.d = id_ref_at(k), .q = iq_ref_at(k)};
            const auto                     c = adaptive.control(ref, {y[0], y[1]}, omega_e, Vdc, dt);
            if (k < plot_until && (k % 2 == 0)) {
                const auto m = adaptive.model();
                tp.push_back(k * dt);
                rR.push_back(m.R / motor_true.R);
                rLd.push_back(m.Ldq.d / motor_true.Ldq.d);
                rlam.push_back(m.lambda / motor_true.lambda);
            }
            ++k;
            return ColVec<2, double>{c.Vdq.d, c.Vdq.q};
        };

        const auto res = sim::simulate_discrete_nonlinear<2, 2, 2, double>(
            dq_map(motor_true, omega_e, dt), measure_currents, record_controller, ColVec<2, double>{}, dt, 2500
        );

        double max_iq = 0.0;
        for (const auto& xk : res.x) {
            max_iq = std::max(max_iq, std::abs(xk[1]));
        }

        const auto m = adaptive.model();
        CHECK(m.R == doctest::Approx(motor_true.R).epsilon(0.1));
        CHECK(m.Ldq.d == doctest::Approx(motor_true.Ldq.d).epsilon(0.15));
        CHECK(m.Ldq.q == doctest::Approx(motor_true.Ldq.q).epsilon(0.15));
        CHECK(m.lambda == doctest::Approx(motor_true.lambda).epsilon(0.15));
        CHECK(max_iq < 50.0); // never blew up despite the initial +80% L error

        plotcheck::xy("predictive_param_convergence.html", "Online KF parameter estimates / true (transient, → 1)", "time (s)", "estimate / true", {{.name = "R", .x = tp, .y = rR}, {.name = "Ld", .x = tp, .y = rLd}, {.name = "lambda", .x = tp, .y = rlam}});

        // --- Adaptive vs fixed-wrong on the continuous RK4 plant: the payoff plot. ---
        const sim::FixedStepSolver fine{sim::RK4<2, double>{}, dt / 10.0};
        const auto                 ref_at = [&](double t) {
            const int kk = static_cast<int>(std::lround(t / dt));
            return DirectQuadrature<double>{.d = id_ref_at(kk), .q = iq_ref_at(kk)};
        };
        const auto iq_trace = [&](auto& any_ctrl) {
            auto controller = [&](double t, const ColVec<2, double>& y) {
                const auto c = any_ctrl.control(ref_at(t), {y[0], y[1]}, omega_e, Vdc, dt);
                return ColVec<2, double>{c.Vdq.d, c.Vdq.q};
            };
            const auto res2 = sim::simulate_sampled<2, 2, 2, double>(
                dq_dynamics(motor_true, omega_e), measure_currents, controller, fine, dt, ColVec<2, double>{}, {0.0, 600 * dt}
            );
            std::vector<double> ts, iqs;
            for (size_t j = 0; j < res2.t.size(); ++j) {
                ts.push_back(res2.t[j]);
                iqs.push_back(res2.x[j][1]);
            }
            return std::pair{ts, iqs};
        };

        motor::AdaptivePredictiveCurrentController<double> adaptive2{make_cfg()};
        motor::PredictiveCurrentController<double>         fixed{guess}; // never adapts
        const auto [t_a, iq_a] = iq_trace(adaptive2);
        const auto [t_f, iq_f] = iq_trace(fixed);

        plotcheck::xy("predictive_adaptive_vs_fixed.html", "Adaptive (self-tuning) vs fixed wrong-model deadbeat, same references", "time (s)", "iq (A)", {{.name = "adaptive", .x = t_a, .y = iq_a}, {.name = "fixed wrong model", .x = t_f, .y = iq_f}});
    }
}
