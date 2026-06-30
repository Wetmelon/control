#include <cmath>
#include <vector>

#include "doctest.h"
#include "plot_check.hpp"
#include "wet/motor/field_weakening.hpp"
#include "wet/motor/foc.hpp"
#include "wet/transforms.hpp"

using namespace wet;

TEST_SUITE("Field weakening") {
    // Surface-PM-ish machine: MTPA is id=0, so field weakening is the whole story.
    // λ/Ld = 25 A < i_max ⇒ an infinite-speed-capable drive (can suppress the magnet
    // flux within the current limit), so the voltage stays held across the whole sweep.
    constexpr double               lambda = 0.05;   // [Wb]
    constexpr double               Vdc = 24.0;      // [V]
    constexpr double               i_max = 30.0;    // [A]
    constexpr double               v_margin = 0.95; // regulate to 95% of the voltage circle
    const DirectQuadrature<double> Ldq{.d = 2e-3, .q = 2e-3};

    TEST_CASE("Feedforward id sits on the voltage ellipse") {
        const double iq = 15.0;
        const double Vmax = design::voltage_circle_radius(Vdc);
        const double w = 400.0; // above base speed for this operating point

        const double id = design::field_weakening_id(iq, w, Vmax, Ldq, lambda);
        CHECK(id < 0.0);

        // (Ld·id + λ)² + (Lq·iq)² must equal (Vmax/ω)².
        const double lhs = std::pow(Ldq.d * id + lambda, 2.0) + std::pow(Ldq.q * iq, 2.0);
        const double rhs = std::pow(Vmax / w, 2.0);
        CHECK(lhs == doctest::Approx(rhs));
    }

    TEST_CASE("No weakening below base speed") {
        const double Vmax = design::voltage_circle_radius(Vdc);
        CHECK(design::field_weakening_id(10.0, 50.0, Vmax, Ldq, lambda) == doctest::Approx(0.0));
        CHECK(design::field_weakening_id(10.0, 0.0, Vmax, Ldq, lambda) == doctest::Approx(0.0));
    }

    // Quasi-static speed sweep: at each speed, iterate the FW regulator against a
    // steady-state voltage model V = ω·(-Lq·iq, Ld·id+λ) to a fixed point (this
    // stands in for the closed current loop), then record the operating point.
    auto sweep = [](motor::FwMethod method) {
        const double                        dt = 1e-4;
        const double                        iq_cmd = 12.0;
        const double                        Vlim = v_margin * design::voltage_circle_radius(Vdc);
        motor::FieldWeakeningConfig<double> cfg{.Ldq = Ldq, .lambda = lambda, .i_max = i_max, .v_margin = v_margin, .ki = 1000.0, .method = method};
        motor::FieldWeakening<double>       fw{cfg};
        DirectQuadrature<double>            idq{.d = 0.0, .q = iq_cmd};

        std::vector<double> spd, id_v, iq_v, vmag, vlim_v;
        for (int s = 0; s <= 120; ++s) {
            const double                   w = (600.0 * s) / 120.0;
            const DirectQuadrature<double> base{.d = 0.0, .q = iq_cmd};
            for (int k = 0; k < 600; ++k) { // settle to the fixed point at this speed
                const DirectQuadrature<double> Vdq{.d = -w * Ldq.q * idq.q, .q = w * (Ldq.d * idq.d + lambda)};
                idq = fw.update(base, Vdq, w, Vdc, dt);
            }
            const double v = w * std::hypot(Ldq.q * idq.q, Ldq.d * idq.d + lambda);
            spd.push_back(w);
            id_v.push_back(idq.d);
            iq_v.push_back(idq.q);
            vmag.push_back(v);
            vlim_v.push_back(Vlim);
        }
        return std::tuple{spd, id_v, iq_v, vmag, vlim_v};
    };

    TEST_CASE("Voltage-feedback FW holds the voltage limit above base speed") {
        auto [spd, id_v, iq_v, vmag, vlim_v] = sweep(motor::FwMethod::VoltageFeedback);
        const double Vlim = vlim_v.front();

        for (std::size_t i = 0; i < spd.size(); ++i) {
            CHECK(vmag[i] <= Vlim * 1.02);                             // never exceeds the circle
            if (spd[i] > 250.0 && id_v[i] > -i_max + 0.5) {            // in FW, before current saturation
                CHECK(id_v[i] < 0.0);                                  // weakening engaged
                CHECK(vmag[i] == doctest::Approx(Vlim).epsilon(0.05)); // pinned at the limit
            }
        }
        CHECK(iq_v.back() < iq_v.front()); // torque current given up at high speed

        plotcheck::xy("fw_voltage_feedback.html", "Voltage-feedback field weakening — quasi-static speed sweep", "electrical speed (rad/s)", "current (A) / voltage (V)", {{.name = "id", .x = spd, .y = id_v}, {.name = "iq", .x = spd, .y = iq_v}, {.name = "|V|", .x = spd, .y = vmag}, {.name = "Vlim", .x = spd, .y = vlim_v}});
    }

    TEST_CASE("Feedforward FW matches feedback and holds the limit") {
        auto [spd, id_v, iq_v, vmag, vlim_v] = sweep(motor::FwMethod::Feedforward);
        const double Vlim = vlim_v.front();

        for (std::size_t i = 0; i < spd.size(); ++i) {
            CHECK(vmag[i] <= Vlim * 1.02);
            if (spd[i] > 250.0 && id_v[i] > -i_max + 0.5) {
                CHECK(id_v[i] < 0.0);
                CHECK(vmag[i] == doctest::Approx(Vlim).epsilon(0.05));
            }
        }

        // Overlay both methods' d-axis trajectories for visual comparison.
        auto [spd_fb, id_fb, iq_fb, vm_fb, vl_fb] = sweep(motor::FwMethod::VoltageFeedback);
        plotcheck::xy("fw_feedforward_vs_feedback.html", "Field weakening: feedforward vs voltage-feedback (id vs speed)", "electrical speed (rad/s)", "id (A)", {{.name = "id feedforward", .x = spd, .y = id_v}, {.name = "id voltage-feedback", .x = spd_fb, .y = id_fb}});
    }
}
