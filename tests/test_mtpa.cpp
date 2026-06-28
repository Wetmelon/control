#include <cmath>
#include <vector>

#include "plot_check.hpp"
#include "wet/motor/foc.hpp"
#include "wet/motor/mtpa.hpp"
#include "wet/transforms.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wet;

namespace {

// Amplitude-invariant electromagnetic torque from a dq operating point.
template<typename T>
T torque_of(const DirectQuadrature<T>& Idq, T lambda, const DirectQuadrature<T>& Ldq, T p) {
    return T{1.5} * p * ((lambda * Idq.q) + ((Ldq.d - Ldq.q) * Idq.d * Idq.q));
}

} // namespace

TEST_SUITE("MTPA current reference") {
    constexpr double p = 4.0;
    constexpr double lambda = 0.1;

    TEST_CASE("Surface PMSM reduces to id = 0") {
        const DirectQuadrature<double> Ldq{.d = 2e-3, .q = 2e-3}; // Ld == Lq
        const double                   Te = 1.5;

        const auto idq = design::mtpa_reference(Te, lambda, Ldq, p);

        CHECK(idq.d == doctest::Approx(0.0));
        CHECK(idq.q == doctest::Approx(design::iq_from_torque(Te, p, lambda)));
        CHECK(torque_of(idq, lambda, Ldq, p) == doctest::Approx(Te));
    }

    TEST_CASE("Salient IPMSM hits the commanded torque on the locus") {
        const DirectQuadrature<double> Ldq{.d = 2e-3, .q = 5e-3}; // Lq > Ld
        const double                   Te = 2.0;

        const auto idq = design::mtpa_reference(Te, lambda, Ldq, p);

        // 1. Produces the requested torque.
        CHECK(torque_of(idq, lambda, Ldq, p) == doctest::Approx(Te));

        // 2. Uses reluctance torque: id is negative.
        CHECK(idq.d < 0.0);

        // 3. Sits on the MTPA locus: (Ld-Lq) id² + λ id - (Ld-Lq) iq² = 0.
        const double Lsal = Ldq.d - Ldq.q;
        const double locus = (Lsal * idq.d * idq.d) + (lambda * idq.d) - (Lsal * idq.q * idq.q);
        CHECK(locus == doctest::Approx(0.0).epsilon(1e-9));
    }

    TEST_CASE("MTPA uses less current than id = 0 for the same torque") {
        const DirectQuadrature<double> Ldq{.d = 2e-3, .q = 6e-3};
        const double                   Te = 2.0;

        const auto   idq = design::mtpa_reference(Te, lambda, Ldq, p);
        const double mtpa_mag = std::hypot(idq.d, idq.q);

        // The naive id=0 point delivering the same torque needs iq = Te/Kt.
        const double iq0 = design::iq_from_torque(Te, p, lambda);
        CHECK(mtpa_mag < std::abs(iq0));
    }

    TEST_CASE("Negative torque is symmetric (iq flips, id stays negative)") {
        const DirectQuadrature<double> Ldq{.d = 2e-3, .q = 5e-3};
        const double                   Te = 2.0;

        const auto pos = design::mtpa_reference(Te, lambda, Ldq, p);
        const auto neg = design::mtpa_reference(-Te, lambda, Ldq, p);

        CHECK(neg.q == doctest::Approx(-pos.q));
        CHECK(neg.d == doctest::Approx(pos.d));
        CHECK(torque_of(neg, lambda, Ldq, p) == doctest::Approx(-Te));
    }

    TEST_CASE("Runtime block matches the design law") {
        const DirectQuadrature<float>     Ldq{.d = 2e-3f, .q = 5e-3f};
        const motor::MtpaReference<float> ref{Ldq, 0.1f, 4.0f};

        const auto idq = ref(1.5f);
        const auto expect = design::mtpa_reference(1.5f, 0.1f, Ldq, 4.0f);
        CHECK(idq.d == doctest::Approx(expect.d));
        CHECK(idq.q == doctest::Approx(expect.q));
    }

    TEST_CASE("Plot: MTPA trajectory vs id=0 over a torque sweep") {
        const DirectQuadrature<double> Ldq{.d = 2e-3, .q = 6e-3}; // salient IPMSM
        const double                   Te_max = 3.0;

        std::vector<double> id_mtpa, iq_mtpa, id_zero, iq_zero;
        std::vector<double> te, mag_mtpa, mag_zero;
        for (int i = 1; i <= 60; ++i) {
            const double Te = (Te_max * i) / 60.0;
            const auto   m = design::mtpa_reference(Te, lambda, Ldq, p);
            const double iq0 = design::iq_from_torque(Te, p, lambda); // id = 0 baseline

            id_mtpa.push_back(m.d);
            iq_mtpa.push_back(m.q);
            id_zero.push_back(0.0);
            iq_zero.push_back(iq0);

            te.push_back(Te);
            mag_mtpa.push_back(std::hypot(m.d, m.q));
            mag_zero.push_back(std::abs(iq0));

            // MTPA must never need more current than the id=0 point for the same torque.
            CHECK(std::hypot(m.d, m.q) <= std::abs(iq0) + 1e-9);
        }

        plotcheck::xy("mtpa_trajectory.html", "MTPA vs id=0 operating points (salient IPMSM, Ld=2mH Lq=6mH)", "id (A)", "iq (A)", {{.name = "MTPA locus", .x = id_mtpa, .y = iq_mtpa, .markers = true}, {.name = "id = 0", .x = id_zero, .y = iq_zero, .markers = true}});

        plotcheck::xy("mtpa_current_savings.html", "Stator current vs torque — MTPA harvests reluctance torque", "torque (Nm)", "|I| (A)", {{.name = "MTPA |I|", .x = te, .y = mag_mtpa}, {.name = "id=0 |I|", .x = te, .y = mag_zero}});
    }
}
