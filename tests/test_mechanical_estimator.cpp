
#include <cstdint>

#include "doctest.h"
#include "wet/matrix/colvec.hpp"
#include "wet/motor/mechanical_estimator.hpp"
#include "wet/systems/discretization.hpp"

using namespace wet;

namespace {
constexpr double J = 2e-4;  // kg·m²
constexpr double b = 1e-3;  // Nm·s
constexpr double Kt = 0.05; // Nm/A
constexpr double Ts = 1e-4; // 10 kHz predict rate

// Deterministic small-amplitude pseudo-noise in [-amp, amp].
struct Noise {
    std::uint32_t s{88172645};
    double        operator()(double amp) {
        s = (s * 1664525u) + 1013904223u;
        return ((static_cast<double>(s >> 8) / static_cast<double>(1u << 24)) - 0.5) * 2.0 * amp;
    }
};
} // namespace

TEST_SUITE("Mechanical Estimator") {

    TEST_CASE("predict-only follows the open-loop discretized model") {
        const auto truth = discretize(design::rotational_load_ss(J, b, Kt), Ts, DiscretizationMethod::ZOH);

        motor::MechanicalEstimator<double> est{motor::MechanicalEstimatorConfig<double>{.J = J, .b = b, .Kt = Kt, .Ts = Ts}};

        ColVec<3, double> x{}; // [theta, omega, tau_load], zero load
        const double      iq = 1.5;
        for (int k = 0; k < 500; ++k) {
            x = truth.A * x + truth.B * ColVec<1, double>{iq};
            est.predict(iq);
        }
        // No measurement noise, matched model -> estimator tracks the model exactly.
        CHECK(est.theta() == doctest::Approx(x[0]).epsilon(1e-9));
        CHECK(est.omega() == doctest::Approx(x[1]).epsilon(1e-9));
    }

    TEST_CASE("estimates speed and an unknown load torque from noisy angle updates") {
        const auto truth = discretize(design::rotational_load_ss(J, b, Kt), Ts, DiscretizationMethod::ZOH);

        motor::MechanicalEstimator<double> est{motor::MechanicalEstimatorConfig<double>{
            .J = J,
            .b = b,
            .Kt = Kt,
            .Ts = Ts,
            .r_encoder = 1e-6
        }};

        ColVec<3, double> x{};
        x[2] = 0.4;            // true (unknown to the estimator) load torque [Nm]
        const double iq = 3.0; // constant current command
        Noise        noise;

        for (int k = 0; k < 40000; ++k) { // 4 s
            x = truth.A * x + truth.B * ColVec<1, double>{iq};
            est.predict(iq);
            if (k % 10 == 0) { // angle update at 1/10 the predict rate (multirate)
                est.update_encoder(x[0] + noise(1e-3));
            }
        }

        CHECK(est.theta() == doctest::Approx(x[0]).epsilon(1e-3));
        CHECK(est.omega() == doctest::Approx(x[1]).epsilon(1e-2));
        CHECK(est.load_torque() == doctest::Approx(0.4).epsilon(0.05)); // load recovered within 5%
    }

    TEST_CASE("sensorless channel uses its own (higher) measurement noise") {
        const auto truth = discretize(design::rotational_load_ss(J, b, Kt), Ts, DiscretizationMethod::ZOH);

        motor::MechanicalEstimator<double> est{motor::MechanicalEstimatorConfig<double>{
            .J = J,
            .b = b,
            .Kt = Kt,
            .Ts = Ts,
            .r_sensorless = 1e-4
        }};

        ColVec<3, double> x{};
        const double      iq = 2.0;
        Noise             noise;
        for (int k = 0; k < 40000; ++k) {
            x = truth.A * x + truth.B * ColVec<1, double>{iq};
            est.predict(iq);
            if (k % 10 == 0) {
                est.update_sensorless(x[0] + noise(1e-2)); // noisier sensorless angle
            }
        }
        CHECK(est.omega() == doctest::Approx(x[1]).epsilon(2e-2));
    }

    TEST_CASE("load accelerometer observes load torque directly") {
        const auto truth = discretize(design::rotational_load_ss(J, b, Kt), Ts, DiscretizationMethod::ZOH);

        motor::MechanicalEstimator<double> est{motor::MechanicalEstimatorConfig<double>{
            .J = J,
            .b = b,
            .Kt = Kt,
            .Ts = Ts,
            .r_accel = 1e-2
        }};

        ColVec<3, double> x{};
        x[2] = 0.3; // true load torque [Nm]
        const double iq = 2.0;
        Noise        noise;
        for (int k = 0; k < 40000; ++k) {
            x = truth.A * x + truth.B * ColVec<1, double>{iq};
            est.predict(iq);
            if (k % 10 == 0) {
                const double alpha = ((Kt * iq) - (b * x[1]) - x[2]) / J; // true load angular accel
                est.update_load_accel(alpha + noise(0.5), iq);
            }
        }
        CHECK(est.load_torque() == doctest::Approx(0.3).epsilon(0.05));
    }

    TEST_CASE("multirate: skipping updates keeps the covariance bounded") {
        motor::MechanicalEstimator<double> est{motor::MechanicalEstimatorConfig<double>{.J = J, .b = b, .Kt = Kt, .Ts = Ts}};
        ColVec<3, double>                  x{};
        const auto                         truth = discretize(design::rotational_load_ss(J, b, Kt), Ts, DiscretizationMethod::ZOH);
        Noise                              noise;

        for (int k = 0; k < 20000; ++k) {
            x = truth.A * x + truth.B * ColVec<1, double>{1.0};
            est.predict(1.0);
            if (k % 100 == 0) { // sparse updates
                est.update_encoder(x[0] + noise(1e-3));
            }
        }
        // Position-state variance stays finite (observable, periodically corrected).
        CHECK(est.covariance()(0, 0) < 1.0);
        CHECK(est.covariance()(0, 0) > 0.0);
    }

    TEST_CASE("reset clears the estimate") {
        motor::MechanicalEstimator<double> est{motor::MechanicalEstimatorConfig<double>{.J = J, .b = b, .Kt = Kt, .Ts = Ts}};
        for (int k = 0; k < 100; ++k) {
            est.predict(5.0);
        }
        CHECK(est.omega() != doctest::Approx(0.0));
        est.reset();
        CHECK(est.theta() == doctest::Approx(0.0));
        CHECK(est.omega() == doctest::Approx(0.0));
        CHECK(est.load_torque() == doctest::Approx(0.0));
    }
}
