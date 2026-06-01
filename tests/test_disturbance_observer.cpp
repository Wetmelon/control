#include "wet/estimation/disturbance_observer.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wetmelon::control;
using estimation::DisturbanceObserver;
using estimation::DisturbanceObserverConfig;

// The disturbance observer is a lightweight SISO estimator:
//   innovation = y_measured − y_predicted
//   d_hat[k+1] = (1 − leak)·d_hat[k] + gain·innovation
// with optional innovation deadband and output-magnitude clamp. It is the
// estimator core behind roadmap #4 (DOB control law). These tests cover config
// validation, the recursion's steady state, the deadband, the clamp, and the
// compensate()/reset() runtime surface.
//
// @see Li et al., "Disturbance Observer-Based Control" (CRC Press, 2016).

TEST_SUITE("Disturbance Observer") {
    TEST_CASE("config validation rejects out-of-range parameters") {
        // Valid baseline.
        CHECK(DisturbanceObserverConfig<double>{0.1, 0.0, 0.0, 0.0, false}.valid());

        // gain must be in [0, 1].
        CHECK_FALSE(DisturbanceObserverConfig<double>{-0.1}.valid());
        CHECK_FALSE(DisturbanceObserverConfig<double>{1.5}.valid());

        // leak must be in [0, 1).
        CHECK_FALSE(DisturbanceObserverConfig<double>{0.1, -0.1}.valid());
        CHECK_FALSE(DisturbanceObserverConfig<double>{0.1, 1.0}.valid());

        // negative deadband / magnitude are invalid.
        CHECK_FALSE(DisturbanceObserverConfig<double>{0.1, 0.0, -1.0}.valid());
        CHECK_FALSE(DisturbanceObserverConfig<double>{0.1, 0.0, 0.0, -1.0}.valid());
    }

    TEST_CASE("synthesize reports steady-state gain and rejects bad config") {
        constexpr DisturbanceObserverConfig<double> good{0.2, 0.1, 0.0, 0.0, false};
        constexpr auto                              res = estimation::synthesize_disturbance_observer(good);
        static_assert(res.success);
        // Constant-innovation steady state: d_hat_ss = gain/leak · innovation.
        CHECK(res.steady_state_gain == doctest::Approx(0.2 / 0.1));

        constexpr DisturbanceObserverConfig<double> bad{2.0}; // gain > 1
        constexpr auto                              res_bad = estimation::synthesize_disturbance_observer(bad);
        static_assert(!res_bad.success);
    }

    TEST_CASE("leaky integrator settles at (gain/leak)·innovation") {
        // The recursion d_hat[k+1] = (1−leak)·d_hat[k] + gain·innovation is a
        // leaky integrator: with a constant innovation it settles at
        // (gain/leak)·innovation. Choosing gain = leak gives unity DC gain, so
        // the estimate converges to the innovation itself. (With leak = 0 it is a
        // pure integrator and would diverge on a constant innovation — that is
        // the intended physics, not a settling estimator.)
        DisturbanceObserver<double> dob{DisturbanceObserverConfig<double>{0.25, 0.25, 0.0, 0.0, false}};

        const double d_true = 3.0;
        for (int k = 0; k < 500; ++k) {
            // y_predicted = 0, y_measured = d_true → innovation = d_true each tick.
            REQUIRE(dob.update(0.0, d_true));
        }
        CHECK(dob.state().disturbance_hat == doctest::Approx(d_true).epsilon(1e-3));
    }

    TEST_CASE("compensate subtracts the disturbance estimate from the command") {
        // gain = leak ⇒ unity-DC-gain leaky integrator settles d_hat at the
        // constant innovation (here 2.0).
        DisturbanceObserver<double> dob{DisturbanceObserverConfig<double>{0.5, 0.5, 0.0, 0.0, false}};
        for (int k = 0; k < 200; ++k) {
            REQUIRE(dob.update(0.0, 2.0)); // drive estimate toward +2
        }
        // u_compensated = u_nominal − d_hat ≈ 10 − 2 = 8.
        CHECK(dob.compensate(10.0) == doctest::Approx(8.0).epsilon(1e-2));
    }

    TEST_CASE("innovation deadband freezes the estimate for small errors") {
        DisturbanceObserver<double> dob{DisturbanceObserverConfig<double>{0.5, 0.0, 0.1, 0.0, false}};
        // Innovation of 0.05 is below the 0.1 deadband → estimate stays put.
        for (int k = 0; k < 50; ++k) {
            REQUIRE(dob.update(0.0, 0.05));
        }
        CHECK(dob.state().disturbance_hat == doctest::Approx(0.0));

        // Innovation above the deadband moves the estimate.
        REQUIRE(dob.update(0.0, 1.0));
        CHECK(dob.state().disturbance_hat > 0.0);
    }

    TEST_CASE("magnitude clamp bounds the estimate") {
        DisturbanceObserver<double> dob{DisturbanceObserverConfig<double>{0.5, 0.0, 0.0, 1.0, true}};
        for (int k = 0; k < 200; ++k) {
            REQUIRE(dob.update(0.0, 100.0)); // would blow past the clamp
        }
        CHECK(dob.state().disturbance_hat <= doctest::Approx(1.0));
        CHECK(dob.state().disturbance_hat > 0.0);
    }

    TEST_CASE("reset clears state") {
        DisturbanceObserver<double> dob{DisturbanceObserverConfig<double>{0.5, 0.0, 0.0, 0.0, false}};
        REQUIRE(dob.update(0.0, 5.0));
        REQUIRE(dob.state().disturbance_hat != 0.0);

        dob.reset();
        CHECK(dob.state().disturbance_hat == doctest::Approx(0.0));
        CHECK_FALSE(dob.state().initialized);
    }

    TEST_CASE("constructing from a design result carries the config") {
        constexpr DisturbanceObserverConfig<double> cfg{0.3, 0.05, 0.0, 0.0, false};
        const auto                                  design = estimation::synthesize_disturbance_observer(cfg);
        REQUIRE(design.success);

        DisturbanceObserver<double> dob{design};
        CHECK(dob.valid());
        CHECK(dob.config().gain == doctest::Approx(0.3));
        CHECK(dob.config().leak == doctest::Approx(0.05));
    }

    TEST_CASE("as<float>() converts the design result") {
        constexpr DisturbanceObserverConfig<double> cfg{0.2, 0.1, 0.01, 5.0, true};
        const auto                                  res = estimation::synthesize_disturbance_observer(cfg);
        const auto                                  rf = res.as<float>();
        CHECK(rf.success == res.success);
        CHECK(rf.config.gain == doctest::Approx(0.2));
        CHECK(rf.config.clamp_enabled);
    }
}
