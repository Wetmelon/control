#include "doctest.h"
#include "wet/backend.hpp"
#include "wet/controllers/lqg.hpp"
#include "wet/controllers/lqr.hpp"
#include "wet/design/riccati.hpp"
#include "wet/estimation/kalman.hpp"
#include "wet/math/math.hpp"
#include "wet/matrix/matrix.hpp"
#include "wet/systems/state_space.hpp"

using namespace wet;

// LQG = LQR (state feedback) + Kalman filter (state estimation), tied together
// by the separation principle. design::discrete_lqg() was previously only covered
// indirectly via test_design/test_api; this exercises the full Tier 2 → Tier 3
// path including the runtime LQG controller built straight from the result
// (which feeds the Kalman filter from result.kalman.sys — the constructor
// dogfooded by lqg.hpp).
//
// @see "Optimal Control" (Anderson & Moore, 1990), §8 (separation principle).

namespace {
// Discrete double integrator at Ts = 0.1 s: x = [position, velocity].
constexpr StateSpace<2, 1, 1, 2, 1> make_plant() {
    return StateSpace<2, 1, 1, 2, 1>{
        Matrix<2, 2>{{1.0, 0.1}, {0.0, 1.0}}, // A
        Matrix<2, 1>{{0.005}, {0.1}},         // B
        Matrix<1, 2>{{1.0, 0.0}},             // C: measure position only
        Matrix<1, 1>::zeros(),                // D
        Matrix<2, 2>::identity(),             // G: process noise on both states
        Matrix<1, 1>::identity()              // H: measurement noise on output
    };
}
} // namespace

TEST_SUITE("LQG") {
    TEST_CASE("design succeeds and reports stable closed loop") {
        constexpr auto         sys = make_plant();
        constexpr auto         Q_lqr = Matrix<2, 2>::identity();
        constexpr Matrix<1, 1> R_lqr{{0.1}};
        constexpr Matrix<2, 2> Q_kf{{0.01, 0.0}, {0.0, 0.01}};
        constexpr Matrix<1, 1> R_kf{{0.1}};

        constexpr auto result = design::discrete_lqg(sys, Q_lqr, R_lqr, Q_kf, R_kf);
        static_assert(result.success, "LQG design must converge at compile time");
        static_assert(result.lqr.success);
        static_assert(result.kalman.success);

        // LQR closed loop A − BK must be inside the unit circle.
        CHECK(result.lqr.is_stable());
        // Kalman error covariance is positive on the diagonal.
        CHECK(result.kalman.P(0, 0) > 0.0);
        CHECK(result.kalman.P(1, 1) > 0.0);
    }

    TEST_CASE("runtime LQG regulates output-feedback plant to zero") {
        // The controller only sees the measured position y = x[0]; it must
        // reconstruct velocity through the Kalman filter and still drive the
        // full state to zero. This is the separation principle working end to end.
        const auto         sys = make_plant();
        const auto         Q_lqr = Matrix<2, 2>::identity();
        const Matrix<1, 1> R_lqr{{0.1}};
        const Matrix<2, 2> Q_kf{{0.01, 0.0}, {0.0, 0.01}};
        const Matrix<1, 1> R_kf{{0.1}};

        const auto result = design::discrete_lqg(sys, Q_lqr, R_lqr, Q_kf, R_kf);
        REQUIRE(result.success);

        // Built straight from the result — exercises kf(result.kalman).
        LQG<2, 1, 1, 2, 1> controller{result};

        ColVec<2> x{{1.0}, {0.0}}; // start displaced by 1 m
        for (int k = 0; k < 200; ++k) {
            const ColVec<1> y{{x[0]}}; // measure position
            controller.update(y);      // Kalman correct
            const ColVec<1> u = controller.control();
            x = sys.A * x + sys.B * u; // advance true plant
            controller.predict(u);     // Kalman predict
        }

        CHECK(x[0] == doctest::Approx(0.0).epsilon(0.02)); // position regulated
        CHECK(x[1] == doctest::Approx(0.0).epsilon(0.02)); // velocity regulated
    }

    TEST_CASE("LQGResult::as<float>() preserves the design") {
        const auto sys = make_plant();
        const auto result = design::discrete_lqg(
            sys, Matrix<2, 2>::identity(), Matrix<1, 1>{{0.1}},
            Matrix<2, 2>{{0.01, 0.0}, {0.0, 0.01}}, Matrix<1, 1>{{0.1}}
        );
        REQUIRE(result.success);

        const auto rf = result.as<float>();
        CHECK(rf.success);
        CHECK(rf.lqr.K(0, 0) == doctest::Approx(static_cast<float>(result.lqr.K(0, 0))));
        CHECK(rf.kalman.L(0, 0) == doctest::Approx(static_cast<float>(result.kalman.L(0, 0))));
    }

    TEST_CASE("lqgreg combines independent Kalman and LQR designs") {
        const auto sys = make_plant();
        const auto kf = design::kalman(sys, Matrix<2, 2>{{0.01, 0.0}, {0.0, 0.01}}, Matrix<1, 1>{{0.1}});
        const auto lqr = design::discrete_lqr(sys.A, sys.B, Matrix<2, 2>::identity(), Matrix<1, 1>{{0.1}});
        REQUIRE(kf.success);
        REQUIRE(lqr.success);

        const auto combined = design::lqg_from_parts(kf, lqr);
        CHECK(combined.success);
        // The combined result reuses the inputs verbatim.
        CHECK(combined.lqr.K(0, 0) == lqr.K(0, 0));
        CHECK(combined.kalman.L(0, 0) == kf.L(0, 0));
    }

    TEST_CASE("LQGResult::to_ss regulator state-space stabilizes the plant") {
        // Validates the prediction-form realization (steady-state L) by closing the
        // loop: y -> compensator -> u -> plant. A correct realization regulates the
        // plant to zero from a displaced start.
        const auto sys = make_plant();
        const auto result = design::discrete_lqg(
            sys, Matrix<2, 2>::identity(), Matrix<1, 1>{{0.1}},
            Matrix<2, 2>{{0.01, 0.0}, {0.0, 0.01}}, Matrix<1, 1>{{0.1}}
        );
        REQUIRE(result.success);

        const auto ss = result.to_ss();            // StateSpace<2,1,1>: in y, out u, state x̂
        CHECK(ss.D(0, 0) == doctest::Approx(0.0)); // strictly proper compensator
        CHECK(ss.Ts == doctest::Approx(sys.Ts));

        ColVec<2> xc{{0.0}, {0.0}}; // compensator (estimator) state
        ColVec<2> xp{{1.0}, {0.0}}; // true plant, displaced
        for (int k = 0; k < 200; ++k) {
            const ColVec<1> y{{xp[0]}};
            const ColVec<1> u = ColVec<1>(ss.C * xc + ss.D * y);
            xp = sys.A * xp + sys.B * u;
            xc = ColVec<2>(ss.A * xc + ss.B * y);
        }
        CHECK(xp[0] == doctest::Approx(0.0).epsilon(0.02));
        CHECK(xp[1] == doctest::Approx(0.0).epsilon(0.02));
    }
}
