#include "doctest.h"
#include "wet/controllers/lqgi.hpp"
#include "wet/matrix/matrix.hpp"
#include "wet/systems/state_space.hpp"

using namespace wetmelon::control;

// LQGI = LQI (state feedback with integral action) + Kalman estimator. The
// integral state drives steady-state output error to zero for constant
// references/disturbances. design::lqgtrack() was only covered indirectly;
// this exercises the design and the runtime tracking loop, including the
// integrator augmentation [x; xi].
//
// @see "Optimal Control" (Anderson & Moore, 1990); integral augmentation is the
//      standard servo/Type-1 construction.

namespace {
// Discrete double integrator at Ts = 0.1 s, measuring position.
constexpr StateSpace<2, 1, 1, 2, 1> make_plant() {
    return StateSpace<2, 1, 1, 2, 1>{
        Matrix<2, 2>{{1.0, 0.1}, {0.0, 1.0}},
        Matrix<2, 1>{{0.005}, {0.1}},
        Matrix<1, 2>{{1.0, 0.0}},
        Matrix<1, 1>::zeros(),
        Matrix<2, 2>::identity(),
        Matrix<1, 1>::identity()
    };
}

// Augmented-state cost: penalize position, velocity, and the integral state.
constexpr Matrix<3, 3> make_Q_aug() {
    Matrix<3, 3> Q{};
    Q(0, 0) = 1.0;  // position
    Q(1, 1) = 1.0;  // velocity
    Q(2, 2) = 10.0; // integral of tracking error — weight it to pull SS error to 0
    return Q;
}
} // namespace

TEST_SUITE("LQGI") {
    TEST_CASE("design succeeds with integral augmentation") {
        constexpr auto         sys = make_plant();
        constexpr auto         Q_aug = make_Q_aug();
        constexpr Matrix<1, 1> R{{0.1}};
        constexpr Matrix<2, 2> Q_kf{{0.01, 0.0}, {0.0, 0.01}};
        constexpr Matrix<1, 1> R_kf{{0.1}};

        constexpr auto result = design::lqgtrack(sys, Q_aug, R, Q_kf, R_kf);
        static_assert(result.success);
        static_assert(result.lqi.success);
        static_assert(result.kalman.success);

        // Gain has a column per augmented state: [x(2) ; xi(1)] = 3.
        CHECK(result.lqi.K(0, 0) != 0.0);
        CHECK(result.lqi.K(0, 2) != 0.0); // integral gain is non-trivial
    }

    TEST_CASE("runtime LQGI tracks a constant reference with zero steady-state error") {
        const auto         sys = make_plant();
        const auto         Q_aug = make_Q_aug();
        const Matrix<1, 1> R{{0.1}};
        const Matrix<2, 2> Q_kf{{0.01, 0.0}, {0.0, 0.01}};
        const Matrix<1, 1> R_kf{{0.1}};

        const auto result = design::lqgtrack(sys, Q_aug, R, Q_kf, R_kf);
        REQUIRE(result.success);

        LQGI<2, 1, 1, 2, 1> controller{result}; // exercises kf(result.kalman)

        const double r = 1.0;         // position setpoint
        ColVec<2>    x{{0.0}, {0.0}}; // true plant state
        double       xi = 0.0;        // integral of (r - y)

        for (int k = 0; k < 600; ++k) {
            const ColVec<1> y{{x[0]}};
            controller.update(y);

            // Augmented state fed to LQI is [x_hat; xi].
            const auto      xhat = controller.kf.state();
            const ColVec<3> x_aug{{xhat[0]}, {xhat[1]}, {xi}};
            const ColVec<1> u = controller.control(x_aug);

            x = sys.A * x + sys.B * u;
            controller.predict(u);
            xi += (r - x[0]); // accumulate tracking error (matches -C augmentation sign convention)
        }

        CHECK(x[0] == doctest::Approx(r).epsilon(0.02)); // output reaches the reference
    }

    TEST_CASE("LQGIResult::as<float>() preserves the design") {
        const auto sys = make_plant();
        const auto result = design::lqgtrack(sys, make_Q_aug(), Matrix<1, 1>{{0.1}}, Matrix<2, 2>{{0.01, 0.0}, {0.0, 0.01}}, Matrix<1, 1>{{0.1}});
        REQUIRE(result.success);

        const auto rf = result.as<float>();
        CHECK(rf.success);
        CHECK(rf.lqi.K(0, 0) == doctest::Approx(static_cast<float>(result.lqi.K(0, 0))));
        CHECK(rf.kalman.L(0, 0) == doctest::Approx(static_cast<float>(result.kalman.L(0, 0))));
    }
}
