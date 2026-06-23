#include <numbers>

#include "wet/backend.hpp"
#include "wet/controllers/lqg.hpp"
#include "wet/controllers/lqgi.hpp"
#include "wet/controllers/lqi.hpp"
#include "wet/controllers/pr.hpp"
#include "wet/design/riccati.hpp"
#include "wet/design/synthesis.hpp"
#include "wet/math/math.hpp"
#include "wet/matrix/matrix.hpp"
#include "wet/systems/state_space.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wet;

TEST_SUITE("Workflow Glue") {
    TEST_CASE("synthesize_lqg builds design analysis and runtime bundles") {
        StateSpace<2, 1, 1, 2, 1> sys{
            .A = Matrix<2, 2>{{1.0, 0.1}, {0.0, 1.0}},
            .B = Matrix<2, 1>{{0.005}, {0.1}},
            .C = Matrix<1, 2>{{1.0, 0.0}},
            .D = Matrix<1, 1>::zeros(),
            .G = Matrix<2, 2>::identity(),
            .H = Matrix<1, 1>::identity(),
            .Ts = 0.1
        };

        const Matrix<2, 2> Q_lqr = Matrix<2, 2>::identity();
        const Matrix<1, 1> R_lqr{{0.1}};
        const Matrix<2, 2> Q_kf{{0.01, 0.0}, {0.0, 0.01}};
        const Matrix<1, 1> R_kf{{0.1}};

        const auto artifacts = design::synthesize_lqg(sys, Q_lqr, R_lqr, Q_kf, R_kf);

        CHECK(artifacts.success);
        CHECK(artifacts.design.success);
        CHECK(artifacts.models.state_feedback_closed_loop.A(0, 0) != doctest::Approx(sys.A(0, 0)));
        CHECK(artifacts.models.observer_error_dynamics(0, 0) != doctest::Approx(sys.A(0, 0)));

        auto                   runtime = artifacts.runtime;
        const ColVec<1, float> y{1.0f};
        const auto             u = runtime.step(y);
        CHECK(u(0, 0) != doctest::Approx(0.0f));
    }

    TEST_CASE("synthesize_lqg_pr adds SISO PR internal model") {
        StateSpace<2, 1, 1, 2, 1> sys{
            .A = Matrix<2, 2>{{1.0, 0.1}, {0.0, 1.0}},
            .B = Matrix<2, 1>{{0.005}, {0.1}},
            .C = Matrix<1, 2>{{1.0, 0.0}},
            .D = Matrix<1, 1>::zeros(),
            .G = Matrix<2, 2>::identity(),
            .H = Matrix<1, 1>::identity(),
            .Ts = 0.001
        };

        const Matrix<2, 2> Q_lqr = Matrix<2, 2>::identity();
        const Matrix<1, 1> R_lqr{{0.1}};
        const Matrix<2, 2> Q_kf{{0.01, 0.0}, {0.0, 0.01}};
        const Matrix<1, 1> R_kf{{0.1}};

        const auto pr = design::pr(
            0.0,
            10.0,
            2.0 * std::numbers::pi,
            5.0,
            0.001
        );

        const auto artifacts = design::synthesize_lqg_pr(sys, Q_lqr, R_lqr, Q_kf, R_kf, pr);

        CHECK(artifacts.success);
        CHECK(artifacts.runtime_pr.pr_design.Ki == doctest::Approx(10.0f));

        auto       runtime = artifacts.runtime_pr;
        const auto u = runtime.step(1.0f, 0.0f);
        CHECK(u(0, 0) != doctest::Approx(0.0f));
    }

    TEST_CASE("synthesize_lqi builds servo artifacts and runtime bundle") {
        StateSpace<2, 1, 1, 0, 0> sys{
            .A = Matrix<2, 2>{{1.0, 0.1}, {0.0, 1.0}},
            .B = Matrix<2, 1>{{0.005}, {0.1}},
            .C = Matrix<1, 2>{{1.0, 0.0}},
            .D = Matrix<1, 1>::zeros(),
            .Ts = 0.1
        };

        Matrix<3, 3> Q_aug{};
        Q_aug(0, 0) = 1.0;
        Q_aug(1, 1) = 1.0;
        Q_aug(2, 2) = 10.0;
        const Matrix<1, 1> R{{0.1}};

        const auto artifacts = design::synthesize_lqi(sys, Q_aug, R);

        CHECK(artifacts.success);
        CHECK(artifacts.design.success);

        auto                   runtime = artifacts.runtime;
        const ColVec<2, float> x{0.0f, 0.0f};
        const ColVec<1, float> y{0.0f};
        const ColVec<1, float> r{1.0f};
        const auto             u = runtime.step(x, y, r);

        CHECK(u(0, 0) != doctest::Approx(0.0f));
    }

    TEST_CASE("synthesize_lqgi builds servo observer artifacts and runtime bundle") {
        StateSpace<2, 1, 1, 2, 1> sys{
            .A = Matrix<2, 2>{{1.0, 0.1}, {0.0, 1.0}},
            .B = Matrix<2, 1>{{0.005}, {0.1}},
            .C = Matrix<1, 2>{{1.0, 0.0}},
            .D = Matrix<1, 1>::zeros(),
            .G = Matrix<2, 2>::identity(),
            .H = Matrix<1, 1>::identity(),
            .Ts = 0.1
        };

        Matrix<3, 3> Q_aug{};
        Q_aug(0, 0) = 1.0;
        Q_aug(1, 1) = 1.0;
        Q_aug(2, 2) = 10.0;

        const Matrix<1, 1> R{{0.1}};
        const Matrix<2, 2> Q_kf{{0.01, 0.0}, {0.0, 0.01}};
        const Matrix<1, 1> R_kf{{0.1}};

        const auto artifacts = design::synthesize_lqgi(sys, Q_aug, R, Q_kf, R_kf);

        CHECK(artifacts.success);
        CHECK(artifacts.design.success);
        CHECK(artifacts.models.observer_error_dynamics(0, 0) != doctest::Approx(sys.A(0, 0)));

        auto                   runtime = artifacts.runtime;
        const ColVec<1, float> y{0.0f};
        const ColVec<1, float> r{1.0f};
        const auto             u = runtime.step(y, r);

        CHECK(u(0, 0) != doctest::Approx(0.0f));
    }

    TEST_CASE("synthesize_lqi drives steady-state tracking error to zero") {
        StateSpace<2, 1, 1, 0, 0> sys{
            .A = Matrix<2, 2>{{1.0, 0.1}, {0.0, 1.0}},
            .B = Matrix<2, 1>{{0.005}, {0.1}},
            .C = Matrix<1, 2>{{1.0, 0.0}},
            .D = Matrix<1, 1>::zeros(),
            .Ts = 0.1
        };

        Matrix<3, 3> Q_aug{};
        Q_aug(0, 0) = 1.0;
        Q_aug(1, 1) = 1.0;
        Q_aug(2, 2) = 10.0;
        const Matrix<1, 1> R{{0.1}};

        const auto artifacts = design::synthesize_lqi(sys, Q_aug, R);
        REQUIRE(artifacts.success);

        // Closed-loop simulation against the (linear, discrete) design plant.
        // The integral state uses a unit discrete integrator, so the runtime
        // must accumulate xi += (r - y) with no Ts scaling; if it did not,
        // integral action would be ~Ts-times too weak and y would never reach r.
        const auto Af = sys.A.as<float>();
        const auto Bf = sys.B.as<float>();
        const auto Cf = sys.C.as<float>();

        auto             runtime = artifacts.runtime;
        ColVec<2, float> x{0.0f, 0.0f};
        const float      r = 1.0f;
        float            y = 0.0f;

        for (int k = 0; k < 5000; ++k) {
            y = (Cf * x)(0, 0);
            const auto u = runtime.step(x, ColVec<1, float>{y}, ColVec<1, float>{r});
            x = ColVec<2, float>(Af * x + Bf * u);
        }

        CHECK(y == doctest::Approx(1.0f).epsilon(0.01));
    }

    TEST_CASE("workflow artifacts synthesize at compile time") {
        constexpr auto artifacts = [] {
            StateSpace<2, 1, 1, 0, 0> sys{
                .A = Matrix<2, 2>{{1.0, 0.1}, {0.0, 1.0}},
                .B = Matrix<2, 1>{{0.005}, {0.1}},
                .C = Matrix<1, 2>{{1.0, 0.0}},
                .D = Matrix<1, 1>::zeros(),
                .Ts = 0.1
            };
            Matrix<3, 3> Q_aug{};
            Q_aug(0, 0) = 1.0;
            Q_aug(1, 1) = 1.0;
            Q_aug(2, 2) = 10.0;
            const Matrix<1, 1> R{{0.1}};
            return design::synthesize_lqi(sys, Q_aug, R);
        }();

        static_assert(artifacts.success, "LQI workflow must synthesize at compile time");
        CHECK(artifacts.success);
    }
}
