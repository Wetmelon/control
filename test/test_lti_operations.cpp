#include "../source/LTI.hpp"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

using namespace control;

TEST_CASE("LTI System Arithmetic Operations") {
    // Create test systems
    const StateSpace plant{
        Matrix::Constant(1, 1, -1.0),  // A
        Matrix::Constant(1, 1, 1.0),   // B
        Matrix::Constant(1, 1, 1.0),   // C
        Matrix::Constant(1, 1, 0.0)    // D
    };

    const StateSpace controller{
        Matrix::Zero(0, 0),          // A - No states (pure gain)
        Matrix::Zero(0, 1),          // B
        Matrix::Zero(1, 0),          // C
        Matrix::Constant(1, 1, 2.0)  // D
    };

    const StateSpace sensor{
        Matrix::Zero(0, 0),          // A
        Matrix::Zero(0, 1),          // B
        Matrix::Zero(1, 0),          // C
        Matrix::Constant(1, 1, 1.0)  // D
    };

    SUBCASE("Series Connection: Controller * Plant") {
        // Open-loop: L(s) = C(s) * G(s) = 2/(s+1)
        auto open_loop = controller * plant;

        // Check dimensions
        CHECK(open_loop.A.rows() == 1);
        CHECK(open_loop.A.cols() == 1);
        CHECK(open_loop.B.rows() == 1);
        CHECK(open_loop.B.cols() == 1);
        CHECK(open_loop.C.rows() == 1);
        CHECK(open_loop.C.cols() == 1);
        CHECK(open_loop.D.rows() == 1);
        CHECK(open_loop.D.cols() == 1);

        // Check values (2/(s+1) should have A=-1, B=2, C=1, D=0)
        CHECK(open_loop.A(0, 0) == doctest::Approx(-1.0));
        CHECK(open_loop.B(0, 0) == doctest::Approx(2.0));
        CHECK(open_loop.C(0, 0) == doctest::Approx(1.0));
        CHECK(open_loop.D(0, 0) == doctest::Approx(0.0));
    }

    SUBCASE("Parallel Connection: Plant + Plant") {
        // Sum: G1(s) + G2(s) should give 2/(s+1)
        auto parallel_sum = plant + plant;

        // Check dimensions (2 states from 2 systems)
        CHECK(parallel_sum.A.rows() == 2);
        CHECK(parallel_sum.A.cols() == 2);
        CHECK(parallel_sum.B.rows() == 2);
        CHECK(parallel_sum.C.cols() == 2);

        // Check A is block diagonal
        CHECK(parallel_sum.A(0, 0) == doctest::Approx(-1.0));
        CHECK(parallel_sum.A(1, 1) == doctest::Approx(-1.0));
        CHECK(parallel_sum.A(0, 1) == doctest::Approx(0.0));
        CHECK(parallel_sum.A(1, 0) == doctest::Approx(0.0));

        // Check C combines both outputs
        CHECK(parallel_sum.C(0, 0) == doctest::Approx(1.0));
        CHECK(parallel_sum.C(0, 1) == doctest::Approx(1.0));

        // D should be sum (0 + 0 = 0)
        CHECK(parallel_sum.D(0, 0) == doctest::Approx(0.0));
    }

    SUBCASE("Parallel Subtraction: Plant - Plant") {
        // Difference: G1(s) - G2(s) should give 0
        auto parallel_diff = plant - plant;

        // Check dimensions
        CHECK(parallel_diff.A.rows() == 2);
        CHECK(parallel_diff.A.cols() == 2);

        // Check A is block diagonal
        CHECK(parallel_diff.A(0, 0) == doctest::Approx(-1.0));
        CHECK(parallel_diff.A(1, 1) == doctest::Approx(-1.0));
        CHECK(parallel_diff.A(0, 1) == doctest::Approx(0.0));
        CHECK(parallel_diff.A(1, 0) == doctest::Approx(0.0));

        // Check C has negation on second system
        CHECK(parallel_diff.C(0, 0) == doctest::Approx(1.0));
        CHECK(parallel_diff.C(0, 1) == doctest::Approx(-1.0));

        // D should be difference (0 - 0 = 0)
        CHECK(parallel_diff.D(0, 0) == doctest::Approx(0.0));
    }

    SUBCASE("Negative Feedback: Unity Feedback on Open Loop") {
        // First create open loop
        auto open_loop = controller * plant;

        // Closed-loop with unity feedback: T(s) = L(s) / (1 + L(s))
        // For L(s) = 2/(s+1), T(s) = 2/(s+3)
        auto closed_loop = feedback(open_loop, sensor, -1);

        // Check dimensions (1 state from open_loop, 0 from sensor)
        CHECK(closed_loop.A.rows() == 1);
        CHECK(closed_loop.A.cols() == 1);

        // Check values for 2/(s+3): A=-3, B=2, C=1, D=0
        CHECK(closed_loop.A(0, 0) == doctest::Approx(-3.0));
        CHECK(closed_loop.B(0, 0) == doctest::Approx(2.0));
        CHECK(closed_loop.C(0, 0) == doctest::Approx(1.0));
        CHECK(closed_loop.D(0, 0) == doctest::Approx(0.0));
    }

    SUBCASE("Complete Control System: Feedback(Controller * Plant, Sensor)") {
        // T(s) = C(s)*G(s) / (1 + C(s)*G(s)*H(s))
        auto control_system = feedback(controller * plant, sensor, -1);

        // Check dimensions
        CHECK(control_system.A.rows() == 1);
        CHECK(control_system.A.cols() == 1);

        // Should be same as previous test (2/(s+3))
        CHECK(control_system.A(0, 0) == doctest::Approx(-3.0));
        CHECK(control_system.B(0, 0) == doctest::Approx(2.0));
        CHECK(control_system.C(0, 0) == doctest::Approx(1.0));
        CHECK(control_system.D(0, 0) == doctest::Approx(0.0));

        // Test step response properties
        auto step_resp = control_system.step(0.0, 5.0);
        CHECK(step_resp.time.size() > 0);
        CHECK(step_resp.output.size() == step_resp.time.size());

        // Check steady-state is approximately 2/3
        CHECK(step_resp.output.back() == doctest::Approx(2.0 / 3.0).epsilon(0.01));
    }
}

TEST_CASE("LTI System Type Safety") {
    const StateSpace continuous{
        Matrix::Constant(1, 1, -1.0),
        Matrix::Constant(1, 1, 1.0),
        Matrix::Constant(1, 1, 1.0),
        Matrix::Constant(1, 1, 0.0)};

    const DiscreteStateSpace discrete{
        Matrix::Constant(1, 1, 0.9),
        Matrix::Constant(1, 1, 0.1),
        Matrix::Constant(1, 1, 1.0),
        Matrix::Constant(1, 1, 0.0),
        0.1  // Ts
    };

    SUBCASE("Same type operations compile") {
        // These should compile fine
        auto result1 = continuous * continuous;
        auto result2 = continuous + continuous;
        auto result3 = continuous - continuous;
        auto result4 = feedback(continuous, continuous);

        CHECK(result1.A.rows() > 0);  // Just verify it compiled
        CHECK(result2.A.rows() > 0);
        CHECK(result3.A.rows() > 0);
        CHECK(result4.A.rows() > 0);
    }

    SUBCASE("Discrete type operations compile") {
        // These should compile fine
        auto result1 = discrete * discrete;
        auto result2 = discrete + discrete;
        auto result3 = discrete - discrete;
        auto result4 = feedback(discrete, discrete);

        CHECK(result1.A.rows() > 0);  // Just verify it compiled
        CHECK(result2.A.rows() > 0);
        CHECK(result3.A.rows() > 0);
        CHECK(result4.A.rows() > 0);
    }

    // Note: Mixed type operations (continuous * discrete) should fail at compile time
    // due to static_assert, so we can't test them here
}
