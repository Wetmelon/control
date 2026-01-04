#include "../inc/state_space.hpp"
#include "doctest.h"

using namespace wetmelon::control;

TEST_CASE("StateSpace Series Connection") {
    // Simple first-order systems: x' = -a*x + u, y = x
    StateSpace<1, 1, 1> sys1{};
    sys1.A(0, 0) = -1.0;
    sys1.B(0, 0) = 1.0;
    sys1.C(0, 0) = 1.0;
    sys1.D(0, 0) = 0.0;

    StateSpace<1, 1, 1> sys2{};
    sys2.A(0, 0) = -2.0;
    sys2.B(0, 0) = 1.0;
    sys2.C(0, 0) = 1.0;
    sys2.D(0, 0) = 0.0;

    auto result = series(sys1, sys2);

    // Expected dimensions
    CHECK(result.A.rows() == 2);
    CHECK(result.A.cols() == 2);
    CHECK(result.B.rows() == 2);
    CHECK(result.B.cols() == 1);
    CHECK(result.C.rows() == 1);
    CHECK(result.C.cols() == 2);
    CHECK(result.D.rows() == 1);
    CHECK(result.D.cols() == 1);

    // Check A matrix structure
    CHECK(result.A(0, 0) == doctest::Approx(-1.0));
    CHECK(result.A(1, 0) == doctest::Approx(1.0)); // B2*C1 = 1*1
    CHECK(result.A(1, 1) == doctest::Approx(-2.0));

    // Check B matrix
    CHECK(result.B(0, 0) == doctest::Approx(1.0));
    CHECK(result.B(1, 0) == doctest::Approx(0.0)); // B2*D1 = 1*0

    // Check C matrix
    CHECK(result.C(0, 0) == doctest::Approx(0.0)); // D2*C1 = 0*1
    CHECK(result.C(0, 1) == doctest::Approx(1.0)); // C2

    // Check D matrix
    CHECK(result.D(0, 0) == doctest::Approx(0.0)); // D2*D1 = 0*0
}

TEST_CASE("StateSpace Series via Operator*") {
    StateSpace<1, 1, 1> sys1{};
    sys1.A(0, 0) = -1.0;
    sys1.B(0, 0) = 1.0;
    sys1.C(0, 0) = 1.0;
    sys1.D(0, 0) = 0.0;

    StateSpace<1, 1, 1> sys2{};
    sys2.A(0, 0) = -2.0;
    sys2.B(0, 0) = 1.0;
    sys2.C(0, 0) = 1.0;
    sys2.D(0, 0) = 0.0;

    // sys1 * sys2 should be equivalent to series(sys2, sys1)
    auto result = sys1 * sys2;

    CHECK(result.A.rows() == 2);
    CHECK(result.A(0, 0) == doctest::Approx(-2.0));
    CHECK(result.A(1, 0) == doctest::Approx(1.0));
}

TEST_CASE("StateSpace Parallel Connection") {
    StateSpace<1, 1, 1> sys1{};
    sys1.A(0, 0) = -1.0;
    sys1.B(0, 0) = 1.0;
    sys1.C(0, 0) = 2.0;
    sys1.D(0, 0) = 0.5;

    StateSpace<1, 1, 1> sys2{};
    sys2.A(0, 0) = -3.0;
    sys2.B(0, 0) = 1.0;
    sys2.C(0, 0) = 1.0;
    sys2.D(0, 0) = 0.3;

    auto result = parallel(sys1, sys2);

    // Expected dimensions
    CHECK(result.A.rows() == 2);
    CHECK(result.A.cols() == 2);
    CHECK(result.B.rows() == 2);
    CHECK(result.B.cols() == 1);
    CHECK(result.C.rows() == 1);
    CHECK(result.C.cols() == 2);
    CHECK(result.D.rows() == 1);
    CHECK(result.D.cols() == 1);

    // Check A matrix (block diagonal)
    CHECK(result.A(0, 0) == doctest::Approx(-1.0));
    CHECK(result.A(0, 1) == doctest::Approx(0.0));
    CHECK(result.A(1, 0) == doctest::Approx(0.0));
    CHECK(result.A(1, 1) == doctest::Approx(-3.0));

    // Check B matrix (stacked)
    CHECK(result.B(0, 0) == doctest::Approx(1.0));
    CHECK(result.B(1, 0) == doctest::Approx(1.0));

    // Check C matrix (side-by-side)
    CHECK(result.C(0, 0) == doctest::Approx(2.0));
    CHECK(result.C(0, 1) == doctest::Approx(1.0));

    // Check D matrix (summed)
    CHECK(result.D(0, 0) == doctest::Approx(0.8)); // 0.5 + 0.3
}

TEST_CASE("StateSpace Parallel via Operator+") {
    StateSpace<1, 1, 1> sys1{};
    sys1.A(0, 0) = -1.0;
    sys1.B(0, 0) = 1.0;
    sys1.C(0, 0) = 2.0;
    sys1.D(0, 0) = 0.5;

    StateSpace<1, 1, 1> sys2{};
    sys2.A(0, 0) = -3.0;
    sys2.B(0, 0) = 1.0;
    sys2.C(0, 0) = 1.0;
    sys2.D(0, 0) = 0.3;

    auto result = sys1 + sys2;

    CHECK(result.A.rows() == 2);
    CHECK(result.D(0, 0) == doctest::Approx(0.8));
}

TEST_CASE("StateSpace Subtraction Connection") {
    StateSpace<1, 1, 1> sys1{};
    sys1.A(0, 0) = -1.0;
    sys1.B(0, 0) = 1.0;
    sys1.C(0, 0) = 2.0;
    sys1.D(0, 0) = 0.5;

    StateSpace<1, 1, 1> sys2{};
    sys2.A(0, 0) = -3.0;
    sys2.B(0, 0) = 1.0;
    sys2.C(0, 0) = 1.0;
    sys2.D(0, 0) = 0.3;

    auto result = subtract(sys1, sys2);

    // Expected dimensions
    CHECK(result.A.rows() == 2);
    CHECK(result.A.cols() == 2);
    CHECK(result.B.rows() == 2);
    CHECK(result.B.cols() == 1);
    CHECK(result.C.rows() == 1);
    CHECK(result.C.cols() == 2);
    CHECK(result.D.rows() == 1);
    CHECK(result.D.cols() == 1);

    // Check A matrix (block diagonal)
    CHECK(result.A(0, 0) == doctest::Approx(-1.0));
    CHECK(result.A(0, 1) == doctest::Approx(0.0));
    CHECK(result.A(1, 0) == doctest::Approx(0.0));
    CHECK(result.A(1, 1) == doctest::Approx(-3.0));

    // Check B matrix (stacked)
    CHECK(result.B(0, 0) == doctest::Approx(1.0));
    CHECK(result.B(1, 0) == doctest::Approx(1.0));

    // Check C matrix (sys1 minus sys2)
    CHECK(result.C(0, 0) == doctest::Approx(2.0));
    CHECK(result.C(0, 1) == doctest::Approx(-1.0)); // -C2

    // Check D matrix (subtracted)
    CHECK(result.D(0, 0) == doctest::Approx(0.2)); // 0.5 - 0.3
}

TEST_CASE("StateSpace Subtraction via Operator-") {
    StateSpace<1, 1, 1> sys1{};
    sys1.A(0, 0) = -1.0;
    sys1.B(0, 0) = 1.0;
    sys1.C(0, 0) = 2.0;
    sys1.D(0, 0) = 0.5;

    StateSpace<1, 1, 1> sys2{};
    sys2.A(0, 0) = -3.0;
    sys2.B(0, 0) = 1.0;
    sys2.C(0, 0) = 1.0;
    sys2.D(0, 0) = 0.3;

    auto result = sys1 - sys2;

    CHECK(result.A.rows() == 2);
    CHECK(result.C(0, 0) == doctest::Approx(2.0));
    CHECK(result.C(0, 1) == doctest::Approx(-1.0));
    CHECK(result.D(0, 0) == doctest::Approx(0.2));
}

TEST_CASE("StateSpace Negative Feedback") {
    // Simple plant: G(s) = 1/(s+1)
    StateSpace<1, 1, 1> plant{};
    plant.A(0, 0) = -1.0;
    plant.B(0, 0) = 1.0;
    plant.C(0, 0) = 1.0;
    plant.D(0, 0) = 0.0;

    // Simple controller: K (proportional gain)
    StateSpace<1, 1, 1> controller{};
    controller.A(0, 0) = 0.0;
    controller.B(0, 0) = 1.0;
    controller.C(0, 0) = 2.0; // Gain = 2
    controller.D(0, 0) = 0.0;

    auto result = feedback(plant, controller);

    // Expected dimensions
    CHECK(result.A.rows() == 2);
    CHECK(result.A.cols() == 2);
    CHECK(result.B.rows() == 2);
    CHECK(result.B.cols() == 1);
    CHECK(result.C.rows() == 1);
    CHECK(result.C.cols() == 2);
    CHECK(result.D.rows() == 1);
    CHECK(result.D.cols() == 1);

    // Check A matrix
    // A = [A1, -B1*C2; B2*C1, A2 - B2*D1*C2]
    // A = [-1, -1*2; 1*1, 0 - 1*0*2]
    // A = [-1, -2; 1, 0]
    CHECK(result.A(0, 0) == doctest::Approx(-1.0));
    CHECK(result.A(0, 1) == doctest::Approx(-2.0));
    CHECK(result.A(1, 0) == doctest::Approx(1.0));
    CHECK(result.A(1, 1) == doctest::Approx(0.0));

    // Check B matrix
    // B = [B1*B2; 0]
    // B = [1*1; 0]
    CHECK(result.B(0, 0) == doctest::Approx(1.0));
    CHECK(result.B(1, 0) == doctest::Approx(0.0));

    // Check C matrix
    // C = [C1 - D1*C2, -D1*C2]
    // C = [1 - 0*2, -0*2]
    // C = [1, 0]
    CHECK(result.C(0, 0) == doctest::Approx(1.0));
    CHECK(result.C(0, 1) == doctest::Approx(0.0));

    // Check D matrix
    // D = D1*B2 = 0*1 = 0
    CHECK(result.D(0, 0) == doctest::Approx(0.0));
}

TEST_CASE("StateSpace Feedback via Operator/") {
    StateSpace<1, 1, 1> plant{};
    plant.A(0, 0) = -1.0;
    plant.B(0, 0) = 1.0;
    plant.C(0, 0) = 1.0;
    plant.D(0, 0) = 0.0;

    StateSpace<1, 1, 1> controller{};
    controller.A(0, 0) = 0.0;
    controller.B(0, 0) = 1.0;
    controller.C(0, 0) = 2.0;
    controller.D(0, 0) = 0.0;

    auto result = plant / controller;

    CHECK(result.A.rows() == 2);
    CHECK(result.A(0, 0) == doctest::Approx(-1.0));
    CHECK(result.A(0, 1) == doctest::Approx(-2.0));
}

TEST_CASE("StateSpace Constexpr") {
    constexpr StateSpace<1, 1, 1> sys1{
        Matrix<1, 1, double>{{-1.0}},
        Matrix<1, 1, double>{{1.0}},
        Matrix<1, 1, double>{{1.0}},
        Matrix<1, 1, double>{{0.0}},
        Matrix<1, 1, double>{},
        Matrix<1, 1, double>{},
        0.0
    };
    constexpr StateSpace<1, 1, 1> sys2{
        Matrix<1, 1, double>{{-2.0}},
        Matrix<1, 1, double>{{1.0}},
        Matrix<1, 1, double>{{1.0}},
        Matrix<1, 1, double>{{0.0}},
        Matrix<1, 1, double>{},
        Matrix<1, 1, double>{},
        0.0
    };

    // Should compile as constexpr
    constexpr auto result_series = series(sys1, sys2);
    constexpr auto result_parallel = parallel(sys1, sys2);
    constexpr auto result_feedback = feedback(sys1, sys2);

    static_assert(result_series.A.rows() == 2);
    static_assert(result_parallel.A.rows() == 2);
    static_assert(result_feedback.A.rows() == 2);
}
