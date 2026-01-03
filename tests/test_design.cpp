#include <cmath>

#include "control_design.hpp"
#include "doctest.h"

using namespace wetmelon::control;

/**
 * @brief Tests for compile-time design:: functions
 *
 * These tests use constexpr to ensure functions are evaluated at compile-time.
 */

TEST_SUITE("DARE: Cross-Term N Support") {
    TEST_CASE("dare with zero cross-term N matches no-N variant") {
        // Verify that dare(A, B, Q, R, N={0}) produces same result as dare(A, B, Q, R)
        Matrix<2, 2> A{{0.95, 0.1}, {0.0, 0.9}};
        Matrix<2, 1> B{{0.1}, {0.1}};
        Matrix<2, 2> Q = Matrix<2, 2>::identity();
        Matrix<1, 1> R = {{1.0}};
        Matrix<2, 1> N{};

        auto P_no_N = dare(A, B, Q, R);
        auto P_with_zero_N = dare(A, B, Q, R, N);

        CHECK(P_no_N.has_value());
        CHECK(P_with_zero_N.has_value());

        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                CHECK(doctest::Approx(P_no_N.value()(i, j)).epsilon(1e-10) == P_with_zero_N.value()(i, j));
            }
        }
    }

    TEST_CASE("dare with non-zero cross-term N produces valid result") {
        Matrix<1, 1> A = {{1.0}};
        Matrix<1, 1> B = {{1.0}};
        Matrix<1, 1> Q = {{1.0}};
        Matrix<1, 1> R = {{1.0}};
        Matrix<1, 1> N = {{0.1}};

        auto P_opt = dare(A, B, Q, R, N);
        CHECK(P_opt.has_value());

        double P = P_opt.value()(0, 0);
        // Verify P > 0 for asymptotically stable result
        CHECK(P > 0.0);
    }
}

TEST_SUITE("Design: Compile-Time LQR with Cross-Term N") {
    TEST_CASE("design::dlqr with cross-term N at compile time") {
        // Use constexpr to ensure compile-time evaluation
        constexpr auto result_no_N = design::dlqr(
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{1.0}}
        );

        constexpr auto result_with_N = design::dlqr(
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{0.5}}
        );

        static_assert(result_no_N.success);
        static_assert(result_with_N.success);

        // Gains should differ
        static_assert(result_no_N.K(0, 0) != result_with_N.K(0, 0));

        // Runtime checks
        CHECK(result_no_N.success);
        CHECK(result_with_N.success);
        CHECK(result_no_N.K(0, 0) != result_with_N.K(0, 0));
    }
}

TEST_SUITE("Design: Compile-Time Result Type Conversions") {
    TEST_CASE("LQRResult::as<U>() at compile time") {
        constexpr auto lqr_d = design::dlqr(
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{1.0}}
        );

        // Convert to float at compile time
        constexpr auto lqr_f = lqr_d.as<float>();

        static_assert(lqr_f.success);
        static_assert(lqr_f.K(0, 0) != 0.0f);

        // Verify conversion preserves value (approximately)
        CHECK(doctest::Approx(static_cast<double>(lqr_f.K(0, 0))).epsilon(1e-6) == lqr_d.K(0, 0));
    }
}

TEST_SUITE("Design: Compile-Time Success Flags") {
    TEST_CASE("LQRResult::success at compile time") {
        constexpr auto result = design::dlqr(
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{1.0}},
            Matrix<1, 1>{{1.0}}
        );

        static_assert(result.success);
        static_assert(result.S(0, 0) > 0.0);

        CHECK(result.success);
        CHECK(result.S(0, 0) > 0.0);
    }
}
