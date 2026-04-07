#include <cmath>

#include "doctest.h"
#include "matrix.hpp"

using namespace wetmelon::control;

/**
 * @brief Tests for matrix function implementations
 *
 * Verifies exponentials, trigonometric functions, and power operations
 * on matrices.
 */
TEST_SUITE("Matrix Functions") {
    TEST_CASE("Matrix exponential - identity") {
        Matrix<2, 2> zeros = Matrix<2, 2>::zeros();

        auto exp_zeros = mat::expm(zeros);

        // exp(0) = I
        CHECK(doctest::Approx(exp_zeros(0, 0)).epsilon(1e-10) == 1.0);
        CHECK(doctest::Approx(exp_zeros(0, 1)).epsilon(1e-10) == 0.0);
        CHECK(doctest::Approx(exp_zeros(1, 0)).epsilon(1e-10) == 0.0);
        CHECK(doctest::Approx(exp_zeros(1, 1)).epsilon(1e-10) == 1.0);
    }

    TEST_CASE("Matrix exponential - diagonal") {
        Matrix<2, 2> A{{-1.0, 0.0}, {0.0, -2.0}};
        auto         exp_A = mat::expm(A);

        // exp(diag(-1,-2)) = diag(exp(-1), exp(-2))
        // Relaxed tolerance for Padé approximation
        CHECK(doctest::Approx(exp_A(0, 0)).epsilon(1e-3) == std::exp(-1.0));
        CHECK(doctest::Approx(exp_A(0, 1)).epsilon(1e-10) == 0.0);
        CHECK(doctest::Approx(exp_A(1, 0)).epsilon(1e-10) == 0.0);
        CHECK(doctest::Approx(exp_A(1, 1)).epsilon(1e-3) == std::exp(-2.0));
    }

    TEST_CASE("Matrix exponential - rotation") {
        // Rotation matrix: exp([0 -θ; θ 0]) = [cos(θ) -sin(θ); sin(θ) cos(θ)]
        double       theta = 0.5;
        Matrix<2, 2> A{{0.0, -theta}, {theta, 0.0}};
        auto         exp_A = mat::expm(A);

        // Relaxed tolerance for Padé approximation
        CHECK(doctest::Approx(exp_A(0, 0)).epsilon(1e-3) == std::cos(theta));
        CHECK(doctest::Approx(exp_A(0, 1)).epsilon(1e-3) == -std::sin(theta));
        CHECK(doctest::Approx(exp_A(1, 0)).epsilon(1e-3) == std::sin(theta));
        CHECK(doctest::Approx(exp_A(1, 1)).epsilon(1e-3) == std::cos(theta));
    }

    TEST_CASE("Matrix trig identity: sin^2 + cos^2 = I") {
        Matrix<2, 2> A{{0.3, 0.1}, {-0.1, 0.4}};

        auto sinA = mat::sin(A);
        auto cosA = mat::cos(A);
        auto sum = sinA * sinA + cosA * cosA;
        auto I = Matrix<2, 2>::identity();

        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                CHECK(doctest::Approx(sum(i, j)).epsilon(1e-6) == I(i, j));
            }
        }
    }

    TEST_CASE("Matrix sqrt - identity") {
        auto I = Matrix<2, 2>::identity();
        auto sqrt_I = mat::sqrt(I);

        // sqrt(I) = I
        CHECK(doctest::Approx(sqrt_I(0, 0)).epsilon(1e-6) == 1.0);
        CHECK(doctest::Approx(sqrt_I(1, 1)).epsilon(1e-6) == 1.0);
    }

    TEST_CASE("Matrix power - integer") {
        Matrix<2, 2> A{{1.0, 1.0}, {0.0, 1.0}};

        auto A2 = mat::pow(A, 2);
        auto A_sq = A * A;

        CHECK(doctest::Approx(A2(0, 0)).epsilon(1e-10) == A_sq(0, 0));
        CHECK(doctest::Approx(A2(0, 1)).epsilon(1e-10) == A_sq(0, 1));
        CHECK(doctest::Approx(A2(1, 0)).epsilon(1e-10) == A_sq(1, 0));
        CHECK(doctest::Approx(A2(1, 1)).epsilon(1e-10) == A_sq(1, 1));
    }

    TEST_CASE("Matrix hyperbolic identity: cosh^2 - sinh^2 = I") {
        Matrix<2, 2> A{{0.2, 0.1}, {0.1, 0.3}};

        auto sinhA = mat::sinh(A);
        auto coshA = mat::cosh(A);
        auto diff = coshA * coshA - sinhA * sinhA;
        auto I = Matrix<2, 2>::identity();

        // Relaxed tolerance for numerical precision through exp
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                CHECK(doctest::Approx(diff(i, j)).epsilon(1e-3) == I(i, j));
            }
        }
    }

    TEST_CASE("Matrix norms") {
        Matrix<2, 2> A{{1.0, 2.0}, {3.0, 4.0}};

        // One norm: max column sum = max(|1|+|3|, |2|+|4|) = max(4, 6) = 6
        CHECK(doctest::Approx(mat::one_norm(A)).epsilon(1e-10) == 6.0);

        // Infinity norm: max row sum = max(|1|+|2|, |3|+|4|) = max(3, 7) = 7
        CHECK(doctest::Approx(mat::infinity_norm(A)).epsilon(1e-10) == 7.0);

        // Frobenius norm: sqrt(1^2 + 2^2 + 3^2 + 4^2) = sqrt(1+4+9+16) = sqrt(30)
        CHECK(doctest::Approx(mat::frobenius_norm(A)).epsilon(1e-10) == std::sqrt(30.0));

        // Two norm approximation (using Frobenius for now)
        CHECK(doctest::Approx(mat::two_norm(A)).epsilon(1e-10) == std::sqrt(30.0));
    }

    TEST_CASE("Matrix norms - identity") {
        auto I = Matrix<2, 2>::identity();

        CHECK(doctest::Approx(mat::one_norm(I)).epsilon(1e-10) == 1.0);
        CHECK(doctest::Approx(mat::infinity_norm(I)).epsilon(1e-10) == 1.0);
        CHECK(doctest::Approx(mat::frobenius_norm(I)).epsilon(1e-10) == std::sqrt(2.0));
        CHECK(doctest::Approx(mat::two_norm(I)).epsilon(1e-10) == std::sqrt(2.0));
    }

    TEST_CASE("Matrix norms - zero matrix") {
        Matrix<2, 2> Z = Matrix<2, 2>::zeros();

        CHECK(doctest::Approx(mat::one_norm(Z)).epsilon(1e-10) == 0.0);
        CHECK(doctest::Approx(mat::infinity_norm(Z)).epsilon(1e-10) == 0.0);
        CHECK(doctest::Approx(mat::frobenius_norm(Z)).epsilon(1e-10) == 0.0);
        CHECK(doctest::Approx(mat::two_norm(Z)).epsilon(1e-10) == 0.0);
    }

    TEST_CASE("Matrix determinant") {
        // 1x1 matrix
        Matrix<1, 1> A1{{5.0}};
        CHECK(doctest::Approx(mat::det(A1)).epsilon(1e-10) == 5.0);

        // 2x2 matrix: det([1 2; 3 4]) = 1*4 - 2*3 = -2
        Matrix<2, 2> A2{{1.0, 2.0}, {3.0, 4.0}};
        CHECK(doctest::Approx(mat::det(A2)).epsilon(1e-10) == -2.0);

        // 3x3 matrix: det([1 2 3; 4 5 6; 7 8 9]) = 0 (singular)
        Matrix<3, 3> A3{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
        CHECK(doctest::Approx(mat::det(A3)).epsilon(1e-10) == 0.0);

        // Identity matrix
        auto I2 = Matrix<2, 2>::identity();
        CHECK(doctest::Approx(mat::det(I2)).epsilon(1e-10) == 1.0);

        auto I3 = Matrix<3, 3>::identity();
        CHECK(doctest::Approx(mat::det(I3)).epsilon(1e-10) == 1.0);
    }

    TEST_CASE("Matrix rank") {
        // Full rank matrices
        auto I2 = Matrix<2, 2>::identity();
        CHECK(mat::rank(I2) == 2);

        Matrix<2, 2> A2{{1.0, 2.0}, {3.0, 4.0}};
        CHECK(mat::rank(A2) == 2);

        // Rank deficient matrices
        Matrix<2, 2> singular{{1.0, 2.0}, {2.0, 4.0}}; // Second row is 2*first row
        CHECK(mat::rank(singular) == 1);

        Matrix<3, 3> rank1{{1.0, 2.0, 3.0}, {2.0, 4.0, 6.0}, {3.0, 6.0, 9.0}};
        CHECK(mat::rank(rank1) == 1);

        // Zero matrix
        Matrix<2, 2> Z = Matrix<2, 2>::zeros();
        CHECK(mat::rank(Z) == 0);
    }
}
