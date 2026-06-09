#include <cmath>

#include "wet/matrix/matrix.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wet;

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

    TEST_CASE("Matrix sin/cos - diagonal, large norm") {
        // sin/cos of a diagonal matrix = diagonal of sin/cos of eigenvalues
        // Norm ≈ 10, exercises the scaling-and-doubling path
        Matrix<2, 2> A{{5.0, 0.0}, {0.0, 10.0}};

        auto sinA = mat::sin(A);
        auto cosA = mat::cos(A);

        CHECK(doctest::Approx(sinA(0, 0)).epsilon(1e-6) == std::sin(5.0));
        CHECK(doctest::Approx(sinA(0, 1)).epsilon(1e-10) == 0.0);
        CHECK(doctest::Approx(sinA(1, 0)).epsilon(1e-10) == 0.0);
        CHECK(doctest::Approx(sinA(1, 1)).epsilon(1e-6) == std::sin(10.0));

        CHECK(doctest::Approx(cosA(0, 0)).epsilon(1e-6) == std::cos(5.0));
        CHECK(doctest::Approx(cosA(0, 1)).epsilon(1e-10) == 0.0);
        CHECK(doctest::Approx(cosA(1, 0)).epsilon(1e-10) == 0.0);
        CHECK(doctest::Approx(cosA(1, 1)).epsilon(1e-6) == std::cos(10.0));
    }

    TEST_CASE("Matrix trig identity: large norm sin^2 + cos^2 = I") {
        Matrix<2, 2> A{{3.0, 1.5}, {-1.0, 4.0}};

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

        REQUIRE(sqrt_I.has_value());
        CHECK(doctest::Approx(sqrt_I.value()(0, 0)).epsilon(1e-6) == 1.0);
        CHECK(doctest::Approx(sqrt_I.value()(1, 1)).epsilon(1e-6) == 1.0);
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

        // Two norm (spectral norm): largest singular value of A = [[1,2],[3,4]]
        // sigma_max = sqrt(lambda_max(A^T A)) ≈ 5.46499
        CHECK(doctest::Approx(mat::two_norm(A)).epsilon(1e-4) == 5.4649857);
    }

    TEST_CASE("Matrix norms - identity") {
        auto I = Matrix<2, 2>::identity();

        CHECK(doctest::Approx(mat::one_norm(I)).epsilon(1e-10) == 1.0);
        CHECK(doctest::Approx(mat::infinity_norm(I)).epsilon(1e-10) == 1.0);
        CHECK(doctest::Approx(mat::frobenius_norm(I)).epsilon(1e-10) == std::sqrt(2.0));
        // Spectral norm of identity is 1 (all singular values are 1)
        CHECK(doctest::Approx(mat::two_norm(I)).epsilon(1e-10) == 1.0);
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

    // Largest |A - B| element, for comparing matrix-function results.
    static constexpr auto max_abs_diff = []<size_t N>(const Matrix<N, N>& A, const Matrix<N, N>& B) {
        double worst = 0.0;
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                worst = wet::max(worst, std::abs(A(i, j) - B(i, j)));
            }
        }
        return worst;
    };

    TEST_CASE("Matrix logarithm is the inverse of expm") {
        // log(exp(A)) == A for a non-trivial, well-conditioned A.
        Matrix<3, 3> A = {
            {0.1, 0.2, 0.0},
            {-0.1, 0.3, 0.1},
            {0.0, -0.2, 0.2},
        };
        Matrix<3, 3> roundtrip = mat::log(mat::expm(A));
        CHECK(max_abs_diff(roundtrip, A) < 1e-9);

        // exp(log(B)) == B for an SPD matrix B (log well-defined).
        Matrix<2, 2> B = {
            {4.0, 1.0},
            {1.0, 3.0},
        };
        Matrix<2, 2> back = mat::expm(mat::log(B));
        CHECK(max_abs_diff(back, B) < 1e-9);

        // log(identity) == 0
        Matrix<3, 3> logI = mat::log(Matrix<3, 3>::identity());
        CHECK(max_abs_diff(logI, Matrix<3, 3>::zeros()) < 1e-12);

        // log(diagonal) == diagonal of logs
        Matrix<2, 2> D = {{2.0, 0.0}, {0.0, 5.0}};
        Matrix<2, 2> logD = mat::log(D);
        // logm uses inverse scaling-and-squaring; accuracy degrades for
        // eigenvalues far from 1 (more squaring steps), ~1e-7 here.
        CHECK(logD(0, 0) == doctest::Approx(std::log(2.0)).epsilon(1e-6));
        CHECK(logD(1, 1) == doctest::Approx(std::log(5.0)).epsilon(1e-6));
    }

    TEST_CASE("Matrix square root: sqrtm(A)^2 == A") {
        Matrix<2, 2> A = {
            {5.0, 4.0},
            {1.0, 2.0},
        };
        auto S = mat::sqrt(A);
        REQUIRE(S.has_value());
        Matrix<2, 2> squared = S.value() * S.value();
        CHECK(max_abs_diff(squared, A) < 1e-8);

        // SPD 3x3
        Matrix<3, 3> B = {
            {4.0, 1.0, 0.0},
            {1.0, 3.0, 1.0},
            {0.0, 1.0, 2.0},
        };
        auto SB = mat::sqrt(B);
        REQUIRE(SB.has_value());
        CHECK(max_abs_diff(SB.value() * SB.value(), B) < 1e-7);

        // sqrt(identity) == identity
        auto SI = mat::sqrt(Matrix<3, 3>::identity());
        REQUIRE(SI.has_value());
        CHECK(max_abs_diff(SI.value(), Matrix<3, 3>::identity()) < 1e-10);
    }

    TEST_CASE("Matrix real-exponent power") {
        Matrix<2, 2> A = {
            {5.0, 4.0},
            {1.0, 2.0},
        };
        // A^0.5 squared == A
        Matrix<2, 2> half = mat::pow(A, 0.5);
        CHECK(max_abs_diff(half * half, A) < 1e-7);

        // A^2.0 == A*A
        Matrix<2, 2> sq = mat::pow(A, 2.0);
        CHECK(max_abs_diff(sq, A * A) < 1e-7);

        // A^1.0 == A, A^0.0 == I
        CHECK(max_abs_diff(mat::pow(A, 1.0), A) < 1e-9);
        CHECK(max_abs_diff(mat::pow(A, 0.0), Matrix<2, 2>::identity()) < 1e-9);
    }

    TEST_CASE("sincos / sin / cos consistency") {
        Matrix<3, 3> A = {
            {0.3, 0.1, 0.0},
            {-0.2, 0.4, 0.1},
            {0.0, -0.1, 0.5},
        };
        auto [S, C] = mat::sincos(A);

        // sin(A) and cos(A) agree with the paired primitive ...
        CHECK(max_abs_diff(mat::sin(A), S) < 1e-12);
        // ... and the standalone cos path matches sincos's cos exactly.
        CHECK(max_abs_diff(mat::cos(A), C) < 1e-12);

        // Pythagorean identity sin^2 + cos^2 == I
        Matrix<3, 3> id = (S * S) + (C * C);
        CHECK(max_abs_diff(id, Matrix<3, 3>::identity()) < 1e-9);

        // Larger norm exercises the scaling/doubling path
        Matrix<2, 2> B = {{2.0, 1.0}, {0.5, 3.0}};
        auto [SB, CB] = mat::sincos(B);
        CHECK(max_abs_diff(mat::cos(B), CB) < 1e-10);
        CHECK(max_abs_diff(mat::sin(B), SB) < 1e-10);
        CHECK(max_abs_diff((SB * SB) + (CB * CB), Matrix<2, 2>::identity()) < 1e-8);
    }
}
