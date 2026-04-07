#include <optional>

#include "cholesky.hpp"
#include "constexpr_complex.hpp"
#include "doctest.h"
#include "matrix.hpp"

using namespace wetmelon::control;
using namespace wetmelon::control::mat;
using wet::complex;

TEST_SUITE("cholesky") {
    TEST_CASE("Cholesky decomposition - basic 2x2 SPD matrix") {
        // Test matrix: [[4, 2], [2, 3]] - known to be SPD
        Matrix<2, 2, double> A = {{4.0, 2.0}, {2.0, 3.0}};

        auto L_opt = cholesky(A);
        REQUIRE(L_opt.has_value());

        const auto& L = L_opt.value();

        // Check that L is lower triangular
        CHECK(L(0, 1) == 0.0);

        // Check that A = L * L^T
        auto L_LT = L * L.transpose();
        CHECK(L_LT(0, 0) == doctest::Approx(A(0, 0)));
        CHECK(L_LT(0, 1) == doctest::Approx(A(0, 1)));
        CHECK(L_LT(1, 0) == doctest::Approx(A(1, 0)));
        CHECK(L_LT(1, 1) == doctest::Approx(A(1, 1)));
    }

    TEST_CASE("Cholesky decomposition - 3x3 SPD matrix") {
        // Test matrix: [[4, 2, 1], [2, 5, 3], [1, 3, 6]]
        Matrix<3, 3, double> A = {
            {4.0, 2.0, 1.0},
            {2.0, 5.0, 3.0},
            {1.0, 3.0, 6.0},
        };

        auto L_opt = cholesky(A);
        REQUIRE(L_opt.has_value());

        const auto& L = L_opt.value();

        // Check that L is lower triangular
        CHECK(L(0, 1) == 0.0);
        CHECK(L(0, 2) == 0.0);
        CHECK(L(1, 2) == 0.0);

        // Check that A = L * L^T
        auto L_LT = L * L.transpose();
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                CHECK(L_LT(i, j) == doctest::Approx(A(i, j)));
            }
        }
    }

    TEST_CASE("Cholesky decomposition - identity matrix") {
        Matrix<3, 3, double> A = Matrix<3, 3, double>::identity();

        auto L_opt = cholesky(A);
        REQUIRE(L_opt.has_value());

        const auto& L = L_opt.value();

        // For identity, L should also be identity
        CHECK(L(0, 0) == doctest::Approx(1.0));
        CHECK(L(1, 1) == doctest::Approx(1.0));
        CHECK(L(2, 2) == doctest::Approx(1.0));
        CHECK(L(0, 1) == 0.0);
        CHECK(L(0, 2) == 0.0);
        CHECK(L(1, 2) == 0.0);
    }

    TEST_CASE("Cholesky decomposition - non-SPD matrix (negative eigenvalue)") {
        // Matrix with negative eigenvalue: [[1, 2], [2, 1]]
        Matrix<2, 2, double> A = {{1.0, 2.0}, {2.0, 1.0}};

        auto L_opt = cholesky(A);
        CHECK(!L_opt.has_value());
    }

    TEST_CASE("Cholesky decomposition - non-SPD matrix (not positive definite)") {
        // Matrix that's not SPD: [[0, 0], [0, 0]]
        Matrix<2, 2, double> A = {{0.0, 0.0}, {0.0, 0.0}};

        auto L_opt = cholesky(A);
        CHECK(!L_opt.has_value());
    }

    TEST_CASE("Cholesky decomposition - non-symmetric matrix") {
        // Non-symmetric matrix: [[4, 1], [2, 3]]
        // Cholesky doesn't check symmetry, so it may succeed or fail
        Matrix<2, 2, double> A = {{4.0, 1.0}, {2.0, 3.0}};

        auto L_opt = cholesky(A);
        // The algorithm doesn't enforce symmetry, so result is implementation-dependent
        // Just check that it either succeeds or fails gracefully
        (void)L_opt; // Suppress unused variable warning
    }

    TEST_CASE("Forward substitution - basic test") {
        // Lower triangular matrix L
        Matrix<3, 3, double> L = {{2.0, 0.0, 0.0}, {1.0, 3.0, 0.0}, {0.5, 0.5, 4.0}};

        // Right-hand side b
        ColVec<3, double> b = {6.0, 8.0, 9.0};

        // Expected solution: L * x = b
        // x(0) = 6/2 = 3
        // x(1) = (8 - 1*3)/3 = 5/3 ≈ 1.6667
        // x(2) = (9 - 0.5*3 - 0.5*5/3)/4 = (9 - 1.5 - 5/6)/4 ≈ 1.7917
        auto x = forward_substitute(L, b);

        CHECK(x(0) == doctest::Approx(3.0));
        CHECK(x(1) == doctest::Approx(5.0 / 3.0));
        CHECK(x(2) == doctest::Approx((9.0 - 1.5 - 5.0 / 6.0) / 4.0));
    }

    TEST_CASE("Backward substitution - basic test") {
        // Upper triangular matrix U (L^T)
        Matrix<3, 3, double> U = {{2.0, 1.0, 0.5}, {0.0, 3.0, 0.5}, {0.0, 0.0, 4.0}};

        // Right-hand side b
        ColVec<3, double> b = {7.0, 5.5, 4.0};

        // Expected solution: U * x = b (treating U as L in backward_substitute)
        // x(2) = 4/4 = 1
        // x(1) = 5.5/3 ≈ 1.8333 (since U(2,1) = 0)
        // x(0) = 7/2 = 3.5 (since U(1,0) = 0, U(2,0) = 0)
        auto x = backward_substitute_transpose(U, b);

        CHECK(x(0) == doctest::Approx(3.5));
        CHECK(x(1) == doctest::Approx(5.5 / 3.0));
        CHECK(x(2) == doctest::Approx(1.0));
    }

    TEST_CASE("Cholesky solve - single RHS") {
        // SPD matrix A
        Matrix<2, 2, double> A = {{4.0, 2.0}, {2.0, 3.0}};

        // Right-hand side B (single column)
        Matrix<2, 1, double> B = {{8.0}, {7.0}};

        // Expected solution: A * x = B
        // x should be approximately [1.25, 1.5]
        auto X_opt = cholesky_solve(A, B);
        REQUIRE(X_opt.has_value());

        const auto& X = X_opt.value();
        CHECK(X(0, 0) == doctest::Approx(1.25));
        CHECK(X(1, 0) == doctest::Approx(1.5));
    }

    TEST_CASE("Cholesky solve - multiple RHS") {
        // SPD matrix A
        Matrix<3, 3, double> A = {{4.0, 2.0, 1.0}, {2.0, 5.0, 3.0}, {1.0, 3.0, 6.0}};

        // Right-hand side B (two columns)
        Matrix<3, 2, double> B = {{7.0, 10.0}, {9.0, 13.0}, {11.0, 15.0}};

        auto X_opt = cholesky_solve(A, B);
        REQUIRE(X_opt.has_value());

        const auto& X = X_opt.value();

        // Verify A * X = B for both columns
        auto AX = A * X;
        CHECK(AX(0, 0) == doctest::Approx(B(0, 0)));
        CHECK(AX(0, 1) == doctest::Approx(B(0, 1)));
        CHECK(AX(1, 0) == doctest::Approx(B(1, 0)));
        CHECK(AX(1, 1) == doctest::Approx(B(1, 1)));
        CHECK(AX(2, 0) == doctest::Approx(B(2, 0)));
        CHECK(AX(2, 1) == doctest::Approx(B(2, 1)));
    }

    TEST_CASE("Cholesky solve - non-SPD matrix") {
        // Non-SPD matrix
        Matrix<2, 2, double> A = {{1.0, 2.0}, {2.0, 1.0}};

        Matrix<2, 1, double> B = {{3.0}, {4.0}};

        auto X_opt = cholesky_solve(A, B);
        CHECK(!X_opt.has_value());
    }

    TEST_CASE("Cholesky decomposition - constexpr") {
        // Test that Cholesky works at compile time
        constexpr Matrix<2, 2, double> A = {{4.0, 2.0}, {2.0, 3.0}};

        constexpr auto L_opt = cholesky(A);
        static_assert(L_opt.has_value(), "Cholesky should succeed at compile time");

        // We can't access .value() in constexpr context for static_assert,
        // but we can verify the function compiles and returns a value
    }

    TEST_CASE("Forward substitution - constexpr") {
        constexpr Matrix<2, 2, double> L = {{2.0, 0.0}, {1.0, 1.0}};
        constexpr ColVec<2, double>    b = {4.0, 1.0};

        constexpr auto x = forward_substitute(L, b);

        static_assert(x(0) > 1.9 && x(0) < 2.1, "x(0) should be ~2.0");
        static_assert(x(1) > -1.1 && x(1) < -0.9, "x(1) should be ~-1.0");
    }

    TEST_CASE("Cholesky solve - float precision") {
        Matrix<2, 2, float> A = {{4.0f, 2.0f}, {2.0f, 3.0f}};
        Matrix<2, 1, float> B = {{8.0f}, {7.0f}};

        auto X_opt = cholesky_solve(A, B);
        REQUIRE(X_opt.has_value());

        const auto& X = X_opt.value();
        CHECK(X(0, 0) == doctest::Approx(1.25));
        CHECK(X(1, 0) == doctest::Approx(1.5));
    }

    TEST_CASE("LU decomposition - basic 2x2 matrix") {
        Matrix<2, 2, double> A = {{4.0, 2.0}, {2.0, 3.0}};

        auto lu_opt = lu_decomposition(A);
        REQUIRE(lu_opt.has_value());

        const auto& [L, U, piv] = lu_opt.value();

        // Check that L is lower triangular with unit diagonal
        CHECK(L(0, 0) == doctest::Approx(1.0));
        CHECK(L(1, 0) == doctest::Approx(0.5));
        CHECK(L(0, 1) == 0.0);
        CHECK(L(1, 1) == doctest::Approx(1.0));

        // Check that U is upper triangular
        CHECK(U(0, 0) == doctest::Approx(4.0));
        CHECK(U(0, 1) == doctest::Approx(2.0));
        CHECK(U(1, 0) == 0.0);
        CHECK(U(1, 1) == doctest::Approx(2.0));

        // Check that A = L * U
        auto LU = L * U;
        CHECK(LU(0, 0) == doctest::Approx(A(0, 0)));
        CHECK(LU(0, 1) == doctest::Approx(A(0, 1)));
        CHECK(LU(1, 0) == doctest::Approx(A(1, 0)));
        CHECK(LU(1, 1) == doctest::Approx(A(1, 1)));
    }

    TEST_CASE("LU decomposition - 3x3 matrix") {
        Matrix<3, 3, double> A = {
            {4.0, 1.0, 1.0},
            {2.0, 3.0, 3.0},
            {2.0, 1.0, 9.0}
        };

        auto lu_opt = lu_decomposition(A);
        REQUIRE(lu_opt.has_value());

        const auto& [L, U, piv] = lu_opt.value();

        // Check that L is lower triangular with unit diagonal
        CHECK(L(0, 0) == doctest::Approx(1.0));
        CHECK(L(1, 0) == doctest::Approx(0.5));
        CHECK(L(2, 0) == doctest::Approx(0.5));
        CHECK(L(1, 1) == doctest::Approx(1.0));
        CHECK(L(2, 1) == doctest::Approx(0.2));
        CHECK(L(2, 2) == doctest::Approx(1.0));
        CHECK(L(0, 1) == 0.0);
        CHECK(L(0, 2) == 0.0);
        CHECK(L(1, 2) == 0.0);

        // Check that U is upper triangular
        CHECK(U(0, 0) == doctest::Approx(4.0));
        CHECK(U(0, 1) == doctest::Approx(1.0));
        CHECK(U(0, 2) == doctest::Approx(1.0));
        CHECK(U(1, 1) == doctest::Approx(2.5));
        CHECK(U(1, 2) == doctest::Approx(2.5));
        CHECK(U(2, 2) == doctest::Approx(8.0));
        CHECK(U(1, 0) == 0.0);
        CHECK(U(2, 0) == 0.0);
        CHECK(U(2, 1) == doctest::Approx(0.0));

        // Check that A = L * U
        auto LU = L * U;
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                CHECK(LU(i, j) == doctest::Approx(A(i, j)));
            }
        }
    }

    TEST_CASE("LU decomposition - singular matrix") {
        // Singular matrix: [[1, 2], [2, 4]] - second row is 2*first row
        Matrix<2, 2, double> A = {{1.0, 2.0}, {2.0, 4.0}};

        auto lu_opt = lu_decomposition(A);
        CHECK(!lu_opt.has_value());
    }

    TEST_CASE("LU decomposition - float precision") {
        Matrix<2, 2, float> A = {{4.0f, 2.0f}, {2.0f, 3.0f}};

        auto lu_opt = lu_decomposition(A);
        REQUIRE(lu_opt.has_value());

        const auto& [L, U, piv] = lu_opt.value();

        // Check that A = L * U
        auto LU = L * U;
        CHECK(LU(0, 0) == doctest::Approx(A(0, 0)));
        CHECK(LU(0, 1) == doctest::Approx(A(0, 1)));
        CHECK(LU(1, 0) == doctest::Approx(A(1, 0)));
        CHECK(LU(1, 1) == doctest::Approx(A(1, 1)));
    }

    TEST_CASE("LU decomposition - complex matrix") {
        using C = complex<double>;
        Matrix<2, 2, C> A = {
            {C{4.0, 1.0}, C{2.0, 0.5}},
            {C{2.0, 0.5}, C{3.0, 2.0}}
        };

        auto lu_opt = lu_decomposition(A);
        REQUIRE(lu_opt.has_value());

        const auto& [L, U, piv] = lu_opt.value();

        // Check that L is lower triangular with unit diagonal
        CHECK(L(0, 0).real() == doctest::Approx(1.0));
        CHECK(L(0, 0).imag() == doctest::Approx(0.0));
        CHECK(L(1, 1).real() == doctest::Approx(1.0));
        CHECK(L(1, 1).imag() == doctest::Approx(0.0));
        CHECK(L(0, 1).real() == doctest::Approx(0.0));
        CHECK(L(0, 1).imag() == doctest::Approx(0.0));

        // Check that U is upper triangular
        CHECK(U(1, 0).real() == doctest::Approx(0.0));
        CHECK(U(1, 0).imag() == doctest::Approx(0.0));

        // Check that A = L * U
        auto LU = L * U;
        CHECK(LU(0, 0).real() == doctest::Approx(A(0, 0).real()));
        CHECK(LU(0, 0).imag() == doctest::Approx(A(0, 0).imag()));
        CHECK(LU(0, 1).real() == doctest::Approx(A(0, 1).real()));
        CHECK(LU(0, 1).imag() == doctest::Approx(A(0, 1).imag()));
        CHECK(LU(1, 0).real() == doctest::Approx(A(1, 0).real()));
        CHECK(LU(1, 0).imag() == doctest::Approx(A(1, 0).imag()));
        CHECK(LU(1, 1).real() == doctest::Approx(A(1, 1).real()));
        CHECK(LU(1, 1).imag() == doctest::Approx(A(1, 1).imag()));
    }

    TEST_CASE("Solve function - SPD matrix (uses Cholesky)") {
        Matrix<2, 2, double> A = {{4.0, 2.0}, {2.0, 3.0}};
        Matrix<2, 1, double> B = {{8.0}, {7.0}};

        auto X_opt = solve(A, B);
        REQUIRE(X_opt.has_value());

        const auto& X = X_opt.value();
        CHECK(X(0, 0) == doctest::Approx(1.25));
        CHECK(X(1, 0) == doctest::Approx(1.5));

        // Verify A * X = B
        auto AX = A * X;
        CHECK(AX(0, 0) == doctest::Approx(B(0, 0)));
        CHECK(AX(1, 0) == doctest::Approx(B(1, 0)));
    }

    TEST_CASE("Solve function - non-SPD matrix (uses LU)") {
        Matrix<2, 2, double> A = {{1.0, 2.0}, {3.0, 4.0}}; // Not SPD
        Matrix<2, 1, double> B = {{5.0}, {11.0}};

        auto X_opt = solve(A, B);
        REQUIRE(X_opt.has_value());

        const auto& X = X_opt.value();
        CHECK(X(0, 0) == doctest::Approx(1.0));
        CHECK(X(1, 0) == doctest::Approx(2.0));

        // Verify A * X = B
        auto AX = A * X;
        CHECK(AX(0, 0) == doctest::Approx(B(0, 0)));
        CHECK(AX(1, 0) == doctest::Approx(B(1, 0)));
    }

    TEST_CASE("Solve function - singular matrix") {
        Matrix<2, 2, double> A = {{1.0, 2.0}, {2.0, 4.0}}; // Singular
        Matrix<2, 1, double> B = {{3.0}, {6.0}};

        auto X_opt = solve(A, B);
        CHECK(!X_opt.has_value());
    }

    TEST_CASE("Solve function - float precision") {
        Matrix<2, 2, float> A = {{4.0f, 2.0f}, {2.0f, 3.0f}};
        Matrix<2, 1, float> B = {{8.0f}, {7.0f}};

        auto X_opt = solve(A, B);
        REQUIRE(X_opt.has_value());

        const auto& X = X_opt.value();
        CHECK(X(0, 0) == doctest::Approx(1.25f));
        CHECK(X(1, 0) == doctest::Approx(1.5f));
    }

    TEST_CASE("Solve function - complex matrix") {
        using C = complex<double>;
        Matrix<2, 2, C> A = {
            {C{4.0, 1.0}, C{2.0, 0.5}},
            {C{2.0, 0.5}, C{3.0, 2.0}}
        };
        Matrix<2, 1, C> B = {
            {C{8.0, 2.0}},
            {C{7.0, 1.5}}
        };

        auto X_opt = solve(A, B);
        REQUIRE(X_opt.has_value());

        const auto& X = X_opt.value();

        // Verify A * X = B
        auto AX = A * X;
        CHECK(AX(0, 0).real() == doctest::Approx(B(0, 0).real()));
        CHECK(AX(0, 0).imag() == doctest::Approx(B(0, 0).imag()));
        CHECK(AX(1, 0).real() == doctest::Approx(B(1, 0).real()));
        CHECK(AX(1, 0).imag() == doctest::Approx(B(1, 0).imag()));
    }

    TEST_CASE("LU decomposition - constexpr") {
        constexpr Matrix<2, 2, double> A = {{4.0, 2.0}, {2.0, 3.0}};

        constexpr auto lu_opt = lu_decomposition(A);
        static_assert(lu_opt.has_value(), "LU decomposition should succeed at compile time");
    }

    TEST_CASE("Solve function - constexpr") {
        constexpr Matrix<2, 2, double> A = {{4.0, 2.0}, {2.0, 3.0}};
        constexpr Matrix<2, 1, double> B = {{8.0}, {7.0}};

        constexpr auto X_opt = solve(A, B);
        static_assert(X_opt.has_value(), "Solve should succeed at compile time");

        // We can't access .value() in constexpr context for static_assert,
        // but we can verify the function compiles
    }
}
