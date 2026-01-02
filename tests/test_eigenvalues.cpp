#include "doctest.h"
#include "eigenvalues.hpp"
#include "matrix.hpp"

TEST_CASE("compute_eigenvalues_qr - 1x1 matrix") {
    Matrix<1, 1> A = {{5.0}};

    auto result = compute_eigenvalues_qr(A);

    CHECK(result.converged);
    CHECK(result.eigenvalues_real(0, 0) == doctest::Approx(5.0));
}

TEST_CASE("compute_eigenvalues_qr - 2x2 diagonal matrix") {
    Matrix<2, 2> A = {
        {3.0, 0.0},
        {0.0, 7.0}
    };

    auto result = compute_eigenvalues_qr(A);

    CHECK(result.converged);
    // Eigenvalues should be 3 and 7 (on diagonal)
    CHECK((result.eigenvalues_real(0, 0) == doctest::Approx(3.0) || result.eigenvalues_real(0, 0) == doctest::Approx(7.0)));
    CHECK((result.eigenvalues_real(1, 1) == doctest::Approx(7.0) || result.eigenvalues_real(1, 1) == doctest::Approx(3.0)));
}

TEST_CASE("compute_eigenvalues_qr - 2x2 symmetric matrix") {
    // Symmetric matrix with known eigenvalues
    Matrix<2, 2> A = {
        {4.0, 1.0},
        {1.0, 4.0}
    };

    auto result = compute_eigenvalues_qr(A);

    CHECK(result.converged);
    // Eigenvalues should be 3 and 5
    double lambda1 = result.eigenvalues_real(0, 0);
    double lambda2 = result.eigenvalues_real(1, 1);

    // Check that we got both eigenvalues (order may vary)
    bool has_3 = (std::abs(lambda1 - 3.0) < 1e-6 || std::abs(lambda2 - 3.0) < 1e-6);
    bool has_5 = (std::abs(lambda1 - 5.0) < 1e-6 || std::abs(lambda2 - 5.0) < 1e-6);

    CHECK(has_3);
    CHECK(has_5);
}

TEST_CASE("compute_eigenvalues_qr - 2x2 identity matrix") {
    Matrix<2, 2> A = Matrix<2, 2>::identity();

    auto result = compute_eigenvalues_qr(A);

    CHECK(result.converged);
    CHECK(result.eigenvalues_real(0, 0) == doctest::Approx(1.0));
    CHECK(result.eigenvalues_real(1, 1) == doctest::Approx(1.0));
}

TEST_CASE("compute_eigenvalues_qr - 3x3 symmetric matrix") {
    // Symmetric matrix
    Matrix<3, 3> A = {
        {6.0, 2.0, 1.0},
        {2.0, 3.0, 1.0},
        {1.0, 1.0, 1.0}
    };

    auto result = compute_eigenvalues_qr(A, 200);

    CHECK(result.converged);

    // Sum of eigenvalues should equal trace
    double trace = A(0, 0) + A(1, 1) + A(2, 2);
    double sum_eigenvalues = result.eigenvalues_real(0, 0) + result.eigenvalues_real(1, 1) + result.eigenvalues_real(2, 2);
    CHECK(sum_eigenvalues == doctest::Approx(trace).epsilon(1e-5));
}

TEST_CASE("compute_eigenvalues_qr - positive definite matrix") {
    // Positive definite matrix (all eigenvalues should be positive)
    Matrix<2, 2> A = {
        {2.0, 0.5},
        {0.5, 2.0}
    };

    auto result = compute_eigenvalues_qr(A);

    CHECK(result.converged);
    CHECK(result.eigenvalues_real(0, 0) > 0.0);
    CHECK(result.eigenvalues_real(1, 1) > 0.0);
}

TEST_CASE("compute_eigenvalues_qr - negative eigenvalues") {
    // Matrix with negative eigenvalues
    Matrix<2, 2> A = {
        {-3.0, 0.0},
        {0.0, -5.0}
    };

    auto result = compute_eigenvalues_qr(A);

    CHECK(result.converged);
    CHECK(result.eigenvalues_real(0, 0) == doctest::Approx(-3.0));
    CHECK(result.eigenvalues_real(1, 1) == doctest::Approx(-5.0));
}

TEST_CASE("compute_eigenvalues_qr - check for positive definiteness") {
    SUBCASE("positive definite matrix") {
        Matrix<3, 3> Q = {
            {2.0, 0.1, 0.1},
            {0.1, 2.0, 0.1},
            {0.1, 0.1, 2.0}
        };

        auto result = compute_eigenvalues_qr(Q, 200);

        if (result.converged) {
            bool all_positive = true;
            for (size_t i = 0; i < 3; ++i) {
                if (result.eigenvalues_real(i, i) <= 0.0) {
                    all_positive = false;
                    break;
                }
            }
            CHECK(all_positive);
        }
    }

    SUBCASE("non-positive definite matrix") {
        Matrix<2, 2> Q = {
            {1.0, 2.0},
            {2.0, 1.0}
        };

        auto result = compute_eigenvalues_qr(Q);

        if (result.converged) {
            // Should have eigenvalues 3 and -1
            double lambda1 = result.eigenvalues_real(0, 0);
            double lambda2 = result.eigenvalues_real(1, 1);

            bool has_negative = (lambda1 < 0.0 || lambda2 < 0.0);
            CHECK(has_negative);
        }
    }
}

TEST_CASE("compute_eigenvalues_qr - zero matrix") {
    Matrix<2, 2> A = Matrix<2, 2>::zeros();

    auto result = compute_eigenvalues_qr(A);

    // Zero matrix should converge immediately (already diagonal)
    // or may not converge if QR decomposition fails
    // Either way, eigenvalues should be zero
    CHECK(result.eigenvalues_real(0, 0) == doctest::Approx(0.0));
    CHECK(result.eigenvalues_real(1, 1) == doctest::Approx(0.0));
}
