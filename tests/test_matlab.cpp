
#include "constexpr_complex.hpp"
#include "matlab.hpp"

using namespace wetmelon::control;

#include "doctest.h"

TEST_CASE("Blkdiag") {
    Matrix<2, 2> mat1 = {{1, 2}, {3, 4}};
    Matrix<3, 3> mat2 = {{5, 6, 7}, {8, 9, 10}, {11, 12, 13}};
    Matrix<1, 1> mat3 = {{14}};

    auto block_diag = matlab::blkdiag(mat1, mat2, mat3);

    CHECK(block_diag.rows() == 6);
    CHECK(block_diag.cols() == 6);

    // Check first block
    CHECK(block_diag(0, 0) == 1);
    CHECK(block_diag(0, 1) == 2);
    CHECK(block_diag(1, 0) == 3);
    CHECK(block_diag(1, 1) == 4);

    // Check second block
    CHECK(block_diag(2, 2) == 5);
    CHECK(block_diag(2, 3) == 6);
    CHECK(block_diag(2, 4) == 7);
    CHECK(block_diag(3, 2) == 8);
    CHECK(block_diag(3, 3) == 9);
    CHECK(block_diag(3, 4) == 10);
    CHECK(block_diag(4, 2) == 11);
    CHECK(block_diag(4, 3) == 12);
    CHECK(block_diag(4, 4) == 13);

    // Check third block
    CHECK(block_diag(5, 5) == 14);

    // Check off-diagonal zeros
    for (size_t r = 0; r < block_diag.rows(); ++r) {
        for (size_t c = 0; c < block_diag.cols(); ++c) {
            if ((r < 2 && c < 2) || (r >= 2 && r < 5 && c >= 2 && c < 5) || (r == 5 && c == 5)) {
                continue; // Skip diagonal blocks
            }
            CHECK(block_diag(r, c) == 0);
        }
    }
}

TEST_CASE("Place - Pole placement for single input") {
    // Simple 2x2 system
    Matrix<2, 2> A = {{0, 1}, {-1, -2}};
    Matrix<2, 1> B = {{0}, {1}};

    auto poles = std::array<wet::complex<double>, 2>{-3.0 + 0i, -4.0 + 0i}; // Desired closed-loop poles

    auto K_opt = matlab::place(A, B, poles);
    REQUIRE(K_opt.has_value());
    auto K = *K_opt;

    CHECK(K.rows() == 1);
    CHECK(K.cols() == 2);

    // Check that the closed-loop poles are approximately the desired ones
    // Closed-loop A_cl = A - B*K
    Matrix<2, 2> A_cl = A - B * K;

    // Compute characteristic polynomial of A_cl
    // For 2x2, det(sI - A_cl) = s^2 - trace*s + det
    double trace = A_cl(0, 0) + A_cl(1, 1);
    double det = A_cl(0, 0) * A_cl(1, 1) - A_cl(0, 1) * A_cl(1, 0);

    // Roots of s^2 + trace*s + det = 0 should be -3 and -4
    // Sum of roots = -trace = 3+4=7
    // Product = det = 12
    CHECK(doctest::Approx(trace).epsilon(1e-6) == -7.0);
    CHECK(doctest::Approx(det).epsilon(1e-6) == 12.0);
}

TEST_CASE("Place - Pole placement with std::complex poles") {
    // Simple 2x2 system
    Matrix<2, 2> A = {{0, 1}, {-1, -2}};
    Matrix<2, 1> B = {{0}, {1}};

    auto poles = std::array{std::complex<double>(-3, 0), std::complex<double>(-4, 0)}; // Desired closed-loop poles

    auto K_opt = matlab::place(A, B, poles);
    REQUIRE(K_opt.has_value());
    auto K = *K_opt;

    CHECK(K.rows() == 1);
    CHECK(K.cols() == 2);

    // Check that the closed-loop poles are approximately the desired ones
    Matrix<2, 2> A_cl = A - B * K;
    double       trace = A_cl(0, 0) + A_cl(1, 1);
    double       det = A_cl(0, 0) * A_cl(1, 1) - A_cl(0, 1) * A_cl(1, 0);
    CHECK(doctest::Approx(trace).epsilon(1e-6) == -7.0);
    CHECK(doctest::Approx(det).epsilon(1e-6) == 12.0);
}

TEST_CASE("Place - Pole placement with _Complex double poles") {
    // Simple 2x2 system
    Matrix<2, 2> A = {{0, 1}, {-1, -2}};
    Matrix<2, 1> B = {{0}, {1}};

    _Complex double p1 = -3.0;
    _Complex double p2 = -4.0;
    auto            poles = std::array{p1, p2}; // Desired closed-loop poles

    auto K_opt = matlab::place(A, B, poles);
    REQUIRE(K_opt.has_value());
    auto K = *K_opt;

    CHECK(K.rows() == 1);
    CHECK(K.cols() == 2);

    // Check that the closed-loop poles are approximately the desired ones
    Matrix<2, 2> A_cl = A - B * K;
    double       trace = A_cl(0, 0) + A_cl(1, 1);
    double       det = A_cl(0, 0) * A_cl(1, 1) - A_cl(0, 1) * A_cl(1, 0);
    CHECK(doctest::Approx(trace).epsilon(1e-6) == -7.0);
    CHECK(doctest::Approx(det).epsilon(1e-6) == 12.0);
}

TEST_CASE("Place - Pole placement with wet::complex poles") {
    // Simple 2x2 system
    Matrix<2, 2> A = {{0, 1}, {-1, -2}};
    Matrix<2, 1> B = {{0}, {1}};

    auto poles = std::array{wet::complex<double>(-3, 0), wet::complex<double>(-4, 0)}; // Desired closed-loop poles

    auto K_opt = matlab::place(A, B, poles);
    REQUIRE(K_opt.has_value());
    auto K = *K_opt;

    CHECK(K.rows() == 1);
    CHECK(K.cols() == 2);

    // Check that the closed-loop poles are approximately the desired ones
    Matrix<2, 2> A_cl = A - B * K;
    double       trace = A_cl(0, 0) + A_cl(1, 1);
    double       det = A_cl(0, 0) * A_cl(1, 1) - A_cl(0, 1) * A_cl(1, 0);
    CHECK(doctest::Approx(trace).epsilon(1e-6) == -7.0);
    CHECK(doctest::Approx(det).epsilon(1e-6) == 12.0);
}

TEST_CASE("PID Tune") {
    // Simple second-order plant: G(s) = 1/(s^2 + 0.5s + 1)
    Matrix<2, 2, double> A = {{0, 1}, {-1, -0.5}};
    Matrix<2, 1, double> B = {{0}, {1}};
    Matrix<1, 2, double> C = {{1, 0}};
    Matrix<1, 1, double> D = {{0}};

    StateSpace<2, 1, 1> sys{A, B, C, D};

    double wc = 1.0; // Desired crossover frequency

    auto pid_result = matlab::pidtune(sys, wc);

    CHECK(pid_result.Kp > 0);
    CHECK(pid_result.Ki > 0);
    CHECK(pid_result.Kd > 0);
    CHECK(pid_result.Ts == 0); // Continuous time
    CHECK(pid_result.Kbc > 0); // Back-calculation gain
}