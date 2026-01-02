#include <algorithm>
#include <complex>

#include "doctest.h"
#include "eigen.hpp"
#include "matrix.hpp"

TEST_CASE("Complex Matrix Support") {
    using Complex = std::complex<double>;

    SUBCASE("Complex matrix creation") {
        Matrix<2, 2, Complex> A;
        A(0, 0) = Complex{1.0, 2.0};
        A(0, 1) = Complex{3.0, -1.0};
        A(1, 0) = Complex{-2.0, 1.0};
        A(1, 1) = Complex{0.0, 1.0};

        CHECK(A(0, 0).real() == 1.0);
        CHECK(A(0, 0).imag() == 2.0);
    }
}

TEST_CASE("Eigen Decomposition") {
    SUBCASE("2x2 symmetric matrix") {
        Mat2 A = {{4.0, 1.0}, {1.0, 3.0}};
        auto result = compute_eigenvalues(A);

        CHECK(result.converged);

        auto lambda1 = result.values[0];
        auto lambda2 = result.values[1];

        CHECK(std::abs(lambda1.imag()) < 1e-6);
        CHECK(std::abs(lambda2.imag()) < 1e-6);

        double ev1 = lambda1.real();
        double ev2 = lambda2.real();

        if (ev1 < ev2)
            std::swap(ev1, ev2);

        CHECK(ev1 == doctest::Approx(4.618).epsilon(0.01));
        CHECK(ev2 == doctest::Approx(2.382).epsilon(0.01));
    }

    SUBCASE("Diagonal matrix") {
        Mat3<double> D = {{2.0, 0.0, 0.0}, {0.0, 5.0, 0.0}, {0.0, 0.0, -3.0}};

        auto result = compute_eigenvalues(D);
        CHECK(result.converged);

        std::array<double, 3> evs;
        for (size_t i = 0; i < 3; ++i) {
            evs[i] = result.values[i].real();
            CHECK(std::abs(result.values[i].imag()) < 1e-6);
        }

        std::sort(evs.begin(), evs.end());

        CHECK(evs[0] == doctest::Approx(-3.0));
        CHECK(evs[1] == doctest::Approx(2.0));
        CHECK(evs[2] == doctest::Approx(5.0));
    }

    SUBCASE("2x2 general matrix (Buck converter closed-loop)") {
        // This is approximately what the Buck converter A-B*K looks like
        // with large values typical of power electronics
        Mat2 A_cl = {{-3e4, -5e3}, {6.8e4, -6.8e3}};

        auto result = compute_eigenvalues(A_cl);
        CHECK(result.converged);

        // Both eigenvalues should be computed (non-zero)
        double ev1 = result.values[0].real();
        double ev2 = result.values[1].real();

        CHECK(ev1 != 0.0);
        CHECK(ev2 != 0.0);
        // trace = -3e4 - 6.8e3 = -36800, so sum of eigenvalues should be this
        CHECK(ev1 + ev2 == doctest::Approx(-36800.0).epsilon(0.01));
    }

    SUBCASE("2x2 consteval eigenvalue computation") {
        // Test that eigenvalues work in consteval context
        constexpr Mat2 A_cl = {{-3e4, -5e3}, {6.8e4, -6.8e3}};
        constexpr auto result = compute_eigenvalues(A_cl);

        static_assert(result.converged, "Eigenvalue computation should converge");

        constexpr double ev1 = result.values[0].real();
        constexpr double ev2 = result.values[1].real();

        // At compile time, verify eigenvalues are non-zero
        static_assert(ev1 != 0.0 || ev2 != 0.0, "At least one eigenvalue should be non-zero");

        CHECK(ev1 != 0.0);
        CHECK(ev2 != 0.0);
        CHECK(ev1 + ev2 == doctest::Approx(-36800.0).epsilon(0.01));
    }

    SUBCASE("2x2 consteval closed-loop poles") {
        // Simulate a simple closed-loop system
        constexpr Mat2         A = {{0.0, -5000.0}, {68027.21, -6802.72}}; // Buck converter approx
        constexpr Matrix<2, 1> B = {{5000.0}, {0.0}};
        constexpr Matrix<1, 2> K = {{1.0, 0.1}}; // Simple gain

        constexpr auto poles = compute_closed_loop_poles(A, B, K);

        // Poles should be non-zero
        CHECK(poles[0] != 0.0);
        CHECK(poles[1] != 0.0);
    }
}
