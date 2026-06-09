#include <algorithm>
#include <array>

#include "wet/controllers/pole_placement.hpp"
#include "wet/matlab.hpp"
#include "wet/matrix/eigen.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wet;

namespace {

// Largest distance between the (sorted, real) spectrum of A − B·K and the
// requested real poles.
template<size_t NX, size_t NU>
double placement_error(const Matrix<NX, NX>& A, const Matrix<NX, NU>& B, const std::array<double, NX>& poles) {
    auto Kopt = design::place(A, B, poles);
    REQUIRE(Kopt.has_value());
    Matrix<NX, NX> ABK = A - (B * Kopt.value());
    auto           r = mat::compute_eigenvalues_qr(ABK);

    std::array<double, NX> got;
    for (size_t i = 0; i < NX; ++i) {
        got[i] = r.eigenvalues_real(i, i);
    }
    std::array<double, NX> want = poles;
    std::sort(got.begin(), got.end());
    std::sort(want.begin(), want.end());
    double worst = 0.0;
    for (size_t i = 0; i < NX; ++i) {
        worst = std::max(worst, std::abs(got[i] - want[i]));
    }
    return worst;
}

} // namespace

TEST_SUITE("pole_placement") {
    TEST_CASE("single-input placement assigns the spectrum exactly") {
        SUBCASE("double integrator, poles {-1, -2}") {
            Matrix<2, 2> A = {{0.0, 1.0}, {0.0, 0.0}};
            Matrix<2, 1> B = {{0.0}, {1.0}};
            CHECK(placement_error(A, B, {-1.0, -2.0}) < 1e-9);
        }
        SUBCASE("3x3 companion, poles {-2, -3, -4}") {
            Matrix<3, 3> A = {{0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {-1.0, -2.0, -3.0}};
            Matrix<3, 1> B = {{0.0}, {0.0}, {1.0}};
            CHECK(placement_error(A, B, {-2.0, -3.0, -4.0}) < 1e-9);
        }
    }

    TEST_CASE("multi-input placement assigns the spectrum exactly") {
        SUBCASE("3x2") {
            Matrix<3, 3> A = {{1.0, 2.0, 0.0}, {0.0, 1.0, 1.0}, {0.0, 0.0, 1.0}};
            Matrix<3, 2> B = {{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
            CHECK(placement_error(A, B, {-1.0, -2.0, -3.0}) < 1e-8);
        }
        SUBCASE("4x2") {
            Matrix<4, 4> A = {{0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {0.0, 0.0, 0.0, 1.0}, {1.0, 2.0, 3.0, 4.0}};
            Matrix<4, 2> B = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 1.0}};
            CHECK(placement_error(A, B, {-1.0, -2.0, -3.0, -4.0}) < 1e-8);
        }
        SUBCASE("repeated pole with multiplicity == NU is assignable") {
            Matrix<4, 4> A = {{0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {0.0, 0.0, 0.0, 1.0}, {1.0, 2.0, 3.0, 4.0}};
            Matrix<4, 2> B = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 1.0}};
            CHECK(placement_error(A, B, {-2.0, -2.0, -3.0, -3.0}) < 1e-7);
        }
    }

    TEST_CASE("square B (NU == NX) places via direct inverse") {
        Matrix<2, 2> A = {{1.0, 2.0}, {3.0, 4.0}};
        Matrix<2, 2> B = Matrix<2, 2>::identity();
        CHECK(placement_error(A, B, {-1.0, -5.0}) < 1e-10);
    }

    TEST_CASE("single-input place matches Ackermann (unique SI gain)") {
        Matrix<3, 3> A = {{0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {-1.0, -2.0, -3.0}};
        Matrix<3, 1> B = {{0.0}, {0.0}, {1.0}};
        auto         Kp = design::place(A, B, std::array<double, 3>{-2.0, -3.0, -4.0});
        REQUIRE(Kp.has_value());

        std::array<wet::complex<double>, 3> pc = {
            wet::complex<double>(-2.0, 0.0), wet::complex<double>(-3.0, 0.0), wet::complex<double>(-4.0, 0.0)
        };
        auto Ka = matlab::acker(A, B, pc);
        REQUIRE(Ka.has_value());

        for (size_t j = 0; j < 3; ++j) {
            CHECK(Kp.value()(0, j) == doctest::Approx(Ka.value()(0, j)).epsilon(1e-9));
        }
    }

    TEST_CASE("place rejects non-assignable problems") {
        Matrix<3, 3> A = {{0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {-1.0, -2.0, -3.0}};

        SUBCASE("multiplicity exceeding the input count (single input, triple pole)") {
            Matrix<3, 1> B = {{0.0}, {0.0}, {1.0}};
            CHECK_FALSE(design::place(A, B, std::array<double, 3>{-2.0, -2.0, -2.0}).has_value());
        }
        SUBCASE("rank-deficient B") {
            Matrix<3, 2> B = {{1.0, 2.0}, {1.0, 2.0}, {1.0, 2.0}}; // rank 1
            CHECK_FALSE(design::place(A, B, std::array<double, 3>{-1.0, -2.0, -3.0}).has_value());
        }
    }

    // Largest distance between the spectrum of A − B·K and a complex pole set,
    // comparing as sorted (real, |imag|) pairs.
    static constexpr auto placement_error_cplx = []<size_t NX, size_t NU>(
                                                     const Matrix<NX, NX>&                       A,
                                                     const Matrix<NX, NU>&                       B,
                                                     const std::array<wet::complex<double>, NX>& poles
                                                 ) {
        auto Kopt = design::place(A, B, poles);
        REQUIRE(Kopt.has_value());
        Matrix<NX, NX> ABK = A - (B * Kopt.value());
        auto           r = mat::compute_eigenvalues_qr(ABK);

        std::array<std::pair<double, double>, NX> got;
        std::array<std::pair<double, double>, NX> want;
        for (size_t i = 0; i < NX; ++i) {
            got[i] = {r.eigenvalues_real(i, i), std::abs(r.eigenvalues_imag(i, i))};
            want[i] = {poles[i].real(), std::abs(poles[i].imag())};
        }
        std::sort(got.begin(), got.end());
        std::sort(want.begin(), want.end());
        double worst = 0.0;
        for (size_t i = 0; i < NX; ++i) {
            worst = std::max(worst, std::abs(got[i].first - want[i].first));
            worst = std::max(worst, std::abs(got[i].second - want[i].second));
        }
        return worst;
    };

    TEST_CASE("complex-conjugate placement assigns the spectrum exactly") {
        using C = wet::complex<double>;
        SUBCASE("3x2: one complex pair + one real") {
            Matrix<3, 3> A = {{1.0, 2.0, 0.0}, {0.0, 1.0, 1.0}, {0.0, 0.0, 1.0}};
            Matrix<3, 2> B = {{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
            CHECK(placement_error_cplx(A, B, std::array<C, 3>{C(-1, 2), C(-1, -2), C(-3, 0)}) < 1e-6);
        }
        SUBCASE("4x2: two complex pairs") {
            Matrix<4, 4> A = {{0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {0.0, 0.0, 0.0, 1.0}, {1.0, 2.0, 3.0, 4.0}};
            Matrix<4, 2> B = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 1.0}};
            CHECK(placement_error_cplx(A, B, std::array<C, 4>{C(-1, 1), C(-1, -1), C(-2, 3), C(-2, -3)}) < 1e-6);
        }
        SUBCASE("4x2: mixed real + complex pair") {
            Matrix<4, 4> A = {{0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {0.0, 0.0, 0.0, 1.0}, {1.0, 2.0, 3.0, 4.0}};
            Matrix<4, 2> B = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 1.0}};
            CHECK(placement_error_cplx(A, B, std::array<C, 4>{C(-1, 2), C(-1, -2), C(-3, 0), C(-4, 0)}) < 1e-6);
        }
        SUBCASE("square 2x2 complex pair") {
            Matrix<2, 2> A = {{1.0, 2.0}, {3.0, 4.0}};
            Matrix<2, 2> B = Matrix<2, 2>::identity();
            CHECK(placement_error_cplx(A, B, std::array<C, 2>{C(-1, 2), C(-1, -2)}) < 1e-9);
        }
        SUBCASE("all-real array forwards to the real (conditioned) path") {
            // Exercise the complex overload with an all-real spectrum.
            Matrix<3, 3> A3 = {{1.0, 2.0, 0.0}, {0.0, 1.0, 1.0}, {0.0, 0.0, 1.0}};
            Matrix<3, 2> B = {{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
            CHECK(placement_error_cplx(A3, B, std::array<C, 3>{C(-1, 0), C(-2, 0), C(-3, 0)}) < 1e-8);
        }
    }

    TEST_CASE("place rejects a dangling (unpaired) complex pole") {
        using C = wet::complex<double>;
        Matrix<3, 3> A = {{1.0, 2.0, 0.0}, {0.0, 1.0, 1.0}, {0.0, 0.0, 1.0}};
        Matrix<3, 2> B = {{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
        // -1+2j without its conjugate is not a real closed-loop spectrum.
        CHECK_FALSE(design::place(A, B, std::array<C, 3>{C(-1, 2), C(-3, 0), C(-4, 0)}).has_value());
    }

    TEST_CASE("matlab::place forwards to design::place") {
        using C = wet::complex<double>;
        Matrix<3, 3>     A = {{1.0, 2.0, 0.0}, {0.0, 1.0, 1.0}, {0.0, 0.0, 1.0}};
        Matrix<3, 2>     B = {{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
        std::array<C, 3> poles = {C(-1, 2), C(-1, -2), C(-3, 0)};

        auto Km = matlab::place(A, B, poles);
        auto Kd = design::place(A, B, poles);
        REQUIRE(Km.has_value());
        REQUIRE(Kd.has_value());
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                CHECK(Km.value()(i, j) == doctest::Approx(Kd.value()(i, j)));
            }
        }
    }

    TEST_CASE("place is constexpr-evaluable") {
        constexpr bool ok = []() consteval {
            Matrix<2, 2> A = {{0.0, 1.0}, {0.0, 0.0}};
            Matrix<2, 1> B = {{0.0}, {1.0}};
            auto         K = design::place(A, B, std::array<double, 2>{-1.0, -2.0});
            if (!K) {
                return false;
            }
            // Double-integrator with poles {-1,-2}: char poly s²+3s+2 ⇒ K = [2, 3].
            return wet::abs(K.value()(0, 0) - 2.0) < 1e-9 && wet::abs(K.value()(0, 1) - 3.0) < 1e-9;
        }();
        static_assert(ok, "place must work at compile time");
        CHECK(ok);
    }
}
