#include <cmath>

#include "wet/analysis.hpp"
#include "wet/design/stability.hpp"
#include "wet/matrix/matrix.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wet;

TEST_CASE("poles() handles systems larger than 4 states") {
    // Diagonal A with known eigenvalues -1..-6 (6 states > old NX<=4 limit).
    Matrix<6, 6, double> A{};
    const double         expected[6] = {-1.0, -2.0, -3.0, -4.0, -5.0, -6.0};
    for (size_t i = 0; i < 6; ++i) {
        A(i, i) = expected[i];
    }
    auto p = analysis::poles(A);
    for (size_t i = 0; i < 6; ++i) {
        CHECK(p[i].real() == doctest::Approx(expected[i]).epsilon(1e-9));
        CHECK(p[i].imag() == doctest::Approx(0.0).epsilon(1e-9));
    }
    CHECK(analysis::is_stable_continuous(A));
}

TEST_CASE("step/impulse/initial response of 1/(s+1)") {
    // 1/(s+1): A=-1, B=1, C=1, D=0
    TransferFunction<1, 2, double> tf{{1.0}, {1.0, 1.0}};
    auto                           t = analysis::linspace(0.0, 5.0, 501);

    auto s = analysis::step(tf, t);
    auto i = analysis::impulse(tf, t);
    for (size_t k = 0; k < t.size(); ++k) {
        CHECK(s.y[k] == doctest::Approx(1.0 - std::exp(-t[k])).epsilon(1e-3));
        CHECK(i.y[k] == doctest::Approx(std::exp(-t[k])).epsilon(1e-3));
    }

    // initial response from x0=2: y(t) = 2 e^{-t}
    auto                 ss = tf.to_state_space();
    Matrix<1, 1, double> x0{};
    x0(0, 0) = 2.0;
    auto in = analysis::initial(ss, x0, t);
    for (size_t k = 0; k < t.size(); ++k) {
        CHECK(in.y[k] == doctest::Approx(2.0 * std::exp(-t[k])).epsilon(1e-3));
    }
}

TEST_CASE("linspace and logspace support float") {
    const auto lin = analysis::linspace(0.0f, 1.0f, 5);
    const auto log = analysis::logspace(1.0f, 100.0f, 3);

    CHECK(std::is_same_v<typename decltype(lin)::value_type, float>);
    CHECK(std::is_same_v<typename decltype(log)::value_type, float>);

    CHECK(lin.front() == doctest::Approx(0.0f));
    CHECK(lin.back() == doctest::Approx(1.0f));
    CHECK(log.front() == doctest::Approx(1.0f));
    CHECK(log.back() == doctest::Approx(100.0f));
}

TEST_SUITE("Stability Analysis") {
    TEST_CASE("Discrete stability check - stable system") {
        // Stable discrete system: eigenvalues inside unit circle
        Matrix<2, 2> A{{0.5, 0.0}, {0.0, 0.8}};
        CHECK(stability::is_stable_discrete(A) == true);
        CHECK(stability::stability_margin_discrete(A) > 0.0);
    }

    TEST_CASE("Discrete stability check - unstable system") {
        // Unstable: eigenvalue outside unit circle
        Matrix<2, 2> A{{1.5, 0.0}, {0.0, 0.5}};
        CHECK(stability::is_stable_discrete(A) == false);
        CHECK(stability::stability_margin_discrete(A) < 0.0);
    }
}
