#include <cmath>

#include "wet/analysis.hpp"
#include "wet/design/stability.hpp"
#include "wet/matrix/matrix.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wet;

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
