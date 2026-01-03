#include <cmath>

#include "control_design.hpp"
#include "doctest.h"

using namespace wetmelon::control;

/**
 * @brief Tests for stability analysis functions
 *
 * Verifies stability checking and margin calculations for discrete systems.
 */

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
