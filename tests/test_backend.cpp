#include "doctest.h"
#include "wet/backend.hpp"

using namespace wet;

/**
 * @brief Tests for the backend-agnostic min/max/minmax value helpers.
 *
 * Guards the initializer_list overloads in particular: ETL's min/max/minmax are
 * binary-only and dangle (return references), so wet supplies by-value versions.
 * These must behave identically on both backends.
 */
TEST_SUITE("backend min/max/minmax") {
    TEST_CASE("minmax(a, b) orders the pair and is stable on ties") {
        CHECK(wet::minmax(3, 1) == pair{1, 3});
        CHECK(wet::minmax(1, 3) == pair{1, 3});
        CHECK(wet::minmax(2, 2) == pair{2, 2});
    }

    TEST_CASE("minmax(initializer_list) finds the extremes") {
        const auto [lo, hi] = wet::minmax({3.0, 1.0, 2.0, 5.0, -1.0});
        CHECK(lo == doctest::Approx(-1.0));
        CHECK(hi == doctest::Approx(5.0));
        // constexpr-usable
        static_assert(wet::minmax({3, 1, 2}).first == 1);
        static_assert(wet::minmax({3, 1, 2}).second == 3);
    }

    TEST_CASE("min/max(initializer_list) match the std/etl-parity contract") {
        CHECK(wet::min({4, 2, 7, 1, 9}) == 1);
        CHECK(wet::max({4, 2, 7, 1, 9}) == 9);
        static_assert(wet::min({4, 2, 7}) == 2);
        static_assert(wet::max({4, 2, 7}) == 7);
    }

    TEST_CASE("single-element lists return that element") {
        CHECK(wet::min({42}) == 42);
        CHECK(wet::max({42}) == 42);
        CHECK(wet::minmax({42}) == pair{42, 42});
    }
}
