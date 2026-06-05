
#include "wet/utility.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wet;

TEST_SUITE("Utility") {
    TEST_CASE("unit conversion helpers support float") {
        const float deg = rad2deg(3.14159265358979323846f);
        const float rad = deg2rad(180.0f);

        CHECK(deg == doctest::Approx(180.0f).epsilon(1e-5));
        CHECK(rad == doctest::Approx(3.14159265358979323846f).epsilon(1e-5));
    }

    TEST_CASE("mag and db conversions support float") {
        const float db = mag2db(10.0f);
        const float mag = db2mag(20.0f);

        CHECK(db == doctest::Approx(20.0f).epsilon(1e-5));
        CHECK(mag == doctest::Approx(10.0f).epsilon(1e-5));
    }
}
