#include "control.hpp"
#include "doctest.h"

using namespace control;

TEST_CASE("Wrap test") {
    SUBCASE("Basic wrapping within range") {
        double x       = 7.0;
        double wrapped = wrap(x, 0.0, 5.0);
        CHECK(wrapped == doctest::Approx(2.0));
    }

    SUBCASE("Negative values wrapping") {
        double x       = -3.0;
        double wrapped = wrap(x, 0.0, 5.0);
        CHECK(wrapped == doctest::Approx(2.0));
    }

    SUBCASE("Values exactly at bounds") {
        double x1       = 0.0;
        double wrapped1 = wrap(x1, 0.0, 5.0);
        CHECK(wrapped1 == doctest::Approx(0.0));

        double x2       = 5.0;
        double wrapped2 = wrap(x2, 0.0, 5.0);
        CHECK(wrapped2 == doctest::Approx(0.0));
    }

    SUBCASE("Large values wrapping") {
        double x       = 23.5;
        double wrapped = wrap(x, -10.0, 10.0);
        CHECK(wrapped == doctest::Approx(3.5));
    }

    SUBCASE("Small values wrapping") {
        double x       = -27.3;
        double wrapped = wrap(x, -10.0, 10.0);
        CHECK(wrapped == doctest::Approx(-7.3));
    }
}

TEST_CASE("Linspace tests") {
    SUBCASE("Basic linspace") {
        auto v = linspace(0.0, 1.0, 5);
        REQUIRE(v.size() == 5);
        CHECK(v[0] == doctest::Approx(0.0));
        CHECK(v[1] == doctest::Approx(0.25));
        CHECK(v[2] == doctest::Approx(0.5));
        CHECK(v[3] == doctest::Approx(0.75));
        CHECK(v[4] == doctest::Approx(1.0));
    }

    SUBCASE("Single element linspace") {
        auto v = linspace(3.14, 2.71, 1);
        REQUIRE(v.size() == 1);
        CHECK(v[0] == doctest::Approx(3.14));
    }

    SUBCASE("Pair overload") {
        auto v = linspace(std::pair<double, double>{-1.0, 1.0}, 3);
        REQUIRE(v.size() == 3);
        CHECK(v[0] == doctest::Approx(-1.0));
        CHECK(v[1] == doctest::Approx(0.0));
        CHECK(v[2] == doctest::Approx(1.0));
    }
}

TEST_CASE("Logspace tests") {
    SUBCASE("Base-10 logspace") {
        auto v = logspace(1.0, 100.0, 3, 10.0);
        REQUIRE(v.size() == 3);
        CHECK(v[0] == doctest::Approx(1.0));
        CHECK(v[1] == doctest::Approx(10.0));
        CHECK(v[2] == doctest::Approx(100.0));
    }

    SUBCASE("Base-2 logspace") {
        auto v = logspace(1.0, 8.0, 4, 2.0);
        REQUIRE(v.size() == 4);
        CHECK(v[0] == doctest::Approx(1.0));
        CHECK(v[1] == doctest::Approx(2.0));
        CHECK(v[2] == doctest::Approx(4.0));
        CHECK(v[3] == doctest::Approx(8.0));
    }

    SUBCASE("Pair overload") {
        auto v = logspace(std::pair<double, double>{1.0, 1000.0}, 4, 10.0);
        REQUIRE(v.size() == 4);
        CHECK(v.front() == doctest::Approx(1.0));
        CHECK(v.back() == doctest::Approx(1000.0));
    }
}

TEST_CASE("Magnitude/decibel and angle conversions") {
    SUBCASE("mag2db / db2mag roundtrip") {
        double mag = 0.5;
        double db  = mag2db(mag);
        CHECK(db == doctest::Approx(-6.0206).epsilon(1e-4));
        CHECK(db2mag(db) == doctest::Approx(mag).epsilon(1e-12));
    }

    SUBCASE("deg2rad / rad2deg roundtrip") {
        double deg = 123.456;
        double rad = deg2rad(deg);
        CHECK(rad2deg(rad) == doctest::Approx(deg).epsilon(1e-12));
    }
}