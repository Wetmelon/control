#include "doctest.h"
#include "wet/utility/lookup.hpp"

using namespace wet;

// Breakpoint lookup tables: 1-D linear/nearest + 2-D bilinear.

TEST_SUITE("Lookup tables") {
    TEST_CASE("Lut1D hits breakpoints and interpolates between") {
        constexpr Lut1D<3, double> lut{{0.0, 10.0, 20.0}, {0.0, 100.0, 0.0}};
        // Exact breakpoints.
        CHECK(lut(0.0) == doctest::Approx(0.0));
        CHECK(lut(10.0) == doctest::Approx(100.0));
        CHECK(lut(20.0) == doctest::Approx(0.0));
        // Linear interpolation within segments.
        CHECK(lut(5.0) == doctest::Approx(50.0));
        CHECK(lut(15.0) == doctest::Approx(50.0));
    }

    TEST_CASE("Lut1D out-of-range: clamp vs linear extrapolation") {
        constexpr Lut1D<2, double> clamp_lut{{0.0, 10.0}, {0.0, 100.0}, Extrapolation::Clamp};
        CHECK(clamp_lut(-5.0) == doctest::Approx(0.0));   // held at first value
        CHECK(clamp_lut(15.0) == doctest::Approx(100.0)); // held at last value

        constexpr Lut1D<2, double> lin_lut{{0.0, 10.0}, {0.0, 100.0}, Extrapolation::Linear};
        CHECK(lin_lut(-5.0) == doctest::Approx(-50.0)); // slope continues
        CHECK(lin_lut(15.0) == doctest::Approx(150.0));
    }

    TEST_CASE("Lut1D nearest-neighbour") {
        constexpr Lut1D<3, double> lut{{0.0, 10.0, 20.0}, {1.0, 2.0, 3.0}};
        CHECK(lut.nearest(2.0) == doctest::Approx(1.0));  // closer to 0
        CHECK(lut.nearest(8.0) == doctest::Approx(2.0));  // closer to 10
        CHECK(lut.nearest(16.0) == doctest::Approx(3.0)); // closer to 20
    }

    TEST_CASE("Lut2D bilinear over a unit grid") {
        constexpr Lut2D<2, 2, double> lut{
            {0.0, 1.0}, // rows
            {0.0, 1.0}, // cols
            {{0.0, 10.0}, {20.0, 30.0}}
        };
        // Corners are the grid values.
        CHECK(lut(0.0, 0.0) == doctest::Approx(0.0));
        CHECK(lut(0.0, 1.0) == doctest::Approx(10.0));
        CHECK(lut(1.0, 0.0) == doctest::Approx(20.0));
        CHECK(lut(1.0, 1.0) == doctest::Approx(30.0));
        // Center is the average of all four.
        CHECK(lut(0.5, 0.5) == doctest::Approx(15.0));
        // Edge midpoints.
        CHECK(lut(0.0, 0.5) == doctest::Approx(5.0));
        CHECK(lut(0.5, 0.0) == doctest::Approx(10.0));
        // Out of range clamps to the grid edge.
        CHECK(lut(-1.0, -1.0) == doctest::Approx(0.0));
        CHECK(lut(2.0, 2.0) == doctest::Approx(30.0));
    }
}
