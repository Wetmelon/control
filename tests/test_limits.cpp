
#include "doctest.h"
#include "wet/power/limits.hpp"
#include "wet/power/transforms.hpp"

using namespace wet;

TEST_SUITE("DC Bus Limiter") {

    using DQ = DirectQuadrature<double>;

    TEST_CASE("computes bus power and current") {
        DcBusLimiter<double> lim{}; // unbounded limits
        const auto           s = lim.evaluate(DQ{.d = 0.0, .q = 20.0}, DQ{.d = 0.0, .q = 10.0}, 48.0);
        CHECK(s.bus_power == doctest::Approx(1.5 * 20.0 * 10.0)); // 300 W
        CHECK(s.bus_current == doctest::Approx(300.0 / 48.0));    // 6.25 A
        CHECK(s.scale == doctest::Approx(1.0));                   // nothing binding
        CHECK(s.ok);
    }

    TEST_CASE("derates on the motoring power cap") {
        DcBusLimiter<double> lim{DcBusLimits<double>{.bus_power_max = 150.0}};
        const auto           s = lim.evaluate(DQ{.d = 0, .q = 20.0}, DQ{.d = 0, .q = 10.0}, 48.0); // 300 W
        CHECK(s.scale == doctest::Approx(150.0 / 300.0));                                          // 0.5
    }

    TEST_CASE("derates on the motoring current cap") {
        // bus_current = [unbounded floor, 3 A cap]
        DcBusLimiter<double> lim{DcBusLimits<double>{.bus_current = {-1e9, 3.0}}};
        const auto           s = lim.evaluate(DQ{.d = 0, .q = 20.0}, DQ{.d = 0, .q = 10.0}, 48.0); // I_bus = 6.25 A
        CHECK(s.scale == doctest::Approx(3.0 / 6.25));
    }

    TEST_CASE("derates regen against the bus-current floor") {
        DcBusLimiter<double> lim{DcBusLimits<double>{.bus_current = {-3.0, 1e9}}};
        // Braking: q-current opposes q-voltage -> negative power -> negative bus current.
        const auto s = lim.evaluate(DQ{.d = 0, .q = 20.0}, DQ{.d = 0, .q = -10.0}, 48.0); // I_bus = -6.25 A
        CHECK(s.bus_current == doctest::Approx(-6.25));
        CHECK(s.scale == doctest::Approx(3.0 / 6.25)); // -3 / -6.25
    }

    TEST_CASE("under/over-voltage raises the disarm gate") {
        DcBusLimiter<double> lim{DcBusLimits<double>{.voltage = {20.0, 55.0}}};
        CHECK_FALSE(lim.evaluate(DQ{.d = 0, .q = 1.0}, DQ{.d = 0, .q = 1.0}, 10.0).ok); // UV
        CHECK_FALSE(lim.evaluate(DQ{.d = 0, .q = 1.0}, DQ{.d = 0, .q = 1.0}, 60.0).ok); // OV
        CHECK(lim.evaluate(DQ{.d = 0, .q = 1.0}, DQ{.d = 0, .q = 1.0}, 48.0).ok);       // in range
    }

    TEST_CASE("unbounded defaults never derate") {
        DcBusLimiter<double> lim{};
        const auto           s = lim.evaluate(DQ{.d = 0, .q = 100.0}, DQ{.d = 0, .q = 500.0}, 48.0);
        CHECK(s.scale == doctest::Approx(1.0));
    }
}
