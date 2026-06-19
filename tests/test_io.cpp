#include "wet/utility/io.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wet;

/**
 * @brief Tests for the tier-2 I/O appliances (io.hpp).
 */

TEST_SUITE("Tier-2 IO appliances") {

    TEST_CASE("AnalogInput validates raw then scales to engineering units") {
        // 0.5–4.5 V -> 0–100 %, with NE43 fault bands [0.25 (0.5 4.5) 4.75].
        io::AnalogInput<double> ai{
            two_point_cal(0.5, 0.0, 4.5, 100.0),
            RangeMonitor<double>{0.25, 0.5, 4.5, 4.75}
        };

        CHECK(ai.update(2.5) == doctest::Approx(50.0)); // mid-scale
        CHECK(ai.valid() == true);

        ai.update(0.1); // below fault floor
        CHECK(ai.faulted() == true);
        CHECK(ai.status() == SignalStatus::FaultLow);

        ai.update(0.3); // saturated low, not a wire fault
        CHECK(ai.faulted() == false);
        CHECK(ai.status() == SignalStatus::UnderRange);
    }

    TEST_CASE("AxisInput chains cal, dead zone, expo, and scale") {
        // Identity cal (already normalized), 10% dead zone, k=1 expo, unit scale.
        io::AxisInput<double> axis{AffineCal<double>{1.0, 0.0}, 0.1, 1.0, 1.0};

        CHECK(axis.update(0.05) == doctest::Approx(0.0));  // inside dead zone
        CHECK(axis.update(1.0) == doctest::Approx(1.0));   // full deflection -> full output
        CHECK(axis.update(-1.0) == doctest::Approx(-1.0)); // symmetric

        // No dead zone / no expo / scaled & inverted: pure -2x.
        io::AxisInput<double> inv{AffineCal<double>{1.0, 0.0}, 0.0, 0.0, 2.0, /*invert=*/true};
        CHECK(inv.update(0.5) == doctest::Approx(-1.0));
    }

    TEST_CASE("Button reports edges, level, and hold") {
        io::Button<double> btn{0.02, 0.5}; // 20 ms debounce, 500 ms hold
        const double       dt = 0.01;

        btn.update(true, dt); // 10 ms, not debounced yet
        CHECK(btn.down() == false);
        btn.update(true, dt); // 20 ms -> commits down
        CHECK(btn.down() == true);
        CHECK(btn.pressed() == true); // rising edge this tick
        btn.update(true, dt);
        CHECK(btn.pressed() == false); // edge is one-shot
        CHECK(btn.held() == false);    // not held long enough

        // Hold it down past 500 ms total.
        for (int i = 0; i < 60; ++i) {
            btn.update(true, dt);
        }
        CHECK(btn.held() == true);

        // Release: falling edge, then down() clears after debounce.
        btn.update(false, dt);
        btn.update(false, dt);
        CHECK(btn.down() == false);
        CHECK(btn.released() == true);
    }

    TEST_CASE("Switch debounces and flags changes") {
        io::Switch<double> sw{0.02}; // 20 ms debounce
        const double       dt = 0.01;

        CHECK(sw.update(false, dt) == false);
        sw.update(true, dt);                // 10 ms, not yet
        CHECK(sw.update(true, dt) == true); // 20 ms -> on
        CHECK(sw.changed() == true);        // flipped this tick
        CHECK(sw.update(true, dt) == true);
        CHECK(sw.changed() == false); // steady

        // A glitch shorter than the debounce window is ignored.
        sw.update(false, dt);
        CHECK(sw.on() == true); // single false sample didn't flip it
        CHECK(sw.changed() == false);
    }
}
