#include <cmath>

#include "wet/controllers/pid.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wetmelon::control;

/**
 * @file test_pid_mode_control.cpp
 * @brief Runtime tests for PIDController's Auto / Tracking mode control and
 *        bumpless-transfer behavior. The static_assert audit in
 *        test_controller_concept.cpp covers the API surface; this file
 *        covers the math.
 */

TEST_SUITE("PID Mode Control") {
    TEST_CASE("PI: disable returns clamped u_track from control()") {
        PIDController<float, PIDMode::PI> pi{design::pid(
            1.0f, 1.0f, 0.0f, 0.01f,
            -5.0f, 5.0f // u clamps
        )};
        REQUIRE(pi.is_enabled());

        pi.disable(2.5f);
        CHECK(!pi.is_enabled());
        CHECK(pi.control(1.0f, 0.0f) == doctest::Approx(2.5f));

        // Clamping still applies in tracking mode.
        pi.disable(100.0f);
        CHECK(pi.control(1.0f, 0.0f) == doctest::Approx(5.0f));

        pi.disable(-100.0f);
        CHECK(pi.control(1.0f, 0.0f) == doctest::Approx(-5.0f));
    }

    TEST_CASE("PI: enable after disable produces bumpless re-engagement") {
        constexpr float                   Ts = 0.01f;
        PIDController<float, PIDMode::PI> pi{design::pid(2.0f, 5.0f, 0.0f, Ts)};

        // Step 1: run in Auto with a non-trivial trajectory so the integrator
        // accumulates an arbitrary value.
        for (int i = 0; i < 20; ++i) {
            (void)pi.control(1.0f, 0.5f);
        }
        REQUIRE(pi.integral != doctest::Approx(0.0f));

        // Step 2: an external command source takes over at u_track = 3.7.
        // While disabled, the PI runs tracking-mode preload every tick.
        const float u_track = 3.7f;
        pi.disable(u_track);
        for (int i = 0; i < 10; ++i) {
            const float u = pi.control(1.0f, 0.5f);
            CHECK(u == doctest::Approx(u_track));
        }

        // Step 3: re-enable and check the next Auto command matches u_track
        // to within numerical noise -- that's the bumpless transfer property.
        pi.enable();
        const float u_after = pi.control(1.0f, 0.5f);
        CHECK(u_after == doctest::Approx(u_track).epsilon(1e-5f));
    }

    TEST_CASE("PID: bumpless transfer with active derivative term") {
        constexpr float                    Ts = 0.01f;
        PIDController<float, PIDMode::PID> pid{design::pid(1.5f, 3.0f, 0.05f, Ts)};

        // Warm up with a varying y so the derivative state isn't trivial.
        for (int i = 0; i < 20; ++i) {
            const float y = 0.5f + (0.01f * static_cast<float>(i));
            (void)pid.control(1.0f, y);
        }

        // Sideline at u_track = -1.2. We hold (r, y) constant across the
        // tracking-to-auto boundary to isolate the bumpless property -- if
        // (r, y) changed during the transition, the controller would
        // correctly respond to that, which is not the property being tested
        // here.
        const float u_track = -1.2f;
        const float r_hold = 1.0f;
        const float y_hold = 0.7f;
        pid.disable(u_track);
        for (int i = 0; i < 20; ++i) {
            const float u = pid.control(r_hold, y_hold);
            CHECK(u == doctest::Approx(u_track));
        }

        // Re-engage at the same (r, y) -- the next Auto command should
        // reproduce u_track to within numerical noise.
        pid.enable();
        const float u_after = pid.control(r_hold, y_hold);
        CHECK(u_after == doctest::Approx(u_track).epsilon(1e-5f));
    }

    TEST_CASE("PID: tracking-mode integrator preload respects i_min/i_max") {
        constexpr float                    Ts = 0.01f;
        PIDController<float, PIDMode::PID> pid{design::pid(
            1.0f, 1.0f, 0.0f, Ts,
            -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity(),
            -2.0f, 2.0f // tight integrator limits
        )};

        // u_track that would require integral = (100 - 0) / 1 = 100, exceeding i_max.
        pid.disable(100.0f);
        (void)pid.control(0.0f, 0.0f);
        CHECK(pid.integral == doctest::Approx(2.0f)); // saturated at i_max

        pid.disable(-100.0f);
        (void)pid.control(0.0f, 0.0f);
        CHECK(pid.integral == doctest::Approx(-2.0f)); // saturated at i_min
    }

    TEST_CASE("P: tracking mode is degenerate but API works for generic code") {
        PIDController<float, PIDMode::P> p{design::pid(
            2.0f, 0.0f, 0.0f, 0.01f,
            -10.0f, 10.0f
        )};

        REQUIRE(p.is_enabled());
        p.disable(4.2f);
        CHECK(!p.is_enabled());

        // In tracking mode, output is u_track regardless of r, y.
        CHECK(p.control(0.0f, 0.0f) == doctest::Approx(4.2f));
        CHECK(p.control(1.0f, 0.5f) == doctest::Approx(4.2f));

        // Re-enabling gives bumpless transfer trivially (no state).
        p.enable();
        CHECK(p.is_enabled());
        CHECK(p.control(1.0f, 0.0f) == doctest::Approx(2.0f)); // Kp*(r-y) = 2.0
    }

    TEST_CASE("reset preserves the runtime mode") {
        PIDController<float, PIDMode::PI> pi{design::pid(1.0f, 1.0f, 0.0f, 0.01f)};

        pi.disable(1.5f);
        REQUIRE(!pi.is_enabled());

        pi.reset();
        // Mode is operator state, not numerical state -- reset() clears the
        // integrator/derivative but leaves the mode where it was.
        CHECK(!pi.is_enabled());
        CHECK(pi.integral == doctest::Approx(0.0f));
    }

    TEST_CASE("back_calculate and mode control compose: disable, then re-engage, then saturate") {
        constexpr float                   Ts = 0.01f;
        PIDController<float, PIDMode::PI> pi{design::pid(
            1.0f, 1.0f, 0.0f, Ts,
            -1.0f, 1.0f, // u clamps for back_calculate to bite
            -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity(),
            0.5f // Kbc
        )};

        // Sideline, then re-engage at u_track = 0.5.
        pi.disable(0.5f);
        (void)pi.control(2.0f, 0.0f);
        pi.enable();
        const float u_first = pi.control(2.0f, 0.0f);
        CHECK(u_first == doctest::Approx(0.5f).epsilon(1e-5f));

        // Now drive the loop further and see the integrator grow until back_calc
        // bites at the u_max = 1.0 rail. The point: mode control and back-calc
        // are independent and both work.
        for (int i = 0; i < 50; ++i) {
            (void)pi.control(5.0f, 0.0f);
        }
        const float u_saturated = pi.control(5.0f, 0.0f);
        CHECK(u_saturated == doctest::Approx(1.0f)); // clamped at u_max
    }
} // TEST_SUITE
