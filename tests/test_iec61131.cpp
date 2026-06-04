#include <cmath>

#include "wet/iec61131.hpp"

#define DOCTEST_CONFIG_INCLUDE_TYPE_TRAITS
#include "doctest.h"

using namespace wetmelon::control::plc;

/**
 * @brief Tests for IEC61131-3 function blocks
 */

TEST_SUITE("IEC61131-3 Function Blocks") {
    TEST_CASE("SR Latch (set-dominant per IEC 61131-3)") {
        SR sr;

        // Initial state
        CHECK(sr.Q1 == false);

        // Set
        CHECK(sr(true, false) == true);
        CHECK(sr.Q1 == true);

        // Reset
        CHECK(sr(false, true) == false);
        CHECK(sr.Q1 == false);

        // Set dominates when S and R are both true (this is what distinguishes
        // SR from RS).
        CHECK(sr(true, true) == true);
        CHECK(sr.Q1 == true);

        // Hold
        CHECK(sr(false, false) == true);
        CHECK(sr.Q1 == true);

        // Reset clears
        CHECK(sr(false, true) == false);
        CHECK(sr.Q1 == false);
    }

    TEST_CASE("SR vs RS differ only when S and R are both asserted") {
        SR sr;
        RS rs;
        sr(true, true); // SR: set wins
        rs(true, true); // RS: reset wins (args are (R, S))
        CHECK(sr.Q1 == true);
        CHECK(rs.Q1 == false);
    }

    TEST_CASE("RS Latch") {
        RS rs;

        // Initial state
        CHECK(rs.Q1 == false);

        // Set
        CHECK(rs(false, true) == true);
        CHECK(rs.Q1 == true);

        // Reset dominant over set
        CHECK(rs(true, true) == false);
        CHECK(rs.Q1 == false);

        // Stay reset
        CHECK(rs(false, false) == false);
        CHECK(rs.Q1 == false);

        // Set again
        CHECK(rs(false, true) == true);
        CHECK(rs.Q1 == true);
    }

    TEST_CASE("TON Timer") {
        TON<float> ton;
        ton.PT = 1.0f; // 1 second preset

        // Initial state
        CHECK(ton.Q == false);
        CHECK(ton.ET == 0.0f);

        // Timer not started
        CHECK(ton(false, 0.1f) == false);
        CHECK(ton.ET == 0.0f);

        // Start timer
        CHECK(ton(true, 0.1f) == false);
        CHECK(ton.ET == 0.1f);

        // Continue timing
        CHECK(ton(true, 0.1f) == false);
        CHECK(ton.ET == 0.2f);

        // Reach preset time
        for (int i = 0; i < 8; ++i) {
            ton(true, 0.1f);
        }
        CHECK(ton.ET == doctest::Approx(1.0f));
        CHECK(ton.Q == true);

        // Stay on
        CHECK(ton(true, 0.1f) == true);
        CHECK(ton.ET == doctest::Approx(1.1f));

        // Reset on input false
        CHECK(ton(false, 0.1f) == false);
        CHECK(ton.ET == 0.0f);
        CHECK(ton.Q == false);
    }

    TEST_CASE("TOF Timer") {
        TOF<float> tof;
        tof.PT = 1.0f; // 1 second preset

        // Initial state
        CHECK(tof.Q == false);
        CHECK(tof.ET == 0.0f);

        // Input on
        CHECK(tof(true, 0.1f) == true);
        CHECK(tof.ET == 0.0f);

        // Input off - start timing
        CHECK(tof(false, 0.1f) == true);
        CHECK(tof.ET == 0.1f);

        // Continue timing
        CHECK(tof(false, 0.1f) == true);
        CHECK(tof.ET == 0.2f);

        // Reach preset time
        for (int i = 0; i < 8; ++i) {
            tof(false, 0.1f);
        }
        CHECK(tof.ET == doctest::Approx(1.0f));
        CHECK(tof.Q == false);

        // Stay off
        CHECK(tof(false, 0.1f) == false);
        CHECK(tof.ET == doctest::Approx(1.1f));

        // Reset on input true
        CHECK(tof(true, 0.1f) == true);
        CHECK(tof.ET == 0.0f);
        CHECK(tof.Q == true);
    }

    TEST_CASE("TP Timer") {
        TP<float> tp;
        tp.PT = 1.0f; // 1 second pulse

        // Initial state
        CHECK(tp.Q == false);
        CHECK(tp.ET == 0.0f);

        // Rising edge triggers pulse
        CHECK(tp(true, 0.1f) == true);
        CHECK(tp.ET == 0.1f); // Timer starts counting immediately

        // Continue pulse
        CHECK(tp(true, 0.1f) == true);
        CHECK(tp.ET == 0.2f);

        // Pulse ends
        for (int i = 0; i < 8; ++i) {
            tp(true, 0.1f);
        }
        CHECK(tp.ET == doctest::Approx(1.0f));
        CHECK(tp.Q == false);

        // No pulse on steady input
        CHECK(tp(true, 0.1f) == false);
        CHECK(tp.ET == doctest::Approx(1.0f)); // ET stays at PT

        // New rising edge
        CHECK(tp(false, 0.1f) == false);
        CHECK(tp(true, 0.1f) == true);
        CHECK(tp.ET == 0.1f); // New pulse starts
    }

    TEST_CASE("CTU Counter") {
        CTU<uint32_t> ctu;
        ctu.PV = 5;

        // Initial state
        CHECK(ctu.CV == 0);
        CHECK(ctu.Q == false);

        // Count up
        CHECK(ctu(true, false) == false); // Rising edge
        CHECK(ctu.CV == 1);

        CHECK(ctu(false, false) == false); // No edge
        CHECK(ctu.CV == 1);

        CHECK(ctu(true, false) == false); // Rising edge
        CHECK(ctu.CV == 2);

        // Reach preset
        ctu(false, false);                // Reset CU to false
        CHECK(ctu(true, false) == false); // CV = 3
        CHECK(ctu.CV == 3);
        ctu(false, false);                // Reset CU to false
        CHECK(ctu(true, false) == false); // CV = 4
        CHECK(ctu.CV == 4);
        ctu(false, false);               // Reset CU to false
        CHECK(ctu(true, false) == true); // CV = 5, Q = true
        CHECK(ctu.CV == 5);
        CHECK(ctu.Q == true);

        // Reset
        CHECK(ctu(false, true) == false);
        CHECK(ctu.CV == 0);
        CHECK(ctu.Q == false);
    }

    TEST_CASE("CTD Counter") {
        CTD<uint32_t> ctd;
        ctd.PV = 5;

        // Load
        CHECK(ctd(false, true) == false);
        CHECK(ctd.CV == 5);

        // Count down
        CHECK(ctd(true, false) == false); // Rising edge
        CHECK(ctd.CV == 4);

        CHECK(ctd(false, false) == false); // No edge
        CHECK(ctd.CV == 4);

        CHECK(ctd(true, false) == false); // Rising edge
        CHECK(ctd.CV == 3);

        // Reach zero
        ctd(false, false);                // Reset CD to false
        CHECK(ctd(true, false) == false); // CV = 2
        CHECK(ctd.CV == 2);
        ctd(false, false);                // Reset CD to false
        CHECK(ctd(true, false) == false); // CV = 1
        CHECK(ctd.CV == 1);
        ctd(false, false);               // Reset CD to false
        CHECK(ctd(true, false) == true); // CV = 0, Q = true
        CHECK(ctd.CV == 0);
        CHECK(ctd.Q == true);

        // Stay at zero
        CHECK(ctd(true, false) == true);
        CHECK(ctd.CV == 0);
    }

    TEST_CASE("CTUD Counter") {
        CTUD<uint32_t> ctud;
        ctud.PV = 3;

        // Initial state
        CHECK(ctud.CV == 0);
        CHECK(ctud.QU == false);
        CHECK(ctud.QD == true); // CV <= 0

        // Load
        ctud(false, false, false, true);
        CHECK(ctud.CV == 3);
        CHECK(ctud.QU == true); // CV >= PV
        CHECK(ctud.QD == false);

        // Count down
        ctud(false, false, false, false); // Reset edges
        ctud(false, true, false, false);  // CD rising edge
        CHECK(ctud.CV == 2);
        CHECK(ctud.QU == false);
        CHECK(ctud.QD == false);

        // Count up
        ctud(false, false, false, false); // Reset edges
        ctud(true, false, false, false);  // CU rising edge
        CHECK(ctud.CV == 3);
        CHECK(ctud.QU == true);
        CHECK(ctud.QD == false);

        // Reset
        ctud(false, false, true, false);
        CHECK(ctud.CV == 0);
        CHECK(ctud.QU == false);
        CHECK(ctud.QD == true);
    }

    TEST_CASE("R_TRIG Rising Edge Detector") {
        R_TRIG rtrig;

        // Initial state
        CHECK(rtrig.Q == false);

        // No edge
        CHECK(rtrig(false) == false);
        CHECK(rtrig.Q == false);

        // Rising edge
        CHECK(rtrig(true) == true);
        CHECK(rtrig.Q == true);

        // Stay high, no edge
        CHECK(rtrig(true) == false);
        CHECK(rtrig.Q == false);

        // Falling edge, no rising
        CHECK(rtrig(false) == false);
        CHECK(rtrig.Q == false);

        // Another rising edge
        CHECK(rtrig(true) == true);
        CHECK(rtrig.Q == true);
    }

    TEST_CASE("F_TRIG Falling Edge Detector") {
        F_TRIG ftrig;

        // Initial state
        CHECK(ftrig.Q == false);

        // No edge
        CHECK(ftrig(true) == false);
        CHECK(ftrig.Q == false);

        // Falling edge
        CHECK(ftrig(false) == true);
        CHECK(ftrig.Q == true);

        // Stay low, no edge
        CHECK(ftrig(false) == false);
        CHECK(ftrig.Q == false);

        // Rising edge, no falling
        CHECK(ftrig(true) == false);
        CHECK(ftrig.Q == false);

        // Another falling edge
        CHECK(ftrig(false) == true);
        CHECK(ftrig.Q == true);
    }

    TEST_CASE("DFF captures D on rising edge only") {
        DFF dff;
        CHECK(dff(true, false) == false); // no edge, D ignored
        CHECK(dff(true, true) == true);   // rising edge captures D=1
        CHECK(dff(false, true) == true);  // CLK high, no edge: holds
        CHECK(dff(false, false) == true); // CLK low: holds
        CHECK(dff(false, true) == false); // rising edge captures D=0
    }

    TEST_CASE("DLATCH is transparent while enabled, holds otherwise") {
        DLATCH d;
        CHECK(d(true, true) == true);   // enabled: follows D
        CHECK(d(false, false) == true); // disabled: holds last
        CHECK(d(false, true) == false); // enabled: follows D
        CHECK(d(true, false) == false); // disabled: holds
    }

    TEST_CASE("TFF toggles on each rising edge") {
        TFF t;
        CHECK(t(false) == false);
        CHECK(t(true) == true);  // rising edge -> toggle
        CHECK(t(true) == true);  // held high, no edge
        CHECK(t(false) == true); // falling, no toggle
        CHECK(t(true) == false); // rising edge -> toggle back
    }

    TEST_CASE("BLINK free-runs with independent on/off times when enabled") {
        BLINK<double> bl{0.020, 0.030}; // 20 ms on, 30 ms off
        const double  dt = 0.010;

        // Starts low (off phase). Off for 30 ms then flips high.
        CHECK(bl(true, dt) == false); // 10
        CHECK(bl(true, dt) == false); // 20
        CHECK(bl(true, dt) == true);  // 30 ms -> high
        // High for 20 ms then flips low.
        CHECK(bl(true, dt) == true);  // 10
        CHECK(bl(true, dt) == false); // 20 ms -> low
        // Disable holds low and resets phase.
        CHECK(bl(false, dt) == false);
    }
} // TEST_SUITE