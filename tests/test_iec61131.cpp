#include <cmath>

#include "doctest.h"
#include "iec61131.hpp"

using namespace wetmelon::control::plc;

/**
 * @brief Tests for IEC61131-3 function blocks
 */

TEST_SUITE("IEC61131-3 Function Blocks") {
    TEST_CASE("SR Latch") {
        SR sr;

        // Initial state
        CHECK(sr.Q1 == false);

        // Set dominant
        CHECK(sr(true, false) == true);
        CHECK(sr.Q1 == true);

        // Reset dominant over set
        CHECK(sr(true, true) == false);
        CHECK(sr.Q1 == false);

        // Stay reset
        CHECK(sr(false, false) == false);
        CHECK(sr.Q1 == false);

        // Set again
        CHECK(sr(true, false) == true);
        CHECK(sr.Q1 == true);
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

} // TEST_SUITE