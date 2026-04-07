#pragma once

#include <cstdint>
#include <limits>

namespace wetmelon::control::plc {

typedef bool     BOOL;  //< Boolean type
typedef uint8_t  BYTE;  //< 8-bit unsigned integer
typedef uint16_t WORD;  //< 16-bit unsigned integer
typedef uint32_t DWORD; //< 32-bit unsigned integer
typedef uint64_t LWORD; //< 64-bit unsigned integer

typedef int8_t  SINT; //< Signed 8-bit integer
typedef int16_t INT;  //< Signed 16-bit integer
typedef int32_t DINT; //< Signed 32-bit integer
typedef int64_t LINT; //< Signed 64-bit integer

typedef uint8_t  USINT; //< Unsigned 8-bit integer
typedef uint16_t UINT;  //< Unsigned 16-bit integer
typedef uint32_t UDINT; //< Unsigned 32-bit integer
typedef uint64_t ULINT; //< Unsigned 64-bit integer
typedef float    REAL;  //< Single-precision floating point
typedef double   LREAL; //< Double-precision floating point

typedef char     CHAR;    //< Character type
typedef wchar_t  WCHAR;   //< Wide character type
typedef char*    STRING;  //< String type (pointer to char)
typedef wchar_t* WSTRING; //< Wide string type (pointer to wchar_t)

/**
 * @defgroup iec61131 IEC61131-3 Functions
 * @brief Standard PLC function blocks for embedded control
 *
 * Implementation of IEC61131-3 standard function blocks commonly used in
 * industrial control and embedded systems.
 */

/**
 * @brief SR Latch (Set-Reset Latch)
 *
 * Bistable function block with set and reset inputs.
 * Q1 = S1 OR (Q1 AND NOT R)
 */
struct SR {
    bool Q1{false}; //!< Current state

    /**
     * @brief Execute SR latch
     * @param S Set input
     * @param R Reset input (dominant over S)
     * @return Current state
     */
    constexpr bool operator()(bool S, bool R) {
        if (R) {
            Q1 = false;
        } else if (S) {
            Q1 = true;
        }
        return Q1;
    }
};

/**
 * @brief RS Latch (Reset-Set Latch)
 *
 * Bistable function block with reset and set inputs.
 * Q1 = NOT R AND (S OR Q1)
 */
struct RS {
    bool Q1{false}; //!< Current state

    /**
     * @brief Execute RS latch
     * @param R Reset input (dominant over S)
     * @param S Set input
     * @return Current state
     */
    constexpr bool operator()(bool R, bool S) {
        if (R) {
            Q1 = false;
        } else if (S) {
            Q1 = true;
        }
        return Q1;
    }
};

/**
 * @brief R_TRIG (Rising Edge Trigger)
 *
 * Detects rising edges on the CLK input.
 */
struct R_TRIG {
    bool CLK{false}; //!< Clock input
    bool Q{false};   //!< Output (true on rising edge)

    /**
     * @brief Execute edge detector
     * @param CLK_input Clock input
     * @return Edge detected
     */
    constexpr bool operator()(bool CLK_input) {
        Q = CLK_input && !CLK;
        CLK = CLK_input;
        return Q;
    }

    /**
     * @brief Reset detector
     */
    constexpr void reset() {
        CLK = false;
        Q = false;
    }
};

/**
 * @brief F_TRIG (Falling Edge Trigger)
 *
 * Detects falling edges on the CLK input.
 */
struct F_TRIG {
    bool CLK{false}; //!< Clock input
    bool Q{false};   //!< Output (true on falling edge)

    /**
     * @brief Execute edge detector
     * @param CLK_input Clock input
     * @return Edge detected
     */
    constexpr bool operator()(bool CLK_input) {
        Q = !CLK_input && CLK;
        CLK = CLK_input;
        return Q;
    }

    /**
     * @brief Reset detector
     */
    constexpr void reset() {
        CLK = false;
        Q = false;
    }
};

/**
 * @brief TON Timer (Timer On Delay)
 *
 * On-delay timer that activates output after specified time.
 */
template<typename T = float>
class TON {
public:
    T    PT{0};    //!< Preset time
    bool Q{false}; //!< Timer output
    T    ET{0};    //!< Elapsed time

    /**
     * @brief Execute timer
     * @param IN Timer input
     * @param dt Time step
     * @return Timer output
     */
    constexpr bool operator()(bool IN, T dt) {
        if (IN) {
            ET += dt;
            if (ET >= PT) {
                Q = true;
            }
        } else {
            ET = 0;
            Q = false;
        }
        return Q;
    }

    /**
     * @brief Reset timer
     */
    constexpr void reset() {
        Q = false;
        ET = 0;
    }
};

/**
 * @brief TOF Timer (Timer Off Delay)
 *
 * Off-delay timer that deactivates output after specified time.
 */
template<typename T = float>
class TOF {
public:
    T    PT{0};    //!< Preset time
    bool Q{false}; //!< Timer output
    T    ET{0};    //!< Elapsed time

    /**
     * @brief Execute timer
     * @param IN Timer input
     * @param dt Time step
     * @return Timer output
     */
    constexpr bool operator()(bool IN, T dt) {
        if (IN) {
            ET = 0;
            Q = true;
        } else {
            ET += dt;
            if (ET >= PT) {
                Q = false;
            }
        }
        return Q;
    }

    /**
     * @brief Reset timer
     */
    constexpr void reset() {
        Q = false;
        ET = 0;
    }
};

/**
 * @brief TP Timer (Timer Pulse)
 *
 * Pulse timer that generates a pulse of specified duration.
 */
template<typename T = float>
class TP {
public:
    T    PT{0};    //!< Pulse time
    bool Q{false}; //!< Timer output
    T    ET{0};    //!< Elapsed time

    /**
     * @brief Execute timer
     * @param IN Timer input (rising edge triggers pulse)
     * @param dt Time step
     * @return Timer output
     */
    constexpr bool operator()(bool IN, T dt) {
        if (IN && !prev_IN) { // Rising edge
            ET = 0;
            Q = true;
        }

        if (Q) {
            ET += dt;
            if (ET >= PT) {
                Q = false;
                // ET stays at PT when pulse ends
            }
        }

        prev_IN = IN;
        return Q;
    }

    /**
     * @brief Reset timer
     */
    constexpr void reset() {
        Q = false;
        ET = 0;
        prev_IN = false;
    }

private:
    bool prev_IN{false}; //!< Previous input state
};

/**
 * @brief CTU Counter (Count Up)
 *
 * Up-counter with reset and count limit.
 */
template<typename T = uint32_t>
class CTU {
public:
    T    PV{0};     //!< Preset value (count limit)
    T    CV{0};     //!< Current value
    bool Q{false};  //!< Counter output (CV >= PV)
    bool CU{false}; //!< Count up input (previous state)

    /**
     * @brief Execute counter
     * @param CU Count up input
     * @param R Reset input
     * @return Counter output
     */
    constexpr bool operator()(bool CU_input, bool R) {
        if (R) {
            CV = 0;
            Q = false;
        } else if (CU_input && !CU) { // Rising edge on CU
            if (CV < std::numeric_limits<T>::max()) {
                ++CV;
            }
            Q = (CV >= PV);
        }
        CU = CU_input;
        return Q;
    }

    /**
     * @brief Reset counter
     */
    constexpr void reset() {
        CV = 0;
        Q = false;
        CU = false;
    }
};

/**
 * @brief CTD Counter (Count Down)
 *
 * Down-counter with load and count limit.
 */
template<typename T = uint32_t>
class CTD {
public:
    T    PV{0};     //!< Preset value (load value)
    T    CV{0};     //!< Current value
    bool Q{false};  //!< Counter output (CV <= 0)
    bool CD{false}; //!< Count down input (previous state)

    /**
     * @brief Execute counter
     * @param CD Count down input
     * @param LD Load input
     * @return Counter output
     */
    constexpr bool operator()(bool CD_input, bool LD) {
        if (LD) {
            CV = PV;
            Q = false;
        } else if (CD_input && !CD) { // Rising edge on CD
            if (CV > 0) {
                --CV;
            }
            Q = (CV == 0);
        }
        CD = CD_input;
        return Q;
    }

    /**
     * @brief Reset counter
     */
    constexpr void reset() {
        CV = 0;
        Q = false;
        CD = false;
    }
};

/**
 * @brief CTUD Counter (Count Up Down)
 *
 * Up-down counter with reset and load.
 */
template<typename T = uint32_t>
class CTUD {
public:
    T    PV{0};     //!< Preset value (load value)
    T    CV{0};     //!< Current value
    bool QU{false}; //!< Up counter output (CV >= PV)
    bool QD{true};  //!< Down counter output (CV <= 0) - initially true since CV=0
    bool CU{false}; //!< Count up input (previous state)
    bool CD{false}; //!< Count down input (previous state)

    /**
     * @brief Execute counter
     * @param CU Count up input
     * @param CD Count down input
     * @param R Reset input
     * @param LD Load input
     */
    constexpr void operator()(bool CU_input, bool CD_input, bool R, bool LD) {
        if (R) {
            CV = 0;
            QU = false;
            QD = true; // CV == 0
        } else if (LD) {
            CV = PV;
            QU = (CV >= PV);
            QD = (CV == 0);
            CU = CD = false; // Reset edge states on load
        } else {
            if (CU_input && !CU) { // Rising edge on CU
                if (CV < std::numeric_limits<T>::max()) {
                    ++CV;
                }
            }
            if (CD_input && !CD) { // Rising edge on CD
                if (CV > 0) {
                    --CV;
                }
            }
            QU = (CV >= PV);
            QD = (CV == 0);
        }
        CU = CU_input;
        CD = CD_input;
    }

    /**
     * @brief Reset counter
     */
    constexpr void reset() {
        CV = 0;
        QU = false;
        QD = true; // CV == 0
        CU = CD = false;
    }
};

} // namespace wetmelon::control::plc