
#include <cstdint>

#include "wet/control.hpp"
#include "wet/matrix/colvec.hpp"
#include "wet/motor/servo.hpp"

constinit static wet::motor::PmacServo motor;

uint64_t now();

// Runs at 1kHz
auto periodic_1kHz(float pos) {
    motor.position_control_step(pos); // Generate velocity controller command
}

// Preemptive task can interrupt periodic_1kHz
auto periodic_8kHz(float vel_tgt, float torque_tgt) {
    constexpr float Ts = 1.0f / 8000.0f;

    // Current and torque control limit refresh based on bus voltage
    motor.recalculate_limits();
    motor.velocity_control_step(vel_tgt, Ts);  // Generate torque controller command
    motor.torque_control_step(torque_tgt, Ts); // Generate DQ current reference
}

struct EncoderData {
    float    angle;
    uint64_t timestamp;
    bool     is_fresh;
};

// Preemptive 24kHz ISR
auto control_isr(wet::ColVec<3, float>& Iabc, float Vdc, EncoderData& enc) {
    constexpr float Ts = 1.0f / 24e3f;

    static uint64_t last_enc_timestamp = 0;
    static bool     first_loop = true;

    // Update encoder observer with a new measurement to keep
    // the observer coherent across preemption levels
    if (enc.is_fresh) {
        if (!first_loop) {
            const uint32_t ticks = (uint32_t)(enc.timestamp - last_enc_timestamp);
            const float    dt = (float)ticks / 400e6f;
            motor.encoder_update_abs(enc.angle, dt);
        }
        last_enc_timestamp = enc.timestamp;
        first_loop = false;
        enc.is_fresh = false;
    }

    return motor.current_control_step(motor.Idq_ref_, Iabc, Vdc, Ts);
}
