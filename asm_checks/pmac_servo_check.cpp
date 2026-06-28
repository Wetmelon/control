
#include "wet/control.hpp"
#include "wet/motor/servo.hpp"

constinit static wet::motor::PmacServo motor;

auto test(const float ref, const wet::motor::ServoFeedback<float> y) {
    motor.set_target(ref);
    return motor.update(y);
}
