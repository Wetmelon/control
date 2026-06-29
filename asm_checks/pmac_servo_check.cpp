
#include "wet/control.hpp"
#include "wet/matrix/colvec.hpp"
#include "wet/motor/servo.hpp"

constinit static wet::motor::PmacServo motor;

auto test(const float ref, const wet::ColVec<3, float>& Iabc, float Vdc, float delta_enc) {
    motor.set_target(ref);
    return motor.update(Iabc, Vdc, delta_enc);
}
