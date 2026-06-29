
#include "wet/control.hpp"
#include "wet/matrix/colvec.hpp"
#include "wet/motor/servo.hpp"

constinit static wet::motor::PmacServo motor;

auto test(float pos, float vel, float tau, const wet::ColVec<3, float>& Iabc, float Vdc, float delta_enc) {
    return motor.update(pos, vel, tau, Iabc, Vdc, delta_enc);
}
