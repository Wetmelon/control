
#include "filters.hpp"
#include "fmt/base.h"

using namespace wetmelon::control;

auto              coeffs1 = design::lowpass_1st(10.0, 0.001).as<float>();
LowPass<1, float> lpf({coeffs1.b0, coeffs1.b1}, {coeffs1.a1}); // 10 Hz cutoff, 1ms sample time

auto              coeffs2 = design::lowpass_2nd(10.0, 0.001, 0.707).as<float>();
LowPass<2, float> lpf2({coeffs2.b0, coeffs2.b1, coeffs2.b2}, {coeffs2.a1, coeffs2.a2}); // 10 Hz, 1ms sample time, Butterworth

LowPass<1, float> lpf1{10.0f, 0.001f}; // 10 Hz cutoff, 1ms sample time

int main() {
    lpf.reset();

    // Simulate step response
    for (int i = 0; i < 101; ++i) {
        float output = lpf(1.0f); // Step input
        if (i % 10 == 0) {
            fmt::print("Step {}: Output = {:.4f}\n", i, output);
        }
    }

    return 0;
}