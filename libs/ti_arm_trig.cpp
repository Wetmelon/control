/*
 *  Copyright (C) 2022 Texas Instruments Incorporated
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *    Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 *    Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the
 *    distribution.
 *
 *    Neither the name of Texas Instruments Incorporated nor the names of
 *    its contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "ti_arm_trig.hpp"

#include <cmath>
#include <cstdint>
#include <numbers>

namespace {

alignas(8) constexpr float pi_consts[5] = {
    1.5707963267f,
    3.1415926535f,
    4.7123889803f,
    6.2831853071f,
    1.2732395447f,
};

alignas(8) constexpr float sin_consts[4] = {
    0.999996615908002773079325846913220383f,
    -0.16664828381895056829366054140948866f,
    0.00830632522715989396465411782615901079f,
    -0.00018363653976946785297280224158683484f
};

alignas(8) constexpr float cos_consts[5] = {
    0.999999953466670136306412430924463351f,
    -0.49999905347076729097546897993796764f,
    0.0416635846931078386653947196040757567f,
    -0.00138537043082318983893723662479142648f,
    0.0000231539316590538762175742441588523467f
};

alignas(8) constexpr float sincos_consts[6] = {
    -0.166666507720947265625f,
    0.008331954479217529296875f,
    -0.00019490718841552734375f,
    -0.499998867511749267578125f,
    4.165589809417724609375e-2f,
    -1.35934352874755859375e-3f
};

alignas(8) constexpr float asin_consts[7] = {
    1.5707961728f,
    -0.2145852647f,
    0.0887556286f,
    -0.0488025043f,
    0.0268999482f,
    -0.0111462294f,
    0.0022959648f,
};

alignas(8) constexpr float atan_consts[8] = {
    4.17232513427734375e-7f,
    0.99997341632843017578125f,
    1.46687030792236328125e-4f,
    -0.330976545810699462890625f,
    -2.6895701885223388671875e-2f,
    0.309777557849884033203125f,
    -0.21780431270599365234375f,
    5.117702484130859375e-2f
};

} // namespace

namespace ti_arm {

float sin(float angle_rad) {
    float a = angle_rad;

    if (a > pi_consts[0]) {
        angle_rad = pi_consts[1] - a;
    }
    if (a > pi_consts[2]) {
        angle_rad = a - pi_consts[3];
    }

    float x2 = angle_rad * angle_rad;
    float x4 = x2 * x2;

    float result = angle_rad * sin_consts[0];
    result += angle_rad * sin_consts[1] * x2;
    result += angle_rad * sin_consts[2] * x4;
    result += angle_rad * sin_consts[3] * x2 * x4;

    return result;
}

float cos(float angle_rad) {
    bool  negate = false;
    float a = angle_rad;

    if (a > pi_consts[0]) {
        angle_rad = a - pi_consts[1];
        negate = true;
    }

    if (a > pi_consts[2]) {
        angle_rad = angle_rad - pi_consts[1];
        negate = false;
    }

    float x2 = angle_rad * angle_rad;
    float x4 = x2 * x2;

    float result = cos_consts[0];
    result += cos_consts[1] * x2;
    result += cos_consts[2] * x4;
    result += cos_consts[3] * x2 * x4;
    result += cos_consts[4] * x4 * x4;

    return negate ? -result : result;
}

SinCosResult sincos(float angle_rad) {
    int32_t r = static_cast<int32_t>(angle_rad * pi_consts[4]);

    uint8_t swap_val = 0x66;
    uint8_t sign_s = 0x1e;
    uint8_t sign_c = 0x78;

    float mod_val = angle_rad - ((r + 1) >> 1) * pi_consts[0];
    swap_val = (swap_val >> r) & 0x1;
    sign_s = (sign_s >> r) & 0x1;
    sign_c = (sign_c >> r) & 0x1;

    float a2 = mod_val * mod_val;
    float a4 = a2 * a2;
    float a6 = a4 * a2;

    float sin_val = mod_val;
    sin_val += mod_val * sincos_consts[0] * a2;
    sin_val += mod_val * sincos_consts[1] * a4;
    sin_val += mod_val * sincos_consts[2] * a6;

    float cos_val = 1.0f;
    cos_val += sincos_consts[3] * a2;
    cos_val += sincos_consts[4] * a4;
    cos_val += sincos_consts[5] * a6;

    if (sign_s) {
        sin_val = -sin_val;
    }
    if (sign_c) {
        cos_val = -cos_val;
    }

    // swap_val encodes whether sin/cos are swapped for this quadrant
    if (swap_val) {
        return {cos_val, sin_val};
    }
    return {sin_val, cos_val};
}

float asin(float x) {
    bool negate = false;

    if (x < 0.0f) {
        x = -x;
        negate = true;
    }

    float sqrt_x = ti_arm::sqrt(1.0f - x);

    float x2 = x * x;
    float x3 = x2 * x;
    float x4 = x2 * x2;
    float x5 = x2 * x3;
    float x6 = x3 * x3;

    float result = asin_consts[0];
    result += asin_consts[1] * x;
    result += asin_consts[2] * x2;
    result += asin_consts[3] * x3;
    result += asin_consts[4] * x4;
    result += asin_consts[5] * x5;
    result += asin_consts[6] * x6;

    result *= sqrt_x;

    if (negate) {
        result -= pi_consts[0];
    } else {
        result = pi_consts[0] - result;
    }

    return result;
}

float acos(float x) {
    bool negate = false;

    if (x < 0.0f) {
        x = -x;
        negate = true;
    }

    float sqrt_x = ti_arm::sqrt(1.0f - x);

    float x2 = x * x;
    float x3 = x2 * x;
    float x4 = x2 * x2;
    float x5 = x2 * x3;
    float x6 = x3 * x3;

    float result = asin_consts[0];
    result += asin_consts[1] * x;
    result += asin_consts[2] * x2;
    result += asin_consts[3] * x3;
    result += asin_consts[4] * x4;
    result += asin_consts[5] * x5;
    result += asin_consts[6] * x6;

    result *= sqrt_x;

    if (negate) {
        result = result - pi_consts[0];
    } else {
        result = pi_consts[0] - result;
    }

    result = pi_consts[0] - result;

    return result;
}

float atan(float x) {
    bool negate = false;
    bool complement = false;

    if (x < 0.0f) {
        x = -x;
        negate = true;
    }

    if (ti_arm::abs(x) > 1.0f) {
        x = 1.0f / x;
        complement = true;
    }

    float x2 = x * x;
    float x3 = x2 * x;
    float x4 = x2 * x2;
    float x5 = x3 * x2;
    float x6 = x3 * x3;
    float x7 = x3 * x4;

    float result = atan_consts[0];
    result += atan_consts[1] * x;
    result += atan_consts[2] * x2;
    result += atan_consts[3] * x3;
    result += atan_consts[4] * x4;
    result += atan_consts[5] * x5;
    result += atan_consts[6] * x6;
    result += atan_consts[7] * x7;

    if (complement) {
        result = pi_consts[0] - result;
    }
    if (negate) {
        result = -result;
    }

    return result;
}

float atan2(float y, float x) {
    float k1 = pi_consts[0];
    float k2 = 0.0f;
    bool  negate = false;
    bool  complement = false;

    float ratio;
    if (ti_arm::abs(x) > ti_arm::abs(y)) {
        ratio = y / x;
    } else {
        ratio = x / y;
        complement = true;
    }

    if (ratio < 0.0f) {
        ratio = -ratio;
        negate = true;
    }

    if (x < 0.0f) {
        if (y > 0.0f) {
            k2 = pi_consts[1];
        } else {
            k2 = -pi_consts[1];
        }
    }
    if (y < 0.0f) {
        k1 = -pi_consts[0];
    }

    float x2 = ratio * ratio;
    float x3 = x2 * ratio;
    float x4 = x2 * x2;
    float x5 = x3 * x2;
    float x6 = x3 * x3;
    float x7 = x3 * x4;

    float result = atan_consts[0];
    result += atan_consts[1] * ratio;
    result += atan_consts[2] * x2;
    result += atan_consts[3] * x3;
    result += atan_consts[4] * x4;
    result += atan_consts[5] * x5;
    result += atan_consts[6] * x6;
    result += atan_consts[7] * x7;

    if (negate) {
        result = -result;
    }
    if (complement) {
        result = k1 - result;
    } else {
        result = k2 + result;
    }

    return result;
}

} // namespace ti_arm
