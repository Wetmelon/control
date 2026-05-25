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

#ifndef TI_ARM_TRIG_HPP_
#define TI_ARM_TRIG_HPP_

#if defined(__GNUC__) || defined(__clang__)
#define TI_ARM_TRIG_TEXT_SECTION __attribute__((section(".trigText")))
#define TI_ARM_TRIG_DATA_SECTION __attribute__((aligned(8), section(".trigData")))
#define TI_ARM_TRIG_ALWAYS_INLINE __attribute__((always_inline)) inline
#else
#define TI_ARM_TRIG_TEXT_SECTION
#define TI_ARM_TRIG_DATA_SECTION
#define TI_ARM_TRIG_ALWAYS_INLINE inline
#endif

float ti_arm_sin(float angleRad);
float ti_arm_cos(float angleRad);
void ti_arm_sincos(float angleRad, float* retValues);
float ti_arm_asin(float x);
float ti_arm_acos(float x);
float ti_arm_atan(float x);
float ti_arm_atan2(float y, float x);

TI_ARM_TRIG_ALWAYS_INLINE float ti_arm_sqrt(float x)
{
    float r = 0.0f;
    __asm("VSQRT.F32 %0, %1" : "=t"(r) : "t"(x));
    return r;
}

TI_ARM_TRIG_ALWAYS_INLINE float ti_arm_abs(float x)
{
    float r = 0.0f;
    __asm("VABS.F32 %0, %1" : "=t"(r) : "t"(x));
    return r;
}

#endif /* TI_ARM_TRIG_HPP_ */
