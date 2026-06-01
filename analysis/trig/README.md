# `wet::` trig — analysis & benchmarks

## Scope

`wet::` ([`inc/wet/math/wet_trig.hpp`](../../inc/wet/math/wet_trig.hpp)) is a header-only
single-precision trig library for Cortex-M7. This directory benchmarks it
against three alternatives:

| name | implementation | input domain | source |
| ---- | -------------- | ------------ | ------ |
| `wet::` | polynomial minimax, `nearbyint` range reduction | any | this repo |
| `ti::` | TI Arm Trig Library, polynomial | `[0, 2π]` | `ti_arm_trig.{hpp,cpp}` |
| `odrv::` | ODrive / CMSIS 512-entry LUT + linear interp | any | `math_utils.hpp` |
| `std::` | newlib `<cmath>` | any | toolchain |

Harness: [`math_utils.cpp`](math_utils.cpp) emits each backend as a standalone
symbol; [`Makefile`](Makefile) cross-compiles to ARM asm (gcc + clang);
[`minimax_trig.py`](minimax_trig.py) fits coefficients and measures float32 error.

## Accuracy — max absolute error (float32 vs float64 reference)

| fn | `wet::` (any) | `ti::` ([0,2π]) | `odrv::` (any) | `std::` (any) |
| -- | ------------- | --------------- | -------------- | ------------- |
| sin    | 8.6e-7  | 7.2e-7 | 1.9e-5 | 3.0e-8 |
| cos    | 8.6e-7  | 2.9e-7 | 1.9e-5 | 3.0e-8 |
| sincos | 8.6e-7  | 1.9e-7 | 1.9e-5 | — |
| asin   | 1.3e-6  | 3.4e-7 | — | 5.9e-8 |
| acos   | 1.3e-6  | 4.3e-7 | — | 1.8e-7 |
| atan   | 5.4e-7  | 1.7e-7 | — | 1.7e-7 |
| atan2  | 7.3e-7  | 3.0e-7 | 2.0e-4 | 2.0e-7 |

- `wet::` sin/cos/sincos use Cody-Waite reduction (three-word π): error holds at
  the ~8.6e-7 polynomial floor to |x| ≈ 20000 rad (see the Cody-Waite section).
  `ti::` is measured on its native `[0, 2π]`; wrapped to full range with
  single-step reduction it grows ~1 ULP/rad. So full-range `wet::` now ≈
  `ti::`-native accuracy.
- `odrv::` sin/cos match CMSIS `arm_sin_f32` (512-LUT + linear interp).
  `odrv::atan2` is a degree-3 polynomial.
- Inverse-trig refit targets: asin +1 term → 2.1e-7, atan +2 terms → 1.8e-7;
  not applied.

## Speed — static instruction count, Cortex-M7 (gcc / clang)

`wet::` sin/cos/sincos include Cody-Waite reduction; inverse trig via Estrin;
literal pool excluded.

```
-O3 -ffast-math -std=c++20 -mcpu=cortex-m7 -mfpu=fpv5-d16 -mfloat-abi=hard
```

| fn | `wet::` | `odrv::` | `ti::` |
| -- | ------- | -------- | ------ |
| sin    | 29 / 25 | 27 / 26 | 23 / 24 |
| cos    | 32 / 28 | 29 / 27 | 39 / 32 |
| sincos | 41 / 38 | 59 / 44 | 48 / 46 |
| asin   | 31 / 28 | — | 35 / 29 |
| acos   | 32 / 27 | — | 32 / 27 |
| atan   | 39 / 32 | — | 43 / 34 |
| atan2  | 40 / 38 | 32 / 33 | 63 / 52 |

- Cody-Waite costs ~+5 over single-step (sin was 24/20 without CW): 3 `vfms` for
  the π-words, one `r·inv_pi`, 3 literal-pool loads. It is always-on, including
  wrapped angles.
- Counts are static (both branch arms); the executed hot path is shorter.
  Cycles require a DWT count on M7.
- `std::` is not compiled here. SPRAD27A Table 3-1 (Cortex-R5F): ~150 cycles
  sin/cos, 87–222 for the rest.

## `-ffast-math`

Analysis builds with `-ffast-math`. The library is header-only;
`Tuprules.lua` (tests only) stays on `-O3`.

- Forward-trig instruction count: clang −3 (sin 22→19), gcc unchanged. FMA
  contraction already occurs at `-O3`.
- Does not convert Horner to Estrin — `c + x·p` has no associativity to
  reassociate. It does reassociate flat-sum form (`c0 + c1·x + …`, e.g. `ti::`)
  into a parallel tree.
- No effect on division: hardware `VDIV` retained (0 `vrecpe`).
- `-funsafe-math-optimizations` reassociation invalidates the Python-measured
  (fixed-order) error for the binary; verify on target if ULP-bound.

## Estrin vs Horner

`wet::detail::estrin_eval` (generic, constexpr) replaces `horner_eval` in the
inverse-trig functions. Pairwise fold (`b[i] = b[2i] + power·b[2i+1]`, power
squared per level): critical path ~log₂(N) vs N.

| poly | critical-path depth | float32 error vs f64 |
| ---- | ------------------- | -------------------- |
| atan (deg-7, N=8) | 7 → ~4 | 9.1e-8 → 1.27e-7 |
| asin (deg-5, N=6) | 5 → ~3 | 7.7e-8 → 1.76e-7 |

- Evaluation rounding cost ~1–2 ULP, below the fit error (atan ~5e-7,
  asin ~1.1e-6).
- gcc instruction count also dropped (atan 52→39, asin 40→31, acos 41→32) as
  Horner-branch duplication collapsed.
- Python mirror: `estrin_f32` / `horner_f32` in `minimax_trig.py`.
- `sin_poly` is hand-written semi-Estrin; `horner_eval` retained.

## Cody-Waite — large-argument reduction

`wet::detail::wrap` uses a three-word-π Cody-Waite reduction (the shipped
`wet C-W` column below); single-step (`ti::`, and `wet:: as-is` = the former
single-step) loses ~1 ULP/rad. Reference is `sin(float64(float32 x))` —
isolates reduction + poly from the float32 input quantization every method
shares ([`plot_codywaite.py`](plot_codywaite.py), `build/codywaite_error.png`).

| \|x\| (rad) | std | ti (wrap) | wet as-is | wet C-W | input floor |
| --- | --- | --- | --- | --- | --- |
| 10   | 5.2e-8 | 1.1e-6 | 1.2e-6 | 5.9e-7 | 4.4e-7 |
| 100  | 5.7e-8 | 1.0e-5 | 1.0e-5 | 8.1e-7 | 3.8e-6 |
| 1000 | 6.2e-8 | 8.9e-5 | 8.9e-5 | 8.7e-7 | 3.0e-5 |
| 5000 | 6.4e-8 | 3.9e-4 | 3.9e-4 | 8.8e-7 | 2.4e-4 |

- `wet:: C-W` is flat at ~8e-7; `wet:: as-is` and `ti::` grow ~linearly (~100×
  gap at |x|=1000, on algorithm error). std is flat ~0.5 ULP.
- The float32 input-quantization floor (`|cos x|·|x − float32(x)|`, irreducible)
  crosses the C-W floor at |x| ≈ 50–100 rad. Net benefit then depends on the
  angle's meaning: canonical float32 state → full ~100×; float32 sample of a
  continuous angle → input floor caps it (~3–4× by |x|=1000).
- Below ~7 rad (wrapped angles) all methods sit near their floors; C-W only
  matters unwrapped — but it is always-on (~+5 instructions vs single-step).
- 3-word π split: `n·PI_HI` exact for n < 2¹³, i.e. |x| ≲ 25700 rad (measured
  flat to 20000; rises to ~1e-6 by 50000). The three subtractions must not be
  reassociated — clang folds them back to single-step under `-ffast-math`
  (verified numerically: error regrows ~1 ULP/rad). `wrap` pins the order
  with `#pragma clang fp reassociate(off)` (survives inlining into
  sin/cos/sincos). gcc 14.2 does not reassociate the `vfms` chain under
  `-ffast-math`, so no gcc pragma is needed; the three-step reduction
  survives inlining without intervention.

## Status

Done:
- Full-range sin/cos/sincos/asin/acos/atan/atan2, branchless hot paths.
- Cody-Waite reduction in `wrap` → forward trig flat at ~8.6e-7 to |x| ≈ 20000
  rad (~+5 instructions; reassociation pinned on clang via pragma, gcc 14.2
  preserves the chain without intervention).
- Standalone `cos` via `sincos` identity.
- `wet::sqrt` (public, bare `vsqrt`, no `sqrtf` fallback).
- `sincos` in `MathBackend`; Park transforms use it.
- Estrin for inverse trig.

Open:
- Inverse-trig refit: asin +1 term, atan +2 terms.
- Optional: radian-domain `sin_poly` refit to trim Cody-Waite cost (~+5 → ~+3).

## References

- **[SPRAD27A]** S. Okur, E. Cohen, *Optimized Trigonometric Functions on TI Arm
  Cores*, TI, July 2022 (rev. Aug 2022). `ti::` accuracy and `std::` / CMSIS
  cycles are Table 3-1 (AM243x, Cortex-R5F, TI Arm Clang 2.0.0, MCU+ SDK 8.2 —
  R5F, not M7). FPv5 latencies (Fig 4-2) and techniques (Sollya minimax,
  branchless conditionals, `[0,2π] → [-π/2, π/2]` reduction) in §2, §4.
- *Arm Cortex-M7 TRM* — FPU instruction timing.