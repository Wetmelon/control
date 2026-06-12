# wet Library Roadmap

Date: 2026-05-25
Owner: controls core
Status: draft

## Table of Contents

- [Purpose](#purpose)
- [Design Constraints](#design-constraints)
- [Architecture layers & build priority](#architecture-layers--build-priority)
- [Dependencies](#dependencies)
- [Standard Synthesis Workflow](#standard-synthesis-workflow)
- [Roadmap](#roadmap)
- [Testing and Documentation](#testing-and-documentation)
- [Decision Items](#decision-items)

## Purpose

`wet` is a **controls** library, built on a deliberately thin, layered base. Generic embedded plumbing comes from ETL (Layer 0). On top of that the library **first** delivers the foundational linear / first-order primitives that *every* controls algorithm leans on — linear algebra, constexpr math, LTI system types, first-order and biquad filters, signal-conditioning blocks — and only on that foundation the controllers, estimators, and synthesis. Work the stack **bottom-up**: see [Architecture layers & build priority](#architecture-layers--build-priority).

This roadmap tracks planned additions across controllers, estimators, and signal-processing/identification, held to a common synthesis workflow using the three-tier pattern:

1. Design function (`design::`, constexpr)
2. Result struct (`.as<U>()`, `bool success`)
3. Runtime controller / estimator / filter, or a synthesis artifact bundle

## Design Constraints

- Constexpr-first design algorithms.
- Runtime path without dynamic allocation.
- Explicit failure signaling for fallible numerical steps.
- Scalar support for `float`, `double`, `wet::complex<float>`, `wet::complex<double>` where applicable.
- Compile-time and runtime validation tests for each new API.
- **Embedded vs host placement is part of the design.** Allocation-free primitives belong in `wet/control.hpp`; anything that pulls `<vector>`, an FFT, or a solver/optimizer belongs behind `wet/toolbox.hpp`. `make embedded-check` enforces the embeddable contract.
- Tuning/adaptation APIs must not require runtime coupling to identification internals; model inputs are optional and must be disable-able.
- **Stay in the controls lane; defer generic embedded plumbing to ETL.** Containers, queues, ring/FIFO buffers, CRC/checksums, and similar general-purpose utilities are out of scope — the [Embedded Template Library](https://www.etlcpp.com) covers them. This library ships only controls/DSP-specific value (controllers, estimators, filters, interpolation tables, encoder/tach, motion).
- **No mandatory third-party dependency; backend-agnostic core.** The embeddable core supports two backends for the handful of `std`-replacement types it needs, selected by the backend profile (#21): the **C++ stdlib** (default — works standalone, no third-party lib) or **ETL** (for freestanding/embedded targets). We do not invent our own container/optional primitives — those come from stdlib or ETL. ETL is a recommended companion for plumbing, never required. See [Dependencies](#dependencies) for which parts need what.
- **Tier 1 / Tier 2 layering.** The library never branches on "kind of plant". Plants are universally described by `StateSpace<NX,NU,NY>` / `TransferFunction<Nn,Nd>`. Tier 1 functions take those universal types (or time-series + order) and are the implementations of truth; Tier 2 wrappers (e.g. `models::two_mass`, `analysis::identify_two_mass`, `CascadePPI`) are thin aliases that call Tier 1 with archetype-specific defaults and unpack results into named physical parameters. Tier 2 must never add capability — removing it leaves the library functionally complete.

## Architecture layers & build priority

The library is layered; build and prioritize **bottom-up**. Each layer depends only on the ones beneath it, and a higher layer is not worked before the foundation it needs is solid. The numbered roadmap items below are grouped here by layer — this is the priority/reading order; the numbers themselves are just stable IDs used by cross-references, not a ranking.

**Layer 0 — generic plumbing & the std-replacement backend.** Containers, queues, ring/FIFO buffers, CRC, checksums, debounce — not ours (see Design Constraints), supplied by [ETL](https://www.etlcpp.com) as an optional companion. The small set of `std`-replacement types the core needs (array/optional/tuple/clamp/numbers) comes from one of two interchangeable backends selected per #21 — the **C++ stdlib** (default) or **ETL** — so the core carries no mandatory third-party dependency. We don't ship our own.

**Layer 1 — foundation: the linear / first-order primitives every controls algorithm uses.** This comes *first*; most of it already exists.
- *In place:* the linear-algebra core (`Matrix`/`ColVec`/`RowVec`, decompositions, `mat::solve`), the constexpr math backend (`wet::sin/sqrt/exp/...`), LTI system types (`StateSpace`/`TransferFunction`), first-order + biquad filters (#2 ☑), and the everyday signal-conditioning blocks / embedded primitives — scaling & calibration, interpolation tables, EWMA·peak·envelope, software timers, encoder/tach (#19 ☑).
- *Foundation complete.* All Layer-1 gaps are done: the freestanding / ETL-backend profile (#21 ☑ — core compiles and the full suite passes under both `std` and `etl` backends), robust MIMO pole placement (#16 ☑, `design::place` via Kautsky–Nichols–Van Dooren + `mat::full_qr`), fast-math-robust filter coefficient designs (#17 ☑), and the robust nonsymmetric eigensolver (Hessenberg + Francis double-shift, real+complex spectra). New work can build on a solid foundation; Layer 2+ is next.

**Layer 2 — core controls & estimation building blocks.** Single-purpose laws/estimators that sit directly on Layer 1: Luenberger/reduced observer (#1 ☑), disturbance observer (#4), UKF (#12), and the identification / excitation / cascade / model-builder infrastructure (#3).

**Layer 3 — advanced controls & estimation.** Composite, adaptive, or optimization-based, built from Layers 1–2: repetitive control (#5), input shaping (#6), online PID tuning (#7), extremum-seeking (#8), harmonic detection & suppression (#9), LPV gain scheduling (#10), super-twisting SMC (#11), H∞ output feedback (#13), constrained MPC (#14), moving-horizon estimation (#15), motion planning (#20).

**Layer 4 — tooling (host-only).** Supports development but does not ship on target: the multi-rate simulation harness (#18) and the host-side identification / FRF parts of #3.

## Dependencies

What each part needs. The headline: the **embeddable core (`wet/control.hpp`) has no mandatory third-party dependency** — it needs only a *backend* (stdlib or ETL) and a *math backend*. Everything heavier is host-only behind `wet/toolbox.hpp`.

| Part | Requires | Optional | Ships on target |
|---|---|---|---|
| **Embeddable core** (`wet/control.hpp`) | one backend profile + the `wet::` math backend | — | ✅ |
| ↳ backend profile (array/optional/tuple/clamp/numbers) | one of: C++ stdlib · ETL | — | ✅ |
| ↳ `wet::` math backend (`wet::sin/sqrt/exp/...`) | freestanding-capable; default impl uses `<cmath>`, swappable via `wet/config.hpp` profile macros | platform intrinsics | ✅ |
| **Host superset** (`wet/toolbox.hpp`): analysis, simulation, MATLAB-style aliases | full hosted C++ stdlib (`<vector>`, `<string>`, …) | — | ❌ |
| ↳ plotting (`plotting/plot*.hpp`) | plotlypp | — | ❌ |
| **Examples** | hosted stdlib + fmt | plotlypp | ❌ |
| **Tests** | hosted stdlib + doctest | — | ❌ |

The two core configurations (selected per #21):

- **With the C++ stdlib** (default) — `std::` backs the std-replacement types. Hosted, and usable standalone (no third-party lib at all).
- **With ETL** — `etl::` backs them, for freestanding/embedded targets without a hosted stdlib; pair with ETL for plumbing too.

We don't invent our own primitives — anything not from the stdlib comes from ETL. Only the **ETL/freestanding** backend is new work (see #21); the stdlib configuration is what exists today.

## Standard Synthesis Workflow

1. Define plant model — `StateSpace` directly, or linearize a nonlinear model at an operating point.
2. Run design stage — `design::synthesize_*` / `design::*`; evaluate convergence and feasibility.
3. Produce artifacts — design result (gains/poles/internals + success), analysis models (closed-loop, error dynamics), runtime bundle (`step(...)`, `reset()`).
4. Convert for deployment — design in `double`, then `.as<float>()` when needed.
5. Verify — static checks (dimensions, constexpr viability, stability) and runtime checks (numerical references, closed-loop expectations).

## Roadmap

Items are listed in **build-priority order**, grouped by the layers from [Architecture layers & build priority](#architecture-layers--build-priority) (foundation first). The item **numbers are stable IDs** — referenced across this doc, `AGENTS.md`, and source/test comments — so they are deliberately *not* sequential here; order on the page is the priority, the number is just the ID. Status tags: ☑ done · ◐ partially built (some sub-parts done, others not) · ⊘ sketched (header stub exists) · ☐ planned.

**Layer 1 — foundation: the linear / first-order primitives every controls algorithm uses (build first).**

### 2. DSP biquad/notch family + utility blocks ☑

Complete the biquad family beyond low-pass and add everyday runtime blocks. Constexpr coefficient designers; allocation-free runtimes. Unblocks harmonic suppression (#9) and is broadly useful.

- Done (`filters/filters.hpp`): RBJ-cookbook biquad designers `design::notch`, `bandpass`, `highpass_2nd`, `peaking`, `lowshelf`, `highshelf` → `SecondOrderCoeffs` (Q-parameterized, designed directly in the digital domain); runtimes `Biquad` (Direct Form I) and `BiquadCascade<N>` (SOS cascade). All constexpr + float/double.
- Done (`filters/blocks.hpp`): utility runtime blocks `MovingAverage<N>`, `RunningRMS<N>`, `MedianFilter<N>`, `RateLimiter` (symmetric + asymmetric slew), `DirtyDerivative` (Tustin band-limited d/dt), `ClampedIntegrator` (forward-Euler + anti-windup clamp; named to avoid colliding with the ODE-solver `Integrator<NX,T>` in `simulation/integrator.hpp`), `Deadband` (continuous-slope), `Hysteresis` (Schmitt trigger). All constexpr, allocation-free, one-sample-per-call; in the `wet/control.hpp` umbrella.
- Done (`filters/differentiator.hpp`, `test_differentiator.cpp`): `RobustExactDifferentiator<T>` (alias `LevantDifferentiator<T>`) — Levant's first-order sliding-mode (super-twisting) differentiator. Model-free clean derivative of a noisy/quantized signal with far less phase lag than an LPF; also exposes a denoised `value()`. `update(f) → ḟ` from a second-derivative bound `L` and the std STA gains (λ0=1.5, λ1=1.1). The matched rate-estimator for the super-twisting controller (#11) and for servo **encoder velocity** (quantization-dominated low-speed differentiation). Constexpr, allocation-free; in the umbrella. Examples: `example_encoder_velocity` (beats raw/LPF'd finite-difference on RMS error *and* near direction reversals), and feeds clean rate in `example_swashplate_stsmc`. Ref: Levant, "Robust exact differentiation via sliding mode technique," Automatica 1998.
- References: Oppenheim & Schafer, "Discrete-Time Signal Processing," 3rd ed., 2009; R. Bristow-Johnson, "Cookbook formulae for audio EQ biquad filter coefficients."
- Acceptance: per-type frequency-response checks ✓; SOS cascade ✓; float/double parity ✓; utility blocks (`tests/test_blocks.cpp`: defining-behaviour + fill/reset edges + constexpr construction) ✓.

### 19. Embedded firmware primitives ☑

The bread-and-butter building blocks an embedded engineer reaches for *before* any synthesis: sensor linearization, scaling/calibration, the workhorse smoother, non-blocking timing, and encoder/speed I/O. The synthesis-heavy items above assume these already exist; today they mostly don't (`utility.hpp` has only unit conversions + `wrap`).

**Scope boundary — defer generic plumbing to ETL.** Containers, queues, byte FIFOs, CRC/checksums, and the like are *not* this library's job; the [Embedded Template Library](https://www.etlcpp.com) already does them well and battle-tested. Pair the two: use `etl::circular_buffer` / `etl::queue_spsc_*` / `etl::crc*` / `etl::debounce` for plumbing, and this library for the controls/DSP-specific helpers below. The core stays zero-dependency — we don't `#include <etl/...>` anywhere; ETL is a *recommended companion*, not a dependency.

**Done.** The controls/DSP-specific primitives shipped, embeddable, with one test TU each (`make embedded-check` green):
- `utility/scaling.hpp` — `lerp`, `inverse_lerp`, `rescale`, `AffineCal` (+`.as<U>()`), `two_point_cal`, `poly_horner` (`clamp` uses `wet::clamp`).
- `utility/lookup.hpp` — `Lut1D<N>` (linear/`nearest`, clamp-or-extrapolate `oob` policy, binary `lut_segment`), `Lut2D<R,C>` (bilinear, edge-clamped).
- `filters/blocks.hpp` (extended) — `ExponentialFilter` + `alpha_from_cutoff`/`alpha_from_time_constant`, `PeakHold`/`MinHold` (optional leak), `EnvelopeDetector` (attack/release).
- `utility/timing.hpp` — `Stopwatch`, `Timeout` (clamped accumulator), `Periodic` (drift-free, catch-up).
- `utility/encoder.hpp` — `QuadratureDecoder` (X1/X2/X4, index reset, illegal-transition rejection), `wrapped_delta` (rollover-safe), `Tachometer` (frequency + period methods, RPM/rad·s⁻¹ conversions).

Dropped from the original scope as ETL duplication: CRC/checksums and the SPSC/overwrite ring buffers (use `etl::crc*`, `etl::queue_spsc_*`, `etl::circular_buffer`).

Decisions resolved: LUT defaults to clamp with an opt-in `Linear` extrapolation and binary-search lookup (no cached hint for now); software timers are `dt`-fed (clock-source agnostic); encoder position is a fixed-width signed counter with `wrapped_delta` for rates.

These are **leaf utilities**, so most deliberately deviate from the three-tier synthesis pattern: they are pure constexpr, allocation-free runtime blocks with no plant model and (usually) no `design::` stage. All are embeddable (`wet/control.hpp`) unless noted. Every accumulator/counter is bounded by construction — no unbounded growth, per the library's overflow-safety rule (see the TON/TOF `ET` clamping and CTU/CTD saturation already in `iec61131.hpp`).

**Tier A — near-universal:**

- **Lookup tables & interpolation** (`utility/lookup.hpp`, new). `Lut1D<N,T>` — monotonic breakpoint table with linear or nearest interpolation and a configurable out-of-range policy (clamp vs linear extrapolation); O(log N) binary search with an optional cached-index hint for the common monotone-query case. `Lut2D<R,C,T>` — bilinear interpolation over a regular grid. Targets: thermistor/sensor linearization (pairs with `thermistor.hpp`), gain-scheduling maps, fan/efficiency/torque-speed curves, ADC fixups.
- **Scaling & calibration** (`utility/scaling.hpp`, new). `lerp(a, b, t)` / `inverse_lerp(a, b, x)` plus a `rescale(x, in_lo, in_hi, out_lo, out_hi)` composed from them (affine map between two ranges), `AffineCal<T>{gain, offset}` with `apply`/`invert`, `two_point_cal(raw0,eng0, raw1,eng1)` → `AffineCal`, and `poly_horner(coeffs, x)` for polynomial cal curves. Counts↔engineering-units conversion is the headline use; composes with `Lut1D` for nonlinear sensors.
- **Exponential filter / EWMA** (add to `filters/blocks.hpp`). `ExponentialFilter<T>` — one-pole IIR by direct smoothing factor `alpha` (`y += alpha*(x − y)`), the most-used embedded smoother. Ships with `alpha_from_cutoff(fc, Ts)` and `alpha_from_time_constant(tau, Ts)` helpers so it interoperates with the existing `fc/Ts`-parameterized first-order `LowPass` in `filters.hpp`. (Distinct from `MovingAverage` (FIR) and `LowPass` (designed biquad): zero buffer, one multiply-add.)
- **Non-blocking software timers** (`utility/timing.hpp`, new). `Stopwatch<T>` (elapsed-time accumulator with `reset`/`elapsed`), `Timeout<T>` (one-shot `expired()` predicate), and `Periodic<T>` (the "do X every N" idiom: `operator()(dt) → bool` firing once per period, with catch-up policy). Tick-count and `T`-seconds (`dt`-fed) variants. Complements the IEC `TON/TOF/TP` PLC-scan timers with the plain scheduling idiom firmware actually uses; accumulators wrap/saturate safely.
- **Peak / hold & envelope** (add to `filters/blocks.hpp`). `PeakHold<T>` / `MinHold<T>` (running extremum with optional decay/leak toward the signal) and `EnvelopeDetector<T>` (asymmetric attack/release follower). Targets: UI bars, fault-threshold capture, crude AGC, inrush capture.

**Tier B — controls-specific I/O:**

- **Quadrature encoder & tach** (`utility/encoder.hpp`, new). `QuadratureDecoder` — A/B (x1/x2/x4) decode with optional Z index, exposing a signed position that handles counter wrap deliberately (the overflow-safety rule is load-bearing here: position deltas are computed in the unsigned domain then sign-extended). `Tachometer<T>` — speed from either pulse-count-per-interval (high speed) or period-between-edges (low speed), with an auto-crossover, returning RPM / rad·s⁻¹ / Hz.

> Generic plumbing (CRC/checksums, SPSC queues, overwrite-oldest circular buffers) was originally scoped here but **cut** — see the ETL scope boundary above.

**References:** R. Lyons, "Understanding Digital Signal Processing," 3rd ed., 2010 (one-pole/EWMA, envelope detection).

**Acceptance:**

- `Lut1D`/`Lut2D` match analytic interpolation at and between breakpoints; out-of-range policy (clamp vs extrapolate) behaves as configured; cached-hint search matches binary search.
- Scaling/cal round-trips (`apply` then `invert`); `two_point_cal` reproduces both anchor points exactly; `poly_horner` matches naïve Horner.
- `ExponentialFilter` step response matches `(1−alpha)^n`; `alpha_from_cutoff` gives the same −3 dB point as the first-order `LowPass`.
- Software timers fire at the right boundaries; accumulators never grow unbounded.
- `PeakHold`/`EnvelopeDetector` track and decay per spec.
- `QuadratureDecoder` counts correct direction/magnitude across a full counter wrap with no spurious jump; `Tachometer` recovers known speeds across the count/period crossover.
- `make embedded-check` stays green (all of #19 is embeddable).

**Decision items (this section):** LUT out-of-range default (clamp vs extrapolate) and index-search strategy (binary vs cached hint); encoder count-type width and wrap convention; software-timer time source abstraction (integer ticks vs `T`-seconds with `dt`).

### 16. Robust MIMO pole placement (true `place`) ☑

Foundational numerical routine: place the eigenvalues of (A − BK) for **multi-input** systems, spending the extra gain freedom to maximize numerical robustness (minimize eigenvector conditioning). Equivalent to MATLAB's `place`.

- **Done (`controllers/pole_placement.hpp`, 2026-06).** `design::place(A, B, poles)` → `wet::optional<K>` via Kautsky–Nichols–Van Dooren. Real-pole path runs KNV **Method 0** (each eigenvector iteratively re-chosen within its admissible subspace to be maximally orthogonal to the others, minimizing κ(X)); complex-conjugate poles are assigned in **real arithmetic** via real (Re v, Im v) eigenvector pairs and 2×2 real Λ-blocks `[[σ,ω],[−ω,σ]]`, so K stays real. Built on a new `mat::full_qr` (full Householder QR — the orthogonal complement of B and the per-eigenvalue admissible null spaces). Single-input reduces to and matches `matlab::acker`; `NU == NX` uses the direct `B⁻¹(A − Λ)` form. Rejects (→ `nullopt`) rank-deficient B, multiplicity > NU (non-defective spectra only), and unpaired complex poles. Fully constexpr. `matlab::place` is the thin alias.
- Naming: `matlab::acker` (single-input Ackermann) already existed; `place` is the new robust MIMO routine — no rename was needed.
- Reference: J. Kautsky, N. K. Nichols, P. Van Dooren, "Robust pole assignment in linear state feedback," Int. J. Control, 1985. https://doi.org/10.1080/00207178508933420
- Acceptance ☑: places assignable MIMO real & complex spectra to ~1e-12 (`test_pole_placement.cpp`); rejects multiplicity > NU, rank-deficient B, and dangling complex poles; single-input matches `acker`; constexpr-evaluable.
- Remaining refinement (optional): the conditioning sweep is applied on the all-real path; complex pairs assign from the orthonormal admissible basis without the extra sweep (placement still exact, conditioning good but not sweep-optimized).

### 17. Fast-math-robust filter coefficient designs ☑

Restructure bilinear filter designers so that the unit-DC-gain identity (and other intended algebraic relations) holds *constructively* rather than emerging from exact cancellation, so the property survives `-fassociative-math` reassociation in downstream user builds.

- **`design::lowpass_2nd` ☑ fixed.** The original bilinear formula didn't merely drift under fast-math — its coefficients placed both poles essentially *on* the unit circle (|p| ≈ 1.0), making it a near-passthrough rather than a low-pass at all (measured |H| ≈ 1.0 at DC, fc, and 10·fc). Rewritten so the numerator taps are derived directly from the unit-DC-gain identity: since the bilinear numerator is ω₀²(1+z⁻¹)² the taps are in fixed ratio 1:2:1, so they are pinned as `dc_sum·{¼, ½, ¼}` with `dc_sum = 1 + a1 + a2`. Unity DC gain now holds by construction (and the ¼/½ factors are exact powers of two), identical under strict-IEEE and `-ffast-math`. Verified: |H| = 1.000 at DC, 0.707 at fc, 0.009 at 10·fc; DC-gain regression test tightened back to 1e-3, settling test back to 0.01.
- **Audit pass ☑ done.** Swept the six RBJ designers (`notch`, `bandpass`, `highpass_2nd`, `peaking`, `lowshelf`, `highshelf`) with tight band-edge gain-identity regression tests evaluated from the *stored* coefficients (`test_biquad.cpp`, "band-edge gain identities hold tightly under -ffast-math"): notch unity passband (DC + Nyquist) to 1e-9, bandpass DC/Nyquist nulls to 1e-12, highpass DC null to 1e-9, lowpass_2nd unity DC to 1e-12, peaking/shelf unity on the flat side to 1e-9. All hold under the runner's `-ffast-math` — as predicted, the RBJ single-expression formulas are not fragile. Reviewed the StateSpace→coeffs paths (`to_coeffs`): they faithfully transcribe the discretized state-space (`b`/`a` read straight from `A`/`B`/`C`/`D`), so there is no cancellation-enforced identity to restructure — DC gain is simply whatever the plant has.
- Reference: Higham, "Accuracy and Stability of Numerical Algorithms" (2nd ed., 2002), §3 on conditioned summation; Oppenheim & Schafer, "Discrete-Time Signal Processing" (3rd ed., 2009), bilinear-transform chapter.
- Acceptance ☑: DC-gain regression tests hold under the runner's `-ffast-math` — `lowpass_2nd` (`test_filters.cpp`: DC-gain at 1e-3, settling at 0.01, plus a cross-zeta unity-gain case), and the six RBJ designers (`test_biquad.cpp`: band-edge gain identities from stored coefficients, 1e-9–1e-12). `to_coeffs` reviewed — no enforced identity to guard.

### 21. Backend-agnostic core: stdlib or ETL (freestanding-capable) ☑

Drop the assumption that a hosted C++ standard library exists, so the embeddable core (`wet/control.hpp`) can compile for **freestanding** targets (no `libstdc++`/`libc++`). Achieved with a *backend profile*, not a hard swap: a thin alias layer maps `wet::array`/`wet::optional`/… to **one of two** backends, selected through the unified `wet/config.hpp` profile macros (shared with the math backend):

- **`stdlib`** (default, hosted) — aliases → `std::`. What exists today; unchanged for hosted users, and usable standalone (no third-party lib).
- **`etl`** — aliases → `etl::` ([Embedded Template Library](https://www.etlcpp.com) supplies the types). Freestanding; pairs naturally with ETL for plumbing.

We deliberately do **not** add a third "invent our own primitives" backend — anything not from the stdlib comes from ETL (see Design Constraints). Both backends preserve the constexpr-first invariant. Toolbox, tests, and examples stay hosted on `std`; only the embeddable umbrella must be freestanding-clean.

**Done (2026-06): the core compiles and the full test suite passes under both backends.** Delivered:
- **Unified config** — a single `wet/config.hpp` discovery point `__has_include`s a macro-only `wet_profile.hpp` (else warns + host defaults), read by *both* facilities. Per-facility macros: containers `WET_BACKEND_ETL` (`backend.hpp`), scalar math `WET_MATH_BACKEND_WET` / `WET_MATH_BACKEND_FREESTANDING` / `WET_MATH_BACKEND_HEADER "h"` (default std). Macro-only profiles let `config.hpp` be read both early (containers) and late (math) without ordering hazards.
- **Aliases + own constants** — the core uses `wet::array/optional/tuple/pair/clamp/min/max/move/forward/...` (from `backend.hpp`, → `std::` or `etl::`) and `wet::numbers::pi_v/...` (our own constexpr constants; resolves the `std::numbers` decision). No `<numbers>` anywhere in the core.
- **Freestanding math backend** — `math_backend.hpp` is `<cmath>`-free (the `StdMathFallback`/`<cmath>` moved to `std_fallback.hpp`, hosted-only); `WET_MATH_BACKEND_FREESTANDING` selects `series_backend.hpp`, which routes runtime math to the constexpr series in `constexpr_math.hpp`. `std::complex` interop in `complex.hpp` is `__has_include`-guarded.
- **Enforcement** — `make freestanding-check` (umbrella compiles under ETL + series math; no *our* header unconditionally pulls a hosted std header). A tup `etl` build **variant** (`Tuprules.lua` honors `CONFIG_BACKEND=ETL`) compiles and **runs the entire test suite** (678 cases / 9739 assertions) on ETL containers + series math, alongside the `std` variant — both green in `make`.

**Feasibility — spiked and confirmed (the gating risk is closed).** The load-bearing question was whether ETL preserves the constexpr-first synthesis invariant. Verified under `-std=c++20`:

- `etl::array` + `etl::optional` evaluate in constant context (ETL gates these on `ETL_CONSTEXPR`, which resolves to real `constexpr` on C++14+, [platform.h](../../libs/etl/include/etl/platform.h)).
- A throwaway LQR-grade DARE solver (matmul, transpose, **Gauss-Jordan inverse → `etl::optional`**, 1000-iteration fixed-point loop) backed by `etl::array`, fully `static_assert`-ed, **converged at compile time** and produced gains *identical* to the real library `design::discrete_lqr` (`K = [2.5857, 3.4434]` on a double-integrator). The riskiest constructs — `etl::array` element read/write in a constexpr loop, returning/unwrapping `etl::optional` in constexpr — all held.

**Core `std` surface to abstract** (everything else used by the core — `<cstddef>`, `<cstdint>`, `<type_traits>`, `<limits>`, `<concepts>`, `<initializer_list>`, most of `<utility>` — is already freestanding on GCC/Clang). The two backends supply each `wet::` alias as:

| `wet::` alias | `stdlib` | `etl` |
|---|---|---|
| `array` | `wet::array` | `etl::array` |
| `optional` / `nullopt` | `wet::optional` | `etl::optional` |
| `tuple` / `pair` | `wet::tuple` / `pair` | `etl::tuple` / `pair` |
| `clamp`/`min`/`max`/`swap` | `<algorithm>` | `etl/algorithm.h` |
| `pi_v` & constants | `std::numbers` | ETL / own constants |
| `index_sequence`, fold (matrix SRA) | `<utility>` (freestanding) | `etl::index_sequence` |
| math (`wet::sin/sqrt/...`) | math backend (`<cmath>`) | math backend |

ETL constexpr-equivalence confirmed for `array`/`optional` (spike above). Out of scope (already host-only behind `toolbox.hpp`, excluded by `embedded-check`): `<vector>`, `<string>`, `<functional>`, `<cstdio>`, `std::complex` (core uses `wet::complex`).

**Build order (all done ☑):**

1. ☑ **`make freestanding-check`** — compiles the `wet/control.hpp` umbrella under the `etl` + series-math backend and asserts no *our* header unconditionally pulls a hosted std header. (Implemented as a compile + source-grep rather than `-nostdinc++`: that flag also strips the freestanding C++ headers — `<cstddef>`, `<type_traits>`, `<limits>` — that ETL itself needs, so it can't compile ETL in this toolchain. The grep enforces the same property on the headers we own; ETL's own transitive includes are its per-target concern.)
2. ☑ **`wet/backend.hpp`** alias layer + `wet::numbers` constants + unified `wet/config.hpp` discovery.
3. ☑ **Core migrated behind the aliases** — every umbrella header off `std::array/optional/tuple/pair/clamp/...`, `std::numbers`, and `<cmath>`/`std::sin…`. The tup `etl` variant runs the whole suite as the regression gate.
4. ☑ Default profile stays `stdlib`; `etl` is opt-in (per-facility macros / the `etl` build variant).

**References:** ISO C++ [freestanding] implementation requirements; ETLCPP documentation (https://www.etlcpp.com).

**Acceptance ☑:**

- `make freestanding-check` green: umbrella compiles under the `etl` + series-math backend with no hosted-header leak from our headers. The `etl` build variant runs the **full test suite (678 cases / 9739 assertions) green** on ETL containers + series math (a stronger check than a single representative synthesis).
- Default (`stdlib`) profile unchanged for hosted users; the `std` variant suite, `make embedded-check`, and `make freestanding-check` all stay green in `make`.
- No `#include <...>` of a hosted header reachable from *our* `wet/control.hpp` headers under the `etl` backend.

**Decision items (this section):**
- ☑ **Profile selection surface → per-facility macros.** One `wet/config.hpp` discovers a macro-only `wet_profile.hpp`; containers key off `WET_BACKEND_ETL`, math off `WET_MATH_BACKEND_*`. Orthogonal, mix-and-match; not a single `WET_BACKEND` enum.
- ☑ **Alias home → one `backend.hpp`** for the container/utility aliases (math backend stays in `math/math_backend.hpp`); both wired through `wet/config.hpp`.
- ☑ **`std::numbers` replacement → our own constants** (`wet::numbers::pi_v/...` in `backend.hpp`), no dependency on `<numbers>` or ETL constants.
- ☐ Error-reporting model under the `etl` backend (ETL's error-handler callback vs the existing `bool success` result structs — keep the latter as the public contract). Deferred until an ETL target actually needs the callback path; `bool success` remains the public contract.

**Layer 2 — core controls & estimation building blocks.**

### 1. Luenberger / reduced-order observer ☑

Deterministic state observers via pole placement; the counterpart to the Kalman family. Prerequisite for output-feedback laws without a stochastic model, and for the disturbance observer (#4).

- Done (`estimation/observer.hpp`): full-order `Observer` and reduced-order (Gopinath) `ReducedOrderObserver`, with `design::synthesize_observer` / `design::synthesize_reduced_observer` via observer-form Ackermann (single-output), `StateSpace` + real/complex-pole overloads. Full-order uses the predictor recursion (error poles eig(A − LC)); reduced-order estimates only the unmeasured states and reads the measured state straight from `y` (ideal for encoder velocity).
- Single-output (NY == 1) only — MIMO-output placement is item #16; use the Kalman filter for MIMO until then.
- References: D. G. Luenberger, "An Introduction to Observers," IEEE TAC, 1971, https://doi.org/10.1109/TAC.1971.1099826; B. Gopinath, "On the Synthesis of Minimal-Order Observers," BSTJ, 1971.
- Acceptance: error dynamics match placed poles ✓; constexpr design ✓; reduced-order reconstructs unmeasured states ✓.

### 4. Disturbance observer + DOB control law ☑

**Built (2026-06):** two complementary disturbance observers in `estimation/disturbance_observer.hpp` (`test_disturbance_observer`, in the umbrella, embeddable + freestanding-clean):
- **Scalar innovation-based** `DisturbanceObserver` + `synthesize_disturbance_observer`: `d̂[k+1] = (1−leak)·d̂ + gain·(y_meas − y_pred)`, `compensate(u) = u − d̂`. Lightweight, model-free.
- **Classical Pn⁻¹·Q (Ohnishi) DOB** `synthesize_classical_dob(Bn, An, Qn, Qd)` → `ClassicalDobResult` + `ClassicalDisturbanceObserver<…>` runtime. Forms the two realized digital filters Fy = Q·Pn⁻¹ = (Qn·An)/(Qd·Bn) and Fu = Q, estimates `d̂ = Fy(y) − Fu(u)`, and `compensate(u_cmd, y) = u_cmd − d̂` (one-sample u-delay breaks the algebraic loop; exact at DC). A **bolt-on around any existing controller**, not a full controller — its distinct value vs ADRC's ESO. Arbitrary plant/Q order (z⁻¹ polynomials), validates causal realizability (leading Bn/An/Qd nonzero), constexpr, `.as<U>()`. Verified: step-load rejection to ~0, DC rejection under 20% model mismatch, open-loop disturbance recovery, constexpr.

**Overlap note:** ESO-based disturbance rejection is *already* delivered by `controllers/adrc.hpp` (ADRC, tested); the classical DOB above is the additive *bolt-on* form.

Observer-based disturbance estimation integrated with a base controller. Targets: PMSM/BLDC servo drives with load-torque disturbances, robot joints with friction/backlash, precision stages with external force. Depends on #1.

- Interface: `design::synthesize_dob(plant, nominal_inverse, q_filter_spec, base_controller)` → `DOBResult` / `DOBArtifacts` + runtime with per-tick compensation.
- References: S. Li et al., "Disturbance Observer-Based Control," CRC Press, 2016, https://doi.org/10.1201/b16570; W.-H. Chen et al., "Disturbance-Observer-Based Control... An Overview," IEEE TIE, 2016, https://doi.org/10.1109/TIE.2015.2478397
- Acceptance: step-load rejection; frequency-domain attenuation; model-mismatch regression.

### 12. UKF (sigma-point filter) ☐

Unscented filter for nonlinear estimation where EKF linearization is poor (map item `sigma-point`).

- Interface: `UKF<NX,NU,NY>` runtime mirroring the EKF API; constexpr sigma-point weights.
- Reference: Julier & Uhlmann, "Unscented Filtering and Nonlinear Estimation," Proc. IEEE, 2004. https://doi.org/10.1109/JPROC.2003.823141
- Acceptance: matches EKF on near-linear systems; outperforms on strongly nonlinear references.

### 3. Identification and excitation infrastructure ◐ (partially built)

**Re-baseline (2026-06):** built & tested — **excitation generators** (`controllers/excitation.hpp`, ~1150 lines, Chirp/PRBS/StepTrain/Ramp/MultiSine + `synthesize_*`, `test_excitation`) and the **generic `Cascade`** (`controllers/cascade.hpp`, `test_cascade`). Not built — `models.hpp` (Tier-2 builders), `utility/ring_log.hpp`, and all host-side identification (`analysis/identification.hpp` `tfest`/`ssest`, `analysis/frf.hpp`). So the on-target *excitation* half is largely done; the model-builders, ring-log, and the whole host-side ID/FRF half remain.

**Decision (2026-06): keep `Cascade` + the `SISOController` concept; redesign holistically when ready (not removed).** Verified the coupling is tiny — the `SISOController`/`SISOControllerWithBackCalculation` concepts (`controllers/controller_concept.hpp`) are used *only* by `cascade.hpp`, its `test_controller_concept.cpp`, and a `static_assert` in `test_cascade.cpp`; the other controllers merely `#include`/`@ref` it (inert). Both are *narrow but working/tested*, so rather than delete-and-rewrite (churn, lost tests, a gap), the concept header is marked **provisional** and will be folded into a **unified controller + observer concept** — one box every controller and observer fits, easy for users to extend (SISO + MIMO/`StateSpace`). Natural companion to #3's cascade-from-model synthesis. The north star: controllers and observers each satisfy a single, documented, user-extensible protocol.

The commissioning workflow: drive the plant with a known excitation signal, log the response, identify a model, synthesize gains, deploy. The library never asks "what kind of plant?" — the user describes their system by constructing a `StateSpace` / `TransferFunction` (directly or via Tier 2 model builders), and identification produces those same universal types from time-series data.

**On-target (embeddable, `wet/control.hpp`):**

- **Excitation generators** (`controllers/excitation.hpp`, new). Runtime signal sources, plant-agnostic, allocation-free, constexpr-configurable. Each exposes `T step(T t)` (or `T step()` with internal sample-time counter) and a `done()` predicate. Implement:
  - `Chirp<T>` — linear and logarithmic sweep; config `(amplitude, f_start_hz, f_end_hz, duration_s, mode)`.
  - `PRBS<T>` — maximum-length pseudo-random binary sequence; config `(amplitude, lfsr_order, clock_period_s, seed)`; reproducible across builds.
  - `StepTrain<T>` — alternating ±A steps with configurable hold; config `(amplitude, hold_s, cycles)`.
  - `Ramp<T>` — rate-limited slope with end-of-segment freeze; config `(target, rate, hold_at_end_s)`.
  - `MultiSine<NTones, T>` — sum of `NTones` sinusoids, `wet::array<Tone, NTones>` where `Tone = {amplitude, freq_hz, phase_rad}`; useful for crest-factor-controlled FRF excitation.
  - Each gets a corresponding `design::synthesize_*` validating the config (positive amplitudes, finite durations, etc.).
- **Generic cascade controller** (`controllers/cascade.hpp`, new). `template<typename Outer, typename Inner> class Cascade` composing any two controllers that satisfy a small `Controller` concept (`T control(T r, T y)`, `void reset()`). Outer-loop output becomes inner-loop reference; inner-loop output goes to the plant. Anti-windup is the inner controller's responsibility (existing `PIDController` already handles it). Tier 2 alias: `template<typename T> using CascadePPI = Cascade<PController<T>, PIDController<T>>;` (a tiny `PController<T>` may need to be extracted from `pid.hpp` — `PIDController` with `Ki=Kd=0` works but a dedicated struct is cleaner). Test against a hand-written cascade composition on a synthetic plant.
- **Data-logger ring buffer** (`utility/ring_log.hpp`, new). `template<std::size_t N, typename... Channels> class RingLog` — fixed-size SOA, `push(t, ch1, ch2, ...)`, overwrite-oldest semantics. Flush via `for_each([](auto t, auto... ch){ ... })` callback that hands ordered tuples to user-supplied transport (UART/SPI/JSON). No allocation, no opinions on wire format. Channel storage is `wet::tuple<wet::array<Channels, N>...>` so flush can stream one channel at a time if needed.
- **Tier 2 model builders** (`models.hpp`, new top-level header, embeddable — these only construct `StateSpace` / `TransferFunction`):
  - `models::single_mass(T M, T c)` → `StateSpace<2,1,1,T>` representing `M·ẍ + c·ẋ = u`, states `[x, ẋ]`, output `x`.
  - `models::second_order(T omega_n, T zeta)` → `StateSpace<2,1,1,T>` for standard underdamped/overdamped second-order in normalized form.
  - `models::two_mass(T M_m, T M_l, T k_belt, T c_belt, T r)` → `StateSpace<4,1,2,T>` (motor torque in; motor angle and load position out). States: `[θ_m, ω_m, x_l, v_l]`.
  - `models::fopdt(T K, T tau, T L)` → `TransferFunction` with explicit dead-time (Padé approximation order via template param, default 1).
  - Each returns the universal type — downstream design / analysis functions never know it came from a Tier 2 builder.

**Host-only (`wet/toolbox.hpp`):**

- **Generic identification** (`analysis/identification.hpp`, overhauled — currently only model-struct placeholders; the existing `estimation/identification.hpp` structs that fit get moved or re-exported, the recursive RLS in `estimation/recursive_least_squares.hpp` stays embedded):
  - `analysis::tfest<Nn, Nd, T>(time, u, y)` → `{TransferFunction<Nn,Nd,T>, FitMetrics}`. SISO output-error / PEM fit at user-specified numerator/denominator orders.
  - `analysis::ssest<NX, T>(time, u, y)` → `{StateSpace<NX,NU,NY,T>, FitMetrics}` (NU/NY deduced from `u`/`y` shapes). Subspace identification (N4SID).
  - `FitMetrics` = `{T residual_rms, T r_squared, T aic, T bic, bool success}`.
  - `analysis::validate(model, held_out_data)` → `ValidationResult` (cross-validation gating).
- **FRF estimation** (`analysis/frf.hpp`, new):
  - `analysis::frfest(time, u, y, freqs)` → `FRFPoints` with `{magnitude, phase, coherence}` per frequency. Windowed Welch-style averaging. Pairs with `Chirp` / `MultiSine` excitation.
- **Tier 2 identification wrappers** (in `analysis/identification.hpp`):
  - `analysis::identify_fopdt_from_step(time, u, y)` → `{FOPDTParameters{K, tau, L}, TransferFunction}` — calls `tfest<1,1>` plus a delay estimator.
  - `analysis::identify_second_order_from_chirp(time, u, y)` → `{SecondOrderParameters{omega_n, zeta, K}, StateSpace<2,1,1>}`.
  - `analysis::identify_two_mass(time, u_torque, theta_m, x_l)` → `{TwoMassParameters{M_m, M_l, k_belt, c_belt, r}, StateSpace<4,1,2>}`. Internally calls `ssest<4>`; applies a similarity transform so the resulting `StateSpace` has the physically meaningful state ordering of `models::two_mass`. Same fit as the generic `ssest`, friendlier output.
  - Pattern: every Tier 2 wrapper returns both the named-parameter struct *and* the universal type. Downstream code can take either.

**Workflow the spec enables:**

```cpp
// 1. On target: excite + log
constexpr auto chirp_cfg = design::synthesize_chirp<float>({.amplitude = 1.0f,
    .f_start_hz = 0.1f, .f_end_hz = 50.0f, .duration_s = 30.0f, .mode = ChirpMode::Log});
Chirp<float>                                     exciter{chirp_cfg};
RingLog<60000, float, float, float, float>       log;
while (!exciter.done()) {
    float u = exciter.step(t);
    plant.apply(u);
    log.push(t, u, theta_m_encoder, x_l_encoder);
    t += Ts;
}
log.flush([](auto t, auto u, auto th, auto x){ uart_csv(t, u, th, x); });

// 2. On host (toolbox build):
auto data        = load_csv("log.csv");
auto [model, fm] = analysis::identify_two_mass(data.t, data.u, data.theta_m, data.x_l);
assert(fm.r_squared > 0.95);
auto cascade     = design::synthesize_cascade_p_pi(model, specs);   // existing or new design fn

// 3. Back on target:
constinit CascadePPI<float> controller{cascade.as<float>()};
```

**Build order for an implementing agent.** Each phase is a clean stopping point; tests + roadmap status update per phase.

1. `controllers/excitation.hpp` + `tests/test_excitation.cpp`. Validate waveforms against closed-form sample values; check `done()` semantics; check `.as<U>()` on configs. Add to `wet/control.hpp` umbrella.
2. `controllers/cascade.hpp` + `tests/test_cascade.cpp`. If a `PController<T>` is needed, extract it from `pid.hpp` (or define alongside). Validate against a hand-written cascade on a 2-state plant. Add `CascadePPI<T>` alias. Add to umbrella.
3. `utility/ring_log.hpp` + `tests/test_ring_log.cpp`. Validate push/overwrite/flush ordering on multi-channel logs. Add to umbrella.
4. `wet/models.hpp` + `tests/test_models.cpp`. Validate each builder against analytical transfer functions (e.g. `models::two_mass` should give a 4-state `StateSpace` whose transfer function matches the textbook two-mass formula). Add to `wet/control.hpp` umbrella (embeddable).
5. `analysis/frf.hpp` + `tests/test_frf.cpp`. Host-only. Validate against synthetic data with known frequency response. Add to `wet/toolbox.hpp`.
6. `analysis/identification.hpp` overhaul (Tier 1 first: `tfest`, `ssest`, `validate`; Tier 2 next: `identify_fopdt_from_step`, `identify_two_mass`, etc.) + `tests/test_identification.cpp`. Host-only. Validate on synthetic data with known plants; check `make embedded-check` stays green. The Tier 2 wrappers must produce the same fit as direct Tier 1 calls plus an extra similarity transform / parameter unpacking step.

**References:** L. Ljung, "System Identification: Theory for the User," 2nd ed., 1999; P. Van Overschee & B. De Moor, "Subspace Identification for Linear Systems," 1996; R. Pintelon & J. Schoukens, "System Identification: A Frequency Domain Approach," 2nd ed., 2012.

**Acceptance:**

- Each excitation generator produces analytically-correct waveforms at sample points; `done()` fires when expected; configs validate.
- `Cascade<Outer, Inner>` output matches a hand-written cascade composition on the same plant exactly.
- `RingLog` push/overwrite/flush ordering is correct on multi-channel logs; no heap allocation; constexpr-friendly construction.
- `models::*` builders produce `StateSpace` / `TransferFunction` matching analytical forms.
- `tfest` / `ssest` recover known plants on noiseless synthetic data within 5% on coefficients (or eigenvalues for state-space).
- Tier 2 ID wrappers (`identify_two_mass`, `identify_fopdt_from_step`, …) recover physical parameters within 10% from chirp- or step-excited synthetic data with realistic encoder noise; return the same `StateSpace` / `TransferFunction` as the equivalent Tier 1 call (up to similarity).
- `make embedded-check` stays green: only excitation, cascade, ring-log, and `models.hpp` reach `wet/control.hpp`; `tfest`/`ssest`/`frfest`/`identify_*` are reachable only via `wet/toolbox.hpp`.

**Notes for implementers:**

- The existing `estimation/identification.hpp` is a placeholder with model structs (`FOPDTModel`, `SOPDTModel`, `ARXModel`, `FitMetrics`, `ValidationResult`). Reuse what fits; the host-side fitters land in `analysis/identification.hpp` (move or re-export structs that pull `<vector>`). Online RLS (`estimation/recursive_least_squares.hpp` ☑) stays embedded and untouched.
- `Cascade<Outer, Inner>` should require only `T control(T r, T y)` and `void reset()` from its inner / outer controllers. A `Controller` concept makes the error messages clean.
- Excitation generators take `T t` (absolute time) rather than tracking internally where possible — keeps them stateless and unit-testable. `PRBS` is the exception (internal LFSR state).
- For `models::two_mass`, choose the state ordering `[θ_m, ω_m, x_l, v_l]` so that `identify_two_mass`'s similarity transform has an obvious target. Document the equations of motion in the header so the user can verify the convention.
- The cascade-from-model design function (`design::synthesize_cascade_p_pi(plant, specs)` referenced in the workflow example) is *not* required by this item; existing tuning rules in `pid_design.hpp` suffice for the outer/inner gains. Treat the cascade-from-model synthesis as a separate (small) future item.

**Layer 3 — advanced controls & estimation.**

### 5. Repetitive controller ◐ (core built)

**Built (2026-06):** plug-in `RepetitiveController<MaxPeriod,T>` + `design::synthesize_repetitive(fs, f0, gain, Q, lead)` (`controllers/repetitive.hpp`, `test_repetitive.cpp`). Internal-model period-delay loop (`w[k]=Q·w[k-N]+e`, `u_rc=k_rc·w[k-N+m]`): one delay rejects the fundamental *and all harmonics*; scalar robustness Q and integer phase-lead m; fixed-size buffer (allocation-free), constexpr. Verified: periodic tracking error → 0, multi-harmonic (fundamental+3rd) disturbance rejection, Q<1 boundedness. Remaining (refinements): a low-pass (FIR) Q-filter option, plant-aware phase-lead/stability-margin design (vs a hand-tuned m), and the LQI/LQGI integration bundle.

Internal-model compensation for periodic disturbances over selected harmonics. Targets: grid-tied inverters, precision motion with periodic trajectories, rotating-machinery ripple. Depends on #2 (delay line + robustness filter).

- Interface: `design::synthesize_repetitive(plant, fundamental_frequency, harmonic_set, robustness_filter)` → `RepetitiveResult` / `RepetitiveArtifacts` + baseline-plus-repetitive runtime.
- Reference: S. Hara et al., "Repetitive Control System," IEEE TAC, 1988. https://doi.org/10.1109/9.1274
- Acceptance: harmonic-rejection benchmarks; stability margins with robustness filter; integration with LQI/LQGI.

### 6. Input shaping (command prefilter) ☑

**Built (2026-06):** `controllers/input_shaper.hpp` (`test_input_shaper`, in the umbrella, embeddable + freestanding-clean). `design::synthesize_input_shaper(fn_hz, zeta, Ts, ShaperType, ei_tol)` → `InputShaperResult` (normalized impulse amplitudes + integer sample delays) for `ShaperType::{ZV, ZVD, ZVDD, EI}` — ZV `[1,K]`, ZVD `[1,2K,K²]`, ZVDD `[1,3K,3K²,K³]` (exact for any ζ), EI tolerable-vibration form. Runtime `InputShaper<MaxDelay,T>` (ring-buffer impulse convolver, pass-through if the buffer is too small for the mode) and `InputShaperBank<NAxes,MaxDelay,T>` for multi-axis. Unity DC gain (amplitudes sum to 1 → steady command unchanged), constexpr, `.as<U>()`. Verified: coefficient normalization/delays, residual-vibration attenuation >50× on a 2nd-order mode, ZVD beating ZV under 15% detuning, undersized-buffer pass-through, multi-axis bank. Pure feedforward (no estimator); pairs with motion #20.
- Remaining (follow-up): `synthesize_input_shaper_from_modes(mode_list, …)` (convolve per-mode shapers for multi-mode plants) — single-mode shaping covers the headline use.
- References: Singer & Seering, "Preshaping Command Inputs to Reduce System Vibration," ASME JDSMC, 1990, https://doi.org/10.1115/1.2894142; Singhose et al., ASME JMD, 1994, https://doi.org/10.1115/1.2919428

### 7. Online PID tuning (relay + IFT) ⊘

Online-first PID/PI tuning that runs in the closed loop without requiring time-series export. The relay autotuner is the lightweight commissioning fallback for any SISO loop (model-free, deterministic, ISR-safe); IFT is the longer-running model-free improvement path. Pairs with the full identification pipeline in #3 — relay is the "no toolchain, no laptop" path, while #3 is the rich workflow when data export is available.

- **Relay autotuner ☑** (`controllers/relay_autotune.hpp`). Åström-Hägglund relay-feedback experiment driver: hysteresis relay in place of the controller induces a sustained limit cycle, then `(Kᵤ, Tᵤ)` are extracted from the limit-cycle period and amplitude. `design::synthesize_relay_autotune(config) → RelayAutotuneResult` plus `RelayAutotuner<T>` runtime exposing `step(y) → {u, status, Ku, Tu}` and a small lifecycle state machine (`Idle → Warmup → Measuring → Done | Failed`). Output `(Kᵤ, Tᵤ)` plugs into the existing tuning rules in `pid_design.hpp` — the autotuner is deliberately orthogonal to the tuning-rule choice. Default recommendation is `design::tyreus_luyben` (gentler than the original Ziegler-Nichols and the modern process-control standard); `ziegler_nichols` retained for textbook compatibility. Hysteresis-corrected gain: `Kᵤ = 4d / (π·√(a² − ε²))`. Acceptance tested on a discretized `1/(s+1)³` plant; describing-function error on `Kᵤ` is inherent to the method (~15–25% on low-order plants — first-harmonic approximation), `Tᵤ` is accurate to a few percent.
- **Biased / asymmetric relay + AMIGO tuning ☐.** Tightens the relay-test output beyond `(Kᵤ, Tᵤ)`: run the relay with asymmetric magnitudes `+d_pos` vs `−d_neg` (or equivalent setpoint bias) so the limit cycle is asymmetric, then track average `u` and average `y` over the measurement window to recover the static gain `Kₛ ≈ ū / ȳ`. Returns `(Kᵤ, Tᵤ, Kₛ)`. Pairs with `design::amigo_kappa_tau(Ku, Tu, Ks, Ts)` — Åström-Hägglund's optimization-derived rule (constrained to a target maximum sensitivity, typically Ms ≤ 1.4) that substantially out-performs ZN / Tyreus-Luyben across the FOPDT process family. Scope: ~30-line extension to `RelayAutotuner` (extra `bias` config field, running `u` and `y` averages over the measurement window) plus the AMIGO formula in `pid_design.hpp`. References: K. J. Åström & T. Hägglund, "Revisiting the Ziegler-Nichols step response method for PID control," Journal of Process Control 14(6), 2004, https://doi.org/10.1016/j.jprocont.2004.01.002; K. J. Åström & T. Hägglund, *Advanced PID Control*, ISA, 2006, Chapters 6 & 8.
- **IFT (iterative feedback tuning) ☐.** Gradient-based model-free tuning that perturbs the closed-loop gains and estimates the gradient of a tracking cost from two or three experiments per iteration. `design::synthesize_ift(reference, cost_weights, learning_rate)` → `IFTResult` / `IFTRuntime` with bumpless transfer between iterations.
- **Safety policy (shared).** Both runtimes must enforce: command-amplitude clamps, slew-rate limits, output saturation passthrough, bumpless transfer to/from the active controller, and rollback to the pre-tuning gains on degraded measurement / timeout.
- References: K. J. Åström & T. Hägglund, "Automatic tuning of simple regulators with specifications on phase and amplitude margins," Automatica 20(5), 1984, https://doi.org/10.1016/0005-1098(84)90014-1; H. Hjalmarsson, M. Gevers, S. Gunnarsson, O. Lequin, "Iterative feedback tuning: theory and applications," IEEE Control Systems Magazine 18(4), 1998, https://doi.org/10.1109/37.710876
- Acceptance (remaining): IFT convergence with bounded excitation; safety-policy tests (clamps, rate limits, bumpless transfer, rollback) for both runtimes.

### 8. Extremum-seeking control (ESC) ☐

Online optimization of a measured objective without an explicit model. Targets: PV MPPT under irradiance/temperature swings, thermal-efficiency optimization, drivetrain efficiency tracking. Header `controllers/esc.hpp` not yet created.

- Interface: `design::synthesize_esc(objective_signal, perturbation_policy, safety_limits)`, `..._mppt(...)` → `ESCResult` / `ESCRuntime` / `ESCSafetyStatus`.
- Reference: Y. Tan et al., "Extremum seeking control for discrete-time systems," IEEE TAC, 2002. https://doi.org/10.1109/9.983370
- Acceptance: MPPT over irradiance/temperature profiles; objective convergence with bounded perturbation; rollback/freeze on degraded measurements.

### 9. Harmonic detection & suppression (anti-chatter) ◐

Online dominant-harmonic detection and suppression via notch/PR/repetitive elements. Targets: lathe-turning chatter, spindle-tool resonance in CNC, gearbox tonal vibration. Depends on #2 (notch) and harmonic estimation. Includes the spectral primitives (windowed RFFT / Goertzel) and a harmonic tracker (`estimation/harmonic_estimation.hpp` is currently a placeholder).

- **Spectral primitive ☑** (`filters/spectral.hpp`, `test_spectral.cpp`, in the umbrella; embeddable + freestanding-clean). Deliberately *basic* — controls layer, not a DSP library. `Goertzel<T>` — a generalized single-bin DFT (arbitrary frequency) via the two-state recurrence (no buffer, O(1) memory): streaming `push(x)`, then `amplitude()`/`magnitude()`/`power()`/`phase()`, auto-restarting per block; exact for coherent sampling. `HarmonicAnalyzer<K,T>` — a Goertzel bank over a fundamental + K−1 harmonics giving per-harmonic `amplitude`/`phase`, `thd()`, fundamental `rms()`, and `total_harmonic_rms()`. Constexpr, allocation-free, float/double. Scope intentionally lean (rectangular/coherent only — no window zoo, no FFT; choose N for coherent sampling). Targets: grid V/I THD, motor torque-ripple. (`SOGI`/`MSTOGI` and `PR`/resonant controllers — the related single-frequency PE primitives — are likewise already built/tested.) **`SogiFll<T>`** (`filters/sogi.hpp`, `test_sogi_fll`): a SOGI with a frequency-locked loop — self-tuning, streaming single-tone tracker (`ω̇ = −Γ·ω·ε·qv′/(v′²+qv′²)`) that *finds* and follows the dominant frequency (no sweep, no prior knowledge) and reports locked frequency/amplitude/phase. The online/adaptive counterpart to the offline Goertzel sweep: use it for grid-frequency sync or to re-tune an input shaper / notch to a *drifting* resonance live. Verified: locks from an offset guess, tracks a frequency step, locks through broadband noise, amplitude recovery, band-clamp, constexpr.
- **Suppression bank ☑** (`controllers/harmonic_suppression.hpp`, `test_harmonic_suppression.cpp`, in the umbrella; embeddable + freestanding-clean). `design::synthesize_harmonic_suppressor(Kp, Ki_fund, w_fund, wc, Ts, harmonics)` → `HarmonicSuppressorResult<N,T>` (one `PRResult` per harmonic via `pr_harmonics`, with spec validation: positive Ts/frequency, no DC target, every resonator strictly below Nyquist), plus the `HarmonicSuppressor<N,T>` runtime — a parallel bank of `PRController` resonators summed on the loop error, satisfying the `(r,y)` protocol, with `set_fundamental()` for grid-frequency adaptation (preserves harmonic ratios). Selective/sparse counterpart to the repetitive comb (#5): target a chosen `{1,5,7,11}` rather than every harmonic of a period. Verified: multi-harmonic reference tracked to ~0 steady-state error, Nyquist/DC rejection, `as<U>()` precision conversion, constexpr.
- Remaining: host-side `analysis::detect_dominant_harmonics(signal_window, sample_rate)` (auto-pick the harmonic set to suppress — wire `HarmonicAnalyzer` output into the synthesizer); turning/milling chatter front-end `design::synthesize_chatter_suppressor_turning(machine_model, spindle_speed, sensor_config)`; adaptive online retune.
- References: Altintas & Budak, "Analytical Prediction of Stability Lobes in Milling," CIRP Annals, 1995, https://doi.org/10.1016/S0007-8506(07)62342-7; Mojiri & Bakhshai, "An Adaptive Notch Filter for Frequency Estimation," IEEE TAC, 2004, https://doi.org/10.1109/TAC.2003.822862; Goertzel, Amer. Math. Monthly, 1958, https://doi.org/10.2307/2308968
- Acceptance: detection latency / false-positive benchmarks; measured harmonic-amplitude reduction; closed-loop stability + actuator effort with adaptive updates.

### 10. LPV gain-scheduled LQG/LQI ⊘

Gain scheduling over an operating-point grid. Targets: UAV dynamics vs airspeed/altitude, vehicle lateral dynamics vs speed, manipulators with configuration-dependent models.

- Workflow: local linear models over the grid → local controller/estimator gains → scheduled runtime bundle with interpolation. Artifacts: schedule map, gain tables, local closed-loop analysis.
- References: Rugh & Shamma, "Research on Gain Scheduling," Automatica, 2000, https://doi.org/10.1016/S0005-1098(00)00058-3; Apkarian & Gahinet, IEEE TAC, 1995, https://doi.org/10.1109/9.384219
- Acceptance: continuity across transitions; stability over the certified envelope.

### 11. Super-twisting SMC ☑

**Built (2026-06):** second-order **super-twisting** algorithm — `design::synthesize_stsmc(L, Ts, λ, k_lin, ε, gain_margin)` (Levant/Moreno gains `k₁=1.5√L`, `k₂=1.1L`) + `design::stsmc(...)` direct-gain factory → `STSMCResult` + `SuperTwistingController` runtime (`controllers/smc.hpp`, `test_stsmc`). Continuous control `u = −k₁(|s|^½sign s + k_lin·s) + v`, `v̇ = −k₂(sign s + 3k_lin|s|^½sign s + 2k_lin²s)`: classic STA at `k_lin=0`, **generalized STA** (linear damping for noisy/under-modelled actuators) at `k_lin>0`, optional boundary layer `ε`. Canonical `control(s)` plus an `(r,y)` surface-builder convenience. Verified: finite-time rejection of a Lipschitz disturbance into an O(Ts) band, continuity vs first-order sign() chattering (the headline test), GSTA convergence, constexpr. Explicit-Euler integral; documented that noisy `s` is the real-world chattering source (filter it / supply `s`).

**Original (first-order baseline):** first-order SMC was already built & tested (`u = −(k/b0)·sign(s)` with boundary layer); #11 added the chattering-free second-order algorithm on top.

Second-order sliding-mode runtime with design-time gain synthesis. Targets: electromechanical drives with bounded matched disturbances, hydraulic/pneumatic actuators, converter current loops.

- Interface: `design::synthesize_stsmc(plant_or_surface, disturbance_bound, bandwidth_targets)` → `STSMCResult` / `STSMCArtifacts` + runtime with optional boundary layer.
- References: Moreno & Osorio, "Strict Lyapunov Functions for the Super-Twisting Algorithm," IEEE TAC, 2012, https://doi.org/10.1109/TAC.2012.2186179; Levant, Int. J. Control, 2003, https://doi.org/10.1080/0020717031000099029
- Acceptance: chattering metrics vs existing SMC; disturbance rejection with bounded effort.

### 13. H-infinity output feedback ⊘

Output-feedback synthesis for weighted generalized plants. Targets: flexible structures with modal uncertainty, flight control with structured uncertainty, active suspension/vibration isolation.

- Interface: `design::synthesize_hinf(augmented_plant, weighting_filters, gamma_search_bounds)` → `HInfResult` / `HInfArtifacts` + S/T analysis models. Workflow: coupled Riccati solves over gamma; select feasible controller.
- References: Doyle et al., "State-Space Solutions to Standard H2 and H-infinity Control Problems," IEEE TAC, 1989, https://doi.org/10.1109/9.29425; Glover & Doyle, Systems & Control Letters, 1988, https://doi.org/10.1016/0167-6911(88)90055-2
- Acceptance: weighted robust-stability/performance; regression vs known references.
- Placement: `toolbox.hpp` (resolved — gamma search allocates / iterates).

### 14. Constrained MPC ⊘

Finite-horizon constrained control with a deterministic runtime iteration budget. Targets: multivariable process control with limits, AV trajectory tracking with actuator limits, power electronics with current/voltage bounds.

- Interface: `design::synthesize_mpc(plant, horizon, weights, constraints, solver_budget)` → `MPCResult` / `MPCArtifacts` + fixed-iteration warm-start runtime.
- References: García, Prett & Morari, "Model Predictive Control: Theory and Practice — A Survey," Automatica, 1989, https://doi.org/10.1016/0005-1098(89)90002-2; Qin & Badgwell, Control Engineering Practice, 2003, https://doi.org/10.1016/S0967-0661(02)00186-7
- Acceptance: deterministic per-tick runtime bound; constraint-satisfaction regression; tracking/regulation tests.
- Placement: `toolbox.hpp` (resolved — QP solver allocates / iterates).

### 15. Moving horizon estimation (MHE) ☐

Optimization-based constrained estimator; toolbox-only (allocates/solver). The estimation counterpart to MPC.

- Reference: Rao, Rawlings & Mayne, "Constrained State Estimation for Nonlinear Discrete-Time Systems," IEEE TAC, 2003. https://doi.org/10.1109/TAC.2003.812777
- Acceptance: matches Kalman on linear/unconstrained problems; respects state constraints on nonlinear references.

### 20. Motion planning / trajectory generation ☐

Point-to-point and multi-segment trajectory generation for actuators and robot axes — the feedforward reference the controllers above track. Two complementary families, because (in the user's words) sometimes you want **time-optimal**, and sometimes you want **derivative-optimal over a fixed time**. Fits the three-tier pattern cleanly: a constexpr `design::` stage solves the boundary-value problem or segment times (with a feasibility/`success` flag), and an allocation-free runtime evaluates `step(t) → {position, velocity, acceleration, jerk}`. Embeddable. Generalizes the pure-feedforward input shaper (#6), which shapes an existing command; this *generates* the command.

**Family 1 — fixed-time, derivative-optimal (arbitrary boundary conditions).** Solve a polynomial boundary-value problem over a *fixed* duration `T` matching boundary conditions on position and its derivatives at both ends:

- `design::synthesize_poly_trajectory<Order>(bc_start, bc_end, T)` → coefficients + `success`. Cubic (match p, v), quintic (match p, v, a), septic/7th (match p, v, a, j). Solved as a small linear system via the existing `mat::solve` (the Vandermonde-style BVP), mirroring how `design::steinhart_hart` now solves its fit.
- Derivative-optimality falls out of the boundary conditions: a quintic with zero velocity/acceleration BCs is exactly the **minimum-jerk** profile over `T` (Flash–Hogan); cubic with zero-velocity BCs is **minimum-acceleration**; septic with zero jerk BCs is **minimum-snap** (Mellinger–Kumar, quadrotors). Tier-2 aliases `design::min_jerk(p0, pT, T)`, `min_accel(...)`, `min_snap(...)` wrap the general BVP with the appropriate zeroed BCs.
- Multi-waypoint splines (C²/C³-continuous piecewise polynomials through a waypoint list) as a later extension once the single-segment BVP is solid.

**Family 2 — constraint-limited, time-optimal.** Minimize duration subject to kinematic limits:

- `design::synthesize_trapezoidal(distance, v_max, a_max)` → segment times + `success` (accel/cruise/decel; degrades to triangular when cruise vanishes). Bang-bang acceleration.
- `design::synthesize_scurve(distance, v_max, a_max, j_max)` → the 7-segment double-S (jerk-limited) profile times. Bounded jerk → no actuator/torque step, the standard for smooth machine motion.
- Each runtime evaluates the piecewise profile and exposes the achieved peak v/a/j and total time.

**Multi-axis coordination.** A bank helper that time-scales each axis's profile to the slowest so a multi-DOF move starts and finishes synchronized (coordinated/“linear” moves), reusing the per-axis runtimes.

**References:** L. Biagiotti & C. Melchiorri, "Trajectory Planning for Automatic Machines and Robots," Springer, 2008 (polynomial, trapezoidal, double-S — the canonical treatment); T. Flash & N. Hogan, "The coordination of arm movements: an experimentally confirmed mathematical model," J. Neurosci. 5(7), 1985, https://doi.org/10.1523/JNEUROSCI.05-07-01688.1985 (minimum jerk); D. Mellinger & V. Kumar, "Minimum snap trajectory generation and control for quadrotors," ICRA 2011, https://doi.org/10.1109/ICRA.2011.5980409; S. Macfarlane & E. A. Croft, "Jerk-bounded manipulator trajectory planning," IEEE T-RA 19(1), 2003, https://doi.org/10.1109/TRA.2002.807548.

**Acceptance:**

- Polynomial trajectories satisfy all boundary conditions exactly at both endpoints; the synthesized quintic with zeroed v/a BCs matches the closed-form minimum-jerk profile to numerical tolerance; the BVP solve reports `success=false` on degenerate `T`.
- Trapezoidal and S-curve profiles respect `v_max`/`a_max`/`j_max`, are continuous in the claimed derivatives (S-curve: jerk-bounded ⇒ continuous acceleration), and report the correct minimum time; triangular/short-move degeneracies handled.
- Multi-axis bank arrives synchronized (all axes reach their endpoints at the same instant).
- All runtimes constexpr-constructible and allocation-free; `make embedded-check` green.

**Decision items (this section):** header layout (`motion/` directory vs single `trajectory.hpp`); whether the BVP coefficient solve is done at `design::` time only (constexpr `mat::solve`) so the runtime stores just coefficients; representation of multi-segment/waypoint trajectories (fixed-`N` segment array vs caller-supplied storage to stay allocation-free).

**Layer 4 — tooling (host-only).**

### 18. Multi-rate simulation harness ☐

The current simulation harness (`simulation/simulate.hpp`) advances every block at a single fixed `Ts`. Real deployments are multi-rate: an ISR-level loop (current/inner-loop control, PWM, observers) runs at 8 kHz–128+ kHz, while RTOS tasks run the outer loops and supervisory logic at 1 / 10 / 100 / 1000 Hz. A faithful closed-loop simulation has to reproduce this rate hierarchy — including the inter-rate effects that single-rate sim hides: sample-and-hold across the rate boundary, the one-sample transport delay an outer task sees on inner-loop state, and aliasing of fast dynamics into slow samplers.

- **Model.** Each block runs at a rate that is an integer divisor of the fastest *discrete* (base) rate; the harness ticks at the base rate (e.g. the ISR rate) and fires each slower block on its decimation boundary. The plant is *not* one of these discrete blocks — it integrates on its own much finer clock (continuous-time approximation, e.g. 1 µs / 1 MHz sub-step, far faster than even the ISR), so fast plant dynamics and inter-sample ripple are resolved between controller ticks. The plant sub-step need not be an integer divisor of the control rates; the harness advances the plant to each control-tick boundary and only samples/holds at those boundaries.
- **Rate-boundary semantics.** Every crossing is a sample-and-hold, including controller→plant. A discrete block's output is held constant (ZOH) on the consumer's clock until that block fires again: ZOH from a slower producer to a faster consumer, and from the fastest discrete block down onto the continuous plant (the plant sees a piecewise-constant input between actuation updates, not a fresh value every integration sub-step). Faster→slower is latch/decimate (the slow sampler reads the held value at its tick). Make the outer→inner reference handoff and the inner→outer state feedback explicit so the simulated transport delay matches what the RTOS actually incurs. This — plus the controller→plant ZOH — is where most "works in sim, oscillates on hardware" gaps come from.
- **Scope / placement.** Host-only (`wet/toolbox.hpp`) — it allocates time-series like the existing `sim::` harness. The per-block runtimes being scheduled are the same allocation-free controllers/estimators from `wet/control.hpp`; the harness only orchestrates *when* each `step()` is called. No new constraint on the embeddable core.
- **Relationship to existing code.** Generalizes `simulate.hpp` (single-rate is the degenerate case: one block, base rate = block rate). Reuses `simulation/integrator.hpp` for the plant sub-step. Pairs naturally with the cascade controllers (#3) and any inner/outer split (current→speed→position).
- **Open questions.** Schedule specification (per-block divisor vs. explicit Hz with validation that each divides the base rate); whether to model task jitter / execution-time offset within a tick or assume ideal periodic firing (ideal first, jitter as a later refinement); how to expose the multi-rate trajectories for plotting when channels are sampled at different rates.
- Reference: Franklin, Powell & Workman, "Digital Control of Dynamic Systems" (3rd ed., 1998), multirate sampling chapter; M. Cimino & P. R. Pagilla, "Conditions for multirate sampled-data control," on inter-sample behavior and rate-boundary effects.
- Acceptance: a two-rate example (fast inner current loop + slow outer loop) reproduces the inner-loop transport delay and sample-and-hold seen on hardware; single-rate results match the existing `simulate.hpp` harness when all blocks share one rate.

## Testing and Documentation

- One test translation unit per new header.
- Compile-time and runtime assertions for each synthesis API.
- End-to-end examples for the first batch of items above.
- Keep `wet/control.hpp` / `wet/toolbox.hpp` placement correct; `make embedded-check` must stay green.
- Update README/CLAUDE.md as features land.

## Decision Items

- ☑ Runtime `step(...)` status policy → **out-of-band `status()`**. `step(r, y) → control` stays the allocation-free hot path; per-tick health is read via a `status()` / `last_status` accessor. Rationale: a degraded controller still actuates, so the runtime is always-value-plus-health (a *product* type), not the sum type `expected<T, E>` models; per-tick unwrap also adds branch cost/noise in ISRs. `tl::expected`/`std::expected` not adopted — wrong shape here, and not freestanding-available (third-party / C++23) against the no-mandatory-dep constraint. Design-time keeps the established `{Result, bool success}` contract.
- ☑ `control.hpp` vs `toolbox.hpp` placement → **H∞ (#13), MPC (#14), MHE (#15) are toolbox** (allocators / iterative solvers); spectral (#9 Goertzel) is embeddable.
- ☑ Repetitive control (#5) rollout → **SISO first**, then MIMO (matches every other rollout).
- ☑ Input shaping (#6) placement → **`controllers/`** (it generates/shapes commands, not a signal filter; pairs with motion #20).
- ☑ First online tuning method → **relay autotuner** (#7, done) with the Tyreus-Luyben default; shared safety policy (clamps, slew, bumpless, rollback) is the cross-cutting requirement on every tuning runtime.
- Default ESC perturbation policy for MPPT (frequency, amplitude schedule, freeze criteria).
- Anti-chatter suppressor update strategy: continuous adaptation vs gated updates.
- **Direction set (not yet built):** FFT/spectral dependency → **in-house Goertzel** for the known-frequency embeddable path (#9); host RFFT/FRF stays a separate toolbox concern. (`filters/spectral.hpp` is still ☐ — see #9.)
- ☑ Observer API shape → **shared `design::synthesize_observer`** returning `L` (#1), kept orthogonal to the (future) robust `place` (#16).
