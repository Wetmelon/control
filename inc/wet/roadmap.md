# wet Library Roadmap

Date: 2026-05-25
Owner: controls core
Status: draft

## Table of Contents

- [wet Library Roadmap](#wet-library-roadmap)
  - [Table of Contents](#table-of-contents)
  - [Purpose](#purpose)
  - [Design Constraints](#design-constraints)
  - [Architecture layers \& build priority](#architecture-layers--build-priority)
  - [Dependencies](#dependencies)
  - [Standard Synthesis Workflow](#standard-synthesis-workflow)
  - [Roadmap](#roadmap)
    - [☑  DSP biquad/notch family + utility blocks](#--dsp-biquadnotch-family--utility-blocks)
    - [☑ Embedded firmware primitives](#-embedded-firmware-primitives)
    - [☑ Robust MIMO pole placement (true `place`)](#-robust-mimo-pole-placement-true-place)
    - [☑ Fast-math-robust filter coefficient designs](#-fast-math-robust-filter-coefficient-designs)
    - [☑ Backend-agnostic core: stdlib or ETL (freestanding-capable)](#-backend-agnostic-core-stdlib-or-etl-freestanding-capable)
    - [☑ Luenberger / reduced-order observer](#-luenberger--reduced-order-observer)
    - [☑ Disturbance observer + DOB control law](#-disturbance-observer--dob-control-law)
    - [☑ UKF (sigma-point filter)](#-ukf-sigma-point-filter)
    - [◐ Identification and excitation infrastructure](#-identification-and-excitation-infrastructure)
    - [☑ Repetitive controller](#-repetitive-controller)
    - [☑ Input shaping (command prefilter)](#-input-shaping-command-prefilter)
    - [⊘ Online PID tuning (relay + IFT)](#-online-pid-tuning-relay--ift)
    - [☑ Extremum-seeking control (ESC)](#-extremum-seeking-control-esc)
    - [◐ Harmonic detection \& suppression (anti-chatter)](#-harmonic-detection--suppression-anti-chatter)
    - [⊘ LPV gain-scheduled LQG/LQI](#-lpv-gain-scheduled-lqglqi)
    - [☑ Super-twisting SMC](#-super-twisting-smc)
    - [⊘ H-infinity output feedback](#-h-infinity-output-feedback)
    - [⊘ Constrained MPC](#-constrained-mpc)
    - [☐ Moving horizon estimation (MHE)](#-moving-horizon-estimation-mhe)
    - [☑ Motion planning / trajectory generation](#-motion-planning--trajectory-generation)
    - [☑ Stewart platform kinematics (6-DOF parallel manipulator)](#-stewart-platform-kinematics-6-dof-parallel-manipulator)
    - [◐ Serial N-DOF manipulator kinematics (rotary-joint arm, N ≤ 6)](#-serial-n-dof-manipulator-kinematics-rotary-joint-arm-n--6)
    - [☑ Motion-system kinematic mappings (Cartesian / CoreXY / polar / delta)](#-motion-system-kinematic-mappings-cartesian--corexy--polar--delta)
    - [☐ Multi-rate simulation harness](#-multi-rate-simulation-harness)
    - [◐ Power-electronics modulation (inverter PWM schemes)](#-power-electronics-modulation-inverter-pwm-schemes)
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

**Layer 3 — advanced controls & estimation.** Composite, adaptive, or optimization-based, built from Layers 1–2: repetitive control (#5), input shaping (#6), online PID tuning (#7), extremum-seeking (#8), harmonic detection & suppression (#9), LPV gain scheduling (#10), super-twisting SMC (#11), H∞ output feedback (#13), constrained MPC (#14), moving-horizon estimation (#15), motion planning (#20), Stewart-platform kinematics (#22), serial-arm kinematics (#23), motion-system kinematic mappings (#24).

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

### ☑  DSP biquad/notch family + utility blocks

Complete the biquad family beyond low-pass and add everyday runtime blocks. Constexpr coefficient designers; allocation-free runtimes. Unblocks harmonic suppression (#9) and is broadly useful.

- Done (`filters/filters.hpp`): RBJ-cookbook biquad designers `design::notch`, `bandpass`, `highpass_2nd`, `peaking`, `lowshelf`, `highshelf` → `SecondOrderCoeffs` (Q-parameterized, designed directly in the digital domain); runtimes `Biquad` (Direct Form I) and `BiquadCascade<N>` (SOS cascade). All constexpr + float/double.
- Done (`filters/blocks.hpp`): utility runtime blocks `MovingAverage<N>`, `RunningRMS<N>`, `MedianFilter<N>`, `RateLimiter` (symmetric + asymmetric slew), `DirtyDerivative` (Tustin band-limited d/dt), `ClampedIntegrator` (forward-Euler + anti-windup clamp; named to avoid colliding with the ODE-solver `Integrator<NX,T>` in `simulation/integrator.hpp`), `Deadband` (continuous-slope), `Hysteresis` (Schmitt trigger). All constexpr, allocation-free, one-sample-per-call; in the `wet/control.hpp` umbrella.
- Done (`filters/differentiator.hpp`, `test_differentiator.cpp`): `RobustExactDifferentiator<T>` (alias `LevantDifferentiator<T>`) — Levant's first-order sliding-mode (super-twisting) differentiator. Model-free clean derivative of a noisy/quantized signal with far less phase lag than an LPF; also exposes a denoised `value()`. `update(f) → ḟ` from a second-derivative bound `L` and the std STA gains (λ0=1.5, λ1=1.1). The matched rate-estimator for the super-twisting controller (#11) and for servo **encoder velocity** (quantization-dominated low-speed differentiation). Constexpr, allocation-free; in the umbrella. Examples: `example_encoder_velocity` (beats raw/LPF'd finite-difference on RMS error *and* near direction reversals), and feeds clean rate in `example_swashplate_stsmc`. Ref: Levant, "Robust exact differentiation via sliding mode technique," Automatica 1998.
- References: Oppenheim & Schafer, "Discrete-Time Signal Processing," 3rd ed., 2009; R. Bristow-Johnson, "Cookbook formulae for audio EQ biquad filter coefficients."
- Acceptance: per-type frequency-response checks ✓; SOS cascade ✓; float/double parity ✓; utility blocks (`tests/test_blocks.cpp`: defining-behaviour + fill/reset edges + constexpr construction) ✓.

### ☑ Embedded firmware primitives

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

### ☑ Robust MIMO pole placement (true `place`)

Foundational numerical routine: place the eigenvalues of (A − BK) for **multi-input** systems, spending the extra gain freedom to maximize numerical robustness (minimize eigenvector conditioning). Equivalent to MATLAB's `place`.

- **Done (`controllers/pole_placement.hpp`, 2026-06).** `design::place(A, B, poles)` → `wet::optional<K>` via Kautsky–Nichols–Van Dooren. Real-pole path runs KNV **Method 0** (each eigenvector iteratively re-chosen within its admissible subspace to be maximally orthogonal to the others, minimizing κ(X)); complex-conjugate poles are assigned in **real arithmetic** via real (Re v, Im v) eigenvector pairs and 2×2 real Λ-blocks `[[σ,ω],[−ω,σ]]`, so K stays real. Built on a new `mat::full_qr` (full Householder QR — the orthogonal complement of B and the per-eigenvalue admissible null spaces). Single-input reduces to and matches `matlab::acker`; `NU == NX` uses the direct `B⁻¹(A − Λ)` form. Rejects (→ `nullopt`) rank-deficient B, multiplicity > NU (non-defective spectra only), and unpaired complex poles. Fully constexpr. `matlab::place` is the thin alias.
- Naming: `matlab::acker` (single-input Ackermann) already existed; `place` is the new robust MIMO routine — no rename was needed.
- Reference: J. Kautsky, N. K. Nichols, P. Van Dooren, "Robust pole assignment in linear state feedback," Int. J. Control, 1985. https://doi.org/10.1080/00207178508933420
- Acceptance ☑: places assignable MIMO real & complex spectra to ~1e-12 (`test_pole_placement.cpp`); rejects multiplicity > NU, rank-deficient B, and dangling complex poles; single-input matches `acker`; constexpr-evaluable.
- Remaining refinement (optional): the conditioning sweep is applied on the all-real path; complex pairs assign from the orthonormal admissible basis without the extra sweep (placement still exact, conditioning good but not sweep-optimized).

### ☑ Fast-math-robust filter coefficient designs

Restructure bilinear filter designers so that the unit-DC-gain identity (and other intended algebraic relations) holds *constructively* rather than emerging from exact cancellation, so the property survives `-fassociative-math` reassociation in downstream user builds.

- **`design::lowpass_2nd` ☑ fixed.** The original bilinear formula didn't merely drift under fast-math — its coefficients placed both poles essentially *on* the unit circle (|p| ≈ 1.0), making it a near-passthrough rather than a low-pass at all (measured |H| ≈ 1.0 at DC, fc, and 10·fc). Rewritten so the numerator taps are derived directly from the unit-DC-gain identity: since the bilinear numerator is ω₀²(1+z⁻¹)² the taps are in fixed ratio 1:2:1, so they are pinned as `dc_sum·{¼, ½, ¼}` with `dc_sum = 1 + a1 + a2`. Unity DC gain now holds by construction (and the ¼/½ factors are exact powers of two), identical under strict-IEEE and `-ffast-math`. Verified: |H| = 1.000 at DC, 0.707 at fc, 0.009 at 10·fc; DC-gain regression test tightened back to 1e-3, settling test back to 0.01.
- **Audit pass ☑ done.** Swept the six RBJ designers (`notch`, `bandpass`, `highpass_2nd`, `peaking`, `lowshelf`, `highshelf`) with tight band-edge gain-identity regression tests evaluated from the *stored* coefficients (`test_biquad.cpp`, "band-edge gain identities hold tightly under -ffast-math"): notch unity passband (DC + Nyquist) to 1e-9, bandpass DC/Nyquist nulls to 1e-12, highpass DC null to 1e-9, lowpass_2nd unity DC to 1e-12, peaking/shelf unity on the flat side to 1e-9. All hold under the runner's `-ffast-math` — as predicted, the RBJ single-expression formulas are not fragile. Reviewed the StateSpace→coeffs paths (`to_coeffs`): they faithfully transcribe the discretized state-space (`b`/`a` read straight from `A`/`B`/`C`/`D`), so there is no cancellation-enforced identity to restructure — DC gain is simply whatever the plant has.
- Reference: Higham, "Accuracy and Stability of Numerical Algorithms" (2nd ed., 2002), §3 on conditioned summation; Oppenheim & Schafer, "Discrete-Time Signal Processing" (3rd ed., 2009), bilinear-transform chapter.
- Acceptance ☑: DC-gain regression tests hold under the runner's `-ffast-math` — `lowpass_2nd` (`test_filters.cpp`: DC-gain at 1e-3, settling at 0.01, plus a cross-zeta unity-gain case), and the six RBJ designers (`test_biquad.cpp`: band-edge gain identities from stored coefficients, 1e-9–1e-12). `to_coeffs` reviewed — no enforced identity to guard.

### ☑ Backend-agnostic core: stdlib or ETL (freestanding-capable)

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

### ☑ Luenberger / reduced-order observer

Deterministic state observers via pole placement; the counterpart to the Kalman family. Prerequisite for output-feedback laws without a stochastic model, and for the disturbance observer (#4).

- Done (`estimation/observer.hpp`): full-order `Observer` and reduced-order (Gopinath) `ReducedOrderObserver`, with `design::synthesize_observer` / `design::synthesize_reduced_observer` via observer-form Ackermann (single-output), `StateSpace` + real/complex-pole overloads. Full-order uses the predictor recursion (error poles eig(A − LC)); reduced-order estimates only the unmeasured states and reads the measured state straight from `y` (ideal for encoder velocity).
- Single-output (NY == 1) only — MIMO-output placement is item #16; use the Kalman filter for MIMO until then.
- References: D. G. Luenberger, "An Introduction to Observers," IEEE TAC, 1971, https://doi.org/10.1109/TAC.1971.1099826; B. Gopinath, "On the Synthesis of Minimal-Order Observers," BSTJ, 1971.
- Acceptance: error dynamics match placed poles ✓; constexpr design ✓; reduced-order reconstructs unmeasured states ✓.

### ☑ Disturbance observer + DOB control law

**Built (2026-06):** two complementary disturbance observers in `estimation/disturbance_observer.hpp` (`test_disturbance_observer`, in the umbrella, embeddable + freestanding-clean):
- **Scalar innovation-based** `DisturbanceObserver` + `synthesize_disturbance_observer`: `d̂[k+1] = (1−leak)·d̂ + gain·(y_meas − y_pred)`, `compensate(u) = u − d̂`. Lightweight, model-free.
- **Classical Pn⁻¹·Q (Ohnishi) DOB** `synthesize_classical_dob(Bn, An, Qn, Qd)` → `ClassicalDobResult` + `ClassicalDisturbanceObserver<…>` runtime. Forms the two realized digital filters Fy = Q·Pn⁻¹ = (Qn·An)/(Qd·Bn) and Fu = Q, estimates `d̂ = Fy(y) − Fu(u)`, and `compensate(u_cmd, y) = u_cmd − d̂` (one-sample u-delay breaks the algebraic loop; exact at DC). A **bolt-on around any existing controller**, not a full controller — its distinct value vs ADRC's ESO. Arbitrary plant/Q order (z⁻¹ polynomials), validates causal realizability (leading Bn/An/Qd nonzero), constexpr, `.as<U>()`. Verified: step-load rejection to ~0, DC rejection under 20% model mismatch, open-loop disturbance recovery, constexpr.

**Overlap note:** ESO-based disturbance rejection is *already* delivered by `controllers/adrc.hpp` (ADRC, tested); the classical DOB above is the additive *bolt-on* form.

Observer-based disturbance estimation integrated with a base controller. Targets: PMSM/BLDC servo drives with load-torque disturbances, robot joints with friction/backlash, precision stages with external force. Depends on #1.

- Interface: `design::synthesize_dob(plant, nominal_inverse, q_filter_spec, base_controller)` → `DOBResult` / `DOBArtifacts` + runtime with per-tick compensation.
- References: S. Li et al., "Disturbance Observer-Based Control," CRC Press, 2016, https://doi.org/10.1201/b16570; W.-H. Chen et al., "Disturbance-Observer-Based Control... An Overview," IEEE TIE, 2016, https://doi.org/10.1109/TIE.2015.2478397
- Acceptance: step-load rejection; frequency-domain attenuation; model-mismatch regression.

### ☑ UKF (sigma-point filter)

Unscented filter for nonlinear estimation where EKF linearization is poor (map item `sigma-point`).

- **Done (`estimation/ukf.hpp`, `test_ukf.cpp`, 2026-06; in the umbrella, embeddable + freestanding-clean).** `UnscentedKalmanFilter<NX,NU,NY,T>` (short alias `UKF`) mirroring the EKF predict/update API, but the user supplies **plain nonlinear maps** `f(x,u)→ColVec<NX>` / `h(x,u)→ColVec<NY>` (concepts `UKFStateFn`/`UKFMeasFn`) — **no Jacobian**. Scaled unscented transform with `UnscentedParams{alpha,beta,kappa}` (defaults 1e-3 / 2 / 0); constexpr mean/covariance weights computed in the constructor. Each step draws the `2·NX+1` sigma points as `x ± col_j(√((NX+λ)P))` via the lower Cholesky factor (`mat::cholesky`), transforms them through the exact nonlinearity, and rebuilds the posterior by the weighted transform; the gain comes from `K = Pxy·Pyy⁻¹` (Cholesky solve). `predict`/`update` both return `bool` (false on non-PD covariance / singular `Pyy`). `.as<U>()`-style type-conversion ctor; `set_state`/`set_covariance` mutators for inter-step intervention (sequential scalar updates with clamping), as in the EKF.
- Reference: Julier & Uhlmann, "Unscented Filtering and Nonlinear Estimation," Proc. IEEE, 2004. https://doi.org/10.1109/JPROC.2003.823141; Wan & van der Merwe, "The Unscented Kalman Filter for Nonlinear Estimation," IEEE AS-SPCC, 2000; Simon, "Optimal State Estimation," 2006, §14.3.
- Acceptance ☑: reduces exactly to a linear KF on linear f/h (recovers unmeasured velocity to 1e-2); tracks a strongly nonlinear range-only measurement onto the correct manifold; predict grows / update shrinks covariance; graceful `false` on singular innovation covariance; float instantiation; `UKF` alias. Full suite green (807 cases / 18611 assertions), `make embedded-check` + `freestanding-check` green.

### ◐ Identification and excitation infrastructure

**Re-baseline (2026-06):** built & tested — **excitation generators** (`controllers/excitation.hpp`, ~1150 lines, Chirp/PRBS/StepTrain/Ramp/MultiSine + `synthesize_*`, `test_excitation`) and the **generic `Cascade`** (`controllers/cascade.hpp`, `test_cascade`). Not built — `models.hpp` (Tier-2 builders) and all host-side identification (`analysis/identification.hpp` `tfest`/`ssest`, `analysis/frf.hpp`). So the on-target *excitation* half is largely done; the model-builders and the whole host-side ID/FRF half remain. (A `utility/ring_log.hpp` data-logger was dropped from scope — fixed-size logging/ring buffers are ETL's domain per the design constraints; users wire excitation→log with `etl::circular_buffer` or their own transport.)

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
Chirp<float> exciter{chirp_cfg};
while (!exciter.done()) {
    float u = exciter.step(t);
    plant.apply(u);
    uart_csv(t, u, theta_m_encoder, x_l_encoder);   // user-supplied transport (or etl::circular_buffer)
    t += Ts;
}

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
3. `wet/models.hpp` + `tests/test_models.cpp`. Validate each builder against analytical transfer functions (e.g. `models::two_mass` should give a 4-state `StateSpace` whose transfer function matches the textbook two-mass formula). Add to `wet/control.hpp` umbrella (embeddable).
4. `analysis/frf.hpp` + `tests/test_frf.cpp`. Host-only. Validate against synthetic data with known frequency response. Add to `wet/toolbox.hpp`.
5. `analysis/identification.hpp` overhaul (Tier 1 first: `tfest`, `ssest`, `validate`; Tier 2 next: `identify_fopdt_from_step`, `identify_two_mass`, etc.) + `tests/test_identification.cpp`. Host-only. Validate on synthetic data with known plants; check `make embedded-check` stays green. The Tier 2 wrappers must produce the same fit as direct Tier 1 calls plus an extra similarity transform / parameter unpacking step.

**References:** L. Ljung, "System Identification: Theory for the User," 2nd ed., 1999; P. Van Overschee & B. De Moor, "Subspace Identification for Linear Systems," 1996; R. Pintelon & J. Schoukens, "System Identification: A Frequency Domain Approach," 2nd ed., 2012.

**Acceptance:**

- Each excitation generator produces analytically-correct waveforms at sample points; `done()` fires when expected; configs validate.
- `Cascade<Outer, Inner>` output matches a hand-written cascade composition on the same plant exactly.
- `models::*` builders produce `StateSpace` / `TransferFunction` matching analytical forms.
- `tfest` / `ssest` recover known plants on noiseless synthetic data within 5% on coefficients (or eigenvalues for state-space).
- Tier 2 ID wrappers (`identify_two_mass`, `identify_fopdt_from_step`, …) recover physical parameters within 10% from chirp- or step-excited synthetic data with realistic encoder noise; return the same `StateSpace` / `TransferFunction` as the equivalent Tier 1 call (up to similarity).
- `make embedded-check` stays green: only excitation, cascade, and `models.hpp` reach `wet/control.hpp`; `tfest`/`ssest`/`frfest`/`identify_*` are reachable only via `wet/toolbox.hpp`.

**Notes for implementers:**

- The existing `estimation/identification.hpp` is a placeholder with model structs (`FOPDTModel`, `SOPDTModel`, `ARXModel`, `FitMetrics`, `ValidationResult`). Reuse what fits; the host-side fitters land in `analysis/identification.hpp` (move or re-export structs that pull `<vector>`). Online RLS (`estimation/recursive_least_squares.hpp` ☑) stays embedded and untouched.
- `Cascade<Outer, Inner>` should require only `T control(T r, T y)` and `void reset()` from its inner / outer controllers. A `Controller` concept makes the error messages clean.
- Excitation generators take `T t` (absolute time) rather than tracking internally where possible — keeps them stateless and unit-testable. `PRBS` is the exception (internal LFSR state).
- For `models::two_mass`, choose the state ordering `[θ_m, ω_m, x_l, v_l]` so that `identify_two_mass`'s similarity transform has an obvious target. Document the equations of motion in the header so the user can verify the convention.
- The cascade-from-model design function (`design::synthesize_cascade_p_pi(plant, specs)` referenced in the workflow example) is *not* required by this item; existing tuning rules in `pid_design.hpp` suffice for the outer/inner gains. Treat the cascade-from-model synthesis as a separate (small) future item.

**Layer 3 — advanced controls & estimation.**

### ☑ Repetitive controller

**Built (2026-06):** plug-in `RepetitiveController<MaxPeriod, T, MaxQHalf>` + `design::synthesize_repetitive(fs, f0, gain, Q, lead)` and `design::synthesize_repetitive_binomial<M>(fs, f0, gain, lead)` (`controllers/repetitive.hpp`, `test_repetitive.cpp`). Internal-model period-delay loop (`w[k]=Q(z)·w[k-N]+e`, `u_rc=k_rc·w[k-N+m]`): one delay rejects the fundamental *and all harmonics*; integer phase-lead m; fixed-size buffer (allocation-free), constexpr.
- **Robustness filter Q:** scalar Q ∈ (0,1] *or* a **zero-phase low-pass FIR** Q(z)=Σ q_i z^{−i} (symmetric taps, `MaxQHalf` half-width). The FIR keeps near-unity gain on the low harmonics (full rejection) but rolls off near Nyquist (robust stability) without adding phase — realizable because Q multiplies the N-delayed signal so the "future" taps read already-buffered samples. Default family is the unity-DC binomial (`synthesize_repetitive_binomial<M>`: M=1→[1,2,1]/4, M=2→[1,4,6,4,1]/16). The scalar Q is the `MaxQHalf=0` special case (existing API unchanged).
- Verified: periodic tracking error → 0, multi-harmonic disturbance rejection (scalar Q=1 perfect; binomial Q strong-but-rolled-off as designed), binomial tap/DC-gain correctness, and the headline robustness case — **a binomial FIR Q stays bounded in an uncompensated-delay loop where the scalar Q=1 diverges to 1e32** (`|Q − k_rc z^{−d}| < 1`).
- Optional follow-ups (not blocking): plant-aware phase-lead/stability-margin auto-design (vs a hand-tuned m), and a packaged LQI/LQGI-plus-repetitive bundle.

Internal-model compensation for periodic disturbances over selected harmonics. Targets: grid-tied inverters, precision motion with periodic trajectories, rotating-machinery ripple. Depends on #2 (delay line + robustness filter).

- Interface: `design::synthesize_repetitive(plant, fundamental_frequency, harmonic_set, robustness_filter)` → `RepetitiveResult` / `RepetitiveArtifacts` + baseline-plus-repetitive runtime.
- Reference: S. Hara et al., "Repetitive Control System," IEEE TAC, 1988. https://doi.org/10.1109/9.1274
- Acceptance: harmonic-rejection benchmarks; stability margins with robustness filter; integration with LQI/LQGI.

### ☑ Input shaping (command prefilter)

**Built (2026-06):** `controllers/input_shaper.hpp` (`test_input_shaper`, in the umbrella, embeddable + freestanding-clean). `design::synthesize_input_shaper(fn_hz, zeta, Ts, ShaperType, ei_tol)` → `InputShaperResult` (normalized impulse amplitudes + integer sample delays) for `ShaperType::{ZV, ZVD, ZVDD, EI}` — ZV `[1,K]`, ZVD `[1,2K,K²]`, ZVDD `[1,3K,3K²,K³]` (exact for any ζ), EI tolerable-vibration form. Runtime `InputShaper<MaxDelay,T>` (ring-buffer impulse convolver, pass-through if the buffer is too small for the mode) and `InputShaperBank<NAxes,MaxDelay,T>` for multi-axis. Unity DC gain (amplitudes sum to 1 → steady command unchanged), constexpr, `.as<U>()`. Verified: coefficient normalization/delays, residual-vibration attenuation >50× on a 2nd-order mode, ZVD beating ZV under 15% detuning, undersized-buffer pass-through, multi-axis bank. Pure feedforward (no estimator); pairs with motion #20.
- Remaining (follow-up): `synthesize_input_shaper_from_modes(mode_list, …)` (convolve per-mode shapers for multi-mode plants) — single-mode shaping covers the headline use.
- References: Singer & Seering, "Preshaping Command Inputs to Reduce System Vibration," ASME JDSMC, 1990, https://doi.org/10.1115/1.2894142; Singhose et al., ASME JMD, 1994, https://doi.org/10.1115/1.2919428

### ⊘ Online PID tuning (relay + IFT)

Online-first PID/PI tuning that runs in the closed loop without requiring time-series export. The relay autotuner is the lightweight commissioning fallback for any SISO loop (model-free, deterministic, ISR-safe); IFT is the longer-running model-free improvement path. Pairs with the full identification pipeline in #3 — relay is the "no toolchain, no laptop" path, while #3 is the rich workflow when data export is available.

- **Relay autotuner ☑** (`controllers/relay_autotune.hpp`). Åström-Hägglund relay-feedback experiment driver: hysteresis relay in place of the controller induces a sustained limit cycle, then `(Kᵤ, Tᵤ)` are extracted from the limit-cycle period and amplitude. `design::synthesize_relay_autotune(config) → RelayAutotuneResult` plus `RelayAutotuner<T>` runtime exposing `step(y) → {u, status, Ku, Tu}` and a small lifecycle state machine (`Idle → Warmup → Measuring → Done | Failed`). Output `(Kᵤ, Tᵤ)` plugs into the existing tuning rules in `pid_design.hpp` — the autotuner is deliberately orthogonal to the tuning-rule choice. Default recommendation is `design::tyreus_luyben` (gentler than the original Ziegler-Nichols and the modern process-control standard); `ziegler_nichols` retained for textbook compatibility. Hysteresis-corrected gain: `Kᵤ = 4d / (π·√(a² − ε²))`. Acceptance tested on a discretized `1/(s+1)³` plant; describing-function error on `Kᵤ` is inherent to the method (~15–25% on low-order plants — first-harmonic approximation), `Tᵤ` is accurate to a few percent.
- **Biased / asymmetric relay + AMIGO tuning ☐.** Tightens the relay-test output beyond `(Kᵤ, Tᵤ)`: run the relay with asymmetric magnitudes `+d_pos` vs `−d_neg` (or equivalent setpoint bias) so the limit cycle is asymmetric, then track average `u` and average `y` over the measurement window to recover the static gain `Kₛ ≈ ū / ȳ`. Returns `(Kᵤ, Tᵤ, Kₛ)`. Pairs with `design::amigo_kappa_tau(Ku, Tu, Ks, Ts)` — Åström-Hägglund's optimization-derived rule (constrained to a target maximum sensitivity, typically Ms ≤ 1.4) that substantially out-performs ZN / Tyreus-Luyben across the FOPDT process family. Scope: ~30-line extension to `RelayAutotuner` (extra `bias` config field, running `u` and `y` averages over the measurement window) plus the AMIGO formula in `pid_design.hpp`. References: K. J. Åström & T. Hägglund, "Revisiting the Ziegler-Nichols step response method for PID control," Journal of Process Control 14(6), 2004, https://doi.org/10.1016/j.jprocont.2004.01.002; K. J. Åström & T. Hägglund, *Advanced PID Control*, ISA, 2006, Chapters 6 & 8.
- **IFT (iterative feedback tuning) ☐.** Gradient-based model-free tuning that perturbs the closed-loop gains and estimates the gradient of a tracking cost from two or three experiments per iteration. `design::synthesize_ift(reference, cost_weights, learning_rate)` → `IFTResult` / `IFTRuntime` with bumpless transfer between iterations.
- **Safety policy (shared).** Both runtimes must enforce: command-amplitude clamps, slew-rate limits, output saturation passthrough, bumpless transfer to/from the active controller, and rollback to the pre-tuning gains on degraded measurement / timeout.
- References: K. J. Åström & T. Hägglund, "Automatic tuning of simple regulators with specifications on phase and amplitude margins," Automatica 20(5), 1984, https://doi.org/10.1016/0005-1098(84)90014-1; H. Hjalmarsson, M. Gevers, S. Gunnarsson, O. Lequin, "Iterative feedback tuning: theory and applications," IEEE Control Systems Magazine 18(4), 1998, https://doi.org/10.1109/37.710876
- Acceptance (remaining): IFT convergence with bounded excitation; safety-policy tests (clamps, rate limits, bumpless transfer, rollback) for both runtimes.

### ☑ Extremum-seeking control (ESC)

**Built (2026-06):** `controllers/esc.hpp` (`test_esc`, in the umbrella, embeddable + freestanding-clean). Classic perturbation-based ESC: `design::synthesize_esc(a, dither_freq, gain, Ts, Maximize|Minimize, hp_cutoff, lp_cutoff, u_init, u_min, u_max)` → `ESCConfig`/`ESCResult` + `ExtremumSeekingController<T>` runtime, plus `synthesize_esc_mppt(...)` convenience. Dither → high-pass (DC removal) → demodulate (∝ ∂J/∂u·a/2) → integrate (`û̇ = ±k·LPF(ξ)`), model-free, allocation-free, constexpr, `.as<U>()`. Safety: `step(J, measurement_valid)` **freezes the integrator** on a flagged-bad measurement (the "rollback/freeze on degraded measurement" behavior), and an optional û clamp band. Verified: climbs an unknown quadratic max / descends a min, **tracks a drifting optimum** (online, not one-shot), freeze-holds û under garbage input, MPPT clamp pins û at the band edge, float, constexpr.
- Reference: Y. Tan et al., "Extremum seeking control for discrete-time systems," IEEE TAC, 2002, https://doi.org/10.1109/9.983370; Ariyur & Krstić, "Real-Time Optimization by Extremum-Seeking Control," Wiley 2003.

- Interface: `design::synthesize_esc(objective_signal, perturbation_policy, safety_limits)`, `..._mppt(...)` → `ESCResult` / `ESCRuntime` / `ESCSafetyStatus`.
- Reference: Y. Tan et al., "Extremum seeking control for discrete-time systems," IEEE TAC, 2002. https://doi.org/10.1109/9.983370
- Acceptance: MPPT over irradiance/temperature profiles; objective convergence with bounded perturbation; rollback/freeze on degraded measurements.

### ◐ Harmonic detection & suppression (anti-chatter)

Online dominant-harmonic detection and suppression via notch/PR/repetitive elements. Targets: lathe-turning chatter, spindle-tool resonance in CNC, gearbox tonal vibration. Depends on #2 (notch) and harmonic estimation. Includes the spectral primitives (windowed RFFT / Goertzel) and a harmonic tracker (`estimation/harmonic_estimation.hpp` is currently a placeholder).

- **Spectral primitive ☑** (`filters/spectral.hpp`, `test_spectral.cpp`, in the umbrella; embeddable + freestanding-clean). Deliberately *basic* — controls layer, not a DSP library. `Goertzel<T>` — a generalized single-bin DFT (arbitrary frequency) via the two-state recurrence (no buffer, O(1) memory): streaming `push(x)`, then `amplitude()`/`magnitude()`/`power()`/`phase()`, auto-restarting per block; exact for coherent sampling. `HarmonicAnalyzer<K,T>` — a Goertzel bank over a fundamental + K−1 harmonics giving per-harmonic `amplitude`/`phase`, `thd()`, fundamental `rms()`, and `total_harmonic_rms()`. Constexpr, allocation-free, float/double. Scope intentionally lean (rectangular/coherent only — no window zoo, no FFT; choose N for coherent sampling). Targets: grid V/I THD, motor torque-ripple. (`SOGI`/`MSTOGI` and `PR`/resonant controllers — the related single-frequency PE primitives — are likewise already built/tested.) **`SogiFll<T>`** (`filters/sogi.hpp`, `test_sogi_fll`): a SOGI with a frequency-locked loop — self-tuning, streaming single-tone tracker (`ω̇ = −Γ·ω·ε·qv′/(v′²+qv′²)`) that *finds* and follows the dominant frequency (no sweep, no prior knowledge) and reports locked frequency/amplitude/phase. The online/adaptive counterpart to the offline Goertzel sweep: use it for grid-frequency sync or to re-tune an input shaper / notch to a *drifting* resonance live. Verified: locks from an offset guess, tracks a frequency step, locks through broadband noise, amplitude recovery, band-clamp, constexpr.
- **Suppression bank ☑** (`controllers/harmonic_suppression.hpp`, `test_harmonic_suppression.cpp`, in the umbrella; embeddable + freestanding-clean). `design::synthesize_harmonic_suppressor(Kp, Ki_fund, w_fund, wc, Ts, harmonics)` → `HarmonicSuppressorResult<N,T>` (one `PRResult` per harmonic via `pr_harmonics`, with spec validation: positive Ts/frequency, no DC target, every resonator strictly below Nyquist), plus the `HarmonicSuppressor<N,T>` runtime — a parallel bank of `PRController` resonators summed on the loop error, satisfying the `(r,y)` protocol, with `set_fundamental()` for grid-frequency adaptation (preserves harmonic ratios). Selective/sparse counterpart to the repetitive comb (#5): target a chosen `{1,5,7,11}` rather than every harmonic of a period. Verified: multi-harmonic reference tracked to ~0 steady-state error, Nyquist/DC rejection, `as<U>()` precision conversion, constexpr.
- Remaining: host-side `analysis::detect_dominant_harmonics(signal_window, sample_rate)` (auto-pick the harmonic set to suppress — wire `HarmonicAnalyzer` output into the synthesizer); turning/milling chatter front-end `design::synthesize_chatter_suppressor_turning(machine_model, spindle_speed, sensor_config)`; adaptive online retune.
- References: Altintas & Budak, "Analytical Prediction of Stability Lobes in Milling," CIRP Annals, 1995, https://doi.org/10.1016/S0007-8506(07)62342-7; Mojiri & Bakhshai, "An Adaptive Notch Filter for Frequency Estimation," IEEE TAC, 2004, https://doi.org/10.1109/TAC.2003.822862; Goertzel, Amer. Math. Monthly, 1958, https://doi.org/10.2307/2308968
- Acceptance: detection latency / false-positive benchmarks; measured harmonic-amplitude reduction; closed-loop stability + actuator effort with adaptive updates.

### ⊘ LPV gain-scheduled LQG/LQI

Gain scheduling over an operating-point grid. Targets: UAV dynamics vs airspeed/altitude, vehicle lateral dynamics vs speed, manipulators with configuration-dependent models.

- Workflow: local linear models over the grid → local controller/estimator gains → scheduled runtime bundle with interpolation. Artifacts: schedule map, gain tables, local closed-loop analysis.
- References: Rugh & Shamma, "Research on Gain Scheduling," Automatica, 2000, https://doi.org/10.1016/S0005-1098(00)00058-3; Apkarian & Gahinet, IEEE TAC, 1995, https://doi.org/10.1109/9.384219
- Acceptance: continuity across transitions; stability over the certified envelope.

### ☑ Super-twisting SMC

**Built (2026-06):** second-order **super-twisting** algorithm — `design::synthesize_stsmc(L, Ts, λ, k_lin, ε, gain_margin)` (Levant/Moreno gains `k₁=1.5√L`, `k₂=1.1L`) + `design::stsmc(...)` direct-gain factory → `STSMCResult` + `SuperTwistingController` runtime (`controllers/smc.hpp`, `test_stsmc`). Continuous control `u = −k₁(|s|^½sign s + k_lin·s) + v`, `v̇ = −k₂(sign s + 3k_lin|s|^½sign s + 2k_lin²s)`: classic STA at `k_lin=0`, **generalized STA** (linear damping for noisy/under-modelled actuators) at `k_lin>0`, optional boundary layer `ε`. Canonical `control(s)` plus an `(r,y)` surface-builder convenience. Verified: finite-time rejection of a Lipschitz disturbance into an O(Ts) band, continuity vs first-order sign() chattering (the headline test), GSTA convergence, constexpr. Explicit-Euler integral; documented that noisy `s` is the real-world chattering source (filter it / supply `s`).

**Original (first-order baseline):** first-order SMC was already built & tested (`u = −(k/b0)·sign(s)` with boundary layer); #11 added the chattering-free second-order algorithm on top.

Second-order sliding-mode runtime with design-time gain synthesis. Targets: electromechanical drives with bounded matched disturbances, hydraulic/pneumatic actuators, converter current loops.

- Interface: `design::synthesize_stsmc(plant_or_surface, disturbance_bound, bandwidth_targets)` → `STSMCResult` / `STSMCArtifacts` + runtime with optional boundary layer.
- References: Moreno & Osorio, "Strict Lyapunov Functions for the Super-Twisting Algorithm," IEEE TAC, 2012, https://doi.org/10.1109/TAC.2012.2186179; Levant, Int. J. Control, 2003, https://doi.org/10.1080/0020717031000099029
- Acceptance: chattering metrics vs existing SMC; disturbance rejection with bounded effort.

### ⊘ H-infinity output feedback

Output-feedback synthesis for weighted generalized plants. Targets: flexible structures with modal uncertainty, flight control with structured uncertainty, active suspension/vibration isolation.

- Interface: `design::synthesize_hinf(augmented_plant, weighting_filters, gamma_search_bounds)` → `HInfResult` / `HInfArtifacts` + S/T analysis models. Workflow: coupled Riccati solves over gamma; select feasible controller.
- References: Doyle et al., "State-Space Solutions to Standard H2 and H-infinity Control Problems," IEEE TAC, 1989, https://doi.org/10.1109/9.29425; Glover & Doyle, Systems & Control Letters, 1988, https://doi.org/10.1016/0167-6911(88)90055-2
- Acceptance: weighted robust-stability/performance; regression vs known references.
- Placement: `toolbox.hpp` (resolved — gamma search allocates / iterates).

### ⊘ Constrained MPC

Finite-horizon constrained control with a deterministic runtime iteration budget. Targets: multivariable process control with limits, AV trajectory tracking with actuator limits, power electronics with current/voltage bounds.

- Interface: `design::synthesize_mpc(plant, horizon, weights, constraints, solver_budget)` → `MPCResult` / `MPCArtifacts` + fixed-iteration warm-start runtime.
- References: García, Prett & Morari, "Model Predictive Control: Theory and Practice — A Survey," Automatica, 1989, https://doi.org/10.1016/0005-1098(89)90002-2; Qin & Badgwell, Control Engineering Practice, 2003, https://doi.org/10.1016/S0967-0661(02)00186-7
- Acceptance: deterministic per-tick runtime bound; constraint-satisfaction regression; tracking/regulation tests.
- Placement: `toolbox.hpp` (resolved — QP solver allocates / iterates).

### ☐ Moving horizon estimation (MHE)

Optimization-based constrained estimator; toolbox-only (allocates/solver). The estimation counterpart to MPC.

- Reference: Rao, Rawlings & Mayne, "Constrained State Estimation for Nonlinear Discrete-Time Systems," IEEE TAC, 2003. https://doi.org/10.1109/TAC.2003.812777
- Acceptance: matches Kalman on linear/unconstrained problems; respects state constraints on nonlinear references.

### ☑ Motion planning / trajectory generation

**Re-baseline (2026-06):** built & tested — all three single-segment generators **plus the multi-axis coordination bank** in `controllers/trajectory.hpp` (`test_trajectory.cpp`, 31 cases incl. a 500-case trapezoidal + 400-case S-curve fuzz; `examples/example_trajectory_gallery.cpp` shows the per-generator grid, `examples/example_coordinated_joints.cpp` the synchronized excavator move). Not built — **multi-waypoint splines** (below) and **Cartesian/task-space moves** (need the kinematics solvers — IK + Jacobian — then optionally TOPP for joint-limit-optimal path timing). The single-segment scope went *beyond* the original plan: trapezoidal and S-curve both take **arbitrary boundary velocities `(Xi, Vi, Xf, Vf)`** and **asymmetric accel/decel limits** (`max_acceleration` bounds the approach-to-cruise ramp, `max_deceleration` the approach-to-target ramp), not just `(distance, v_max, a_max)`. The originally-listed **online bang-bang generator was dropped** — it is sliding-mode control (chattery acceleration), the wrong tool for a smooth feedforward *reference*; everything is precomputed.

Point-to-point trajectory generation for actuators and robot axes — the feedforward reference the controllers above track. Two complementary families, because (in the user's words) sometimes you want **time-optimal**, and sometimes you want **derivative-optimal over a fixed time**. Fits the three-tier pattern cleanly: a constexpr `design::` stage solves the boundary-value problem or segment times (with a `success` flag), and an allocation-free runtime evaluates `step(dt)` / `eval(t) → {position, velocity, acceleration, jerk}`. Embeddable. Generalizes the pure-feedforward input shaper (#6), which shapes an existing command; this *generates* the command. Shared types: `TrajectoryLimits{max_velocity, max_acceleration, max_deceleration, max_jerk}` (with `valid()` / `valid_jerk_limited()`), `TrajectoryState{position, velocity, acceleration, jerk}`, `TrajectoryBoundary`.

**Family 1 — fixed-time, derivative-optimal (arbitrary boundary conditions). ☑ built.** Solve a polynomial boundary-value problem over a *fixed* duration `T` matching boundary conditions on position and its derivatives at both ends:

- `design::synthesize_poly_trajectory<Order>(bc_start, bc_end, T)` → `PolyTrajectory<Order>` (coefficients + `success`). Cubic (match p, v), quintic (match p, v, a), septic/7th (match p, v, a, j). Solved as a small linear system via the existing `mat::solve` (the Vandermonde-style BVP), mirroring how `design::steinhart_hart` solves its fit. Runtime `PolynomialTrajectory<Order, T>` evaluates with a Horner-with-derivatives (synthetic-division) recurrence.
- Derivative-optimality falls out of the boundary conditions: a quintic with zero velocity/acceleration BCs is exactly the **minimum-jerk** profile over `T` (Flash–Hogan); cubic with zero-velocity BCs is **minimum-acceleration**; septic with zero jerk BCs is **minimum-snap** (Mellinger–Kumar, quadrotors). Tier-2 aliases `design::min_jerk(p0, pT, T)`, `min_accel(...)`, `min_snap(...)` wrap the general BVP with the appropriate zeroed BCs.

**Family 2 — constraint-limited, time-optimal. ☑ built.** Minimize duration subject to (asymmetric) kinematic limits, arbitrary `Vi`/`Vf`:

- `design::synthesize_trapezoidal(Xi, Vi, Xf, Vf, limits)` (+ rest-to-rest `(Xi, Xf, limits)` overload) → `TrapezoidalProfile` with segment times + `success` (accel/cruise/decel; degrades to triangular when cruise vanishes). Piecewise-constant acceleration. Accepts over-speed/wrong-direction `Vi` ("handbrake"). ODrive's symmetric rest-to-rest `planTrapezoidal` is the special case. Runtime `TrapezoidalTrajectory<T>`.
- `design::synthesize_scurve(Xi, Vi, Xf, Vf, limits)` (+ rest-to-rest overload) → `ScurveProfile` (≤7 constant-jerk segments). Bounded jerk ⇒ continuous acceleration (C²), the standard for smooth machine motion. The junction (peak) velocity is found by bisection on region displacement (a sign-change bracket — robust without closed-form sign logic); converges to the trapezoidal as `j_max → ∞`. Runtime `ScurveTrajectory<T>`.
- Region displacement uses the exact identity `Tr·(v_start + v_end)/2` (the per-region velocity profile is point-symmetric), so distance bookkeeping is identical across both Family-2 generators.

**Multi-axis coordination. ☑ built.** `TrajectoryBank<NAxes, Trajectory>` (homogeneous per-axis runtime — `TrapezoidalTrajectory` / `ScurveTrajectory` / `PolynomialTrajectory`). Plans each axis at its own min-time, takes `T_sync = max_i duration_i`, and **time-scales** every axis to it: axis `i` is evaluated at `t·kᵢ` (`kᵢ = duration_i / T_sync`) with derivatives scaled `·kᵢ`/`·kᵢ²`/`·kᵢ³`. Per-axis limits stay satisfied (time-scaling only shrinks v/a/j), all axes land together — the joint-space **PTP / "MoveJ"** move. `eval(t)` / `step(dt)` return a `wet::array<TrajectoryState, NAxes>`. **This is *joint-space* only — it does NOT preserve the Cartesian tool path** (independent per-axis profiles synced in *time* bow the path, even on a gantry). Use it for free-space moves; for a straight tool path use the Cartesian move below.

**Cartesian / task-space (LIN) move. ☑ built (global-K version).** `CartesianMove<NJoints, PathFn, IkFn, T>` (`controllers/cartesian_move.hpp`, `make_cartesian_move` factory, `test_cartesian_move.cpp`; `LinearPath` helper in `kinematics/pose.hpp`). **Path-preserving by construction:** one shared scalar `s(t)` (an S-curve along the path) parameterizes a fixed geometry `p(s)`, every joint is `qᵢ(s)=IK(p(s))` — re-timing only changes speed *along* the path. Limit handling is a single **global** time-scale `K = max(1, max|q̇ᵢ|/v_maxᵢ, √(max|q̈ᵢ|/a_maxᵢ))` found by a finite IK sweep (no Jacobian — joint sensitivities are finite-differenced; runtime evaluates `IK(p(s(t/K)))` each tick). Generic over the path and IK callables (works with any motion map; a future serial arm). Paced by the single worst point on the path (crawls through a singularity skim; a narrow rate-peak can slip a coarse sweep → raise `samples`). The **time-optimal pointwise** version is TOPP — see below.

**Time-optimal path parameterization (TOPP). ☑ built (2026-06).** `ToppMove<NJoints, NGrid, PathFn, IkFn, T>` (`controllers/topp.hpp`, `make_topp_move<NGrid>` factory, `test_topp.cpp`; in the umbrella, embedded + freestanding-clean). The pointwise-minimum-time counterpart to the global-K `CartesianMove`: same path-preserving `qᵢ(s)=IK(p(s))` construction and the same callable interface, but instead of one global slow factor it drives *every* point of the path to its own velocity/acceleration limit. Reparameterizes by squared path speed `x(s)=ṡ²` (so `dx/ds=2s̈` and the per-joint velocity/acceleration limits are linear in `(x,s̈)` at each `s`), builds a velocity MVC and an acceleration MVC (per-point monotone bisection on feasibility of the box-constraint `s̈`-interval), then a forward (max-accel) / backward (max-decel) pass on a uniform `NGrid` grid yields the minimum-time `x(s)` honouring boundary speeds `(ṡ₀, ṡ_f)`; segment times integrate exactly under the constant-`s̈` model `Δt=2Δs/(ṡᵢ+ṡᵢ₊₁)`. Allocation-free (fixed `NGrid` arrays) and constexpr; the solve runs in the constructor, the runtime samples `q(s(t))` by the chain rule. Verified: matches the analytic trapezoid/triangle time on a straight 1-joint path; respects joint vel/accel limits along straight and curved (arc, `q''≠0`) paths; strictly faster than the global-K `CartesianMove` on the same path; honours nonzero start/end speeds. Build note: surfaced (and fixed in both `topp.hpp` and `cartesian_move.hpp`) an endpoint finite-difference bug — clamping one side of the central-difference stencil while still dividing by the full `2h` halved `q'` at the path ends, which *doubled* the acceleration bound there; the stencil centre is now nudged inward while position is still read at the true `s`. **Still ☐ — pointwise jerk-limited TOPP** (the present version is acceleration-bounded; bang-bang in `s̈`).

**Multi-waypoint splines. ☑ built (2026-06).** `controllers/trajectory.hpp` (`design::SplineProfile<NPts, Order, T>` + `design::synthesize_spline<NPts, Order>` + Tier-2 `design::cubic_spline` / `design::quintic_spline`, runtime `SplineTrajectory<NPts, Order, T>`; `test_spline.cpp`). Interpolates a waypoint list `(times[], points[])` with one degree-`Order` polynomial per interval, joined by one **global linear system** (interpolation at every knot + derivative-continuity `1…Order−1` at the interior knots + clamped end conditions) solved with `mat::solve` — the multi-segment generalization of `synthesize_poly_trajectory` (which *is* the `NPts == 2` single-segment case, verified by a reduction test). **Cubic (Order 3) ⇒ C²** (continuous acceleration); **quintic (Order 5) ⇒ C⁴** (continuous jerk *and* snap — the "C³-and-better" smooth option). Clamped boundary conditions: end velocity for cubic, end velocity + acceleration for quintic (default zero ⇒ rest-to-rest). Allocation-free (fixed `NPts`/`Order`), constexpr-constructible, `.as<U>()`; the runtime drops into a `TrajectoryBank`. Verified: exact waypoint interpolation; exact derivative continuity at interior knots (one-sided segment limits, to ~1e-12); clamped BCs honoured; single-segment ≡ the BVP polynomial; constexpr + float rebind; non-increasing times rejected. Pairs with TOPP (spline gives the geometric path, TOPP times it).

**References:** L. Biagiotti & C. Melchiorri, "Trajectory Planning for Automatic Machines and Robots," Springer, 2008 (polynomial, trapezoidal, double-S — the canonical treatment); T. Flash & N. Hogan, "The coordination of arm movements: an experimentally confirmed mathematical model," J. Neurosci. 5(7), 1985, https://doi.org/10.1523/JNEUROSCI.05-07-01688.1985 (minimum jerk); D. Mellinger & V. Kumar, "Minimum snap trajectory generation and control for quadrotors," ICRA 2011, https://doi.org/10.1109/ICRA.2011.5980409; R. Béarée, "FIR filter-based online jerk-constrained trajectory generation" (the asymmetric trapezoidal / double-S structure; also the basis for ODrive's `planTrapezoidal`); S. Macfarlane & E. A. Croft, "Jerk-bounded manipulator trajectory planning," IEEE T-RA 19(1), 2003, https://doi.org/10.1109/TRA.2002.807548.

**Acceptance:**

- ☑ Polynomial trajectories satisfy all boundary conditions exactly at both endpoints; the synthesized quintic with zeroed v/a BCs matches the closed-form minimum-jerk profile to numerical tolerance; the BVP solve reports `success=false` on degenerate `T`.
- ☑ Trapezoidal and S-curve profiles respect `v_max`/`a_max`/`d_max`/`j_max`, are continuous in the claimed derivatives (S-curve: jerk-bounded ⇒ continuous acceleration), and report the correct minimum time; triangular/short-move/handbrake degeneracies handled (verified by fuzz).
- ☑ Multi-axis bank arrives synchronized (all axes reach their endpoints at the same instant; time-scaled axes stay within their own limits — verified).
- ☑ All runtimes constexpr-constructible and allocation-free; `make embedded-check` green.

**Decision items (resolved):** single `trajectory.hpp` (not a `motion/` directory); the BVP coefficient solve and all segment-time solves run at `design::` time so the runtime stores only coefficients / segment tables; multi-segment profiles use a fixed-`N` segment array (`ScurveProfile` holds `array<…, 7>`) to stay allocation-free.

### ☑ Stewart platform kinematics (6-DOF parallel manipulator)

**Built (2026-06):** `kinematics/stewart.hpp` + `test_stewart.cpp` (9 cases, incl. a 200-sample workspace round-trip fuzz), in the umbrella, embedded + freestanding-clean. `StewartGeometry<T>` (six base + six platform anchors, stroke window, home height) → `design::synthesize_stewart` / the Tier-2 `design::stewart_symmetric(base_radius, platform_radius, base_half_angle, platform_half_angle, home_height, stroke_min, stroke_max)` builder for the symmetric 6-6 hexagon → `StewartConfig<T>` (+ `.as<U>()`). Runtime `StewartPlatform<T>` exposes `inverse(pose) → {lengths, reachable}` (closed-form `Lᵢ = ‖t + R·pᵢ − bᵢ‖`), `jacobian(pose)` (rows `ûᵢᵀ·[I, −[R·pᵢ]ₓ]`, finite-difference-verified), and `forward(lengths, guess, iters, tol) → {Pose, converged, residual}` (Newton–Raphson on the 6×6 Jacobian via `mat::solve`, warm-started; quadratic to machine precision on a non-singular Jacobian, graceful `converged=false` on a singular one). Pose stored/composed as `(quaternion, translation)`; the Newton orientation step builds the incremental quaternion `exp(½·δθ)` from the rotation vector directly. Two notes from the build: a *similar* base/platform polygon (equal half-angles) makes the neutral pose a Stewart singularity (rank-deficient Jacobian) — the symmetric builder takes distinct half-angles; and the Newton orientation update exposed (and motivated fixing) a latent `Quaternion/DCM::from_axis_angle` bug — its `eps` guard compared the *squared* axis norm to a linear tolerance, silently rejecting short rotation vectors (now `eps²`, a true zero-axis guard; regression test in `test_rotation.cpp`).

Closed-form **inverse** and iterative **forward** kinematics for the Gough–Stewart platform — the 6-leg parallel manipulator behind motion-cueing rigs (flight/driving simulators), hexapod fixtures, and precision pointing/isolation stages. Given a desired platform pose, the inverse map yields the six actuator commands the controllers above drive; it is the geometric layer that turns a 6-DOF motion-cueing trajectory (#20, #6) into per-leg setpoints. Fits the three-tier pattern: a constexpr `design::` stage validates the geometry and precomputes the fixed base/platform anchor tables, and an allocation-free runtime evaluates the per-tick kinematics. Embeddable.

**Shared geometry foundation (applies to #22–#24).** All three kinematics items build on the existing constexpr geometry types in `utility/geometry.hpp` — `Quaternion<T>`, `DCM<T>`, `Euler<T,Order>`, `Vec3<T>` (on `wet::Matrix`/`Mat3`/`Mat4`) — so no new rotation math is needed; `mat::solve` backs the Jacobian steps. **Pose is represented as `Pose{ Translation3<T> translation, Quaternion<T> orientation }` throughout — translation-3-vector + unit-quaternion, *not* a 4×4 `Transform4`** (a new `kinematics/pose.hpp`). This is both the public interchange type *and* the internal accumulation type: composing the FK chain as `(q, t)` pairs — `q = q₁⊗q₂`, `t = t₁ + q₁·rotate(t₂)` — is ~half the FLOPs of a `Mat4×Mat4` per joint (≈31 vs 64 multiplies), carries 7 scalars instead of 16, and the quaternion renormalizes cheaply where a 4×4 rotation block drifts from orthonormal over the chain. `Translation3<T>` is a thin `Vec3` wrapper with domain-named conveniences (`distance()`, `norm()`). `Transform4<T>` is retained only as an **interop/export convenience** (textbook DH matrix, point-cloud transforms) via the existing `to_quaternion_translation()` / `from_quaternion_translation()` — never the working representation. (Open: whether `geometry.hpp` is promoted out of `utility/` now that it is a first-class robotics dependency.)

**Geometry / configuration.** A `StewartGeometry<T>` describes the rig: six base anchor points `bᵢ` (fixed frame) and six platform anchor points `pᵢ` (moving frame), plus actuator stroke limits `[Lmin, Lmax]` and a home height. `design::synthesize_stewart(geometry)` → `StewartConfig` + `success` validates non-degenerate anchors, reachable home pose, and consistent leg count. A Tier-2 convenience `design::stewart_symmetric(base_radius, platform_radius, base_angle, platform_angle, ...)` builds the common symmetric 6-3 / 6-6 hexagon layouts (anchors on two alternating-spaced rings) so users do not hand-enter twelve coordinate triples.

**Inverse kinematics (closed-form, the hot path).** Pose is `(translation t ∈ ℝ³, orientation R ∈ SO(3))`, with `R` from roll/pitch/yaw (or a quaternion). Each leg length is the closed-form
`Lᵢ = ‖ t + R·pᵢ − bᵢ ‖`, with the unit leg vector `ûᵢ = (t + R·pᵢ − bᵢ)/Lᵢ` available for the actuator Jacobian. `StewartPlatform<T>::inverse(pose) → {array<T,6> lengths, bool reachable}` — pure, allocation-free, exact, runs every control tick to convert a commanded pose into six leg-length (or leg-velocity, via the Jacobian) setpoints; `reachable` flags any leg outside `[Lmin, Lmax]`.

**Forward kinematics (iterative).** No closed form exists for the general Gough–Stewart platform; recover pose from six measured leg lengths by Newton–Raphson on the 6×6 actuator Jacobian `J` (rows `ûᵢᵀ · [I, −[R·pᵢ]ₓ]`), reusing `mat::solve`. `StewartPlatform<T>::forward(lengths, pose_guess, iters) → {Pose, bool converged, T residual}` — fixed iteration budget (deterministic runtime), warm-started from the previous solved pose (the natural seed in a real loop). Host-side validation can sweep the workspace; on target it is the optional feedback path when leg encoders are present.

**Workspace / Jacobian helpers.** `jacobian(pose)`, leg-length-limit and singularity checks (Jacobian conditioning), and a reachable-workspace probe so a motion-cueing washout filter (#20 territory) can be clamped to the achievable envelope before commanding a pose.

- Interface: `design::synthesize_stewart(geometry)` / `design::stewart_symmetric(...)` → `StewartConfig` + `StewartPlatform<T>` runtime with `inverse(pose)`, `forward(lengths, guess)`, `jacobian(pose)`.
- References: D. Stewart, "A Platform with Six Degrees of Freedom," Proc. IMechE, 1965, https://doi.org/10.1243/PIME_PROC_1965_180_029_02; J.-P. Merlet, "Parallel Robots," 2nd ed., Springer, 2006 (forward-kinematics methods, singularity analysis); B. Dasgupta & T. S. Mruthyunjaya, "The Stewart platform manipulator: a review," Mechanism and Machine Theory 35(1), 2000, https://doi.org/10.1016/S0094-114X(99)00006-3.
- Acceptance: inverse kinematics reproduces leg lengths for known poses to numerical tolerance, and round-trips against forward kinematics (inverse∘forward = identity within Newton tolerance) across the workspace; forward Newton iteration converges from a home-pose seed and reports `converged=false` / residual on singular or out-of-reach inputs; stroke-limit and singularity flags fire at the geometric boundaries; all runtimes constexpr-constructible and allocation-free; `make embedded-check` green.

**Decision items (this section):** orientation: accept RPY Euler / quaternion / rotation matrix at the boundary for convenience, but **store and compose as quaternion** internally (see the shared geometry-foundation note); forward-kinematics iteration budget and convergence-tolerance defaults; whether the symmetric-layout builder lives as a Tier-2 `design::stewart_symmetric` or a separate `models::`-style geometry helper.

### ◐ Serial N-DOF manipulator kinematics (rotary-joint arm, N ≤ 6)

**Built (2026-06): the N-generic numerical foundation.** `kinematics/serial_arm.hpp` + `test_serial_arm.cpp` (9 cases incl. a 50-sample FK cross-check vs an independent DH 4×4 product, a finite-difference Jacobian check, and a 150-sample IK workspace round-trip), in the umbrella, embedded + freestanding-clean. `DhJoint<T>` (standard/distal DH `a, α, d, θ_offset` + angle limits `[q_min, q_max]`) → `DhChain<N,T>` → `design::synthesize_serial_arm(chain)` (+ Tier-2 `design::arm_spherical_wrist(base_height, upper_arm, forearm, tool)` builder) → `SerialArmConfig<N,T>` (`success`, `spherical_wrist` flag via Pieper's criterion `a₅=a₆=0, d₅=0`, `.as<U>()`). Runtime `SerialArm<N,T>`: `forward(q) → Pose` (`A₁…A_N` chain, exact), `frames(q)` (cumulative `{0}…{N}`), `jacobian(q)` (geometric 6×N, finite-difference-verified), `inverse(target, seed, mask, iters, λ, tol) → {joints, converged, residual}` (damped-least-squares `Δq = Jᵀ(JJᵀ+λ²I)⁻¹·e` via `mat::solve`, 6×6 for every N, warm-started; orientation error as a shortest-arc rotation vector; joints clamped to limits), `manipulability(q)` (Yoshikawa `√det(JJᵀ)`, or `√det(JᵀJ)` for `N<6`) + `near_singular(q)`. An optional 6-DOF `TaskMask` (`task_full`/`task_position`) controls a chosen Cartesian subspace, so under-actuated `N<6` arms target only the reachable DOF; free `select_nearest(solutions, count, q_ref)` branch-tracking helper (wrapped joint distance). **Still ☐ — the spherical-wrist closed-form (Pieper) 8-branch enumeration** (`SerialArm<6>::inverse → {array<JointSet,8>, count}`); `spherical_wrist` already gates its future auto-selection. The DLS path already solves `N==6` numerically, so the closed form is a speed/branch-completeness refinement, not a capability gap.

**Prismatic joints + SCARA (2026-06).** `DhJoint` gained a `JointType{Revolute, Prismatic}` field (defaults Revolute — old 6-field aggregate inits are unchanged, guarded by a regression test): a prismatic joint's variable extends `d` instead of `θ`, and its Jacobian column is `[z_j; 0]`. This unlocks mixed-type chains, the headline being the **SCARA** (RRPR). A new `kinematics/scara.hpp` (+ `test_scara.cpp`, in the umbrella, embedded + freestanding-clean) makes both SCARA flavours one call: **series** `design::scara_arm(link1, link2, base_height, z_stroke, tool)` → `SerialArmConfig<4>` (shoulder + elbow + prismatic Z + wrist; solve IK against the new `task_position_yaw` mask); **parallel** `FiveBar<T>` / `design::five_bar_symmetric(base_separation, proximal, distal)` — a planar five-bar 2-DOF parallel mechanism (the parallel cousin of the delta) with closed-form inverse (circle intersection per arm), closed-form forward (intersect the two distal circles), a 2×2 velocity Jacobian, and reachability / parallel-singularity flags. Verified: prismatic FK/Jacobian (finite-difference); SCARA FK vs the closed form + position+yaw IK round-trip; five-bar forward∘inverse to machine precision across the workspace + Jacobian vs finite difference.

#### Original plan (the closed-form path remains TODO)

The series counterpart to the Stewart platform (#22): forward/inverse kinematics for an articulated revolute-joint arm — the canonical industrial robot. Where the Stewart platform is a *parallel* mechanism (easy closed-form inverse, iterative forward), the serial arm is the mirror image: **trivial closed-form forward, multi-solution inverse**. Same three-tier shape, embeddable, allocation-free; the kinematics produce joint setpoints that the per-joint trajectory generators (#20) then time-profile under velocity/acceleration limits.

**N-axis by construction (N ≤ 6).** The arm is templated `SerialArm<N,T>`: forward kinematics, the geometric Jacobian (`6×N`), and the numerical inverse solver are all naturally N-generic and carry **no extra cost** for the general case — the damped-least-squares inverse pivots on `JJᵀ`, which is `6×6` for *any* N. The single piece that is intrinsically 6-DOF is the **closed-form** Pieper path (the 3-for-position / 3-for-orientation decoupling and its 8 branches); it is a compile-time specialization enabled only when `N == 6` and the wrist is spherical, with everything else transparently falling back to the N-generic numerical solver. For `N < 6` the arm is under-actuated for a full 6-DOF pose, so the solver targets a caller-supplied **task mask** (e.g. position-only for a 3R arm, position + yaw for a 5-axis) and least-squares any unreachable DOF.

**Geometry / configuration.** A `DhChain<N,T>` of Denavit–Hartenberg parameters `(aᵢ, αᵢ, dᵢ, θ_offsetᵢ)` per joint, plus per-joint angle limits `[θminᵢ, θmaxᵢ]`. `design::synthesize_serial_arm(dh, joint_limits)` → `SerialArmConfig` + `success` validates the chain and (at `N == 6`) flags whether the wrist is **spherical** (last three axes intersect — Pieper's criterion), which selects the IK path. Tier-2 builders for common layouts (e.g. `design::arm_spherical_wrist(...)`) so users needn't hand-enter the DH table for a standard elbow-manipulator.

**Forward kinematics (closed-form, trivial, any N).** `SerialArm<N,T>::forward(joint_angles) → Pose` evaluates the homogeneous transform chain `T = A₁(θ₁)·…·A_N(θ_N)`. Pure, exact, constexpr — same cost for any geometry or axis count.

**Inverse kinematics — scoped to make it tractable.**
- **Spherical-wrist (Pieper) closed-form (`N == 6` only)** — the headline path, covering essentially every classic industrial 6-axis arm (PUMA-class). IK **decouples**: the first three joints solve the wrist-centre *position*, the last three solve *orientation*. `SerialArm<6,T>::inverse(pose) → {array<JointSet,8> solutions, uint8 count}` enumerates the up-to-**8** branches (shoulder left/right × elbow up/down × wrist flip), filtered against joint limits, with a `select_nearest(solutions, current_q)` helper for continuous branch tracking across a trajectory.
- **General numerical solver (any N ≤ 6, the N-generic path)** — for non-spherical-wrist geometries and all `N < 6` arms (no closed form — the general 6R case is the 16th-degree Raghavan–Roth problem, deliberately **out of scope** analytically), damped-least-squares Jacobian iteration `Δq = Jᵀ(JJᵀ + λ²I)⁻¹ Δx` from a seed pose, reusing `mat::solve`. `JJᵀ` is `6×6` for every N, so the solver is axis-count-agnostic. Accepts an optional **task mask** (which of the 6 Cartesian DOF to control) so under-actuated `N < 6` arms target only the reachable DOF. Fixed iteration budget (deterministic), warm-started from the current pose, with `converged` / residual reporting and singularity-robust damping.

**Jacobian / singularity helpers.** `jacobian(q)` (geometric `6×N` Jacobian, Cartesian↔joint velocity), `manipulability(q)` (`√det(JJᵀ)` / conditioning) and the standard singularity flags (wrist-aligned, elbow-extended, shoulder). Cartesian velocity/acceleration commands map to joint rates through `J`; **velocity/acceleration *limits* are not enforced here** — those belong to the per-joint trajectory generators (#20), which the kinematics feeds.

- Interface: `design::synthesize_serial_arm(dh, joint_limits)` / `design::arm_spherical_wrist(...)` → `SerialArmConfig` + `SerialArm<N,T>` runtime with `forward(q)`, `inverse(pose[, task_mask])` (closed-form when `N == 6` + spherical-wrist, numerical otherwise), `jacobian(q)`, `select_nearest(...)`.
- References: J. Denavit & R. S. Hartenberg, "A kinematic notation for lower-pair mechanisms based on matrices," ASME J. Appl. Mech., 1955; D. L. Pieper, "The Kinematics of Manipulators under Computer Control," PhD thesis, Stanford, 1968 (spherical-wrist decoupling); M. Raghavan & B. Roth, "Inverse Kinematics of the General 6R Manipulator and Related Linkages," ASME J. Mech. Des. 115(3), 1993, https://doi.org/10.1115/1.2919218 (why the general case is hard — scoped out); J. Craig, "Introduction to Robotics: Mechanics and Control," 3rd ed., 2005.
- Acceptance: forward kinematics matches the DH transform chain for known joint sets to numerical tolerance; closed-form inverse round-trips against forward (forward∘inverse = identity for each returned branch) and enumerates the correct solution count, filtered to joint limits; `select_nearest` tracks a continuous branch through a Cartesian path without flips; numerical fallback converges from a seed and reports `converged=false` / residual at singularities and unreachable targets; manipulability/singularity flags fire at the geometric boundaries; all runtimes constexpr-constructible and allocation-free; `make embedded-check` green.

**Decision items (this section):** how aggressively to scope IK (spherical-wrist closed-form only, vs always-available numerical fallback — likely both, with the closed-form path auto-selected when `N == 6` and Pieper's criterion holds); task-mask representation for under-actuated `N < 6` arms (fixed 6-bit DOF mask vs caller-supplied weight vector); orientation representation at the boundary (shared with #22 — RPY/quaternion/matrix); whether the DH convention is standard or modified (Craig) DH, and whether to expose both; how branch identity (shoulder/elbow/wrist configuration tags) is reported alongside the raw solution sets.

### ☑ Motion-system kinematic mappings (Cartesian / CoreXY / polar / delta)

**Built (2026-06):** `kinematics/motion_maps.hpp` + the shared `kinematics/pose.hpp` foundation (`Pose<T>` = `Translation3` + `Quaternion`, with compose/inverse/`transform_point`/`Transform4` interop; `Translation3<T>` thin `Vec3`). `test_kinematics.cpp` (11 cases incl. 2000-sample round-trip fuzz on each delta). All in the umbrella, embedded + freestanding clean. Naming landed slightly differently from the sketch below: `CoreXY<T>` / `PolarMap<T>` / `CartesianMap<N,T>` are plain runtime structs with static/`const` `forward`/`inverse` (no `design::synthesize_*` stage — the maps are trivial enough that a synthesis tier added nothing); the deltas are `RotaryDelta<T>` / `LinearDelta<T>` built from `RotaryDeltaGeometry` / `LinearDeltaGeometry`, returning `DeltaInverse{array<T,3> actuators, bool reachable}` / `DeltaForward{Pose, bool valid}`. Rotary-delta FK is the canonical quadratic-intersection (Trossen/marginallyclever derivation); linear-delta FK is sphere trilateration. Decision resolved: trivial maps *do* get thin runtime types for interface uniformity; rotary vs linear delta are **two** types (the geometry and the FK/IK math differ enough that one parameterized type would be muddier).

The "everything that isn't a 6-DOF manipulator" item: the small, closed-form coordinate transforms that map machine actuators to task space for the common motion architectures — 3D printers, plotters, pick-&-place gantries, polar/R-θ machines, and delta robots. Most are trivial (the point is to give them a documented, tested home so users stop reinventing `atan2`), but two — **CoreXY** and **delta** — carry real coupling worth a proper synthesis stage. Three-tier shape, embeddable, allocation-free; like #22/#23 these emit per-actuator setpoints that the trajectory generators (#20) time-profile.

**Group A — direct / linearly-coupled maps (closed-form both directions).**
- **Cartesian gantry** (`models`-style identity map): per-axis scale + offset from `utility/scaling.hpp` (#19); the "kinematics" is the identity. Included only as a named, documented passthrough so a Cartesian machine uses the *same* `forward/inverse` interface as the coupled machines.
- **CoreXY / H-bot:** the two motors jointly drive X/Y — `[Δx, Δy] = ½·[[1, 1], [1, −1]]·[Δa, Δb]` (and its exact inverse). `CoreXyMapping<T>::forward(a, b) → (x, y)` / `inverse(x, y) → (a, b)`. Linear, exact, constexpr; a real coupling but a trivial 2×2.
- **Polar / R-θ:** `inverse(x, y) → (r=√(x²+y²), θ=atan2(y, x))`, `forward(r, θ) → (r·cosθ, r·sinθ)`, with optional Z passthrough. Closed-form; the only subtlety is θ-unwrapping for continuous motion (reuse `utility/wrap`).

**Group B — delta (genuine parallel mechanism, the substantive one).**
- **Linear delta** (3 vertical rails + parallelogram arms) and **rotary delta** (3 base servos): a true parallel kinematic chain in the #22 family but 3-DOF translational. Closed-form **inverse** (each leg solved independently from the target XYZ — a sphere/circle intersection per arm), and a closed-form-ish **forward** (intersection of three spheres). `DeltaRobot<T>::inverse(x, y, z) → {array<T,3> joints, bool reachable}` / `forward(joints) → {Pose, bool valid}`. Workspace/reach checks and arm-singularity flags, mirroring #22's helpers.

- Interface: `design::synthesize_corexy(...)` / `synthesize_polar(...)` / `synthesize_delta(geometry)` → config + `success`, with `CoreXyMapping<T>` / `PolarMapping<T>` / `DeltaRobot<T>` runtimes exposing the common `forward(...)` / `inverse(...)` pair (and `reachable` where a workspace exists).
- References: I. Maslov (CoreXY), the CoreXY mechanism description, https://corexy.com/theory.html; R. Clavel, "Conception d'un robot parallèle rapide à 4 degrés de liberté," PhD thesis, EPFL, 1991 (delta robot); J.-P. Merlet, "Parallel Robots," 2nd ed., Springer, 2006 (delta inverse/forward kinematics).
- Acceptance: each mapping round-trips (`forward∘inverse = identity`) to numerical tolerance across its workspace; CoreXY matrix and its inverse reproduce known motor↔Cartesian moves exactly; polar handles the origin and θ-wrap without discontinuity; delta inverse reproduces leg solutions for known poses and forward recovers the pose, with `reachable`/`valid` flags firing at the workspace boundary; all runtimes constexpr-constructible and allocation-free; `make embedded-check` green.

**Decision items (this section):** whether the trivial Cartesian/polar maps warrant runtime types at all or are better left as documented `scaling.hpp`/`wrap` recipes (likely: thin types for interface uniformity); how much delta-robot geometry to expose (linear vs rotary delta as one parameterized type vs two); shared `Pose` / workspace-flag conventions with #22/#23.

**Layer 4 — tooling (host-only).**

### ☐ Multi-rate simulation harness

The current simulation harness (`simulation/simulate.hpp`) advances every block at a single fixed `Ts`. Real deployments are multi-rate: an ISR-level loop (current/inner-loop control, PWM, observers) runs at 8 kHz–128+ kHz, while RTOS tasks run the outer loops and supervisory logic at 1 / 10 / 100 / 1000 Hz. A faithful closed-loop simulation has to reproduce this rate hierarchy — including the inter-rate effects that single-rate sim hides: sample-and-hold across the rate boundary, the one-sample transport delay an outer task sees on inner-loop state, and aliasing of fast dynamics into slow samplers.

- **Model.** Each block runs at a rate that is an integer divisor of the fastest *discrete* (base) rate; the harness ticks at the base rate (e.g. the ISR rate) and fires each slower block on its decimation boundary. The plant is *not* one of these discrete blocks — it integrates on its own much finer clock (continuous-time approximation, e.g. 1 µs / 1 MHz sub-step, far faster than even the ISR), so fast plant dynamics and inter-sample ripple are resolved between controller ticks. The plant sub-step need not be an integer divisor of the control rates; the harness advances the plant to each control-tick boundary and only samples/holds at those boundaries.
- **Rate-boundary semantics.** Every crossing is a sample-and-hold, including controller→plant. A discrete block's output is held constant (ZOH) on the consumer's clock until that block fires again: ZOH from a slower producer to a faster consumer, and from the fastest discrete block down onto the continuous plant (the plant sees a piecewise-constant input between actuation updates, not a fresh value every integration sub-step). Faster→slower is latch/decimate (the slow sampler reads the held value at its tick). Make the outer→inner reference handoff and the inner→outer state feedback explicit so the simulated transport delay matches what the RTOS actually incurs. This — plus the controller→plant ZOH — is where most "works in sim, oscillates on hardware" gaps come from.
- **Scope / placement.** Host-only (`wet/toolbox.hpp`) — it allocates time-series like the existing `sim::` harness. The per-block runtimes being scheduled are the same allocation-free controllers/estimators from `wet/control.hpp`; the harness only orchestrates *when* each `step()` is called. No new constraint on the embeddable core.
- **Relationship to existing code.** Generalizes `simulate.hpp` (single-rate is the degenerate case: one block, base rate = block rate). Reuses `simulation/integrator.hpp` for the plant sub-step. Pairs naturally with the cascade controllers (#3) and any inner/outer split (current→speed→position).
- **Open questions.** Schedule specification (per-block divisor vs. explicit Hz with validation that each divides the base rate); whether to model task jitter / execution-time offset within a tick or assume ideal periodic firing (ideal first, jitter as a later refinement); how to expose the multi-rate trajectories for plotting when channels are sampled at different rates.
- Reference: Franklin, Powell & Workman, "Digital Control of Dynamic Systems" (3rd ed., 1998), multirate sampling chapter; M. Cimino & P. R. Pagilla, "Conditions for multirate sampled-data control," on inter-sample behavior and rate-boundary effects.
- Acceptance: a two-rate example (fast inner current loop + slow outer loop) reproduces the inner-loop transport delay and sample-and-hold seen on hardware; single-rate results match the existing `simulate.hpp` harness when all blocks share one rate.

### ◐ Power converter modulation & control (DC-DC, PFC, multilevel)

Modulator and control building blocks for the standard switching-converter topologies, not just three-phase inverters. Layered by where each piece lands: **gating/modulation** leaf utilities in `utility/modulation.hpp` (pure constexpr duty/carrier math, no plant model — they deviate from the three-tier pattern like the other `utility/` blocks); **control laws** (MPPT, PFC loops) in `controllers/` on the existing PI/PR/observer infrastructure; **averaged converter models** as Tier-2 `StateSpace`/`TransferFunction` builders for design + simulation. The common thread is *actuation*: turn a voltage/current command into switch states.

**Cross-cutting modulator primitives** (the shared layer under everything below):
- PWM carrier comparison — trailing-edge (default), leading-edge (current-mode), and dual-edge / **center-aligned** (symmetric, lower harmonic content + the natural place to sample current).
- Complementary gating + **dead-time** insertion, and dead-time *compensation* (volt-second error correction from device drop + dead-time, keyed on current sign).
- **Interleaving** — N-phase carrier offsets (360°/N) for multiphase buck / interleaved boost ripple cancellation and current sharing.

**DC-DC.**
- Duty feedforward for **buck** (`D = Vo/Vi`), **boost** (`D = 1 − Vi/Vo`), **buck-boost**; closed loop is voltage-mode or (peak/average) current-mode on the existing controllers — the topology-specific part is the modulator + slope-compensation for peak-current mode.
- **MPPT (boost for PV/solar):** Perturb-&-Observe and Incremental-Conductance trackers. Note overlap with **extremum-seeking (#8)** — ESC *is* a gradient MPPT; P&O/IncCond are the discrete domain-standard alternatives worth having explicitly. Interleaved boost for higher power.

**PFC / AC-DC (chargers, front-ends).**
- **Boost PFC** with average-current-mode control: inner current loop shapes inductor current to a rectified-sinusoid reference (multiplier = voltage-loop output × |v_in| template), outer DC-bus voltage loop. Pairs with the existing PR/PI controllers and a PLL/template generator for the line.
- **Totem-pole bridgeless PFC** (the AC→DC charger case): a high-frequency leg (GaN/SiC) runs the boost PWM while a **line-frequency leg (SCR or slow MOSFETs)** unfolds/commutates at the mains zero-crossing — needs zero-cross detection, polarity-dependent gating, and zero-crossing current-distortion handling. The control loops are standard; the topology-specific value is the commutation/gating sequencer.

**Three-phase VSI carrier/space-vector schemes** (the original inverter-PWM scope):
- **Done:** continuous **SVPWM** via min-max zero-sequence injection (`svpwm_zero_sequence`, `svm_duty_cycles`, `SvmDuties`), with the carrier ⇔ space-vector equivalence (Hava/Kerkman/Lipo).
- To add: **SPWM** (baseline, no injection), **THIPWM** (closed-form 1/6 injection, cheaper cousin of the shipped offset), the **DPWM family** (DPWMMAX/MIN, DPWM0–3, GDPWM — clamp 60° windows to cut switching loss ~33%, placement keyed to load PF), **overmodulation** (Mode I/II → six-step, pairs with the existing `is_clipped` flag), and **random/spread-spectrum PWM** (EMI flattening, low priority).

**Multilevel (NPC / ANPC).**
- **3-level NPC** (neutral-point-clamped): phase output ∈ {+Vdc/2, 0, −Vdc/2}; modulation via phase-disposition (PD) level-shifted carriers or 3-level SVM, plus **neutral-point voltage balancing** (redundant small-vector selection or carrier offset).
- **ANPC** (active NPC): clamp diodes replaced by active switches → freedom to **distribute switching/conduction loss** across devices via switch-state selection (loss balancing), on top of NPC modulation. Larger effort; do NPC first, ANPC as the loss-balancing extension.

References: R. W. Erickson & D. Maksimović, *Fundamentals of Power Electronics*, 3rd ed., 2020 (DC-DC, PFC, current-mode); D. G. Holmes & T. A. Lipo, *Pulse Width Modulation for Power Converters*, IEEE Press, 2003 (inverter PWM, multilevel carriers/SVM); A. M. Hava et al., IEEE T-PEL 14(1), 1999 (CPWM/DPWM/GDPWM); L. Ma et al. and Bruckner/Holmes on ANPC loss distribution.

Acceptance (per piece): duty feedforward matches the converter's CCM conversion ratio; center-aligned/interleaved carriers produce the expected phase relationships and ripple cancellation; MPPT converges to the array's true MPP under irradiance steps (and matches ESC on a smooth P-V curve); boost-PFC current tracks the rectified-sine template with unity displacement factor; totem-pole commutation sequences correctly across the zero-crossing; NPC neutral-point stays balanced under load; ANPC switch-state selection equalizes per-device loss. Modulator primitives constexpr + embeddable with one test TU; control laws follow the three-tier pattern.

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
