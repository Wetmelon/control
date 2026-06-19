---
name: review-control-header
description: >
  Deep review of a single controller/filter/estimator header in inc/wet (e.g.
  controllers/*.hpp, filters/*.hpp, estimation/*.hpp). Reads the whole header and
  checks it on five axes: correctness (design + runtime + numerics), test
  coverage, library-style/API/Doxygen consistency, reuse (redundancy, the matrix
  library, concepts), and StateSpace/TF composability for simulation. Use when
  the user says "review <header>", "check <controller>", "is the ESO right", or
  invokes /review-control-header. Pair with /ponytail-review for pure complexity.
---

# Review a control header

One header, full review. Output is a severity-ranked findings list, not prose.
This is the method that caught a runtime controller delegating to a deleted
overload, an uninitialized member used in a rescale, and an output that rode past
its clamp — read the *whole* header and trace the math, don't skim.

**Bias every recommendation toward numerical robustness.** When two options are
equally simple, recommend the more numerically sound one — and when a less robust
form is in use, flag it and name the better one even if it costs a few lines:
factored/triangular solves over explicit inverses, Joseph-form covariance updates,
`hypot`/`log1p`/`expm1` over naive algebra, Kahan/pairwise summation on long
accumulations, Tustin/ZOH over forward Euler, symmetric eigen paths for symmetric
matrices. Correctness and conditioning win over brevity; never recommend the
flimsier algorithm to save code.

## Method

Work the phases in order. Read the entire header first (it's one file); don't
review from the diff alone — the bug is often in code the diff didn't touch.
Cross-read a known-good sibling (e.g. `controllers/pid.hpp`) as the style anchor.

### 1. Map the header

Most headers in this repo have two halves. Identify them:
- **`design::` half** — `constexpr` synthesis: a `XResult` struct (gains, success
  flag, `as<U>()` converter) and `x(...)` factory/synthesis functions.
- **runtime half** — the `XController`/`X` class: state members, `control(...)`,
  `reset()`, optional `back_calculate(...)`, converting constructor.

List every public method on the runtime class. For each, confirm it actually
**compiles and resolves** — a templated `control()` that delegates to an overload
that no longer exists compiles as a header and only fails when instantiated, so
it slips through. (This is exactly how the ADRC runtime was found gutted.)

### 2. Design half

- Do the synthesized gains match the cited formula/reference? Re-derive at least
  one (e.g. ESO `beta` for poles at `-wo`, Tustin biquad coefficients).
- Does `as<U>()` convert **every** field (including any newly-added member)?
- Does the converting ctor copy **every** state member? A missing one is a silent
  state-loss bug across precision conversions.
- `success`/validity: is an invalid spec rejected (returns `success=false`) rather
  than producing garbage gains?
- **Composability:** if the controller/filter is representable as an LTI system,
  a `design::` function should return a `StateSpace` or `TransferFunction` so it
  drops into the simulation/analysis tooling (Bode, `series`/`feedback`,
  `discretize`). Lead-lag/PR/biquad all do this — flag a new LTI block that only
  ships a runtime class and no `to_ss()`/`to_tf()`/builder. (The *runtime* class
  need not be built on that SS — see phase 3.)

### 3. Runtime half

- **Every state member used by a method must be initialized by every constructor.**
  An uninitialized member read in `set_*`/`control` is a real bug (harmonic
  suppressor's `w_fund_` defaulted to 1 and broke `set_fundamental`).
- **`(r, y)` protocol**: does `control(r, y, ...)` compute error and apply the law
  correctly? Sign of the feedback, setpoint weighting, reference path.
- **Anti-windup** (`back_calculate`): sign of the unwind (saturation excess bleeds
  *out* of the integrator), and consistency with the in-loop clamp behavior.
- **Bumpless transfer / tracking**: does the preload reproduce the applied command
  on re-enable, accounting for the integrator's own next-tick update?
- **Output limits**: is the *applied* command actually bounded? A dithered or
  feed-forward term added after the clamp can exceed the limit (ESC).
- `reset()` clears all accumulator/history state (and the right amount — not mode
  or config that should persist).
- **Runtime ≠ simulation SS.** The runtime may use a more optimal realization than
  the design-half state space (precomputed IIR coefficients, direct difference
  equation, scalar recurrences) — that's expected and fine. Only check the
  realization is *equivalent* to the SS/TF, not that it mirrors its structure.

### 4. Numerics & discretization

- **`Ts` handling**: `Ts` is passed per-call (not stored) by repo convention.
  Guard non-positive `Ts` — a divide (`d/Ts`) or backward step needs it. PID guards
  and holds state; flag any controller that divides by `Ts` unguarded (SMC/STSMC).
- Discretization method matches the doc (Tustin/ZOH/Euler) and the coefficients
  are normalized (e.g. divided by `a0`). Check the DC-gain / unity-gain identity.
- Nyquist / sample-rate validity: is a dither/resonant frequency required to be
  sufficiently below `fs/2` (not merely below it)?
- NaN/Inf paths: division by a zero gain (`b0`, `Kbc`, `gain`), `sqrt` of a
  possibly-negative argument, `Ts=0` in a coefficient computation.

### 5. Style, reuse & abstraction

- **API style** matches the library: descriptive long names primary with MATLAB
  short aliases where applicable; `design::` factories named like their siblings;
  `constexpr`, `[[nodiscard]]`, `reset()`, `as<U>()`, braces always, `std::numbers`
  over literal constants. Flag deviations against the sibling anchor.
- **Doxygen style** matches: `@brief` + the control law in math (`@f[...@f]` or
  `$...$`), the MATLAB equivalent, an academic reference (`@see` with DOI), and a
  worked `@code` example. Flag thin or missing doc on a public symbol.
- **Redundancy / commonization:** is this a re-implementation of something that
  already exists (a hand-rolled biquad vs `Biquad`, a private clamp vs
  `wet::clamp`, a bespoke edge detector vs `logic.hpp`)? Name the existing symbol
  to reuse, or propose factoring a shared helper if two headers grew the same code.
- **Leverage the matrix library:** state propagation, gain application, and
  covariance math should use `Matrix`/`ColVec`/`mat::` ops, not hand-indexed
  loops, where it's clearer and not a hot-path regression. If a needed operation
  is missing and would be **reusable** (e.g. a block-companion builder, a
  Householder step, a Joseph-form update), propose adding it to the matrix library
  rather than inlining a one-off.
- **Concepts:** would a concept (a scalar/floating constraint, an
  `LtiSystem`/`SisoController` shape) replace a bare `typename` to improve
  overload clarity and error messages? Note where `requires` would tighten an
  interface that's currently structurally typed.
- **Purity & `const`:** prefer pure functions and `const`-correct methods.
  `design::` synthesis should be free functions returning a result by value (not
  mutating an out-param or hidden state); runtime accessors and any method that
  doesn't change state must be `const` (and `[[nodiscard]]`). Don't fear returning
  matrices/structs by value — guaranteed copy elision / NRVO makes it free. Flag
  reference-out-params and non-`const` methods that only read.
- **Minimize class state (YAGNI):** a runtime class should store only what
  `control()`/`reset()` actually need between calls — gains and live accumulators.
  Flag members that are cached derivations of other members, config that could be
  a `control()` argument, or "might need it later" fields. The leaner the state,
  the cheaper `reset()`, conversion, and reasoning about correctness.

### 6. Verify

- Build: `make` (formats, builds, runs the doctest suite). Never bare `tup`.
- **Every function tested.** Grep the matching `tests/test_<name>.cpp`. Two failure
  modes: (a) the runtime path is untested — the suite only checks `design::` gains
  and never constructs the controller and calls `control()` (green but unproven,
  as `test_adrc.cpp` was); (b) a public function/overload has no case at all. List
  the untested symbols, and add tracking / disturbance-rejection / equivalence
  self-checks if asked.

## Output format

Be terse. One finding per bullet, `file:line` clickable, severity-grouped. State
the fix, don't explain it — the reader is a 20-year controls engineer. Skip empty
sections and any preamble.

```
**Verdict:** <one line — what's correct, count by severity>.

## Critical
- **<header>:<line>** — <what's wrong + fix, one sentence>.

## Minor
- **<header>:<line>** — <issue + fix>.

## Notes
- <style / reuse / composability — one line each>.
```

End with a one-line offer to fix criticals and add missing tests. Do not apply
fixes unless asked.
