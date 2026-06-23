# Unified controller & observer concepts (SISO + MIMO)

Status: planning · Owner: controls core · Feeds roadmap #3 (Cascade + unified
concept), #18 (multi-rate sim), #25 (kinematic estimator)

The goal: one small C++20 concept (or a tiny family of them) that every
controller, filter, and observer in the library satisfies, so generic code —
`Cascade<Outer, Inner>`, the simulation harness, gain-scheduled bundles — can
hold "a controller" or "an observer" without caring which one. The blocker is
that the **shipped runtimes already disagree** on what `control()`/`step()`
take and return. This doc catalogs the disagreement and the design space before
we pick. Nothing here is decided yet.

## What exists today (the real signatures)

Pulled from the headers, not idealized:

| Block | Signature | Consumes | Returns |
|-------|-----------|----------|---------|
| `PID`, `ADRC` | `T control(T r, T y, T Ts)` | ref + meas + explicit Ts | scalar command |
| `PR` | `T control(T r, T y)` and `T control(T error)` | ref + meas, or pre-differenced error | scalar command |
| `SMC` | `T control(T r, T y, T Ts, T phi)` | ref + meas + Ts + boundary-layer param | scalar command |
| `LQR` | `ColVec<NU> control(ColVec<NX> x)` / `control(x, x_ref)` | full **state** vector | vector command |
| `LQI` | `control(ColVec<NX> x, ColVec<NY> r, ColVec<NY> y)` / `control(x_aug)` | state + ref + meas | vector command |
| `LQG` | `predict(u)` + `update(y, u)`, then `control()` / `control(x_ref)` | meas + input via embedded KF; `control()` reads estimated state | vector command |
| `LQGI` | `predict(u)` + `update(y, u)`, then `control(r, y)` / `control(x_aug)` | meas + input via embedded KF; integral state on tracking error | vector command |
| `Kalman`, `EKF` | `predict(u)` then correct; `state()` | input + meas, **two-phase** | state estimate |
| `LuenbergerObserver` | `step(ColVec<NY> y, ColVec<NU> u)` | meas + input, **one-phase** | state estimate |
| `DisturbanceObserver` | `T estimate(T y, T u)` / `T step(T x)` | meas + input | disturbance / lumped state |

Common ground: every one has `reset()`. After that they diverge on five axes.

## The five axes of disagreement

1. **Scalar `T` vs `ColVec<N, T>` (SISO vs MIMO).** The headline. PID/PR/SMC/ADRC
   are scalar; LQR/LQI/LQG are vector. A single concept must either span both or
   admit `N == 1` is just a `ColVec<1>`.
2. **What the controller consumes.** Three incompatible shapes:
   - **error-feedback** `(r, y)` — PID, PR, SMC, ADRC. Computes its own error.
   - **state-feedback** `(x)` — LQR. Caller supplies the full state (often from
     an observer).
   - **composite** — LQG. Holds its own estimator: you feed measurements to its
     `predict`/`update` and then `control()` reads the estimated state. It still
     consumes `(y, u)` — just through a separate two-phase estimator interface,
     not through `control()`. This is the in-tree proof that "a controller" is
     not always a single `f(r, y) → u` call; some are estimator-update-then-law.
3. **Explicit `Ts` vs baked-in.** PID/SMC/ADRC take `Ts` every call (supports
   runtime rate changes); PR/LQ* bake it at construction. A uniform `step()`
   forces a choice.
4. **Controller vs observer shape.** Controller: `command = f(setpoint, meas)`.
   Observer: `state = f(meas, input)`. Different in/out *roles* even when the
   tensor shapes coincide — a concept that calls both "a block with `step`"
   risks erasing the distinction the cascade actually depends on.
5. **One-phase vs two-phase estimators.** `step(y,u)` vs `predict(u)`+`correct(y)`.
   The two-phase split matters for multi-rate (#18): predict at the fast rate,
   correct only when a slow measurement lands.

## Unified `step` vs exposed phases

LQG/LQGI currently expose `predict` / `update` / `control` as three separate
calls (thin pass-throughs of the embedded KF + LQR — no deliberate design). The
question is whether the concept should require a *single* `step(r, y) → u` that
folds them. The answer is **both, layered:**

- The concept requires a unified `step(r, y) → u` (or `step(y, u)` for
  observers). For LQG that is internally `predict(u_prev) → update(y) →
  u = control(r) → cache u` — the single-rate ergonomic path. This is what
  `Cascade` and the simulation harness call.
- The two-phase `predict` / `update` pair stays as an *optional refinement* a
  block may also expose, for callers that need it. Multi-rate (#18) drives the
  fast loop with `predict` and fires `update` only when a measurement lands;
  `update`'s `bool` return gates the skip.

Two things a folded `step` must get right, and why they aren't a free merge:

- **`u` causality.** `predict` consumes the *previously applied* command, not
  this tick's. A unified `step` must cache last-`u` internally and fix the
  order `predict(u_prev) → update(y) → control`.
- **Measurement availability.** A single `step(y)` assumes a fresh `y` every
  call — false under multi-rate. The unified path is the single-rate
  convenience; it does not replace the exposed phases, it sits on top of them.

## Design tensions (no decision yet)

- **One concept or a family?** A single `Block { reset(); }` is too weak to be
  useful. Likely a small family: `Controller` (setpoint+meas → command),
  `StateFeedback` (state → command), `Observer` (meas+input → state). Question:
  do `Cascade`/sim need all three, or can adapters collapse them to one?
- **Unify SISO into MIMO, or keep both first-class?** Treating scalar as
  `ColVec<1>` makes the concept uniform but taxes the common SISO path (PID on a
  motor) with vector ceremony and may pessimize codegen. Alternative: the
  concept is parameterized on the signal type, `T` *or* `ColVec<N,T>`, and SISO
  blocks stay scalar.
- **Where does the error get computed?** If the concept is `(r, y) → u`, LQR
  doesn't fit without an adapter that turns `(r,y)` into a state estimate. If
  it's `(state) → u`, PID doesn't fit. The cascade protocol the roadmap sketched
  (`T control(T r, T y)`) is the *error-feedback* shape — it silently excludes
  LQR/LQG unless we wrap them.
- **`Ts` placement.** Pass per-`step` (uniform, supports rate changes, but
  pollutes blocks that don't need it) vs construction-time (clean call site, but
  no runtime rate change). Could be a trait: `static constexpr bool needs_ts`.
- **Adapters vs native conformance.** Cheapest path may be: keep the existing
  signatures, write thin adapter wrappers (`AsController`, `AsObserver`) that
  present the concept's shape. Costs nothing at the leaves, concentrates the
  ugliness in one place. Risk: adapter proliferation if the concept is wrong.

## Candidate shapes to evaluate

Sketches, not proposals — each is a strawman to argue against:

```cpp
// A) Single error-feedback concept, signal type as the unifier
template <class C>
concept Controller = requires(C c, typename C::signal r, typename C::signal y) {
    { c.control(r, y) } -> std::same_as<typename C::signal>;  // signal = T or ColVec<N,T>
    c.reset();
};
// LQR/LQG do NOT satisfy this without a state-estimating adapter.

// B) Two concepts split by what's consumed
template <class C> concept ErrorController = /* control(r, y) -> u */;
template <class C> concept StateController = /* control(x)    -> u */;
// Cascade<Outer, Inner> requires Outer::output == Inner::reference; both may be either kind.

// C) Observer concept, two-phase with optional collapse
template <class O>
concept Observer = requires(O o, typename O::input u, typename O::meas y) {
    o.predict(u);     // fast rate
    o.correct(y);     // slow rate (when meas available)
    { o.state() };    // -> ColVec<NX,T>
    o.reset();
};
// step(y,u) blocks get a default predict+correct shim; KF/EKF native.
```

## Open questions

- Does `Cascade` need to compose controller↔observer (estimator in the loop), or
  only controller↔controller? Answer narrows the concept count.
- MIMO `Cascade`: outer output dimension must match inner reference dimension —
  a compile-time `requires` on `NU_outer == NY_inner`. Worth encoding in the
  concept or left to the composition site?
- Do we want runtime polymorphism (type-erased `IController`) anywhere, or is
  everything static? (decisions.md D-something on allocation-free / embedded.)
- How does this interact with `StateSpace` composability in tiers.md — should a
  controller *be* expressible as a `StateSpace` for analysis, separate from its
  runtime concept conformance?

## Acceptance (when this graduates from planning)

- A concept (or family) that every shipped controller and observer satisfies —
  natively or via one documented adapter each.
- `Cascade<Outer, Inner>` compiles for at least one controller↔controller and,
  if in scope, one controller↔observer pairing, SISO and MIMO.
- No regression to the SISO call site (PID-on-a-motor stays `control(r, y, Ts)`
  ergonomically — adapter, not rewrite).

## Related docs

- [roadmap.md](roadmap.md) #3 — the Cascade + unified-concept work item.
- [tiers.md](tiers.md) — these are all Tier-3 runtime objects; the concept is
  the protocol Tier-3 shares.
- [decisions.md](decisions.md) — D2 universal types, D3 three-tier pattern,
  allocation-free / embedded constraints the concept must honor.
</content>
</invoke>
