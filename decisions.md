# wet Library — Decision Log
- [wet Library — Decision Log](#wet-library--decision-log)
  - [Foundational](#foundational)
    - [D1 — No mandatory third-party dependency; backend-agnostic core](#d1--no-mandatory-third-party-dependency-backend-agnostic-core)
    - [D2 — Plants are universal types; Tier 1 / Tier 2 layering](#d2--plants-are-universal-types-tier-1--tier-2-layering)
    - [D3 — Three-tier synthesis pattern](#d3--three-tier-synthesis-pattern)
    - [D4 — Embedded vs host placement is part of the design](#d4--embedded-vs-host-placement-is-part-of-the-design)
    - [D5 — Stay in the controls lane; defer generic plumbing to ETL](#d5--stay-in-the-controls-lane-defer-generic-plumbing-to-etl)
    - [D6 — Explicit failure signaling](#d6--explicit-failure-signaling)
  - [Architecture](#architecture)
    - [D7 — Runtime `step()` status policy → out-of-band `status()`](#d7--runtime-step-status-policy--out-of-band-status)
    - [D8 — std-replacement backend → stdlib or ETL, per-facility macros](#d8--std-replacement-backend--stdlib-or-etl-per-facility-macros)
    - [D9 — `std::numbers` replacement → own `wet::numbers` constants](#d9--stdnumbers-replacement--own-wetnumbers-constants)
    - [D10 — Math backend → `<cmath>`-free, freestanding series option](#d10--math-backend--cmath-free-freestanding-series-option)
    - [D11 — Observer API → shared `design::synthesize_observer`](#d11--observer-api--shared-designsynthesize_observer)
  - [Per-feature](#per-feature)
    - [D12 — Repetitive control rollout → SISO first](#d12--repetitive-control-rollout--siso-first)
    - [D13 — Input-shaping placement → `trajectory/`](#d13--input-shaping-placement--trajectory)
    - [D14 — First online tuning method → relay autotuner](#d14--first-online-tuning-method--relay-autotuner)
    - [D15 — Spectral dependency → in-house Goertzel (embeddable)](#d15--spectral-dependency--in-house-goertzel-embeddable)
    - [D16 — H∞ / MPC / MHE → `toolbox.hpp`](#d16--h--mpc--mhe--toolboxhpp)
    - [D17 — Trajectory module shape](#d17--trajectory-module-shape)
    - [D18 — Pose representation → quaternion + translation](#d18--pose-representation--quaternion--translation)
    - [D19 — Embedded-primitive conventions (LUT / timers / encoder)](#d19--embedded-primitive-conventions-lut--timers--encoder)
  - [Open decisions](#open-decisions)
    - [O1 — Default ESC perturbation policy for MPPT](#o1--default-esc-perturbation-policy-for-mppt)
    - [O2 — Anti-chatter suppressor update strategy](#o2--anti-chatter-suppressor-update-strategy)
    - [O3 — Error-reporting model under the ETL backend](#o3--error-reporting-model-under-the-etl-backend)
    - [O4 — Promote `geometry.hpp` out of `utility/`](#o4--promote-geometryhpp-out-of-utility)

Why the library is the way it is. Each entry is a resolved design decision and the
reasoning behind it; open decisions still under discussion are at the end.

- **README** answers *what's in the box* (inventory).
- **roadmap** answers *what's next* (planned work).
- **this log** answers *why is it this way* (decisions + rationale).

Entries have a stable ID (`D#` / `O#`) so source comments, the roadmap, and the
README can cross-reference them. `Refs` points back to the roadmap item number(s)
and/or the relevant header.

## Foundational

### D1 — No mandatory third-party dependency; backend-agnostic core
**Status:** Accepted

The embeddable core (`wet/control.hpp`) carries no required third-party library. It
needs only a *backend profile* (C++ stdlib **or** ETL) for the handful of
std-replacement types, and a *math backend*. Everything heavier is host-only behind
`wet/toolbox.hpp`. See [[D8]].

**Why:** the library must drop into a freestanding/embedded target with no hosted
stdlib, while staying usable standalone on a workstation. We do not invent our own
container/optional primitives — those come from stdlib or ETL.

### D2 — Plants are universal types; Tier 1 / Tier 2 layering
**Status:** Accepted

The library never branches on "kind of plant." A motor, a thermal loop, a two-mass
drive — all are just a `StateSpace<NX,NU,NY>` / `TransferFunction<Nn,Nd>` with
different numbers. The algorithms (**Tier 1** — `design::discrete_lqr`, `design::place`,
…) only ever see that universal type and hold all the actual math, once.

**Tier 2** wrappers are pure sugar for common archetypes: `models::two_mass(M_m, M_l,
k, c, r)` just builds the right `StateSpace<4,1,2>` from named physical parameters,
hands it to Tier 1, and unpacks the result back into named fields. It computes nothing
new.

**Why:** avoid a combinatorial explosion of plant-specific APIs (`design_lqr_for_motor`,
`design_lqr_for_pendulum`, …) and keep the algorithms free of domain assumptions. The
test for a legitimate Tier 2 wrapper: removing it must leave the library functionally
complete — you could always construct the `StateSpace` by hand and call Tier 1.

### D3 — Three-tier synthesis pattern
**Status:** Accepted

Every synthesis follows: (1) a constexpr `design::` function, (2) a result struct
with `.as<U>()` + `bool success`, (3) an allocation-free runtime controller/estimator.

**Why:** design in `double` at compile time, deploy in `float` with no heap; fallible
numerical steps signal failure explicitly rather than throwing or asserting. Leaf
utilities (scaling, lookup, timers) deliberately deviate — pure runtime blocks with
no plant model need no `design::` stage.

### D4 — Embedded vs host placement is part of the design
**Status:** Accepted · **Refs:** roadmap #13/#14/#15 (see [[D16]])

Allocation-free primitives belong in `wet/control.hpp`; anything that pulls
`<vector>`, an FFT, or a solver/optimizer belongs behind `wet/toolbox.hpp`.
`make embedded-check` enforces the embeddable contract.

**Why:** the same source serves an MCU and a workstation; the boundary is mechanical
and testable, not a matter of discipline.

### D5 — Stay in the controls lane; defer generic plumbing to ETL
**Status:** Accepted

Containers, queues, ring/FIFO buffers, CRC/checksums, debounce and similar
general-purpose utilities are out of scope — the [ETL](https://www.etlcpp.com) covers
them. This library ships only controls/DSP-specific value. The core never
`#include`s ETL; ETL is a recommended companion, never a dependency.

**Why:** avoid reinventing battle-tested plumbing; keep the library focused and the
core dependency-free. (CRC/checksums and SPSC/circular buffers were explicitly cut
from the embedded-primitives scope on these grounds — use `etl::crc*`,
`etl::queue_spsc_*`, `etl::circular_buffer`.)

### D6 — Explicit failure signaling
**Status:** Accepted

Fallible numerical steps return `{Result, bool success}` (design time) or
`wet::optional` (e.g. matrix inversion, DARE convergence). No exceptions in the
numerical path.

**Why:** constexpr-compatible, freestanding-compatible, and forces callers to handle
non-convergence. See [[D7]] for the runtime-tick counterpart.

## Architecture

### D7 — Runtime `step()` status policy → out-of-band `status()`
**Status:** Accepted (2026-06)

`step(r, y) → control` stays the allocation-free hot path; per-tick health is read via
a separate `status()` / `last_status` accessor — not an `expected<T, E>` return.

**Why:** a degraded controller still actuates, so the runtime is always
value-plus-health (a *product* type), not a sum type. Per-tick unwrap also adds branch
cost/noise in ISRs. `tl::expected`/`std::expected` is the wrong shape here and isn't
freestanding-available. Design time keeps the [[D6]] `{Result, bool success}` contract.

### D8 — std-replacement backend → stdlib or ETL, per-facility macros
**Status:** Accepted (2026-06) · **Refs:** roadmap #21, `backend.hpp`, `config.hpp`

Core uses `wet::` aliases (`array`/`optional`/`tuple`/`clamp`/…) over **one of two
interchangeable backends**: C++ stdlib (default, hosted, standalone) or ETL
(freestanding). Selected by per-facility `wet/config.hpp` macros (`WET_BACKEND_ETL`
for containers; `WET_MATH_BACKEND_*` for math) — orthogonal and mix-and-match, not a
single `WET_BACKEND` enum. Container/utility aliases live in one `backend.hpp`; the
math backend stays in `math/math_backend.hpp`. There is no third "invent our own"
backend. See [[D1]], [[O3]].

**Why:** one core, two deployment profiles, no mandatory third-party dep. Per-facility
macros let `config.hpp` be read both early (containers) and late (math) without
ordering hazards.

### D9 — `std::numbers` replacement → own `wet::numbers` constants
**Status:** Accepted (2026-06) · **Refs:** roadmap #21, `backend.hpp`

The core uses its own constexpr `wet::numbers::pi_v<T>` etc. — no dependency on
`<numbers>` or ETL constants anywhere in the core.

**Why:** keeps the no-mandatory-dep / freestanding guarantee ([[D1]], [[D8]]) without
relying on a C++20 stdlib header being present.

### D10 — Math backend → `<cmath>`-free, freestanding series option
**Status:** Accepted (2026-06) · **Refs:** roadmap #21, `math/math_backend.hpp`

`math_backend.hpp` is `<cmath>`-free; the `StdMathFallback`/`<cmath>` path moved to
`std_fallback.hpp` (hosted-only). `WET_MATH_BACKEND_FREESTANDING` routes runtime math
to the constexpr series in `constexpr_math.hpp`. `std::complex` interop is
`__has_include`-guarded. Selected independently of the container backend ([[D8]]).

**Why:** a freestanding target without `<cmath>` still needs `sin`/`sqrt`/`exp`/…;
the constexpr series already exist for compile-time evaluation, so reuse them at
runtime rather than require a hosted math library.

### D11 — Observer API → shared `design::synthesize_observer`
**Status:** Accepted

A single `design::synthesize_observer` returns the gain `L`, kept orthogonal to the
robust `design::place` (roadmap #16).

**Why:** one documented observer-synthesis entry point; pole-placement robustness is a
separate concern.

### D20 — Namespace organization: a horizontal `wet::` core + a few vertical domains
**Status:** Accepted (2026-06) · **Refs:** roadmap #3 (unified block concept), #31/#37/#38, API-naming convention

**`wet::` is the horizontal library.** The general-purpose control-engineering surface —
scalar/complex math, LTI models, controllers (PID/LQR/LQG/lead-lag/SMC/PR/ADRC/ESC/…),
estimators (Kalman/EKF/UKF/observers/disturbance), filters, trajectory profiling, and the
embedded toolbox primitives — all live directly in `wet::`. This is the product; do **not**
carve it into `control::`/`estimation::`/`filters::` sub-namespaces — that splits the core
along academic lines for no caller benefit. Namespaces break out only for two reasons below.

**Vertical domains** name an application area whose blocks are not general-purpose:

| Namespace | Scope |
| --------- | ----- |
| `motor::` | motor drive: FOC, SVPWM modulation, thermal, mechanical/parameter estimators, `PmacServo`, DC-bus limiting, encoder, actuator (roadmap #31) |
| `power::` | grid-tied power electronics: sync/PLL control use, droop/VSM, sequence control, islanding/LVRT, battery SOC/SOH + energy management (roadmap #38) |
| `hydraulic::` | valve flow-linearization + deadband comp, load-sensing pump, hydrostatic dual-path drive (roadmap #37) |
| `robotics::` | manipulator kinematics: serial-arm DH/IK, SCARA, delta, Stewart (geometry only) |

**Cross-cutting layers** stay flat and never nest under a domain:

| Namespace | Role |
| --------- | ---- |
| `mat::` | linear algebra |
| `design::` | all offline synthesis — Riccati, pole placement, PID tuning, **and** sysid excitation/identification — returning rich result structs (`design::lqr` → full result, not just `K`) |
| `analysis::` | host-side frequency/time evaluation (bode/nyquist/step/margins/norms) |
| `sim::` / `plot::` | host simulation + plotting |
| `matlab::` | MATLAB name+behaviour aliases — a thin compatibility skin (`matlab::lqr` returns only `K`); every entry is a `using` or one-line forwarder, **never** a second implementation body |

**Boundary calls (settled):**
- **Shared three-phase math stays in `wet::`** (`wet/transforms.hpp`: Clarke/Park, symmetrical
  components, instantaneous power, plus the SOGI/PLL family). Both `motor::` and `power::` depend
  on it; parking it under either creates a wrong-way dependency — it is generic AC vocabulary, like
  `mat::` is to everyone.
- **Trajectory stays in `wet::`** — trapezoidal/S-curve/spline profiling is general motion (a motor
  axis uses it too); only manipulator geometry is `robotics::`.
- **No `dsp::`/`sysid::` (yet).** DSP overlaps the `wet::` filters with no seam; sysid is a
  design-time activity → its excitation/identification code lands in `design::`. Carve either out
  only if it grows its own runtime surface. (ponytail: YAGNI — don't pre-split.)

**Header granularity = one feature, not one namespace.** A feature's synthesis (`design::`) and its
runtime block (`wet::` or its domain) live **together** in one header — `lqr.hpp` holds `design::lqr`
*and* the LQR runtime. Two public namespaces per header (design tier + runtime tier of D3) is the
intended shape, not arbitrary mixing. (`detail::` is exempt; PID is being unified to this shape — the
lone split.) The **only** forced split-out is heavy host-only synthesis (H∞/MPC/MHE → `toolbox.hpp`,
[[D16]]), which cannot ship to target. Folder layout tracks the namespace but is not binding — on each
migrated header re-check that file location and contents still belong.

**Single-implementation guarantee.** One canonical symbol per algorithm; all aliases resolve to it,
so they cannot diverge. This is the structural fix for the MATLAB/Simulink failure mode where
same-named blocks ship different implementations. Enforced by a build test asserting each facade
symbol forwards to its canonical (no independent logic) — "test + rule", not convention alone.

**Why:** `wet::` is one coherent control library, not a pile of toolboxes; splitting it by academic
category would fragment the core for no caller gain. Vertical domains isolate application-specific
blocks (and the user's deep domains — motor/power/hydraulic); a generic root keeps domain assumptions
from leaking into common code; the MATLAB layer stays a faithful compatibility skin rather than a fork.

## Per-feature

### D12 — Repetitive control rollout → SISO first
**Status:** Accepted · **Refs:** roadmap #5

SISO repetitive control first, then MIMO — matching every other rollout in the library.

### D13 — Input-shaping placement → `trajectory/`
**Status:** Accepted · **Refs:** roadmap #6, `trajectory/input_shaper.hpp`

Input shaping lives in the `trajectory/` module (it generates/shapes commands, not a
signal filter) and pairs with motion planning (#20) in the same module.

### D14 — First online tuning method → relay autotuner
**Status:** Accepted · **Refs:** roadmap #7, `design/relay_autotune.hpp`

The Åström–Hägglund relay autotuner is the first/default online tuning method, with a
**Tyreus–Luyben** default tuning rule (gentler than Ziegler–Nichols; the modern
process-control standard). The shared safety policy — command clamps, slew limits,
output-saturation passthrough, bumpless transfer, rollback on degraded measurement /
timeout — is a cross-cutting requirement on **every** tuning runtime.

### D15 — Spectral dependency → in-house Goertzel (embeddable)
**Status:** Accepted · **Refs:** roadmap #9, `filters/spectral.hpp`

The known-frequency embeddable path uses an in-house Goertzel; host-side RFFT/FRF
stays a separate `toolbox.hpp` concern. Spectral (Goertzel) is embeddable; see [[D16]].

### D16 — H∞ / MPC / MHE → `toolbox.hpp`
**Status:** Accepted · **Refs:** roadmap #13/#14/#15

H∞ (#13), constrained MPC (#14), and MHE (#15) are host-only (`toolbox.hpp`) — they
allocate / run iterative solvers (gamma search, QP). Spectral (#9 Goertzel) is
embeddable. Instance of [[D4]].

### D17 — Trajectory module shape
**Status:** Accepted (2026-06) · **Refs:** roadmap #20, `trajectory/*`

A single `trajectory/` module (not a `motion/` directory). The BVP coefficient solve
and all segment-time solves run at `design::` time, so the runtime stores only
coefficients / segment tables. Multi-segment profiles use a fixed-`N` segment array
(`ScurveProfile` holds `array<…, 7>`) to stay allocation-free.

### D18 — Pose representation → quaternion + translation
**Status:** Accepted (2026-06) · **Refs:** roadmap #22/#23/#24, `kinematics/pose.hpp`

`Pose{ Translation3<T> translation, Quaternion<T> orientation }` is both the public
interchange type and the internal accumulation type throughout kinematics. Orientation
is accepted as RPY/quaternion/matrix at the boundary but **stored and composed as
quaternion**. `Transform4<T>` is retained only as an interop/export convenience (DH
matrices, point-cloud transforms), never the working representation.

**Why:** composing the FK chain as `(q, t)` pairs is ~half the FLOPs of `Mat4×Mat4`
per joint (~31 vs 64 multiplies), carries 7 scalars instead of 16, and a quaternion
renormalizes cheaply where a 4×4 rotation block drifts from orthonormal over a chain.

### D19 — Embedded-primitive conventions (LUT / timers / encoder)
**Status:** Accepted (2026-06) · **Refs:** roadmap #19, `utility/{lookup,timing,encoder}.hpp`

- **LUT** defaults to clamp out-of-range, with opt-in linear extrapolation; binary-search
  lookup (no cached-index hint for now).
- **Software timers** are `dt`-fed (clock-source agnostic), complementing the IEC
  `TON/TOF/TP` scan timers.
- **Encoder** position is a fixed-width signed counter; rates use `wrapped_delta`
  (rollover-safe, computed in the unsigned domain then sign-extended).

**Why:** every accumulator/counter is bounded by construction (the library's
overflow-safety rule); a `dt`-fed time source keeps the blocks portable across clocks.

## Open decisions

### O1 — Default ESC perturbation policy for MPPT
**Status:** Open · **Refs:** roadmap #8

Dither frequency, amplitude schedule, and freeze criteria for the extremum-seeking
MPPT convenience path are not yet fixed.

### O2 — Anti-chatter suppressor update strategy
**Status:** Open · **Refs:** roadmap #9

Continuous adaptation vs gated updates for the online harmonic-suppressor retune.

### O3 — Error-reporting model under the ETL backend
**Status:** Open (deferred) · **Refs:** roadmap #21

Whether to wire ETL's error-handler callback path. Deferred until an ETL target
actually needs it; the [[D6]] `bool success` result structs remain the public contract
regardless.

### O4 — Promote `geometry.hpp` out of `utility/`
**Status:** Open · **Refs:** roadmap #22

Now that `utility/geometry.hpp` (Quaternion/DCM/Euler/Vec3) is a first-class robotics
dependency for #22–#24, whether it should move out of `utility/`.
