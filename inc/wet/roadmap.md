# wetmelon::control Library Roadmap

Date: 2026-05-25
Owner: controls core
Status: draft

## Table of Contents

- [Purpose](#purpose)
- [Design Constraints](#design-constraints)
- [Standard Synthesis Workflow](#standard-synthesis-workflow)
- [Roadmap](#roadmap)
- [Testing and Documentation](#testing-and-documentation)
- [Decision Items](#decision-items)

## Purpose

Track planned additions across controllers, estimators, and signal-processing/identification, held to a common synthesis workflow using the three-tier pattern:

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
- **Tier 1 / Tier 2 layering.** The library never branches on "kind of plant". Plants are universally described by `StateSpace<NX,NU,NY>` / `TransferFunction<Nn,Nd>`. Tier 1 functions take those universal types (or time-series + order) and are the implementations of truth; Tier 2 wrappers (e.g. `models::two_mass`, `analysis::identify_two_mass`, `CascadePPI`) are thin aliases that call Tier 1 with archetype-specific defaults and unpack results into named physical parameters. Tier 2 must never add capability — removing it leaves the library functionally complete.

## Standard Synthesis Workflow

1. Define plant model — `StateSpace` directly, or linearize a nonlinear model at an operating point.
2. Run design stage — `design::synthesize_*` / `design::*`; evaluate convergence and feasibility.
3. Produce artifacts — design result (gains/poles/internals + success), analysis models (closed-loop, error dynamics), runtime bundle (`step(...)`, `reset()`).
4. Convert for deployment — design in `double`, then `.as<float>()` when needed.
5. Verify — static checks (dimensions, constexpr viability, stability) and runtime checks (numerical references, closed-loop expectations).

## Roadmap

Ordered roughly by leverage and dependency: earlier items unblock later ones. Status tags: ☑ done · ⊘ sketched (header exists) · ☐ planned.

### 1. Luenberger / reduced-order observer ☑

Deterministic state observers via pole placement; the counterpart to the Kalman family. Prerequisite for output-feedback laws without a stochastic model, and for the disturbance observer (#4).

- Done (`estimation/observer.hpp`): full-order `Observer` and reduced-order (Gopinath) `ReducedOrderObserver`, with `design::synthesize_observer` / `design::synthesize_reduced_observer` via observer-form Ackermann (single-output), `StateSpace` + real/complex-pole overloads. Full-order uses the predictor recursion (error poles eig(A − LC)); reduced-order estimates only the unmeasured states and reads the measured state straight from `y` (ideal for encoder velocity).
- Single-output (NY == 1) only — MIMO-output placement is item #16; use the Kalman filter for MIMO until then.
- References: D. G. Luenberger, "An Introduction to Observers," IEEE TAC, 1971, https://doi.org/10.1109/TAC.1971.1099826; B. Gopinath, "On the Synthesis of Minimal-Order Observers," BSTJ, 1971.
- Acceptance: error dynamics match placed poles ✓; constexpr design ✓; reduced-order reconstructs unmeasured states ✓.

### 2. DSP biquad/notch family + utility blocks ⊘

Complete the biquad family beyond low-pass and add everyday runtime blocks. Constexpr coefficient designers; allocation-free runtimes. Unblocks harmonic suppression (#9) and is broadly useful.

- Done (`filters/filters.hpp`): RBJ-cookbook biquad designers `design::notch`, `bandpass`, `highpass_2nd`, `peaking`, `lowshelf`, `highshelf` → `SecondOrderCoeffs` (Q-parameterized, designed directly in the digital domain); runtimes `Biquad` (Direct Form I) and `BiquadCascade<N>` (SOS cascade). All constexpr + float/double.
- Pending: utility runtime blocks `MovingAverage`, `RunningRMS`, `MedianFilter<N>`, `RateLimiter`, `DirtyDerivative`, anti-windup `Integrator`, `Deadband`, `Hysteresis`.
- References: Oppenheim & Schafer, "Discrete-Time Signal Processing," 3rd ed., 2009; R. Bristow-Johnson, "Cookbook formulae for audio EQ biquad filter coefficients."
- Acceptance: per-type frequency-response checks ✓; SOS cascade ✓; float/double parity ✓; utility blocks (pending).

### 3. Identification and excitation infrastructure ⊘

The commissioning workflow: drive the plant with a known excitation signal, log the response, identify a model, synthesize gains, deploy. The library never asks "what kind of plant?" — the user describes their system by constructing a `StateSpace` / `TransferFunction` (directly or via Tier 2 model builders), and identification produces those same universal types from time-series data.

**On-target (embeddable, `wet/control.hpp`):**

- **Excitation generators** (`controllers/excitation.hpp`, new). Runtime signal sources, plant-agnostic, allocation-free, constexpr-configurable. Each exposes `T step(T t)` (or `T step()` with internal sample-time counter) and a `done()` predicate. Implement:
  - `Chirp<T>` — linear and logarithmic sweep; config `(amplitude, f_start_hz, f_end_hz, duration_s, mode)`.
  - `PRBS<T>` — maximum-length pseudo-random binary sequence; config `(amplitude, lfsr_order, clock_period_s, seed)`; reproducible across builds.
  - `StepTrain<T>` — alternating ±A steps with configurable hold; config `(amplitude, hold_s, cycles)`.
  - `Ramp<T>` — rate-limited slope with end-of-segment freeze; config `(target, rate, hold_at_end_s)`.
  - `MultiSine<NTones, T>` — sum of `NTones` sinusoids, `std::array<Tone, NTones>` where `Tone = {amplitude, freq_hz, phase_rad}`; useful for crest-factor-controlled FRF excitation.
  - Each gets a corresponding `design::synthesize_*` validating the config (positive amplitudes, finite durations, etc.).
- **Generic cascade controller** (`controllers/cascade.hpp`, new). `template<typename Outer, typename Inner> class Cascade` composing any two controllers that satisfy a small `Controller` concept (`T control(T r, T y)`, `void reset()`). Outer-loop output becomes inner-loop reference; inner-loop output goes to the plant. Anti-windup is the inner controller's responsibility (existing `PIDController` already handles it). Tier 2 alias: `template<typename T> using CascadePPI = Cascade<PController<T>, PIDController<T>>;` (a tiny `PController<T>` may need to be extracted from `pid.hpp` — `PIDController` with `Ki=Kd=0` works but a dedicated struct is cleaner). Test against a hand-written cascade composition on a synthetic plant.
- **Data-logger ring buffer** (`utility/ring_log.hpp`, new). `template<std::size_t N, typename... Channels> class RingLog` — fixed-size SOA, `push(t, ch1, ch2, ...)`, overwrite-oldest semantics. Flush via `for_each([](auto t, auto... ch){ ... })` callback that hands ordered tuples to user-supplied transport (UART/SPI/JSON). No allocation, no opinions on wire format. Channel storage is `std::tuple<std::array<Channels, N>...>` so flush can stream one channel at a time if needed.
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

### 4. Disturbance observer + DOB control law ⊘

Observer-based disturbance estimation integrated with a base controller. Targets: PMSM/BLDC servo drives with load-torque disturbances, robot joints with friction/backlash, precision stages with external force. Estimator core sketched in `estimation/disturbance_observer.hpp`; this adds the control-law wiring. Depends on #1.

- Interface: `design::synthesize_dob(plant, nominal_inverse, q_filter_spec, base_controller)` → `DOBResult` / `DOBArtifacts` + runtime with per-tick compensation.
- References: S. Li et al., "Disturbance Observer-Based Control," CRC Press, 2016, https://doi.org/10.1201/b16570; W.-H. Chen et al., "Disturbance-Observer-Based Control... An Overview," IEEE TIE, 2016, https://doi.org/10.1109/TIE.2015.2478397
- Acceptance: step-load rejection; frequency-domain attenuation; model-mismatch regression.

### 5. Repetitive controller ⊘

Internal-model compensation for periodic disturbances over selected harmonics. Targets: grid-tied inverters, precision motion with periodic trajectories, rotating-machinery ripple. Depends on #2 (delay line + robustness filter).

- Interface: `design::synthesize_repetitive(plant, fundamental_frequency, harmonic_set, robustness_filter)` → `RepetitiveResult` / `RepetitiveArtifacts` + baseline-plus-repetitive runtime.
- Reference: S. Hara et al., "Repetitive Control System," IEEE TAC, 1988. https://doi.org/10.1109/9.1274
- Acceptance: harmonic-rejection benchmarks; stability margins with robustness filter; integration with LQI/LQGI.

### 6. Input shaping (command prefilter) ⊘

Feedforward command shaping for resonance suppression (ZV/ZVD/EI shapers) on reference trajectories. Targets: 3D-printer gantries, pick-and-place/CNC moves, flexible stages. Pure feedforward (no estimator).

- Interface: `design::synthesize_input_shaper(natural_frequency, damping_ratio, shaper_type)`, `..._from_modes(mode_list, shaper_type)` → `InputShaperResult` / `InputShaperRuntime` + multi-axis bank helper.
- References: Singer & Seering, "Preshaping Command Inputs to Reduce System Vibration," ASME JDSMC, 1990, https://doi.org/10.1115/1.2894142; Singhose et al., ASME JMD, 1994, https://doi.org/10.1115/1.2919428
- Acceptance: coefficient normalization / delay ordering; residual-vibration attenuation under nominal/detuned resonance.
- Open: place in `controllers/` vs `filters/`.

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

### 9. Harmonic detection & suppression (anti-chatter) ⊘

Online dominant-harmonic detection and suppression via notch/PR/repetitive elements. Targets: lathe-turning chatter, spindle-tool resonance in CNC, gearbox tonal vibration. Depends on #2 (notch) and harmonic estimation. Includes the spectral primitives (windowed RFFT / Goertzel, host-only) and a harmonic tracker (`estimation/harmonic_estimation.hpp` is currently a placeholder).

- Interface: `analysis::detect_dominant_harmonics(signal_window, sample_rate)` (host); `design::synthesize_harmonic_suppressor(harmonic_set, suppressor_type, robustness_settings)`; `design::synthesize_chatter_suppressor_turning(machine_model, spindle_speed, sensor_config)`.
- References: Altintas & Budak, "Analytical Prediction of Stability Lobes in Milling," CIRP Annals, 1995, https://doi.org/10.1016/S0007-8506(07)62342-7; Mojiri & Bakhshai, "An Adaptive Notch Filter for Frequency Estimation," IEEE TAC, 2004, https://doi.org/10.1109/TAC.2003.822862; Goertzel, Amer. Math. Monthly, 1958, https://doi.org/10.2307/2308968
- Acceptance: detection latency / false-positive benchmarks; measured harmonic-amplitude reduction; closed-loop stability + actuator effort with adaptive updates.

### 10. LPV gain-scheduled LQG/LQI ⊘

Gain scheduling over an operating-point grid. Targets: UAV dynamics vs airspeed/altitude, vehicle lateral dynamics vs speed, manipulators with configuration-dependent models.

- Workflow: local linear models over the grid → local controller/estimator gains → scheduled runtime bundle with interpolation. Artifacts: schedule map, gain tables, local closed-loop analysis.
- References: Rugh & Shamma, "Research on Gain Scheduling," Automatica, 2000, https://doi.org/10.1016/S0005-1098(00)00058-3; Apkarian & Gahinet, IEEE TAC, 1995, https://doi.org/10.1109/9.384219
- Acceptance: continuity across transitions; stability over the certified envelope.

### 11. Super-twisting SMC ⊘

Second-order sliding-mode runtime with design-time gain synthesis. Targets: electromechanical drives with bounded matched disturbances, hydraulic/pneumatic actuators, converter current loops.

- Interface: `design::synthesize_stsmc(plant_or_surface, disturbance_bound, bandwidth_targets)` → `STSMCResult` / `STSMCArtifacts` + runtime with optional boundary layer.
- References: Moreno & Osorio, "Strict Lyapunov Functions for the Super-Twisting Algorithm," IEEE TAC, 2012, https://doi.org/10.1109/TAC.2012.2186179; Levant, Int. J. Control, 2003, https://doi.org/10.1080/0020717031000099029
- Acceptance: chattering metrics vs existing SMC; disturbance rejection with bounded effort.

### 12. UKF (sigma-point filter) ☐

Unscented filter for nonlinear estimation where EKF linearization is poor (map item `sigma-point`).

- Interface: `UKF<NX,NU,NY>` runtime mirroring the EKF API; constexpr sigma-point weights.
- Reference: Julier & Uhlmann, "Unscented Filtering and Nonlinear Estimation," Proc. IEEE, 2004. https://doi.org/10.1109/JPROC.2003.823141
- Acceptance: matches EKF on near-linear systems; outperforms on strongly nonlinear references.

### 13. H-infinity output feedback ⊘

Output-feedback synthesis for weighted generalized plants. Targets: flexible structures with modal uncertainty, flight control with structured uncertainty, active suspension/vibration isolation.

- Interface: `design::synthesize_hinf(augmented_plant, weighting_filters, gamma_search_bounds)` → `HInfResult` / `HInfArtifacts` + S/T analysis models. Workflow: coupled Riccati solves over gamma; select feasible controller.
- References: Doyle et al., "State-Space Solutions to Standard H2 and H-infinity Control Problems," IEEE TAC, 1989, https://doi.org/10.1109/9.29425; Glover & Doyle, Systems & Control Letters, 1988, https://doi.org/10.1016/0167-6911(88)90055-2
- Acceptance: weighted robust-stability/performance; regression vs known references.
- Open: `control.hpp` vs `toolbox.hpp` placement (gamma search may allocate).

### 14. Constrained MPC ⊘

Finite-horizon constrained control with a deterministic runtime iteration budget. Targets: multivariable process control with limits, AV trajectory tracking with actuator limits, power electronics with current/voltage bounds.

- Interface: `design::synthesize_mpc(plant, horizon, weights, constraints, solver_budget)` → `MPCResult` / `MPCArtifacts` + fixed-iteration warm-start runtime.
- References: García, Prett & Morari, "Model Predictive Control: Theory and Practice — A Survey," Automatica, 1989, https://doi.org/10.1016/0005-1098(89)90002-2; Qin & Badgwell, Control Engineering Practice, 2003, https://doi.org/10.1016/S0967-0661(02)00186-7
- Acceptance: deterministic per-tick runtime bound; constraint-satisfaction regression; tracking/regulation tests.
- Open: `control.hpp` vs `toolbox.hpp` placement.

### 15. Moving horizon estimation (MHE) ☐

Optimization-based constrained estimator; toolbox-only (allocates/solver). The estimation counterpart to MPC.

- Reference: Rao, Rawlings & Mayne, "Constrained State Estimation for Nonlinear Discrete-Time Systems," IEEE TAC, 2003. https://doi.org/10.1109/TAC.2003.812777
- Acceptance: matches Kalman on linear/unconstrained problems; respects state constraints on nonlinear references.

### 16. Robust MIMO pole placement (true `place`) ☐

Foundational numerical routine: place the eigenvalues of (A − BK) for **multi-input** systems, spending the extra gain freedom to maximize numerical robustness (minimize eigenvector conditioning). Equivalent to MATLAB's `place`. Upgrades the Ackermann-only placement used today by `matlab::place`, the Luenberger/reduced observers (#1, via the dual `acker(Aᵀ, Cᵀ, p)ᵀ`), and any pole-placement controller to true MIMO.

- Naming fix: the current `matlab::place` is single-input Ackermann — i.e. MATLAB's `acker`, not `place`. Plan: rename it to `acker` and reserve `place` for this robust MIMO routine.
- Interface: `design::acker(A, B, desired_poles)` → gain `K`.
- Reference: J. Kautsky, N. K. Nichols, P. Van Dooren, "Robust pole assignment in linear state feedback," Int. J. Control, 1985. https://doi.org/10.1080/00207178508933420
- Acceptance: places assignable MIMO spectra; rejects pole multiplicity exceeding the input count; better conditioning than naive placement; matches MATLAB `place` on references.
- Note: foundational — pull earlier than its number if MIMO observers/controllers are needed.

### 17. Fast-math-robust filter coefficient designs ⊘

Restructure bilinear filter designers so that the unit-DC-gain identity (and other intended algebraic relations) holds *constructively* rather than emerging from exact cancellation, so the property survives `-fassociative-math` reassociation in downstream user builds.

- **`design::lowpass_2nd` ☑ fixed.** The original bilinear formula didn't merely drift under fast-math — its coefficients placed both poles essentially *on* the unit circle (|p| ≈ 1.0), making it a near-passthrough rather than a low-pass at all (measured |H| ≈ 1.0 at DC, fc, and 10·fc). Rewritten so the numerator taps are derived directly from the unit-DC-gain identity: since the bilinear numerator is ω₀²(1+z⁻¹)² the taps are in fixed ratio 1:2:1, so they are pinned as `dc_sum·{¼, ½, ¼}` with `dc_sum = 1 + a1 + a2`. Unity DC gain now holds by construction (and the ¼/½ factors are exact powers of two), identical under strict-IEEE and `-ffast-math`. Verified: |H| = 1.000 at DC, 0.707 at fc, 0.009 at 10·fc; DC-gain regression test tightened back to 1e-3, settling test back to 0.01.
- Audit pass (remaining): sweep the other biquad designers (`notch`, `bandpass`, `highpass_2nd`, `peaking`, `lowshelf`, `highshelf`) and the StateSpace→coeffs paths the same way. The RBJ cookbook formulas compute b/a from single expressions and are mostly less fragile, but worth verifying with a `-ffast-math` regression sweep.
- Reference: Higham, "Accuracy and Stability of Numerical Algorithms" (2nd ed., 2002), §3 on conditioned summation; Oppenheim & Schafer, "Discrete-Time Signal Processing" (3rd ed., 2009), bilinear-transform chapter.
- Acceptance: a DC-gain regression test that holds settled value within 1e-3 of 1.0 even when the runner is built with `-ffast-math`; afterwards tighten the LowPass2nd tolerance back to 0.01. **Done for `lowpass_2nd`** (`test_filters.cpp`: DC-gain at 1e-3, settling at 0.01, plus a cross-zeta unity-gain case); remaining biquads still to sweep.

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

- Runtime status-return policy for `step(...)` in bundles with fallible observer/estimator updates.
- `control.hpp` vs `toolbox.hpp` placement for H-infinity, MPC, MHE, and spectral tools (allocators/solvers → toolbox).
- Repetitive control rollout scope: SISO first or direct MIMO.
- Module placement for input shaping: `controllers/` vs `filters/`.
- Default first online tuning method and shared safety policy (excitation, clamps, rollback).
- Default ESC perturbation policy for MPPT (frequency, amplitude schedule, freeze criteria).
- Anti-chatter suppressor update strategy: continuous adaptation vs gated updates.
- FFT/spectral dependency: in-house Goertzel/RFFT vs optional third-party (host only).
- Observer API shape: shared `design::synthesize_observer` returning `L`, vs folding into existing `place`.
